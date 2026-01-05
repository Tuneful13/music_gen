import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
from peft import LoraConfig, get_peft_model, TaskType

import representation

class MusicLLM(nn.Module):
    def __init__(self, model_name_or_path="Qwen/Qwen2.5-1.5B", music_config=None, use_lora=True):
        super().__init__()
        
        # 1. Configuración y Cuerpo del LLM (Backbone)
        self.llm_config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.hidden_size = getattr(self.llm_config, "hidden_size", 896)
        
        base_model = AutoModel.from_pretrained(model_name_or_path, config=self.llm_config)

        # --- INTEGRACIÓN DE LoRA ---
        if use_lora:
            lora_config = LoraConfig(
                r=16, 
                lora_alpha=32,
                # Ajustamos las capas de atención de Qwen para ser entrenables
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                lora_dropout=0.1,
                bias="none",
                task_type=TaskType.FEATURE_EXTRACTION # Usamos el LLM como extractor de características
            )
            self.llm_body = get_peft_model(base_model, lora_config)
            print("[*] LoRA inyectado correctamente en Qwen")
        else:
            self.llm_body = base_model

        # 2. Adaptador de Entrada (Suma de los 6 atributos iniciales)
        self.music_embeddings = nn.ModuleList([
            nn.Embedding(v_size, self.hidden_size) for v_size in music_config['n_tokens']
        ])
        self.emb_norm = nn.LayerNorm(self.hidden_size)
        self.dropout = nn.Dropout(0.1)

        # 3. REALIMENTACIÓN INTRA-PASO: Embeddings de dependencia
        # Estos permiten que el modelo "recuerde" qué Tipo eligió antes de predecir el Pitch
        self.intra_step_embeddings = nn.ModuleList([
            nn.Embedding(v_size, self.hidden_size) for v_size in music_config['n_tokens']
        ])

        # 4. Cabezales de predicción
        self.output_heads = nn.ModuleList([
            nn.Linear(self.hidden_size, v_size) for v_size in music_config['n_tokens']
        ])

    def forward(self, x_music, attention_mask=None, target_types=None):
        # --- PROCESAMIENTO INICIAL ---
        inputs_embeds = 0
        for i, layer in enumerate(self.music_embeddings):
            inputs_embeds += layer(x_music[:, :, i])
        
        inputs_embeds = self.dropout(self.emb_norm(inputs_embeds))
        outputs = self.llm_body(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        h = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]

        logits_list = []
        h_current = h 

        # Generamos la máscara de instrumentos permitidos HASTA el momento actual
        instr_mask = self._generate_instr_mask(x_music)

        for i in range(6):
            logits = self.output_heads[i](h_current)
            
            # --- CORRECCIÓN DE MÁSCARA EN TRAIN ---
            if i == 5: 
                # Si en train.py pasamos los tipos de los targets:
                if target_types is not None:
                    # Usamos el tipo real que debe predecir
                    is_note = (target_types == 3).unsqueeze(-1)
                else:
                    # Si no los pasamos, una heurística común es asumir que si
                    # el input es nota, el target probablemente también (aunque es menos preciso)
                    # O mejor aún, no filtrar por tipo en train para que el modelo aprenda 
                    # por sí solo la relación Tipo -> Instrumento.
                    is_note = torch.ones_like(logits, dtype=torch.bool)
                
                logits = torch.where(is_note, logits.masked_fill(~instr_mask, -float('inf')), logits)
            
            logits_list.append(logits)
            
            if i < 5:
                # Realimentación: usamos el valor real (Teacher Forcing)
                h_current = h_current + self.intra_step_embeddings[i](x_music[:, :, i])
        
        return logits_list

    @torch.no_grad()
    def generate(self, start_tokens, max_len, temperature=1.0, eos_token=None, top_k=20):
        self.eval()
        generated = start_tokens
        device = start_tokens.device
        batch_size = start_tokens.shape[0]
        
        # 1. Rastrear instrumentos permitidos por cada secuencia del batch
        allowed_instruments = [set() for _ in range(batch_size)]
        for b in range(batch_size):
            for row in start_tokens[b]:
                if row[0].item() == 1: # Tipo: Instrument-change
                    allowed_instruments[b].add(row[5].item())
        
        # 2. Rastrear último beat por cada secuencia del batch (Monotonicidad)
        last_beats = start_tokens[:, -1, 1].clone() # [batch_size]

        for _ in range(max_len):
            # Forward del LLM (último estado)
            inputs_embeds = 0
            for i, layer in enumerate(self.music_embeddings):
                inputs_embeds += layer(generated[:, :, i])
            
            inputs_embeds = self.emb_norm(inputs_embeds)
            outputs = self.llm_body(inputs_embeds=inputs_embeds)
            h_last = outputs.last_hidden_state[:, -1, :]

            current_step_tokens = []
            h_acc = h_last
            chosen_types = None # Guardará los tipos elegidos para este paso [batch_size]

            for i in range(6):
                logits = self.output_heads[i](h_acc) / temperature
                
                # --- APLICAR RESTRICCIONES POR LOTE ---
                if i == 1: # BEAT
                    for b in range(batch_size):
                        mask = torch.arange(logits.size(-1), device=device) < last_beats[b]
                        logits[b, mask] = -float('Inf')

                elif i == 5: # INSTRUMENTO
                    for b in range(batch_size):
                        # Solo restringimos si el tipo elegido en i=0 fue NOTA (3)
                        if chosen_types[b] == 3 and len(allowed_instruments[b]) > 0:
                            mask = torch.ones(logits.size(-1), device=device) * -float('Inf')
                            for inst_id in allowed_instruments[b]:
                                mask[inst_id] = 0
                            logits[b] += mask
                
                # Muestreo
                k = min(top_k, logits.size(-1))
                v, _ = torch.topk(logits, k)
                logits[logits < v[:, [-1]]] = -float('Inf')
                probs = F.softmax(logits, dim=-1)
                sample = torch.multinomial(probs, num_samples=1)
                current_step_tokens.append(sample)
                
                # Actualizar estados de restricción para el siguiente paso/atributo
                val_tensor = sample.squeeze(-1)
                if i == 0: 
                    chosen_types = val_tensor
                elif i == 1: 
                    last_beats = val_tensor
                elif i == 5:
                    for b in range(batch_size):
                        if chosen_types[b] == 1: # Si declaramos instrumento
                            allowed_instruments[b].add(val_tensor[b].item())

                if i < 5:
                    h_acc = h_acc + self.intra_step_embeddings[i](val_tensor)

            # Concatenar resultado
            next_step = torch.cat(current_step_tokens, dim=-1).unsqueeze(1)
            generated = torch.cat((generated, next_step), dim=1)

            if eos_token is not None:
                # Si todos los elementos del batch han generado EOS (opcional: simplificado a solo el primero)
                if current_step_tokens[0][0].item() == eos_token:
                    break
        return generated
    
    def _generate_instr_mask(self, x_music):
        """
        Versión altamente eficiente y vectorizada (sin bucles Python).
        Genera la máscara [Batch, Seq, Num_Instruments].
        """
        batch_size, seq_len, _ = x_music.shape
        num_instr = self.output_heads[5].out_features
        device = x_music.device
        
        # 1. Identificar dónde ocurren las declaraciones (Tipo 1)
        is_type_1 = (x_music[:, :, 0] == 1) # [Batch, Seq]
        instr_ids = x_music[:, :, 5]        # [Batch, Seq]
        
        # 2. Crear un mapa de "apariciones iniciales"
        # Inicializamos en ceros: [Batch, Seq, Num_Instr]
        mask = torch.zeros((batch_size, seq_len, num_instr), device=device, dtype=torch.float32)
        
        # Usamos scatter para poner un 1 solo en el instrumento declarado en ese paso exacto
        # Solo scattereamos donde is_type_1 es True para no marcar el instrumento 0 por error
        mask.scatter_(2, instr_ids.unsqueeze(-1), is_type_1.unsqueeze(-1).float())
        
        # 3. Propagar la declaración hacia el futuro usando suma acumulativa (cumsum)
        # Si un instrumento aparece en el paso 5, el cumsum será >= 1 para todos los pasos >= 5
        mask = torch.cumsum(mask, dim=1)
        
        # Convertir a booleano (True si el instrumento ha aparecido al menos una vez)
        instr_mask = (mask > 0)
        
        # --- AJUSTES DE SEGURIDAD ---
        
        # A. Permitir siempre el token 0 (suele ser PAD o SOS) para evitar bloqueos
        instr_mask[:, :, 0] = True
        
        # B. Manejo de pasos iniciales: Si en un paso no hay NINGÚN instrumento permitido todavía
        # (por ejemplo, al principio de la canción antes de los Tipo 1), permitimos todos 
        # para que el modelo pueda predecir los Tipo 1 libremente sin recibir -inf.
        no_instr_declared = ~instr_mask.any(dim=-1, keepdim=True)
        instr_mask = instr_mask | no_instr_declared
        
        return instr_mask

    def print_trainable_parameters(self):
        """Utilidad para ver qué estamos entrenando (LoRA + Heads + Embeddings)"""
        trainable_params = 0
        all_param = 0
        for _, param in self.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(f"Entrenable: {trainable_params} | Total: {all_param} | %: {100 * trainable_params / all_param:.2f}%")

# --- PRUEBA DE FUNCIONAMIENTO ---
if __name__ == "__main__":
    encoding = {"n_tokens": [10, 256, 128, 128, 64, 128]} # Dummy config
    model = MusicLLM(music_config=encoding, use_lora=True)
    model.print_trainable_parameters()
    
    # Simular entrada [batch, seq, attributes]
    mock_in = torch.randint(0, 10, (1, 10, 6))
    out = model(mock_in)
    print(f"Salida exitosa: {len(out)} cabezales de logits.")