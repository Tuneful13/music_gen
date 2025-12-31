import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

import representation


model_name="Qwen/Qwen2.5-1.5B"

class MusicLLM(nn.Module):
    def __init__(self, model_name_or_path="Qwen/Qwen2.5-1.5B", music_config=None):
        super().__init__()
        
        # 1. Configuración y Cuerpo del LLM
        self.llm_config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.hidden_size = getattr(self.llm_config, "hidden_size", 
                           getattr(self.llm_config, "d_model", 
                           getattr(self.llm_config, "n_embd", None)))
        
        self.llm_body = AutoModel.from_pretrained(model_name_or_path, config=self.llm_config)

        # 2. Adaptador de Entrada (Dimensiones alineadas al LLM)
        self.music_embeddings = nn.ModuleList([
            nn.Embedding(v_size, self.hidden_size) 
            for v_size in music_config['n_tokens']
        ])
        
        # --- LA PIEZA CLAVE: LayerNorm Post-Suma ---
        # Esto estabiliza la suma de los 6 embeddings antes de entrar al LLM
        self.emb_norm = nn.LayerNorm(self.hidden_size)
        self.dropout = nn.Dropout(0.1) # Opcional: añade robustez

        # 3. Adaptador de Salida (Cabezales de predicción)
        self.output_heads = nn.ModuleList([
            nn.Linear(self.hidden_size, v_size)
            for v_size in music_config['n_tokens']
        ])

    def forward(self, x_music, attention_mask=None):
        # --- ENTRADA ---
        inputs_embeds = 0
        for i, layer in enumerate(self.music_embeddings):
            inputs_embeds += layer(x_music[:, :, i])
        
        # Aplicamos la normalización y el dropout
        # Esto asegura que la media sea 0 y la varianza 1, tal como espera el LLM
        inputs_embeds = self.emb_norm(inputs_embeds)
        inputs_embeds = self.dropout(inputs_embeds)
            
        # --- PROCESAMIENTO (LLM) ---
        outputs = self.llm_body(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]
            
        # --- SALIDA ---
        logits_list = [head(hidden_states) for head in self.output_heads]
        return logits_list
    
    def freeze_backbone(self, freeze=True):
        """
        Si freeze=True, congela todos los parámetros del LLM (Qwen).
        Solo se entrenarán los embeddings de música y los cabezales de salida.
        """
        for param in self.llm_body.parameters():
            param.requires_grad = not freeze
        
        status = "CONGELADO" if freeze else "DESCONGELADO"
        print(f"[*] El cuerpo del LLM (Qwen) ahora está: {status}")


# --- BLOQUE DE COMPROBACIÓN (MAIN) ---
if __name__ == "__main__":
    # 1. Parámetros de prueba
    encoding = representation.load_encoding("encoding.json")
    batch_size = 2
    seq_len = 1024  # Tamaño máximo que definimos antes
    
    # 2. Inicializar modelo
    # Nota: Si no tienes GPU, esto usará la CPU.
    model = MusicLLM(music_config=encoding)
    model.eval() # Modo evaluación
    
    # 3. Crear datos ficticios dinámicamente
    # Usamos un bucle para generar cada columna basada en su n_token correspondiente
    mock_columns = []
    for max_val in encoding['n_tokens']:
        col = torch.randint(0, max_val, (batch_size, seq_len))
        mock_columns.append(col)
    
    mock_input = torch.stack(mock_columns, dim=-1)

    print(f" Input shape: {mock_input.shape} (Batch, Seq, Atributos)")

    # 4. Ejecutar el modelo
    with torch.no_grad():
        logits = model(mock_input)

    # 5. Comprobar resultados
    print("\n Verificación de salida:")
    n_tokens_names = ["Tipo", "Beat", "Posición", "Pitch", "Duración", "Instrumento"]
    
    for i, l in enumerate(logits):
        print(f"🔹 {n_tokens_names[i]}: Logits shape {l.shape}")
        # La forma esperada es (batch, seq_len, n_tokens_especifico)
        expected_shape = (batch_size, seq_len, encoding['n_tokens'][i])
        assert l.shape == expected_shape, f"Error en dimensión {i}"

    print("\n ¡Todo funciona! El modelo procesa los 6 atributos y devuelve las predicciones alineadas.")