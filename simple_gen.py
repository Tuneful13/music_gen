import torch
import pathlib
import numpy as np
import muspy
import representation
from compund_transformer import MusicLLM
import gc

# ==========================================
# CONFIGURACIÓN
# ==========================================
CHECKPOINT_PATH = "exp/qwen_music_v1/checkpoints/best_model.pt"
ENCODING_PATH = "encoding.json"
SOUNDFONT_PATH = "./data/MS_Basic.sf2" 
OUTPUT_WAV = "generacion_final.wav"

SEQ_LEN = 512       
TEMPERATURE = 1.0 
TOP_K = 40          
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    print(f"[*] Cargando recursos...")
    encoding = representation.load_encoding(ENCODING_PATH)
    
    model = MusicLLM(music_config=encoding)
    
    print(f"[*] Cargando pesos en CPU...")
    try:
        state_dict = torch.load(CHECKPOINT_PATH, map_location="cpu")
        if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"[!] Error al cargar checkpoint: {e}")
        return

    model.to(DEVICE)
    model.eval()
    del state_dict
    gc.collect()

    # Generación
    sos_token = encoding["type_code_map"]["start-of-song"]
    eos_token = encoding["type_code_map"]["end-of-song"]
    
    tgt_start = torch.zeros((1, 1, 6), dtype=torch.long, device=DEVICE)
    tgt_start[0, 0, 0] = sos_token

    print(f"[*] Generando música...")
    with torch.no_grad():
        generated_tensor = model.generate(
            tgt_start, 
            max_len=SEQ_LEN, 
            temperature=TEMPERATURE,
            eos_token=eos_token,
            top_k=TOP_K
        )

    generated_np = generated_tensor.cpu().numpy()[0]

    # Decodificación
    print(f"[*] Decodificando...")
    try:
        music = representation.decode(generated_np, encoding)
        
        if len(music.tracks) == 0 or sum(len(t.notes) for t in music.tracks) == 0:
            print("\n[!] El modelo no generó ninguna nota válida. Prueba a subir la temperatura.")
            return

        print("\n" + "="*40)
        print(f"{'MÉTRICAS MUSICALES':^40}")
        print("="*40)

        metricas_a_probar = [
            ("Pitch Class Entropy", muspy.pitch_class_entropy),
            ("Scale Consistency", muspy.scale_consistency),
        ]
        
        for name, func in metricas_a_probar:
            try:
                val = func(music)
                print(f" {name:20}: {val:.4f}")
            except:
                print(f" {name:20}: [N/A]")

        # 2. Ejecutar Groove Consistency (necesita un parámetro extra)
        try:
            # Comparamos el ritmo cada 4 tiempos (un compás estándar)
            g_val = muspy.groove_consistency(music, 4 * music.resolution)
            print(f" {'Groove Consistency':20}: {g_val:.4f}")
        except:
            print(f" {'Groove Consistency':20}: [N/A - Poco material]")
            
        print("="*40 + "\n")

        # Renderizado
        print(f"[*] Renderizando audio a {OUTPUT_WAV}...")
        representation.advanced_decode(
            generated_np, 
            encoding, 
            soundfont=SOUNDFONT_PATH, 
            output_wav=OUTPUT_WAV
        )
        print(f"[+] ¡Éxito! Generación completada.")

    except Exception as e:
        print(f"[!] Error inesperado: {e}")

if __name__ == "__main__":
    main()