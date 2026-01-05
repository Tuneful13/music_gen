import argparse
import logging
import pathlib
import pprint
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data
import tqdm

import dataset
from compound_transformer import MusicLLM 
import representation
import utils

@utils.resolve_paths
def parse_args(args=None, namespace=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", choices=("sod", "lmd", "lmd_full", "snd"), required=True, help="dataset key")
    parser.add_argument("-n", "--names", type=pathlib.Path, help="input names")
    parser.add_argument("-i", "--in_dir", type=pathlib.Path, help="input data directory")
    parser.add_argument("-o", "--out_dir", type=pathlib.Path, help="output directory")
    parser.add_argument("-ns", "--n_samples", default=50, type=int, help="number of samples")
    parser.add_argument("-s", "--shuffle", action="store_true")
    parser.add_argument("--use_csv", action="store_true")
    parser.add_argument("--model_steps", type=int, help="step of the trained model")
    
    # Sampling
    parser.add_argument("--seq_len", default=1024, type=int)
    parser.add_argument("--temperature", default=1.0, type=float) # Simplificado a float
    parser.add_argument("--top_k", default=20, type=int, help="Top-K sampling")
    
    parser.add_argument("-g", "--gpu", type=int)
    parser.add_argument("-j", "--jobs", default=1, type=int)
    parser.add_argument("-q", "--quiet", action="store_true")
    return parser.parse_args(args=args, namespace=namespace)

def save_pianoroll(filename, music, size=None, **kwargs):
    music.show_pianoroll(track_label="program", **kwargs)
    if size is not None: plt.gcf().set_size_inches(size)
    plt.savefig(filename)
    plt.close()

def save_result(filename, data, sample_dir, encoding):
    """Guarda los resultados en todos los formatos musicales."""
    # 1. Datos crudos
    np.save(sample_dir / "npy" / f"{filename}.npy", data)
    representation.save_csv_codes(sample_dir / "csv" / f"{filename}.csv", data)
    
    # 2. Decodificación a Música
    music = representation.decode(data, encoding)
    music.save(sample_dir / "json" / f"{filename}.json")
    music.write(sample_dir / "mid" / f"{filename}.mid")
    
    # 3. Visualización
    save_pianoroll(sample_dir / "png" / f"{filename}.png", music, (20, 5), preset="frame")
    
    # 4. Audio (Requiere FluidSynth)
    SF2_PATH = "data/MS_Basic.sf2"
    wav_path = sample_dir / "wav" / f"{filename}.wav"
    music.write(str(wav_path), soundfont_path=SF2_PATH)
    
    # Convertir a MP3
    mp3_path = sample_dir / "mp3" / f"{filename}.mp3"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(wav_path), "-b:a", "192k", str(mp3_path)])

def main():
    args = parse_args()

    if args.dataset is not None:
        if args.names is None: args.names = pathlib.Path(f"data/{args.dataset}/processed/test-names.txt")
        if args.in_dir is None: args.in_dir = pathlib.Path(f"data/{args.dataset}/processed/notes/")
        if args.out_dir is None: args.out_dir = pathlib.Path(f"exp/test_{args.dataset}")

    logging.basicConfig(level=logging.ERROR if args.quiet else logging.INFO, format="%(message)s")
    
    # Cargar argumentos de entrenamiento para mantener consistencia
    train_args = utils.load_json(args.out_dir / "train-args.json")
    device = torch.device(f"cuda:{args.gpu}" if args.gpu is not None and torch.cuda.is_available() else "cpu")
    encoding = representation.load_encoding(args.in_dir / "encoding.json")

    # Carpetas de salida
    sample_dir = args.out_dir / "samples"
    for fmt in ["npy", "csv", "txt", "json", "png", "mid", "wav", "mp3"]:
        (sample_dir / fmt).mkdir(exist_ok=True, parents=True)

    # Dataloader
    test_dataset = dataset.MusicDataset(args.names, args.in_dir, encoding, max_seq_len=train_args["max_seq_len"], max_beat=train_args["max_beat"], use_csv=args.use_csv)
    test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=args.shuffle, num_workers=args.jobs, collate_fn=dataset.MusicDataset.collate)

    # --- CARGA DEL MODELO (ADAPTADO A QWEN) ---
    logging.info(f"Cargando MusicLLM (Qwen) en {device}...")
    logging.info(f"[*] Inicializando arquitectura del modelo en CPU...")
    model = MusicLLM(
        model_name_or_path=train_args.get("model_name", "Qwen/Qwen2.5-1.5B"),
        music_config={"n_tokens": encoding["n_tokens"]},
    )

    checkpoint_dir = args.out_dir / "checkpoints"
    checkpoint_path = checkpoint_dir / ("best_model.pt" if args.model_steps is None else f"model_{args.model_steps}.pt")

    logging.info(f"[*] Cargando pesos desde {checkpoint_path} a la RAM del sistema...")
    try:
        # 1. Cargamos el archivo directamente a la CPU para no saturar la VRAM
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        
        # 2. Si el checkpoint es un diccionario de entrenamiento completo, extraemos solo el modelo
        if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        
        # 3. Cargamos los pesos en la estructura del modelo (que sigue en CPU)
        model.load_state_dict(state_dict)
        logging.info("[*] Pesos cargados en la arquitectura correctamente.")

    except Exception as e:
        logging.error(f"[!] Error crítico al cargar el checkpoint: {e}")
        sys.exit(1)

    # 4. Ahora que el modelo está completo en CPU, lo movemos entero a la GPU
    logging.info(f"[*] Transfiriendo modelo a {device}...")
    model.to(device)
    model.eval()

    # Tokens especiales
    sos = encoding["type_code_map"]["start-of-song"]
    eos = encoding["type_code_map"]["end-of-song"]
    beat_0 = encoding["beat_code_map"][0]
    beat_4 = encoding["beat_code_map"][4]
    beat_16 = encoding["beat_code_map"][16]

    with torch.no_grad():
        data_iter = iter(test_loader)
        for i in tqdm.tqdm(range(args.n_samples), ncols=80):
            try:
                batch = next(data_iter)
            except StopIteration:
                break

            # 1. Ground truth (Original)
            truth_np = batch["seq"][0].numpy()
            save_result(f"{i}_truth", truth_np, sample_dir, encoding)

            # --- FUNCION INTERNA PARA GENERAR Y GUARDAR ---
            def generate_and_save(prefix_tokens, suffix_name):
                # En tu clase MusicLLM, generate() devuelve la secuencia COMPLETA (prefix + new)
                full_seq = model.generate(
                    prefix_tokens.to(device),
                    max_len=args.seq_len - prefix_tokens.shape[1],
                    temperature=args.temperature,
                    eos_token=eos,
                    top_k=args.top_k
                )
                res_np = full_seq[0].cpu().numpy()
                save_result(f"{i}_{suffix_name}", res_np, sample_dir, encoding)

            # 2. Generación Incondicionada (Desde SOS)
            tgt_start = torch.zeros((1, 1, 6), dtype=torch.long, device=device)
            tgt_start[:, 0, 0] = sos
            generate_and_save(tgt_start, "unconditioned")

            # 3. Generación informada por instrumentos (Prefacio)
            # Buscamos donde empieza el primer beat
            prefix_idx = int(np.argmax(batch["seq"][0, :, 1].numpy() >= beat_0))
            if prefix_idx > 0:
                tgt_start = batch["seq"][:1, :prefix_idx]
                generate_and_save(tgt_start, "instrument-informed")

            # 4. Continuación de 4 beats
            cond_4_idx = int(np.argmax(batch["seq"][0, :, 1].numpy() >= beat_4))
            if cond_4_idx > 0:
                tgt_start = batch["seq"][:1, :cond_4_idx]
                generate_and_save(tgt_start, "4-beat-continuation")

if __name__ == "__main__":
    main()