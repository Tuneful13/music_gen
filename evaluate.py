import argparse
import logging
import pathlib
import pprint
import sys
from collections import defaultdict

import muspy
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
import tqdm

import dataset
from compound_transformer import MusicLLM 
import representation
import utils

@utils.resolve_paths
def parse_args(args=None, namespace=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", choices=("sod", "lmd", "lmd_full", "snd"), required=True)
    parser.add_argument("-n", "--names", type=pathlib.Path)
    parser.add_argument("-i", "--in_dir", type=pathlib.Path)
    parser.add_argument("-o", "--out_dir", type=pathlib.Path)
    parser.add_argument("-ns", "--n_samples", type=int)
    parser.add_argument("--use_csv", action="store_true")
    parser.add_argument("--model_steps", type=int)
    parser.add_argument("--seq_len", default=512, type=int) # Bajado un poco por velocidad
    parser.add_argument("--temperature", default=1.0, type=float)
    parser.add_argument("-g", "--gpu", type=int)
    parser.add_argument("-j", "--jobs", default=0, type=int)
    parser.add_argument("-q", "--quiet", action="store_true")
    return parser.parse_args(args=args, namespace=namespace)


def evaluate(data, encoding, filename, eval_dir):
    # (Se mantiene igual que tu original)
    pathlib.Path(eval_dir / "npy").mkdir(parents=True, exist_ok=True)
    pathlib.Path(eval_dir / "csv").mkdir(parents=True, exist_ok=True)
    pathlib.Path(eval_dir / "json").mkdir(parents=True, exist_ok=True)
    
    np.save(eval_dir / "npy" / f"{filename}.npy", data)
    representation.save_csv_codes(eval_dir / "csv" / f"{filename}.csv", data)
    
    music = representation.decode(data, encoding)
    music.trim(music.resolution * 64)
    music.save(eval_dir / "json" / f"{filename}.json")

    if not music.tracks:
        return {"pitch_class_entropy": np.nan, "scale_consistency": np.nan, "groove_consistency": np.nan}

    return {
        "pitch_class_entropy": muspy.pitch_class_entropy(music),
        "scale_consistency": muspy.scale_consistency(music),
        "groove_consistency": muspy.groove_consistency(music, 4 * music.resolution),
    }

def main():
    args = parse_args()

    if args.dataset is not None:
        if args.names is None: args.names = pathlib.Path(f"data/{args.dataset}/processed/test-names.txt")
        if args.in_dir is None: args.in_dir = pathlib.Path(f"data/{args.dataset}/processed/notes/")
        if args.out_dir is None: args.out_dir = pathlib.Path(f"exp/test_{args.dataset}")

    eval_dir = args.out_dir / "eval"
    eval_dir.mkdir(exist_ok=True, parents=True)

    logging.basicConfig(level=logging.ERROR if args.quiet else logging.INFO, format="%(message)s")

    device = torch.device(f"cuda:{args.gpu}" if args.gpu is not None and torch.cuda.is_available() else "cpu")
    encoding = representation.load_encoding(args.in_dir / "encoding.json")

    logging.info(f"Cargando modelo MusicLLM (Qwen backbone)...")
    model = MusicLLM(music_config=encoding).to(device)
    
    checkpoint_dir = args.out_dir / "checkpoints"
    checkpoint_filename = checkpoint_dir / ("best_model.pt" if args.model_steps is None else f"model_{args.model_steps}.pt")
    
    # Cargar pesos (con bypass para llm_body si es necesario)
    model.load_state_dict(torch.load(checkpoint_filename, map_location=device))
    model.freeze_backbone(True)
    model.eval()

    test_dataset = dataset.MusicDataset(args.names, args.in_dir, encoding, max_seq_len=1024)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, collate_fn=dataset.MusicDataset.collate)

    sos = encoding["type_code_map"]["start-of-song"]
    results = defaultdict(list)
    n_samples = len(test_loader) if args.n_samples is None else args.n_samples
    test_iter = iter(test_loader)
    eos = encoding["type_code_map"]["end-of-song"]

    with torch.no_grad():
        for i in tqdm.tqdm(range(n_samples)):
            # Preparar inicio
            tgt_start = torch.zeros((1, 1, 6), dtype=torch.long, device=device)
            tgt_start[:, 0, 0] = sos
            
            # LLAMADA SIMPLE AL MODELO
            generated_tensor = model.generate(
                tgt_start, 
                max_len=args.seq_len, 
                temperature=args.temperature,
                eos_token=eos,
                top_k=40 # Recomendado para que no genere notas "locas"
            )
            
            generated_np = generated_tensor.cpu().numpy()[0]
            results["unconditioned"].append(evaluate(generated_np, encoding, f"sample_{i}", eval_dir / "unconditioned"))

    # Mostrar resultados finales
    for exp, res in results.items():
        logging.info(f"\n--- Resultados: {exp} ---")
        for key in res[0].keys():
            vals = [r[key] for r in res if not np.isnan(r[key])]
            logging.info(f"{key}: mean={np.mean(vals):.4f}")

if __name__ == "__main__":
    main()