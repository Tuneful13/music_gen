import argparse
import logging
import pathlib
import pprint
import shutil
import sys
import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
import tqdm
import wandb # <--- Integración de WandB

import dataset
from compund_transformer import MusicLLM 
import representation
import utils

@utils.resolve_paths
def parse_args(args=None, namespace=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", choices=("sod", "lmd", "lmd_full", "snd"), required=True, help="dataset key")
    parser.add_argument("-t", "--train_names", type=pathlib.Path, help="training names")
    parser.add_argument("-v", "--valid_names", type=pathlib.Path, help="validation names")
    parser.add_argument("-i", "--in_dir", type=pathlib.Path, help="input data directory")
    parser.add_argument("-o", "--out_dir", type=pathlib.Path, help="output directory")
    # Data
    parser.add_argument("-bs", "--batch_size", default=8, type=int, help="batch size")
    parser.add_argument("--use_csv", action="store_true", help="whether to save outputs in CSV format")
    parser.add_argument("--aug", action=argparse.BooleanOptionalAction, default=True, help="whether to use data augmentation")
    # Model
    parser.add_argument("--max_seq_len", default=1024, type=int, help="maximum sequence length")
    parser.add_argument("--max_beat", default=256, type=int, help="maximum number of beats")
    parser.add_argument("--dim", default=1536, type=int, help="model dimension (Qwen 1.5B default)")
    # Training
    parser.add_argument("--steps", default=200000, type=int, help="number of steps")
    parser.add_argument("--valid_steps", default=1000, type=int, help="validation frequency")
    parser.add_argument("--early_stopping", action=argparse.BooleanOptionalAction, default=True, help="whether to use early stopping")
    parser.add_argument("-e", "--early_stopping_tolerance", default=20, type=int, help="tolerance")
    parser.add_argument("-lr", "--learning_rate", default=0.00005, type=float, help="learning rate")
    parser.add_argument("--lr_warmup_steps", default=5000, type=int, help="warmup steps")
    parser.add_argument("--lr_decay_steps", default=100000, type=int, help="decay end steps")
    parser.add_argument("--lr_decay_multiplier", default=0.1, type=float, help="multiplier")
    parser.add_argument("--grad_norm_clip", default=1.0, type=float, help="gradient norm clipping")
    # Others
    parser.add_argument("-g", "--gpu", type=int, help="gpu number")
    parser.add_argument("-j", "--jobs", default=4, type=int, help="number of workers")
    parser.add_argument("-q", "--quiet", action="store_true", help="show warnings only")
    return parser.parse_args(args=args, namespace=namespace)

def get_lr_multiplier(step, warmup_steps, decay_end_steps, decay_end_multiplier):
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    if step > decay_end_steps:
        return decay_end_multiplier
    position = (step - warmup_steps) / (decay_end_steps - warmup_steps)
    return 1 - (1 - decay_end_multiplier) * position

def compute_loss(logits_list, targets, mask, n_tokens):
    """Calcula la pérdida combinada para los 6 atributos con máscara."""
    total_loss = 0
    individual_losses = []
    for i in range(6):
        l_i = F.cross_entropy(
            logits_list[i].reshape(-1, n_tokens[i]), 
            targets[:, :, i].reshape(-1), 
            reduction='none'
        )
        masked_l_i = (l_i * mask.reshape(-1)).sum() / (mask.sum() + 1e-8)
        total_loss += masked_l_i
        individual_losses.append(masked_l_i)
    return total_loss, individual_losses

def main():
    
    args = parse_args()

    wandb.init(project="music-llm-mmt", config=vars(args))

    # Configuración de rutas
    if args.dataset is not None:
        if args.train_names is None: args.train_names = pathlib.Path(f"data/{args.dataset}/processed/train-names.txt")
        if args.valid_names is None: args.valid_names = pathlib.Path(f"data/{args.dataset}/processed/valid-names.txt")
        if args.in_dir is None: args.in_dir = pathlib.Path(f"data/{args.dataset}/processed/notes/")
        if args.out_dir is None: args.out_dir = pathlib.Path(f"exp/test_{args.dataset}")

    args.out_dir.mkdir(exist_ok=True, parents=True)
    (args.out_dir / "checkpoints").mkdir(exist_ok=True)

    logging.basicConfig(
        level=logging.ERROR if args.quiet else logging.INFO,
        format="%(message)s",
        handlers=[logging.FileHandler(args.out_dir / "train.log", "w"), logging.StreamHandler(sys.stdout)],
    )

    device = torch.device(f"cuda:{args.gpu}" if args.gpu is not None else "cpu")
    
    # Carga de Encoding
    encoding_path = args.in_dir / "encoding.json"
    encoding = representation.load_encoding(encoding_path)
    n_tokens = encoding['n_tokens'] 
    logging.info(f"Vocabulario detectado: {n_tokens}")

    # Datasets
    train_dataset = dataset.MusicDataset(args.train_names, args.in_dir, encoding, max_seq_len=args.max_seq_len, max_beat=args.max_beat, use_augmentation=args.aug, use_csv=args.use_csv)
    train_loader = torch.utils.data.DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=args.jobs, collate_fn=dataset.MusicDataset.collate)
    
    valid_dataset = dataset.MusicDataset(args.valid_names, args.in_dir, encoding, max_seq_len=args.max_seq_len, max_beat=args.max_beat, use_csv=args.use_csv)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, args.batch_size, num_workers=args.jobs, collate_fn=dataset.MusicDataset.collate)

    # Modelo
    logging.info(f"Cargando modelo...")
    model = MusicLLM(music_config={"n_tokens": n_tokens}).to(device)
    model.freeze_backbone(True)

    n_trainables = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Parámetros entrenables: {n_trainables}")

    # Optimizador
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), args.learning_rate)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: get_lr_multiplier(step, args.lr_warmup_steps, args.lr_decay_steps, args.lr_decay_multiplier))

    # CSV de Loss local
    loss_csv = open(args.out_dir / "loss.csv", "w")
    loss_csv.write("step,train_loss,valid_loss,type,beat,pos,pitch,dur,inst\n")

    step, min_val_loss = 0, float("inf")
    count_early_stopping = 0
    train_iterator = iter(train_loader)

    while step < args.steps:
        model.train()
        recent_losses = []

        logging.info(f"Training Step {step}...")
        for _ in (pbar := tqdm.tqdm(range(args.valid_steps), ncols=80)):
            try:
                batch = next(train_iterator)
            except StopIteration:
                train_iterator = iter(train_loader)
                batch = next(train_iterator)

            seq = batch["seq"].to(device)
            mask = batch["mask"].to(device)

            # Shifting autoregresivo
            inputs, targets, input_mask = seq[:, :-1, :], seq[:, 1:, :], mask[:, :-1]

            optimizer.zero_grad()
            logits_list = model(inputs, attention_mask=input_mask)
            loss, _ = compute_loss(logits_list, targets, input_mask, n_tokens)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm_clip)
            optimizer.step()
            scheduler.step()

            # WandB Train Log
            wandb.log({"train/loss": loss.item(), "train/lr": scheduler.get_last_lr()[0], "step": step})

            recent_losses.append(float(loss))
            if len(recent_losses) > 10: del recent_losses[0]
            pbar.set_postfix(loss=f"{np.mean(recent_losses):8.4f}")
            step += 1

        # Validación
        model.eval()
        logging.info("Validating...")
        with torch.no_grad():
            total_loss, total_losses = 0, [0] * 6
            count = 0
            for batch in valid_loader:
                seq, mask = batch["seq"].to(device), batch["mask"].to(device)
                inputs, targets, input_mask = seq[:, :-1, :], seq[:, 1:, :], mask[:, :-1]

                logits_list = model(inputs, attention_mask=input_mask)
                loss, ind_losses = compute_loss(logits_list, targets, input_mask, n_tokens)

                total_loss += float(loss) * len(batch)
                for idx in range(6): total_losses[idx] += float(ind_losses[idx]) * len(batch)
                count += len(batch)

        val_loss = total_loss / count
        individual_losses = [l / count for l in total_losses]

        # WandB Val Log
        wandb.log({
            "val/loss": val_loss,
            "val/type_loss": individual_losses[0],
            "val/beat_loss": individual_losses[1],
            "val/pos_loss": individual_losses[2],
            "val/pitch_loss": individual_losses[3],
            "val/dur_loss": individual_losses[4],
            "val/inst_loss": individual_losses[5],
            "step": step
        })

        logging.info(f"Step {step} | Val Loss: {val_loss:.4f} | Pitch Loss: {individual_losses[3]:.4f}")
        
        # Guardar Localmente
        loss_csv.write(f"{step},{np.mean(recent_losses)},{val_loss}," + ",".join([f"{l}" for l in individual_losses]) + "\n")
        checkpoint_filename = args.out_dir / "checkpoints" / f"model_{step}.pt"
        torch.save(model.state_dict(), checkpoint_filename)

        if val_loss < min_val_loss:
            min_val_loss = val_loss
            shutil.copyfile(checkpoint_filename, args.out_dir / "checkpoints" / "best_model.pt")
            count_early_stopping = 0
        elif args.early_stopping:
            count_early_stopping += 1
            if count_early_stopping > args.early_stopping_tolerance:
                logging.info("Early stopping!"); break

    wandb.finish()
    loss_csv.close()

if __name__ == "__main__":
    main()