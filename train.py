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
import wandb
from dotenv import load_dotenv

import dataset
from compound_transformer import MusicLLM 
import representation
import utils

@utils.resolve_paths
def parse_args(args=None, namespace=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", choices=("sod", "lmd", "lmd_full", "snd"), required=True, help="dataset key")
    parser.add_argument("-t", "--train_names", type=pathlib.Path, help="training names")
    parser.add_argument("-v", "--valid_names", type=pathlib.Path, help="validation names")
    parser.add_argument("-i", "--in_dir", type=pathlib.Path, help="input data directory")
    parser.add_argument("-o", "--out_dir", type=pathlib.Path, help="output directory")
    parser.add_argument("-bs", "--batch_size", default=8, type=int, help="batch size")
    parser.add_argument("--use_csv", action="store_true", help="save in CSV format")
    parser.add_argument("--aug", action=argparse.BooleanOptionalAction, default=True, help="data augmentation")
    parser.add_argument("--max_seq_len", default=1024, type=int, help="max sequence length")
    parser.add_argument("--max_beat", default=256, type=int, help="max beats")
    parser.add_argument("--dim", default=1536, type=int, help="model dimension")
    parser.add_argument("--steps", default=200000, type=int, help="number of steps")
    parser.add_argument("--valid_steps", default=1000, type=int, help="validation frequency")
    parser.add_argument("--early_stopping", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("-e", "--early_stopping_tolerance", default=20, type=int)
    parser.add_argument("-lr", "--learning_rate", default=0.0001, type=float)
    parser.add_argument("--lr_warmup_steps", default=2000, type=int)
    parser.add_argument("--lr_decay_steps", default=150000, type=int)
    parser.add_argument("--lr_decay_multiplier", default=0.1, type=float)
    parser.add_argument("--grad_norm_clip", default=0.5, type=float)
    parser.add_argument("-g", "--gpu", type=int, help="gpu number")
    parser.add_argument("-j", "--jobs", default=4, type=int, help="dataloader workers")
    parser.add_argument("-q", "--quiet", action="store_true")
    parser.add_argument("-m", "--model_name", type=str, default="Qwen/Qwen2.5-1.5B", help="Hugging Face model name")
    parser.add_argument("--resume", action="store_true", help="Continuar desde el último checkpoint")
    parser.add_argument("--start_step", type=int, default=None, help="Forzar el inicio en un paso específico")
    return parser.parse_args(args=args, namespace=namespace)

def get_lr_multiplier(step, warmup_steps, decay_end_steps, decay_end_multiplier):
    if step < warmup_steps: return (step + 1) / warmup_steps
    if step > decay_end_steps: return decay_end_multiplier
    position = (step - warmup_steps) / (decay_end_steps - warmup_steps)
    return 1 - (1 - decay_end_multiplier) * position

def compute_loss(logits_list, targets, mask, n_tokens):
    total_loss = 0
    individual_losses = []
    for i in range(6):
        l_i = F.cross_entropy(logits_list[i].reshape(-1, n_tokens[i]), targets[:, :, i].reshape(-1), reduction='none')
        masked_l_i = (l_i * mask.reshape(-1)).sum() / (mask.sum() + 1e-8)
        total_loss += masked_l_i
        individual_losses.append(masked_l_i)
    return total_loss, individual_losses

def main():
    args = parse_args()
    load_dotenv()

    # Iniciar WandB con opción de resume
    wandb.init(project="music-llm-qwen", config=vars(args), resume="allow")

    if args.dataset is not None:
        if args.train_names is None: args.train_names = pathlib.Path(f"data/{args.dataset}/processed/train-names.txt")
        if args.valid_names is None: args.valid_names = pathlib.Path(f"data/{args.dataset}/processed/valid-names.txt")
        if args.in_dir is None: args.in_dir = pathlib.Path(f"data/{args.dataset}/processed/notes/")
        if args.out_dir is None: args.out_dir = pathlib.Path(f"exp/test_{args.dataset}")

    args.out_dir.mkdir(exist_ok=True, parents=True)
    (args.out_dir / "checkpoints").mkdir(exist_ok=True)

    logging.basicConfig(level=logging.ERROR if args.quiet else logging.INFO, format="%(message)s",
        handlers=[logging.FileHandler(args.out_dir / "train.log", "a" if args.resume else "w"), logging.StreamHandler(sys.stdout)])

    device = torch.device(f"cuda:{args.gpu}" if args.gpu is not None else "cpu")
    encoding = representation.load_encoding("encoding.json")
    n_tokens = encoding['n_tokens']

    # Datasets
    train_dataset = dataset.MusicDataset(args.train_names, args.in_dir, encoding, max_seq_len=args.max_seq_len, max_beat=args.max_beat, use_augmentation=args.aug, use_csv=args.use_csv)
    train_loader = torch.utils.data.DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=args.jobs, collate_fn=dataset.MusicDataset.collate)
    valid_dataset = dataset.MusicDataset(args.valid_names, args.in_dir, encoding, max_seq_len=args.max_seq_len, max_beat=args.max_beat, use_csv=args.use_csv)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, args.batch_size, num_workers=args.jobs, collate_fn=dataset.MusicDataset.collate)

    # 1. Inicializar modelo en CPU primero
    logging.info(f"[*] Inicializando arquitectura del modelo en CPU...")
    model = MusicLLM(model_name_or_path=args.model_name, music_config={"n_tokens": n_tokens}, use_lora=True)
    model.llm_body.gradient_checkpointing_enable()

    # Optimizador y Scheduler iniciales
    lora_params = [p for n, p in model.named_parameters() if "llm_body" in n and p.requires_grad]
    music_params = [p for n, p in model.named_parameters() if "llm_body" not in n and p.requires_grad]
    
    optimizer = torch.optim.AdamW([
        {'params': lora_params, 'lr': args.learning_rate * 0.1},
        {'params': music_params, 'lr': args.learning_rate}
    ], weight_decay=0.01)
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda s: get_lr_multiplier(s, args.lr_warmup_steps, args.lr_decay_steps, args.lr_decay_multiplier))
    scaler = torch.amp.GradScaler('cuda')

    # Variables de estado
    step = args.start_step if args.start_step is not None else 0
    min_val_loss = float("inf")
    count_early_stopping = 0
    
    # 2. Lógica de Resume Segura (CPU -> GPU)
    checkpoint_path = args.out_dir / "checkpoints" / "last_checkpoint.pt"
    if not checkpoint_path.exists():
        checkpoint_path = args.out_dir / "checkpoints" / "best_model.pt"

    if args.resume and checkpoint_path.exists():
        logging.info(f"[*] Cargando pesos desde {checkpoint_path} a la RAM del sistema...")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # FORMATO NUEVO
            model.load_state_dict(checkpoint['model_state_dict'])
            logging.info("[*] Pesos del modelo cargados en CPU. Transfiriendo a GPU...")
            model.to(device)
            
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
            if args.start_step is None:
                step = checkpoint['step']
            min_val_loss = checkpoint.get('min_val_loss', float("inf"))
            count_early_stopping = checkpoint.get('count_early_stopping', 0)
            logging.info(f"[*] Reanudando en el paso {step}")
        else:
            # FORMATO VIEJO
            model.load_state_dict(checkpoint)
            logging.info(f"[*] Formato antiguo. Pesos cargados en CPU. Transfiriendo a GPU...")
            model.to(device)
            logging.info(f"[!] Reanudando desde paso manual: {step}")
    else:
        logging.info(f"[*] Transfiriendo modelo base a {device}...")
        model.to(device)

    loss_csv = open(args.out_dir / "loss.csv", "a" if args.resume else "w")
    if not args.resume:
        loss_csv.write("step,train_loss,valid_loss,type,beat,pos,pitch,dur,inst\n")

    train_iterator = iter(train_loader)

    

    while step < args.steps:
        model.train()
        recent_losses = []
        logging.info(f"Training Step {step}...")
        
        for _ in (pbar := tqdm.tqdm(range(args.valid_steps), ncols=80)):
            try: batch = next(train_iterator)
            except StopIteration: train_iterator = iter(train_loader); batch = next(train_iterator)

            seq, mask = batch["seq"].to(device), batch["mask"].to(device)
            inputs, targets, input_mask = seq[:, :-1, :], seq[:, 1:, :], mask[:, :-1]

            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                # Pasamos los tipos de los targets para que la máscara solo actúe en notas
                logits_list = model(inputs, attention_mask=input_mask, target_types=targets[:, :, 0])
                loss, _ = compute_loss(logits_list, targets, input_mask, n_tokens)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            if step % 50 == 0:
                wandb.log({"train/loss": loss.item(), "train/lr": optimizer.param_groups[1]['lr'], "step": step})
            recent_losses.append(loss.item())
            pbar.set_postfix(loss=f"{np.mean(recent_losses[-10:]):8.4f}")
            step += 1

        # --- VALIDACIÓN ---
        model.eval()
        logging.info("Validating...")
        with torch.no_grad():
            total_loss, total_losses, count = 0, [0]*6, 0
            for batch in valid_loader:
                seq, mask = batch["seq"].to(device), batch["mask"].to(device)
                inputs, targets, input_mask = seq[:, :-1, :], seq[:, 1:, :], mask[:, :-1]
                with torch.amp.autocast('cuda'):
                    logits_list = model(inputs, attention_mask=input_mask, target_types=targets[:, :, 0])
                    loss, ind_losses = compute_loss(logits_list, targets, input_mask, n_tokens)
                
                batch_size = len(batch["seq"])
                total_loss += loss.item() * batch_size
                for idx in range(6): total_losses[idx] += ind_losses[idx].item() * batch_size
                count += batch_size

        val_loss = total_loss / count
        individual_losses = [l / count for l in total_losses]

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
        
        logging.info(f"Step {step} | Val Loss: {val_loss:.4f} | Pitch: {individual_losses[3]:.4f}")
        loss_csv.write(f"{step},{np.mean(recent_losses)},{val_loss}," + ",".join([f"{l:.6f}" for l in individual_losses]) + "\n")
        loss_csv.flush()

        # --- GUARDAR CHECKPOINT COMPLETO ---
        checkpoint_data = {
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'min_val_loss': min_val_loss,
            'count_early_stopping': count_early_stopping
        }
        
        checkpoint_filename = args.out_dir / "checkpoints" / f"model_{step}.pt"
        torch.save(checkpoint_data, checkpoint_filename)
        torch.save(checkpoint_data, args.out_dir / "checkpoints" / "last_checkpoint.pt")

        if val_loss < min_val_loss:
            min_val_loss = val_loss
            shutil.copyfile(checkpoint_filename, args.out_dir / "checkpoints" / "best_model.pt")
            count_early_stopping = 0
            logging.info(f"[*] ¡Nuevo récord! Modelo guardado en best_model.pt")
        elif args.early_stopping:
            count_early_stopping += 1
            if count_early_stopping > args.early_stopping_tolerance: 
                logging.info("Early stopping triggered.")
                break

    wandb.finish(); loss_csv.close()

if __name__ == "__main__": main()