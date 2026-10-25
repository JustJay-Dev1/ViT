"""
Train the baseline ViT (the base paper's optimal configuration) on PlantVillage.

Typical run on an RTX 3050 6GB:
    python train.py --data_dir "D:/datasets/plantvillage dataset/color" --epochs 50 --batch_size 32 --amp

Quick pipeline check first (a few minutes):
    python train.py --data_dir "<path>" --epochs 1 --limit_samples 300

Resume after an interruption (power cut, sleep, Ctrl+C):
    python train.py --data_dir "<path>" --epochs 50 --batch_size 32 --amp --resume checkpoints/last.pt

Outputs (in --output_dir):
    best_model.pt   best checkpoint by validation accuracy (used by evaluate.py)
    last.pt         latest state, for --resume
    history.json    per-epoch loss / accuracy / time
    figures/        training_curves.png (refreshed every epoch), class_distribution.png
"""

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from model import VisionTransformer, count_parameters, model_size_mb
from dataset_grouped import build_grouped_dataloaders
from utils import set_seed, get_device, AverageMeter, Timer
from visualize import plot_training_curves, plot_class_distribution


def parse_args():
    p = argparse.ArgumentParser()
    # Data
    p.add_argument("--data_dir", type=str, required=True,
                   help="Folder that directly contains the 38 class subfolders")
    p.add_argument("--output_dir", type=str, default="./checkpoints_grouped")
    p.add_argument("--split_dir", type=str, required=True,
                   help="Directory containing train.txt, val.txt, and test.txt group-aware manifests")
    p.add_argument("--match_checkpoint_config", type=str, required=True,
                   help="Existing baseline best_model.pt; training settings are copied from its saved config")
    p.add_argument("--val_split", type=float, default=0.15)
    p.add_argument("--test_split", type=float, default=0.15)
    p.add_argument("--limit_samples", type=int, default=None,
                   help="Use only N images in total (fast sanity check)")
    # Model (defaults = base paper's optimal configuration)
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--patch_size", type=int, default=16)
    p.add_argument("--dim", type=int, default=512)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--mlp_dim", type=int, default=1024)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--positional_encoding", type=str, default="sinusoidal",
                   choices=["sinusoidal", "learned"])
    # Training
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--warmup_epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--label_smoothing", type=float, default=0.1)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--amp", action="store_true", help="Mixed precision (GPU only, recommended)")
    p.add_argument("--resume", type=str, default=None, help="Path to last.pt to continue training")
    return p.parse_args()


def make_scheduler(optimizer, epochs, warmup_epochs):
    """Linear warm-up, then cosine decay (never below 1% of the base LR)."""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / max(1, warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
        return max(0.01, 0.5 * (1 + math.cos(math.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)


def make_scaler(enabled):
    if not enabled:
        return None
    try:
        return torch.amp.GradScaler("cuda")
    except AttributeError:  # older PyTorch
        return torch.cuda.amp.GradScaler()


def run_epoch(model, loader, criterion, device, desc, optimizer=None, scaler=None, grad_clip=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    loss_meter, acc_meter = AverageMeter(), AverageMeter()

    bar = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)
    for images, labels in bar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.set_grad_enabled(is_train):
            if scaler is not None:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    if grad_clip:
                        scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if grad_clip:
                        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()

        acc = (outputs.argmax(dim=1) == labels).float().mean().item()
        loss_meter.update(loss.item(), images.size(0))
        acc_meter.update(acc, images.size(0))
        bar.set_postfix(loss=f"{loss_meter.avg:.4f}", acc=f"{acc_meter.avg * 100:.2f}%")

    return loss_meter.avg, acc_meter.avg


def apply_reference_config(args):
    """Copy relevant hyperparameters from the existing baseline checkpoint.

    This makes the grouped-split rerun comparable to the original run while
    retaining the new split directory and output directory.
    """
    reference_path = os.path.abspath(os.path.expanduser(args.match_checkpoint_config))
    if not os.path.isfile(reference_path):
        raise SystemExit(f"Baseline checkpoint not found: {reference_path}")
    checkpoint = torch.load(reference_path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config")
    if not isinstance(config, dict):
        raise SystemExit("The baseline checkpoint has no saved config dictionary.")

    keys = (
        "image_size", "patch_size", "dim", "depth", "heads", "mlp_dim",
        "dropout", "positional_encoding", "batch_size", "epochs",
        "warmup_epochs", "lr", "weight_decay", "label_smoothing",
        "grad_clip", "num_workers", "seed", "amp", "limit_samples",
    )
    copied = []
    for key in keys:
        if key in config:
            setattr(args, key, config[key])
            copied.append(key)
    print("Copied baseline training configuration:", ", ".join(copied))
    print(f"Grouped manifests: {os.path.abspath(args.split_dir)}")
    print(f"New output directory: {os.path.abspath(args.output_dir)}")


def main():
    args = parse_args()
    apply_reference_config(args)
    set_seed(args.seed)
    device = get_device()
    os.makedirs(args.output_dir, exist_ok=True)
    fig_dir = os.path.join(args.output_dir, "figures")

    print(f"Device: {device}")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name} | {props.total_memory / 1024**3:.1f} GB VRAM")
    else:
        print("WARNING: no GPU detected - training will be extremely slow. "
              "Check that you installed the CUDA build of PyTorch.")

    train_loader, val_loader, _, class_names = build_grouped_dataloaders(
        data_dir=args.data_dir,
        split_dir=args.split_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    num_classes = len(class_names)

    # Class distribution figure (good slide: shows dataset imbalance)
    sub = train_loader.dataset
    targets = [sub.dataset.targets[i] for i in sub.indices]
    plot_class_distribution(np.bincount(targets, minlength=num_classes), class_names,
                            os.path.join(fig_dir, "class_distribution.png"))

    model = VisionTransformer(
        image_size=args.image_size, patch_size=args.patch_size, num_classes=num_classes,
        dim=args.dim, depth=args.depth, heads=args.heads, mlp_dim=args.mlp_dim,
        dropout=args.dropout, positional_encoding=args.positional_encoding,
    ).to(device)
    n_params = count_parameters(model)
    print(f"Model: {n_params:,} parameters ({n_params / 1e6:.2f}M) | ~{model_size_mb(model):.2f} MB")

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = make_scheduler(optimizer, args.epochs, args.warmup_epochs)
    scaler = make_scaler(args.amp and device.type == "cuda")

    start_epoch, best_val_acc, history = 1, 0.0, []
    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state_dict"])
        optimizer.load_state_dict(ck["optimizer_state_dict"])
        scheduler.load_state_dict(ck["scheduler_state_dict"])
        if scaler is not None and ck.get("scaler_state_dict"):
            scaler.load_state_dict(ck["scaler_state_dict"])
        start_epoch = ck["epoch"] + 1
        best_val_acc = ck["best_val_acc"]
        history = ck["history"]
        print(f"Resumed from {args.resume}: continuing at epoch {start_epoch}/{args.epochs}")

    epoch_times = []
    with Timer() as total_timer:
        for epoch in range(start_epoch, args.epochs + 1):
            t0 = time.time()
            tr_loss, tr_acc = run_epoch(model, train_loader, criterion, device,
                                        f"Epoch {epoch}/{args.epochs} [train]",
                                        optimizer, scaler, args.grad_clip)
            va_loss, va_acc = run_epoch(model, val_loader, criterion, device,
                                        f"Epoch {epoch}/{args.epochs} [val]  ")
            scheduler.step()
            secs = time.time() - t0
            epoch_times.append(secs)
            eta_h = np.mean(epoch_times) * (args.epochs - epoch) / 3600

            print(f"Epoch {epoch:3d}/{args.epochs} | train loss {tr_loss:.4f} acc {tr_acc * 100:6.2f}% | "
                  f"val loss {va_loss:.4f} acc {va_acc * 100:6.2f}% | "
                  f"lr {optimizer.param_groups[0]['lr']:.2e} | {secs / 60:.1f} min | ETA {eta_h:.1f} h")
            if epoch == start_epoch and device.type == "cuda":
                print(f"  (peak GPU memory: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB "
                      f"of {props.total_memory / 1024**3:.1f} GB)")

            history.append({"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
                            "val_loss": va_loss, "val_acc": va_acc, "seconds": secs})

            if va_acc > best_val_acc:
                best_val_acc = va_acc
                torch.save({"model_state_dict": model.state_dict(), "class_names": class_names,
                            "config": vars(args), "val_acc": va_acc, "epoch": epoch},
                           os.path.join(args.output_dir, "best_model.pt"))
                print(f"  -> new best model saved (val acc {va_acc * 100:.2f}%)")

            torch.save({"model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
                        "epoch": epoch, "best_val_acc": best_val_acc, "history": history,
                        "class_names": class_names, "config": vars(args)},
                       os.path.join(args.output_dir, "last.pt"))
            with open(os.path.join(args.output_dir, "history.json"), "w") as f:
                json.dump(history, f, indent=2)
            plot_training_curves(history, os.path.join(fig_dir, "training_curves.png"))

    print(f"\nFinished. Time this session: {total_timer.elapsed / 3600:.2f} h | "
          f"best validation accuracy: {best_val_acc * 100:.2f}%")
    print(f"Next: python evaluate_grouped.py --checkpoint {os.path.join(args.output_dir, 'best_model.pt')} "
          f"--data_dir \"{args.data_dir}\" --split_dir \"{args.split_dir}\"")


if __name__ == "__main__":
    main()
