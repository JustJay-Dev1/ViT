"""Train the teacher with paper-disclosed settings and resumable checkpoints."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score
from torch import nn
from torch.optim import AdamW
from tqdm.auto import tqdm

from data import load_hf_dataset, prepare_or_load_splits, build_dataloaders, split_manifest_sha256
from model import build_teacher, count_parameters


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_torch_save(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def save_rng_state() -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def run_validation(model, loader, criterion, device, amp: bool):
    model.eval()
    losses, all_targets, all_preds = [], [], []
    with torch.inference_mode():
        for images, labels in tqdm(loader, desc="validation", leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if amp and device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(images)
                    loss = criterion(logits, labels)
            else:
                logits = model(images)
                loss = criterion(logits, labels)
            losses.append(float(loss.item()) * images.size(0))
            all_targets.extend(labels.cpu().tolist())
            all_preds.extend(logits.argmax(dim=1).cpu().tolist())
    return sum(losses) / len(loader.dataset), accuracy_score(all_targets, all_preds), all_targets, all_preds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_id", default="Project-AgML/plant_village_classification")
    parser.add_argument("--cache_dir", default="/content/hf_cache")
    parser.add_argument("--split_manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--results_dir", default=None, help="Directory for history.json (defaults to output_dir)")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--micro_batch_size", type=int, default=16)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--resume", default="auto", help="auto, none, or explicit checkpoint path")
    args = parser.parse_args()

    if args.epochs < 1 or args.micro_batch_size < 1 or args.grad_accum_steps < 1:
        raise ValueError("epochs, micro_batch_size and grad_accum_steps must be positive.")
    if args.save_every < 1:
        raise ValueError("save_every must be positive.")

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("No CUDA GPU is available. Stop here rather than accidentally training on CPU.")
    print(f"Device: {device} | GPU: {torch.cuda.get_device_name(0)}")
    print(f"Dataset source: {args.dataset_id}")

    ds, class_names = load_hf_dataset(args.dataset_id, cache_dir=args.cache_dir)
    manifest = prepare_or_load_splits(ds, class_names, args.split_manifest, seed=args.seed, dataset_id=args.dataset_id)
    manifest_hash = split_manifest_sha256(args.split_manifest)
    print(f"Dataset: {len(ds):,} images | {len(class_names)} classes")
    print(f"Split counts: {manifest['counts']}")
    print("NOTE: row-level stratified split, not leaf-grouped; see README.md.")

    train_loader, val_loader, _ = build_dataloaders(
        ds, manifest,
        micro_batch_size=args.micro_batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
    )
    model = build_teacher(num_classes=len(class_names)).to(device)
    parameter_count = count_parameters(model)
    print(f"Teacher parameters: {parameter_count:,} ({parameter_count / 1e6:.3f}M)")
    effective_batch = args.micro_batch_size * args.grad_accum_steps
    print(f"Micro-batch={args.micro_batch_size}; accumulation={args.grad_accum_steps}; "
          f"effective batch={effective_batch}")
    print("Optimizer: AdamW(lr=1e-4); weight_decay=0.01 (PyTorch default; not specified in paper)")
    print("AMP: fp16 | LR schedule: fixed (no scheduler reported in paper)")

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=1e-4)  # weight_decay follows PyTorch default 0.01
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    last_path = output_dir / "last.pt"
    best_path = output_dir / "best_model.pt"
    results_dir = Path(args.results_dir) if args.results_dir else output_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    history_path = results_dir / "history.json"
    best_val_acc = -1.0
    history = []
    start_epoch = 1
    ckpt_path = None
    if args.resume == "auto" and last_path.exists():
        ckpt_path = last_path
    elif args.resume not in ("auto", "none"):
        ckpt_path = Path(args.resume)

    if ckpt_path is not None:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if ckpt.get("split_manifest_sha256") != manifest_hash:
            raise RuntimeError("Checkpoint split hash does not match the current manifest; refusing unsafe resume.")
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = int(ckpt["epoch"]) + 1
        best_val_acc = float(ckpt["best_val_acc"])
        history = ckpt.get("history", [])
        restore_rng_state(ckpt.get("rng_state", {}))
        if best_path.exists():
            best_ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
            if best_ckpt.get("split_manifest_sha256") != manifest_hash:
                raise RuntimeError("Best checkpoint split hash does not match the current manifest.")
            best_val_acc = max(best_val_acc, float(best_ckpt.get("best_val_acc", -1.0)))
        print(f"Resuming from {ckpt_path} at epoch {start_epoch}/{args.epochs}; best validation accuracy so far={best_val_acc*100:.2f}%")
    elif args.resume == "none" and last_path.exists():
        raise RuntimeError(
            "A checkpoint already exists. To avoid accidental overwrite, move it aside or use --resume auto."
        )

    if start_epoch > args.epochs:
        print(f"Training is already complete: checkpoint epoch {start_epoch - 1} >= target {args.epochs}.")
        return

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        epoch_start = time.time()
        running_loss = 0.0
        correct = 0
        total = 0
        optimizer.zero_grad(set_to_none=True)
        loader_len = len(train_loader)

        progress = tqdm(enumerate(train_loader), total=loader_len,
                        desc=f"Epoch {epoch}/{args.epochs}", leave=True)
        for step, (images, labels) in progress:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            group_start = (step // args.grad_accum_steps) * args.grad_accum_steps
            group_size = min(args.grad_accum_steps, loader_len - group_start)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(images)
                loss = criterion(logits, labels)
                scaled_loss = loss / group_size
            scaler.scale(scaled_loss).backward()

            should_step = ((step + 1) % args.grad_accum_steps == 0) or (step + 1 == loader_len)
            if should_step:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            batch_n = images.size(0)
            running_loss += float(loss.item()) * batch_n
            correct += int((logits.detach().argmax(1) == labels).sum().item())
            total += batch_n
            if (step + 1) % 20 == 0 or step + 1 == loader_len:
                progress.set_postfix(
                    loss=f"{running_loss / max(1,total):.4f}",
                    acc=f"{100*correct/max(1,total):.2f}%",
                    lr=f"{optimizer.param_groups[0]['lr']:.1e}",
                )

        train_loss = running_loss / max(total, 1)
        train_acc = correct / max(total, 1)
        val_loss, val_acc, _, _ = run_validation(model, val_loader, criterion, device, amp=True)
        epoch_seconds = time.time() - epoch_start
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "validation_loss": val_loss,
            "validation_accuracy": val_acc,
            "seconds": epoch_seconds,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(record)
        print(
            f"Epoch {epoch}/{args.epochs} | train loss={train_loss:.4f}, acc={train_acc*100:.2f}% | "
            f"val loss={val_loss:.4f}, acc={val_acc*100:.2f}% | {epoch_seconds/60:.1f} min"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            safe_torch_save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": class_names,
                    "config": model.config,
                    "optimizer_name": "AdamW",
                    "learning_rate": 1e-4,
                    "epoch": epoch,
                    "best_val_acc": best_val_acc,
                    "split_manifest_sha256": manifest_hash,
                    "training_config": vars(args),
                    "dataset_id": args.dataset_id,
                    "dataset_fingerprint": manifest["dataset_fingerprint"],
                    "split_counts": manifest["counts"],
                    "split_method": manifest["split_method"],
                },
                best_path,
            )
            print(f"  -> saved new best model (val acc {best_val_acc*100:.2f}%)")

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "epoch": epoch,
            "best_val_acc": best_val_acc,
            "history": history,
            "class_names": class_names,
            "config": vars(args),
            "split_manifest_sha256": manifest_hash,
            "rng_state": save_rng_state(),
        }
        if epoch == start_epoch or epoch % args.save_every == 0 or epoch == args.epochs:
            safe_torch_save(checkpoint, last_path)
            print(f"  -> resumable checkpoint saved at epoch {epoch}")
        with history_path.open("w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    print(f"Finished. Best validation accuracy: {best_val_acc*100:.2f}%")
    print(f"Best checkpoint: {best_path}")
    print("Next step: evaluate the held-out test split exactly once using src/evaluate.py")


if __name__ == "__main__":
    main()
