"""Evaluate the saved teacher on the untouched test partition."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from tqdm.auto import tqdm

from data import load_hf_dataset, prepare_or_load_splits, build_dataloaders, split_manifest_sha256
from model import build_teacher


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_id", default="Project-AgML/plant_village_classification")
    parser.add_argument("--cache_dir", default="/content/hf_cache")
    parser.add_argument("--split_manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds, class_names = load_hf_dataset(args.dataset_id, cache_dir=args.cache_dir)
    manifest = prepare_or_load_splits(ds, class_names, args.split_manifest, seed=args.seed, dataset_id=args.dataset_id)
    expected_hash = split_manifest_sha256(args.split_manifest)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if ckpt.get("split_manifest_sha256") != expected_hash:
        raise RuntimeError("Checkpoint split hash differs from evaluation split manifest.")
    if ckpt.get("class_names") != class_names:
        raise RuntimeError("Class ordering in checkpoint differs from current dataset; refusing evaluation.")

    _, _, test_loader = build_dataloaders(
        ds, manifest,
        micro_batch_size=16,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
    )
    model = build_teacher(num_classes=len(class_names)).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    y_true, y_pred = [], []
    with torch.inference_mode():
        for images, labels in tqdm(test_loader, desc="test"):
            images = images.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(images)
            y_true.extend(labels.tolist())
            y_pred.extend(logits.argmax(1).cpu().tolist())

    metrics = {
        "dataset_id": args.dataset_id,
        "num_test_images": len(y_true),
        "num_classes": len(class_names),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "weighted_precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "weighted_recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "classification_report": classification_report(
            y_true, y_pred, labels=list(range(len(class_names))), target_names=class_names,
            output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(len(class_names)))).tolist(),
        "checkpoint_epoch": int(ckpt["epoch"]),
        "checkpoint_val_accuracy": float(ckpt["best_val_acc"]),
        "split_manifest_sha256": expected_hash,
        "split_method": manifest["split_method"],
    }
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "test_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Test accuracy: {metrics['accuracy']*100:.4f}%")
    print(f"Weighted F1: {metrics['weighted_f1']*100:.4f}% | Macro F1: {metrics['macro_f1']*100:.4f}%")
    print(f"Results saved to {out / 'test_metrics.json'}")


if __name__ == "__main__":
    main()
