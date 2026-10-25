"""
Evaluate a trained checkpoint on the held-out test split and generate all
presentation figures.

Usage:
    python evaluate.py --checkpoint checkpoints/best_model.pt --data_dir "<path to 38-class folder>"

Produces (in checkpoints/figures/ by default):
    confusion_matrix.png, per_class_f1.png, correct_predictions.png,
    misclassified_examples.png, attention_maps.png, paper_comparison.png
and eval_results.json with Accuracy / Precision / Recall / F1 / size / latency.
"""

import argparse
import json
import os
import time

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from model import VisionTransformer, count_parameters, model_size_mb
from dataset import build_dataloaders
from utils import get_device, set_seed
import visualize as viz


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--figures_dir", type=str, default=None)
    p.add_argument("--output_json", type=str, default=None)
    return p.parse_args()


@torch.no_grad()
def benchmark_latency(model, device, image_size, runs=50, warmup=10):
    """Single-image (batch=1) inference latency, excluding data loading."""
    x = torch.randn(1, 3, image_size, image_size, device=device)
    for _ in range(warmup):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.time()
    for _ in range(runs):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.time() - start) / runs * 1000


def main():
    args = parse_args()
    device = get_device()
    ck_dir = os.path.dirname(os.path.abspath(args.checkpoint))
    fig_dir = args.figures_dir or os.path.join(ck_dir, "figures")
    out_json = args.output_json or os.path.join(ck_dir, "eval_results.json")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg, class_names = ckpt["config"], ckpt["class_names"]
    set_seed(cfg.get("seed", 42))

    model = VisionTransformer(
        image_size=cfg["image_size"], patch_size=cfg["patch_size"], num_classes=len(class_names),
        dim=cfg["dim"], depth=cfg["depth"], heads=cfg["heads"], mlp_dim=cfg["mlp_dim"],
        dropout=0.0, positional_encoding=cfg["positional_encoding"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Same seed + split ratios as training -> the identical held-out test set.
    _, _, test_loader, _ = build_dataloaders(
        data_dir=args.data_dir, image_size=cfg["image_size"], batch_size=args.batch_size,
        val_split=cfg["val_split"], test_split=cfg["test_split"],
        num_workers=args.num_workers, seed=cfg.get("seed", 42),
        limit_samples=cfg.get("limit_samples"),
    )

    all_preds, all_labels = [], []
    correct_samples, wrong_samples = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            probs = torch.softmax(model(images.to(device)), dim=1).cpu()
            conf, preds = probs.max(dim=1)
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())
            for i in range(images.size(0)):
                s = {"img": images[i], "true": labels[i].item(), "pred": preds[i].item(), "conf": conf[i].item()}
                if s["true"] == s["pred"] and len(correct_samples) < 16:
                    correct_samples.append(s)
                elif s["true"] != s["pred"] and len(wrong_samples) < 16:
                    wrong_samples.append(s)

    n = len(class_names)
    acc = accuracy_score(all_labels, all_preds)
    prec, rec, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="weighted", zero_division=0)
    _, _, f1_per_class, _ = precision_recall_fscore_support(
        all_labels, all_preds, labels=list(range(n)), average=None, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(n)))

    results = {
        "accuracy": acc, "precision_weighted": prec, "recall_weighted": rec, "f1_weighted": f1,
        "f1_macro": float(np.mean(f1_per_class)),
        "num_parameters": count_parameters(model), "model_size_mb": model_size_mb(model),
        "latency_ms_per_image_batch1": benchmark_latency(model, device, cfg["image_size"]),
        "device": str(device), "num_test_images": len(all_labels),
    }

    print("\n=== Test set results ===")
    for k, v in results.items():
        print(f"{k:30s}: {v:.4f}" if isinstance(v, float) else f"{k:30s}: {v}")

    with open(out_json, "w") as f:
        json.dump({**results, "confusion_matrix": cm.tolist(), "class_names": class_names,
                   "f1_per_class": [float(x) for x in f1_per_class]}, f, indent=2)

    # ---- Presentation figures ----
    print("\nGenerating figures ...")
    viz.plot_confusion_matrix(cm, class_names, os.path.join(fig_dir, "confusion_matrix.png"))
    viz.plot_per_class_f1(f1_per_class, class_names, os.path.join(fig_dir, "per_class_f1.png"))
    viz.plot_prediction_grid(correct_samples, class_names, os.path.join(fig_dir, "correct_predictions.png"),
                             "Correct predictions (with model confidence)")
    viz.plot_prediction_grid(wrong_samples, class_names, os.path.join(fig_dir, "misclassified_examples.png"),
                             "Misclassified test images")
    attn_samples = correct_samples[:4] + wrong_samples[:2]
    viz.plot_attention_maps(model, attn_samples, class_names, device, os.path.join(fig_dir, "attention_maps.png"))
    viz.plot_paper_comparison(
        {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1},
        os.path.join(fig_dir, "paper_comparison.png"))

    print(f"Figures saved to: {fig_dir}")
    print(f"Metrics saved to: {out_json}")


if __name__ == "__main__":
    main()
