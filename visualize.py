"""
Presentation-ready figures for the ViT plant-disease project.

Everything here saves PNGs (200 dpi) that can be dropped straight into slides.
train.py refreshes the training-curve figure after every epoch; evaluate.py
produces the rest from the held-out test set.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # save to files only; no display needed
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import torch
import torch.nn.functional as F

MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])

# Numbers reported by the base paper on PlantVillage (in %)
PAPER_METRICS = {"Accuracy": 99.77, "Precision": 99.61, "Recall": 99.69, "F1-score": 99.64}

BLUE, ORANGE, GREEN, RED, GREY = "#2E5395", "#E07B39", "#3A9D5D", "#C0392B", "#7F8C8D"

plt.rcParams.update({
    "font.size": 12, "axes.titlesize": 15, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 100,
})


def _save(fig, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def pretty(name: str, maxlen: int = 38) -> str:
    name = name.replace("___", " - ").replace("_", " ").replace(",", "")
    return name if len(name) <= maxlen else name[: maxlen - 1] + "…"


def _denorm(img: torch.Tensor) -> np.ndarray:
    arr = img.detach().cpu().permute(1, 2, 0).numpy() * STD + MEAN
    return np.clip(arr, 0, 1)


# --------------------------------------------------------------------------- #
# 1. Training curves (refreshed every epoch)
# --------------------------------------------------------------------------- #
def plot_training_curves(history, out_path):
    if not history:
        return
    ep = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(ep, [h["train_loss"] for h in history], color=BLUE, lw=2.5, marker="o", ms=4, label="Train")
    axes[0].plot(ep, [h["val_loss"] for h in history], color=ORANGE, lw=2.5, marker="o", ms=4, label="Validation")
    axes[0].set(title="Loss", xlabel="Epoch", ylabel="Cross-entropy loss")
    axes[0].legend()

    tr = [h["train_acc"] * 100 for h in history]
    va = [h["val_acc"] * 100 for h in history]
    axes[1].plot(ep, tr, color=BLUE, lw=2.5, marker="o", ms=4, label="Train")
    axes[1].plot(ep, va, color=ORANGE, lw=2.5, marker="o", ms=4, label="Validation")
    best = int(np.argmax(va))
    axes[1].scatter([ep[best]], [va[best]], s=220, marker="*", color=GREEN, zorder=5,
                    label=f"Best val: {va[best]:.2f}% (epoch {ep[best]})")
    axes[1].set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy (%)")
    axes[1].legend(loc="lower right")

    for ax in axes:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    fig.suptitle("Baseline ViT training on PlantVillage", fontsize=17, fontweight="bold", y=1.02)
    _save(fig, out_path)


# --------------------------------------------------------------------------- #
# 2. Class distribution of the training split
# --------------------------------------------------------------------------- #
def plot_class_distribution(counts, class_names, out_path):
    order = np.argsort(counts)
    fig, ax = plt.subplots(figsize=(10, max(6, 0.28 * len(class_names))))
    ax.barh([pretty(class_names[i]) for i in order], np.array(counts)[order], color=BLUE)
    ax.set(title="Training images per class", xlabel="Number of images")
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="y", visible=False)
    _save(fig, out_path)


# --------------------------------------------------------------------------- #
# 3. Confusion matrix (row-normalised)
# --------------------------------------------------------------------------- #
def plot_confusion_matrix(cm, class_names, out_path):
    cm = np.asarray(cm, dtype=float)
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    n = len(class_names)
    fig, ax = plt.subplots(figsize=(15, 13))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    labels = [pretty(c, 30) for c in class_names]
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set(title="Confusion matrix (each row sums to 100%)", xlabel="Predicted class", ylabel="True class")
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="Fraction of true class")
    _save(fig, out_path)


# --------------------------------------------------------------------------- #
# 4. Per-class F1
# --------------------------------------------------------------------------- #
def plot_per_class_f1(f1_scores, class_names, out_path):
    f1_scores = np.asarray(f1_scores)
    order = np.argsort(f1_scores)
    colors = [GREEN if f1_scores[i] >= 0.99 else ORANGE if f1_scores[i] >= 0.95 else RED for i in order]
    fig, ax = plt.subplots(figsize=(10, max(6, 0.28 * len(class_names))))
    ax.barh([pretty(class_names[i]) for i in order], f1_scores[order] * 100, color=colors)
    ax.axvline(f1_scores.mean() * 100, color=GREY, ls="--", lw=1.5,
               label=f"Macro average: {f1_scores.mean() * 100:.2f}%")
    lo = max(0, min(90, f1_scores.min() * 100 - 3))
    ax.set(title="F1-score per disease class", xlabel="F1-score (%)", xlim=(lo, 100.5))
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower left")
    _save(fig, out_path)


# --------------------------------------------------------------------------- #
# 5. Prediction grids (correct / misclassified examples)
# --------------------------------------------------------------------------- #
def plot_prediction_grid(samples, class_names, out_path, title, cols=4):
    """samples: list of dicts with keys img (normalised CHW tensor), true, pred, conf."""
    if not samples:
        return
    rows = int(np.ceil(len(samples) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 4.4 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, s in zip(axes, samples):
        ax.imshow(_denorm(s["img"]))
        ok = s["true"] == s["pred"]
        ax.set_title(f"True: {pretty(class_names[s['true']], 26)}\n"
                     f"Pred: {pretty(class_names[s['pred']], 26)} ({s['conf'] * 100:.0f}%)",
                     fontsize=9, color=GREEN if ok else RED)
        ax.axis("off")
    for ax in axes[len(samples):]:
        ax.axis("off")
    fig.suptitle(title, fontsize=17, fontweight="bold")
    fig.tight_layout()
    _save(fig, out_path)


# --------------------------------------------------------------------------- #
# 6. Attention rollout: what the model looks at
# --------------------------------------------------------------------------- #
@torch.no_grad()
def attention_rollout(model, x):
    """Attention rollout (Abnar & Zuidema, 2020): multiplies head-averaged attention
    matrices across layers (with residual identity added), then reads the CLS token's
    attention to every image patch. Returns (B, g, g) maps normalised to [0, 1]."""
    model.eval()
    for b in model.encoder_blocks:
        b.save_attn = True
    model(x)
    attn = [b.attn_weights for b in model.encoder_blocks]
    for b in model.encoder_blocks:
        b.save_attn = False
        b.attn_weights = None

    B, N, _ = attn[0].shape
    eye = torch.eye(N, device=x.device).unsqueeze(0)
    rollout = eye.expand(B, -1, -1).clone()
    for a in attn:
        a = a + eye
        a = a / a.sum(dim=-1, keepdim=True)
        rollout = a @ rollout

    mask = rollout[:, 0, 1:]
    g = int(round(mask.shape[1] ** 0.5))
    mask = mask.reshape(B, g, g)
    mask = mask - mask.amin(dim=(1, 2), keepdim=True)
    mask = mask / (mask.amax(dim=(1, 2), keepdim=True) + 1e-8)
    return mask.cpu()


def plot_attention_maps(model, samples, class_names, device, out_path, max_samples=6):
    """Top row: leaf photo. Bottom row: same photo with attention heatmap overlay."""
    samples = samples[:max_samples]
    if not samples:
        return
    x = torch.stack([s["img"] for s in samples]).to(device)
    masks = attention_rollout(model, x)
    H, W = x.shape[-2:]
    masks_up = F.interpolate(masks.unsqueeze(1), size=(H, W), mode="bilinear", align_corners=False)[:, 0].numpy()

    n = len(samples)
    fig, axes = plt.subplots(2, n, figsize=(3.3 * n, 7.2))
    axes = np.asarray(axes).reshape(2, n)
    for i, s in enumerate(samples):
        img = _denorm(s["img"])
        heat = plt.cm.jet(masks_up[i])[..., :3]
        overlay = np.clip(0.55 * img + 0.45 * heat, 0, 1)
        ok = s["true"] == s["pred"]
        axes[0, i].imshow(img)
        axes[0, i].set_title(f"True: {pretty(class_names[s['true']], 24)}\n"
                             f"Pred: {pretty(class_names[s['pred']], 24)}",
                             fontsize=8.5, color=GREEN if ok else RED)
        axes[1, i].imshow(overlay)
        for r in range(2):
            axes[r, i].axis("off")
    axes[0, 0].text(-0.08, 0.5, "Input", transform=axes[0, 0].transAxes, rotation=90,
                    va="center", ha="right", fontsize=13, fontweight="bold")
    axes[1, 0].text(-0.08, 0.5, "Attention", transform=axes[1, 0].transAxes, rotation=90,
                    va="center", ha="right", fontsize=13, fontweight="bold")
    fig.suptitle("Where the ViT looks (attention rollout: red = high attention)", fontsize=16, fontweight="bold")
    fig.tight_layout()
    _save(fig, out_path)


# --------------------------------------------------------------------------- #
# 7. Reproduction vs. base paper
# --------------------------------------------------------------------------- #
def plot_paper_comparison(ours, out_path):
    """ours: dict with keys accuracy, precision, recall, f1 (fractions in [0, 1])."""
    keys = ["Accuracy", "Precision", "Recall", "F1-score"]
    ours_vals = [ours["accuracy"] * 100, ours["precision"] * 100, ours["recall"] * 100, ours["f1"] * 100]
    paper_vals = [PAPER_METRICS[k] for k in keys]

    x = np.arange(len(keys))
    w = 0.36
    fig, ax = plt.subplots(figsize=(10, 5.5))
    b1 = ax.bar(x - w / 2, paper_vals, w, color=GREY, label="Base paper (Ouamane et al., 2025)")
    b2 = ax.bar(x + w / 2, ours_vals, w, color=BLUE, label="Our reproduction")
    for bars in (b1, b2):
        for r in bars:
            ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.03, f"{r.get_height():.2f}",
                    ha="center", va="bottom", fontsize=10)
    lo = np.floor(min(paper_vals + ours_vals)) - 1
    ax.set(xticks=x, xticklabels=keys, ylim=(lo, 100.6), ylabel="%",
           title="Baseline reproduction vs. base paper")
    ax.text(0.99, 0.02, "Note: y-axis is truncated", transform=ax.transAxes, ha="right",
            fontsize=9, color=GREY, style="italic")
    ax.legend(loc="lower left")
    ax.grid(axis="x", visible=False)
    _save(fig, out_path)
