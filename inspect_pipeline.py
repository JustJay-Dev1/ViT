"""Print verified dataset, preprocessing, split, and batch details for PPT screenshots.

Run from your project root, where dataset.py is located:
    python inspect_pipeline_for_ppt.py --data_dir "D:/datasets/plantvillage dataset/color"

This is an inspection script only. It does NOT train or change the model.
"""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np
from torchvision.datasets import ImageFolder

from dataset import build_dataloaders, build_transforms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Folder containing the 38 class subfolders")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve()
    if not data_dir.is_dir():
        raise SystemExit(f"ERROR: Dataset folder not found: {data_dir}")

    print("=" * 68)
    print("STEP 1 — DATASET INPUT")
    print("=" * 68)
    print(f"Dataset path : {data_dir}")
    raw_ds = ImageFolder(str(data_dir))
    counts = Counter(raw_ds.targets)
    print(f"Image count  : {len(raw_ds):,}")
    print(f"Class count  : {len(raw_ds.classes)}")
    print("Sample class names:")
    for name in raw_ds.classes[:8]:
        print(f"  - {name}")
    if len(raw_ds.classes) > 8:
        print(f"  ... and {len(raw_ds.classes) - 8} more classes")
    print("Images per class (min / max):", min(counts.values()), "/", max(counts.values()))

    print("\n" + "=" * 68)
    print("STEP 2 — PREPROCESSING TRANSFORMS")
    print("=" * 68)
    train_tf, eval_tf = build_transforms(args.image_size)
    print("Training transforms (augmentation is training-only):")
    for step, transform in enumerate(train_tf.transforms, start=1):
        print(f"  {step}. {transform}")
    print("Validation/test transforms:")
    for step, transform in enumerate(eval_tf.transforms, start=1):
        print(f"  {step}. {transform}")

    print("\n" + "=" * 68)
    print("STEP 3 — TRAIN / VALIDATION / TEST SPLIT")
    print("=" * 68)
    train_loader, val_loader, test_loader, class_names = build_dataloaders(
        data_dir=str(data_dir),
        image_size=args.image_size,
        batch_size=args.batch_size,
        val_split=0.15,
        test_split=0.15,
        num_workers=0,
        seed=args.seed,
    )
    print(f"Train samples      : {len(train_loader.dataset):,}")
    print(f"Validation samples : {len(val_loader.dataset):,}")
    print(f"Test samples       : {len(test_loader.dataset):,}")
    print(f"Classes in loader  : {len(class_names)}")
    print(f"Split seed         : {args.seed}")
    print("Split type         : random image-level split (not leaf-grouped)")

    print("\n" + "=" * 68)
    print("STEP 4 — PREPROCESSED TRAINING BATCH")
    print("=" * 68)
    images, labels = next(iter(train_loader))
    print(f"Image batch shape  : {tuple(images.shape)}")
    print(f"Label batch shape  : {tuple(labels.shape)}")
    print(f"Image tensor dtype : {images.dtype}")
    print(f"Normalized range   : {images.min().item():.3f} to {images.max().item():.3f}")
    print("Note: normalized values can be outside [0, 1] because ImageNet normalization is applied.")
    print("\nInspection complete. No model training was run.")


if __name__ == "__main__":
    main()
