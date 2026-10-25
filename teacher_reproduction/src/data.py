"""PlantVillage 39-class data loading and reproducible split manifest.

Dataset source: Project-AgML/plant_village_classification on Hugging Face.
The dataset contains 55,448 rows and 39 labels at the time this bundle was made.
It does not expose leaf_id/group metadata. Therefore, the generated split is
stratified by class and reproducible, but it is NOT claimed to be leaf-grouped.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

DATASET_ID = "Project-AgML/plant_village_classification"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_hf_dataset(dataset_id: str = DATASET_ID, cache_dir: str | None = None):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install the `datasets` package first: pip install datasets") from exc

    ds = load_dataset(dataset_id, split="train", cache_dir=cache_dir)
    if "image" not in ds.column_names or "label" not in ds.column_names:
        raise ValueError(
            f"Expected columns `image` and `label`; received {ds.column_names}. "
            "Stop and inspect the selected dataset revision before training."
        )
    label_feature = ds.features["label"]
    class_names = getattr(label_feature, "names", None)
    if not class_names:
        raise ValueError("The label feature must expose a stable ClassLabel names list.")
    if len(class_names) != 39:
        raise ValueError(f"Expected 39 classes per reference dataset; found {len(class_names)}.")
    if "Background_without_leaves" not in class_names:
        raise ValueError("The expected Background_without_leaves class is missing; inspect the dataset revision before training.")
    if len(ds) != 55_448:
        raise ValueError(
            f"Expected the audited 55,448-row dataset release, but loaded {len(ds):,} rows. "
            "Stop and inspect the dataset revision before generating a split or training."
        )
    return ds, list(class_names)


def _atomic_json_write(path: Path, content: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(content, f, indent=2)
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)


def prepare_or_load_splits(
    ds,
    class_names: list[str],
    manifest_path: str,
    seed: int = 42,
    dataset_id: str = DATASET_ID,
) -> dict[str, Any]:
    """Create or validate a fixed stratified 80/10/10 split over source-row IDs."""
    manifest_file = Path(manifest_path)
    fingerprint = getattr(ds, "_fingerprint", "unavailable")
    source_labels = np.asarray(ds["label"], dtype=np.int64)
    if len(source_labels) != len(ds):
        raise RuntimeError("Dataset label count does not match dataset row count.")

    if manifest_file.exists():
        with manifest_file.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
        checks = {
            "dataset_id": manifest.get("dataset_id") == dataset_id,
            "row_count": manifest.get("row_count") == len(ds),
            "class_names": manifest.get("class_names") == class_names,
            "fingerprint": manifest.get("dataset_fingerprint") == fingerprint,
            "seed": manifest.get("seed") == seed,
            "class_counts": isinstance(manifest.get("class_counts"), dict),
        }
        failed = [key for key, passed in checks.items() if not passed]
        if failed:
            raise RuntimeError(
                "The saved split manifest does not match the currently loaded dataset/config "
                f"({', '.join(failed)}). Do NOT regenerate silently. Inspect the source/revision "
                "and deliberately create a new manifest if required."
            )
        _validate_indices(manifest, len(ds))
        return manifest

    indices = np.arange(len(ds), dtype=np.int64)
    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=0.10,
        random_state=seed,
        shuffle=True,
        stratify=source_labels,
    )
    train_val_labels = source_labels[train_val_idx]
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=1 / 9,  # 10% of total after 10% has been set aside for test
        random_state=seed,
        shuffle=True,
        stratify=train_val_labels,
    )

    split_lists = {
        "train": [int(i) for i in train_idx],
        "validation": [int(i) for i in val_idx],
        "test": [int(i) for i in test_idx],
    }
    split_class_counts = {}
    for split_name, split_indices in split_lists.items():
        counts = np.bincount(source_labels[np.asarray(split_indices, dtype=np.int64)], minlength=len(class_names))
        split_class_counts[split_name] = {class_names[i]: int(counts[i]) for i in range(len(class_names))}
    manifest: dict[str, Any] = {
        "dataset_id": dataset_id,
        "dataset_config": "default single train split, 39 classes",
        "dataset_fingerprint": fingerprint,
        "row_count": len(ds),
        "class_names": class_names,
        "split_method": "stratified row-level 80/10/10; not leaf-grouped",
        "seed": seed,
        "counts": {key: len(value) for key, value in split_lists.items()},
        "class_counts": split_class_counts,
        "splits": split_lists,
        "limitations": [
            "The reference paper's PlantVillage split counts sum to 55,484, while this public 39-class source contains 55,448 rows.",
            "The paper does not publish exact split manifests or a random seed, so this is a reproducible approximate protocol, not the exact original split.",
            "The source does not expose physical-leaf group IDs, so the random stratified split cannot guarantee leaf-level separation.",
        ],
    }
    _validate_indices(manifest, len(ds))
    _atomic_json_write(manifest_file, manifest)
    return manifest


def _validate_indices(manifest: dict[str, Any], row_count: int) -> None:
    splits = manifest.get("splits", {})
    expected = {"train", "validation", "test"}
    if set(splits) != expected:
        raise ValueError(f"Split manifest must contain {sorted(expected)}.")
    arrays = {k: np.asarray(v, dtype=np.int64) for k, v in splits.items()}
    for name, arr in arrays.items():
        if arr.size == 0 or arr.min() < 0 or arr.max() >= row_count:
            raise ValueError(f"Invalid indices in {name} split.")
        if len(np.unique(arr)) != len(arr):
            raise ValueError(f"Duplicate row indices in {name} split.")
    if np.intersect1d(arrays["train"], arrays["validation"]).size:
        raise ValueError("Row-level overlap between train and validation splits.")
    if np.intersect1d(arrays["train"], arrays["test"]).size:
        raise ValueError("Row-level overlap between train and test splits.")
    if np.intersect1d(arrays["validation"], arrays["test"]).size:
        raise ValueError("Row-level overlap between validation and test splits.")
    all_indices = np.concatenate(list(arrays.values()))
    if len(all_indices) != row_count or len(np.unique(all_indices)) != row_count:
        raise ValueError("Split indices must form a partition of all source rows.")


def build_transforms(image_size: int = 224):
    # The paper lists these operations but does not state all parameter values.
    # Scale/translation/shear values below are disclosed implementation choices.
    train_tf = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomRotation(10),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), shear=10),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_tf, eval_tf


class HFImageDataset(Dataset):
    def __init__(self, source_ds, indices: list[int], transform) -> None:
        self.source_ds = source_ds
        self.indices = indices
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        row = self.source_ds[int(self.indices[idx])]
        image = row["image"].convert("RGB")
        image = self.transform(image)
        return image, int(row["label"])


def build_dataloaders(
    ds,
    manifest: dict[str, Any],
    micro_batch_size: int = 16,
    eval_batch_size: int = 32,
    num_workers: int = 2,
):
    train_tf, eval_tf = build_transforms(224)
    train_set = HFImageDataset(ds, manifest["splits"]["train"], train_tf)
    val_set = HFImageDataset(ds, manifest["splits"]["validation"], eval_tf)
    test_set = HFImageDataset(ds, manifest["splits"]["test"], eval_tf)
    common = {
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
    }
    train_loader = DataLoader(
        train_set, batch_size=micro_batch_size, shuffle=True, drop_last=False, **common
    )
    val_loader = DataLoader(
        val_set, batch_size=eval_batch_size, shuffle=False, drop_last=False, **common
    )
    test_loader = DataLoader(
        test_set, batch_size=eval_batch_size, shuffle=False, drop_last=False, **common
    )
    return train_loader, val_loader, test_loader


def split_manifest_sha256(manifest_path: str) -> str:
    h = hashlib.sha256()
    with open(manifest_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()
