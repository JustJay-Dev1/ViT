"""ImageFolder data loaders that consume explicit group-aware split manifests."""

from pathlib import Path

from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(image_size: int = 224):
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(int(image_size * 1.14)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, eval_tf


def _normalize_manifest_line(line: str) -> str:
    """Return a case-insensitive ImageFolder-relative path: Class/file.jpg."""
    line = line.strip().replace("\\", "/")
    if not line:
        return ""
    if line.lower().startswith("raw/color/"):
        line = line[len("raw/color/"):]
    parts = line.split("/")
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError(
            f"Invalid split entry {line!r}; expected 'Class_Name/image.jpg' "
            "or 'raw/color/Class_Name/image.jpg'."
        )
    return line.casefold()


def _read_split(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing split manifest: {path}")
    lines = [_normalize_manifest_line(s) for s in path.read_text(encoding="utf-8-sig").splitlines()]
    lines = [s for s in lines if s]
    if len(lines) != len(set(lines)):
        raise ValueError(f"Duplicate image paths found in split manifest: {path}")
    return lines


def build_grouped_dataloaders(
    data_dir: str,
    split_dir: str,
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 4,
):
    """Build train/validation/test loaders from explicit split manifests.

    Required files in split_dir: train.txt, val.txt, test.txt. These manifests
    must partition the local ImageFolder dataset exactly once. Group isolation
    is established by the split-generation script, not inferred here.
    """
    data_root = Path(data_dir).expanduser().resolve()
    split_root = Path(split_dir).expanduser().resolve()
    if not data_root.is_dir():
        raise NotADirectoryError(f"Dataset directory not found: {data_root}")

    train_tf, eval_tf = build_transforms(image_size)
    eval_dataset = datasets.ImageFolder(str(data_root), transform=eval_tf)
    train_aug_dataset = datasets.ImageFolder(str(data_root), transform=train_tf)
    if eval_dataset.classes != train_aug_dataset.classes:
        raise RuntimeError("ImageFolder class ordering differs between datasets.")

    path_to_index = {}
    for index, (image_path, _label) in enumerate(eval_dataset.samples):
        relative = Path(image_path).resolve().relative_to(data_root).as_posix().casefold()
        if relative in path_to_index:
            raise ValueError(f"Duplicate relative image path in ImageFolder: {relative}")
        path_to_index[relative] = index

    split_paths = {
        name: _read_split(split_root / f"{name}.txt")
        for name in ("train", "val", "test")
    }
    normalized_sets = {name: set(paths) for name, paths in split_paths.items()}

    pair_overlaps = {
        "train/val": normalized_sets["train"] & normalized_sets["val"],
        "train/test": normalized_sets["train"] & normalized_sets["test"],
        "val/test": normalized_sets["val"] & normalized_sets["test"],
    }
    nonempty_overlaps = {k: len(v) for k, v in pair_overlaps.items() if v}
    if nonempty_overlaps:
        raise ValueError(f"Split manifests overlap by image path: {nonempty_overlaps}")

    manifest_union = set().union(*normalized_sets.values())
    local_paths = set(path_to_index)
    unknown = sorted(manifest_union - local_paths)
    missing = sorted(local_paths - manifest_union)
    if unknown or missing:
        examples = []
        if unknown:
            examples.append(f"unknown manifest paths={unknown[:3]}")
        if missing:
            examples.append(f"local images omitted from manifests={missing[:3]}")
        raise ValueError(
            "Split manifests must cover every local image exactly once; " + "; ".join(examples)
        )

    indices = {
        name: [path_to_index[p] for p in split_paths[name]]
        for name in ("train", "val", "test")
    }
    train_labels = {eval_dataset.samples[i][1] for i in indices["train"]}
    val_labels = {eval_dataset.samples[i][1] for i in indices["val"]}
    test_labels = {eval_dataset.samples[i][1] for i in indices["test"]}
    all_labels = set(range(len(eval_dataset.classes)))
    for split_name, labels in (("train", train_labels), ("val", val_labels), ("test", test_labels)):
        if labels != all_labels:
            absent = [eval_dataset.classes[i] for i in sorted(all_labels - labels)]
            raise ValueError(f"Some classes are missing from {split_name}: {absent}")

    train_dataset = Subset(train_aug_dataset, indices["train"])
    val_dataset = Subset(eval_dataset, indices["val"])
    test_dataset = Subset(eval_dataset, indices["test"])

    common_loader_args = {
        "num_workers": num_workers,
        "pin_memory": True,
        "persistent_workers": num_workers > 0,
    }
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        **common_loader_args,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        **common_loader_args,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        **common_loader_args,
    )

    print(f"Dataset loaded: {len(eval_dataset):,} images, {len(eval_dataset.classes)} classes")
    print(
        f"  train={len(train_dataset):,}  val={len(val_dataset):,}  "
        f"test={len(test_dataset):,}  (explicit grouped manifests)"
    )
    return train_loader, val_loader, test_loader, eval_dataset.classes
