"""
PlantVillage data loading.

Expected folder layout (this is how the dataset ships on Kaggle / GitHub —
one folder per class, e.g. spMohanty/PlantVillage-Dataset "color" folder):

    data_dir/
        Apple___Apple_scab/
            img001.jpg
            img002.jpg
            ...
        Apple___Black_rot/
            ...
        Tomato___healthy/
            ...
        ... (38 classes total)

This script takes that single folder and creates train/val/test splits
internally — the raw download does not need to be pre-split.
"""

import torch
from torch.utils.data import DataLoader, random_split
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


def build_dataloaders(
    data_dir: str,
    image_size: int = 224,
    batch_size: int = 32,
    val_split: float = 0.15,
    test_split: float = 0.15,
    num_workers: int = 2,
    seed: int = 42,
    limit_samples: int | None = None,
):
    """Returns (train_loader, val_loader, test_loader, class_names)."""
    train_tf, eval_tf = build_transforms(image_size)

    # Load once with eval transform just to get consistent split indices;
    # we rebuild train subset with train_tf below.
    base_dataset = datasets.ImageFolder(data_dir, transform=eval_tf)
    class_names = base_dataset.classes

    if limit_samples is not None:
        # Fast sanity-check mode: use only a small slice of the data.
        indices = torch.randperm(len(base_dataset), generator=torch.Generator().manual_seed(seed))
        indices = indices[:limit_samples].tolist()
        base_dataset = torch.utils.data.Subset(base_dataset, indices)

    n_total = len(base_dataset)
    n_val = int(n_total * val_split)
    n_test = int(n_total * test_split)
    n_train = n_total - n_val - n_test

    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset, test_subset = random_split(
        base_dataset, [n_train, n_val, n_test], generator=generator
    )

    # Apply train-time augmentation only to the training subset.
    train_dataset_aug = datasets.ImageFolder(data_dir, transform=train_tf)
    if isinstance(base_dataset, torch.utils.data.Subset):
        # limit_samples mode: map subset indices through to the augmented dataset
        underlying_indices = [base_dataset.indices[i] for i in train_subset.indices]
    else:
        underlying_indices = train_subset.indices
    train_subset = torch.utils.data.Subset(train_dataset_aug, underlying_indices)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0, drop_last=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0)

    print(f"Dataset loaded: {n_total} images, {len(class_names)} classes")
    print(f"  train={n_train}  val={n_val}  test={n_test}")

    return train_loader, val_loader, test_loader, class_names
