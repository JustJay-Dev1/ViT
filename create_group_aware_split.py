#!/usr/bin/env python3
"""Create reproducible group-aware PlantVillage train/validation/test manifests.

Inputs are the official color_train.txt, color_test.txt and leaf-map.json.
The function mirrors the leaf_id assignment in the PlantVillage Hugging Face
loader, then uses StratifiedGroupKFold with 20 folds to approximate a 70/15/15
image split while keeping each derived leaf_id in exactly one split.

Generated manifests contain paths relative to the local ImageFolder root:
    Class_Name/image.jpg

This script creates manifests/reports only. It does not open images, modify the
dataset, train a model, or change existing checkpoints.
"""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


def read_manifest(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig") as f:
        return [line.strip().replace("\\", "/") for line in f if line.strip()]


def parse_path(line: str) -> tuple[str, str, str]:
    parts = line.split("/")
    if len(parts) < 4 or parts[0].lower() != "raw" or parts[1].lower() != "color":
        raise ValueError(f"Unexpected manifest path: {line}")
    # Existing local data_dir points directly at raw/color, so write class/file.
    return parts[2], parts[3], f"{parts[2]}/{parts[3]}"


def derive_leaf_id(class_name: str, filename: str, leaf_map: dict[str, Any]) -> tuple[str, str]:
    """Mirror leaf_id assignment in the official PlantVillage loader."""
    ident = filename.replace("_final_masked", "")
    if "___" in ident:
        ident = ident.split("___")[-1]
    ident = ident.split("copy")[0]
    for ext in (".jpg", ".JPG", ".png", ".PNG"):
        ident = ident.replace(ext, "")
    ident = ident.strip()
    key = ident.lower().strip()

    if key in leaf_map:
        suggestions = leaf_map[key]
        if isinstance(suggestions, list) and len(suggestions) == 1:
            return str(suggestions[0]), "mapped_single_suggestion"
        if isinstance(suggestions, list):
            match = next((str(s) for s in suggestions if class_name in str(s)), None)
            if match is not None:
                return match, "mapped_class_match"
        return f"fallback_{ident}", "fallback_map_class_mismatch"

    return f"fallback_{ident}", "fallback_map_key_missing"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train_file", required=True, help="Official color_train.txt")
    ap.add_argument("--test_file", required=True, help="Official color_test.txt")
    ap.add_argument("--leaf_map", required=True, help="Official leaf_grouping/leaf-map.json")
    ap.add_argument("--output_dir", default="grouped_split", help="Output split directory")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    train_file, test_file, map_file = map(
        lambda x: Path(x).expanduser().resolve(),
        (args.train_file, args.test_file, args.leaf_map),
    )
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for path, label in ((train_file, "Training manifest"), (test_file, "Test manifest"), (map_file, "Leaf map")):
        if not path.is_file():
            raise SystemExit(f"ERROR: {label} not found: {path}")

    original_train = read_manifest(train_file)
    original_test = read_manifest(test_file)
    if len(original_train) != len(set(original_train)) or len(original_test) != len(set(original_test)):
        raise SystemExit("ERROR: Duplicate paths found within an input manifest.")
    overlap = set(original_train) & set(original_test)
    if overlap:
        raise SystemExit(f"ERROR: The official manifests overlap by {len(overlap)} paths.")
    paths = original_train + original_test

    with map_file.open("r", encoding="utf-8") as f:
        leaf_map = json.load(f)
    if not isinstance(leaf_map, dict):
        raise SystemExit("ERROR: leaf-map JSON root must be an object.")

    rows = []
    for path in paths:
        cls, filename, local_relative_path = parse_path(path)
        leaf_id, assignment_type = derive_leaf_id(cls, filename, leaf_map)
        rows.append({
            "source_path": path,
            "relative_path": local_relative_path,
            "class_name": cls,
            "filename": filename,
            "leaf_id": leaf_id,
            "leaf_id_assignment_type": assignment_type,
        })

    group_classes: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        group_classes[row["leaf_id"]].add(row["class_name"])
    cross_class_groups = {g: sorted(c) for g, c in group_classes.items() if len(c) > 1}
    if cross_class_groups:
        examples = list(cross_class_groups.items())[:5]
        raise SystemExit(
            "ERROR: Some group IDs map to multiple classes; review before splitting. "
            f"Examples: {examples}"
        )

    labels = np.asarray([r["class_name"] for r in rows])
    groups = np.asarray([r["leaf_id"] for r in rows])
    fold_assignment = np.full(len(rows), -1, dtype=np.int32)
    splitter = StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=args.seed)
    for fold_idx, (_, heldout_idx) in enumerate(splitter.split(np.zeros(len(rows)), labels, groups)):
        fold_assignment[heldout_idx] = fold_idx
    if np.any(fold_assignment < 0):
        raise SystemExit("ERROR: Some images were not assigned to a fold.")

    # Three folds (~15%) each for validation and test; remaining fourteen for training.
    # This is close to 70/15/15 and all group IDs are held within one partition.
    for i, row in enumerate(rows):
        fold = int(fold_assignment[i])
        if fold <= 2:
            row["split"] = "val"
        elif fold <= 5:
            row["split"] = "test"
        else:
            row["split"] = "train"

    split_rows = {name: [r for r in rows if r["split"] == name] for name in ("train", "val", "test")}
    split_groups = {name: {r["leaf_id"] for r in split_rows[name]} for name in split_rows}
    overlaps = {
        "train_val": len(split_groups["train"] & split_groups["val"]),
        "train_test": len(split_groups["train"] & split_groups["test"]),
        "val_test": len(split_groups["val"] & split_groups["test"]),
    }
    if any(overlaps.values()):
        raise SystemExit(f"ERROR: Group overlap detected after splitting: {overlaps}")

    all_paths = []
    for name, records in split_rows.items():
        manifest_path = output_dir / f"{name}.txt"
        relative_paths = [r["relative_path"] for r in records]
        if len(relative_paths) != len(set(relative_paths)):
            raise SystemExit(f"ERROR: Duplicate relative paths in {name} split.")
        manifest_path.write_text("\n".join(relative_paths) + "\n", encoding="utf-8")
        all_paths.extend(relative_paths)

    if len(all_paths) != len(set(all_paths)) or len(all_paths) != len(rows):
        raise SystemExit("ERROR: Generated manifests do not form an exact image partition.")

    class_names = sorted(set(labels.tolist()))
    class_counts = {}
    for cls in class_names:
        class_counts[cls] = {
            name: sum(1 for r in split_rows[name] if r["class_name"] == cls)
            for name in ("train", "val", "test")
        }
        class_counts[cls]["total"] = sum(class_counts[cls].values())

    assignment_counts = Counter(r["leaf_id_assignment_type"] for r in rows)
    summary = {
        "source_train_manifest_images": len(original_train),
        "source_test_manifest_images": len(original_test),
        "total_images": len(rows),
        "source_path_overlap": len(overlap),
        "classes": len(class_names),
        "leaf_map_keys": len(leaf_map),
        "unique_leaf_group_ids": len(set(groups.tolist())),
        "group_assignment_methods": dict(assignment_counts),
        "split_sizes": {name: len(split_rows[name]) for name in ("train", "val", "test")},
        "split_percentages": {
            name: round(len(split_rows[name]) / len(rows) * 100, 4)
            for name in ("train", "val", "test")
        },
        "unique_groups_per_split": {name: len(split_groups[name]) for name in split_groups},
        "cross_split_group_overlaps": overlaps,
        "all_38_classes_present_in_each_split": {
            name: len({r["class_name"] for r in split_rows[name]}) == len(class_names)
            for name in split_rows
        },
        "class_counts": class_counts,
        "method": "StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=seed); folds 0-2 validation, 3-5 test, 6-19 train.",
        "group_id_method": "Mirrors leaf_id assignment in official PlantVillage Hugging Face loader. Fallback IDs are derived from normalized filename identifiers when the map does not provide a class-matched assignment.",
        "limitations": "Group disjointness is guaranteed for the derived group IDs, not independently verified biological identity for every image. Treat as metadata-based group-aware partitioning, not field validation.",
        "seed": args.seed,
    }
    (output_dir / "split_report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (output_dir / "split_details.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("\n=== Group-aware PlantVillage split ===")
    for name in ("train", "val", "test"):
        print(f"{name:>5}: {len(split_rows[name]):,} images ({summary['split_percentages'][name]:.2f}%), {len(split_groups[name]):,} groups")
    print(f"Classes: {len(class_names)} in each split")
    print(f"Unique derived groups: {summary['unique_leaf_group_ids']:,}")
    print(f"Cross-split group overlaps: {overlaps}")
    print(f"Output directory: {output_dir}")
    print("Use these manifests only with the grouped dataset loader, not the original random_split loader.")


if __name__ == "__main__":
    main()
