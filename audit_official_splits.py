#!/usr/bin/env python3
"""
Audit PlantVillage official train/test manifests against a local color dataset.

This script does not modify images, model code, checkpoints, or manifests.
It checks:
  1) whether every manifest image exists in the local ImageFolder directory,
  2) whether train/test path lists overlap or omit local images,
  3) whether train/test include all classes,
  4) possible cross-split leaf/group-ID collisions using the leaf_id logic
     from the official PlantVillage Hugging Face dataset loader.

Important:
A fallback-ID collision is a potential same-leaf overlap requiring review,
not proof of model leakage. The output deliberately labels these separately.
"""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def read_manifest(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig") as f:
        return [line.strip().replace("\\", "/") for line in f if line.strip()]


def parse_manifest_path(line: str) -> tuple[str, str]:
    # Expected form: raw/color/<class-name>/<filename>
    parts = line.split("/")
    if len(parts) < 4 or parts[0].lower() != "raw" or parts[1].lower() != "color":
        raise ValueError(f"Unexpected manifest path format: {line}")
    return parts[2], parts[3]


def local_relative_map(data_dir: Path) -> dict[str, Path]:
    """Map case-folded class/filename paths to local paths."""
    out: dict[str, Path] = {}
    for p in data_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            rel = p.relative_to(data_dir).as_posix()
            out[rel.casefold()] = p
    return out


def official_loader_leaf_id(class_name: str, file_name: str, leaf_map: dict[str, Any]) -> tuple[str, str, str]:
    """
    Mirror the leaf_id fallback logic in the official dataset loader.
    Returns (leaf_id, assignment_type, normalized_identifier).
    """
    ident = file_name.replace("_final_masked", "")
    if "___" in ident:
        ident = ident.split("___")[-1]
    ident = ident.split("copy")[0]
    for ext in (".jpg", ".JPG", ".png", ".PNG"):
        ident = ident.replace(ext, "")
    ident = ident.strip()
    lookup_key = ident.lower().strip()

    if lookup_key in leaf_map:
        suggestions = leaf_map[lookup_key]
        if isinstance(suggestions, list) and len(suggestions) == 1:
            # The official loader uses a single suggestion as-is.
            return str(suggestions[0]), "mapped_single_suggestion", ident
        if isinstance(suggestions, list):
            matched = next((str(s) for s in suggestions if class_name in str(s)), None)
            if matched is not None:
                return matched, "mapped_class_match", ident
        return f"fallback_{ident}", "fallback_map_class_mismatch", ident

    return f"fallback_{ident}", "fallback_map_key_missing", ident


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="Local color folder containing the 38 class subfolders")
    ap.add_argument("--train_file", required=True, help="Path to official color_train.txt")
    ap.add_argument("--test_file", required=True, help="Path to official color_test.txt")
    ap.add_argument("--leaf_map", required=True, help="Path to official leaf-map.json")
    ap.add_argument("--output_dir", default="official_split_audit", help="Output folder for audit reports")
    args = ap.parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve()
    train_file = Path(args.train_file).expanduser().resolve()
    test_file = Path(args.test_file).expanduser().resolve()
    map_file = Path(args.leaf_map).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for p, label in ((data_dir, "Dataset folder"), (train_file, "Training manifest"),
                     (test_file, "Test manifest"), (map_file, "Leaf map")):
        if not p.exists():
            raise SystemExit(f"ERROR: {label} not found: {p}")
    if not data_dir.is_dir():
        raise SystemExit(f"ERROR: --data_dir is not a directory: {data_dir}")

    train_lines = read_manifest(train_file)
    test_lines = read_manifest(test_file)
    train_set, test_set = set(train_lines), set(test_lines)
    exact_overlap = train_set & test_set

    with map_file.open("r", encoding="utf-8") as f:
        leaf_map = json.load(f)
    if not isinstance(leaf_map, dict):
        raise SystemExit("ERROR: leaf-map JSON root must be an object.")

    local_map = local_relative_map(data_dir)
    local_keys = set(local_map)

    details: list[dict[str, Any]] = []
    group_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    class_counts = {"train": Counter(), "test": Counter()}
    missing_local = {"train": [], "test": []}
    manifest_local_keys = set()

    for split, lines in (("train", train_lines), ("test", test_lines)):
        for line in lines:
            try:
                class_name, file_name = parse_manifest_path(line)
            except ValueError as e:
                raise SystemExit(str(e)) from e

            rel_key = f"{class_name}/{file_name}".casefold()
            found_path = local_map.get(rel_key)
            if found_path is None:
                missing_local[split].append(line)
            else:
                manifest_local_keys.add(rel_key)

            leaf_id, assignment_type, identifier = official_loader_leaf_id(
                class_name, file_name, leaf_map
            )
            class_counts[split][class_name] += 1

            row = {
                "split": split,
                "manifest_path": line,
                "class_name": class_name,
                "filename": file_name,
                "exists_locally": found_path is not None,
                "leaf_id_from_official_loader_logic": leaf_id,
                "leaf_id_assignment_type": assignment_type,
                "normalized_identifier": identifier,
            }
            details.append(row)
            group_records[leaf_id].append(row)

    local_only_keys = sorted(local_keys - manifest_local_keys)
    overlap_groups = []
    for leaf_id, rows in group_records.items():
        splits = {r["split"] for r in rows}
        if {"train", "test"}.issubset(splits):
            train_rows = [r for r in rows if r["split"] == "train"]
            test_rows = [r for r in rows if r["split"] == "test"]
            types = sorted({r["leaf_id_assignment_type"] for r in rows})
            overlap_groups.append({
                "leaf_id": leaf_id,
                "assignment_types": "|".join(types),
                "train_image_count": len(train_rows),
                "test_image_count": len(test_rows),
                "total_image_count": len(rows),
                "classes": "|".join(sorted({r["class_name"] for r in rows})),
                "train_paths": " | ".join(r["manifest_path"] for r in train_rows),
                "test_paths": " | ".join(r["manifest_path"] for r in test_rows),
            })

    train_unique, test_unique = len(train_set), len(test_set)
    total_manifest_unique = len(train_set | test_set)
    local_images = len(local_map)
    total_classes = sorted(set(class_counts["train"]) | set(class_counts["test"]))

    overlap_by_type = Counter()
    overlap_test_images_by_class = Counter()
    overlap_train_images_by_class = Counter()
    overlap_test_images = 0
    overlap_train_images = 0
    for group in overlap_groups:
        overlap_by_type[group["assignment_types"]] += 1
        overlap_test_images += group["test_image_count"]
        overlap_train_images += group["train_image_count"]
        # Parse per group rows for per-class counts
        for row in group_records[group["leaf_id"]]:
            if row["split"] == "test":
                overlap_test_images_by_class[row["class_name"]] += 1
            else:
                overlap_train_images_by_class[row["class_name"]] += 1

    report = {
        "dataset_dir": str(data_dir),
        "train_manifest": str(train_file),
        "test_manifest": str(test_file),
        "leaf_map": str(map_file),
        "leaf_map_keys": len(leaf_map),
        "local_image_count": local_images,
        "train_manifest_line_count": len(train_lines),
        "train_manifest_unique_paths": train_unique,
        "test_manifest_line_count": len(test_lines),
        "test_manifest_unique_paths": test_unique,
        "exact_path_overlap_between_train_and_test": len(exact_overlap),
        "unique_manifest_paths_union": total_manifest_unique,
        "local_images_missing_from_both_manifests": len(local_only_keys),
        "train_manifest_paths_missing_locally": len(missing_local["train"]),
        "test_manifest_paths_missing_locally": len(missing_local["test"]),
        "train_classes": len(class_counts["train"]),
        "test_classes": len(class_counts["test"]),
        "class_counts": {
            cls: {
                "train": class_counts["train"].get(cls, 0),
                "test": class_counts["test"].get(cls, 0),
                "total": class_counts["train"].get(cls, 0) + class_counts["test"].get(cls, 0),
            }
            for cls in total_classes
        },
        "possible_cross_split_leaf_id_collisions": len(overlap_groups),
        "train_images_in_colliding_leaf_ids": overlap_train_images,
        "test_images_in_colliding_leaf_ids": overlap_test_images,
        "overlap_group_assignment_types": dict(overlap_by_type),
        "test_images_in_colliding_groups_by_class": dict(overlap_test_images_by_class),
        "train_images_in_colliding_groups_by_class": dict(overlap_train_images_by_class),
        "interpretation": (
            "Official documentation describes the provided split as leaf-group-aware, but this script "
            "independently checks the current loader's leaf_id assignment logic. Any cross-split leaf-ID "
            "collision—especially fallback IDs—must be inspected before claiming a fully group-disjoint "
            "evaluation. A collision is a potential overlap, not by itself proof of performance leakage."
        ),
    }

    (output_dir / "official_split_audit_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    with (output_dir / "official_split_audit_details.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(details[0].keys()))
        writer.writeheader()
        writer.writerows(details)

    overlap_fields = [
        "leaf_id", "assignment_types", "train_image_count", "test_image_count",
        "total_image_count", "classes", "train_paths", "test_paths"
    ]
    with (output_dir / "possible_group_overlaps.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=overlap_fields)
        writer.writeheader()
        writer.writerows(sorted(overlap_groups, key=lambda r: r["leaf_id"]))

    with (output_dir / "local_manifest_mismatches.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as f:
        writer = csv.writer(f)
        writer.writerow(["type", "manifest_path_or_local_relative_path"])
        for split, paths in missing_local.items():
            for path in paths:
                writer.writerow([f"{split}_path_missing_locally", path])
        for rel in local_only_keys:
            writer.writerow(["local_image_not_in_manifests", rel])

    print("\n=== Official PlantVillage split audit ===")
    print(f"Local images found: {local_images:,}")
    print(f"Train manifest: {len(train_lines):,} lines / {train_unique:,} unique paths")
    print(f"Test manifest:  {len(test_lines):,} lines / {test_unique:,} unique paths")
    print(f"Exact path overlap train/test: {len(exact_overlap):,}")
    print(f"Unique manifest paths union: {total_manifest_unique:,}")
    print(f"Train paths missing locally: {len(missing_local['train']):,}")
    print(f"Test paths missing locally: {len(missing_local['test']):,}")
    print(f"Local images not in either manifest: {len(local_only_keys):,}")
    print(f"Possible cross-split leaf-ID collisions: {len(overlap_groups):,}")
    print(f"Train images in colliding IDs: {overlap_train_images:,}")
    print(f"Test images in colliding IDs: {overlap_test_images:,}")
    print(f"\nReports saved under: {output_dir}")
    print("Read the JSON report and possible_group_overlaps.csv before creating a new split.")


if __name__ == "__main__":
    main()
