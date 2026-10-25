#!/usr/bin/env python3
"""Audit whether PlantVillage leaf-map.json can be matched to a local ImageFolder dataset.

This script is diagnostic only: it does not modify images, checkpoints, or training code.
It follows the identifier-normalization approach used by the official PlantVillage loader,
then requires the mapping's class label to match the local class-folder name before counting
an image as safely group-mapped.

Example (PowerShell):
  python audit_leaf_map.py --data_dir "D:\\Applications\\Codes\\plant_disease_vit\\datasets\\plantvillage dataset\\color" --leaf_map "D:\\Applications\\Codes\\plant_disease_vit\\leaf-map.json"
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def normalize_identifier(filename: str) -> str:
    """Normalize an image basename as in the official PlantVillage loader."""
    name = filename.replace("_final_masked", "")
    if "___" in name:
        name = name.split("___")[-1]
    name = name.split("copy")[0]
    suffix = Path(name).suffix
    if suffix:
        name = name[: -len(suffix)]
    return name.strip().lower()


def parse_suggestions(raw: Any) -> list[str]:
    if isinstance(raw, list):
        return [str(x) for x in raw if isinstance(x, str)]
    if isinstance(raw, str):
        return [raw]
    return []


def image_files(data_dir: Path) -> list[Path]:
    return sorted(
        p for p in data_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def audit(data_dir: Path, map_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with map_path.open("r", encoding="utf-8") as f:
        leaf_map = json.load(f)
    if not isinstance(leaf_map, dict):
        raise ValueError("leaf-map.json must contain a JSON object mapping image identifiers to suggestions")

    paths = image_files(data_dir)
    if not paths:
        raise ValueError(f"No image files found under: {data_dir}")

    totals = Counter()
    per_class: dict[str, Counter] = defaultdict(Counter)
    group_image_counts = Counter()
    rows: list[dict[str, Any]] = []

    for path in paths:
        # For an ImageFolder layout, the immediate parent directory is the class name.
        class_name = path.parent.name
        key = normalize_identifier(path.name)
        suggestions = parse_suggestions(leaf_map.get(key))
        totals["images"] += 1
        per_class[class_name]["images"] += 1

        matching = []
        for suggestion in suggestions:
            if ":::" not in suggestion:
                continue
            suggested_class, _leaf_part = suggestion.rsplit(":::" , 1)
            if suggested_class == class_name:
                matching.append(suggestion)
        matching = sorted(set(matching))

        if not suggestions:
            status = "no_map_key"
            group_id = ""
            totals["no_map_key"] += 1
            per_class[class_name]["no_map_key"] += 1
        elif len(matching) == 1:
            status = "mapped_class_match"
            group_id = matching[0]
            totals["mapped_class_match"] += 1
            per_class[class_name]["mapped_class_match"] += 1
            group_image_counts[group_id] += 1
        elif len(matching) > 1:
            status = "multiple_matching_groups"
            group_id = ""
            totals["multiple_matching_groups"] += 1
            per_class[class_name]["multiple_matching_groups"] += 1
        else:
            status = "key_found_no_class_match"
            group_id = ""
            totals["key_found_no_class_match"] += 1
            per_class[class_name]["key_found_no_class_match"] += 1

        rows.append({
            "relative_path": str(path.relative_to(data_dir)),
            "class_name": class_name,
            "normalized_identifier": key,
            "map_key_found": bool(suggestions),
            "mapping_status": status,
            "group_id": group_id,
            "suggestions": " | ".join(suggestions),
        })

    classes_report = []
    for class_name in sorted(per_class):
        c = per_class[class_name]
        n = c["images"]
        mapped = c["mapped_class_match"]
        classes_report.append({
            "class_name": class_name,
            "images": n,
            "mapped_class_match": mapped,
            "mapping_coverage_percent": round(mapped / n * 100, 2) if n else 0.0,
            "no_map_key": c["no_map_key"],
            "key_found_no_class_match": c["key_found_no_class_match"],
            "multiple_matching_groups": c["multiple_matching_groups"],
        })

    n = totals["images"]
    n_mapped = totals["mapped_class_match"]
    report: dict[str, Any] = {
        "data_dir": str(data_dir.resolve()),
        "leaf_map": str(map_path.resolve()),
        "leaf_map_keys": len(leaf_map),
        "images_scanned": n,
        "mapped_class_match": n_mapped,
        "mapping_coverage_percent": round(n_mapped / n * 100, 2) if n else 0.0,
        "no_map_key": totals["no_map_key"],
        "key_found_no_class_match": totals["key_found_no_class_match"],
        "multiple_matching_groups": totals["multiple_matching_groups"],
        "unique_matched_group_ids": len(group_image_counts),
        "matched_groups_containing_multiple_images": sum(1 for v in group_image_counts.values() if v > 1),
        "max_images_in_one_matched_group": max(group_image_counts.values(), default=0),
        "class_report": classes_report,
        "interpretation": (
            "This is a coverage audit, not proof of leakage or proof of a leak-free split. "
            "Only images with an exact class-matched leaf-map entry count as group-mapped. "
            "If coverage is incomplete or any class has low coverage, do not treat unmapped images "
            "as safely independent leaves; use the official predefined split or obtain matching metadata."
        ),
    }
    return report, rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", required=True, help="Folder directly containing the 38 class subfolders")
    parser.add_argument("--leaf_map", required=True, help="Path to leaf-map.json")
    parser.add_argument("--report_json", default="leaf_map_audit_report.json")
    parser.add_argument("--details_csv", default="leaf_map_audit_details.csv")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser()
    map_path = Path(args.leaf_map).expanduser()
    if not data_dir.is_dir():
        raise SystemExit(f"ERROR: dataset folder does not exist: {data_dir}")
    if not map_path.is_file():
        raise SystemExit(f"ERROR: leaf map file does not exist: {map_path}")

    report, rows = audit(data_dir, map_path)
    report_path = Path(args.report_json).expanduser()
    csv_path = Path(args.details_csv).expanduser()
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("=== PlantVillage leaf-map coverage audit ===")
    print(f"Images scanned:                     {report['images_scanned']:,}")
    print(f"leaf-map keys:                      {report['leaf_map_keys']:,}")
    print(f"Exact class-matched group entries:  {report['mapped_class_match']:,} ({report['mapping_coverage_percent']:.2f}%)")
    print(f"No map key:                         {report['no_map_key']:,}")
    print(f"Key found but no class match:       {report['key_found_no_class_match']:,}")
    print(f"Multiple matching groups:           {report['multiple_matching_groups']:,}")
    print(f"Unique matched group IDs:            {report['unique_matched_group_ids']:,}")
    print("\nPer-class coverage:")
    for row in report["class_report"]:
        print(f"  {row['class_name']}: {row['mapped_class_match']:,}/{row['images']:,} ({row['mapping_coverage_percent']:.2f}%)")
    print(f"\nJSON report: {report_path.resolve()}")
    print(f"CSV details: {csv_path.resolve()}")
    print("\nNOTE: Low or incomplete coverage means this map alone cannot establish a reliable group-aware split.")


if __name__ == "__main__":
    main()
