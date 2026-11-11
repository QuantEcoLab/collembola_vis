#!/usr/bin/env python3
"""Validate production v7 dedup output against the validated standalone v7 run."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import cv2
import pandas as pd
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def norm_id(value: Any) -> str:
    if value is None or str(value).strip() == "" or str(value).strip().lower() == "nan":
        return ""
    try:
        return str(int(float(value)))
    except Exception:
        return str(value)


def norm_drop(value: Any) -> str:
    parts = [norm_id(part) for part in str(value).split(";")]
    return ";".join(part for part in parts if part)


def pair_decisions(df: pd.DataFrame) -> dict[tuple[int, int], dict[str, str | bool]]:
    out = {}
    for row in df.fillna("").to_dict("records"):
        key = (int(float(row["detection_id_a"])), int(float(row["detection_id_b"])))
        out[key] = {
            "predicted_duplicate": truthy(row.get("predicted_duplicate", False)),
            "action": str(row.get("action", "")),
            "keep_detection_id": norm_id(row.get("keep_detection_id", "")),
            "drop_detection_id": norm_drop(row.get("drop_detection_id", "")),
        }
    return out


def det_id_from_name(path: Path) -> int | None:
    match = re.search(r"_det(\d+)_", path.name)
    if not match:
        return None
    return int(match.group(1))


def validate_no_drop_files(source_dir: Path, drop_ids: set[int]) -> list[str]:
    errors = []
    final_dirs = ["crops", "predicted_crops", "masks", "envelope_predicted_crops", "envelope_masks", "envelope_comparison"]
    for dirname in final_dirs:
        d = source_dir / dirname
        if not d.exists():
            continue
        for path in d.rglob("*"):
            if not path.is_file():
                continue
            det_id = det_id_from_name(path)
            if det_id in drop_ids:
                errors.append(str(path.relative_to(REPO_ROOT)))
    return errors


def validate_contours_fullres(final: pd.DataFrame, image_path: Path) -> list[int]:
    Image.MAX_IMAGE_PIXELS = None
    width, height = Image.open(image_path).size
    bad_ids = []
    for row in final.fillna("").to_dict("records"):
        if not truthy(row.get("has_mask", False)) or not row.get("mask_path"):
            continue
        mask = cv2.imread(str(resolve_path(row["mask_path"])), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            bad_ids.append(int(row["detection_id"]))
            continue
        contours, _ = cv2.findContours((mask > 0).astype("uint8") * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            bad_ids.append(int(row["detection_id"]))
            continue
        contour = max(contours, key=cv2.contourArea)
        x = int(math.floor(float(row["bbox_x1"])))
        y = int(math.floor(float(row["bbox_y1"])))
        contour[:, 0, 0] += x
        contour[:, 0, 1] += y
        xs = contour[:, 0, 0]
        ys = contour[:, 0, 1]
        if xs.min() < 0 or ys.min() < 0 or xs.max() >= width or ys.max() >= height:
            bad_ids.append(int(row["detection_id"]))
    return bad_ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate integrated production v7 dedup against standalone v7")
    parser.add_argument("--current-root", type=Path, default=Path("data/v7_production_integration_regression_current"))
    parser.add_argument("--reference-root", type=Path, default=Path("data/duplicate_classifier_v7_fragment_aware/single_source_regression_v7_after_fix"))
    parser.add_argument("--source", default="C_1_2_HDPE005")
    parser.add_argument("--image-root", type=Path, default=Path("data/final_holdout_v1"))
    args = parser.parse_args()

    source = args.source
    current_dir = resolve_path(args.current_root) / source
    reference_dir = resolve_path(args.reference_root) / source
    image_path = resolve_path(args.image_root) / f"{source}.jpg"
    current_audit = pd.read_csv(current_dir / f"{source}_v7_dedup.csv")
    reference_audit = pd.read_csv(reference_dir / f"{source}_v7_dedup.csv")
    status = pd.read_csv(current_dir / f"{source}_v7_detection_status.csv")
    raw = pd.read_csv(current_dir / f"{source}_segmentation_results_before_v7.csv")
    final = pd.read_csv(current_dir / f"{source}_segmentation_results.csv")

    current_decisions = pair_decisions(current_audit)
    reference_decisions = pair_decisions(reference_audit)
    mismatches = []
    for key in sorted(set(current_decisions) | set(reference_decisions)):
        if current_decisions.get(key) != reference_decisions.get(key):
            mismatches.append({"pair": key, "current": current_decisions.get(key), "reference": reference_decisions.get(key)})

    status_rows = status.fillna("").to_dict("records")
    drop_ids = {int(row["detection_id"]) for row in status_rows if row["final_status"] == "dropped"}
    keep_ids = {int(row["detection_id"]) for row in status_rows if row["final_status"] == "kept"}
    final_ids = {int(v) for v in final["detection_id"].tolist()}
    raw_ids = {int(v) for v in raw["detection_id"].tolist()}

    validation = {
        "source": source,
        "candidate_pairs_match": set(current_decisions) == set(reference_decisions),
        "keep_drop_decisions_match_reference": len(mismatches) == 0,
        "decision_mismatches": mismatches,
        "raw_detection_count": len(raw),
        "final_detection_count": len(final),
        "drop_count": len(drop_ids),
        "keep_count": len(keep_ids),
        "all_drop_ids_removed_from_final_csv": len(drop_ids & final_ids) == 0,
        "all_keep_ids_preserved_in_final_csv": keep_ids <= final_ids,
        "final_count_matches_unique_kept_ids": len(final) == len(keep_ids) == len(final_ids),
        "raw_ids_match_status_ids": raw_ids == {int(row["detection_id"]) for row in status_rows},
        "drop_files_in_final_dirs": validate_no_drop_files(current_dir, drop_ids),
        "bad_fullres_contour_detection_ids": validate_contours_fullres(final, image_path),
        "final_overlay_exists": (current_dir / f"{source}_trunk_seg_overlay.jpg").exists(),
        "raw_overlay_exists": (current_dir / f"{source}_trunk_seg_overlay_before_v7.jpg").exists(),
    }
    validation["all_drop_ids_removed_from_final_artifacts"] = not validation["drop_files_in_final_dirs"]
    validation["all_contours_in_fullres_bounds"] = not validation["bad_fullres_contour_detection_ids"]
    validation["passed"] = all([
        validation["candidate_pairs_match"],
        validation["keep_drop_decisions_match_reference"],
        validation["all_drop_ids_removed_from_final_csv"],
        validation["all_keep_ids_preserved_in_final_csv"],
        validation["final_count_matches_unique_kept_ids"],
        validation["raw_ids_match_status_ids"],
        validation["all_drop_ids_removed_from_final_artifacts"],
        validation["all_contours_in_fullres_bounds"],
        validation["final_overlay_exists"],
        validation["raw_overlay_exists"],
    ])
    out_path = resolve_path(args.current_root) / f"{source}_production_v7_validation.json"
    out_path.write_text(json.dumps(validation, indent=2), encoding="utf-8")
    print(json.dumps(validation, indent=2))
    if not validation["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
