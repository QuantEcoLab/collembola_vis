#!/usr/bin/env python3
"""Generate audit artifacts for the v7 final end-to-end holdout run."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def rel_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def normalize_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def draw_bbox_overlay(image_path: Path, rows: pd.DataFrame, output_path: Path, title: str) -> None:
    Image.MAX_IMAGE_PIXELS = None
    image_bgr = cv2.cvtColor(np.array(Image.open(image_path).convert("RGB")), cv2.COLOR_RGB2BGR)
    overlay = image_bgr.copy()
    for _, row in rows.iterrows():
        x1 = int(math.floor(float(row["x1"])))
        y1 = int(math.floor(float(row["y1"])))
        x2 = int(math.ceil(float(row["x2"])))
        y2 = int(math.ceil(float(row["y2"])))
        det_id = int(row["source_detection_id"])
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 220, 255), 2, cv2.LINE_AA)
        cv2.putText(overlay, str(det_id), (x1 + 4, max(24, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 255), 2, cv2.LINE_AA)
    cv2.putText(overlay, title, (32, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 220, 255), 3, cv2.LINE_AA)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), overlay, [cv2.IMWRITE_PNG_COMPRESSION, 3])


def contour_from_mask(mask: np.ndarray) -> np.ndarray | None:
    contours, _ = cv2.findContours((mask > 0).astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def draw_final_contour_overlay(image_path: Path, rows: pd.DataFrame, output_path: Path, title: str) -> None:
    Image.MAX_IMAGE_PIXELS = None
    image_bgr = cv2.cvtColor(np.array(Image.open(image_path).convert("RGB")), cv2.COLOR_RGB2BGR)
    overlay = image_bgr.copy()
    for row in rows.to_dict("records"):
        x1 = int(math.floor(float(row["bbox_x1"])))
        y1 = int(math.floor(float(row["bbox_y1"])))
        x2 = int(math.ceil(float(row["bbox_x2"])))
        y2 = int(math.ceil(float(row["bbox_y2"])))
        det_id = int(row["detection_id"])
        color = (0, 255, 0)
        if normalize_bool(row.get("has_mask", False)) and str(row.get("mask_path", "")):
            mask = cv2.imread(str(resolve_path(row["mask_path"])), cv2.IMREAD_GRAYSCALE)
            contour = contour_from_mask(mask) if mask is not None else None
            if contour is not None:
                global_contour = contour.copy()
                global_contour[:, 0, 0] += x1
                global_contour[:, 0, 1] += y1
                cv2.drawContours(overlay, [global_contour], -1, color, 2, cv2.LINE_AA)
            else:
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        else:
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        cv2.putText(overlay, str(det_id), (x1 + 4, max(24, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
    cv2.putText(overlay, title, (32, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 0), 3, cv2.LINE_AA)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), overlay, [cv2.IMWRITE_PNG_COMPRESSION, 3])


def load_summary(output_root: Path) -> list[dict[str, Any]]:
    summary_path = output_root / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return summary.get("images", [])


def source_name(item: dict[str, Any]) -> str:
    return Path(item["output_dir"]).name


def write_viewer_script(output_root: Path, sources: list[str]) -> Path:
    script_path = output_root / "view_fullres_overlays.py"
    sources_literal = repr(sources)
    script_path.write_text(
        f"""#!/usr/bin/env python3
from pathlib import Path
import cv2

ROOT = Path(__file__).resolve().parent
SOURCES = {sources_literal}

for source in SOURCES:
    for name in [f"{{source}}_before_v7_fullres.png", f"{{source}}_after_v7_fullres.png"]:
        path = ROOT / "final_artifacts" / source / name
        img = cv2.imread(str(path))
        if img is None:
            print(f"Missing: {{path}}")
            continue
        preview = cv2.resize(img, None, fx=0.12, fy=0.12, interpolation=cv2.INTER_AREA)
        cv2.imshow(str(path), preview)
        print(f"Showing {{path}}. Press any key for next image, Esc to stop.")
        key = cv2.waitKey(0)
        cv2.destroyAllWindows()
        if key == 27:
            raise SystemExit
""",
        encoding="utf-8",
    )
    return script_path


def run_measurements(image_path: Path, detections_csv: Path, output_csv: Path, um_per_pixel: float) -> None:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "measure_organisms_fast.py"),
        "--image",
        str(image_path),
        "--detections",
        str(detections_csv),
        "--output",
        str(output_csv),
        "--um-per-pixel",
        str(um_per_pixel),
    ]
    subprocess.run(cmd, check=True)


def process_source(item: dict[str, Any], artifacts_root: Path, um_per_pixel: float) -> dict[str, Any]:
    source = source_name(item)
    source_dir = resolve_path(item["output_dir"])
    image_path = resolve_path(item["image"])
    out_dir = artifacts_root / source
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_csv = source_dir / "detections" / f"{source}_detections.csv"
    final_csv = source_dir / f"{source}_segmentation_results.csv"
    audit_csv = source_dir / f"{source}_v7_dedup.csv"
    raw = pd.read_csv(raw_csv).reset_index().rename(columns={"index": "source_detection_id"})
    final = pd.read_csv(final_csv)
    audit = pd.read_csv(audit_csv)

    duplicate_rows = audit[audit["action"].eq("auto_remove")].copy()
    drop_ids = sorted({int(v) for v in duplicate_rows["drop_detection_id"].dropna().tolist()})
    keep_decision_ids = sorted({int(v) for v in duplicate_rows["keep_detection_id"].dropna().tolist()})
    final_ids = sorted(int(v) for v in final["detection_id"].tolist())
    raw_ids = sorted(int(v) for v in raw["source_detection_id"].tolist())
    kept_ids = sorted(set(raw_ids) - set(drop_ids))

    pd.DataFrame({"source_detection_id": kept_ids}).to_csv(out_dir / f"{source}_kept_ids.csv", index=False)
    pd.DataFrame({"source_detection_id": drop_ids}).to_csv(out_dir / f"{source}_drop_ids.csv", index=False)
    duplicate_rows.to_csv(out_dir / f"{source}_drop_decisions.csv", index=False)
    raw.to_csv(out_dir / f"{source}_raw_detections_before_v7.csv", index=False)
    final.to_csv(out_dir / f"{source}_final_segmentation_after_v7.csv", index=False)

    measurement_input = raw[raw["source_detection_id"].isin(final_ids)].copy()
    measurement_input.to_csv(out_dir / f"{source}_measurement_input_yolo.csv", index=False)
    measurements_csv = out_dir / f"{source}_measurements_kept.csv"
    run_measurements(image_path, out_dir / f"{source}_measurement_input_yolo.csv", measurements_csv, um_per_pixel)
    measurements = pd.read_csv(measurements_csv)
    measurements.insert(1, "source_detection_id", measurement_input["source_detection_id"].to_list())
    measurements.to_csv(measurements_csv, index=False)

    before_rows = raw.copy()
    after_rows = raw[raw["source_detection_id"].isin(final_ids)].copy()
    draw_bbox_overlay(image_path, before_rows, out_dir / f"{source}_before_v7_fullres.png", f"{source} before v7: {len(before_rows)}")
    draw_final_contour_overlay(image_path, final, out_dir / f"{source}_after_v7_fullres.png", f"{source} after v7: {len(final)}")

    Image.MAX_IMAGE_PIXELS = None
    width, height = Image.open(image_path).size
    bbox_errors = []
    for row in final.to_dict("records"):
        x1, y1, x2, y2 = [float(row[k]) for k in ["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]]
        if x1 < 0 or y1 < 0 or x2 > width or y2 > height or x2 <= x1 or y2 <= y1:
            bbox_errors.append(int(row["detection_id"]))
    missing_final_ids = sorted(set(kept_ids) - set(final_ids))
    unexpected_final_ids = sorted(set(final_ids) - set(kept_ids))
    drop_ids_in_final = sorted(set(drop_ids) & set(final_ids))

    validation = {
        "source": source,
        "image_path": rel_path(image_path),
        "raw_detections_before_v7": len(raw),
        "final_detections_after_v7": len(final),
        "drop_ids": drop_ids,
        "drop_count": len(drop_ids),
        "kept_count": len(kept_ids),
        "keep_decision_ids": keep_decision_ids,
        "candidate_pairs": int(len(audit)),
        "predicted_duplicate_pairs": int(len(duplicate_rows)),
        "unsafe_groups": int(audit["unsafe_group"].fillna(False).map(normalize_bool).sum()),
        "image_width": width,
        "image_height": height,
        "bbox_out_of_bounds_detection_ids": bbox_errors,
        "drop_ids_in_final": drop_ids_in_final,
        "missing_final_kept_ids": missing_final_ids,
        "unexpected_final_ids": unexpected_final_ids,
        "counts_match_summary": len(final) == int(item["detections_after_v7_dedup"]),
        "no_drop_ids_in_final": len(drop_ids_in_final) == 0,
        "all_final_bboxes_in_bounds": len(bbox_errors) == 0,
        "measurement_rows": int(len(measurements)),
        "measurement_rows_match_final": len(measurements) == len(final),
    }
    (out_dir / f"{source}_validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    return validation


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate v7 final end-to-end audit artifacts")
    parser.add_argument("--output-root", type=Path, default=Path("data/v7_final_end_to_end_test"))
    parser.add_argument("--manifest", type=Path, default=Path("data/final_holdout_v1/manifest.csv"))
    parser.add_argument("--um-per-pixel", type=float, default=8.57)
    args = parser.parse_args()

    output_root = resolve_path(args.output_root)
    artifacts_root = output_root / "final_artifacts"
    items = load_summary(output_root)
    sources = [source_name(item) for item in items]

    validations = [process_source(item, artifacts_root, args.um_per_pixel) for item in items]
    pd.DataFrame(validations).to_csv(artifacts_root / "validation_summary.csv", index=False)

    manifest = pd.read_csv(resolve_path(args.manifest))
    manifest["source"] = manifest["filename"].map(lambda name: Path(str(name)).stem)
    manifest_subset = manifest[manifest["source"].isin(sources)].copy()
    manifest_subset.to_csv(artifacts_root / "source_identification.csv", index=False)

    viewer_script = write_viewer_script(output_root, sources)
    final_summary = {
        "sources": sources,
        "artifact_root": rel_path(artifacts_root),
        "source_identification_csv": rel_path(artifacts_root / "source_identification.csv"),
        "validation_summary_csv": rel_path(artifacts_root / "validation_summary.csv"),
        "viewer_command": f"python {rel_path(viewer_script)}",
        "total_raw_detections_before_v7": int(sum(v["raw_detections_before_v7"] for v in validations)),
        "total_final_detections_after_v7": int(sum(v["final_detections_after_v7"] for v in validations)),
        "total_drop_count": int(sum(v["drop_count"] for v in validations)),
        "all_no_drop_ids_in_final": all(v["no_drop_ids_in_final"] for v in validations),
        "all_final_bboxes_in_bounds": all(v["all_final_bboxes_in_bounds"] for v in validations),
        "all_measurement_rows_match_final": all(v["measurement_rows_match_final"] for v in validations),
    }
    (artifacts_root / "final_artifact_summary.json").write_text(json.dumps(final_summary, indent=2), encoding="utf-8")
    print(json.dumps(final_summary, indent=2))


if __name__ == "__main__":
    main()
