#!/usr/bin/env python3
"""Run detector + trunk segmentation model on unseen full-source images."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.infer_tiled import infer_tiled  # noqa: E402


def safe_name(path: Path) -> str:
    return path.stem.replace(" ", "_").replace("(", "").replace(")", "")


def ensure_dirs(base: Path) -> dict[str, Path]:
    dirs = {
        "crops": base / "crops",
        "predicted_crops": base / "predicted_crops",
        "masks": base / "masks",
        "envelope_masks": base / "envelope_masks",
        "envelope_predicted_crops": base / "envelope_predicted_crops",
        "envelope_comparison": base / "envelope_comparison",
        "contact_sheets": base / "contact_sheets",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def fit_panel(img: np.ndarray, height: int = 180) -> np.ndarray:
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((height, height, 3), dtype=np.uint8)
    scale = height / float(h)
    width = max(1, int(round(w * scale)))
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)


def write_contact_sheet(items: list[tuple[int, np.ndarray, np.ndarray, np.ndarray]], output_path: Path, limit: int = 80) -> None:
    panels = []
    for detection_id, crop, overlay, mask_bgr in items[:limit]:
        trio = [fit_panel(crop), fit_panel(overlay), fit_panel(mask_bgr)]
        max_h = max(p.shape[0] for p in trio)
        padded = []
        for p in trio:
            if p.shape[0] < max_h:
                p = cv2.copyMakeBorder(p, 0, max_h - p.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(245, 245, 245))
            padded.append(p)
        row = np.concatenate(padded, axis=1)
        cv2.putText(row, f"id {detection_id}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(row, f"id {detection_id}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
        panels.append(row)
    if not panels:
        return
    max_w = max(p.shape[1] for p in panels)
    padded_rows = []
    for p in panels:
        if p.shape[1] < max_w:
            p = cv2.copyMakeBorder(p, 0, 0, 0, max_w - p.shape[1], cv2.BORDER_CONSTANT, value=(245, 245, 245))
        padded_rows.append(p)
    sheet = np.concatenate(padded_rows, axis=0)
    cv2.imwrite(str(output_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])


def prediction_mask_from_result(result: Any, crop_shape: tuple[int, int]) -> tuple[np.ndarray | None, float | None]:
    if result.masks is None or result.boxes is None or len(result.masks) == 0 or len(result.boxes) == 0:
        return None, None
    confs = result.boxes.conf.detach().cpu().numpy() if result.boxes.conf is not None else np.ones(len(result.masks))
    best_idx = int(np.argmax(confs))
    crop_h, crop_w = crop_shape
    mask = np.zeros((crop_h, crop_w), dtype=np.uint8)

    polygons = result.masks.xy
    if polygons and len(polygons) > best_idx and len(polygons[best_idx]) >= 3:
        pts = np.asarray(polygons[best_idx], dtype=np.float32)
        pts[:, 0] = np.clip(pts[:, 0], 0, crop_w - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, crop_h - 1)
        cv2.fillPoly(mask, [pts.astype(np.int32)], 255)
    else:
        mask_data = result.masks.data[best_idx].detach().cpu().numpy()
        mask = cv2.resize((mask_data > 0.5).astype(np.uint8) * 255, (crop_w, crop_h), interpolation=cv2.INTER_NEAREST)

    if not np.any(mask):
        return None, float(confs[best_idx])
    return mask, float(confs[best_idx])


def largest_component(mask: np.ndarray) -> np.ndarray:
    mask_bool = mask > 0
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_bool.astype(np.uint8), connectivity=8)
    if num_labels <= 1:
        return mask_bool.astype(np.uint8) * 255
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == largest).astype(np.uint8) * 255


def contour_from_mask(mask: np.ndarray) -> np.ndarray | None:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def odd_kernel_from_radius(radius: int) -> int:
    radius = max(0, int(radius))
    return max(1, radius * 2 + 1)


def dilation_radius(mask_shape: tuple[int, int], ratio: float, max_radius: int) -> int:
    h, w = mask_shape
    short_side = max(1, min(h, w))
    radius = int(round(short_side * ratio))
    return max(1, min(int(max_radius), radius))


def resample_closed_contour(contour: np.ndarray, n_points: int) -> np.ndarray | None:
    pts = contour.reshape(-1, 2).astype(np.float32)
    if len(pts) < 3:
        return None
    closed = np.vstack([pts, pts[0]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    total = float(seg.sum())
    if total <= 0:
        return None
    distances = np.concatenate([[0.0], np.cumsum(seg)])
    samples = np.linspace(0.0, total, max(8, int(n_points)), endpoint=False)
    xs = np.interp(samples, distances, closed[:, 0])
    ys = np.interp(samples, distances, closed[:, 1])
    return np.stack([xs, ys], axis=1)


def circular_moving_average(points: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return points
    if window % 2 == 0:
        window += 1
    window = min(window, len(points) if len(points) % 2 == 1 else len(points) - 1)
    if window < 3:
        return points
    pad = window // 2
    padded = np.vstack([points[-pad:], points, points[:pad]])
    kernel = np.ones(window, dtype=np.float32) / float(window)
    xs = np.convolve(padded[:, 0], kernel, mode="valid")
    ys = np.convolve(padded[:, 1], kernel, mode="valid")
    return np.stack([xs, ys], axis=1).astype(np.float32)


def rasterize_contour(points: np.ndarray, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    h, w = shape
    pts = np.round(points).astype(np.int32)
    pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
    contour = pts.reshape(-1, 1, 2)
    out = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(out, [contour], -1, 255, thickness=cv2.FILLED)
    return out, contour


def raw_coverage(raw_mask: np.ndarray, envelope_mask: np.ndarray) -> float:
    raw = raw_mask > 0
    raw_area = int(raw.sum())
    if raw_area == 0:
        return 0.0
    covered = int(np.logical_and(raw, envelope_mask > 0).sum())
    return covered / float(raw_area)


def touches_border(mask: np.ndarray) -> bool:
    return bool(np.any(mask[0, :] > 0) or np.any(mask[-1, :] > 0) or np.any(mask[:, 0] > 0) or np.any(mask[:, -1] > 0))


def make_envelope_mask(
    raw_mask: np.ndarray,
    dilate_ratio: float,
    dilate_max: int,
    smooth_window: int,
    resample_points: int,
    min_coverage: float,
) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any]]:
    raw_largest = largest_component(raw_mask)
    raw_area = int(np.count_nonzero(raw_largest))
    radius = dilation_radius(raw_largest.shape[:2], dilate_ratio, dilate_max)
    fallback_used = False

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (odd_kernel_from_radius(radius), odd_kernel_from_radius(radius)))
    dilated = cv2.dilate(raw_largest, kernel, iterations=1)
    dilated_contour = contour_from_mask(dilated)
    if dilated_contour is None:
        return raw_largest.copy(), contour_from_mask(raw_largest), {
            "raw_mask_area": raw_area,
            "envelope_mask_area": raw_area,
            "raw_coverage_by_envelope": 1.0 if raw_area else 0.0,
            "envelope_area_ratio": 1.0,
            "raw_contour_points": 0,
            "envelope_contour_points": 0,
            "dilation_radius_px": radius,
            "smoothing_window": smooth_window,
            "envelope_warning": True,
            "fallback_used": True,
        }

    raw_contour = contour_from_mask(raw_largest)
    raw_points = int(len(raw_contour)) if raw_contour is not None else 0
    sampled = resample_closed_contour(dilated_contour, resample_points)
    if sampled is None:
        fallback_used = True
        envelope_mask = dilated.copy()
        envelope_contour = dilated_contour
    else:
        smoothed = circular_moving_average(sampled, smooth_window)
        envelope_mask, envelope_contour = rasterize_contour(smoothed, raw_largest.shape[:2])
        coverage = raw_coverage(raw_largest, envelope_mask)
        if coverage < min_coverage:
            fallback_used = True
            envelope_mask = dilated.copy()
            envelope_contour = dilated_contour

    # Safety fallback: if even the selected envelope misses raw pixels, union it with the dilated mask.
    coverage = raw_coverage(raw_largest, envelope_mask)
    if coverage < min_coverage:
        fallback_used = True
        envelope_mask = cv2.bitwise_or(envelope_mask, dilated)
        envelope_contour = contour_from_mask(envelope_mask)
        coverage = raw_coverage(raw_largest, envelope_mask)

    envelope_area = int(np.count_nonzero(envelope_mask))
    area_ratio = float(envelope_area / raw_area) if raw_area else 0.0
    warning = bool(coverage < min_coverage or area_ratio > 1.35 or touches_border(envelope_mask) or fallback_used)
    metrics = {
        "raw_mask_area": raw_area,
        "envelope_mask_area": envelope_area,
        "raw_coverage_by_envelope": coverage,
        "envelope_area_ratio": area_ratio,
        "raw_contour_points": raw_points,
        "envelope_contour_points": int(len(envelope_contour)) if envelope_contour is not None else 0,
        "dilation_radius_px": radius,
        "smoothing_window": smooth_window,
        "envelope_warning": warning,
        "fallback_used": fallback_used,
    }
    return envelope_mask, envelope_contour, metrics


def write_comparison(crop: np.ndarray, raw_overlay: np.ndarray, envelope_overlay: np.ndarray, output_path: Path) -> None:
    panels = [fit_panel(crop), fit_panel(raw_overlay), fit_panel(envelope_overlay)]
    max_h = max(p.shape[0] for p in panels)
    padded = []
    for panel in panels:
        if panel.shape[0] < max_h:
            panel = cv2.copyMakeBorder(panel, 0, max_h - panel.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(245, 245, 245))
        padded.append(panel)
    cv2.imwrite(str(output_path), np.concatenate(padded, axis=1))


def process_image(
    image_path: Path,
    detector_model: Path,
    seg_model: YOLO,
    out_root: Path,
    conf: float,
    device: str,
    imgsz: int,
    use_envelope_contour: bool,
    envelope_dilate_ratio: float,
    envelope_dilate_max: int,
    envelope_smooth_window: int,
    envelope_resample_points: int,
    envelope_min_coverage: float,
) -> dict[str, Any]:
    image_name = safe_name(image_path)
    out_dir = out_root / image_name
    out_dir.mkdir(parents=True, exist_ok=True)
    dirs = ensure_dirs(out_dir)

    detections = infer_tiled(
        image_path=image_path,
        model_path=detector_model,
        tile_size=1280,
        overlap=256,
        conf_threshold=conf,
        iou_threshold=0.5,
        output_dir=out_dir / "detections",
        device=device,
    )

    detections_csv = out_dir / "detections" / f"{image_path.stem}_detections.csv"
    df = pd.read_csv(detections_csv)

    Image.MAX_IMAGE_PIXELS = None
    image_rgb = np.array(Image.open(image_path).convert("RGB"))
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    full_overlay = image_bgr.copy()
    h, w = image_bgr.shape[:2]

    rows = []
    contact_items = []
    mask_count = 0
    no_mask_count = 0
    envelope_count = 0
    envelope_warning_count = 0
    fallback_count = 0

    for detection_id, row in df.iterrows():
        x1 = max(0, int(np.floor(row["x1"])))
        y1 = max(0, int(np.floor(row["y1"])))
        x2 = min(w, int(np.ceil(row["x2"])))
        y2 = min(h, int(np.ceil(row["y2"])))
        crop_bgr = image_bgr[y1:y2, x1:x2]
        if crop_bgr.size == 0:
            no_mask_count += 1
            continue

        sample_id = f"{image_name}_det{detection_id:04d}"
        crop_path = dirs["crops"] / f"{sample_id}.png"
        pred_path = dirs["predicted_crops"] / f"{sample_id}_pred.png"
        mask_path = dirs["masks"] / f"{sample_id}_mask.png"
        envelope_pred_path = dirs["envelope_predicted_crops"] / f"{sample_id}_envelope_pred.png"
        envelope_mask_path = dirs["envelope_masks"] / f"{sample_id}_envelope_mask.png"
        comparison_path = dirs["envelope_comparison"] / f"{sample_id}_comparison.png"
        cv2.imwrite(str(crop_path), crop_bgr)

        result = seg_model.predict(crop_bgr, imgsz=imgsz, conf=0.15, device=device, verbose=False)[0]
        mask, seg_conf = prediction_mask_from_result(result, crop_bgr.shape[:2])
        overlay = crop_bgr.copy()
        envelope_overlay = crop_bgr.copy()
        envelope_metrics = {
            "raw_mask_area": 0,
            "envelope_mask_area": 0,
            "raw_coverage_by_envelope": "",
            "envelope_area_ratio": "",
            "raw_contour_points": 0,
            "envelope_contour_points": 0,
            "dilation_radius_px": "",
            "smoothing_window": envelope_smooth_window,
            "envelope_warning": False,
            "fallback_used": False,
        }

        has_mask = mask is not None
        if has_mask:
            mask_count += 1
            raw_contour = contour_from_mask(mask)
            contour_for_full_overlay = raw_contour
            if raw_contour is not None:
                cv2.drawContours(overlay, [raw_contour], -1, (0, 255, 255), 2, cv2.LINE_AA)
                envelope_metrics["raw_mask_area"] = int(np.count_nonzero(mask))
                envelope_metrics["envelope_mask_area"] = int(np.count_nonzero(mask))
                envelope_metrics["raw_coverage_by_envelope"] = 1.0
                envelope_metrics["envelope_area_ratio"] = 1.0
                envelope_metrics["raw_contour_points"] = int(len(raw_contour))
                envelope_metrics["envelope_contour_points"] = int(len(raw_contour))

                if use_envelope_contour:
                    envelope_mask, envelope_contour, envelope_metrics = make_envelope_mask(
                        mask,
                        dilate_ratio=envelope_dilate_ratio,
                        dilate_max=envelope_dilate_max,
                        smooth_window=envelope_smooth_window,
                        resample_points=envelope_resample_points,
                        min_coverage=envelope_min_coverage,
                    )
                    cv2.imwrite(str(envelope_mask_path), envelope_mask)
                    if envelope_contour is not None:
                        cv2.drawContours(envelope_overlay, [envelope_contour], -1, (0, 255, 255), 2, cv2.LINE_AA)
                        contour_for_full_overlay = envelope_contour
                    else:
                        envelope_overlay = overlay.copy()
                    envelope_count += 1
                    if envelope_metrics["envelope_warning"]:
                        envelope_warning_count += 1
                    if envelope_metrics["fallback_used"]:
                        fallback_count += 1
                    cv2.imwrite(str(envelope_pred_path), envelope_overlay)
                    write_comparison(crop_bgr, overlay, envelope_overlay, comparison_path)

                if contour_for_full_overlay is not None:
                    global_contour = contour_for_full_overlay.copy()
                    global_contour[:, 0, 0] += x1
                    global_contour[:, 0, 1] += y1
                    cv2.drawContours(full_overlay, [global_contour], -1, (0, 255, 255), 2, cv2.LINE_AA)
                    cx, cy = contour_for_full_overlay.reshape(-1, 2).mean(axis=0).astype(int)
                    cv2.putText(overlay, str(detection_id), (cx + 3, cy - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
                    if use_envelope_contour:
                        cv2.putText(envelope_overlay, str(detection_id), (cx + 3, cy - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
                        cv2.imwrite(str(envelope_pred_path), envelope_overlay)
                        write_comparison(crop_bgr, overlay, envelope_overlay, comparison_path)
                    gx, gy = global_contour.reshape(-1, 2).mean(axis=0).astype(int)
                    cv2.putText(full_overlay, str(detection_id), (gx + 4, gy - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.imwrite(str(mask_path), mask)
            mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        else:
            no_mask_count += 1
            mask_path = Path("")
            mask_bgr = np.zeros(crop_bgr.shape, dtype=np.uint8)
            cv2.putText(overlay, "NO MASK", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imwrite(str(pred_path), overlay)
        contact_items.append((int(detection_id), crop_bgr, overlay, mask_bgr))
        rows.append({
            "detection_id": int(detection_id),
            "det_confidence": float(row["confidence"]),
            "seg_confidence": seg_conf if seg_conf is not None else "",
            "has_mask": bool(has_mask),
            "bbox_x1": float(row["x1"]),
            "bbox_y1": float(row["y1"]),
            "bbox_x2": float(row["x2"]),
            "bbox_y2": float(row["y2"]),
            "crop_path": str(crop_path),
            "prediction_crop_path": str(pred_path),
            "mask_path": str(mask_path) if has_mask else "",
            "envelope_prediction_crop_path": str(envelope_pred_path) if has_mask and use_envelope_contour else "",
            "envelope_mask_path": str(envelope_mask_path) if has_mask and use_envelope_contour else "",
            "envelope_comparison_path": str(comparison_path) if has_mask and use_envelope_contour else "",
            **envelope_metrics,
        })

    overlay_path = out_dir / f"{image_name}_trunk_seg_overlay.jpg"
    cv2.imwrite(str(overlay_path), full_overlay, [cv2.IMWRITE_JPEG_QUALITY, 92])
    results_csv = out_dir / f"{image_name}_segmentation_results.csv"
    pd.DataFrame(rows).to_csv(results_csv, index=False)
    contact_path = dirs["contact_sheets"] / f"{image_name}_contact_sheet.jpg"
    write_contact_sheet(contact_items, contact_path)

    summary = {
        "image": str(image_path),
        "output_dir": str(out_dir),
        "detections": int(len(df)),
        "masks": int(mask_count),
        "no_mask": int(no_mask_count),
        "envelope_masks": int(envelope_count),
        "envelope_warnings": int(envelope_warning_count),
        "fallbacks": int(fallback_count),
        "overlay_path": str(overlay_path),
        "results_csv": str(results_csv),
        "contact_sheet": str(contact_path),
        "envelope_comparison_dir": str(dirs["envelope_comparison"]),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test trunk segmentation generalization on new full images")
    parser.add_argument("--images-dir", type=Path, default=Path("data/slike_test_nove"))
    parser.add_argument("--detector-model", type=Path, default=Path("models/yolo11n_tiled_best.pt"))
    parser.add_argument("--seg-model", type=Path, default=Path("runs/segment/runs/segment/trunk_seg_v1_test20/weights/best.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/web_outputs/trunk_seg_generalization_test"))
    parser.add_argument("--conf", type=float, default=0.6)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--envelope-dilate-ratio", type=float, default=0.03)
    parser.add_argument("--envelope-dilate-max", type=int, default=7)
    parser.add_argument("--envelope-smooth-window", type=int, default=9)
    parser.add_argument("--envelope-resample-points", type=int, default=80)
    parser.add_argument("--envelope-min-coverage", type=float, default=0.99)
    parser.add_argument("--use-envelope-contour", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    images = sorted(p for p in args.images_dir.iterdir() if p.is_file() and p.suffix.lower() in image_exts)
    if not images:
        raise SystemExit(f"No image files found in {args.images_dir}")
    missing = [p for p in [args.detector_model, args.seg_model] if not p.exists()]
    if missing:
        raise SystemExit(f"Missing model files: {missing}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    seg_model = YOLO(args.seg_model)
    summaries = []
    for image_path in images:
        summaries.append(process_image(
            image_path,
            args.detector_model,
            seg_model,
            args.output_dir,
            args.conf,
            args.device,
            args.imgsz,
            args.use_envelope_contour,
            args.envelope_dilate_ratio,
            args.envelope_dilate_max,
            args.envelope_smooth_window,
            args.envelope_resample_points,
            args.envelope_min_coverage,
        ))

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps({"images": summaries}, indent=2), encoding="utf-8")
    print(json.dumps({"summary_path": str(summary_path), "images": summaries}, indent=2))


if __name__ == "__main__":
    main()
