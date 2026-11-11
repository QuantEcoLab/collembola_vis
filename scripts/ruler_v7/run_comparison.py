import argparse
import csv
import json
import math
import shutil
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from ultralytics import YOLO

Image.MAX_IMAGE_PIXELS = None
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/trunk_dataset_combined_v7/ruler_unseen_visual_comparison/v5_vs_v7"
DETECTOR = ROOT / "models/yolo11n_tiled_best.pt"
V5 = ROOT / "runs/segment/runs/segment/trunk_seg_v5_test20/weights/last.pt"
V7 = ROOT / "runs/segment/runs/segment/trunk_seg_v7_ruler_finetune/weights/best.pt"
IMAGES = [
    ROOT / "data/slike_test_v5_nove/C_3_2_Collembola011.jpg",
    ROOT / "data/slike_test_v5_nove/C_4_1_PS_500-1000_014.jpg",
    ROOT / "data/slike_test_v5_nove/C_5_1_PS_500-1000_017.jpg",
    ROOT / "data/final_holdout_v1/C_1_2_Collembola005.jpg",
    ROOT / "data/final_holdout_v1/C_1_3_Collembola006.jpg",
]
PARAMS = {
    "detector_tile_size": 1280,
    "detector_overlap": 256,
    "detector_stride": 1024,
    "detector_confidence": 0.6,
    "detector_global_bbox_nms_iou": 0.5,
    "segmentation_confidence": 0.15,
    "segmentation_imgsz": 320,
    "device": "cpu",
    "envelope_mode": False,
    "mask_iou_dedup": False,
    "v7_duplicate_classifier": False,
    "restricted_v5_classifier_cleanup": False,
}
CONTOUR_COLOR = (0, 255, 255)
CONTOUR_WIDTH = 2


def safe_name(path: Path) -> str:
    return path.stem.replace(" ", "_").replace("(", "").replace(")", "")


def rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def tile_image(image_path: Path, tile_size: int = 1280, overlap: int = 256):
    img = Image.open(image_path)
    img_w, img_h = img.size
    tiles = []
    stride = tile_size - overlap
    tile_id = 0
    y = 0
    y_end = 0
    while y < img_h:
        x = 0
        x_end = 0
        while x < img_w:
            x_end = min(x + tile_size, img_w)
            y_end = min(y + tile_size, img_h)
            x_start = max(0, x_end - tile_size) if x_end == img_w else x
            y_start = max(0, y_end - tile_size) if y_end == img_h else y
            tiles.append((img.crop((x_start, y_start, x_end, y_end)), x_start, y_start, tile_id))
            tile_id += 1
            if x_end == img_w:
                break
            x += stride
        if y_end == img_h:
            break
        y += stride
    return img, tiles


def nms_global(detections, iou_threshold=0.5):
    if not detections:
        return []
    boxes = np.array([[d["x1"], d["y1"], d["x2"], d["y2"]] for d in detections])
    scores = np.array([d["conf"] for d in detections])
    order = scores.argsort()[::-1]
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        inter = w * h
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_order = (boxes[order[1:], 2] - boxes[order[1:], 0]) * (boxes[order[1:], 3] - boxes[order[1:], 1])
        union = area_i + area_order - inter
        iou = inter / union
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    return [detections[i] for i in keep]


def normalize_binary_mask(mask: Any) -> np.ndarray:
    arr = np.asarray(mask)
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[:, :, 0]
    if arr.ndim != 2:
        raise ValueError(f"Expected mask HxW, got {arr.shape}")
    return arr > 0


def prediction_mask_from_result(result: Any, crop_shape: tuple[int, int]):
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


def contour_from_mask(mask: np.ndarray):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def overlay_crop(crop_bgr: np.ndarray, mask: np.ndarray | None, detection_id: int, label: str):
    overlay = crop_bgr.copy()
    if mask is not None:
        contour = contour_from_mask(normalize_binary_mask(mask).astype(np.uint8) * 255)
        if contour is not None:
            cv2.drawContours(overlay, [contour], -1, CONTOUR_COLOR, CONTOUR_WIDTH, cv2.LINE_AA)
            cx, cy = contour.reshape(-1, 2).mean(axis=0).astype(int)
            cv2.putText(overlay, str(detection_id), (cx + 3, cy - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.55, CONTOUR_COLOR, 2, cv2.LINE_AA)
    else:
        cv2.putText(overlay, "NO MASK", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(overlay, label, (8, max(18, overlay.shape[0] - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(overlay, label, (8, max(18, overlay.shape[0] - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    return overlay


def add_banner(img: np.ndarray, text: str, h: int = 34):
    out = cv2.copyMakeBorder(img, h, 0, 0, 0, cv2.BORDER_CONSTANT, value=(245, 245, 245))
    cv2.putText(out, text, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2, cv2.LINE_AA)
    return out


def make_triptych(orig: np.ndarray, v5: np.ndarray, v7: np.ndarray, det_id: int, v5_conf, v7_conf):
    panels = [
        add_banner(orig, f"original | id {det_id}"),
        add_banner(v5, f"v5_last | conf {v5_conf if v5_conf is not None else 'NA'}"),
        add_banner(v7, f"v7_best | conf {v7_conf if v7_conf is not None else 'NA'}"),
    ]
    max_h = max(p.shape[0] for p in panels)
    padded = []
    for p in panels:
        if p.shape[0] < max_h:
            p = cv2.copyMakeBorder(p, 0, max_h - p.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(245, 245, 245))
        padded.append(p)
    return np.concatenate(padded, axis=1)


def mask_iou(a: np.ndarray | None, b: np.ndarray | None):
    if a is None or b is None:
        return ""
    aa = normalize_binary_mask(a)
    bb = normalize_binary_mask(b)
    union = int(np.logical_or(aa, bb).sum())
    return 0.0 if union == 0 else float(np.logical_and(aa, bb).sum() / union)


def fit_panel(img: np.ndarray, width: int = 900):
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((width, width, 3), dtype=np.uint8)
    new_h = max(1, int(round(h * (width / float(w)))))
    return cv2.resize(img, (width, new_h), interpolation=cv2.INTER_AREA)


def write_contact_sheet(image_paths: list[Path], output_path: Path, cols: int = 3, thumb_w: int = 900):
    thumbs = []
    for p in image_paths:
        img = cv2.imread(str(p))
        if img is not None:
            thumbs.append(fit_panel(img, thumb_w))
    if not thumbs:
        return
    tile_h = max(t.shape[0] for t in thumbs)
    rows = math.ceil(len(thumbs) / cols)
    sheet = np.full((rows * tile_h, cols * thumb_w, 3), 245, dtype=np.uint8)
    for i, t in enumerate(thumbs):
        y = (i // cols) * tile_h
        x = (i % cols) * thumb_w
        sheet[y:y + t.shape[0], x:x + t.shape[1]] = t
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])


def draw_full_overlay(image_bgr: np.ndarray, rows: list[dict[str, Any]], model_key: str):
    overlay = image_bgr.copy()
    for row in rows:
        mask = row.get(f"{model_key}_mask")
        if mask is None:
            continue
        contour = contour_from_mask(normalize_binary_mask(mask).astype(np.uint8) * 255)
        if contour is None:
            continue
        global_contour = contour.copy()
        global_contour[:, 0, 0] += int(row["crop_x1"])
        global_contour[:, 0, 1] += int(row["crop_y1"])
        cv2.drawContours(overlay, [global_contour], -1, CONTOUR_COLOR, CONTOUR_WIDTH, cv2.LINE_AA)
        gx, gy = global_contour.reshape(-1, 2).mean(axis=0).astype(int)
        cv2.putText(overlay, str(row["detection_id"]), (gx + 4, gy - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, CONTOUR_COLOR, 2, cv2.LINE_AA)
    return overlay


def run_detector(image_path: Path, model: YOLO, out_dir: Path):
    det_dir = out_dir / "detections"
    det_dir.mkdir(parents=True, exist_ok=True)
    img, tiles = tile_image(image_path, PARAMS["detector_tile_size"], PARAMS["detector_overlap"])
    img_w, img_h = img.size
    all_det = []
    for tile_idx, (tile_img, x_offset, y_offset, tile_id) in enumerate(tiles, start=1):
        results = model(tile_img, conf=PARAMS["detector_confidence"], device=PARAMS["device"], verbose=False)
        for result in results:
            boxes = result.boxes
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                all_det.append({
                    "x1": float(x1 + x_offset), "y1": float(y1 + y_offset),
                    "x2": float(x2 + x_offset), "y2": float(y2 + y_offset),
                    "conf": float(boxes.conf[i].cpu().numpy()),
                    "class": int(boxes.cls[i].cpu().numpy()),
                    "tile_id": tile_id,
                })
        if tile_idx % 25 == 0 or tile_idx == len(tiles):
            print(f"  detector {image_path.stem}: tile {tile_idx}/{len(tiles)}", flush=True)
    final = nms_global(all_det, PARAMS["detector_global_bbox_nms_iou"])
    csv_path = det_dir / f"{safe_name(image_path)}_detections.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["detection_id", "x1", "y1", "x2", "y2", "width", "height", "confidence", "class", "tile_id"])
        w.writeheader()
        for idx, d in enumerate(final):
            w.writerow({
                "detection_id": idx, "x1": d["x1"], "y1": d["y1"], "x2": d["x2"], "y2": d["y2"],
                "width": d["x2"] - d["x1"], "height": d["y2"] - d["y1"],
                "confidence": d["conf"], "class": d["class"], "tile_id": d["tile_id"],
            })
    meta = {
        "image_path": rel(image_path), "image_size": [img_w, img_h], "model_path": str(DETECTOR),
        "detector_source": "git blob feature/v5-trunk-contour-model:models/yolo11n_tiled_best.pt extracted to temp path",
        "tile_size": PARAMS["detector_tile_size"], "overlap": PARAMS["detector_overlap"],
        "stride": PARAMS["detector_stride"], "conf_threshold": PARAMS["detector_confidence"],
        "iou_threshold": PARAMS["detector_global_bbox_nms_iou"], "num_tiles": len(tiles),
        "detections_before_nms": len(all_det), "detections_after_nms": len(final),
    }
    (det_dir / f"{safe_name(image_path)}_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return final, meta


def process_image(image_path: Path, detector_model: YOLO, v5_model: YOLO, v7_model: YOLO):
    source = safe_name(image_path)
    out_dir = OUT / source
    if out_dir.exists():
        shutil.rmtree(out_dir)
    for sub in ["crops/original", "crops/v5_overlay", "crops/v7_overlay", "crops/triptych", "masks/v5", "masks/v7", "full_overlays", "contact_sheets"]:
        (out_dir / sub).mkdir(parents=True, exist_ok=True)
    print(f"Processing {source}", flush=True)
    detections, meta = run_detector(image_path, detector_model, out_dir)
    image_rgb = np.array(Image.open(image_path).convert("RGB"))
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    h, w = image_bgr.shape[:2]
    rows = []
    csv_rows = []
    triptychs = []
    for det_id, det in enumerate(detections):
        x1 = max(0, int(np.floor(det["x1"])))
        y1 = max(0, int(np.floor(det["y1"])))
        x2 = min(w, int(np.ceil(det["x2"])))
        y2 = min(h, int(np.ceil(det["y2"])))
        crop_bgr = image_bgr[y1:y2, x1:x2]
        if crop_bgr.size == 0:
            continue
        sample = f"{source}_det{det_id:04d}"
        orig_path = out_dir / "crops/original" / f"{sample}.png"
        v5_path = out_dir / "crops/v5_overlay" / f"{sample}_v5_overlay.png"
        v7_path = out_dir / "crops/v7_overlay" / f"{sample}_v7_overlay.png"
        trip_path = out_dir / "crops/triptych" / f"{sample}_triptych.jpg"
        v5_mask_path = out_dir / "masks/v5" / f"{sample}_v5_mask.png"
        v7_mask_path = out_dir / "masks/v7" / f"{sample}_v7_mask.png"
        cv2.imwrite(str(orig_path), crop_bgr)
        r5 = v5_model.predict(crop_bgr, imgsz=PARAMS["segmentation_imgsz"], conf=PARAMS["segmentation_confidence"], device=PARAMS["device"], verbose=False)[0]
        r7 = v7_model.predict(crop_bgr, imgsz=PARAMS["segmentation_imgsz"], conf=PARAMS["segmentation_confidence"], device=PARAMS["device"], verbose=False)[0]
        m5, c5 = prediction_mask_from_result(r5, crop_bgr.shape[:2])
        m7, c7 = prediction_mask_from_result(r7, crop_bgr.shape[:2])
        if m5 is not None:
            cv2.imwrite(str(v5_mask_path), m5)
        if m7 is not None:
            cv2.imwrite(str(v7_mask_path), m7)
        ov5 = overlay_crop(crop_bgr, m5, det_id, "v5_last")
        ov7 = overlay_crop(crop_bgr, m7, det_id, "v7_best")
        cv2.imwrite(str(v5_path), ov5)
        cv2.imwrite(str(v7_path), ov7)
        trip = make_triptych(crop_bgr, ov5, ov7, det_id, round(c5, 4) if c5 is not None else None, round(c7, 4) if c7 is not None else None)
        cv2.imwrite(str(trip_path), trip, [cv2.IMWRITE_JPEG_QUALITY, 92])
        triptychs.append(trip_path)
        area5 = int(np.count_nonzero(m5)) if m5 is not None else 0
        area7 = int(np.count_nonzero(m7)) if m7 is not None else 0
        rows.append({"detection_id": det_id, "crop_x1": x1, "crop_y1": y1, "v5_mask": m5, "v7_mask": m7})
        csv_rows.append({
            "source_image": rel(image_path), "source_id": source, "detection_id": det_id,
            "bbox_x1": det["x1"], "bbox_y1": det["y1"], "bbox_x2": det["x2"], "bbox_y2": det["y2"],
            "detector_confidence": det["conf"],
            "v5_segmentation_confidence": c5 if c5 is not None else "",
            "v7_segmentation_confidence": c7 if c7 is not None else "",
            "v5_mask_area_px": area5, "v7_mask_area_px": area7,
            "v5_v7_mask_pixel_iou": mask_iou(m5, m7),
            "original_crop_path": rel(orig_path), "v5_overlay_path": rel(v5_path), "v7_overlay_path": rel(v7_path),
            "triptych_path": rel(trip_path),
            "v5_mask_path": rel(v5_mask_path) if m5 is not None else "", "v7_mask_path": rel(v7_mask_path) if m7 is not None else "",
        })
        if (det_id + 1) % 100 == 0 or (det_id + 1) == len(detections):
            print(f"  segment {source}: crop {det_id + 1}/{len(detections)}", flush=True)
    v5_full_path = out_dir / "full_overlays" / f"{source}_v5_last_full_overlay.jpg"
    v7_full_path = out_dir / "full_overlays" / f"{source}_v7_best_full_overlay.jpg"
    cv2.imwrite(str(v5_full_path), draw_full_overlay(image_bgr, rows, "v5"), [cv2.IMWRITE_JPEG_QUALITY, 92])
    cv2.imwrite(str(v7_full_path), draw_full_overlay(image_bgr, rows, "v7"), [cv2.IMWRITE_JPEG_QUALITY, 92])
    contact_path = out_dir / "contact_sheets" / f"{source}_all_triptychs_contact_sheet.jpg"
    write_contact_sheet(triptychs, contact_path, cols=3, thumb_w=900)
    per_csv = out_dir / f"{source}_comparison.csv"
    pd.DataFrame(csv_rows).to_csv(per_csv, index=False)
    summary = {
        "source_image": rel(image_path), "source_id": source, "detections_after_global_bbox_nms": len(csv_rows),
        "detector_metadata": meta, "v5_full_overlay": rel(v5_full_path), "v7_full_overlay": rel(v7_full_path),
        "contact_sheet": rel(contact_path), "comparison_csv": rel(per_csv),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary, csv_rows, triptychs


def main():
    global DETECTOR
    parser = argparse.ArgumentParser(description="Compare raw v5/v7 masks on the same detector boxes")
    parser.add_argument("--detector-model", type=Path, default=DETECTOR)
    args = parser.parse_args()
    DETECTOR = args.detector_model.resolve()
    for child in OUT.iterdir() if OUT.exists() else []:
        if child.name != "run_comparison.py":
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    OUT.mkdir(parents=True, exist_ok=True)
    missing = [str(p) for p in [DETECTOR, V5, V7, *IMAGES] if not p.exists()]
    if missing:
        raise SystemExit(f"Missing required inputs: {missing}")
    print("Loading models", flush=True)
    detector_model = YOLO(DETECTOR)
    v5_model = YOLO(V5)
    v7_model = YOLO(V7)
    summaries = []
    all_rows = []
    overall_trips = []
    for image_path in IMAGES:
        summary, rows, trips = process_image(image_path, detector_model, v5_model, v7_model)
        summaries.append(summary)
        all_rows.extend(rows)
        if trips:
            step = max(1, len(trips) // 30)
            overall_trips.extend(trips[::step][:30])
    all_csv = OUT / "v5_vs_v7_common_bbox_comparison.csv"
    pd.DataFrame(all_rows).to_csv(all_csv, index=False)
    overall_contact = OUT / "v5_vs_v7_overall_sample_contact_sheet.jpg"
    write_contact_sheet(overall_trips, overall_contact, cols=3, thumb_w=900)
    summary = {
        "status": "completed",
        "leakage_free": True,
        "manual_ground_truth_metrics": False,
        "models": {
            "detector_requested": "models/yolo11n_tiled_best.pt",
            "detector_used": str(DETECTOR),
            "detector_used_note": "resolved --detector-model; default models/yolo11n_tiled_best.pt",
            "v5_segmentation": rel(V5),
            "v7_segmentation": rel(V7),
        },
        "parameters": PARAMS,
        "contour_style": {"color_bgr": list(CONTOUR_COLOR), "width_px": CONTOUR_WIDTH, "line_type": "cv2.LINE_AA"},
        "images": summaries,
        "combined_csv": rel(all_csv),
        "overall_sample_contact_sheet": rel(overall_contact),
        "completed_at_unix": time.time(),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUT / "README.md").write_text(f"""# Leakage-Free v5 vs v7 Ruler Visual Comparison

This is a visual comparison only. No manual ground truth labels were used and no metrics against GT were computed. No winner is declared here.

## Leakage Status

The five selected source IDs were previously audited as absent from `data/trunk_dataset_combined_v7/yolo_seg` and are not any of the v7 ruler training sources (`C_2_1_Collembola007`, `C_4_3_PS_500034`, `K_3_Collembola003`).

## Models

- Detector requested: `models/yolo11n_tiled_best.pt`
- Detector used: `{DETECTOR}`
- Detector note: resolved `--detector-model`; default is `models/yolo11n_tiled_best.pt`.
- v5 segmentation: `{rel(V5)}`
- v7 segmentation: `{rel(V7)}`

## Fixed Parameters

- Detector-first pipeline from Git history: `scripts/test_trunk_seg_generalization.py` + `scripts/infer_tiled.py`
- Detector tile size: 1280
- Detector overlap: 256
- Detector stride: 1024
- Detector confidence: 0.6
- Global bbox NMS IoU: 0.5
- Segmentation confidence: 0.15
- Segmentation imgsz: 320
- Envelope mode: disabled
- Mask-IoU dedup: disabled
- v7 duplicate classifier: disabled
- restricted v5 classifier cleanup: disabled
- Contour display: BGR `(0,255,255)`, width 2 px, `cv2.LINE_AA`

## Outputs

- Combined CSV: `v5_vs_v7_common_bbox_comparison.csv`
- Overall sample contact sheet: `v5_vs_v7_overall_sample_contact_sheet.jpg`
- Per-source folders contain detector CSV/metadata, original crops, v5 overlays, v7 overlays, triptychs, masks, full-resolution overlays and all-crop contact sheets.
""", encoding="utf-8")
    print(json.dumps({
        "summary": rel(OUT / "summary.json"),
        "combined_csv": rel(all_csv),
        "overall_contact": rel(overall_contact),
        "counts": {s["source_id"]: s["detections_after_global_bbox_nms"] for s in summaries},
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
