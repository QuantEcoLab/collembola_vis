#!/usr/bin/env python3
"""Leakage-free end-to-end v5/v7 trunk pipeline visual comparison.

This script intentionally writes only under this output directory. It uses the
detector-first pipeline parameters recovered from Git history, runs the detector
once per source image, then runs v5 and v7 segmentation plus the same final
v7 fragment-aware classifier dedup separately for each model.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import joblib
import numpy as np
import pandas as pd
from PIL import Image
from ultralytics import YOLO


Image.MAX_IMAGE_PIXELS = None

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/trunk_dataset_combined_v7/ruler_unseen_visual_comparison/final_pipeline_v5_vs_v7"
DETECTOR = ROOT / "models/yolo11n_tiled_best.pt"
DEDUP_MODEL = ROOT / "models/duplicate_pair_classifier_v7_fragment_aware.joblib"
V5_SEG = ROOT / "runs/segment/runs/segment/trunk_seg_v5_test20/weights/last.pt"
V7_SEG = ROOT / "runs/segment/runs/segment/trunk_seg_v7_ruler_finetune/weights/best.pt"

IMAGES = [
    ROOT / "data/slike_test_v5_nove/C_3_2_Collembola011.jpg",
    ROOT / "data/slike_test_v5_nove/C_4_1_PS_500-1000_014.jpg",
    ROOT / "data/slike_test_v5_nove/C_5_1_PS_500-1000_017.jpg",
    ROOT / "data/final_holdout_v1/C_1_2_Collembola005.jpg",
    ROOT / "data/final_holdout_v1/C_1_3_Collembola006.jpg",
]

PARAMS = {
    "tile_size": 1280,
    "overlap": 256,
    "stride": 1024,
    "detector_conf": 0.6,
    "bbox_nms_iou": 0.5,
    "seg_conf": 0.15,
    "seg_imgsz": 320,
    "dedup_threshold": 0.50,
    "device": "cpu",
    "envelope_mode": False,
    "mask_iou_dedup": False,
    "restricted_v5_cleanup": False,
}

CONTOUR_COLOR = (0, 255, 255)
CONTOUR_WIDTH = 2

FEATURES = [
    "mask_iou", "bbox_iou", "mask_area_ratio", "centroid_distance",
    "detector_conf_min", "detector_conf_max", "detector_conf_abs_diff", "detector_conf_ratio",
    "segmentation_conf_min", "segmentation_conf_max", "segmentation_conf_abs_diff", "segmentation_conf_ratio",
    "mask_area_min", "mask_area_max", "mask_area_abs_diff",
    "normalized_centroid_distance", "bbox_area_ratio", "small_mask_in_large_bbox", "small_bbox_in_large_bbox",
    "expanded_overlap", "fragment_score", "bbox_gap_px", "mask_gap_px",
    "small_bbox_inside_large_bbox_fraction", "small_mask_inside_large_bbox_fraction", "small_mask_inside_dilated_large_mask_fraction",
    "width_ratio", "height_ratio", "aspect_ratio_difference", "major_axis_angle_difference",
    "centroid_distance_over_large_bbox_diagonal", "contour_perimeter_ratio", "solidity_difference",
    "eccentricity_difference", "orientation_difference", "convex_hull_area_ratio",
    "masks_connect_with_small_dilation", "min_dilation_radius_to_connect_masks",
]


def rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def safe_name(path: Path) -> str:
    return path.stem.replace(" ", "_").replace("(", "").replace(")", "")


def required_paths() -> list[Path]:
    return [DETECTOR, DEDUP_MODEL, V5_SEG, V7_SEG, *IMAGES]


def tile_image(image_path: Path):
    img = Image.open(image_path)
    img_w, img_h = img.size
    tiles = []
    tile_size = PARAMS["tile_size"]
    stride = PARAMS["stride"]
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


def nms_global(detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
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
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_o = (boxes[order[1:], 2] - boxes[order[1:], 0]) * (boxes[order[1:], 3] - boxes[order[1:], 1])
        iou = inter / (area_i + area_o - inter)
        order = order[np.where(iou <= PARAMS["bbox_nms_iou"])[0] + 1]
    return [detections[i] for i in keep]


def run_detector_once(image_path: Path, detector: YOLO, out_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    img, tiles = tile_image(image_path)
    all_detections = []
    for tile_idx, (tile, x_off, y_off, tile_id) in enumerate(tiles, start=1):
        results = detector(tile, conf=PARAMS["detector_conf"], device=PARAMS["device"], verbose=False)
        for result in results:
            boxes = result.boxes
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                all_detections.append({
                    "x1": float(x1 + x_off), "y1": float(y1 + y_off),
                    "x2": float(x2 + x_off), "y2": float(y2 + y_off),
                    "conf": float(boxes.conf[i].cpu().numpy()),
                    "class": int(boxes.cls[i].cpu().numpy()),
                    "tile_id": tile_id,
                })
        if tile_idx % 25 == 0 or tile_idx == len(tiles):
            print(f"  detector {image_path.stem}: tile {tile_idx}/{len(tiles)}", flush=True)
    final = nms_global(all_detections)
    for det_id, det in enumerate(final):
        det["detection_id"] = det_id
    det_dir = out_dir / "detector"
    det_dir.mkdir(parents=True, exist_ok=True)
    csv_path = det_dir / f"{safe_name(image_path)}_detections_after_bbox_nms.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["detection_id", "x1", "y1", "x2", "y2", "width", "height", "confidence", "class", "tile_id"])
        writer.writeheader()
        for det in final:
            writer.writerow({
                "detection_id": det["detection_id"], "x1": det["x1"], "y1": det["y1"], "x2": det["x2"], "y2": det["y2"],
                "width": det["x2"] - det["x1"], "height": det["y2"] - det["y1"],
                "confidence": det["conf"], "class": det["class"], "tile_id": det["tile_id"],
            })
    meta = {
        "image_path": rel(image_path), "image_size": list(img.size), "num_tiles": len(tiles),
        "detections_before_bbox_nms": len(all_detections), "detections_after_bbox_nms": len(final),
        "detections_csv": rel(csv_path), "detector_model": str(DETECTOR), **PARAMS,
    }
    (det_dir / f"{safe_name(image_path)}_detector_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return final, meta


def normalize_mask(mask: Any) -> np.ndarray:
    arr = np.asarray(mask)
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[:, :, 0]
    if arr.ndim != 2:
        raise ValueError(f"Expected HxW mask, got {arr.shape}")
    return arr > 0


def result_mask(result: Any, crop_shape: tuple[int, int]) -> tuple[np.ndarray | None, float | None]:
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
        data = result.masks.data[best_idx].detach().cpu().numpy()
        mask = cv2.resize((data > 0.5).astype(np.uint8) * 255, (crop_w, crop_h), interpolation=cv2.INTER_NEAREST)
    return (mask if np.any(mask) else None), float(confs[best_idx])


def contour_from_mask(mask: np.ndarray | None) -> np.ndarray | None:
    if mask is None:
        return None
    contours, _ = cv2.findContours((normalize_mask(mask).astype(np.uint8) * 255), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return max(contours, key=cv2.contourArea) if contours else None


def bbox_area(bbox: tuple[float, float, float, float]) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def ratio_minmax(a: float, b: float) -> float:
    return float(min(a, b) / max(a, b)) if max(a, b) > 0 else 0.0


def bbox_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    inter = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    union = bbox_area(a) + bbox_area(b) - inter
    return float(inter / union) if union > 0 else 0.0


def bbox_gap(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    return float(math.hypot(max(a[0] - b[2], b[0] - a[2], 0.0), max(a[1] - b[3], b[1] - a[3], 0.0)))


def angle_diff(a: float, b: float) -> float:
    if np.isnan(a) or np.isnan(b):
        return np.nan
    return float(abs((a - b + 90.0) % 180.0 - 90.0))


def contour_props(mask: np.ndarray | None) -> dict[str, float]:
    if mask is None:
        return {"perimeter": np.nan, "solidity": np.nan, "eccentricity": np.nan, "orientation": np.nan, "hull_area": np.nan}
    contour = contour_from_mask(mask)
    if contour is None:
        return {"perimeter": np.nan, "solidity": np.nan, "eccentricity": np.nan, "orientation": np.nan, "hull_area": np.nan}
    area = float(cv2.contourArea(contour))
    perimeter = float(cv2.arcLength(contour, True))
    hull = cv2.convexHull(contour)
    hull_area = float(cv2.contourArea(hull))
    solidity = area / hull_area if hull_area > 0 else np.nan
    pts = contour.reshape(-1, 2).astype(np.float64)
    orientation = np.nan
    eccentricity = np.nan
    if len(pts) >= 3:
        centered = pts - pts.mean(axis=0)
        cov = np.cov(centered, rowvar=False)
        vals, vecs = np.linalg.eigh(cov)
        vals = np.sort(np.maximum(vals, 0.0))[::-1]
        if vals[0] > 0:
            eccentricity = math.sqrt(max(0.0, 1.0 - vals[1] / vals[0]))
        major = vecs[:, np.argmax(np.linalg.eigvalsh(cov))]
        orientation = math.degrees(math.atan2(float(major[1]), float(major[0])))
    return {"perimeter": perimeter, "solidity": solidity, "eccentricity": eccentricity, "orientation": orientation, "hull_area": hull_area}


def mask_centroid(det: dict[str, Any]) -> tuple[float, float]:
    mask = det.get("mask")
    if mask is not None:
        ys, xs = np.nonzero(normalize_mask(mask))
        if len(xs):
            return float(xs.mean() + det["mask_x"]), float(ys.mean() + det["mask_y"])
    b = det["bbox"]
    return (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0


def paste_mask(canvas: np.ndarray, det: dict[str, Any], origin_x: int, origin_y: int) -> None:
    mask = det.get("mask")
    if mask is None:
        return
    m = normalize_mask(mask)
    x = int(det["mask_x"] - origin_x)
    y = int(det["mask_y"] - origin_y)
    h, w = m.shape[:2]
    canvas[y:y + h, x:x + w] |= m


def pair_features(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    bbox_a, bbox_b = a["bbox"], b["bbox"]
    mask_a, mask_b = a.get("mask"), b.get("mask")
    area_a = int(normalize_mask(mask_a).sum()) if mask_a is not None else int(bbox_area(bbox_a))
    area_b = int(normalize_mask(mask_b).sum()) if mask_b is not None else int(bbox_area(bbox_b))
    small, large = (a, b) if area_a <= area_b else (b, a)
    sb, lb = small["bbox"], large["bbox"]
    ca, cb = mask_centroid(a), mask_centroid(b)
    bbox_area_a, bbox_area_b = bbox_area(bbox_a), bbox_area(bbox_b)
    small_bbox_area = max(1.0, bbox_area(sb))
    large_diag = max(1.0, math.hypot(lb[2] - lb[0], lb[3] - lb[1]))
    props_a, props_b = contour_props(mask_a), contour_props(mask_b)
    vals: dict[str, Any] = {
        "bbox_iou": bbox_iou(bbox_a, bbox_b),
        "bbox_area_ratio": ratio_minmax(bbox_area_a, bbox_area_b),
        "small_bbox_in_large_bbox": max(0.0, min(sb[2], lb[2]) - max(sb[0], lb[0])) * max(0.0, min(sb[3], lb[3]) - max(sb[1], lb[1])) / small_bbox_area,
        "small_bbox_inside_large_bbox_fraction": max(0.0, min(sb[2], lb[2]) - max(sb[0], lb[0])) * max(0.0, min(sb[3], lb[3]) - max(sb[1], lb[1])) / small_bbox_area,
        "bbox_gap_px": bbox_gap(bbox_a, bbox_b),
        "width_ratio": ratio_minmax(bbox_a[2] - bbox_a[0], bbox_b[2] - bbox_b[0]),
        "height_ratio": ratio_minmax(bbox_a[3] - bbox_a[1], bbox_b[3] - bbox_b[1]),
        "aspect_ratio_difference": abs(((bbox_a[2] - bbox_a[0]) / max(1.0, bbox_a[3] - bbox_a[1])) - ((bbox_b[2] - bbox_b[0]) / max(1.0, bbox_b[3] - bbox_b[1]))),
        "centroid_distance": float(math.hypot(ca[0] - cb[0], ca[1] - cb[1])),
        "centroid_distance_over_large_bbox_diagonal": float(math.hypot(ca[0] - cb[0], ca[1] - cb[1])) / large_diag,
        "normalized_centroid_distance": float(math.hypot(ca[0] - cb[0], ca[1] - cb[1])) / max(1.0, (math.hypot(bbox_a[2] - bbox_a[0], bbox_a[3] - bbox_a[1]) + math.hypot(bbox_b[2] - bbox_b[0], bbox_b[3] - bbox_b[1])) / 2.0),
        "contour_perimeter_ratio": ratio_minmax(props_a["perimeter"], props_b["perimeter"]) if not np.isnan(props_a["perimeter"]) and not np.isnan(props_b["perimeter"]) else np.nan,
        "solidity_difference": abs(props_a["solidity"] - props_b["solidity"]) if not np.isnan(props_a["solidity"]) and not np.isnan(props_b["solidity"]) else np.nan,
        "eccentricity_difference": abs(props_a["eccentricity"] - props_b["eccentricity"]) if not np.isnan(props_a["eccentricity"]) and not np.isnan(props_b["eccentricity"]) else np.nan,
        "orientation_difference": angle_diff(props_a["orientation"], props_b["orientation"]),
        "major_axis_angle_difference": angle_diff(props_a["orientation"], props_b["orientation"]),
        "convex_hull_area_ratio": ratio_minmax(props_a["hull_area"], props_b["hull_area"]) if not np.isnan(props_a["hull_area"]) and not np.isnan(props_b["hull_area"]) else np.nan,
        "detector_conf_min": min(a["det_confidence"], b["det_confidence"]),
        "detector_conf_max": max(a["det_confidence"], b["det_confidence"]),
        "detector_conf_abs_diff": abs(a["det_confidence"] - b["det_confidence"]),
        "detector_conf_ratio": ratio_minmax(a["det_confidence"], b["det_confidence"]),
        "segmentation_conf_min": min(a["seg_confidence"], b["seg_confidence"]),
        "segmentation_conf_max": max(a["seg_confidence"], b["seg_confidence"]),
        "segmentation_conf_abs_diff": abs(a["seg_confidence"] - b["seg_confidence"]),
        "segmentation_conf_ratio": ratio_minmax(a["seg_confidence"], b["seg_confidence"]),
    }
    if mask_a is None or mask_b is None:
        vals.update({"mask_iou": np.nan, "mask_area_ratio": ratio_minmax(area_a, area_b), "mask_area_min": min(area_a, area_b), "mask_area_max": max(area_a, area_b), "mask_area_abs_diff": abs(area_a - area_b)})
        return vals
    x1, y1 = int(math.floor(min(bbox_a[0], bbox_b[0]))), int(math.floor(min(bbox_a[1], bbox_b[1])))
    x2, y2 = int(math.ceil(max(bbox_a[2], bbox_b[2]))), int(math.ceil(max(bbox_a[3], bbox_b[3])))
    canvas_a = np.zeros((max(1, y2 - y1), max(1, x2 - x1)), dtype=bool)
    canvas_b = np.zeros_like(canvas_a)
    paste_mask(canvas_a, a, x1, y1)
    paste_mask(canvas_b, b, x1, y1)
    small_canvas = canvas_a if area_a <= area_b else canvas_b
    large_canvas = canvas_b if area_a <= area_b else canvas_a
    inter = int((canvas_a & canvas_b).sum())
    union = int((canvas_a | canvas_b).sum())
    small_area = max(1, int(small_canvas.sum()))
    ys, xs = np.nonzero(small_canvas)
    inside_bbox = ((xs + x1 >= lb[0]) & (xs + x1 <= lb[2]) & (ys + y1 >= lb[1]) & (ys + y1 <= lb[3])).sum() if len(xs) else 0
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    dilated_large = cv2.dilate(large_canvas.astype(np.uint8), kernel, iterations=1) > 0
    dist = cv2.distanceTransform((~large_canvas).astype(np.uint8), cv2.DIST_L2, 3)
    mask_gap = 0.0 if inter > 0 else float(dist[small_canvas].min()) if small_canvas.any() else np.nan
    vals.update({
        "mask_iou": inter / union if union else 0.0,
        "mask_area_ratio": ratio_minmax(area_a, area_b),
        "mask_area_min": min(area_a, area_b),
        "mask_area_max": max(area_a, area_b),
        "mask_area_abs_diff": abs(area_a - area_b),
        "small_mask_in_large_bbox": float(inside_bbox / small_area),
        "small_mask_inside_large_bbox_fraction": float(inside_bbox / small_area),
        "small_mask_inside_dilated_large_mask_fraction": float((small_canvas & dilated_large).sum() / small_area),
        "mask_gap_px": mask_gap,
        "masks_connect_with_small_dilation": bool(mask_gap <= 10.0) if not np.isnan(mask_gap) else False,
        "min_dilation_radius_to_connect_masks": mask_gap,
        "fragment_score": 100.0 * float(inside_bbox / small_area) * (1.0 - (inter / union if union else 0.0)),
        "expanded_overlap": float((small_canvas & dilated_large).sum() / small_area),
    })
    return vals


def candidate_pairs(rows: list[dict[str, Any]], cell_size: int = 512) -> list[tuple[int, int]]:
    grid: dict[tuple[int, int], list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        if row.get("mask") is None:
            continue
        b = row["bbox"]
        expand = max(10.0, 0.75 * math.hypot(b[2] - b[0], b[3] - b[1]) + 10.0)
        for gy in range(int(math.floor((b[1] - expand) / cell_size)), int(math.floor((b[3] + expand) / cell_size)) + 1):
            for gx in range(int(math.floor((b[0] - expand) / cell_size)), int(math.floor((b[2] + expand) / cell_size)) + 1):
                grid[(gx, gy)].append(idx)
    pairs = set()
    for indices in grid.values():
        for i, a in enumerate(indices):
            for b in indices[i + 1:]:
                pairs.add(tuple(sorted((a, b))))
    return sorted(pairs)


def connected_groups(edges: list[tuple[int, int]]) -> list[list[int]]:
    adj: dict[int, set[int]] = defaultdict(set)
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)
    groups, visited = [], set()
    for det_id in sorted(adj):
        if det_id in visited:
            continue
        stack, members = [det_id], []
        visited.add(det_id)
        while stack:
            cur = stack.pop()
            members.append(cur)
            for nb in sorted(adj[cur]):
                if nb not in visited:
                    visited.add(nb)
                    stack.append(nb)
        groups.append(sorted(members))
    return groups


def best_keep_id(members: list[int], by_id: dict[int, dict[str, Any]]) -> int:
    return max(members, key=lambda det_id: (by_id[det_id]["det_confidence"], by_id[det_id]["seg_confidence"], -det_id))


def apply_classifier_dedup(rows: list[dict[str, Any]], payload: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    model = payload["model"]
    features = payload.get("features", FEATURES)
    by_id = {int(row["detection_id"]): row for row in rows}
    audit, edges = [], []
    for a_idx, b_idx in candidate_pairs(rows):
        a, b = rows[a_idx], rows[b_idx]
        vals = pair_features(a, b)
        reason = []
        if float(vals.get("mask_iou") or 0.0) >= 0.20:
            reason.append("mask_iou")
        if float(vals.get("bbox_iou") or 0.0) >= 0.15:
            reason.append("bbox_iou")
        if float(vals.get("normalized_centroid_distance") or 999.0) <= 0.75:
            reason.append("centroid")
        if float(vals.get("mask_gap_px") or 999.0) <= 10.0:
            reason.append("mask_gap")
        if float(vals.get("bbox_gap_px") or 999.0) <= 10.0:
            reason.append("bbox_gap")
        if float(vals.get("fragment_score") or 0.0) >= 25.0:
            reason.append("fragment")
        if not reason:
            continue
        frame = pd.DataFrame([{feature: vals.get(feature, np.nan) for feature in features}], columns=features)
        prob = float(model.predict_proba(frame)[:, 1][0])
        pred = prob >= PARAMS["dedup_threshold"]
        a_id, b_id = int(a["detection_id"]), int(b["detection_id"])
        if pred:
            edges.append((a_id, b_id))
        audit.append({
            "detection_id_a": min(a_id, b_id), "detection_id_b": max(a_id, b_id),
            "duplicate_probability": prob, "predicted_duplicate": bool(pred),
            "reason": ";".join(reason), "action": "keep_both", "keep_detection_id": "", "drop_detection_id": "",
            **{k: vals.get(k, np.nan) for k in features},
        })
    groups = connected_groups(edges)
    drop_ids: set[int] = set()
    group_info: dict[int, dict[str, Any]] = {}
    unsafe_groups = 0
    for group_id, members in enumerate(groups, start=1):
        unsafe = len(members) > 3
        if unsafe:
            unsafe_groups += 1
            keep_id, group_drop = "", set()
        else:
            keep_id = best_keep_id(members, by_id)
            group_drop = {det_id for det_id in members if det_id != keep_id}
            drop_ids.update(group_drop)
        for det_id in members:
            group_info[det_id] = {"group_id": group_id, "size": len(members), "unsafe": unsafe, "keep": keep_id, "drop": group_drop}
    for row in audit:
        if not row["predicted_duplicate"]:
            continue
        info = group_info[int(row["detection_id_a"])]
        row["duplicate_group_id"] = info["group_id"]
        row["group_size"] = info["size"]
        row["unsafe_group"] = info["unsafe"]
        if info["unsafe"]:
            row["action"] = "unsafe_group_no_action"
            continue
        row["keep_detection_id"] = info["keep"]
        pair_drop = [d for d in [int(row["detection_id_a"]), int(row["detection_id_b"])] if d in info["drop"]]
        row["drop_detection_id"] = ";".join(str(d) for d in pair_drop)
        row["action"] = "auto_remove" if pair_drop else "keep_both"
    kept = [row for row in rows if int(row["detection_id"]) not in drop_ids]
    stats = {
        "classifier_dedup_threshold": PARAMS["dedup_threshold"],
        "candidate_pairs_evaluated": len(audit),
        "predicted_duplicate_pairs": sum(1 for row in audit if row["predicted_duplicate"]),
        "duplicate_groups": len(groups),
        "unsafe_groups": unsafe_groups,
        "removed_by_classifier_dedup": len(drop_ids),
    }
    return kept, audit, stats


def draw_crop_overlay(crop: np.ndarray, mask: np.ndarray | None, det_id: int, label: str) -> np.ndarray:
    out = crop.copy()
    contour = contour_from_mask(mask)
    if contour is not None:
        cv2.drawContours(out, [contour], -1, CONTOUR_COLOR, CONTOUR_WIDTH, cv2.LINE_AA)
        cx, cy = contour.reshape(-1, 2).mean(axis=0).astype(int)
        cv2.putText(out, str(det_id), (cx + 3, cy - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.55, CONTOUR_COLOR, 2, cv2.LINE_AA)
    else:
        cv2.putText(out, "NO MASK", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(out, label, (8, max(20, out.shape[0] - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(out, label, (8, max(20, out.shape[0] - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def draw_full_overlay(image: np.ndarray, rows: list[dict[str, Any]]) -> np.ndarray:
    out = image.copy()
    for row in rows:
        contour = contour_from_mask(row.get("mask"))
        if contour is None:
            continue
        global_contour = contour.copy()
        global_contour[:, 0, 0] += int(row["crop_x1"])
        global_contour[:, 0, 1] += int(row["crop_y1"])
        cv2.drawContours(out, [global_contour], -1, CONTOUR_COLOR, CONTOUR_WIDTH, cv2.LINE_AA)
        gx, gy = global_contour.reshape(-1, 2).mean(axis=0).astype(int)
        cv2.putText(out, str(row["detection_id"]), (gx + 4, gy - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, CONTOUR_COLOR, 2, cv2.LINE_AA)
    return out


def add_banner(img: np.ndarray, text: str, height: int = 38) -> np.ndarray:
    out = cv2.copyMakeBorder(img, height, 0, 0, 0, cv2.BORDER_CONSTANT, value=(245, 245, 245))
    cv2.putText(out, text, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    return out


def resize_width(img: np.ndarray, width: int) -> np.ndarray:
    h, w = img.shape[:2]
    if w == 0 or h == 0:
        return np.zeros((width, width, 3), dtype=np.uint8)
    return cv2.resize(img, (width, max(1, int(round(h * width / w)))), interpolation=cv2.INTER_AREA)


def side_by_side(original: np.ndarray, v5: np.ndarray, v7: np.ndarray, labels: tuple[str, str, str]) -> np.ndarray:
    panels = [add_banner(original, labels[0]), add_banner(v5, labels[1]), add_banner(v7, labels[2])]
    max_h = max(p.shape[0] for p in panels)
    padded = [cv2.copyMakeBorder(p, 0, max_h - p.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(245, 245, 245)) for p in panels]
    return np.concatenate(padded, axis=1)


def write_contact_sheet(paths: list[Path], output: Path, cols: int = 3, width: int = 900) -> None:
    thumbs = []
    for path in paths:
        img = cv2.imread(str(path))
        if img is not None:
            thumbs.append(resize_width(img, width))
    if not thumbs:
        return
    tile_h = max(t.shape[0] for t in thumbs)
    sheet = np.full((math.ceil(len(thumbs) / cols) * tile_h, cols * width, 3), 245, dtype=np.uint8)
    for i, thumb in enumerate(thumbs):
        y, x = (i // cols) * tile_h, (i % cols) * width
        sheet[y:y + thumb.shape[0], x:x + thumb.shape[1]] = thumb
    cv2.imwrite(str(output), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])


def segment_one_model(
    source: str,
    model_label: str,
    seg_model: YOLO,
    image_bgr: np.ndarray,
    detections: list[dict[str, Any]],
    out_dir: Path,
    dedup_payload: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    model_dir = out_dir / model_label
    for sub in ["all_crops", "all_masks", "all_overlays", "kept_crops", "kept_masks"]:
        (model_dir / sub).mkdir(parents=True, exist_ok=True)
    h, w = image_bgr.shape[:2]
    rows = []
    for det in detections:
        det_id = int(det["detection_id"])
        x1, y1 = max(0, int(np.floor(det["x1"]))), max(0, int(np.floor(det["y1"])))
        x2, y2 = min(w, int(np.ceil(det["x2"]))), min(h, int(np.ceil(det["y2"])))
        crop = image_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        result = seg_model.predict(crop, imgsz=PARAMS["seg_imgsz"], conf=PARAMS["seg_conf"], device=PARAMS["device"], verbose=False)[0]
        mask, seg_conf = result_mask(result, crop.shape[:2])
        stem = f"{source}_det{det_id:04d}"
        crop_path = model_dir / "all_crops" / f"{stem}.png"
        mask_path = model_dir / "all_masks" / f"{stem}_mask.png"
        overlay_path = model_dir / "all_overlays" / f"{stem}_overlay.png"
        cv2.imwrite(str(crop_path), crop)
        if mask is not None:
            cv2.imwrite(str(mask_path), mask)
        cv2.imwrite(str(overlay_path), draw_crop_overlay(crop, mask, det_id, model_label))
        rows.append({
            "detection_id": det_id,
            "bbox": (float(det["x1"]), float(det["y1"]), float(det["x2"]), float(det["y2"])),
            "bbox_x1": float(det["x1"]), "bbox_y1": float(det["y1"]), "bbox_x2": float(det["x2"]), "bbox_y2": float(det["y2"]),
            "crop_x1": x1, "crop_y1": y1, "crop_x2": x2, "crop_y2": y2,
            "det_confidence": float(det["conf"]), "seg_confidence": float(seg_conf or 0.0),
            "has_mask": mask is not None, "mask": mask, "mask_x": x1, "mask_y": y1,
            "mask_area_px": int(np.count_nonzero(mask)) if mask is not None else 0,
            "crop_path": crop_path, "mask_path": mask_path if mask is not None else Path(""), "overlay_path": overlay_path,
        })
        if (det_id + 1) % 100 == 0 or det_id + 1 == len(detections):
            print(f"  {model_label} segment {source}: {det_id + 1}/{len(detections)}", flush=True)
    kept, audit, dedup_stats = apply_classifier_dedup(rows, dedup_payload)
    kept_ids = {int(row["detection_id"]) for row in kept}
    for row in kept:
        shutil.copy2(row["crop_path"], model_dir / "kept_crops" / row["crop_path"].name)
        if row["has_mask"] and row["mask_path"]:
            shutil.copy2(row["mask_path"], model_dir / "kept_masks" / row["mask_path"].name)
    full_overlay = draw_full_overlay(image_bgr, kept)
    full_overlay_path = model_dir / f"{source}_{model_label}_final_full_overlay.jpg"
    cv2.imwrite(str(full_overlay_path), full_overlay, [cv2.IMWRITE_JPEG_QUALITY, 92])
    final_csv = model_dir / f"{source}_{model_label}_final.csv"
    pd.DataFrame([{k: rel(v) if isinstance(v, Path) and str(v) else v for k, v in row.items() if k not in {"mask", "bbox"}} for row in kept]).to_csv(final_csv, index=False)
    audit_csv = model_dir / f"{source}_{model_label}_classifier_dedup.csv"
    pd.DataFrame(audit).to_csv(audit_csv, index=False)
    summary = {
        "model_label": model_label,
        "initial_detections_after_bbox_nms": len(detections),
        "segmented_rows": len(rows),
        "classifier_removed": len(rows) - len(kept),
        "final_count": len(kept),
        "kept_detection_ids": sorted(kept_ids),
        "final_csv": rel(final_csv),
        "classifier_dedup_csv": rel(audit_csv),
        "full_overlay": rel(full_overlay_path),
        **dedup_stats,
    }
    (model_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return kept, summary


def zoom_regions(image_bgr: np.ndarray) -> dict[str, tuple[int, int, int, int]]:
    h, w = image_bgr.shape[:2]
    return {
        "ruler_left": (int(w * 0.28), int(h * 0.07), int(w * 0.62), int(h * 0.30)),
        "ruler_center": (int(w * 0.36), int(h * 0.08), int(w * 0.76), int(h * 0.34)),
        "ruler_right": (int(w * 0.48), int(h * 0.08), int(w * 0.92), int(h * 0.36)),
    }


def write_side_by_sides(source: str, image_bgr: np.ndarray, v5_summary: dict[str, Any], v7_summary: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    side_dir = out_dir / "side_by_side"
    side_dir.mkdir(parents=True, exist_ok=True)
    v5_img = cv2.imread(str(ROOT / v5_summary["full_overlay"]))
    v7_img = cv2.imread(str(ROOT / v7_summary["full_overlay"]))
    preview_w = 1400
    full_sbs = side_by_side(resize_width(image_bgr, preview_w), resize_width(v5_img, preview_w), resize_width(v7_img, preview_w), ("original", "v5 final", "v7 final"))
    full_path = side_dir / f"{source}_full_side_by_side.jpg"
    cv2.imwrite(str(full_path), full_sbs, [cv2.IMWRITE_JPEG_QUALITY, 90])
    zoom_paths = []
    for name, (x1, y1, x2, y2) in zoom_regions(image_bgr).items():
        crop = side_by_side(image_bgr[y1:y2, x1:x2], v5_img[y1:y2, x1:x2], v7_img[y1:y2, x1:x2], (f"original {name}", "v5 final", "v7 final"))
        path = side_dir / f"{source}_{name}_zoom_side_by_side.jpg"
        cv2.imwrite(str(path), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
        zoom_paths.append(path)
    return {"full_side_by_side": rel(full_path), "zoom_side_by_side": [rel(p) for p in zoom_paths]}


def process_image(image_path: Path, detector: YOLO, v5: YOLO, v7: YOLO, dedup_payload: dict[str, Any]) -> dict[str, Any]:
    source = safe_name(image_path)
    out_dir = OUT / source
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Processing {source}", flush=True)
    detections, detector_meta = run_detector_once(image_path, detector, out_dir)
    image_bgr = cv2.cvtColor(np.array(Image.open(image_path).convert("RGB")), cv2.COLOR_RGB2BGR)
    v5_kept, v5_summary = segment_one_model(source, "v5_last", v5, image_bgr, detections, out_dir, dedup_payload)
    v7_kept, v7_summary = segment_one_model(source, "v7_best", v7, image_bgr, detections, out_dir, dedup_payload)
    side = write_side_by_sides(source, image_bgr, v5_summary, v7_summary, out_dir)
    summary = {
        "source_image": rel(image_path), "source_id": source,
        "detections_before_bbox_nms": detector_meta["detections_before_bbox_nms"],
        "detections_after_bbox_nms": detector_meta["detections_after_bbox_nms"],
        "v5": v5_summary, "v7": v7_summary, **side,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def write_readme(summary: dict[str, Any]) -> None:
    lines = [
        "# Final Pipeline v5 vs v7 Visual Comparison",
        "",
        "Leakage-free visual comparison on five full-resolution ruler/scale images. No manual ground truth labels were used and no winner is declared.",
        "",
        "## Models",
        f"- Detector requested: `models/yolo11n_tiled_best.pt`",
        f"- Detector used: `{rel(DETECTOR)}` (resolved --detector-model)",
        f"- v5 segmentation: `{rel(V5_SEG)}`",
        f"- v7 segmentation: `{rel(V7_SEG)}`",
        f"- Final dedup classifier: `{rel(DEDUP_MODEL)}`",
        "",
        "## Parameters",
        "- tile size: 1280",
        "- overlap: 256",
        "- stride: 1024",
        "- detector confidence: 0.6",
        "- global bbox NMS IoU: 0.5",
        "- segmentation confidence: 0.15",
        "- segmentation imgsz: 320",
        "- final classifier dedup threshold: 0.50",
        "- envelope mode: disabled",
        "- mask-IoU dedup: disabled",
        "- restricted v5 cleanup: disabled",
        "",
        "## Pipeline Order",
        "1. Run tiled detector once per full-resolution image.",
        "2. Apply global bbox NMS.",
        "3. Send the same post-NMS bbox crops to v5 and v7 segmentation separately.",
        "4. Apply the same v7 fragment-aware classifier dedup separately to v5 and v7 results.",
        "5. Save final overlays, final CSVs, kept crops/masks, side-by-side full views and ruler zooms.",
        "",
        "## Output Index",
    ]
    for item in summary["images"]:
        lines.append(f"- `{item['source_id']}`: {item['detections_after_bbox_nms']} bbox-NMS detections, v5 final {item['v5']['final_count']}, v7 final {item['v7']['final_count']}, side-by-side `{item['full_side_by_side']}`")
    (OUT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def check_inputs() -> dict[str, Any]:
    missing = [str(path) for path in required_paths() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required inputs: " + json.dumps(missing, indent=2))
    payload = joblib.load(DEDUP_MODEL)
    features = payload.get("features", [])
    return {
        "all_inputs_exist": True,
        "dedup_threshold": PARAMS["dedup_threshold"],
        "mask_iou_dedup_enabled": PARAMS["mask_iou_dedup"],
        "restricted_v5_cleanup_enabled": PARAMS["restricted_v5_cleanup"],
        "dedup_model_features": len(features),
        "dedup_features_match_expected": all(feature in features for feature in FEATURES[:15]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run leakage-free final end-to-end v5/v7 visual comparison")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and print planned run without loading YOLO models")
    parser.add_argument("--check", action="store_true", help="Validate inputs and fixed parameters")
    parser.add_argument("--detector-model", type=Path, default=DETECTOR)
    parser.add_argument("--dedup-model", type=Path, default=DEDUP_MODEL)
    return parser.parse_args()


def main() -> None:
    global DETECTOR, DEDUP_MODEL
    args = parse_args()
    DETECTOR = args.detector_model.resolve()
    DEDUP_MODEL = args.dedup_model.resolve()
    checks = check_inputs()
    if args.dry_run or args.check:
        print(json.dumps({"status": "ok", "output_dir": rel(OUT), "checks": checks, "parameters": PARAMS, "images": [rel(p) for p in IMAGES]}, indent=2))
        return
    for child in list(OUT.iterdir()) if OUT.exists() else []:
        if child.name == Path(__file__).name:
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()
    OUT.mkdir(parents=True, exist_ok=True)
    print("Loading models", flush=True)
    detector = YOLO(DETECTOR)
    v5 = YOLO(V5_SEG)
    v7 = YOLO(V7_SEG)
    dedup_payload = joblib.load(DEDUP_MODEL)
    images = [process_image(path, detector, v5, v7, dedup_payload) for path in IMAGES]
    overall_contact_inputs = []
    for item in images:
        overall_contact_inputs.append(ROOT / item["full_side_by_side"])
        overall_contact_inputs.extend(ROOT / p for p in item["zoom_side_by_side"])
    overall_contact = OUT / "overall_full_and_ruler_zoom_contact_sheet.jpg"
    write_contact_sheet(overall_contact_inputs, overall_contact, cols=2, width=1100)
    summary = {
        "status": "completed", "leakage_free": True, "manual_ground_truth_metrics": False,
        "models": {"detector": str(DETECTOR), "v5_seg": rel(V5_SEG), "v7_seg": rel(V7_SEG), "dedup_classifier": str(DEDUP_MODEL)},
        "parameters": PARAMS, "checks": checks, "images": images,
        "overall_contact_sheet": rel(overall_contact), "completed_at_unix": time.time(),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_readme(summary)
    print(json.dumps({
        "summary": rel(OUT / "summary.json"),
        "overall_contact_sheet": rel(overall_contact),
        "counts": {item["source_id"]: {"bbox_nms": item["detections_after_bbox_nms"], "v5_final": item["v5"]["final_count"], "v7_final": item["v7"]["final_count"]} for item in images},
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
