#!/usr/bin/env python3
"""Run detector + trunk segmentation model on unseen full-source images."""

import argparse
import json
import math
import shutil
import sys
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

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.infer_tiled import infer_tiled  # noqa: E402
from scripts.train_duplicate_classifier_v7_fragment_aware import FEATURES as V7_FEATURES  # noqa: E402
from scripts.train_duplicate_classifier_v7_fragment_aware import pair_mask_features as v7_pair_mask_features  # noqa: E402


def safe_name(path: Path) -> str:
    return path.stem.replace(" ", "_").replace("(", "").replace(")", "")


def resolve_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


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


def normalize_binary_mask(mask: Any) -> np.ndarray:
    mask_arr = np.asarray(mask)
    if mask_arr.ndim == 3 and mask_arr.shape[2] == 1:
        mask_arr = mask_arr[:, :, 0]
    elif mask_arr.ndim != 2:
        raise ValueError(f"Expected mask shape HxW or HxWx1, got {mask_arr.shape}")
    return mask_arr > 0


def row_mask(row: dict[str, Any]) -> tuple[np.ndarray, int, int] | None:
    if not row.get("has_mask") or not row.get("mask_path"):
        return None
    mask = cv2.imread(str(row["mask_path"]), cv2.IMREAD_GRAYSCALE)
    if mask is None or not np.any(mask):
        return None
    x = max(0, int(np.floor(float(row["bbox_x1"]))))
    y = max(0, int(np.floor(float(row["bbox_y1"]))))
    return normalize_binary_mask(mask), x, y


def source_mask_canvas(mask: np.ndarray, x: int, y: int, origin_x: int, origin_y: int, width: int, height: int) -> np.ndarray:
    mask = normalize_binary_mask(mask)
    canvas = np.zeros((height, width), dtype=bool)
    x1, y1 = x - origin_x, y - origin_y
    x2, y2 = x1 + mask.shape[1], y1 + mask.shape[0]
    dst_x1, dst_y1 = max(0, x1), max(0, y1)
    dst_x2, dst_y2 = min(width, x2), min(height, y2)
    if dst_x2 <= dst_x1 or dst_y2 <= dst_y1:
        return canvas
    src_x1, src_y1 = dst_x1 - x1, dst_y1 - y1
    src_x2, src_y2 = src_x1 + (dst_x2 - dst_x1), src_y1 + (dst_y2 - dst_y1)
    canvas[dst_y1:dst_y2, dst_x1:dst_x2] = mask[src_y1:src_y2, src_x1:src_x2]
    return canvas


def source_mask_iou(a: dict[str, Any], b: dict[str, Any]) -> float | None:
    ma = row_mask(a)
    mb = row_mask(b)
    if ma is None or mb is None:
        return None
    mask_a, ax1, ay1 = ma
    mask_b, bx1, by1 = mb
    ax2, ay2 = ax1 + mask_a.shape[1], ay1 + mask_a.shape[0]
    bx2, by2 = bx1 + mask_b.shape[1], by1 + mask_b.shape[0]
    ux1, uy1 = min(ax1, bx1), min(ay1, by1)
    ux2, uy2 = max(ax2, bx2), max(ay2, by2)
    ca = source_mask_canvas(mask_a, ax1, ay1, ux1, uy1, ux2 - ux1, uy2 - uy1)
    cb = source_mask_canvas(mask_b, bx1, by1, ux1, uy1, ux2 - ux1, uy2 - uy1)
    union = int(np.logical_or(ca, cb).sum())
    if union == 0:
        return 0.0
    inter = int(np.logical_and(ca, cb).sum())
    return float(inter / union)


def bbox_gap_px(a: dict[str, Any], b: dict[str, Any]) -> float:
    ax1, ay1, ax2, ay2 = float(a["bbox_x1"]), float(a["bbox_y1"]), float(a["bbox_x2"]), float(a["bbox_y2"])
    bx1, by1, bx2, by2 = float(b["bbox_x1"]), float(b["bbox_y1"]), float(b["bbox_x2"]), float(b["bbox_y2"])
    gap_x = max(0.0, max(ax1, bx1) - min(ax2, bx2))
    gap_y = max(0.0, max(ay1, by1) - min(ay2, by2))
    return float(np.hypot(gap_x, gap_y))


def spatial_candidate_pairs(rows: list[dict[str, Any]], max_gap: float = 10.0, cell_size: int = 512) -> list[tuple[int, int]]:
    grid: dict[tuple[int, int], list[int]] = {}
    for idx, row in enumerate(rows):
        x1 = float(row["bbox_x1"]) - max_gap
        y1 = float(row["bbox_y1"]) - max_gap
        x2 = float(row["bbox_x2"]) + max_gap
        y2 = float(row["bbox_y2"]) + max_gap
        gx1 = int(np.floor(x1 / cell_size))
        gy1 = int(np.floor(y1 / cell_size))
        gx2 = int(np.floor(x2 / cell_size))
        gy2 = int(np.floor(y2 / cell_size))
        for gy in range(gy1, gy2 + 1):
            for gx in range(gx1, gx2 + 1):
                grid.setdefault((gx, gy), []).append(idx)

    pairs: set[tuple[int, int]] = set()
    for indices in grid.values():
        for i, a_idx in enumerate(indices):
            for b_idx in indices[i + 1:]:
                a, b = rows[a_idx], rows[b_idx]
                if bbox_gap_px(a, b) <= max_gap:
                    pairs.add(tuple(sorted((a_idx, b_idx))))
    return sorted(pairs)


def mask_centroid(row: dict[str, Any]) -> tuple[float, float]:
    rm = row_mask(row)
    if rm is not None:
        mask, x, y = rm
        ys, xs = np.nonzero(normalize_binary_mask(mask))
        if len(xs):
            return float(xs.mean() + x), float(ys.mean() + y)
    return (
        (float(row["bbox_x1"]) + float(row["bbox_x2"])) / 2.0,
        (float(row["bbox_y1"]) + float(row["bbox_y2"])) / 2.0,
    )


def better_duplicate_member(a: dict[str, Any], b: dict[str, Any]) -> tuple[dict[str, Any], str]:
    det_a = float(a.get("det_confidence") or 0.0)
    det_b = float(b.get("det_confidence") or 0.0)
    if abs(det_a - det_b) > 1e-6:
        return (a if det_a > det_b else b), "higher_detector_confidence"
    seg_a = float(a.get("seg_confidence") or 0.0)
    seg_b = float(b.get("seg_confidence") or 0.0)
    if abs(seg_a - seg_b) > 1e-6:
        return (a if seg_a > seg_b else b), "segmentation_confidence_tiebreak"
    return a, "segmentation_confidence_tiebreak"


def deduplicate_by_mask_iou(rows: list[dict[str, Any]], threshold: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    start = time.perf_counter()
    masked = [r for r in rows if r.get("has_mask") and r.get("mask_path")]
    by_id = {int(r["detection_id"]): r for r in masked}
    total_possible_pairs = len(masked) * (len(masked) - 1) // 2
    candidate_pairs = spatial_candidate_pairs(masked)
    edges: list[tuple[int, int, float]] = []
    mask_iou_calculations = 0
    for a_idx, b_idx in candidate_pairs:
        a = masked[a_idx]
        b = masked[b_idx]
        mask_iou_calculations += 1
        miou = source_mask_iou(a, b)
        if miou is not None and miou >= threshold:
            edges.append((int(a["detection_id"]), int(b["detection_id"]), miou))

    adjacency: dict[int, set[int]] = {det_id: set() for det_id in by_id}
    for a_id, b_id, _miou in edges:
        adjacency[a_id].add(b_id)
        adjacency[b_id].add(a_id)

    groups: dict[int, list[int]] = {}
    visited: set[int] = set()
    group_id = 0
    for det_id in sorted(adjacency):
        if det_id in visited or not adjacency[det_id]:
            continue
        group_id += 1
        stack = [det_id]
        members = []
        visited.add(det_id)
        while stack:
            current = stack.pop()
            members.append(current)
            for neighbor in sorted(adjacency[current]):
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)
        groups[group_id] = sorted(members)

    group_by_id = {det_id: gid for gid, members in groups.items() for det_id in members}
    group_keep: dict[int, tuple[int, str]] = {}
    drop_ids: set[int] = set()
    for gid, members in groups.items():
        best = by_id[members[0]]
        for det_id in members[1:]:
            best, _reason = better_duplicate_member(best, by_id[det_id])
        keep_id = int(best["detection_id"])
        max_detector_conf = max(float(by_id[det_id].get("det_confidence") or 0.0) for det_id in members)
        tied_at_max = sum(1 for det_id in members if abs(float(by_id[det_id].get("det_confidence") or 0.0) - max_detector_conf) <= 1e-6)
        keep_reason = "segmentation_confidence_tiebreak" if tied_at_max > 1 else "higher_detector_confidence"
        group_keep[gid] = (keep_id, keep_reason)
        drop_ids.update(det_id for det_id in members if det_id != keep_id)

    audit_rows = []
    for a_id, b_id, miou in edges:
        gid = group_by_id[a_id]
        keep_id, keep_reason = group_keep[gid]
        if a_id == keep_id:
            drop_id = b_id
        elif b_id == keep_id:
            drop_id = a_id
        else:
            pair_best, _pair_reason = better_duplicate_member(by_id[a_id], by_id[b_id])
            drop_id = b_id if int(pair_best["detection_id"]) == a_id else a_id
        a = by_id[a_id]
        b = by_id[b_id]
        audit_rows.append({
            "detection_id_a": a_id,
            "detection_id_b": b_id,
            "mask_iou": miou,
            "detector_conf_a": float(a.get("det_confidence") or 0.0),
            "detector_conf_b": float(b.get("det_confidence") or 0.0),
            "segmentation_conf_a": float(a.get("seg_confidence") or 0.0),
            "segmentation_conf_b": float(b.get("seg_confidence") or 0.0),
            "duplicate_group_id": gid,
            "keep_detection_id": keep_id,
            "drop_detection_id": drop_id,
            "keep_reason": keep_reason,
            "threshold": float(threshold),
        })

    kept_rows = [r for r in rows if int(r["detection_id"]) not in drop_ids]
    stats = {
        "dedup_detections": int(len(masked)),
        "dedup_total_possible_pairs": int(total_possible_pairs),
        "dedup_spatial_prefilter_pairs": int(len(candidate_pairs)),
        "dedup_mask_iou_calculations": int(mask_iou_calculations),
        "dedup_duplicate_groups": int(len(groups)),
        "dedup_removed_detections": int(len(drop_ids)),
        "dedup_seconds": float(time.perf_counter() - start),
    }
    return kept_rows, audit_rows, stats


def draw_full_overlay(image_bgr: np.ndarray, rows: list[dict[str, Any]], use_envelope_contour: bool) -> np.ndarray:
    full_overlay = image_bgr.copy()
    for row in rows:
        if not row.get("has_mask"):
            continue
        mask_path = row.get("envelope_mask_path") if use_envelope_contour and row.get("envelope_mask_path") else row.get("mask_path")
        if not mask_path:
            continue
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        mask = (normalize_binary_mask(mask).astype(np.uint8) * 255)
        contour = contour_from_mask(mask)
        if contour is None:
            continue
        x1 = max(0, int(np.floor(float(row["bbox_x1"]))))
        y1 = max(0, int(np.floor(float(row["bbox_y1"]))))
        global_contour = contour.copy()
        global_contour[:, 0, 0] += x1
        global_contour[:, 0, 1] += y1
        cv2.drawContours(full_overlay, [global_contour], -1, (0, 255, 255), 2, cv2.LINE_AA)
        gx, gy = global_contour.reshape(-1, 2).mean(axis=0).astype(int)
        cv2.putText(full_overlay, str(row["detection_id"]), (gx + 4, gy - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    return full_overlay


def v7_row_detection(row: dict[str, Any]) -> dict[str, Any] | None:
    rm = row_mask(row)
    if rm is None:
        return None
    mask, _x, _y = rm
    return {
        "row": {
            "det_confidence": row.get("det_confidence") or 0.0,
            "seg_confidence": row.get("seg_confidence") or 0.0,
        },
        "bbox": (float(row["bbox_x1"]), float(row["bbox_y1"]), float(row["bbox_x2"]), float(row["bbox_y2"])),
        "mask": mask,
    }


def bbox_area_xyxy(bbox: tuple[float, float, float, float]) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def bbox_diagonal_row(row: dict[str, Any]) -> float:
    return float(math.hypot(float(row["bbox_x2"]) - float(row["bbox_x1"]), float(row["bbox_y2"]) - float(row["bbox_y1"])))


def v7_conservative_candidate_pairs(rows: list[dict[str, Any]], cell_size: int = 512) -> list[tuple[int, int]]:
    masked = [idx for idx, row in enumerate(rows) if row.get("has_mask") and row.get("mask_path")]
    grid: dict[tuple[int, int], list[int]] = {}
    for idx in masked:
        row = rows[idx]
        expansion = max(10.0, 0.75 * bbox_diagonal_row(row) + 10.0)
        gx1 = int(math.floor((float(row["bbox_x1"]) - expansion) / cell_size))
        gy1 = int(math.floor((float(row["bbox_y1"]) - expansion) / cell_size))
        gx2 = int(math.floor((float(row["bbox_x2"]) + expansion) / cell_size))
        gy2 = int(math.floor((float(row["bbox_y2"]) + expansion) / cell_size))
        for gy in range(gy1, gy2 + 1):
            for gx in range(gx1, gx2 + 1):
                grid.setdefault((gx, gy), []).append(idx)
    pairs: set[tuple[int, int]] = set()
    for indices in grid.values():
        for i, a_idx in enumerate(indices):
            for b_idx in indices[i + 1:]:
                pairs.add(tuple(sorted((a_idx, b_idx))))
    return sorted(pairs)


def v7_feature_row(a: dict[str, Any], b: dict[str, Any], features: list[str]) -> tuple[dict[str, Any], str]:
    det_a = v7_row_detection(a)
    det_b = v7_row_detection(b)
    if det_a is None or det_b is None:
        return {feature: np.nan for feature in features}, "missing_mask"
    vals = v7_pair_mask_features(det_a, det_b)
    det_conf_a = float(a.get("det_confidence") or 0.0)
    det_conf_b = float(b.get("det_confidence") or 0.0)
    seg_conf_a = float(a.get("seg_confidence") or 0.0)
    seg_conf_b = float(b.get("seg_confidence") or 0.0)
    vals.update({
        "detector_conf_min": min(det_conf_a, det_conf_b),
        "detector_conf_max": max(det_conf_a, det_conf_b),
        "detector_conf_abs_diff": abs(det_conf_a - det_conf_b),
        "detector_conf_ratio": min(det_conf_a, det_conf_b) / max(det_conf_a, det_conf_b) if max(det_conf_a, det_conf_b) > 0 else 0.0,
        "segmentation_conf_min": min(seg_conf_a, seg_conf_b),
        "segmentation_conf_max": max(seg_conf_a, seg_conf_b),
        "segmentation_conf_abs_diff": abs(seg_conf_a - seg_conf_b),
        "segmentation_conf_ratio": min(seg_conf_a, seg_conf_b) / max(seg_conf_a, seg_conf_b) if max(seg_conf_a, seg_conf_b) > 0 else 0.0,
    })
    reasons = []
    if float(vals.get("mask_iou") or 0.0) >= 0.20:
        reasons.append("mask_iou")
    if float(vals.get("bbox_iou") or 0.0) >= 0.15:
        reasons.append("bbox_iou")
    if float(vals.get("normalized_centroid_distance") or 999.0) <= 0.75:
        reasons.append("centroid")
    if float(vals.get("mask_gap_px") or 999.0) <= 10.0:
        reasons.append("mask_gap")
    if float(vals.get("bbox_gap_px") or 999.0) <= 10.0:
        reasons.append("bbox_gap")
    if float(vals.get("fragment_score") or 0.0) >= 25.0:
        reasons.append("fragment")
    return {feature: vals.get(feature, np.nan) for feature in features}, ";".join(reasons)


def connected_duplicate_groups(edges: list[tuple[int, int]]) -> list[list[int]]:
    adjacency: dict[int, set[int]] = defaultdict(set)
    for a_id, b_id in edges:
        adjacency[a_id].add(b_id)
        adjacency[b_id].add(a_id)
    groups = []
    visited: set[int] = set()
    for det_id in sorted(adjacency):
        if det_id in visited:
            continue
        stack = [det_id]
        visited.add(det_id)
        members = []
        while stack:
            cur = stack.pop()
            members.append(cur)
            for nb in sorted(adjacency[cur]):
                if nb not in visited:
                    visited.add(nb)
                    stack.append(nb)
        groups.append(sorted(members))
    return groups


def best_v7_keep_id(members: list[int], by_id: dict[int, dict[str, Any]]) -> int:
    return max(
        members,
        key=lambda det_id: (
            float(by_id[det_id].get("det_confidence") or 0.0),
            float(by_id[det_id].get("seg_confidence") or 0.0),
            -int(det_id),
        ),
    )


def deduplicate_by_v7_classifier(rows: list[dict[str, Any]], model_path: Path, threshold: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    start = time.perf_counter()
    payload = joblib.load(model_path)
    model = payload["model"]
    features = payload.get("features", V7_FEATURES)
    by_id = {int(row["detection_id"]): row for row in rows}
    candidate_pairs = v7_conservative_candidate_pairs(rows)
    audit_rows = []
    edges = []
    for a_idx, b_idx in candidate_pairs:
        a = rows[a_idx]
        b = rows[b_idx]
        a_id = int(a["detection_id"])
        b_id = int(b["detection_id"])
        feats, reason = v7_feature_row(a, b, features)
        if not reason:
            continue
        prob = float(model.predict_proba(pd.DataFrame([feats], columns=features))[:, 1][0])
        predicted = prob >= threshold
        if predicted:
            edges.append((a_id, b_id))
        audit_rows.append({
            "detection_id_a": min(a_id, b_id),
            "detection_id_b": max(a_id, b_id),
            "duplicate_probability": prob,
            "predicted_duplicate": bool(predicted),
            "duplicate_group_id": "",
            "group_size": "",
            "unsafe_group": False,
            "detector_conf_a": float(a.get("det_confidence") or 0.0),
            "detector_conf_b": float(b.get("det_confidence") or 0.0),
            "segmentation_conf_a": float(a.get("seg_confidence") or 0.0),
            "segmentation_conf_b": float(b.get("seg_confidence") or 0.0),
            "keep_detection_id": "",
            "drop_detection_id": "",
            "action": "keep_both",
            "reason": reason,
            **feats,
        })

    groups = connected_duplicate_groups(edges)
    group_info: dict[int, dict[str, Any]] = {}
    group_by_member: dict[int, int] = {}
    drop_ids: set[int] = set()
    unsafe_groups = 0
    for group_id, members in enumerate(groups, start=1):
        unsafe = len(members) > 3
        if unsafe:
            unsafe_groups += 1
            keep_id = ""
            group_drop_ids: set[int] = set()
        else:
            keep_id = best_v7_keep_id(members, by_id)
            group_drop_ids = {det_id for det_id in members if det_id != keep_id}
            drop_ids.update(group_drop_ids)
        for det_id in members:
            group_by_member[det_id] = group_id
            group_info[det_id] = {"size": len(members), "unsafe": unsafe, "keep_id": keep_id, "drop_ids": group_drop_ids}

    for row in audit_rows:
        a_id = int(row["detection_id_a"])
        b_id = int(row["detection_id_b"])
        row["final_status_a"] = "dropped" if a_id in drop_ids else "kept"
        row["final_status_b"] = "dropped" if b_id in drop_ids else "kept"
        row["replacement_detection_id_a"] = ""
        row["replacement_detection_id_b"] = ""
        if not row["predicted_duplicate"]:
            continue
        info = group_info[a_id]
        row["duplicate_group_id"] = group_by_member[a_id]
        row["group_size"] = info["size"]
        row["unsafe_group"] = bool(info["unsafe"])
        if info["unsafe"]:
            row["action"] = "unsafe_group_no_action"
            continue
        row["keep_detection_id"] = info["keep_id"]
        if a_id in info["drop_ids"]:
            row["replacement_detection_id_a"] = info["keep_id"]
        if b_id in info["drop_ids"]:
            row["replacement_detection_id_b"] = info["keep_id"]
        pair_drop_ids = [det_id for det_id in [a_id, b_id] if det_id in info["drop_ids"]]
        row["drop_detection_id"] = ";".join(str(det_id) for det_id in pair_drop_ids)
        row["action"] = "auto_remove" if pair_drop_ids else "keep_both"

    kept_rows = [row for row in rows if int(row["detection_id"]) not in drop_ids]
    stats = {
        "v7_dedup_enabled": True,
        "v7_dedup_model": str(model_path),
        "v7_dedup_threshold": float(threshold),
        "v7_dedup_candidate_pairs": int(len(audit_rows)),
        "v7_dedup_predicted_duplicate_pairs": int(sum(1 for row in audit_rows if row["predicted_duplicate"])),
        "v7_dedup_duplicate_groups": int(len(groups)),
        "v7_dedup_unsafe_groups": int(unsafe_groups),
        "v7_dedup_removed_detections": int(len(drop_ids)),
        "v7_dedup_seconds": float(time.perf_counter() - start),
    }
    return kept_rows, audit_rows, stats


def v7_detection_status_rows(rows: list[dict[str, Any]], audit_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    status = {
        int(row["detection_id"]): {
            "detection_id": int(row["detection_id"]),
            "final_status": "kept",
            "replacement_detection_id": "",
            "duplicate_group_id": "",
            "group_size": "",
            "unsafe_group": False,
        }
        for row in rows
    }
    for row in audit_rows:
        if not row.get("predicted_duplicate"):
            continue
        group_id = row.get("duplicate_group_id", "")
        group_size = row.get("group_size", "")
        unsafe = bool(row.get("unsafe_group", False))
        keep_id = row.get("keep_detection_id", "")
        for key in ("detection_id_a", "detection_id_b"):
            det_id = int(row[key])
            status[det_id]["duplicate_group_id"] = group_id
            status[det_id]["group_size"] = group_size
            status[det_id]["unsafe_group"] = unsafe
        for drop_id in str(row.get("drop_detection_id", "")).split(";"):
            if not drop_id:
                continue
            det_id = int(drop_id)
            status[det_id]["final_status"] = "dropped"
            status[det_id]["replacement_detection_id"] = keep_id
    return [status[det_id] for det_id in sorted(status)]


def move_dropped_detection_artifacts(rows: list[dict[str, Any]], drop_ids: set[int], out_dir: Path) -> None:
    out_dir = resolve_path(out_dir)
    path_columns = [
        "crop_path",
        "prediction_crop_path",
        "mask_path",
        "envelope_prediction_crop_path",
        "envelope_mask_path",
        "envelope_comparison_path",
    ]
    dropped_root = out_dir / "dropped_v7"
    for row in rows:
        if int(row["detection_id"]) not in drop_ids:
            continue
        for col in path_columns:
            value = row.get(col)
            if not value:
                continue
            src = Path(value)
            if not src.is_absolute():
                src = resolve_path(src)
            if not src.exists() or not src.is_file():
                continue
            try:
                relative = src.relative_to(out_dir)
            except ValueError:
                relative = Path(src.name)
            dst = dropped_root / relative
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
            row[col] = str(dst.relative_to(REPO_ROOT))


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
    mask_dedup: bool,
    mask_dedup_iou: float,
    v7_dedup: bool = False,
    v7_dedup_model: Path = Path("models/duplicate_pair_classifier_v7_fragment_aware.joblib"),
    v7_dedup_threshold: float = 0.50,
    allow_overwrite: bool = False,
) -> dict[str, Any]:
    image_name = safe_name(image_path)
    out_dir = out_root / image_name
    if out_dir.exists() and any(out_dir.iterdir()) and not allow_overwrite:
        raise FileExistsError(f"Output for {image_name} already exists: {out_dir}. Use --allow-overwrite to replace files.")
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

    raw_rows_before_dedup = [dict(row) for row in rows]
    raw_overlay_before_dedup = full_overlay.copy()
    raw_segmentation_csv = out_dir / f"{image_name}_segmentation_results_before_v7.csv"
    raw_overlay_path = out_dir / f"{image_name}_trunk_seg_overlay_before_v7.jpg"

    detections_before_mask_dedup = len(rows)
    mask_dedup_audit_rows: list[dict[str, Any]] = []
    mask_dedup_stats = {
        "dedup_detections": 0,
        "dedup_total_possible_pairs": 0,
        "dedup_spatial_prefilter_pairs": 0,
        "dedup_mask_iou_calculations": 0,
        "dedup_duplicate_groups": 0,
        "dedup_removed_detections": 0,
        "dedup_seconds": 0.0,
    }
    if mask_dedup:
        rows, mask_dedup_audit_rows, mask_dedup_stats = deduplicate_by_mask_iou(rows, mask_dedup_iou)
        kept_ids = {int(r["detection_id"]) for r in rows}
        contact_items = [item for item in contact_items if item[0] in kept_ids]
        full_overlay = draw_full_overlay(image_bgr, rows, use_envelope_contour)
    detections_after_mask_dedup = len(rows)

    detections_before_v7_dedup = len(rows)
    v7_dedup_audit_rows: list[dict[str, Any]] = []
    v7_dedup_stats = {
        "v7_dedup_enabled": False,
        "v7_dedup_model": str(v7_dedup_model),
        "v7_dedup_threshold": float(v7_dedup_threshold),
        "v7_dedup_candidate_pairs": 0,
        "v7_dedup_predicted_duplicate_pairs": 0,
        "v7_dedup_duplicate_groups": 0,
        "v7_dedup_unsafe_groups": 0,
        "v7_dedup_removed_detections": 0,
        "v7_dedup_seconds": 0.0,
    }
    if v7_dedup:
        rows, v7_dedup_audit_rows, v7_dedup_stats = deduplicate_by_v7_classifier(rows, v7_dedup_model, v7_dedup_threshold)
        kept_ids = {int(r["detection_id"]) for r in rows}
        drop_ids = {int(r["detection_id"]) for r in raw_rows_before_dedup} - kept_ids
        move_dropped_detection_artifacts(raw_rows_before_dedup, drop_ids, out_dir)
        contact_items = [item for item in contact_items if item[0] in kept_ids]
        full_overlay = draw_full_overlay(image_bgr, rows, use_envelope_contour)
        pd.DataFrame(raw_rows_before_dedup).to_csv(raw_segmentation_csv, index=False)
        cv2.imwrite(str(raw_overlay_path), raw_overlay_before_dedup, [cv2.IMWRITE_JPEG_QUALITY, 92])

    overlay_path = out_dir / f"{image_name}_trunk_seg_overlay.jpg"
    cv2.imwrite(str(overlay_path), full_overlay, [cv2.IMWRITE_JPEG_QUALITY, 92])
    results_csv = out_dir / f"{image_name}_segmentation_results.csv"
    pd.DataFrame(rows).to_csv(results_csv, index=False)
    mask_dedup_csv = out_dir / f"{image_name}_mask_dedup.csv"
    if mask_dedup:
        pd.DataFrame(mask_dedup_audit_rows, columns=[
            "detection_id_a",
            "detection_id_b",
            "mask_iou",
            "detector_conf_a",
            "detector_conf_b",
            "segmentation_conf_a",
            "segmentation_conf_b",
            "duplicate_group_id",
            "keep_detection_id",
            "drop_detection_id",
            "keep_reason",
            "threshold",
        ]).to_csv(mask_dedup_csv, index=False)
    v7_dedup_csv = out_dir / f"{image_name}_v7_dedup.csv"
    v7_status_csv = out_dir / f"{image_name}_v7_detection_status.csv"
    if v7_dedup:
        v7_audit_columns = [
            "detection_id_a",
            "detection_id_b",
            "duplicate_probability",
            "predicted_duplicate",
            "duplicate_group_id",
            "group_size",
            "unsafe_group",
            "detector_conf_a",
            "detector_conf_b",
            "segmentation_conf_a",
            "segmentation_conf_b",
            "keep_detection_id",
            "drop_detection_id",
            "action",
            "reason",
            "final_status_a",
            "final_status_b",
            "replacement_detection_id_a",
            "replacement_detection_id_b",
            *V7_FEATURES,
        ]
        pd.DataFrame(v7_dedup_audit_rows, columns=v7_audit_columns).to_csv(v7_dedup_csv, index=False)
        pd.DataFrame(v7_detection_status_rows(raw_rows_before_dedup, v7_dedup_audit_rows)).to_csv(v7_status_csv, index=False)
    contact_path = dirs["contact_sheets"] / f"{image_name}_contact_sheet.jpg"
    write_contact_sheet(contact_items, contact_path)

    summary = {
        "image": str(image_path),
        "output_dir": str(out_dir),
        "detections": int(len(rows)),
        "detections_before_mask_dedup": int(detections_before_mask_dedup),
        "mask_dedup_enabled": bool(mask_dedup),
        "mask_dedup_iou": float(mask_dedup_iou),
        "mask_dedup_pairs": int(len(mask_dedup_audit_rows)),
        "mask_dedup_removed": int(detections_before_mask_dedup - detections_after_mask_dedup),
        **mask_dedup_stats,
        "detections_before_v7_dedup": int(detections_before_v7_dedup),
        "raw_segmentation_csv_before_v7": str(raw_segmentation_csv) if v7_dedup else "",
        "raw_overlay_before_v7": str(raw_overlay_path) if v7_dedup else "",
        "v7_dedup_csv": str(v7_dedup_csv) if v7_dedup else "",
        "v7_detection_status_csv": str(v7_status_csv) if v7_dedup else "",
        "v7_dedup_removed": int(detections_before_v7_dedup - len(rows)),
        "detections_after_v7_dedup": int(len(rows)),
        **v7_dedup_stats,
        "masks": int(sum(1 for r in rows if r.get("has_mask"))),
        "no_mask": int(sum(1 for r in rows if not r.get("has_mask"))),
        "envelope_masks": int(sum(1 for r in rows if r.get("envelope_mask_path"))),
        "envelope_warnings": int(sum(1 for r in rows if bool(r.get("envelope_warning")))),
        "fallbacks": int(sum(1 for r in rows if bool(r.get("fallback_used")))),
        "overlay_path": str(overlay_path),
        "results_csv": str(results_csv),
        "mask_dedup_csv": str(mask_dedup_csv) if mask_dedup else "",
        "contact_sheet": str(contact_path),
        "envelope_comparison_dir": str(dirs["envelope_comparison"]),
    }
    if mask_dedup:
        print(json.dumps({"image": image_name, "mask_dedup_debug": mask_dedup_stats}, indent=2))
    if v7_dedup:
        print(json.dumps({"image": image_name, "v7_dedup_debug": v7_dedup_stats}, indent=2))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test trunk segmentation generalization on new full images")
    parser.add_argument("--images-dir", type=Path, default=Path("data/slike_test_nove"))
    parser.add_argument("--image-stems", nargs="*", default=None, help="Optional list of image stems to process from --images-dir")
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
    parser.add_argument("--mask-dedup", action="store_true")
    parser.add_argument("--mask-dedup-iou", type=float, default=0.30)
    parser.add_argument("--v7-dedup", dest="v7_dedup", action="store_true", help="Enable validated v7 duplicate cleanup post-processing")
    parser.add_argument("--no-v7-dedup", dest="v7_dedup", action="store_false", help="Disable v7 duplicate cleanup post-processing")
    parser.add_argument("--v7-dedup-model", type=Path, default=Path("models/duplicate_pair_classifier_v7_fragment_aware.joblib"))
    parser.add_argument("--v7-dedup-threshold", type=float, default=0.50)
    parser.add_argument("--allow-overwrite", action="store_true", help="Allow writing into existing per-source output directories")
    parser.set_defaults(v7_dedup=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mask_dedup and args.v7_dedup:
        raise ValueError("--mask-dedup and --v7-dedup are mutually exclusive; enable only one dedup mode")
    image_exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    images = sorted(p for p in args.images_dir.iterdir() if p.is_file() and p.suffix.lower() in image_exts)
    if args.image_stems:
        wanted = set(args.image_stems)
        images = [p for p in images if p.stem in wanted]
        missing_stems = sorted(wanted - {p.stem for p in images})
        if missing_stems:
            raise SystemExit(f"Requested --image-stems not found in {args.images_dir}: {missing_stems}")
    if not images:
        raise SystemExit(f"No image files found in {args.images_dir}")
    model_paths = [args.detector_model, args.seg_model]
    if args.v7_dedup:
        model_paths.append(args.v7_dedup_model)
    missing = [p for p in model_paths if not p.exists()]
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
            args.mask_dedup,
            args.mask_dedup_iou,
            args.v7_dedup,
            args.v7_dedup_model,
            args.v7_dedup_threshold,
            args.allow_overwrite,
        ))

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps({"images": summaries}, indent=2), encoding="utf-8")
    print(json.dumps({"summary_path": str(summary_path), "images": summaries}, indent=2))


if __name__ == "__main__":
    main()
