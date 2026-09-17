#!/usr/bin/env python3
"""Train/evaluate fragment-aware duplicate classifier v7 as an isolated experiment."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import cv2
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "data/duplicate_classifier_v7_fragment_aware"
MODEL_PATH = REPO_ROOT / "models/duplicate_pair_classifier_v7_fragment_aware.joblib"
V6_TRAINING = REPO_ROOT / "data/duplicate_classifier_v6/training_pairs.csv"
UNSEEN_REVIEW = REPO_ROOT / "data/duplicate_classifier_final_v1/unseen_v1/unseen_manual_review.csv"
UNSEEN_AUDIT_ROOT = REPO_ROOT / "data/duplicate_classifier_final_v1/unseen_v1/cleanup"
V5_MODEL = REPO_ROOT / "models/duplicate_pair_classifier_v5.joblib"
FINAL_MODEL = REPO_ROOT / "models/duplicate_pair_classifier_final_v1.joblib"

SEG_OUTPUT_ROOTS = [
    REPO_ROOT / "data/duplicate_classifier_final_v1/unseen_v1/raw",
    REPO_ROOT / "data/web_outputs/active_learning_v1_raw",
    REPO_ROOT / "data/web_outputs/trunk_seg_generalization_test_v5_new",
    REPO_ROOT / "data/web_outputs/trunk_seg_v5_regression_no_dedup",
    REPO_ROOT / "data/web_outputs/trunk_seg_v5_regression_mask_dedup",
]

BASE_FEATURES = [
    "mask_iou", "bbox_iou", "mask_area_ratio", "centroid_distance",
    "detector_conf_min", "detector_conf_max", "detector_conf_abs_diff", "detector_conf_ratio",
    "segmentation_conf_min", "segmentation_conf_max", "segmentation_conf_abs_diff", "segmentation_conf_ratio",
    "mask_area_min", "mask_area_max", "mask_area_abs_diff",
]

FRAGMENT_FEATURES = [
    "normalized_centroid_distance", "bbox_area_ratio", "small_mask_in_large_bbox", "small_bbox_in_large_bbox",
    "expanded_overlap", "fragment_score", "bbox_gap_px", "mask_gap_px",
    "small_bbox_inside_large_bbox_fraction", "small_mask_inside_large_bbox_fraction", "small_mask_inside_dilated_large_mask_fraction",
    "width_ratio", "height_ratio", "aspect_ratio_difference", "major_axis_angle_difference",
    "centroid_distance_over_large_bbox_diagonal", "contour_perimeter_ratio", "solidity_difference",
    "eccentricity_difference", "orientation_difference", "convex_hull_area_ratio",
    "masks_connect_with_small_dilation", "min_dilation_radius_to_connect_masks",
]

FEATURES = BASE_FEATURES + FRAGMENT_FEATURES
THRESHOLDS = [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.97, 0.99]


def normalize_label(value: Any) -> str:
    label = str(value).strip().upper()
    if label in {"D", "DUPLICATE", "DUPLICATE_SPLIT", "DUPLICATE_FRAGMENT"}:
        return "D"
    if label in {"S", "SEPARATE"}:
        return "S"
    return ""


def pair_key(row: pd.Series | dict[str, Any]) -> tuple[str, int, int]:
    a = int(float(row["detection_id_a"]))
    b = int(float(row["detection_id_b"]))
    if a > b:
        a, b = b, a
    return str(row["source_image"]), a, b


def safe_float(value: Any, default: float = np.nan) -> float:
    try:
        if value is None or str(value).strip() == "":
            return default
        return float(value)
    except Exception:
        return default


def ratio_minmax(a: float, b: float) -> float:
    return float(min(a, b) / max(a, b)) if max(a, b) > 0 else 0.0


def bbox_area(bbox: tuple[float, float, float, float]) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def bbox_intersection(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def bbox_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    inter = bbox_intersection(a, b)
    union = bbox_area(a) + bbox_area(b) - inter
    return inter / union if union > 0 else 0.0


def bbox_gap(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    dx = max(a[0] - b[2], b[0] - a[2], 0.0)
    dy = max(a[1] - b[3], b[1] - a[3], 0.0)
    return float(math.hypot(dx, dy))


def angle_diff(a: float, b: float) -> float:
    if np.isnan(a) or np.isnan(b):
        return np.nan
    diff = abs((a - b + 90.0) % 180.0 - 90.0)
    return float(diff)


def contour_props(mask: np.ndarray) -> dict[str, float]:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return {"perimeter": np.nan, "solidity": np.nan, "eccentricity": np.nan, "orientation": np.nan, "hull_area": np.nan}
    contour = max(contours, key=cv2.contourArea)
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


def load_detection(source: str, det_id: int) -> dict[str, Any] | None:
    for root in SEG_OUTPUT_ROOTS:
        csv_path = root / source / f"{source}_segmentation_results.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path).fillna("")
        df["detection_id"] = df["detection_id"].astype(int)
        match = df[df["detection_id"].eq(det_id)]
        if match.empty:
            continue
        row = match.iloc[0].to_dict()
        bbox = (safe_float(row["bbox_x1"]), safe_float(row["bbox_y1"]), safe_float(row["bbox_x2"]), safe_float(row["bbox_y2"]))
        mask_path = Path(str(row.get("mask_path", "")))
        if not mask_path.is_absolute():
            mask_path = REPO_ROOT / mask_path
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return {"row": row, "bbox": bbox, "mask": None}
        return {"row": row, "bbox": bbox, "mask": mask > 0}
    return None


def mask_centroid(mask: np.ndarray | None, bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    if mask is not None:
        ys, xs = np.nonzero(mask)
        if len(xs):
            return float(xs.mean() + math.floor(bbox[0])), float(ys.mean() + math.floor(bbox[1]))
    return (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0


def paste_mask(canvas: np.ndarray, mask: np.ndarray, bbox: tuple[float, float, float, float], origin_x: int, origin_y: int) -> None:
    x = int(math.floor(bbox[0])) - origin_x
    y = int(math.floor(bbox[1])) - origin_y
    h, w = mask.shape[:2]
    canvas[y:y + h, x:x + w] |= mask


def pair_mask_features(a: dict[str, Any], b: dict[str, Any]) -> dict[str, float]:
    bbox_a, bbox_b = a["bbox"], b["bbox"]
    mask_a, mask_b = a.get("mask"), b.get("mask")
    area_a = int(mask_a.sum()) if mask_a is not None else bbox_area(bbox_a)
    area_b = int(mask_b.sum()) if mask_b is not None else bbox_area(bbox_b)
    small, large = (a, b) if area_a <= area_b else (b, a)
    sb, lb = small["bbox"], large["bbox"]
    sm, lm = small.get("mask"), large.get("mask")
    ca, cb = mask_centroid(mask_a, bbox_a), mask_centroid(mask_b, bbox_b)
    large_diag = max(1.0, math.hypot(lb[2] - lb[0], lb[3] - lb[1]))
    bbox_area_a, bbox_area_b = bbox_area(bbox_a), bbox_area(bbox_b)
    small_bbox_area, large_bbox_area = bbox_area(sb), bbox_area(lb)
    props_a = contour_props(mask_a) if mask_a is not None else contour_props(np.zeros((1, 1), dtype=np.uint8))
    props_b = contour_props(mask_b) if mask_b is not None else contour_props(np.zeros((1, 1), dtype=np.uint8))
    out = {
        "bbox_iou": bbox_iou(bbox_a, bbox_b),
        "bbox_area_ratio": ratio_minmax(bbox_area_a, bbox_area_b),
        "small_bbox_in_large_bbox": bbox_intersection(sb, lb) / small_bbox_area if small_bbox_area > 0 else 0.0,
        "small_bbox_inside_large_bbox_fraction": bbox_intersection(sb, lb) / small_bbox_area if small_bbox_area > 0 else 0.0,
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
    }
    if sm is None or lm is None:
        return out
    x1 = int(math.floor(min(bbox_a[0], bbox_b[0])))
    y1 = int(math.floor(min(bbox_a[1], bbox_b[1])))
    x2 = int(math.ceil(max(bbox_a[2], bbox_b[2])))
    y2 = int(math.ceil(max(bbox_a[3], bbox_b[3])))
    canvas_small = np.zeros((max(1, y2 - y1), max(1, x2 - x1)), dtype=bool)
    canvas_large = np.zeros_like(canvas_small)
    paste_mask(canvas_small, sm, sb, x1, y1)
    paste_mask(canvas_large, lm, lb, x1, y1)
    inter = int((canvas_small & canvas_large).sum())
    union = int((canvas_small | canvas_large).sum())
    small_area = max(1, int(canvas_small.sum()))
    out["mask_iou"] = inter / union if union else 0.0
    out["mask_area_ratio"] = ratio_minmax(int(canvas_small.sum()), int(canvas_large.sum()))
    out["mask_area_min"] = min(int(canvas_small.sum()), int(canvas_large.sum()))
    out["mask_area_max"] = max(int(canvas_small.sum()), int(canvas_large.sum()))
    out["mask_area_abs_diff"] = abs(int(canvas_small.sum()) - int(canvas_large.sum()))
    sx1, sy1 = int(math.floor(sb[0])) - x1, int(math.floor(sb[1])) - y1
    ys, xs = np.nonzero(canvas_small)
    inside_bbox = ((xs + x1 >= lb[0]) & (xs + x1 <= lb[2]) & (ys + y1 >= lb[1]) & (ys + y1 <= lb[3])).sum()
    out["small_mask_in_large_bbox"] = float(inside_bbox / small_area)
    out["small_mask_inside_large_bbox_fraction"] = float(inside_bbox / small_area)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    dilated_large = cv2.dilate(canvas_large.astype(np.uint8), kernel, iterations=1) > 0
    out["small_mask_inside_dilated_large_mask_fraction"] = float((canvas_small & dilated_large).sum() / small_area)
    dist = cv2.distanceTransform((~canvas_large).astype(np.uint8), cv2.DIST_L2, 3)
    out["mask_gap_px"] = 0.0 if inter > 0 else float(dist[canvas_small].min()) if canvas_small.any() else np.nan
    out["masks_connect_with_small_dilation"] = bool(out["mask_gap_px"] <= 10.0)
    out["min_dilation_radius_to_connect_masks"] = out["mask_gap_px"]
    if "fragment_score" not in out or np.isnan(safe_float(out.get("fragment_score"))):
        out["fragment_score"] = 100.0 * out["small_mask_inside_large_bbox_fraction"] * (1.0 - out["mask_iou"])
    return out


def enrich_row(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    source, a_id, b_id = pair_key(out)
    a = load_detection(source, a_id)
    b = load_detection(source, b_id)
    if a is not None and b is not None:
        out.update({k: v for k, v in pair_mask_features(a, b).items() if k not in out or str(out.get(k, "")).strip() == "" or pd.isna(out.get(k))})
        ra, rb = a["row"], b["row"]
        det_a, det_b = safe_float(ra.get("det_confidence"), 0.0), safe_float(rb.get("det_confidence"), 0.0)
        seg_a, seg_b = safe_float(ra.get("seg_confidence"), 0.0), safe_float(rb.get("seg_confidence"), 0.0)
        out.update({
            "detector_conf_min": min(det_a, det_b), "detector_conf_max": max(det_a, det_b),
            "detector_conf_abs_diff": abs(det_a - det_b), "detector_conf_ratio": ratio_minmax(det_a, det_b),
            "segmentation_conf_min": min(seg_a, seg_b), "segmentation_conf_max": max(seg_a, seg_b),
            "segmentation_conf_abs_diff": abs(seg_a - seg_b), "segmentation_conf_ratio": ratio_minmax(seg_a, seg_b),
        })
    return out


def standardize_existing() -> pd.DataFrame:
    df = pd.read_csv(V6_TRAINING).fillna("")
    rows = []
    for _, row in df.iterrows():
        label = normalize_label(row.get("label", ""))
        if label not in {"D", "S"}:
            continue
        item = row.to_dict()
        source, a, b = pair_key(item)
        item.update({"source_image": source, "detection_id_a": a, "detection_id_b": b, "label": label, "hard_fragment_D": False, "unseen_v1_caught_D": False, "source_review_csv": str(V6_TRAINING.relative_to(REPO_ROOT))})
        rows.append(item)
    return pd.DataFrame(rows)


def load_unseen_audit_flags() -> dict[tuple[str, int, int], bool]:
    flags = {}
    for path in sorted(UNSEEN_AUDIT_ROOT.glob("*/*_classifier_audit.csv")):
        audit = pd.read_csv(path).fillna("")
        for _, row in audit.iterrows():
            key = pair_key(row)
            pred = str(row.get("predicted_duplicate", "")).strip().lower() in {"true", "1", "yes"}
            action = str(row.get("action", "")).strip() == "auto_remove"
            flags[key] = bool(pred and action)
    return flags


def standardize_unseen() -> pd.DataFrame:
    review = pd.read_csv(UNSEEN_REVIEW).fillna("")
    audit_flags = load_unseen_audit_flags()
    rows = []
    for _, row in review.iterrows():
        label = normalize_label(row.get("review_status", ""))
        if label not in {"D", "S"}:
            continue
        item = row.to_dict()
        source, a, b = pair_key(item)
        key = (source, a, b)
        caught = bool(audit_flags.get(key, False))
        item.update({"source_image": source, "detection_id_a": a, "detection_id_b": b, "label": label, "hard_fragment_D": bool(label == "D" and not caught), "unseen_v1_caught_D": bool(label == "D" and caught), "source_review_csv": str(UNSEEN_REVIEW.relative_to(REPO_ROOT))})
        rows.append(item)
    return pd.DataFrame(rows)


def build_dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    combined = pd.concat([standardize_existing(), standardize_unseen()], ignore_index=True, sort=False)
    enriched = pd.DataFrame([enrich_row(row) for row in combined.to_dict("records")])
    for feature in FEATURES:
        if feature in enriched.columns:
            enriched[feature] = pd.to_numeric(enriched[feature], errors="coerce")
        else:
            enriched[feature] = np.nan
    buckets: dict[tuple[str, int, int], list[dict[str, Any]]] = {}
    for row in enriched.to_dict("records"):
        buckets.setdefault(pair_key(row), []).append(row)
    resolved = []
    conflicts = []
    for key, items in buckets.items():
        labels = sorted({item["label"] for item in items})
        if len(labels) > 1:
            for item in items:
                conflicts.append({**item, "dedupe_source_image": key[0], "dedupe_detection_id_a": key[1], "dedupe_detection_id_b": key[2], "conflict_labels": ";".join(labels)})
            continue
        keep = items[0]
        keep["hard_fragment_D"] = any(bool(item.get("hard_fragment_D")) for item in items)
        keep["unseen_v1_caught_D"] = any(bool(item.get("unseen_v1_caught_D")) for item in items)
        keep["merged_sources"] = ";".join(sorted({str(item.get("source_review_csv", "")) for item in items}))
        resolved.append(keep)
    return pd.DataFrame(resolved), pd.DataFrame(conflicts)


def make_splits(x: pd.DataFrame, y: np.ndarray, groups: np.ndarray) -> tuple[str, list[tuple[np.ndarray, np.ndarray]]]:
    n_splits = min(5, len(np.unique(groups)))
    try:
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=20260910)
        return "StratifiedGroupKFold", list(splitter.split(x, y, groups))
    except Exception:
        splitter = GroupKFold(n_splits=n_splits)
        return "GroupKFold", list(splitter.split(x, y, groups))


def metrics_at(y_true: np.ndarray, y_prob: np.ndarray, threshold: float, hard: np.ndarray, easy: np.ndarray) -> dict[str, float | int]:
    pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    d_precision, d_recall, d_f1, _ = precision_recall_fscore_support(y_true, pred, pos_label=1, average="binary", zero_division=0)
    s_precision, s_recall, s_f1, _ = precision_recall_fscore_support(1 - y_true, 1 - pred, pos_label=1, average="binary", zero_division=0)
    return {
        "D_precision": float(d_precision), "D_recall": float(d_recall), "D_F1": float(d_f1),
        "S_precision": float(s_precision), "S_recall": float(s_recall), "S_F1": float(s_f1),
        "S_false_positive_count": int(fp), "S_false_positive_rate": float(fp / max(1, int((y_true == 0).sum()))),
        "confusion_TN": int(tn), "confusion_FP_S_to_D": int(fp), "confusion_FN_D_to_S": int(fn), "confusion_TP": int(tp),
        "hard_fragment_D_recall": float(((pred == 1) & hard).sum() / max(1, int(hard.sum()))),
        "easy_known_D_recall": float(((pred == 1) & easy).sum() / max(1, int(easy.sum()))),
        "hard_fragment_D_caught": int(((pred == 1) & hard).sum()),
        "hard_fragment_D_total": int(hard.sum()),
        "easy_known_D_caught": int(((pred == 1) & easy).sum()),
        "easy_known_D_total": int(easy.sum()),
    }


def sample_weights(labels: np.ndarray, hard: np.ndarray, mode: str) -> np.ndarray | None:
    if mode == "natural":
        return None
    weights = np.ones(len(labels), dtype=float)
    if mode in {"balanced", "balanced_hard3"}:
        pos = max(1, int(labels.sum()))
        neg = max(1, int((labels == 0).sum()))
        weights[labels == 1] = len(labels) / (2.0 * pos)
        weights[labels == 0] = len(labels) / (2.0 * neg)
    if mode == "hard3":
        weights[hard] = 3.0
    if mode == "balanced_hard3":
        weights[hard] *= 3.0
    return weights


def fit_with_optional_weight(model: Any, x: pd.DataFrame, y: np.ndarray, weights: np.ndarray | None) -> None:
    if weights is None:
        model.fit(x, y)
        return
    try:
        model.fit(x, y, clf__sample_weight=weights)
    except Exception:
        try:
            model.fit(x, y, sample_weight=weights)
        except Exception:
            model.fit(x, y)


def evaluate_model(name: str, estimator: Any, weight_mode: str, x: pd.DataFrame, y: np.ndarray, hard: np.ndarray, splits: list[tuple[np.ndarray, np.ndarray]], used: pd.DataFrame) -> tuple[np.ndarray, list[dict[str, Any]]]:
    probs = np.zeros(len(used), dtype=float)
    rows = []
    for fold, (train_idx, val_idx) in enumerate(splits, start=1):
        model = clone(estimator)
        weights = sample_weights(y[train_idx], hard[train_idx], weight_mode)
        fit_with_optional_weight(model, x.iloc[train_idx], y[train_idx], weights)
        probs[val_idx] = model.predict_proba(x.iloc[val_idx])[:, 1]
        for idx in val_idx:
            rows.append({
                "model": name, "weight_mode": weight_mode, "fold": fold,
                "source_image": used.iloc[idx]["source_image"], "detection_id_a": int(used.iloc[idx]["detection_id_a"]), "detection_id_b": int(used.iloc[idx]["detection_id_b"]),
                "label": used.iloc[idx]["label"], "hard_fragment_D": bool(hard[idx]), "prob_D": float(probs[idx]),
            })
    return probs, rows


def choose_best(threshold_df: pd.DataFrame, model_filter: pd.Series) -> pd.Series:
    sub = threshold_df[model_filter].copy()
    zero = sub[sub["S_false_positive_count"].eq(0)]
    if not zero.empty:
        sub = zero
    return sub.sort_values(["S_false_positive_count", "hard_fragment_D_recall", "D_recall", "D_F1", "threshold"], ascending=[True, False, False, False, False]).iloc[0]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    resolved, conflicts = build_dataset()
    resolved.to_csv(OUT_DIR / "training_pairs.csv", index=False)
    conflicts.to_csv(OUT_DIR / "conflicts.csv", index=False)
    used = resolved.dropna(subset=["label", "source_image", "detection_id_a", "detection_id_b"]).copy()
    for feature in FEATURES:
        used[feature] = pd.to_numeric(used[feature], errors="coerce")
    used.to_csv(OUT_DIR / "feature_table.csv", index=False)

    x = used[FEATURES]
    y = used["label"].eq("D").astype(int).to_numpy()
    hard = used["hard_fragment_D"].astype(bool).to_numpy()
    easy = (used["label"].eq("D") & ~used["hard_fragment_D"].astype(bool)).to_numpy()
    groups = used["source_image"].astype(str).to_numpy()
    cv_name, splits = make_splits(x, y, groups)

    base_models = {
        "LogisticRegression": Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=5000, random_state=20260910))]),
        "RandomForestClassifier": Pipeline([("impute", SimpleImputer(strategy="median")), ("clf", RandomForestClassifier(n_estimators=700, max_depth=8, min_samples_leaf=3, random_state=20260910, n_jobs=-1))]),
        "HistGradientBoostingClassifier": Pipeline([("impute", SimpleImputer(strategy="median")), ("clf", HistGradientBoostingClassifier(max_iter=350, learning_rate=0.035, max_leaf_nodes=12, l2_regularization=0.05, random_state=20260910))]),
        "GradientBoostingClassifier": Pipeline([("impute", SimpleImputer(strategy="median")), ("clf", GradientBoostingClassifier(n_estimators=350, learning_rate=0.035, max_depth=2, random_state=20260910))]),
        "ExtraTreesClassifier": Pipeline([("impute", SimpleImputer(strategy="median")), ("clf", ExtraTreesClassifier(n_estimators=700, max_depth=10, min_samples_leaf=2, random_state=20260910, n_jobs=-1))]),
    }
    variants = []
    for name, model in base_models.items():
        variants.append((name, "natural", model))
        variants.append((name, "hard3", model))
        if name in {"LogisticRegression", "RandomForestClassifier", "ExtraTreesClassifier"}:
            balanced = clone(model)
            balanced.set_params(clf__class_weight="balanced")
            variants.append((name, "class_weight_balanced", balanced))
        variants.append((name, "balanced", model))
        variants.append((name, "balanced_hard3", model))

    pred_rows = []
    threshold_rows = []
    probs_by_variant = {}
    for name, weight_mode, estimator in variants:
        variant = f"v7_{name}_{weight_mode}"
        probs, rows = evaluate_model(variant, estimator, weight_mode if weight_mode != "class_weight_balanced" else "natural", x, y, hard, splits, used)
        probs_by_variant[variant] = probs
        pred_rows.extend(rows)
        for threshold in THRESHOLDS:
            threshold_rows.append({"model": variant, "threshold": threshold, **metrics_at(y, probs, threshold, hard, easy)})

    baseline_rows = []
    for label, model_path in [("v5_fixed_classifier", V5_MODEL), ("final_v1_fixed_classifier", FINAL_MODEL)]:
        payload = joblib.load(model_path)
        feats = payload["features"]
        probs = payload["model"].predict_proba(used[feats])[:, 1]
        for threshold in THRESHOLDS:
            row = {"model": label, "threshold": threshold, **metrics_at(y, probs, threshold, hard, easy)}
            threshold_rows.append(row)
            baseline_rows.append(row)

    threshold_df = pd.DataFrame(threshold_rows)
    threshold_df.to_csv(OUT_DIR / "threshold_analysis.csv", index=False)
    pd.DataFrame(pred_rows).to_csv(OUT_DIR / "cross_validation_predictions.csv", index=False)

    comparison = []
    for model in sorted(threshold_df["model"].unique()):
        best = choose_best(threshold_df, threshold_df["model"].eq(model)).to_dict()
        comparison.append(best)
    comparison_df = pd.DataFrame(comparison)
    comparison_df.to_csv(OUT_DIR / "model_comparison.csv", index=False)

    hard_df = threshold_df[["model", "threshold", "hard_fragment_D_recall", "hard_fragment_D_caught", "hard_fragment_D_total", "easy_known_D_recall", "S_false_positive_count", "D_recall"]].copy()
    hard_df.to_csv(OUT_DIR / "hard_fragment_analysis.csv", index=False)

    v7_best = choose_best(threshold_df, threshold_df["model"].str.startswith("v7_"))
    v5_best = choose_best(threshold_df, threshold_df["model"].eq("v5_fixed_classifier"))
    final_best = choose_best(threshold_df, threshold_df["model"].eq("final_v1_fixed_classifier"))
    baseline_best_hard = max(float(v5_best["hard_fragment_D_recall"]), float(final_best["hard_fragment_D_recall"]))
    baseline_best_recall = max(float(v5_best["D_recall"]), float(final_best["D_recall"]))
    v7_better = bool(
        int(v7_best["S_false_positive_count"]) <= min(int(v5_best["S_false_positive_count"]), int(final_best["S_false_positive_count"]))
        and float(v7_best["hard_fragment_D_recall"]) > baseline_best_hard
        and float(v7_best["D_recall"]) >= baseline_best_recall
    )

    best_model_name = str(v7_best["model"])
    _, base_name, weight_mode = best_model_name.split("_", 2)
    estimator = clone(base_models[base_name])
    final_weights = sample_weights(y, hard, weight_mode if weight_mode != "class_weight_balanced" else "natural")
    if weight_mode == "class_weight_balanced" and base_name in {"LogisticRegression", "RandomForestClassifier", "ExtraTreesClassifier"}:
        estimator.set_params(clf__class_weight="balanced")
    fit_with_optional_weight(estimator, x, y, final_weights)
    if v7_better:
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": estimator, "features": FEATURES, "label_positive": "D", "recommended_threshold": float(v7_best["threshold"]), "training_dataset": str(OUT_DIR / "training_pairs.csv"), "experiment": "v7_fragment_aware"}, MODEL_PATH)

    try:
        perm = permutation_importance(estimator, x, y, n_repeats=12, random_state=20260910, scoring="f1", n_jobs=-1)
        importance = pd.DataFrame({"feature": FEATURES, "importance_mean": perm.importances_mean, "importance_std": perm.importances_std}).sort_values("importance_mean", ascending=False)
    except Exception:
        clf = estimator.named_steps.get("clf") if hasattr(estimator, "named_steps") else estimator
        vals = getattr(clf, "feature_importances_", np.full(len(FEATURES), np.nan))
        importance = pd.DataFrame({"feature": FEATURES, "importance_mean": vals, "importance_std": np.nan}).sort_values("importance_mean", ascending=False)
    importance.to_csv(OUT_DIR / "feature_importance.csv", index=False)

    counts = used["label"].value_counts().to_dict()
    summary = {
        "training_pairs_csv": str((OUT_DIR / "training_pairs.csv").relative_to(REPO_ROOT)),
        "feature_table_csv": str((OUT_DIR / "feature_table.csv").relative_to(REPO_ROOT)),
        "D_examples": int(counts.get("D", 0)),
        "S_examples": int(counts.get("S", 0)),
        "total_examples": int(len(used)),
        "hard_fragment_D": int(hard.sum()),
        "unseen_v1_caught_D": int(used["unseen_v1_caught_D"].astype(bool).sum()),
        "conflicts_removed": int(len(conflicts)),
        "cv": cv_name,
        "source_groups": int(len(np.unique(groups))),
        "features": FEATURES,
        "best_v7": v7_best.to_dict(),
        "v5_best": v5_best.to_dict(),
        "final_v1_best": final_best.to_dict(),
        "v7_objectively_better_for_fragment_duplicates": v7_better,
        "model_saved": str(MODEL_PATH.relative_to(REPO_ROOT)) if v7_better else "not_saved_v7_not_objectively_better",
        "main_pipeline_changed": False,
        "new_unseen_test_prepared": False,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
