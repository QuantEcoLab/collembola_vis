"""Exp03 masks measured only by curved centerline and integrated local sections."""

import csv
import hashlib
import json
import math
import shutil
import uuid
from pathlib import Path

from backend.config import settings
from backend.jobs.models import Job, JobType
from backend.services.centerline_measurement import FORMULA, METHOD, measure_mask, summarize

REPO = Path(__file__).resolve().parents[2]
EXP03_SHA256 = "3fe97dd43f387815f8ee947eb5d02a6b0a7afadc1be59942a6b4a1f49f9aa0da"
MODEL_NAME = "exp03_no_os"


def annotation_snapshot(image_id):
    """Read saved human corrections without writing annotations or changing IDs."""
    path = settings.annotations_dir / f"{image_id}.json"
    if not path.is_file():
        return None, None
    data = json.loads(path.read_text())
    if data.get("image_id") != image_id:
        raise ValueError("Spremljene oznake pripadaju drugoj fotografiji")
    digest = hashlib.sha256(json.dumps({k: data.get(k) for k in ("image_id", "image_filename", "source_job_id", "boxes")},
                                      sort_keys=True, allow_nan=False).encode()).hexdigest()
    return data, digest


def find_saved_source(image_id):
    """Find a completed same-image exp03 mask job, including archived calculations."""
    candidates = []
    for path in settings.outputs_dir.glob("*/result.json"):
        try:
            data = json.loads(path.read_text())
            result = data.get("result") or {}
            if (data.get("status") == "completed" and result.get("image_id") == image_id
                    and result.get("model_name") == MODEL_NAME and result.get("checkpoint_sha256") == EXP03_SHA256
                    and Path(result.get("mask_records_path", "")).is_file()):
                candidates.append(data)
        except (OSError, ValueError):
            continue
    return max(candidates, key=lambda d: d.get("completed_at") or d.get("created_at") or "") if candidates else None


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_checkpoint() -> Path:
    path = settings.exp03_checkpoint
    path = path.resolve() if path.is_absolute() else (REPO / path).resolve()
    if not path.is_file():
        raise RuntimeError(f"Eksperimentalni exp03_no_os checkpoint nije pronađen: {path}")
    if file_sha256(path) != EXP03_SHA256:
        raise RuntimeError("SHA-256 exp03_no_os checkpointa ne odgovara fiksiranom modelu; obrada je zaustavljena.")
    return path


def valid_calibration(value):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError("Kalibracija mora biti konačan pozitivan broj u µm/px.")
    return float(value)


def _legacy_pipeline():
    from scripts.ruler_v7 import run_final_pipeline_comparison
    return run_final_pipeline_comparison


def measure_final_masks(records, um_per_pixel, *, geometries=None, progress_callback=None):
    """Measure retained masks using the approved method; exclude unreliable totals."""
    import cv2
    calibration = valid_calibration(um_per_pixel)
    measurements = []
    for index, record in enumerate(records):
        if not record["kept_after_dedup"] or not record["has_mask"]:
            continue
        mask_path = Path(record["mask_path"]).resolve()
        if not mask_path.is_relative_to(settings.outputs_dir.resolve()):
            raise ValueError("Invalid saved mask path")
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None or not mask.any():
            continue
        mask = mask > 0
        row, geometry = measure_mask(mask, record, calibration)
        measurements.append(row)
        if geometries is not None:
            geometries.append(geometry)
        if progress_callback and (index % 10 == 0 or index == len(records) - 1):
            progress_callback((index + 1) / max(1, len(records)), f"Središnja linija i lokalni presjeci {index + 1}/{len(records)}")
    return measurements, summarize(measurements, calibration)


def _write_measurements(job, records, contours, overlay_path, detection_job_id, pipeline_metadata, progress_callback=None):
    import pandas as pd

    output = settings.outputs_dir / job.id
    output.mkdir(parents=True, exist_ok=True)
    geometries = []
    measurements, volume_summary = measure_final_masks(records, job.params.get("um_per_pixel"), geometries=geometries,
                                                     progress_callback=progress_callback)
    image = Path(job.params["image_path"])
    csv_path = output / f"{image.stem}_exp03_measurements.csv"
    excel_path = csv_path.with_suffix(".xlsx")
    columns = ["detection_id", "annotation_id", "segmentation_confidence", "measurement_method", "measurement_status", "measurement_reliable",
               "volume_included", "volume_mm3", "volume_unit", "length_px", "width_px", "width_min_px", "width_max_px",
               "length_mm", "width_mm", "area_px", "area_mm2", "mask_area_px", "perimeter_px", "perimeter_mm", "calibration_um_per_px",
               "dedup_status", "touches_crop_edge", "mask_components", "warning", "warning_codes", "volume_method", "width_definition",
               "centroid_x_px", "centroid_y_px", "local_section_count", "model_name"]
    frame = pd.DataFrame(measurements, columns=columns)
    frame.to_csv(csv_path, index=False, encoding="utf-8-sig")
    section_rows = [{k: v for k, v in section.items() if k not in ("left", "right")} for g in geometries for section in g["cross_sections"]]
    sections_frame = pd.DataFrame(section_rows, columns=["detection_id", "s_px", "x_px", "y_px", "width_px", "delta_l_px", "volume_px3", "width_mm", "delta_l_mm", "volume_mm3", "volume_included"])
    widths_path = output / "centerline_local_widths.csv"
    sections_frame.to_csv(widths_path, index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        frame.to_excel(writer, sheet_name="Measurements", index=False)
        pd.DataFrame([volume_summary]).to_excel(writer, sheet_name="Volume summary", index=False)
        pd.DataFrame([{"model": MODEL_NAME, "checkpoint_sha256": EXP03_SHA256,
                       "calibration_um_per_px": job.params.get("um_per_pixel"),
                        "measurement_method": METHOD, "volume_method": FORMULA,
                        "summary_policy": "Only measurement_reliable and calibrated rows enter totals; candidates remain diagnostic",
                        "width_definition": "Local perpendicular body cross-sections; median is a descriptive width only"}]).to_excel(writer, sheet_name="Method", index=False)
        sections_frame.to_excel(writer, sheet_name="Local widths", index=False)
    records_path = output / "exp03_mask_records.json"
    records_path.write_text(json.dumps(records, allow_nan=False))
    contour_path = output / "exp03_contours.json"
    # Full section data are in CSV/Excel; keep the viewer geometry compact.
    viewer_geometries = [{**g, "cross_sections": g["cross_sections"][::max(1, len(g["cross_sections"]) // 24)]} for g in geometries]
    contour_path.write_text(json.dumps({"image_id": job.params["image_id"], "measurement_method": METHOD,
                                       "coordinate_system": "global_image_pixels", "organisms": viewer_geometries}, allow_nan=False))
    target_overlay = output / "exp03_overlay.jpg"
    # The overview shows reliability and the curved path, never a fitted ellipse.
    import cv2
    import numpy as np
    overview = cv2.imread(str(image))
    if overview is not None:
        for g in viewer_geometries:
            color = (60, 200, 50) if g["measurement_reliable"] else (0, 145, 255)
            for path in g["contours"]:
                cv2.polylines(overview, [np.round(path).astype(np.int32)], True, color, 2)
            if len(g["centerline"]) > 1:
                cv2.polylines(overview, [np.round(g["centerline"]).astype(np.int32)], False, color, 2)
        cv2.imwrite(str(target_overlay), overview, [cv2.IMWRITE_JPEG_QUALITY, 92])
    elif Path(overlay_path).resolve() != target_overlay.resolve():
        shutil.copy2(overlay_path, target_overlay)
    return {"model_name": MODEL_NAME, "experimental": False, "checkpoint_sha256": EXP03_SHA256,
            "measurement_method": METHOD, "method": METHOD, "local_widths_path": str(widths_path),
            "image_id": job.params["image_id"], "image_path": str(image), "image_stem": image.stem,
            "num_organisms": len(measurements), "csv_path": str(csv_path), "excel_path": str(excel_path),
            "overlay_path": str(target_overlay), "trunk_overlay_path": str(target_overlay),
            "contours_path": str(contour_path), "mask_records_path": str(records_path),
            "measurements": measurements, "volume_summary": volume_summary,
            "calibration_um_per_px": valid_calibration(job.params.get("um_per_pixel")),
            "volume_description": "Integracija lokalnih kružnih presjeka duž zakrivljene središnje linije. Nepouzdane jedinke isključene su iz ukupnog volumena.",
            "volume_formula": FORMULA, "length_definition": "Arc length of cleaned curved body centerline",
            "detection_job_id": detection_job_id, "pipeline": pipeline_metadata, "gt_saved": False}


def run_contours(job: Job, progress_callback):
    """Fixed detector/segmentation/dedup followed automatically by existing mask measurement."""
    checkpoint = validate_checkpoint()
    from backend.jobs.manager import job_manager
    from backend.jobs.models import JobStatus

    if job.params.get("annotation_snapshot") is not None:
        snapshot = job.params["annotation_snapshot"]
        source = job_manager.get(job.params.get("source_contour_job_id", "")) if job.params.get("source_contour_job_id") else None
        if source is None or source.result.get("pipeline", {}).get("annotations_sha256") != job.params.get("annotations_sha256"):
            # New/changed human boxes require segmentation, not reuse of stale crops.
            job.params.pop("source_contour_job_id", None)
            output = settings.outputs_dir / job.id
            output.mkdir(parents=True, exist_ok=True)
            path = output / "reviewed_detections.csv"
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["x1", "y1", "x2", "y2", "confidence", "annotation_id"])
                writer.writeheader()
                for b in snapshot["boxes"]:
                    if b["status"] != "rejected":
                        writer.writerow({**{k: b[k] for k in ("x1", "y1", "x2", "y2")}, "confidence": b.get("conf", 1), "annotation_id": b["id"]})
            job.params["detections_csv"] = str(path)

    if job.params.get("source_contour_job_id"):
        source = job_manager.get(job.params["source_contour_job_id"])
        if source is None or source.type not in (JobType.CONTOURS, JobType.MEASUREMENT) or source.status != JobStatus.COMPLETED:
            raise ValueError("Izvorni posao kontura nije dovršen.")
        if source.result.get("image_id") != job.params["image_id"]:
            raise ValueError("Konture pripadaju drugoj fotografiji.")
        if source.result.get("model_name") != MODEL_NAME or source.result.get("checkpoint_sha256") != EXP03_SHA256:
            raise ValueError("Spremljene maske ne pripadaju fiksiranom exp03_no_os modelu.")
        progress_callback(.2, "Ponovni izračun iz spremljenih konačnih maski; bez inference…")
        records = json.loads(Path(source.result["mask_records_path"]).read_text())
        contours = json.loads(Path(source.result["contours_path"]).read_text())["organisms"]
        result = _write_measurements(job, records, contours, source.result["overlay_path"], source.result.get("detection_job_id"),
                                     {**source.result["pipeline"], "measurement_only": True, "source_contour_job_id": source.id},
                                     lambda p, m: progress_callback(.2 + .75 * p, m))
        progress_callback(1., "Mjerenja iz spremljenih exp03 maski su dovršena.")
        return result

    import cv2
    import numpy as np
    from PIL import Image
    from ultralytics import YOLO
    from backend.services.detection import run_detection

    legacy = _legacy_pipeline()
    output = settings.outputs_dir / job.id
    output.mkdir(parents=True, exist_ok=True)
    detector_job = job_manager.get(job.params.get("detection_job_id", "")) if job.params.get("detection_job_id") else None
    if detector_job is None and not job.params.get("detections_csv"):
        detector_id = uuid.uuid4().hex[:12]
        params = {"image_path": job.params["image_path"], "model_path": str(settings.default_model),
                  "conf": legacy.PARAMS["detector_conf"], "iou": legacy.PARAMS["bbox_nms_iou"],
                  "tile_size": legacy.PARAMS["tile_size"], "overlap": legacy.PARAMS["overlap"], "device": legacy.PARAMS["device"]}
        detected = run_detection(Job(id=detector_id, type=JobType.DETECTION, params=params),
                                 lambda p, m: progress_callback(p * .35, m))
        detector_job = job_manager.register_completed(JobType.DETECTION, params, detected, job_id=detector_id)
    with Path(job.params.get("detections_csv") or detector_job.result["csv_path"]).open(newline="", encoding="utf-8-sig") as handle:
        detections = list(csv.DictReader(handle))
    Image.MAX_IMAGE_PIXELS = None
    with Image.open(job.params["image_path"]) as opened:
        image = cv2.cvtColor(np.array(opened.convert("RGB")), cv2.COLOR_RGB2BGR)
    height, width = image.shape[:2]
    network = YOLO(str(checkpoint)) if detections else None
    rows = []
    mask_dir = output / "masks"
    mask_dir.mkdir(exist_ok=True)
    detector_hash = file_sha256(settings.default_model)
    dedup_hash = file_sha256(legacy.DEDUP_MODEL)
    for index, detection in enumerate(detections):
        bbox = tuple(float(detection[key]) for key in ["x1", "y1", "x2", "y2"])
        x1, y1 = max(0, int(np.floor(bbox[0]))), max(0, int(np.floor(bbox[1])))
        x2, y2 = min(width, int(np.ceil(bbox[2]))), min(height, int(np.ceil(bbox[3])))
        crop = image[y1:y2, x1:x2]
        if not crop.size:
            raise ValueError(f"Prazan bbox crop za detection {index}")
        result = network.predict(crop, imgsz=legacy.PARAMS["seg_imgsz"], conf=legacy.PARAMS["seg_conf"],
                                 device=legacy.PARAMS["device"], verbose=False)[0]
        mask, confidence = legacy.result_mask(result, crop.shape[:2])
        mask_path = mask_dir / f"det{index:05d}.png"
        if mask is not None:
            cv2.imwrite(str(mask_path), mask)
        rows.append({"detection_id": index, "bbox": bbox, "det_confidence": float(detection.get("confidence", detection.get("conf", 1))),
                     "annotation_id": detection.get("annotation_id"),
                     "seg_confidence": float(confidence or 0), "mask": mask, "mask_x": x1, "mask_y": y1,
                     "crop_x1": x1, "crop_y1": y1, "crop_box": [x1, y1, x2, y2], "mask_path": str(mask_path),
                     "has_mask": mask is not None, "segmentation_confidence": confidence})
        progress_callback(.35 + .45 * (index + 1) / len(detections), f"exp03_no_os konture {index + 1}/{len(detections)}")
    progress_callback(.82, "Postojeći fragment-aware classifier dedup…")
    if rows:
        kept, audit, dedup_stats = legacy.apply_classifier_dedup(rows, legacy.joblib.load(legacy.DEDUP_MODEL))
    else:
        kept, audit, dedup_stats = [], [], {"removed_by_classifier_dedup": 0}
    kept_ids = {row["detection_id"] for row in kept}
    (output / "dedup_audit.json").write_text(json.dumps(audit, default=lambda v: v.item() if hasattr(v, "item") else str(v)))
    overlay = legacy.draw_full_overlay(image, kept)
    overlay_path = output / "exp03_overlay.jpg"
    cv2.imwrite(str(overlay_path), overlay, [cv2.IMWRITE_JPEG_QUALITY, 92])
    records = []
    contours = []
    for row in rows:
        records.append({key: value for key, value in row.items() if key not in ["mask", "bbox"]} | {"kept_after_dedup": row["detection_id"] in kept_ids})
        if row["detection_id"] in kept_ids and row["mask"] is not None:
            paths, _ = cv2.findContours((row["mask"] > 0).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            contours.append({"detection_id": row["detection_id"], "bbox": row["crop_box"],
                             "contours": [[[int(p[0]) + row["mask_x"], int(p[1]) + row["mask_y"]] for p in contour[:, 0, :]] for contour in paths if len(contour) >= 3]})
    progress_callback(.9, "Mjerenje zakrivljene linije i lokalnih kružnih presjeka…")
    metadata = {"parameters": dict(legacy.PARAMS), "bbox_padding_px": 0, "dedup": dedup_stats,
                "annotations_sha256": job.params.get("annotations_sha256"),
                "detections": len(rows), "without_mask": sum(not row["has_mask"] for row in rows),
                "removed_duplicates": len(rows) - len(kept_ids), "kept_boxes": len(kept_ids),
                "detector_sha256": detector_hash, "dedup_sha256": dedup_hash}
    result = _write_measurements(job, records, contours, overlay_path, detector_job.id if detector_job else None, metadata,
                                 lambda p, m: progress_callback(.9 + .08 * p, m))
    if file_sha256(checkpoint) != EXP03_SHA256 or file_sha256(settings.default_model) != detector_hash or file_sha256(legacy.DEDUP_MODEL) != dedup_hash:
        raise RuntimeError("Hash modela ili classifiera promijenjen je tijekom obrade.")
    progress_callback(1., "Konture i mjerenja exp03_no_os su dovršeni.")
    return result
