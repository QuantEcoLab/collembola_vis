"""Experimental exp03 contours and existing cylinder measurements, without GT writes."""

import csv
import hashlib
import json
import math
import shutil
import uuid
from pathlib import Path

from backend.config import settings
from backend.jobs.models import Job, JobType

REPO = Path(__file__).resolve().parents[2]
EXP03_SHA256 = "3fe97dd43f387815f8ee947eb5d02a6b0a7afadc1be59942a6b4a1f49f9aa0da"
MODEL_NAME = "exp03_no_os"


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


def measure_final_masks(records, um_per_pixel):
    """Reuse application's mask axes, perimeter and cylinder formula, never resegment."""
    import cv2
    import numpy as np
    from scripts.measure_organisms import measure_organism_from_mask

    calibration = valid_calibration(um_per_pixel)
    measurements = []
    for record in records:
        if not record["kept_after_dedup"] or not record["has_mask"]:
            continue
        mask_path = Path(record["mask_path"]).resolve()
        if not mask_path.is_relative_to(settings.outputs_dir.resolve()):
            raise ValueError("Invalid saved mask path")
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None or not mask.any():
            continue  # Never use the existing empty-mask bbox volume fallback.
        mask = mask > 0
        height, width = mask.shape
        # Unit calibration is used only internally to obtain the existing pixel axes.
        # All physical outputs are explicitly null unless the image has a real calibration.
        measured = measure_organism_from_mask(mask, calibration if calibration is not None else 1.0, [0, 0, width, height])
        major = measured["major_axis_px"]
        minor = measured["minor_axis_px"]
        axes_valid = math.isfinite(major) and math.isfinite(minor) and major > 0 and minor > 0
        edge = bool(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any())
        components = int(cv2.connectedComponents(mask.astype(np.uint8))[0] - 1)
        warnings = []
        if edge:
            warnings.append("Maska dodiruje rub cropa")
        if components > 1:
            warnings.append("Maska ima odvojene komponente")
        if not axes_valid:
            warnings.append("Degenerirane osi maske")
        physical = calibration is not None
        volume_available = physical and axes_valid
        x, y = record["crop_box"][:2]
        measurements.append({
            "detection_id": record["detection_id"],
            "segmentation_confidence": record["segmentation_confidence"],
            "volume_mm3": float(measured["volume_um3"] / 1e9) if volume_available else None,
            "volume_unit": "mm³" if volume_available else "nije dostupan bez kalibracije" if not physical else "nije dostupan za degeneriranu masku",
            "length_px": major, "width_px": minor,
            "length_mm": float(measured["length_um"] / 1000) if physical else None,
            "width_mm": float(measured["width_um"] / 1000) if physical else None,
            "area_px": measured["area_px"],
            "area_mm2": float(measured["area_um2"] / 1e6) if physical else None,
            "perimeter_px": measured["perimeter_px"],
            "perimeter_mm": float(measured["perimeter_px"] * calibration / 1000) if physical else None,
            "calibration_um_per_px": calibration,
            "dedup_status": "kept", "touches_crop_edge": edge, "mask_components": components,
            "warning": "; ".join(warnings), "volume_method": "cylinder_estimate",
            "width_definition": "maximum width of the existing equivalent ellipse (minor axis)",
            "centroid_x_px": measured["centroid_x_px"] + x,
            "centroid_y_px": measured["centroid_y_px"] + y,
            "model_name": MODEL_NAME,
        })
    volumes = [m["volume_mm3"] for m in measurements if m["volume_mm3"] is not None]
    summary = {"available": calibration is not None, "count": len(volumes), "unit": "mm³" if calibration is not None else None,
               "mean": float(np.mean(volumes)) if volumes else None,
               "median": float(np.median(volumes)) if volumes else None,
               "min": min(volumes) if volumes else None, "max": max(volumes) if volumes else None,
               "total": float(sum(volumes)) if volumes else 0.0 if calibration is not None and not measurements else None}
    return measurements, summary


def _write_measurements(job, records, contours, overlay_path, detection_job_id, pipeline_metadata):
    import pandas as pd

    output = settings.outputs_dir / job.id
    output.mkdir(parents=True, exist_ok=True)
    measurements, volume_summary = measure_final_masks(records, job.params.get("um_per_pixel"))
    image = Path(job.params["image_path"])
    csv_path = output / f"{image.stem}_exp03_measurements.csv"
    excel_path = csv_path.with_suffix(".xlsx")
    columns = ["detection_id", "segmentation_confidence", "volume_mm3", "volume_unit", "length_px", "width_px",
               "length_mm", "width_mm", "area_px", "area_mm2", "perimeter_px", "perimeter_mm", "calibration_um_per_px",
               "dedup_status", "touches_crop_edge", "mask_components", "warning", "volume_method", "width_definition",
               "centroid_x_px", "centroid_y_px", "model_name"]
    frame = pd.DataFrame(measurements, columns=columns)
    frame.to_csv(csv_path, index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        frame.to_excel(writer, sheet_name="Measurements", index=False)
        pd.DataFrame([volume_summary]).to_excel(writer, sheet_name="Volume summary", index=False)
        pd.DataFrame([{"model": MODEL_NAME, "checkpoint_sha256": EXP03_SHA256,
                       "calibration_um_per_px": job.params.get("um_per_pixel"),
                       "volume_method": "V = pi * (equivalent ellipse width / 2)^2 * length; estimated, not direct 3D"}]).to_excel(writer, sheet_name="Method", index=False)
    records_path = output / "exp03_mask_records.json"
    records_path.write_text(json.dumps(records, allow_nan=False))
    contour_path = output / "exp03_contours.json"
    contour_path.write_text(json.dumps({"image_id": job.params["image_id"], "coordinate_system": "global_image_pixels", "organisms": contours}))
    target_overlay = output / "exp03_overlay.jpg"
    if Path(overlay_path).resolve() != target_overlay.resolve():
        shutil.copy2(overlay_path, target_overlay)
    return {"model_name": MODEL_NAME, "experimental": True, "checkpoint_sha256": EXP03_SHA256,
            "image_id": job.params["image_id"], "image_path": str(image), "image_stem": image.stem,
            "num_organisms": len(measurements), "csv_path": str(csv_path), "excel_path": str(excel_path),
            "overlay_path": str(target_overlay), "trunk_overlay_path": str(target_overlay),
            "contours_path": str(contour_path), "mask_records_path": str(records_path),
            "measurements": measurements, "volume_summary": volume_summary,
            "calibration_um_per_px": valid_calibration(job.params.get("um_per_pixel")),
            "volume_description": "Procijenjeni volumen cilindričnom aproksimacijom; nije izravno izmjeren 3D volumen.",
            "axis_definition": "Postojeće regionprops osi ekvivalentne elipse: duljina = velika os; najveća širina elipse = mala os.",
            "detection_job_id": detection_job_id, "pipeline": pipeline_metadata, "gt_saved": False}


def run_contours(job: Job, progress_callback):
    """Fixed detector/segmentation/dedup followed automatically by existing mask measurement."""
    checkpoint = validate_checkpoint()
    from backend.jobs.manager import job_manager
    from backend.jobs.models import JobStatus

    if job.params.get("source_contour_job_id"):
        source = job_manager.get(job.params["source_contour_job_id"])
        if source is None or source.type != JobType.CONTOURS or source.status != JobStatus.COMPLETED:
            raise ValueError("Izvorni posao kontura nije dovršen.")
        if source.result.get("image_id") != job.params["image_id"]:
            raise ValueError("Konture pripadaju drugoj fotografiji.")
        if source.result.get("model_name") != MODEL_NAME or source.result.get("checkpoint_sha256") != EXP03_SHA256:
            raise ValueError("Spremljene maske ne pripadaju fiksiranom exp03_no_os modelu.")
        progress_callback(.2, "Ponovni izračun iz spremljenih konačnih maski; bez inference…")
        records = json.loads(Path(source.result["mask_records_path"]).read_text())
        contours = json.loads(Path(source.result["contours_path"]).read_text())["organisms"]
        result = _write_measurements(job, records, contours, source.result["overlay_path"], source.result.get("detection_job_id"),
                                     {**source.result["pipeline"], "measurement_only": True, "source_contour_job_id": source.id})
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
    if detector_job is None:
        detector_id = uuid.uuid4().hex[:12]
        params = {"image_path": job.params["image_path"], "model_path": str(settings.default_model),
                  "conf": legacy.PARAMS["detector_conf"], "iou": legacy.PARAMS["bbox_nms_iou"],
                  "tile_size": legacy.PARAMS["tile_size"], "overlap": legacy.PARAMS["overlap"], "device": legacy.PARAMS["device"]}
        detected = run_detection(Job(id=detector_id, type=JobType.DETECTION, params=params),
                                 lambda p, m: progress_callback(p * .35, m))
        detector_job = job_manager.register_completed(JobType.DETECTION, params, detected, job_id=detector_id)
    with Path(detector_job.result["csv_path"]).open(newline="", encoding="utf-8-sig") as handle:
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
    progress_callback(.9, "Mjerenje konačnih maski postojećim osima i cilindričnom metodom…")
    metadata = {"parameters": dict(legacy.PARAMS), "bbox_padding_px": 0, "dedup": dedup_stats,
                "detections": len(rows), "without_mask": sum(not row["has_mask"] for row in rows),
                "removed_duplicates": len(rows) - len(kept_ids), "kept_boxes": len(kept_ids),
                "detector_sha256": detector_hash, "dedup_sha256": dedup_hash}
    result = _write_measurements(job, records, contours, overlay_path, detector_job.id, metadata)
    if file_sha256(checkpoint) != EXP03_SHA256 or file_sha256(settings.default_model) != detector_hash or file_sha256(legacy.DEDUP_MODEL) != dedup_hash:
        raise RuntimeError("Hash modela ili classifiera promijenjen je tijekom obrade.")
    progress_callback(1., "Konture i mjerenja exp03_no_os su dovršeni.")
    return result
