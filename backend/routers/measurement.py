"""Measurement endpoints."""

import csv
import json
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from backend.auth import get_current_user
from backend.config import settings
from backend.jobs.manager import job_manager
from backend.jobs.models import JobType
from backend.services.image import get_image_info

router = APIRouter(prefix="/api/measurement", tags=["measurement"], dependencies=[Depends(get_current_user)])


class MeasurementRequest(BaseModel):
    image_id: str
    detections_csv: str = ""
    um_per_pixel: float | None = None
    method: str = "centerline_circular_sections_v2"
    device: str | None = None
    use_annotations: bool = False


def _annotations_to_csv(image_id: str) -> Path:
    """Read saved annotation file and write a detections CSV with non-rejected boxes.
    Returns the path to the generated CSV."""
    ann_path = settings.annotations_dir / f"{image_id}.json"
    if not ann_path.exists():
        raise FileNotFoundError(f"No annotations found for image {image_id}")

    data = json.loads(ann_path.read_text())
    boxes = [b for b in data.get("boxes", []) if b.get("status") != "rejected"]

    out_path = settings.annotations_dir / f"{image_id}_{uuid.uuid4().hex[:8]}_for_measurement.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["x1", "y1", "x2", "y2", "width", "height", "confidence", "class"])
        writer.writeheader()
        for b in boxes:
            x1, y1, x2, y2 = b["x1"], b["y1"], b["x2"], b["y2"]
            writer.writerow({
                "x1": x1, "y1": y1,
                "x2": x2, "y2": y2,
                "width": x2 - x1,
                "height": y2 - y1,
                "confidence": b.get("conf", 1.0),
                "class": 0,
            })
    return out_path


@router.post("/run")
async def run_measurement(req: MeasurementRequest):
    """Submit a measurement job."""
    info = get_image_info(req.image_id)
    if info is None:
        raise HTTPException(404, "Image not found")

    from backend.services.contours import valid_calibration, validate_checkpoint, annotation_snapshot
    try:
        valid_calibration(req.um_per_pixel)
        validate_checkpoint()
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    except RuntimeError as exc:
        raise HTTPException(503, str(exc))
    detections_csv = None
    if not req.use_annotations and req.detections_csv:
        csv_path = Path(req.detections_csv).resolve()
        if not csv_path.is_relative_to(settings.outputs_dir.resolve()):
            raise HTTPException(400, "Invalid detections path")
        detections_csv = req.detections_csv

    params = {
        "image_id": req.image_id, "image_path": info["path"],
        "um_per_pixel": req.um_per_pixel,
        "use_annotations": req.use_annotations,
        "method": "centerline_circular_sections_v2",
    }
    if detections_csv:
        params["detections_csv"] = detections_csv
    if req.use_annotations:
        snapshot, digest = annotation_snapshot(req.image_id)
        if snapshot is None:
            raise HTTPException(400, "Najprije spremite ručne oznake")
        params.update(annotation_snapshot=snapshot, annotations_sha256=digest)
    if req.device:
        params["device"] = req.device

    job = job_manager.submit(JobType.MEASUREMENT, params)
    return {"job_id": job.id, "status": job.status.value}


@router.get("/result/{job_id}")
async def get_measurement_result(job_id: str):
    """Get measurement results for a completed job."""
    job = job_manager.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    return job.to_dict()
