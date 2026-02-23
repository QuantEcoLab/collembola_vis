"""Measurement endpoints."""

import csv
import json
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
    detections_csv: str
    um_per_pixel: float
    method: str = "fast"
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

    out_path = settings.annotations_dir / f"{image_id}_for_measurement.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["x1", "y1", "x2", "y2", "conf"])
        writer.writeheader()
        for b in boxes:
            writer.writerow({
                "x1": b["x1"], "y1": b["y1"],
                "x2": b["x2"], "y2": b["y2"],
                "conf": b.get("conf", 1.0),
            })
    return out_path


@router.post("/run")
async def run_measurement(req: MeasurementRequest):
    """Submit a measurement job."""
    info = get_image_info(req.image_id)
    if info is None:
        raise HTTPException(404, "Image not found")

    if req.use_annotations:
        try:
            detections_csv = str(_annotations_to_csv(req.image_id))
        except FileNotFoundError as e:
            raise HTTPException(404, str(e))
    else:
        detections_csv = req.detections_csv

    params = {
        "image_path": info["path"],
        "detections_csv": detections_csv,
        "um_per_pixel": req.um_per_pixel,
        "method": req.method,
    }
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
