"""Measurement endpoints."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.jobs.manager import job_manager
from backend.jobs.models import JobType
from backend.services.image import get_image_info

router = APIRouter(prefix="/api/measurement", tags=["measurement"])


class MeasurementRequest(BaseModel):
    image_id: str
    detections_csv: str
    um_per_pixel: float
    method: str = "fast"
    device: str | None = None


@router.post("/run")
async def run_measurement(req: MeasurementRequest):
    """Submit a measurement job."""
    info = get_image_info(req.image_id)
    if info is None:
        raise HTTPException(404, "Image not found")

    params = {
        "image_path": info["path"],
        "detections_csv": req.detections_csv,
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
