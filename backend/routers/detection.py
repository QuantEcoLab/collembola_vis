"""Detection endpoints."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.config import settings
from backend.jobs.manager import job_manager
from backend.jobs.models import JobType
from backend.services.image import get_image_info

router = APIRouter(prefix="/api/detection", tags=["detection"])


class DetectionRequest(BaseModel):
    image_id: str
    model_path: str | None = None
    conf: float | None = None
    iou: float | None = None
    tile_size: int | None = None
    overlap: int | None = None
    device: str | None = None


@router.post("/run")
async def run_detection(req: DetectionRequest):
    """Submit a detection job."""
    info = get_image_info(req.image_id)
    if info is None:
        raise HTTPException(404, "Image not found")

    params = {"image_path": info["path"]}
    if req.model_path:
        params["model_path"] = req.model_path
    if req.conf is not None:
        params["conf"] = req.conf
    if req.iou is not None:
        params["iou"] = req.iou
    if req.tile_size is not None:
        params["tile_size"] = req.tile_size
    if req.overlap is not None:
        params["overlap"] = req.overlap
    if req.device is not None:
        params["device"] = req.device

    job = job_manager.submit(JobType.DETECTION, params)
    return {"job_id": job.id, "status": job.status.value}


@router.get("/result/{job_id}")
async def get_detection_result(job_id: str):
    """Get detection results for a completed job."""
    job = job_manager.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    return job.to_dict()
