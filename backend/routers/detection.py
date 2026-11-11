"""Detection endpoints."""

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from backend.auth import get_current_user
from backend.config import settings
from backend.jobs.manager import job_manager
from backend.jobs.models import JobStatus, JobType
from backend.services.annotation_state import detection_job_boxes, detection_job_image_id
from backend.services.image import get_image_info
from backend.services.image_identity import capture_identity

router = APIRouter(prefix="/api/detection", tags=["detection"], dependencies=[Depends(get_current_user)])


class DetectionRequest(BaseModel):
    image_id: str
    model_path: str | None = None
    conf: float | None = None
    iou: float | None = None
    tile_size: int | None = None
    overlap: int | None = None
    extra_tile_sizes: list[int] | None = None
    extra_conf: list[float] | None = None
    contain: float | None = None
    device: str | None = None


@router.post("/run")
async def run_detection(req: DetectionRequest):
    """Submit a detection job."""
    info = get_image_info(req.image_id)
    if info is None:
        raise HTTPException(404, "Image not found")

    params = {"image_path": info["path"], "image_id": req.image_id,
              "image_identity": capture_identity(req.image_id, info["path"])}
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
    if req.extra_tile_sizes is not None:
        params["extra_tile_sizes"] = req.extra_tile_sizes
    if req.extra_conf is not None:
        params["extra_conf"] = req.extra_conf
    if (req.extra_tile_sizes is None) != (req.extra_conf is None) or len(req.extra_tile_sizes or []) != len(req.extra_conf or []):
        raise HTTPException(422, "extra_tile_sizes and extra_conf must be given together, one confidence per tile size")
    if req.contain is not None:
        params["contain"] = req.contain
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


@router.get("/boxes/{job_id}")
async def get_detection_boxes(job_id: str):
    """Return bounding boxes from a completed detection job as JSON."""
    job = job_manager.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    if job.status != JobStatus.COMPLETED:
        raise HTTPException(400, "Job not completed yet")
    if not job.result.get("csv_path"):
        raise HTTPException(404, "No CSV result for this job")
    try:
        boxes = detection_job_boxes(job)
    except FileNotFoundError:
        raise HTTPException(404, "Detections CSV not found on disk")
    # Lets the workspace refuse boxes of a job that belongs to another photo.
    image_id = detection_job_image_id(job)
    return {"image_id": image_id, "boxes": boxes}
