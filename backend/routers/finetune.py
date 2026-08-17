"""Fine-tune endpoints (single-image and all-annotations)."""

import json
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from backend.auth import require_admin
from backend.config import settings
from backend.jobs.manager import job_manager
from backend.jobs.models import JobType
from backend.services.image import get_image_info

router = APIRouter(
    prefix="/api/finetune",
    tags=["finetune"],
    dependencies=[Depends(require_admin)],
)


class FinetuneRequest(BaseModel):
    image_id: str
    annotation_file: str   # image_id used as annotation file key
    base_model: str | None = None
    epochs: int = 15
    device: str | None = None
    tile_size: int | None = None
    overlap: int | None = None


@router.post("/run")
async def run_finetune(req: FinetuneRequest):
    """Submit a fine-tune job."""
    info = get_image_info(req.image_id)
    if info is None:
        raise HTTPException(404, "Image not found")

    annotation_path = settings.annotations_dir / f"{req.annotation_file}.json"
    if not annotation_path.exists():
        raise HTTPException(404, f"Annotation file not found: {annotation_path}")

    params: dict = {
        "image_path": info["path"],
        "annotation_path": str(annotation_path),
        "epochs": req.epochs,
    }
    if req.base_model:
        params["base_model"] = req.base_model
    if req.device:
        params["device"] = req.device
    if req.tile_size is not None:
        params["tile_size"] = req.tile_size
    if req.overlap is not None:
        params["overlap"] = req.overlap

    job = job_manager.submit(JobType.FINETUNE, params)
    return {"job_id": job.id, "status": job.status.value}


class FinetuneAllRequest(BaseModel):
    base_model: str | None = None
    epochs: int = 20
    device: str | None = None
    tile_size: int | None = None
    overlap: int | None = None
    min_added: int = 10  # only use images where user added > this many boxes


@router.post("/run-all")
async def run_finetune_all(req: FinetuneAllRequest):
    """Fine-tune on all images where users manually added > min_added new boxes."""
    params: dict = {"epochs": req.epochs, "min_added": req.min_added}
    if req.base_model:
        params["base_model"] = req.base_model
    if req.device:
        params["device"] = req.device
    if req.tile_size is not None:
        params["tile_size"] = req.tile_size
    if req.overlap is not None:
        params["overlap"] = req.overlap

    job = job_manager.submit(JobType.FINETUNE_ALL, params)
    return {"job_id": job.id, "status": job.status.value}
