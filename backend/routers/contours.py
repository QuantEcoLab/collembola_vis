"""Fixed experimental contour workflow; accepts no inference parameter overrides."""

import json
import math
import threading
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, field_validator

from backend.auth import get_current_user
from backend.config import settings
from backend.jobs.manager import job_manager
from backend.jobs.models import JobStatus, JobType
from backend.services.contours import MODEL_NAME, EXP03_SHA256, validate_checkpoint
from backend.services.image import get_image_info

router = APIRouter(prefix="/api/contours", tags=["experimental-contours"], dependencies=[Depends(get_current_user)])
_submission_lock = threading.Lock()


class ContourRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    image_id: str
    detection_job_id: str | None = None
    source_contour_job_id: str | None = None
    um_per_pixel: float | None = None

    @field_validator("um_per_pixel", mode="before")
    @classmethod
    def positive_calibration(cls, value):
        if value is not None and (type(value) not in (int, float) or not math.isfinite(value) or value <= 0):
            raise ValueError("Kalibracija mora biti konačna i veća od nule (µm/px)")
        return value


@router.get("/model")
async def contour_model():
    try:
        validate_checkpoint()
    except RuntimeError as exc:
        raise HTTPException(503, str(exc))
    return {"name": MODEL_NAME, "experimental": True, "sha256": EXP03_SHA256}


@router.post("/run")
async def run_contours(req: ContourRequest):
    info = get_image_info(req.image_id)
    if info is None:
        raise HTTPException(404, "Fotografija nije učitana ili više nije dostupna.")
    try:
        validate_checkpoint()
    except RuntimeError as exc:
        raise HTTPException(503, str(exc))
    if req.detection_job_id and req.source_contour_job_id:
        raise HTTPException(400, "Odaberite postojeće detekcije ili ponovno mjerenje maski, ne oboje.")
    if req.detection_job_id:
        detection = job_manager.get(req.detection_job_id)
        if detection is None or detection.type != JobType.DETECTION or detection.status != JobStatus.COMPLETED:
            raise HTTPException(400, "Izvorni posao detekcije nije dovršen.")
        if Path(detection.result.get("image_path", "")).resolve() != Path(info["path"]).resolve():
            raise HTTPException(400, "Detekcije pripadaju drugoj fotografiji.")
        path = Path(detection.result.get("csv_path", "")).resolve()
        if not path.is_relative_to(settings.outputs_dir.resolve()) or not path.is_file():
            raise HTTPException(400, "Nedostaje izvorni CSV detekcija.")
        expected = {"conf": .6, "iou": .5, "tile_size": 1280, "overlap": 256}
        provenance = dict(detection.params)
        if not provenance:
            metadata_path = Path(detection.result.get("metadata_path", "")).resolve()
            if not metadata_path.is_relative_to(settings.outputs_dir.resolve()) or not metadata_path.is_file():
                raise HTTPException(400, "Nije moguće potvrditi parametre starih detekcija; ponovite standardnu detekciju.")
            metadata = json.loads(metadata_path.read_text())
            provenance = {"conf": metadata.get("conf_threshold"), "iou": metadata.get("iou_threshold"),
                          "tile_size": metadata.get("tile_size"), "overlap": metadata.get("overlap"), "model_path": metadata.get("model_path")}
        defaults = {"conf": settings.default_conf, "iou": settings.default_iou, "tile_size": settings.default_tile_size, "overlap": settings.default_overlap}
        if any(provenance.get(k, defaults[k]) != v for k, v in expected.items()):
            raise HTTPException(400, "Za exp03 koristite standardne detekcije (conf 0.6, IoU 0.5, tile 1280, overlap 256).")
        if Path(provenance.get("model_path") or settings.default_model).resolve() != settings.default_model.resolve():
            raise HTTPException(400, "Detekcije ne koriste postojeći zadani detector.")
    if req.source_contour_job_id:
        source = job_manager.get(req.source_contour_job_id)
        if source is None or source.type != JobType.CONTOURS or source.status != JobStatus.COMPLETED or source.result.get("image_id") != req.image_id or source.result.get("model_name") != MODEL_NAME or source.result.get("checkpoint_sha256") != EXP03_SHA256:
            raise HTTPException(400, "Nema dovršenih exp03 kontura za ovu fotografiju.")
    params = {"image_id": req.image_id, "image_path": info["path"], "um_per_pixel": req.um_per_pixel}
    if req.detection_job_id:
        params["detection_job_id"] = req.detection_job_id
    if req.source_contour_job_id:
        params["source_contour_job_id"] = req.source_contour_job_id
    with _submission_lock:
        active = next((job for job in job_manager.list_jobs() if job.type == JobType.CONTOURS and job.params.get("image_id") == req.image_id and job.status in [JobStatus.PENDING, JobStatus.RUNNING]), None)
        job = active or job_manager.submit(JobType.CONTOURS, params)
    return {"job_id": job.id, "status": job.status.value, "model_name": MODEL_NAME, "experimental": True}
