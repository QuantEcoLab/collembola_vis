"""Annotations endpoints — save/load/export per-image box annotations."""

import csv
import io
import json
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel

from backend.auth import get_current_user
from backend.config import settings
from backend.jobs.manager import job_manager
from backend.jobs.models import JobStatus, JobType
from backend.services.annotation_state import (detection_job_boxes, detection_job_image_id, merge_detections,
                                               read_annotation)

router = APIRouter(
    prefix="/api/annotations",
    tags=["annotations"],
    dependencies=[Depends(get_current_user)],
)


class AnnotatedBox(BaseModel):
    id: str
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float
    status: str  # "accepted" | "rejected" | "added"


class AnnotationFile(BaseModel):
    image_id: str
    image_filename: str
    source_job_id: str
    created_at: str
    boxes: list[AnnotatedBox]


def _annotation_path(image_id: str) -> Path:
    return settings.annotations_dir / f"{image_id}.json"


@router.get("/{image_id}")
async def get_annotations(image_id: str):
    """Load saved annotations for an image."""
    path = _annotation_path(image_id)
    if not path.exists():
        raise HTTPException(404, "No annotations found for this image")
    return json.loads(path.read_text())


@router.post("/{image_id}")
async def save_annotations(image_id: str, data: AnnotationFile):
    """Full-replace save annotations for an image."""
    if data.image_id != image_id:
        raise HTTPException(400, "image_id mismatch")
    path = _annotation_path(image_id)
    payload = data.model_dump()
    payload["saved_at"] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(payload, indent=2))
    return {"ok": True, "path": str(path)}


class MergeRequest(BaseModel):
    detection_job_id: str


@router.post("/{image_id}/merge-detections")
async def merge_new_detections(image_id: str, req: MergeRequest):
    """Preview: boxes of a newer detection job merged into the saved annotations of a photo.

    The reviewer's added and rejected boxes are kept; nothing is written until the merged boxes are
    saved with POST /api/annotations/{image_id}.
    """
    job = job_manager.get(req.detection_job_id)
    if job is None or job.type != JobType.DETECTION or job.status != JobStatus.COMPLETED:
        raise HTTPException(400, "Posao detekcije nije dovršen.")
    if detection_job_image_id(job) != image_id:
        raise HTTPException(400, "Detekcije pripadaju drugoj fotografiji.")
    try:
        detections = detection_job_boxes(job)
    except FileNotFoundError:
        raise HTTPException(404, "Nedostaje CSV detekcija.")
    annotation = read_annotation(image_id)
    boxes, summary = merge_detections(annotation, detections)
    return {"image_id": image_id, "source_job_id": job.id,
            "previous_source_job_id": (annotation or {}).get("source_job_id"), "boxes": boxes, "summary": summary}


@router.get("/{image_id}/export")
async def export_annotations(image_id: str, format: str = "json"):
    """Download annotations as JSON or CSV."""
    path = _annotation_path(image_id)
    if not path.exists():
        raise HTTPException(404, "No annotations found for this image")
    data = json.loads(path.read_text())

    if format == "csv":
        output = io.StringIO()
        writer = csv.DictWriter(
            output,
            fieldnames=["id", "x1", "y1", "x2", "y2", "conf", "status"],
        )
        writer.writeheader()
        for box in data.get("boxes", []):
            writer.writerow({k: box.get(k, "") for k in writer.fieldnames})
        content = output.getvalue()
        return Response(
            content=content,
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{image_id}_annotations.csv"'},
        )

    return Response(
        content=json.dumps(data, indent=2),
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{image_id}_annotations.json"'},
    )
