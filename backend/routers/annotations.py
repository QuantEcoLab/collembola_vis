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
