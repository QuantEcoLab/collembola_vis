"""Projects endpoints."""

import csv
import io
import json

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

from backend import db_projects
from backend.auth import get_current_user
from backend.config import settings
from backend.jobs.manager import job_manager
from backend.jobs.models import JobType
from backend.services.image import get_image_info


def _annotation_path(image_id: str):
    return settings.annotations_dir / f"{image_id}.json"


def _enrich_image(img: dict) -> dict:
    """Add annotation stats to a project image row."""
    path = _annotation_path(img["image_id"])
    if path.exists():
        try:
            data = json.loads(path.read_text())
            boxes = data.get("boxes", [])
            img["has_annotation"] = True
            img["annotation_total"] = len(boxes)
            img["annotation_accepted"] = sum(
                1 for b in boxes if b.get("status") != "rejected"
            )
        except Exception:
            img["has_annotation"] = False
            img["annotation_total"] = 0
            img["annotation_accepted"] = 0
    else:
        img["has_annotation"] = False
        img["annotation_total"] = 0
        img["annotation_accepted"] = 0
    return img

router = APIRouter(
    prefix="/api/projects",
    tags=["projects"],
    dependencies=[Depends(get_current_user)],
)


def _require_owner(project_id: str, username: str) -> dict:
    project = db_projects.get_project(project_id)
    if project is None:
        raise HTTPException(404, "Project not found")
    if project["created_by"] != username:
        raise HTTPException(403, "Only the project owner can perform this action")
    return project


# ── Request models ─────────────────────────────────────────────────────────────

class CreateProjectRequest(BaseModel):
    name: str
    description: str = ""


class UpdateProjectRequest(BaseModel):
    name: str
    description: str = ""


class AddImageRequest(BaseModel):
    image_id: str
    filename: str


class BatchDetectRequest(BaseModel):
    conf: float | None = None
    tile_size: int | None = None
    overlap: int | None = None
    device: str | None = None


class UpdateJobsRequest(BaseModel):
    detection_job_id: str | None = None
    measurement_job_id: str | None = None


class BatchMeasureRequest(BaseModel):
    um_per_pixel: float
    method: str = "fast"
    device: str | None = None


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("")
async def create_project(
    req: CreateProjectRequest,
    username: str = Depends(get_current_user),
):
    pid = db_projects.create_project(req.name, req.description, username)
    return {"id": pid, "name": req.name}


@router.get("")
async def list_projects():
    return db_projects.list_projects()


@router.get("/{project_id}")
async def get_project(project_id: str):
    project = db_projects.get_project(project_id)
    if project is None:
        raise HTTPException(404, "Project not found")
    project["images"] = [_enrich_image(img) for img in project["images"]]
    return project


@router.get("/{project_id}/annotations/export")
async def export_project_annotations(project_id: str, format: str = "json"):
    """Export all annotations for every image in the project as a single file."""
    project = db_projects.get_project(project_id)
    if project is None:
        raise HTTPException(404, "Project not found")

    images = project.get("images", [])
    combined = []
    for img in images:
        path = _annotation_path(img["image_id"])
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text())
            combined.append({
                "image_id": img["image_id"],
                "filename": img["filename"],
                "boxes": data.get("boxes", []),
                "saved_at": data.get("saved_at"),
            })
        except Exception:
            pass

    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in project["name"])

    if format == "csv":
        output = io.StringIO()
        writer = csv.DictWriter(
            output,
            fieldnames=["image_id", "filename", "box_id", "x1", "y1", "x2", "y2", "conf", "status"],
        )
        writer.writeheader()
        for entry in combined:
            for box in entry["boxes"]:
                writer.writerow({
                    "image_id": entry["image_id"],
                    "filename": entry["filename"],
                    "box_id": box.get("id", ""),
                    "x1": box.get("x1", ""),
                    "y1": box.get("y1", ""),
                    "x2": box.get("x2", ""),
                    "y2": box.get("y2", ""),
                    "conf": box.get("conf", ""),
                    "status": box.get("status", ""),
                })
        return Response(
            content=output.getvalue(),
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{safe_name}_annotations.csv"'},
        )

    payload = {"project_id": project_id, "project_name": project["name"], "images": combined}
    return Response(
        content=json.dumps(payload, indent=2),
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{safe_name}_annotations.json"'},
    )


@router.patch("/{project_id}")
async def update_project(
    project_id: str,
    req: UpdateProjectRequest,
    username: str = Depends(get_current_user),
):
    _require_owner(project_id, username)
    db_projects.update_project(project_id, req.name, req.description)
    return {"ok": True}


@router.delete("/{project_id}")
async def delete_project(
    project_id: str,
    username: str = Depends(get_current_user),
):
    _require_owner(project_id, username)
    db_projects.delete_project(project_id)
    return {"ok": True}


@router.post("/{project_id}/images")
async def add_image(
    project_id: str,
    req: AddImageRequest,
    username: str = Depends(get_current_user),
):
    if db_projects.get_project(project_id) is None:
        raise HTTPException(404, "Project not found")
    # Verify image exists
    info = get_image_info(req.image_id)
    if info is None:
        raise HTTPException(404, "Image not found")
    row_id = db_projects.add_image(project_id, req.image_id, req.filename, username)
    return {"id": row_id}


@router.delete("/{project_id}/images/{image_id}")
async def remove_image(
    project_id: str,
    image_id: str,
    username: str = Depends(get_current_user),
):
    if db_projects.get_project(project_id) is None:
        raise HTTPException(404, "Project not found")
    db_projects.remove_image(project_id, image_id)
    return {"ok": True}


@router.post("/{project_id}/detect")
async def batch_detect(
    project_id: str,
    req: BatchDetectRequest,
    username: str = Depends(get_current_user),
):
    project = db_projects.get_project(project_id)
    if project is None:
        raise HTTPException(404, "Project not found")
    images = project.get("images", [])
    if not images:
        raise HTTPException(400, "Project has no images")

    # Build image entries with resolved paths
    image_entries = []
    for img in images:
        info = get_image_info(img["image_id"])
        if info is None:
            continue
        image_entries.append({
            "image_id": img["image_id"],
            "image_path": info["path"],
            "filename": img["filename"],
        })

    if not image_entries:
        raise HTTPException(400, "No valid images found in project")

    params = {
        "project_id": project_id,
        "image_entries": image_entries,
    }
    if req.conf is not None:
        params["conf"] = req.conf
    if req.tile_size is not None:
        params["tile_size"] = req.tile_size
    if req.overlap is not None:
        params["overlap"] = req.overlap
    if req.device is not None:
        params["device"] = req.device

    job = job_manager.submit(JobType.BATCH, params)
    return {"job_id": job.id, "status": job.status.value}


@router.patch("/{project_id}/images/{image_id}/jobs")
async def update_image_jobs(
    project_id: str,
    image_id: str,
    req: UpdateJobsRequest,
    username: str = Depends(get_current_user),
):
    project = db_projects.get_project(project_id)
    if project is None:
        raise HTTPException(404, "Project not found")
    if req.detection_job_id is not None:
        db_projects.set_detection_job(project_id, image_id, req.detection_job_id)
    if req.measurement_job_id is not None:
        db_projects.set_measurement_job(project_id, image_id, req.measurement_job_id)
    return {"ok": True}


@router.post("/{project_id}/measure")
async def batch_measure(
    project_id: str,
    req: BatchMeasureRequest,
    username: str = Depends(get_current_user),
):
    project = db_projects.get_project(project_id)
    if project is None:
        raise HTTPException(404, "Project not found")

    images = [_enrich_image(img) for img in project["images"]]
    measurable = [img for img in images if img["detection_job_id"] or img["has_annotation"]]
    if not measurable:
        raise HTTPException(400, "No images with detections or annotations")

    image_entries = []
    for img in measurable:
        info = get_image_info(img["image_id"])
        if info is None:
            continue
        image_entries.append({
            "image_id": img["image_id"],
            "image_path": info["path"],
            "filename": img["filename"],
            "detection_job_id": img["detection_job_id"],
            "use_annotations": img["has_annotation"],
        })

    if not image_entries:
        raise HTTPException(400, "No valid images found in project")

    params = {
        "project_id": project_id,
        "image_entries": image_entries,
        "um_per_pixel": req.um_per_pixel,
        "method": req.method,
        "device": req.device,
    }
    job = job_manager.submit(JobType.BATCH_MEASURE, params)
    return {"job_id": job.id, "status": job.status.value}
