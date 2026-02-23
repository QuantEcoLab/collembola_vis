"""Community detection submissions router."""

import csv
import io
import json
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from backend import db
from backend.auth import get_current_user

router = APIRouter(prefix="/api/community", tags=["community"])


class SubmitRequest(BaseModel):
    image_name: str
    image_width: int | None = None
    image_height: int | None = None
    um_per_pixel: float | None = None
    conf_threshold: float | None = None
    boxes: list[dict]


@router.post("/submit")
async def submit(req: SubmitRequest, username: str = Depends(get_current_user)):
    entry_id = db.insert(
        username=username,
        submitted_at=datetime.now(timezone.utc).isoformat(),
        image_name=req.image_name,
        image_width=req.image_width,
        image_height=req.image_height,
        num_detections=len(req.boxes),
        um_per_pixel=req.um_per_pixel,
        conf_threshold=req.conf_threshold,
        boxes=req.boxes,
    )
    return {"id": entry_id, "num_detections": len(req.boxes)}


@router.get("/list")
async def list_entries(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    search: str = Query("", max_length=200),
    _: str = Depends(get_current_user),
):
    return db.list_recent(limit=limit, offset=offset, search=search)


@router.get("/stats")
async def stats(_: str = Depends(get_current_user)):
    return db.get_stats()


@router.get("/{entry_id}")
async def get_entry(entry_id: str, _: str = Depends(get_current_user)):
    entry = db.get_by_id(entry_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="Entry not found")
    return entry


@router.get("/{entry_id}/export")
async def export_entry(
    entry_id: str,
    format: str = Query("json", pattern="^(json|csv)$"),
    _: str = Depends(get_current_user),
):
    entry = db.get_by_id(entry_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="Entry not found")

    stem = entry["image_name"].rsplit(".", 1)[0]

    if format == "json":
        content = json.dumps(entry["boxes"], indent=2)
        return StreamingResponse(
            io.BytesIO(content.encode()),
            media_type="application/json",
            headers={"Content-Disposition": f'attachment; filename="{stem}_community.json"'},
        )

    # CSV
    buf = io.StringIO()
    if entry["boxes"]:
        keys = list(entry["boxes"][0].keys())
        writer = csv.DictWriter(buf, fieldnames=keys)
        writer.writeheader()
        writer.writerows(entry["boxes"])
    content_bytes = buf.getvalue().encode()
    return StreamingResponse(
        io.BytesIO(content_bytes),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{stem}_community.csv"'},
    )
