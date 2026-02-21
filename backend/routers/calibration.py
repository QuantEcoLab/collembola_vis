"""Calibration endpoints."""

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from backend.auth import get_current_user
from backend.services.calibration import (
    auto_calibrate,
    get_calibration,
    manual_calibrate,
    save_calibration,
)
from backend.services.image import get_image_info

router = APIRouter(prefix="/api/calibration", tags=["calibration"], dependencies=[Depends(get_current_user)])


class AutoCalibrateRequest(BaseModel):
    image_id: str
    ruler_mm: float = 10.0


class ManualCalibrateRequest(BaseModel):
    image_id: str
    point1: list[float]
    point2: list[float]
    known_mm: float


@router.post("/auto")
async def run_auto_calibration(req: AutoCalibrateRequest):
    """Run automatic ruler detection on an image."""
    info = get_image_info(req.image_id)
    if info is None:
        raise HTTPException(404, "Image not found")

    result = auto_calibrate(Path(info["path"]), req.ruler_mm)

    if result.get("um_per_pixel") is None:
        return result  # return with method=auto_failed

    image_stem = Path(info["filename"]).stem
    cal_id = save_calibration(req.image_id, image_stem, result)
    result["calibration_id"] = cal_id
    return result


@router.post("/manual")
async def run_manual_calibration(req: ManualCalibrateRequest):
    """Compute calibration from two clicked points."""
    info = get_image_info(req.image_id)
    if info is None:
        raise HTTPException(404, "Image not found")

    result = manual_calibrate(req.image_id, req.point1, req.point2, req.known_mm)

    image_stem = Path(info["filename"]).stem
    cal_id = save_calibration(req.image_id, image_stem, result)
    result["calibration_id"] = cal_id
    return result


@router.get("/{image_id}")
async def get_image_calibration(image_id: str):
    """Retrieve saved calibration for an image."""
    info = get_image_info(image_id)
    if info is None:
        raise HTTPException(404, "Image not found")

    image_stem = Path(info["filename"]).stem
    cal = get_calibration(image_stem)
    if cal is None:
        raise HTTPException(404, "No calibration found for this image")
    return cal
