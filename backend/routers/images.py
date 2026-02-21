"""Image upload and management endpoints."""

from fastapi import APIRouter, Depends, HTTPException, UploadFile

from backend.auth import get_current_user
from backend.services.image import delete_image, get_image_info, list_images, save_upload

router = APIRouter(prefix="/api/images", tags=["images"], dependencies=[Depends(get_current_user)])


@router.post("/upload")
async def upload_image(file: UploadFile):
    """Upload an image and generate a thumbnail."""
    if not file.filename:
        raise HTTPException(400, "No filename provided")

    content = await file.read()
    if len(content) == 0:
        raise HTTPException(400, "Empty file")

    result = save_upload(file.filename, content)
    return result


@router.get("")
async def get_images():
    """List all uploaded images."""
    return list_images()


@router.get("/{image_id}")
async def get_image(image_id: str):
    """Get info about a specific image."""
    info = get_image_info(image_id)
    if info is None:
        raise HTTPException(404, "Image not found")
    return info


@router.delete("/{image_id}")
async def remove_image(image_id: str):
    """Delete an uploaded image."""
    if not delete_image(image_id):
        raise HTTPException(404, "Image not found")
    return {"ok": True}
