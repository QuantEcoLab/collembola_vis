"""Image upload handling and thumbnail generation."""

import uuid
from pathlib import Path

from PIL import Image

from backend.config import settings


def save_upload(filename: str, content: bytes) -> dict:
    """Save an uploaded image and generate a thumbnail.

    Returns dict with image_id, filename, path, thumbnail_path, width, height.
    """
    image_id = uuid.uuid4().hex[:12]
    image_dir = settings.uploads_dir / image_id
    image_dir.mkdir(parents=True, exist_ok=True)

    # Save original
    original_path = image_dir / filename
    original_path.write_bytes(content)

    # Open and get dimensions
    Image.MAX_IMAGE_PIXELS = None
    img = Image.open(original_path)
    width, height = img.size

    # Generate thumbnail
    thumbnail_path = image_dir / "thumbnail.jpg"
    max_side = settings.thumbnail_max_side
    if max(width, height) > max_side:
        ratio = max_side / max(width, height)
        new_size = (int(width * ratio), int(height * ratio))
        thumb = img.resize(new_size, Image.LANCZOS)
    else:
        thumb = img.copy()
    thumb = thumb.convert("RGB")
    thumb.save(thumbnail_path, "JPEG", quality=85)

    return {
        "image_id": image_id,
        "filename": filename,
        "path": str(original_path),
        "thumbnail_path": str(thumbnail_path),
        "width": width,
        "height": height,
        "thumbnail_width": thumb.width,
        "thumbnail_height": thumb.height,
    }


def get_image_info(image_id: str) -> dict | None:
    """Get info about a previously uploaded image."""
    image_dir = settings.uploads_dir / image_id
    if not image_dir.exists():
        return None

    # Find the original file (not thumbnail)
    files = [f for f in image_dir.iterdir() if f.name != "thumbnail.jpg"]
    if not files:
        return None

    original = files[0]
    thumbnail = image_dir / "thumbnail.jpg"

    Image.MAX_IMAGE_PIXELS = None
    img = Image.open(original)
    width, height = img.size
    img.close()

    return {
        "image_id": image_id,
        "filename": original.name,
        "path": str(original),
        "thumbnail_path": str(thumbnail) if thumbnail.exists() else None,
        "width": width,
        "height": height,
    }


def list_images() -> list[dict]:
    """List all uploaded images."""
    if not settings.uploads_dir.exists():
        return []

    images = []
    for image_dir in sorted(settings.uploads_dir.iterdir()):
        if image_dir.is_dir():
            info = get_image_info(image_dir.name)
            if info:
                images.append(info)
    return images


def register_server_path(image_path: Path) -> dict:
    """Register an existing server-side image without copying it.

    Creates a symlink and generates a thumbnail only.
    Returns the same dict shape as save_upload.
    """
    image_id = uuid.uuid4().hex[:12]
    image_dir = settings.uploads_dir / image_id
    image_dir.mkdir(parents=True, exist_ok=True)

    # Symlink to the original — no copy
    link = image_dir / image_path.name
    link.symlink_to(image_path.resolve())

    Image.MAX_IMAGE_PIXELS = None
    img = Image.open(image_path)
    width, height = img.size

    thumbnail_path = image_dir / "thumbnail.jpg"
    max_side = settings.thumbnail_max_side
    if max(width, height) > max_side:
        ratio = max_side / max(width, height)
        new_size = (int(width * ratio), int(height * ratio))
        thumb = img.resize(new_size, Image.LANCZOS)
    else:
        thumb = img.copy()
    thumb = thumb.convert("RGB")
    thumb.save(thumbnail_path, "JPEG", quality=85)

    return {
        "image_id": image_id,
        "filename": image_path.name,
        "path": str(image_path.resolve()),
        "thumbnail_path": str(thumbnail_path),
        "width": width,
        "height": height,
        "thumbnail_width": thumb.width,
        "thumbnail_height": thumb.height,
    }


def delete_image(image_id: str) -> bool:
    """Delete an uploaded image and its thumbnail."""
    import shutil

    image_dir = settings.uploads_dir / image_id
    if not image_dir.exists():
        return False
    shutil.rmtree(image_dir)
    return True
