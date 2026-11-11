"""Byte identity and native dimensions of a registered photograph, without rewriting it."""
from pathlib import Path

from PIL import Image

from backend.services.detector_registry import sha


def capture_identity(image_id: str, path: str | Path) -> dict:
    path = Path(path)
    stat = path.stat()
    digest = sha(path)
    Image.MAX_IMAGE_PIXELS = None
    with Image.open(path) as image:
        width, height = image.size
    after = path.stat()
    if (stat.st_size, stat.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError('Fotografija je promijenjena tijekom provjere identiteta.')
    return {'image_id': image_id, 'filename': path.name, 'width': width, 'height': height,
            'sha256': digest, 'registered_path': str(path.parent.resolve() / path.name),
            'resolved_path': str(path.resolve())}
