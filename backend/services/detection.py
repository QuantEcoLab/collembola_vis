"""Detection service — wraps scripts/infer_tiled.py."""

import sys
from pathlib import Path
from typing import Any

# Ensure repo root is on sys.path so we can import scripts
_repo_root = Path(__file__).resolve().parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from backend.config import settings
from backend.jobs.models import Job


def run_detection(job: Job, progress_callback) -> dict[str, Any]:
    """Execute tiled inference for a job.

    Expected job.params:
        image_path: str
        model_path: str (optional, default from settings)
        conf: float (optional)
        iou: float (optional)
        tile_size: int (optional)
        overlap: int (optional)
        device: str (optional)
    """
    from scripts.infer_tiled import infer_tiled

    params = job.params
    image_path = Path(params["image_path"])
    output_dir = settings.outputs_dir / job.id

    detections = infer_tiled(
        image_path=image_path,
        model_path=params.get("model_path", str(settings.default_model)),
        tile_size=params.get("tile_size", settings.default_tile_size),
        overlap=params.get("overlap", settings.default_overlap),
        conf_threshold=params.get("conf", settings.default_conf),
        iou_threshold=params.get("iou", settings.default_iou),
        output_dir=str(output_dir),
        device=params.get("device", settings.default_device),
        progress_callback=progress_callback,
    )

    image_stem = image_path.stem
    csv_path = output_dir / f"{image_stem}_detections.csv"
    overlay_path = output_dir / f"{image_stem}_overlay.jpg"
    metadata_path = output_dir / f"{image_stem}_metadata.json"

    return {
        "num_detections": len(detections),
        "csv_path": str(csv_path),
        "overlay_path": str(overlay_path),
        "metadata_path": str(metadata_path),
        "image_path": str(image_path),
        "image_stem": image_stem,
    }
