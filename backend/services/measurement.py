"""Measurement service — wraps measurement scripts."""

import sys
from pathlib import Path
from typing import Any

_repo_root = Path(__file__).resolve().parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from backend.config import settings
from backend.jobs.models import Job


def run_measurement(job: Job, progress_callback) -> dict[str, Any]:
    """Execute measurement for a job.

    Expected job.params:
        image_path: str
        detections_csv: str
        um_per_pixel: float
        method: str ('fast' or 'sam', default 'fast')
        device: str (optional, for SAM)
    """
    params = job.params
    image_path = Path(params["image_path"])
    detections_csv = Path(params["detections_csv"])
    method = params.get("method", "fast")

    output_dir = settings.outputs_dir / job.id
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / f"{image_path.stem}_measurements.csv"

    um_per_pixel = params["um_per_pixel"]

    if method == "sam":
        from scripts.measure_organisms import measure_organisms

        df = measure_organisms(
            image_path=image_path,
            detections_csv=detections_csv,
            output_csv=output_csv,
            um_per_pixel=um_per_pixel,
            device=params.get("device", "cuda"),
            progress_callback=progress_callback,
        )
    else:
        from scripts.measure_organisms_fast import measure_organisms_fast

        df = measure_organisms_fast(
            image_path=image_path,
            detections_csv=detections_csv,
            output_csv=output_csv,
            um_per_pixel=um_per_pixel,
            progress_callback=progress_callback,
        )

    return {
        "num_organisms": len(df),
        "csv_path": str(output_csv),
        "method": method,
        "image_path": str(image_path),
        "summary": {
            col: {
                "mean": round(float(df[col].mean()), 4),
                "median": round(float(df[col].median()), 4),
                "min": round(float(df[col].min()), 4),
                "max": round(float(df[col].max()), 4),
            }
            for col in df.select_dtypes(include="number").columns
            if col not in ("detection_id", "class")
        },
    }
