"""Compatibility entry point: all app measurements use the centerline method."""

import sys
from pathlib import Path
from typing import Any

_repo_root = Path(__file__).resolve().parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from backend.config import settings
from backend.jobs.models import Job


def run_measurement(job: Job, progress_callback) -> dict[str, Any]:
    """Reuse same-photo exp03 masks or segment reviewed boxes, never old axes."""
    from backend.services.contours import annotation_snapshot, find_saved_source, run_contours
    params = job.params
    params.setdefault("image_id", Path(params["image_path"]).parent.name)
    if params.get("use_annotations") and "annotation_snapshot" not in params:
        snapshot, digest = annotation_snapshot(params["image_id"])
        if snapshot is None:
            raise ValueError("Nema spremljenih ručnih oznaka")
        params.update(annotation_snapshot=snapshot, annotations_sha256=digest)
    if not params.get("source_contour_job_id"):
        # Reuse saved masks only if they were made from the boxes measured now.
        if params.get("annotation_snapshot") is not None:
            source = find_saved_source(params["image_id"], annotations_sha256=params.get("annotations_sha256"))
        elif params.get("detection_job_id"):
            source = find_saved_source(params["image_id"], detection_job_id=params["detection_job_id"])
        elif params.get("detections_csv"):
            source = None  # boxes of unknown origin: segment them
        else:
            source = find_saved_source(params["image_id"])
        if source:
            params["source_contour_job_id"] = source["id"]
    return run_contours(job, progress_callback)
