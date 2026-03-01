"""Batch measurement service — runs fast measurement for each image in a project sequentially."""

import csv
import json
import sys
import uuid
from pathlib import Path
from typing import Any

_repo_root = Path(__file__).resolve().parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from backend import db_projects
from backend.config import settings
from backend.jobs.manager import job_manager
from backend.jobs.models import Job, JobType


def _annotations_to_csv(image_id: str) -> Path:
    """Read saved annotation file and write a detections CSV with non-rejected boxes."""
    ann_path = settings.annotations_dir / f"{image_id}.json"
    if not ann_path.exists():
        raise FileNotFoundError(f"No annotations found for image {image_id}")

    data = json.loads(ann_path.read_text())
    boxes = [b for b in data.get("boxes", []) if b.get("status") != "rejected"]

    out_path = settings.annotations_dir / f"{image_id}_for_measurement.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["x1", "y1", "x2", "y2", "width", "height", "confidence", "class"]
        )
        writer.writeheader()
        for b in boxes:
            x1, y1, x2, y2 = b["x1"], b["y1"], b["x2"], b["y2"]
            writer.writerow({
                "x1": x1, "y1": y1,
                "x2": x2, "y2": y2,
                "width": x2 - x1,
                "height": y2 - y1,
                "confidence": b.get("conf", 1.0),
                "class": 0,
            })
    return out_path


def run_batch_measurement(job: Job, progress_callback) -> dict[str, Any]:
    """Execute fast measurement for each image in a project sequentially.

    Expected job.params:
        project_id: str
        image_entries: list of {image_id, image_path, filename, detection_job_id, use_annotations}
        um_per_pixel: float
        method: str ('fast' | 'sam', default 'fast')
        device: str | None
    """
    from scripts.measure_organisms_fast import measure_organisms_fast

    params = job.params
    project_id = params["project_id"]
    image_entries = params["image_entries"]
    um_per_pixel = params["um_per_pixel"]
    method = params.get("method", "fast")
    total = len(image_entries)

    results = []

    for i, entry in enumerate(image_entries):
        image_id = entry["image_id"]
        image_path = Path(entry["image_path"])
        filename = entry["filename"]
        image_stem = image_path.stem

        progress_callback(
            i / total,
            f"Measuring {filename} ({i + 1}/{total})",
        )

        # Resolve detections CSV
        try:
            if entry.get("use_annotations"):
                detections_csv = _annotations_to_csv(image_id)
            else:
                det_job_id = entry["detection_job_id"]
                detections_csv = settings.outputs_dir / det_job_id / f"{image_stem}_detections.csv"

            meas_job_id = uuid.uuid4().hex[:12]
            output_dir = settings.outputs_dir / meas_job_id
            output_dir.mkdir(parents=True, exist_ok=True)
            output_csv = output_dir / f"{image_stem}_measurements.csv"

            sam_overlay_path = None
            if method == "sam":
                from scripts.measure_organisms import measure_organisms, _normalize_device
                sam_device = _normalize_device(params.get("device", "cuda"))
                df, sam_overlay_path = measure_organisms(
                    image_path=image_path,
                    detections_csv=detections_csv,
                    output_csv=output_csv,
                    um_per_pixel=um_per_pixel,
                    device=sam_device,
                    progress_callback=None,
                )
            else:
                df = measure_organisms_fast(
                    image_path=image_path,
                    detections_csv=detections_csv,
                    output_csv=output_csv,
                    um_per_pixel=um_per_pixel,
                    progress_callback=None,
                )

            # Register as a completed MEASUREMENT job so workspace can load it.
            # Pass meas_job_id so the job ID matches the output directory —
            # the workspace loads CSVs via outputFileUrl(job.id, filename).
            sub_result: dict[str, Any] = {
                "num_organisms": len(df),
                "csv_path": str(output_csv),
                "method": method,
                "image_path": str(image_path),
                "image_stem": image_path.stem,
            }
            if method == "sam" and sam_overlay_path is not None:
                sub_result["overlay_path"] = str(sam_overlay_path)
            sub_job = job_manager.register_completed(
                JobType.MEASUREMENT,
                params={
                    "image_path": str(image_path),
                    "detections_csv": str(detections_csv),
                    "um_per_pixel": um_per_pixel,
                    "method": method,
                },
                result=sub_result,
                job_id=meas_job_id,
            )

            db_projects.set_measurement_job(project_id, image_id, sub_job.id)

            results.append({
                "image_id": image_id,
                "filename": filename,
                "measurement_job_id": sub_job.id,
                "num_organisms": len(df),
            })

        except Exception as e:
            results.append({
                "image_id": image_id,
                "filename": filename,
                "measurement_job_id": None,
                "error": str(e),
                "num_organisms": 0,
            })

        progress_callback(
            (i + 1) / total,
            f"Done {filename} ({i + 1}/{total})",
        )

    measured = sum(1 for r in results if r.get("measurement_job_id"))
    return {
        "project_id": project_id,
        "results": results,
        "num_measured": measured,
    }
