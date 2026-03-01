"""Batch process service — detect, auto-accept as annotations, then measure each image."""

import csv
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_repo_root = Path(__file__).resolve().parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from backend import db_projects
from backend.config import settings
from backend.jobs.manager import job_manager
from backend.jobs.models import Job, JobType


def _save_as_annotations(image_id: str, filename: str, detections: list, det_job_id: str) -> None:
    """Write all detections to the annotation JSON as accepted boxes."""
    boxes = [
        {
            "id": f"auto_{i}",
            "x1": float(det["x1"]),
            "y1": float(det["y1"]),
            "x2": float(det["x2"]),
            "y2": float(det["y2"]),
            "conf": float(det["conf"]),
            "status": "accepted",
        }
        for i, det in enumerate(detections)
    ]
    ann_data = {
        "image_id": image_id,
        "image_filename": filename,
        "source_job_id": det_job_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "boxes": boxes,
    }
    ann_path = settings.annotations_dir / f"{image_id}.json"
    ann_path.parent.mkdir(parents=True, exist_ok=True)
    ann_path.write_text(json.dumps(ann_data, indent=2))


def _annotations_to_csv(image_id: str) -> Path:
    """Convert saved annotation JSON to a CSV suitable for measure_organisms_fast."""
    ann_path = settings.annotations_dir / f"{image_id}.json"
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
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "width": x2 - x1, "height": y2 - y1,
                "confidence": b.get("conf", 1.0),
                "class": 0,
            })
    return out_path


def run_batch_process(job: Job, progress_callback) -> dict[str, Any]:
    """Detect and measure every image in a project sequentially.

    For each image:
      1. Run tiled YOLO inference → detection CSV + overlay
      2. Save all detections as accepted annotations (writes annotations/{image_id}.json)
      3. Convert annotations to measurement CSV
      4. Run fast ellipse measurement

    Expected job.params:
        project_id: str
        image_entries: list of {image_id, image_path, filename}
        um_per_pixel: float
        conf: float (optional)
        tile_size: int (optional)
        overlap: int (optional)
        device: str (optional)
    """
    from scripts.infer_tiled import infer_tiled
    from scripts.measure_organisms_fast import measure_organisms_fast

    params = job.params
    project_id = params["project_id"]
    image_entries = params["image_entries"]
    um_per_pixel = params["um_per_pixel"]
    total = len(image_entries)

    results = []

    for i, entry in enumerate(image_entries):
        image_id = entry["image_id"]
        image_path = Path(entry["image_path"])
        filename = entry["filename"]
        image_stem = image_path.stem
        base = i / total

        # ── Detection ────────────────────────────────────────────────────────
        progress_callback(base, f"Detecting {filename} ({i + 1}/{total})")

        det_job_id = uuid.uuid4().hex[:12]
        det_dir = settings.outputs_dir / det_job_id

        try:
            detections = infer_tiled(
                image_path=image_path,
                model_path=str(settings.default_model),
                tile_size=params.get("tile_size", settings.default_tile_size),
                overlap=params.get("overlap", settings.default_overlap),
                conf_threshold=params.get("conf", settings.default_conf),
                iou_threshold=params.get("iou", settings.default_iou),
                output_dir=str(det_dir),
                device=params.get("device", settings.default_device),
                progress_callback=None,
            )

            # Use det_job_id as the registered job ID so outputFileUrl resolves correctly
            det_sub_job = job_manager.register_completed(
                JobType.DETECTION,
                params={"image_path": str(image_path)},
                result={
                    "num_detections": len(detections),
                    "csv_path": str(det_dir / f"{image_stem}_detections.csv"),
                    "overlay_path": str(det_dir / f"{image_stem}_overlay.jpg"),
                    "metadata_path": str(det_dir / f"{image_stem}_metadata.json"),
                    "image_path": str(image_path),
                    "image_stem": image_stem,
                },
                job_id=det_job_id,
            )
            db_projects.set_detection_job(project_id, image_id, det_sub_job.id)

            if len(detections) == 0:
                # Save empty annotation file so workspace shows the annotation state
                _save_as_annotations(image_id, filename, [], det_sub_job.id)
                results.append({
                    "image_id": image_id,
                    "filename": filename,
                    "detection_job_id": det_sub_job.id,
                    "measurement_job_id": None,
                    "num_detections": 0,
                    "num_organisms": 0,
                })
                progress_callback((i + 1) / total, f"Done {filename} — no detections ({i + 1}/{total})")
                continue

            # ── Save detections as accepted annotations ───────────────────────
            progress_callback(base + 0.3 / total, f"Saving annotations {filename} ({i + 1}/{total})")
            _save_as_annotations(image_id, filename, detections, det_sub_job.id)
            detections_csv = _annotations_to_csv(image_id)

            # ── Measurement ──────────────────────────────────────────────────
            progress_callback(base + 0.6 / total, f"Measuring {filename} ({i + 1}/{total})")

            meas_job_id = uuid.uuid4().hex[:12]
            meas_dir = settings.outputs_dir / meas_job_id
            meas_dir.mkdir(parents=True, exist_ok=True)
            meas_csv = meas_dir / f"{image_stem}_measurements.csv"

            df = measure_organisms_fast(
                image_path=image_path,
                detections_csv=detections_csv,
                output_csv=meas_csv,
                um_per_pixel=um_per_pixel,
                progress_callback=None,
            )

            # Use meas_job_id as the registered job ID so outputFileUrl resolves correctly
            meas_sub_job = job_manager.register_completed(
                JobType.MEASUREMENT,
                params={
                    "image_path": str(image_path),
                    "detections_csv": str(detections_csv),
                    "um_per_pixel": um_per_pixel,
                    "method": "fast",
                },
                result={
                    "num_organisms": len(df),
                    "csv_path": str(meas_csv),
                    "method": "fast",
                    "image_path": str(image_path),
                    "image_stem": image_stem,
                },
                job_id=meas_job_id,
            )
            db_projects.set_measurement_job(project_id, image_id, meas_sub_job.id)

            results.append({
                "image_id": image_id,
                "filename": filename,
                "detection_job_id": det_sub_job.id,
                "measurement_job_id": meas_sub_job.id,
                "num_detections": len(detections),
                "num_organisms": len(df),
            })

        except Exception as e:
            import traceback
            traceback.print_exc()
            results.append({
                "image_id": image_id,
                "filename": filename,
                "error": str(e),
            })

        progress_callback((i + 1) / total, f"Done {filename} ({i + 1}/{total})")

    processed = sum(1 for r in results if r.get("measurement_job_id"))
    return {
        "project_id": project_id,
        "results": results,
        "num_processed": processed,
    }
