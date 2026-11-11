"""Batch detection service — runs tiled inference for each image in a project sequentially."""

import sys
from pathlib import Path
from typing import Any

_repo_root = Path(__file__).resolve().parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from backend import db_projects
from backend.config import settings
from backend.jobs.manager import job_manager
from backend.jobs.models import Job, JobType
from backend.services.cuda_guard import ensure_inference_device
from backend.services.detection import detection_profile
from backend.services.image_identity import capture_identity


def run_batch_detection(job: Job, progress_callback) -> dict[str, Any]:
    """Execute tiled inference for each image in a project sequentially.

    Expected job.params:
        project_id: str
        image_entries: list of {image_id, image_path, filename}
        conf: float (optional)
        tile_size: int (optional)
        overlap: int (optional)
        device: str (optional)
    """
    from scripts.infer_tiled import infer_tiled

    params = job.params
    project_id = params["project_id"]
    image_entries = params["image_entries"]
    total = len(image_entries)
    device = params.get("device", settings.default_device)
    ensure_inference_device(device)
    profile = detection_profile(params)

    results = []

    for i, entry in enumerate(image_entries):
        image_id = entry["image_id"]
        image_path = Path(entry["image_path"])
        filename = entry["filename"]

        progress_callback(
            i / total,
            f"Detecting {filename} ({i + 1}/{total})",
        )

        # Use a sub-job-id as the output dir so workspace can later find the CSV
        import uuid
        det_job_id = uuid.uuid4().hex[:12]
        output_dir = settings.outputs_dir / det_job_id

        try:
            identity = capture_identity(image_id, image_path)
            detections = infer_tiled(
                image_path=image_path,
                model_path=profile["model_path"],
                tile_size=profile["tile_size"],
                overlap=profile["overlap"],
                conf_threshold=profile["conf"],
                iou_threshold=profile["iou"],
                output_dir=str(output_dir),
                device=device,
                progress_callback=None,
                extra_tile_sizes=profile["extra_tile_sizes"],
                extra_conf_thresholds=profile["extra_conf"],
                contain_threshold=profile["contain"],
                contain_keep_confident=profile["contain_keep_confident"],
            )

            image_stem = image_path.stem
            csv_path = output_dir / f"{image_stem}_detections.csv"
            overlay_path = output_dir / f"{image_stem}_overlay.jpg"
            metadata_path = output_dir / f"{image_stem}_metadata.json"

            if capture_identity(image_id, image_path) != identity:
                raise ValueError("Fotografija je promijenjena tijekom detekcije")
            # Register as a completed DETECTION job so workspace can load it.
            # Pass det_job_id so the job ID matches the output directory.
            sub_job = job_manager.register_completed(
                JobType.DETECTION,
                params={"image_path": str(image_path), "image_id": image_id, "image_identity": identity},
                result={
                    "image_id": image_id,
                    "image_identity": identity,
                    "num_detections": len(detections),
                    "csv_path": str(csv_path),
                    "overlay_path": str(overlay_path),
                    "metadata_path": str(metadata_path),
                    "image_path": str(image_path),
                    "image_stem": image_stem,
                    "detector_profile": profile,
                },
                job_id=det_job_id,
            )

            # Persist detection_job_id in project_images
            db_projects.set_detection_job(project_id, image_id, sub_job.id)

            results.append({
                "image_id": image_id,
                "filename": filename,
                "detection_job_id": sub_job.id,
                "num_detections": len(detections),
            })

        except Exception as e:
            results.append({
                "image_id": image_id,
                "filename": filename,
                "detection_job_id": None,
                "error": str(e),
                "num_detections": 0,
            })

        progress_callback(
            (i + 1) / total,
            f"Done {filename} ({i + 1}/{total})",
        )

    return {
        "project_id": project_id,
        "results": results,
    }
