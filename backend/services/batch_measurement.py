"""Project measurements use the same centerline workflow as the workspace."""
import uuid
from pathlib import Path

from backend import db_projects
from backend.services.annotation_state import annotations_take_precedence, read_annotation
from backend.jobs.manager import job_manager
from backend.jobs.models import Job, JobStatus, JobType
from backend.services.measurement import run_measurement


def run_batch_measurement(job: Job, progress_callback) -> dict:
    entries = job.params["image_entries"]
    results = []
    for index, entry in enumerate(entries):
        image_id = entry["image_id"]
        source = job_manager.get(entry["detection_job_id"]) if entry.get("detection_job_id") else None
        if source is not None and source.status != JobStatus.COMPLETED:
            source = None
        # Human-saved annotations win; an annotation file without saved_at (written from raw
        # detections at import) is superseded by the photo's current detection job.
        use_annotations = annotations_take_precedence(read_annotation(image_id), source.id if source else None)
        child = Job(id=uuid.uuid4().hex[:12], type=JobType.MEASUREMENT, params={
            "image_id": image_id, "image_path": entry["image_path"],
            "um_per_pixel": job.params.get("um_per_pixel"),
            "use_annotations": use_annotations,
        })
        if source:
            child.params["detections_csv"] = source.result["csv_path"]
            child.params["detection_job_id"] = source.id
        try:
            result = run_measurement(child, lambda p, m: progress_callback((index + p) / max(1, len(entries)), m))
            sub = job_manager.register_completed(JobType.MEASUREMENT, child.params, result, job_id=child.id)
            db_projects.set_measurement_job(job.params["project_id"], image_id, sub.id)
            if result.get("detection_job_id"):
                db_projects.set_detection_job(job.params["project_id"], image_id, result["detection_job_id"])
            results.append({"image_id": image_id, "filename": entry["filename"], "measurement_job_id": sub.id,
                            "num_organisms": result["num_organisms"], "volume_summary": result["volume_summary"]})
        except Exception as exc:
            results.append({"image_id": image_id, "filename": entry["filename"], "measurement_job_id": None,
                            "num_organisms": 0, "error": str(exc)})
    progress_callback(1., "Mjerenja središnje linije dovršena")
    return {"project_id": job.params["project_id"], "results": results,
            "num_measured": sum(bool(r.get("measurement_job_id")) for r in results)}
