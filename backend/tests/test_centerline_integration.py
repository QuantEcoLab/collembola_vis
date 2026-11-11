import csv
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from backend.config import settings
from backend.jobs.models import Job, JobStatus, JobType
from backend.services import contours, measurement, batch_measurement
from backend.services.centerline_measurement import METHOD


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    for name in ("outputs_dir", "annotations_dir"):
        path = tmp_path / name
        path.mkdir()
        monkeypatch.setattr(settings, name, path)
    return tmp_path


def record(index, mask):
    path = settings.outputs_dir / f"mask-{index}.png"
    cv2.imwrite(str(path), mask.astype(np.uint8) * 255)
    return {"detection_id": index, "crop_box": [0, 0, mask.shape[1], mask.shape[0]],
            "mask_path": str(path), "has_mask": True, "kept_after_dedup": True, "segmentation_confidence": .9}


def test_unreliable_candidates_keep_diagnostics_but_never_enter_any_volume_statistic(isolated):
    y, x = np.mgrid[:120, :180]
    normal = (x - np.clip(x, 40, 130))**2 + (y - 60)**2 <= 12**2
    edge = normal[55:, :]
    multi = normal.copy()
    multi[3:7, 3:7] = True
    geometry = []
    rows, summary = contours.measure_final_masks([record(0, normal), record(1, edge), record(2, multi)], 8., geometries=geometry)
    assert rows[0]["measurement_reliable"] and rows[0]["volume_included"]
    assert all(not row["volume_included"] and not row["measurement_reliable"] for row in rows[1:])
    assert all(row["volume_mm3"] is not None for row in rows)  # Candidates stay visible.
    assert summary["count"] == summary["reliable_count"] == 1
    assert summary["excluded_count"] == 2
    assert all(summary[key] == rows[0]["volume_mm3"] for key in ("total", "mean", "median", "min", "max"))
    assert geometry[0]["centerline"] and geometry[0]["cross_sections"]
    assert all(not section["volume_included"] for g in geometry[1:] for section in g["cross_sections"])
    assert all(r["measurement_method"] == METHOD for r in rows)


@pytest.mark.parametrize("old_method", ["fast", "sam"])
def test_legacy_api_method_names_cannot_select_old_measurement_algorithms(isolated, monkeypatch, old_method):
    monkeypatch.setattr(contours, "find_saved_source", lambda image_id, **filters: None)
    dispatch = Mock(return_value={"measurement_method": METHOD})
    monkeypatch.setattr(contours, "run_contours", dispatch)
    job = Job(id="compat", type=JobType.MEASUREMENT, params={"image_id": "photo", "image_path": "photo.png", "method": old_method, "um_per_pixel": 8.})
    assert measurement.run_measurement(job, lambda *a: None)["measurement_method"] == METHOD
    dispatch.assert_called_once()
    assert dispatch.call_args.args[0] is job


def test_batch_uses_same_method_and_never_overwrites_saved_human_annotations(isolated, monkeypatch):
    ann = settings.annotations_dir / "photo.json"
    original = json.dumps({"image_id": "photo", "image_filename": "photo.png", "source_job_id": "det", "saved_at": "2026-11-11T13:00:00+00:00",
                           "boxes": [{"id": "added-1", "x1": 1, "y1": 1, "x2": 9, "y2": 9, "conf": 1, "status": "added"}]}).encode()
    ann.write_bytes(original)
    run = Mock(return_value={"measurement_method": METHOD, "num_organisms": 2, "volume_summary": {"count": 1, "excluded_count": 1}})
    monkeypatch.setattr(batch_measurement, "run_measurement", run)
    monkeypatch.setattr(batch_measurement, "job_manager", SimpleNamespace(register_completed=lambda *a, **kw: SimpleNamespace(id=kw["job_id"])))
    db = Mock()
    monkeypatch.setattr(batch_measurement.db_projects, "set_measurement_job", db)
    job = Job(id="batch", type=JobType.BATCH_MEASURE, params={"project_id": "project", "um_per_pixel": 8.,
        "image_entries": [{"image_id": "photo", "image_path": "photo.png", "filename": "photo.png"}]})
    result = batch_measurement.run_batch_measurement(job, lambda *a: None)
    assert result["num_measured"] == 1 and ann.read_bytes() == original
    assert run.call_args.args[0].params["use_annotations"]
    db.assert_called_once()


def test_changed_annotations_invalidate_old_mask_reuse_without_altering_human_file(isolated, monkeypatch):
    from backend.jobs import manager
    saved = {"image_id": "photo", "image_filename": "photo.png", "source_job_id": "det", "boxes": [
        {"id": "box-0", "x1": 1, "y1": 1, "x2": 12, "y2": 14, "conf": .9, "status": "accepted"},
        {"id": "box-1", "x1": 20, "y1": 1, "x2": 32, "y2": 14, "conf": .8, "status": "rejected"},
        {"id": "added-1", "x1": 40, "y1": 1, "x2": 52, "y2": 14, "conf": 1, "status": "added"}]}
    path = settings.annotations_dir / "photo.json"
    path.write_text(json.dumps(saved));before = path.read_bytes()
    snapshot, digest = contours.annotation_snapshot("photo")
    old = SimpleNamespace(result={"pipeline": {"annotations_sha256": "different"}})
    monkeypatch.setattr(manager.job_manager, "get", lambda key: old)
    monkeypatch.setattr(contours, "validate_checkpoint", lambda: Path("fixed.pt"))
    monkeypatch.setattr(contours, "_legacy_pipeline", Mock(side_effect=AssertionError("fresh segmentation required")))
    job = Job(id="changed", type=JobType.CONTOURS, params={"image_id": "photo", "image_path": "photo.png",
        "source_contour_job_id": "old", "annotation_snapshot": snapshot, "annotations_sha256": digest})
    with pytest.raises(AssertionError, match="fresh segmentation required"):
        contours.run_contours(job, lambda *a: None)
    assert "source_contour_job_id" not in job.params
    with Path(job.params["detections_csv"]).open() as f:
        rows = list(csv.DictReader(f))
    assert [r["annotation_id"] for r in rows] == ["box-0", "added-1"]
    assert path.read_bytes() == before
