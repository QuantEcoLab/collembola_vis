"""A newer detection run reaches the user's photos: unreviewed annotation files never hide it.

Annotation files without ``saved_at`` were written from raw detections at project import (no human
review). They must not keep a photo on the old detector's boxes once a newer detection job is current,
in the workspace (useRefinement, frontend tests), in project measurement, or through reuse of masks
segmented from older boxes. Human-saved files always win, and a newer job can be merged into them
while keeping the reviewer's added and rejected boxes.
"""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.auth import get_current_user
from backend.config import settings
from backend.jobs.models import Job, JobStatus, JobType
from backend.routers import annotations as annotations_router
from backend.routers import contours as contours_router
from backend.services import annotation_state, batch_measurement, measurement
from backend.services import contours as contour_service
from backend.services.annotation_state import annotations_take_precedence, merge_detections


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    for name in ("outputs_dir", "annotations_dir", "uploads_dir"):
        path = tmp_path / name
        path.mkdir()
        monkeypatch.setattr(settings, name, path)
    return tmp_path


def box(id_, x1, status="accepted", y1=10, size=40, conf=.9):
    return {"id": id_, "x1": float(x1), "y1": float(y1), "x2": float(x1 + size), "y2": float(y1 + size), "conf": conf,
            "status": status}


def annotation(boxes, source="old-job", saved=True, image_id="photo"):
    data = {"image_id": image_id, "image_filename": "photo.jpg", "source_job_id": source,
            "created_at": "2026-11-11T12:18:34+00:00", "boxes": boxes}
    if saved:
        data["saved_at"] = "2026-11-11T13:00:00+00:00"
    return data


def write_annotation(data):
    path = settings.annotations_dir / f"{data['image_id']}.json"
    path.write_text(json.dumps(data))
    return path


def detection_job(job_id, rows, image_id="photo"):
    out = settings.outputs_dir / job_id
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "photo_detections.csv"
    csv_path.write_text("x1,y1,x2,y2,width,height,confidence,class\n" + "".join(
        f"{b['x1']},{b['y1']},{b['x2']},{b['y2']},{b['x2'] - b['x1']},{b['y2'] - b['y1']},{b['conf']},0\n" for b in rows))
    return Job(id=job_id, type=JobType.DETECTION, status=JobStatus.COMPLETED, params={"image_id": image_id},
               result={"image_id": image_id, "csv_path": str(csv_path), "image_path": f"data/uploads/{image_id}/photo.jpg"})


# ── which boxes a photo uses ────────────────────────────────────────────────────────────────────


def test_human_saved_annotations_always_win():
    assert annotations_take_precedence(annotation([box("a", 0)], source="old-job"), "new-job")
    assert annotations_take_precedence(annotation([box("a", 0)]), None)


def test_unreviewed_annotation_file_is_superseded_by_a_newer_detection_job():
    auto = annotation([box("box-0", 0)], source="old-job", saved=False)
    assert not annotations_take_precedence(auto, "new-job")
    assert annotations_take_precedence(auto, "old-job")  # same boxes as its own job
    assert annotations_take_precedence(auto, None)  # nothing else to show
    assert not annotations_take_precedence({**auto, "saved_at": None}, "new-job")
    assert not annotations_take_precedence(None, "new-job")


# ── merging a newer job into reviewed annotations ──────────────────────────────────────────────


def test_merge_keeps_reviewer_decisions_and_replaces_model_boxes():
    reviewed = annotation([
        box("box-0", 0),                       # accepted, re-detected below -> replaced by the new box
        box("box-1", 100),                     # accepted, not re-detected -> kept (a reviewer saw it)
        box("box-2", 200, "rejected"),         # rejected texture blob -> kept as rejected
        box("added-x", 300, "added"),          # drawn by the reviewer -> kept
        box("box-3", 400, size=60),            # accepted animal ...
    ])
    new = [box("box-0", 2), box("box-1", 202), box("box-2", 303), box("box-3", 500), box("box-4", 405, size=50)]
    boxes, summary = merge_detections(reviewed, new)
    by_id = {b["id"]: b for b in boxes}
    assert len(by_id) == len(boxes), "ids must stay unique after the merge"
    assert by_id["box-0"]["x1"] == 2 and by_id["box-0"]["status"] == "accepted"
    assert "box-1" not in by_id and by_id["box-3"]["x1"] == 500     # new boxes on rejected/added objects dropped
    assert by_id["box-4"]["x1"] == 405                               # covers old box-3 (inside)
    assert by_id["prev-box-1"]["x1"] == 100 and by_id["prev-box-1"]["status"] == "accepted"
    assert by_id["prev-box-2"]["status"] == "rejected" and by_id["added-x"]["status"] == "added"
    assert "prev-box-3" not in by_id and "prev-box-0" not in by_id
    assert summary == {"human_saved": True, "new_detections": 5, "new_kept": 3, "new_dropped_on_rejected": 1,
                       "new_dropped_on_added": 1, "new_dropped_inside_previous": 0, "previous_accepted": 3,
                       "previous_accepted_kept": 1, "added_kept": 1, "rejected_kept": 1}


def test_merge_a_partial_new_box_does_not_displace_a_reviewed_box():
    """A smaller new box lying inside a reviewed accepted box (a body part) is dropped; the reviewed box stays."""
    full = {"id": "box-5", "x1": 0., "y1": 10., "x2": 100., "y2": 50., "conf": .9, "status": "accepted"}
    reviewed = annotation([full])
    partial = {"id": "box-0", "x1": 70., "y1": 15., "x2": 100., "y2": 45., "conf": .95, "status": "accepted"}
    elsewhere = box("box-1", 300)
    boxes, summary = merge_detections(reviewed, [partial, elsewhere])
    by_id = {b["id"]: b for b in boxes}
    assert set(by_id) == {"prev-box-5", "box-1"} and by_id["prev-box-5"]["x2"] == 100.
    assert summary["new_dropped_inside_previous"] == 1 and summary["previous_accepted_kept"] == 1
    # a new box that contains the reviewed box (or matches it) still replaces it
    bigger = {"id": "box-2", "x1": 0., "y1": 5., "x2": 110., "y2": 55., "conf": .9, "status": "accepted"}
    boxes, summary = merge_detections(reviewed, [bigger])
    assert [b["id"] for b in boxes] == ["box-2"] and summary["previous_accepted_kept"] == 0


def test_merge_into_an_unreviewed_file_takes_only_the_new_boxes():
    auto = annotation([box("box-0", 0), box("box-1", 100)], saved=False)
    boxes, summary = merge_detections(auto, [box("box-0", 500)])
    assert [b["id"] for b in boxes] == ["box-0"] and boxes[0]["x1"] == 500
    assert summary["previous_accepted_kept"] == 0 and not summary["human_saved"]


@pytest.fixture
def annotations_api(isolated, monkeypatch):
    jobs = {}
    monkeypatch.setattr(annotations_router, "job_manager", SimpleNamespace(get=lambda key: jobs.get(key)))
    app = FastAPI()
    app.include_router(annotations_router.router)
    app.dependency_overrides[get_current_user] = lambda: "tester"
    with TestClient(app) as client:
        yield client, jobs


def test_merge_endpoint_previews_without_writing_and_refuses_other_photos(annotations_api):
    client, jobs = annotations_api
    path = write_annotation(annotation([box("box-0", 0), box("box-1", 200, "rejected"), box("added-1", 300, "added")]))
    before = path.read_bytes()
    jobs["new"] = detection_job("new", [box("b", 1), box("b", 201), box("b", 600)])
    jobs["other"] = detection_job("other", [box("b", 1)], image_id="someone-else")

    body = client.post("/api/annotations/photo/merge-detections", json={"detection_job_id": "new"}).json()
    assert body["source_job_id"] == "new" and body["previous_source_job_id"] == "old-job"
    assert sorted((b["id"], b["status"]) for b in body["boxes"]) == [
        ("added-1", "added"), ("box-0", "accepted"), ("box-2", "accepted"), ("prev-box-1", "rejected")]
    assert path.read_bytes() == before  # a preview only; the editor saves

    assert client.post("/api/annotations/photo/merge-detections", json={"detection_job_id": "other"}).status_code == 400
    assert client.post("/api/annotations/photo/merge-detections", json={"detection_job_id": "missing"}).status_code == 400


# ── project measurement (Process / batch measurement) ─────────────────────────────────────────


def run_batch(monkeypatch, jobs, entry):
    run = Mock(return_value={"measurement_method": "m", "num_organisms": 1, "volume_summary": {}})
    monkeypatch.setattr(batch_measurement, "run_measurement", run)
    monkeypatch.setattr(batch_measurement, "job_manager", SimpleNamespace(
        get=lambda key: jobs.get(key), register_completed=lambda *a, **kw: SimpleNamespace(id=kw["job_id"])))
    monkeypatch.setattr(batch_measurement.db_projects, "set_measurement_job", Mock())
    monkeypatch.setattr(batch_measurement.db_projects, "set_detection_job", Mock())
    job = Job(id="batch", type=JobType.BATCH_MEASURE, params={"project_id": "p", "um_per_pixel": 8., "image_entries": [
        {"image_id": "photo", "image_path": "photo.jpg", "filename": "photo.jpg", **entry}]})
    batch_measurement.run_batch_measurement(job, lambda *a: None)
    return run.call_args.args[0].params


def test_project_measurement_uses_the_newer_job_instead_of_an_unreviewed_file(isolated, monkeypatch):
    write_annotation(annotation([box("box-0", 0)], source="old-job", saved=False))
    jobs = {"new-job": detection_job("new-job", [box("b", 0)])}
    params = run_batch(monkeypatch, jobs, {"detection_job_id": "new-job"})
    assert not params["use_annotations"]
    assert params["detection_job_id"] == "new-job" and params["detections_csv"] == jobs["new-job"].result["csv_path"]


@pytest.mark.parametrize("saved,detection_job_id,expected", [
    (True, "new-job", True),      # reviewer's work wins
    (False, "old-job", True),     # the file is a copy of the current job
    (False, None, True),          # nothing else to measure
    (False, "deleted-job", True),  # current job is gone: keep the file's boxes
    (False, "failed-job", True),   # a failed re-run does not replace anything
])
def test_project_measurement_keeps_annotations_when_they_should_win(isolated, monkeypatch, saved, detection_job_id, expected):
    write_annotation(annotation([box("box-0", 0)], source="old-job", saved=saved))
    jobs = {"new-job": detection_job("new-job", [box("b", 0)]), "old-job": detection_job("old-job", [box("b", 0)]),
            "failed-job": Job(id="failed-job", type=JobType.DETECTION, status=JobStatus.FAILED, result={})}
    params = run_batch(monkeypatch, jobs, {"detection_job_id": detection_job_id} if detection_job_id else {})
    assert params["use_annotations"] is expected


# ── masks segmented from older boxes are not reused for newer boxes ───────────────────────────


def saved_masks(job_id, *, detection_job_id=None, annotations_sha256=None, completed_at="2026-11-11T10:00:00"):
    out = settings.outputs_dir / job_id
    out.mkdir(parents=True, exist_ok=True)
    (out / "exp03_mask_records.json").write_text("[]")
    result = {"image_id": "photo", "model_name": contour_service.MODEL_NAME, "checkpoint_sha256": contour_service.EXP03_SHA256,
              "mask_records_path": str(out / "exp03_mask_records.json"), "detection_job_id": detection_job_id,
              "pipeline": {"annotations_sha256": annotations_sha256}}
    (out / "result.json").write_text(json.dumps({"id": job_id, "status": "completed", "completed_at": completed_at, "result": result}))
    return result


def test_saved_masks_are_found_only_for_the_boxes_being_measured(isolated):
    saved_masks("from-old", detection_job_id="old-job", completed_at="2026-11-11T12:00:00")
    saved_masks("from-new", detection_job_id="new-job", completed_at="2026-11-11T11:00:00")
    saved_masks("from-review", detection_job_id="old-job", annotations_sha256="abc", completed_at="2026-11-11T13:00:00")
    find = contour_service.find_saved_source
    assert find("photo")["id"] == "from-review"  # unfiltered: the latest, as before
    assert find("photo", detection_job_id="new-job")["id"] == "from-new"
    assert find("photo", detection_job_id="old-job")["id"] == "from-old"  # not the masks of the reviewed boxes
    assert find("photo", annotations_sha256="abc")["id"] == "from-review"
    assert find("photo", annotations_sha256="other") is None
    assert find("photo", detection_job_id="never-segmented") is None


def test_measurement_reuses_masks_only_from_the_same_detection_job(isolated, monkeypatch):
    saved_masks("from-old", detection_job_id="old-job")
    dispatch = Mock(return_value={"measurement_method": "m"})
    monkeypatch.setattr(contour_service, "run_contours", dispatch)
    job = Job(id="m", type=JobType.MEASUREMENT, params={"image_id": "photo", "image_path": "photo.jpg",
              "detection_job_id": "new-job", "detections_csv": "x.csv"})
    measurement.run_measurement(job, lambda *a: None)
    assert "source_contour_job_id" not in job.params  # new boxes are segmented, not measured from old masks
    job.params["detection_job_id"] = "old-job"
    measurement.run_measurement(job, lambda *a: None)
    assert job.params["source_contour_job_id"] == "from-old"


class Manager:
    def __init__(self):
        self.jobs = {}
        self.submissions = []

    def get(self, key):
        return self.jobs.get(key)

    def list_jobs(self):
        return list(self.jobs.values())

    def submit(self, kind, params):
        job = Job(id=f"job{len(self.submissions)}", type=kind, params=params)
        self.jobs[job.id] = job
        self.submissions.append(job)
        return job


@pytest.fixture
def contours_api(isolated, monkeypatch):
    manager = Manager()
    monkeypatch.setattr(contours_router, "job_manager", manager)
    monkeypatch.setattr(contours_router, "validate_checkpoint", lambda: Path("fixed.pt"))
    monkeypatch.setattr(contours_router, "get_image_info", lambda key: {"path": str(isolated / "photo.jpg"), "image_id": key,
                                                                         "filename": "photo.jpg"})
    app = FastAPI()
    app.include_router(contours_router.router)
    app.dependency_overrides[get_current_user] = lambda: "tester"
    with TestClient(app) as client:
        yield client, manager


def test_contours_after_a_new_detection_run_segment_the_new_boxes(contours_api, isolated):
    client, manager = contours_api
    from backend.services.detection import detection_profile
    new = detection_job("new-job", [box("b", 0)])
    new.result.update(image_path=str(isolated / "photo.jpg"), detector_profile=detection_profile({}))
    manager.jobs["new-job"] = new
    manager.jobs["masks-old"] = Job(id="masks-old", type=JobType.CONTOURS, status=JobStatus.COMPLETED,
                                    result=saved_masks("masks-old", detection_job_id="old-job"))
    manager.jobs["masks-new"] = Job(id="masks-new", type=JobType.CONTOURS, status=JobStatus.COMPLETED,
                                    result=saved_masks("masks-new", detection_job_id="new-job"))

    stale = client.post("/api/contours/run", json={"image_id": "photo", "detection_job_id": "new-job",
                                                   "source_contour_job_id": "masks-old"})
    assert stale.status_code == 200
    params = manager.submissions[-1].params
    assert params["detection_job_id"] == "new-job" and "source_contour_job_id" not in params
    manager.submissions[-1].status = JobStatus.COMPLETED  # let the next request submit

    same = client.post("/api/contours/run", json={"image_id": "photo", "detection_job_id": "new-job",
                                                  "source_contour_job_id": "masks-new"})
    assert same.status_code == 200
    assert manager.submissions[-1].params["source_contour_job_id"] == "masks-new"


# ── provenance: the detector that produced the measured boxes ──────────────────────────────────


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture
def worker(isolated, monkeypatch):
    """run_contours with the segmentation network, dedup and overlay replaced by fakes."""
    import ultralytics
    from backend.jobs import manager as manager_module

    image_path = isolated / "photo.png"
    cv2.imwrite(str(image_path), np.zeros((100, 100, 3), dtype=np.uint8))
    jobs = {}
    monkeypatch.setattr(manager_module.job_manager, "get", lambda key: jobs.get(key))
    checkpoint = isolated / "fixed.pt"
    checkpoint.write_bytes(b"synthetic")
    monkeypatch.setattr(contour_service, "validate_checkpoint", lambda: checkpoint)
    original_sha = contour_service.file_sha256
    monkeypatch.setattr(contour_service, "file_sha256",
                        lambda path: contour_service.EXP03_SHA256 if Path(path) == checkpoint else original_sha(Path(path)))
    current = isolated / "current_detector.pt"
    current.write_bytes(b"current standard detector")
    monkeypatch.setattr(settings, "default_model", current)
    classifier = isolated / "classifier.joblib"
    classifier.write_bytes(b"classifier")

    class Network:
        def __init__(self, path):
            pass

        def predict(self, crop, **kwargs):
            return [(np.ones(crop.shape[:2], dtype=np.uint8) * 255, .9)]

    monkeypatch.setattr(ultralytics, "YOLO", Network)
    legacy = SimpleNamespace(PARAMS={"seg_imgsz": 320, "seg_conf": .15, "device": "cpu"}, DEDUP_MODEL=classifier,
                             result_mask=lambda result, shape: result,
                             apply_classifier_dedup=lambda rows, payload: (rows, [], {"removed_by_classifier_dedup": 0}),
                             joblib=SimpleNamespace(load=lambda path: {}), draw_full_overlay=lambda image, rows: image.copy())
    monkeypatch.setattr(contour_service, "_legacy_pipeline", lambda: legacy)

    def run(params):
        return contour_service.run_contours(Job(id="contours", type=JobType.CONTOURS, params={
            "image_id": "photo", "image_path": str(image_path), "um_per_pixel": 8., **params}), lambda *a: None)

    return SimpleNamespace(jobs=jobs, run=run, current=current, isolated=isolated)


def old_detector_job(worker, job_id="old-job"):
    old_model = worker.isolated / "old_detector.pt"
    old_model.write_bytes(b"legacy detector")
    job = detection_job(job_id, [box("b", 10, y1=10, size=30)])
    job.result["detector_profile"] = {"model_path": str(old_model), "conf": .6, "iou": .5, "tile_size": 1280, "overlap": 256,
                                      "extra_tile_sizes": [], "extra_conf": [], "contain": None}
    worker.jobs[job_id] = job
    return job, old_model


def test_measured_annotation_boxes_record_the_detector_that_produced_them(worker):
    job, old_model = old_detector_job(worker)
    snapshot = annotation([box("box-0", 10, size=30), box("added-1", 50, "added", size=30)], source=job.id)
    result = worker.run({"annotation_snapshot": snapshot, "annotations_sha256": "digest"})
    assert result["pipeline"]["detector_sha256"] == sha(old_model) != sha(worker.current)
    assert result["pipeline"]["detector_job_id"] == job.id


def test_measured_detections_csv_records_the_job_that_wrote_it(worker):
    job, old_model = old_detector_job(worker)
    result = worker.run({"detections_csv": job.result["csv_path"]})
    assert result["pipeline"]["detector_sha256"] == sha(old_model)
    assert result["pipeline"]["detector_job_id"] == job.id


def test_boxes_of_unknown_origin_record_no_detector(worker):
    csv_path = worker.isolated / "elsewhere.csv"
    csv_path.write_text("x1,y1,x2,y2,confidence\n10,10,40,40,.9\n")
    result = worker.run({"detections_csv": str(csv_path)})
    assert result["pipeline"]["detector_sha256"] is None and result["pipeline"]["detector_job_id"] is None
