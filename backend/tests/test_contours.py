import asyncio
import csv
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np
import openpyxl
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.auth import get_current_user
from backend.config import settings
from backend.jobs.models import Job, JobStatus, JobType
from backend.routers import contours as router_module
from backend.services import contours as service
from scripts.measure_organisms import measure_organism_from_mask


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    for name in ["outputs_dir", "uploads_dir", "annotations_dir", "calibration_dir"]:
        path = tmp_path / name
        path.mkdir()
        monkeypatch.setattr(settings, name, path)
    return tmp_path


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
def api(isolated, monkeypatch):
    manager = Manager()
    monkeypatch.setattr(router_module, "job_manager", manager)
    monkeypatch.setattr(router_module, "validate_checkpoint", lambda: Path("fixed.pt"))
    monkeypatch.setattr(router_module, "get_image_info", lambda key: {"path": str(isolated / "image.png"), "image_id": key})
    app = FastAPI()
    app.include_router(router_module.router)
    app.dependency_overrides[get_current_user] = lambda: "tester"
    with TestClient(app) as client:
        yield client, manager


def test_no_image_has_clear_error_and_never_submits(api, monkeypatch):
    client, manager = api
    monkeypatch.setattr(router_module, "get_image_info", lambda key: None)
    response = client.post("/api/contours/run", json={"image_id": "missing", "um_per_pixel": None})
    assert response.status_code == 404
    assert "Fotografija" in response.json()["detail"]
    assert not manager.submissions


def test_fixed_call_and_pending_duplicate_are_coalesced_without_calibration_default(api):
    client, manager = api
    first = client.post("/api/contours/run", json={"image_id": "photo", "um_per_pixel": None})
    second = client.post("/api/contours/run", json={"image_id": "photo", "um_per_pixel": None})
    assert first.status_code == second.status_code == 200
    assert first.json()["job_id"] == second.json()["job_id"]
    assert len(manager.submissions) == 1
    job = manager.submissions[0]
    assert job.type == JobType.CONTOURS
    assert job.params["um_per_pixel"] is None
    assert set(job.params) == {"image_id", "image_path", "um_per_pixel"}


@pytest.mark.parametrize("value", [0, -1, True, "8.57"])
def test_invalid_calibration_is_rejected(api, value):
    client, manager = api
    assert client.post("/api/contours/run", json={"image_id": "photo", "um_per_pixel": value}).status_code == 422
    assert not manager.submissions


def test_inference_overrides_are_not_exposed(api):
    client, manager = api
    assert client.post("/api/contours/run", json={"image_id": "photo", "conf": .1}).status_code == 422
    assert not manager.submissions


def test_other_photo_or_nonstandard_detections_are_rejected(api, isolated):
    client, manager = api
    path = settings.outputs_dir / "detections.csv"
    path.write_text("x1,y1,x2,y2,confidence\n1,1,10,10,.9\n")
    detection = Job(id="detection", type=JobType.DETECTION, status=JobStatus.COMPLETED,
                    params={"conf": .6, "iou": .5, "tile_size": 1280, "overlap": 256},
                    result={"image_path": str(isolated / "other.png"), "csv_path": str(path)})
    manager.jobs[detection.id] = detection
    assert client.post("/api/contours/run", json={"image_id": "photo", "detection_job_id": detection.id}).status_code == 400
    detection.result["image_path"] = str(isolated / "image.png")
    detection.params["conf"] = .2
    assert client.post("/api/contours/run", json={"image_id": "photo", "detection_job_id": detection.id}).status_code == 400
    detection.params["conf"] = .6
    assert client.post("/api/contours/run", json={"image_id": "photo", "detection_job_id": detection.id}).status_code == 200


def test_sha_mismatch_blocks_request(api, monkeypatch):
    client, manager = api
    def invalid():
        raise RuntimeError("SHA-256 mismatch")
    monkeypatch.setattr(router_module, "validate_checkpoint", invalid)
    response = client.post("/api/contours/run", json={"image_id": "photo"})
    assert response.status_code == 503
    assert not manager.submissions


def test_real_checkpoint_verifier_rejects_wrong_bytes(isolated, monkeypatch):
    path = isolated / "wrong.pt"
    path.write_bytes(b"wrong model")
    monkeypatch.setattr(settings, "exp03_checkpoint", path)
    with pytest.raises(RuntimeError, match="SHA-256"):
        service.validate_checkpoint()


def mask_record(index, mask, *, kept=True, available=True):
    path = settings.outputs_dir / f"mask-{index}.png"
    cv2.imwrite(str(path), mask)
    return {"detection_id": index, "mask_path": str(path), "crop_box": [100, 200, 100 + mask.shape[1], 200 + mask.shape[0]],
            "kept_after_dedup": kept, "has_mask": available, "segmentation_confidence": .9}


def test_only_kept_nonempty_masks_use_existing_axes_and_cylinder_formula(isolated):
    mask = np.zeros((40, 30), dtype=np.uint8)
    mask[5:34, 7:22] = 255
    rows = [mask_record(17, mask), mask_record(18, mask, kept=False), mask_record(19, np.zeros_like(mask), available=False)]
    measured, summary = service.measure_final_masks(rows, 8.57)
    assert len(measured) == summary["count"] == 1
    assert measured[0]["detection_id"] == 17
    existing = measure_organism_from_mask(mask > 0, 8.57, [0, 0, 30, 40])
    assert measured[0]["length_px"] == existing["major_axis_px"]
    assert measured[0]["width_px"] == existing["minor_axis_px"]
    assert measured[0]["volume_mm3"] == pytest.approx(existing["volume_um3"] / 1e9)
    assert measured[0]["volume_mm3"] == pytest.approx(np.pi * (measured[0]["width_mm"] / 2) ** 2 * measured[0]["length_mm"])
    assert summary["total"] == measured[0]["volume_mm3"]
    assert measured[0]["centroid_x_px"] == existing["centroid_x_px"] + 100


def test_no_calibration_has_pixel_measures_and_no_invented_physical_volume(isolated):
    mask = np.zeros((30, 25), dtype=np.uint8)
    mask[:20, 4:18] = 255
    measured, summary = service.measure_final_masks([mask_record(9, mask)], None)
    assert measured[0]["length_px"] > 0
    assert measured[0]["touches_crop_edge"]
    assert measured[0]["volume_mm3"] is None
    assert measured[0]["length_mm"] is None
    assert measured[0]["calibration_um_per_px"] is None
    assert not summary["available"] and summary["total"] is None


@pytest.mark.parametrize("calibration", [None, 10.])
def test_csv_excel_exports_include_volume_unit_and_do_not_write_gt(isolated, calibration):
    mask = np.zeros((30, 25), dtype=np.uint8)
    mask[3:25, 5:20] = 255
    records = [mask_record(27, mask)]
    manual = settings.annotations_dir / "existing.json"
    manual.write_text('{"preserve":"original human work"}')
    before = manual.read_bytes()
    job = Job(id="exports", type=JobType.CONTOURS, params={"image_id": "photo", "image_path": str(isolated / "image.png"), "um_per_pixel": calibration})
    output = settings.outputs_dir / job.id
    output.mkdir()
    overlay = output / "exp03_overlay.jpg"
    cv2.imwrite(str(overlay), np.zeros((30, 25, 3), dtype=np.uint8))
    result = service._write_measurements(job, records, [], overlay, None, {})
    with Path(result["csv_path"]).open(encoding="utf-8-sig", newline="") as handle:
        exported = list(csv.DictReader(handle))[0]
    if calibration is None:
        assert exported["volume_mm3"] == ""
        assert "bez kalibracije" in exported["volume_unit"]
    else:
        assert float(exported["volume_mm3"]) > 0
        assert exported["volume_unit"] == "mm³"
    book = openpyxl.load_workbook(result["excel_path"], read_only=True)
    try:
        values = list(book["Measurements"].values)
        assert "volume_mm3" in values[0] and "volume_unit" in values[0]
        volume = values[1][values[0].index("volume_mm3")]
        assert volume is None if calibration is None else volume > 0
    finally:
        book.close()
    assert manual.read_bytes() == before and len(list(settings.annotations_dir.iterdir())) == 1


def test_measurement_only_reuses_masks_without_detector_or_segmentation(isolated, monkeypatch):
    from backend.jobs import manager as manager_module
    mask = np.zeros((30, 25), dtype=np.uint8)
    mask[4:24, 6:19] = 255
    original = Job(id="original", type=JobType.CONTOURS, status=JobStatus.COMPLETED)
    directory = settings.outputs_dir / original.id
    directory.mkdir()
    records = directory / "records.json"
    records.write_text(json.dumps([mask_record(17, mask)]))
    contours = directory / "contours.json"
    contours.write_text('{"organisms":[]}')
    overlay = directory / "overlay.jpg"
    cv2.imwrite(str(overlay), np.zeros((30, 25, 3), dtype=np.uint8))
    original.result = {"image_id": "photo", "model_name": service.MODEL_NAME, "checkpoint_sha256": service.EXP03_SHA256,
                       "mask_records_path": str(records), "contours_path": str(contours), "overlay_path": str(overlay), "pipeline": {}}
    monkeypatch.setattr(manager_module.job_manager, "get", lambda key: original)
    monkeypatch.setattr(service, "validate_checkpoint", lambda: Path("fixed.pt"))
    monkeypatch.setattr(service, "_legacy_pipeline", Mock(side_effect=AssertionError("Inference must not run")))
    job = Job(id="recalibrated", type=JobType.CONTOURS, params={"image_id": "photo", "image_path": str(isolated / "image.png"), "source_contour_job_id": original.id, "um_per_pixel": 10.})
    result = service.run_contours(job, lambda *args: None)
    assert result["num_organisms"] == 1 and result["volume_summary"]["count"] == 1
    assert result["measurements"][0]["calibration_um_per_px"] == 10.


def test_app_startup_checks_sha_before_database_initialization(monkeypatch):
    import backend.main as main
    db_init = Mock()
    monkeypatch.setattr(main.db, "init_db", db_init)
    monkeypatch.setattr(main, "validate_checkpoint", Mock(side_effect=RuntimeError("checkpoint SHA failure")))
    async def startup():
        async with main.lifespan(main.app):
            pass
    with pytest.raises(RuntimeError, match="SHA failure"):
        asyncio.run(startup())
    db_init.assert_not_called()


def test_degenerate_mask_does_not_invent_zero_physical_volume(isolated):
    mask = np.zeros((5, 5), dtype=np.uint8)
    mask[2, 2] = 255
    rows, summary = service.measure_final_masks([mask_record(3, mask)], 8.57)
    assert rows[0]["volume_mm3"] is None
    assert "Degenerirane" in rows[0]["warning"]
    assert summary["count"] == 0 and summary["total"] is None


def test_worker_reuses_detector_crops_dedup_and_measures_only_final_masks(isolated, monkeypatch):
    import ultralytics
    from backend.jobs import manager as manager_module
    from backend.services import detection as detection_service

    image_path = isolated / "photo.png"
    cv2.imwrite(str(image_path), np.zeros((100, 100, 3), dtype=np.uint8))
    csv_path = settings.outputs_dir / "original_detections.csv"
    csv_path.write_text("x1,y1,x2,y2,width,height,confidence,class\n10.4,12.4,30.1,40.1,19.7,27.7,.9,0\n40,15,60,45,20,30,.8,0\n65,20,85,50,20,30,.7,0\n")
    source = Job(id="detector", type=JobType.DETECTION, status=JobStatus.COMPLETED, result={"csv_path": str(csv_path)})
    monkeypatch.setattr(manager_module.job_manager, "get", lambda key: source)
    detector = Mock(side_effect=AssertionError("Existing detector must not run again"))
    monkeypatch.setattr(detection_service, "run_detection", detector)
    checkpoint = isolated / "fixed.pt"
    checkpoint.write_bytes(b"synthetic checkpoint used only with mocked network")
    monkeypatch.setattr(service, "validate_checkpoint", lambda: checkpoint)
    original_sha = service.file_sha256
    monkeypatch.setattr(service, "file_sha256", lambda path: service.EXP03_SHA256 if Path(path) == checkpoint else original_sha(Path(path)))
    detector_file = isolated / "detector.pt"
    detector_file.write_bytes(b"unchanged detector")
    monkeypatch.setattr(settings, "default_model", detector_file)
    classifier_file = isolated / "classifier.joblib"
    classifier_file.write_bytes(b"unchanged classifier")
    calls = []
    class Network:
        def __init__(self, path):
            assert Path(path) == checkpoint
        def predict(self, crop, **kwargs):
            calls.append((crop.shape, kwargs))
            return [(np.ones(crop.shape[:2], dtype=np.uint8) * 255, .9)] if len(calls) < 3 else [(None, None)]
    monkeypatch.setattr(ultralytics, "YOLO", Network)
    dedup = Mock(side_effect=lambda rows, payload: ([rows[0], rows[2]], [], {"removed_by_classifier_dedup": 1}))
    legacy = SimpleNamespace(PARAMS={"seg_imgsz": 320, "seg_conf": .15, "device": "cpu"}, DEDUP_MODEL=classifier_file,
        result_mask=lambda result, shape: result, apply_classifier_dedup=dedup, joblib=SimpleNamespace(load=lambda path: {}),
        draw_full_overlay=lambda image, rows: image.copy())
    monkeypatch.setattr(service, "_legacy_pipeline", lambda: legacy)
    result = service.run_contours(Job(id="worker", type=JobType.CONTOURS, params={
        "image_id": "photo", "image_path": str(image_path), "detection_job_id": source.id, "um_per_pixel": 8.57,
    }), lambda *args: None)
    detector.assert_not_called()
    dedup.assert_called_once()
    assert len(calls) == 3
    assert calls[0][0] == (29, 21, 3)  # Legacy floor/ceil, no additional padding.
    assert all(kwargs == {"imgsz": 320, "conf": .15, "device": "cpu", "verbose": False} for _, kwargs in calls)
    assert [row["detection_id"] for row in result["measurements"]] == [0]
    assert result["num_organisms"] == result["volume_summary"]["count"] == 1
    assert result["pipeline"]["kept_boxes"] == 2 and result["pipeline"]["without_mask"] == 1
    assert result["measurements"][0]["touches_crop_edge"]
    assert not result["gt_saved"] and not list(settings.annotations_dir.iterdir())
