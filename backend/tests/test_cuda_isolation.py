"""Regression: a CPU exp03 contour run must not take GPU detection down in the shared backend process.

Production failure (2026-11-11): POST /api/contours/run ran ultralytics with device "cpu" inside the
uvicorn process. ``select_device("cpu")`` sets ``os.environ["CUDA_VISIBLE_DEVICES"] = ""`` process-wide;
torch then enumerated CUDA with no visible device and cached that for the life of the process, so
every later detection with device "0" failed ("Invalid CUDA 'device=0' requested") until restart.

No GPU is needed: ``FakeCudaRuntime`` models torch's enumeration (first query reads the environment,
the answer is cached), and the real ultralytics ``select_device`` does the clobbering.
"""
import asyncio
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np
import pytest
import torch
from ultralytics.utils import torch_utils
from ultralytics.utils.torch_utils import select_device

from backend.config import settings
from backend.jobs.models import Job, JobStatus, JobType
from backend.services import contours as contour_service
from backend.services import detection as detection_service

PHYSICAL_GPUS = 4


class FakeCudaRuntime:
    """torch.cuda without a GPU.

    ``is_available`` reads CUDA_VISIBLE_DEVICES on the first call and caches the device count for the
    rest of the process, like c10::cuda::device_count(). ``device_count`` re-reads the variable, like
    torch's NVML-based count before CUDA is initialised.
    """

    def __init__(self):
        self.cached_count = None

    @staticmethod
    def visible():
        value = os.environ.get("CUDA_VISIBLE_DEVICES")
        if value is None:
            return PHYSICAL_GPUS
        return len([part for part in value.split(",") if part.strip()])

    def is_available(self):
        if self.cached_count is None:
            self.cached_count = self.visible()
        return self.cached_count > 0

    def device_count(self):
        return self.visible()


class FakeUltralyticsModel:
    """Handles ``device`` the way the real predictor does on first use, then returns no boxes."""

    instances = []

    def __init__(self, path):
        self.path = path
        self.devices = []
        FakeUltralyticsModel.instances.append(self)

    def _setup(self, device):
        selected = select_device(device, verbose=False)  # BasePredictor.setup_model
        torch.cuda.is_available()  # AutoBackend: cuda = torch.cuda.is_available() and device.type != "cpu"
        self.devices.append(selected)
        return selected

    def __call__(self, image, conf=None, device=None, verbose=None):  # infer_tiled's detector call
        self._setup(device)
        return []

    def predict(self, crop, **kwargs):  # exp03 segmentation call
        self._setup(kwargs["device"])
        return [(np.ones(crop.shape[:2], dtype=np.uint8) * 255, .9)]


@pytest.fixture
def cuda(monkeypatch, tmp_path):
    """Fresh service process: CUDA_VISIBLE_DEVICES unset (as in collembola.service), CUDA not yet queried."""
    import scripts.infer_tiled as infer_module

    for name in ["outputs_dir", "uploads_dir", "annotations_dir", "calibration_dir"]:
        path = tmp_path / name
        path.mkdir()
        monkeypatch.setattr(settings, name, path)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    runtime = FakeCudaRuntime()
    monkeypatch.setattr(torch.cuda, "is_available", runtime.is_available)
    monkeypatch.setattr(torch.cuda, "device_count", runtime.device_count)
    monkeypatch.setattr(torch_utils, "get_gpu_info", lambda index: "fake GPU")
    monkeypatch.setattr(infer_module, "YOLO", FakeUltralyticsModel)
    monkeypatch.setattr(torch, "set_num_threads", lambda count: None)  # keep the test runner's thread pool
    FakeUltralyticsModel.instances = []
    image_path = tmp_path / "photo.png"
    cv2.imwrite(str(image_path), np.full((100, 100, 3), 30, dtype=np.uint8))
    return SimpleNamespace(runtime=runtime, image_path=image_path, tmp_path=tmp_path)


def contour_job_from_existing_detections(cuda, monkeypatch):
    """Exp03 CPU segmentation of crops from an existing detection job (the path user2 used)."""
    import ultralytics
    from backend.jobs import manager as manager_module

    csv_path = settings.outputs_dir / "original_detections.csv"
    csv_path.write_text("x1,y1,x2,y2,width,height,confidence,class\n10,12,30,40,20,28,.9,0\n")
    source = Job(id="detector", type=JobType.DETECTION, status=JobStatus.COMPLETED, result={"csv_path": str(csv_path)})
    monkeypatch.setattr(manager_module.job_manager, "get", lambda key: source)
    checkpoint = cuda.tmp_path / "fixed.pt"
    checkpoint.write_bytes(b"synthetic checkpoint used only with the fake network")
    monkeypatch.setattr(contour_service, "validate_checkpoint", lambda: checkpoint)
    original_sha = contour_service.file_sha256
    monkeypatch.setattr(contour_service, "file_sha256",
                        lambda path: contour_service.EXP03_SHA256 if Path(path) == checkpoint else original_sha(Path(path)))
    for name in ("detector.pt", "classifier.joblib"):
        (cuda.tmp_path / name).write_bytes(b"unchanged")
    monkeypatch.setattr(settings, "default_model", cuda.tmp_path / "detector.pt")
    monkeypatch.setattr(ultralytics, "YOLO", FakeUltralyticsModel)
    legacy = SimpleNamespace(
        PARAMS={"seg_imgsz": 320, "seg_conf": .15, "device": "cpu"}, DEDUP_MODEL=cuda.tmp_path / "classifier.joblib",
        result_mask=lambda result, shape: result, joblib=SimpleNamespace(load=lambda path: {}),
        apply_classifier_dedup=lambda rows, payload: (rows, [], {"removed_by_classifier_dedup": 0}),
        draw_full_overlay=lambda image, rows: image.copy())
    monkeypatch.setattr(contour_service, "_legacy_pipeline", lambda: legacy)
    return Job(id="contours", type=JobType.CONTOURS, params={
        "image_id": "photo", "image_path": str(cuda.image_path), "detection_job_id": source.id, "um_per_pixel": 8.57})


def gpu_detection(cuda):
    return detection_service.run_detection(Job(id="gpu-detection", type=JobType.DETECTION, params={
        "image_path": str(cuda.image_path), "image_id": "photo", "model_path": "detector.pt", "device": "0"}),
        lambda *args: None)


def test_cpu_contour_segmentation_then_gpu_detection_in_one_process(cuda, monkeypatch):
    job = contour_job_from_existing_detections(cuda, monkeypatch)
    contour_service.run_contours(job, lambda *args: None)
    segmentation = next(model for model in FakeUltralyticsModel.instances if model.path.endswith("fixed.pt"))

    result = gpu_detection(cuda)  # failed in production with "Invalid CUDA 'device=0' requested"

    assert result["num_detections"] == 0
    assert FakeUltralyticsModel.instances[-1].devices[-1] == torch.device("cuda:0")
    assert [device.type for device in segmentation.devices] == ["cpu"]  # exp03 masks stay on CPU
    assert cuda.runtime.cached_count == PHYSICAL_GPUS  # the CPU run never hid the GPUs from torch


def test_cpu_contour_detector_run_keeps_cuda_and_json_params(cuda, monkeypatch):
    """Without a source detection job the contour pipeline runs the tiled detector on CPU itself."""
    import ultralytics
    from backend.jobs import manager as manager_module

    checkpoint = cuda.tmp_path / "fixed.pt"
    checkpoint.write_bytes(b"synthetic")
    monkeypatch.setattr(contour_service, "validate_checkpoint", lambda: checkpoint)
    original_sha = contour_service.file_sha256
    monkeypatch.setattr(contour_service, "file_sha256",
                        lambda path: contour_service.EXP03_SHA256 if Path(path) == checkpoint else original_sha(Path(path)))
    for name in ("detector.pt", "classifier.joblib"):
        (cuda.tmp_path / name).write_bytes(b"unchanged")
    monkeypatch.setattr(settings, "default_model", cuda.tmp_path / "detector.pt")
    monkeypatch.setattr(ultralytics, "YOLO", FakeUltralyticsModel)
    registered = []
    monkeypatch.setattr(manager_module.job_manager, "register_completed",
                        lambda kind, params, result, job_id=None: registered.append(params) or Job(id=job_id, type=kind, params=params, result=result))
    legacy = SimpleNamespace(PARAMS={"detector_conf": .6, "bbox_nms_iou": .5, "tile_size": 1280, "overlap": 256, "device": "cpu",
                                     "seg_imgsz": 320, "seg_conf": .15},
                             DEDUP_MODEL=cuda.tmp_path / "classifier.joblib", draw_full_overlay=lambda image, rows: image.copy())
    monkeypatch.setattr(contour_service, "_legacy_pipeline", lambda: legacy)

    contour_service.run_contours(Job(id="contours", type=JobType.CONTOURS, params={
        "image_id": "photo", "image_path": str(cuda.image_path), "um_per_pixel": None}), lambda *args: None)

    assert gpu_detection(cuda)["num_detections"] == 0
    assert [device.type for device in FakeUltralyticsModel.instances[0].devices] == ["cpu"]
    assert registered[0]["device"] == "cpu"  # persisted job params stay JSON strings
    assert cuda.runtime.cached_count == PHYSICAL_GPUS


def test_requested_cpu_detection_does_not_disable_later_gpu_detection(cuda):
    """The advanced detection dialog offers device "cpu" for ordinary detections too."""
    detection_service.run_detection(Job(id="cpu-detection", type=JobType.DETECTION, params={
        "image_path": str(cuda.image_path), "image_id": "photo", "model_path": "detector.pt", "device": "cpu"}),
        lambda *args: None)
    assert gpu_detection(cuda)["num_detections"] == 0


def test_env_clobbered_by_other_code_after_startup_pin_is_restored(cuda, monkeypatch):
    """Defensive guard: any other library calling select_device('cpu') after startup cannot break detection."""
    from backend.services import cuda_guard

    monkeypatch.setattr(cuda_guard, "STARTUP_CUDA_VISIBLE_DEVICES", None)
    assert cuda_guard.pin_cuda_runtime()  # app startup
    select_device("cpu", verbose=False)  # e.g. a CPU fine-tune in the same process
    torch.cuda.is_available()
    assert os.environ["CUDA_VISIBLE_DEVICES"] == ""

    assert gpu_detection(cuda)["num_detections"] == 0


def test_env_clobbered_before_cuda_was_queried_is_restored_before_detection(cuda, monkeypatch):
    from backend.services import cuda_guard

    monkeypatch.setattr(cuda_guard, "STARTUP_CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    assert gpu_detection(cuda)["num_detections"] == 0
    assert cuda.runtime.cached_count == 1


def test_cuda_hidden_for_the_whole_process_gives_clear_error_without_running_detector(cuda, monkeypatch):
    from backend.services import cuda_guard

    monkeypatch.setattr(cuda_guard, "STARTUP_CUDA_VISIBLE_DEVICES", None)
    select_device("cpu", verbose=False)
    torch.cuda.is_available()  # CUDA enumerated while hidden: no GPU for the rest of this process
    with pytest.raises(RuntimeError, match="ponovno pokrenite servis"):
        gpu_detection(cuda)
    assert not FakeUltralyticsModel.instances


def test_app_startup_pins_cuda_before_any_job_handler_is_registered(monkeypatch):
    import backend.main as main

    calls = []
    monkeypatch.setattr(main, "validate_checkpoint", lambda: calls.append("checkpoint"))
    monkeypatch.setattr(main, "pin_cuda_runtime", lambda: calls.append("pin"))
    monkeypatch.setattr(main.db, "init_db", lambda: None)
    monkeypatch.setattr(main.db_projects, "init_project_db", lambda: None)
    monkeypatch.setattr(main.job_manager, "register_handler", lambda kind, handler: calls.append(kind))

    async def startup():
        async with main.lifespan(main.app):
            pass

    asyncio.run(startup())
    assert calls[:2] == ["checkpoint", "pin"]
    assert JobType.DETECTION in calls[2:]
