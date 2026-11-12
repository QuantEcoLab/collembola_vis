"""Standard detection profile: fine-tuned detector, extra 640-tile pass and fragment suppression."""
import hashlib
import json
import re
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch

import scripts.infer_tiled as infer_module
from backend.config import settings
from backend.jobs.models import Job, JobStatus, JobType
from backend.services import detection as detection_service
from backend.services.detection import detection_profile, legacy_profile, recorded_profile, same_profile


def _old_tile_grid(img_w, img_h, tile_size, overlap):
    """Tile grid of the original tile_image() loop, kept here as the reference."""
    out, stride, y = [], tile_size - overlap, 0
    while y < img_h:
        x = 0
        while x < img_w:
            x_end, y_end = min(x + tile_size, img_w), min(y + tile_size, img_h)
            x_start = max(0, x_end - tile_size) if x_end == img_w else x
            y_start = max(0, y_end - tile_size) if y_end == img_h else y
            out.append((x_start, y_start, x_end, y_end))
            if x_end == img_w:
                break
            x += stride
        if y_end == img_h:
            break
        y += stride
    return out


@pytest.mark.parametrize("size", [(10493, 10365), (10446, 10026), (1280, 1280), (900, 700), (2049, 1281)])
@pytest.mark.parametrize("tile", [(1280, 256), (640, 256)])
def test_tile_grid_is_unchanged(size, tile):
    assert infer_module.tile_positions(*size, *tile) == _old_tile_grid(*size, *tile)


def box(x1, y1, x2, y2, conf=0.9):
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "conf": conf, "class": 0}


def test_fragment_inside_a_larger_box_is_dropped_but_neighbours_are_kept():
    animal = box(100, 100, 200, 160, .7)
    fragment = box(150, 110, 195, 150, .95)        # 100% inside the animal box, higher confidence
    touching = box(190, 100, 260, 150, .8)         # overlaps the animal but is mostly outside it
    separate = box(400, 400, 450, 450, .5)
    kept = infer_module.suppress_contained([animal, fragment, touching, separate], 0.8)
    assert kept == [animal, touching, separate]
    assert infer_module.suppress_contained([animal, fragment], None) == [animal, fragment]


def test_keep_confident_spares_animals_inside_a_box_spanning_two_touching_animals():
    pair = box(100, 100, 220, 160, .6)              # one box over two touching animals
    left = box(100, 100, 160, 160, .9)              # each animal's own box scores higher
    right = box(158, 100, 220, 160, .85)
    fragment = box(105, 105, 150, 150, .5)          # a body part of the left animal scores lower
    kept = infer_module.suppress_contained([pair, left, right, fragment], 0.8, keep_confident=True)
    assert kept == [pair, left, right]
    assert infer_module.suppress_contained([pair, left, right, fragment], 0.8) == [pair]


def test_cli_profiles_match_the_backend_standard_profile(monkeypatch):
    calls = []
    monkeypatch.setattr(infer_module, "infer_tiled", lambda **kwargs: calls.append(kwargs) or [])
    infer_module.main(["--image", "x.jpg", "--profile", "standard"])
    infer_module.main(["--image", "x.jpg", "--profile", "standard", "--conf", ".3", "--no-contain"])
    infer_module.main(["--image", "x.jpg"])
    standard = detection_profile({})
    assert {k: calls[0][v] for k, v in (("model_path", "model_path"), ("conf", "conf_threshold"), ("iou", "iou_threshold"),
            ("tile_size", "tile_size"), ("overlap", "overlap"), ("extra_tile_sizes", "extra_tile_sizes"),
            ("extra_conf", "extra_conf_thresholds"), ("contain", "contain_threshold"),
            ("contain_keep_confident", "contain_keep_confident"))} == {**standard, "model_path": str(settings.default_model)}
    assert calls[1]["conf_threshold"] == .3 and calls[1]["contain_threshold"] is None
    assert calls[2]["conf_threshold"] == .6 and calls[2]["contain_threshold"] is None and not calls[2]["contain_keep_confident"]


class FakeBoxes:
    def __init__(self, rows):
        rows = np.array(rows, dtype=np.float32).reshape(-1, 5)
        self.xyxy = torch.tensor(rows[:, :4])
        self.conf = torch.tensor(rows[:, 4])
        self.cls = torch.zeros(len(rows))

    def __len__(self):
        return len(self.xyxy)


class ScaleAwareModel:
    """One animal at full-image (300, 300)-(400, 360). The 1280 grid finds it whole; the 640 grid
    also returns a fragment of it and a small juvenile at (900, 100)-(930, 120) that only it sees."""

    calls = []

    def __init__(self, path):
        self.path = path

    def __call__(self, image, conf=None, device=None, verbose=None):
        w, h = image.size
        ScaleAwareModel.calls.append((w, conf))
        x0, y0 = image.info["offset"] if "offset" in image.info else (0, 0)
        rows = []
        if w == 1280:
            rows.append([300, 300, 400, 360, .9])
        else:
            rows += [[310, 305, 360, 355, .95], [900, 100, 930, 120, .7]]
        rows = [[r[0] - x0, r[1] - y0, r[2] - x0, r[3] - y0, r[4]] for r in rows
                if x0 <= r[0] and r[2] <= x0 + w and y0 <= r[1] and r[3] <= y0 + h and r[4] >= conf]
        return [SimpleNamespace(boxes=FakeBoxes(rows))]


@pytest.fixture
def scale_model(monkeypatch, tmp_path):
    real_crop = infer_module.Image.Image.crop

    def crop(self, bbox):  # remember where each tile came from so the fake can place its boxes
        tile = real_crop(self, bbox)
        tile.info["offset"] = (bbox[0], bbox[1])
        return tile

    monkeypatch.setattr(infer_module.Image.Image, "crop", crop)
    monkeypatch.setattr(infer_module, "YOLO", ScaleAwareModel)
    ScaleAwareModel.calls = []
    path = tmp_path / "plate.png"
    cv2.imwrite(str(path), np.full((1280, 1280, 3), 40, dtype=np.uint8))
    return path


def read_csv(output_dir, stem="plate"):
    import pandas as pd
    return pd.read_csv(output_dir / f"{stem}_detections.csv")


def test_extra_640_pass_adds_small_animals_and_fragments_are_suppressed(scale_model, tmp_path):
    out = tmp_path / "out"
    infer_module.infer_tiled(scale_model, "model.pt", conf_threshold=.5, output_dir=out, device="0",
                             extra_tile_sizes=[640], extra_conf_thresholds=[.6], contain_threshold=.8)
    df = read_csv(out)
    assert sorted(map(tuple, df[["x1", "y1", "x2", "y2"]].round().astype(int).values.tolist())) == [
        (300, 300, 400, 360), (900, 100, 930, 120)]
    assert {c for w, c in ScaleAwareModel.calls if w == 640} == {.6}
    meta = json.loads((out / "plate_metadata.json").read_text())
    assert meta["extra_tile_sizes"] == [640] and meta["contain_threshold"] == .8
    assert meta["num_tiles"] == 1 and meta["num_tiles_all_passes"] == 1 + 9


def test_without_extra_options_the_old_single_pass_is_run(scale_model, tmp_path):
    out = tmp_path / "out"
    infer_module.infer_tiled(scale_model, "model.pt", conf_threshold=.5, output_dir=out, device="0")
    assert len(read_csv(out)) == 1
    assert {w for w, _ in ScaleAwareModel.calls} == {1280}


def test_standard_profile_comes_from_settings_and_requests_can_override():
    profile = detection_profile({})
    assert profile == {"model_path": str(settings.default_model), "conf": settings.default_conf,
                       "iou": settings.default_iou, "tile_size": settings.default_tile_size,
                       "overlap": settings.default_overlap, "extra_tile_sizes": list(settings.default_extra_tile_sizes),
                       "extra_conf": list(settings.default_extra_conf), "contain": settings.default_contain,
                       "contain_keep_confident": settings.default_contain_keep_confident}
    custom = detection_profile({"conf": .3, "extra_tile_sizes": [], "extra_conf": [], "contain": None})
    assert custom["conf"] == .3 and custom["extra_tile_sizes"] == [] and custom["contain"] is None


def test_run_detection_uses_and_records_the_standard_profile(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "outputs_dir", tmp_path)
    image = tmp_path / "photo.png"
    cv2.imwrite(str(image), np.full((50, 50, 3), 30, dtype=np.uint8))
    calls = []
    monkeypatch.setattr(infer_module, "infer_tiled", lambda **kwargs: calls.append(kwargs) or [])
    monkeypatch.setattr(detection_service, "ensure_inference_device", lambda device: None)
    result = detection_service.run_detection(
        Job(id="det", type=JobType.DETECTION, params={"image_path": str(image), "image_id": "photo"}), lambda *a: None)
    expected = detection_profile({})
    assert result["detector_profile"] == expected
    assert calls[0]["model_path"] == expected["model_path"]
    assert calls[0]["conf_threshold"] == expected["conf"]
    assert calls[0]["extra_tile_sizes"] == expected["extra_tile_sizes"]
    assert calls[0]["extra_conf_thresholds"] == expected["extra_conf"]
    assert calls[0]["contain_threshold"] == expected["contain"]


def test_profiles_recorded_before_keep_confident_existed_compare_as_off():
    recorded = {k: v for k, v in detection_profile({"contain_keep_confident": False}).items() if k != "contain_keep_confident"}
    job = Job(id="v2", type=JobType.DETECTION, params={}, status=JobStatus.COMPLETED, result={"detector_profile": recorded})
    assert recorded_profile(job)["contain_keep_confident"] is False
    assert same_profile(recorded, detection_profile({"contain_keep_confident": False}))
    assert not same_profile(recorded, detection_profile({"contain_keep_confident": True}))


def test_jobs_without_a_recorded_profile_are_legacy_detections():
    old = Job(id="old", type=JobType.DETECTION, status=JobStatus.COMPLETED,
              params={"image_path": "x", "conf": .6, "iou": .5, "tile_size": 1280, "overlap": 256, "device": "0"})
    assert same_profile(recorded_profile(old), legacy_profile())
    assert not same_profile(recorded_profile(old), detection_profile({}))
    low = Job(id="low", type=JobType.DETECTION, status=JobStatus.COMPLETED, params={"conf": .2, "tile_size": 640})
    assert not same_profile(recorded_profile(low), legacy_profile())
    new = Job(id="new", type=JobType.DETECTION, status=JobStatus.COMPLETED, params={"image_path": "x"},
              result={"detector_profile": detection_profile({})})
    assert same_profile(recorded_profile(new), detection_profile({}))


def test_startup_refuses_missing_or_unexpected_detector_weights(tmp_path, monkeypatch):
    """A deploy that forgets the (untracked) weights fails at startup, not at every detection."""
    from backend.services.detection import validate_detector

    weights = tmp_path / "detector.pt"
    monkeypatch.setattr(settings, "default_model", weights)
    monkeypatch.setattr(settings, "default_model_sha256", hashlib.sha256(b"expected weights").hexdigest())
    with pytest.raises(RuntimeError, match="missing"):
        validate_detector()
    weights.write_bytes(b"other weights")
    with pytest.raises(RuntimeError, match="expected"):
        validate_detector()
    weights.write_bytes(b"expected weights")
    assert validate_detector() == weights


def test_gpu_index_is_resolved_among_the_visible_gpus_without_touching_the_environment(monkeypatch):
    """--device 0 with CUDA_VISIBLE_DEVICES=2 must stay on physical GPU 2 (ultralytics would rewrite the variable)."""
    import os

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    assert infer_module.ultralytics_device("0") == torch.device("cuda:0")
    assert infer_module.ultralytics_device("cuda:1") == torch.device("cuda:1")
    assert infer_module.ultralytics_device(torch.device("cuda:1")) == torch.device("cuda:1")
    with pytest.raises(ValueError, match="not available"):
        infer_module.ultralytics_device("2")
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "2,3"
    assert infer_module.ultralytics_device("0,1") == "0,1"  # multi-GPU strings are left to ultralytics
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    assert infer_module.ultralytics_device("0") == "0"  # no GPU: ultralytics reports its usual error


def test_frontend_standard_conf_matches_the_backend_default():
    """The advanced detection dialog pre-fills the standard profile's confidence."""
    src = (Path(__file__).resolve().parents[2] / "frontend/src/components/AdvancedDetectionModal.tsx").read_text()
    found = re.search(r"export const STANDARD_CONF = ([0-9.]+)", src)
    assert found and float(found.group(1)) == settings.default_conf
