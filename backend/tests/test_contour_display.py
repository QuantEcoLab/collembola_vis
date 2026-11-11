import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.auth import get_current_user
from backend.config import settings
from backend.jobs.models import JobStatus
from backend.routers import contours as router_module
from backend.services.contour_display import draw_outer_contours


def test_display_uses_original_photo_and_external_mask_boundary_without_changing_results(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, 'outputs_dir', tmp_path)
    photo = tmp_path / 'original.png'
    cv2.imwrite(str(photo), np.zeros((80, 80, 3), np.uint8))
    mask = np.zeros((40, 40), np.uint8)
    mask[5:35, 5:35] = 255
    mask[14:26, 14:26] = 0  # hole must not introduce an inner boundary
    mask_path = tmp_path / 'mask.png'
    cv2.imwrite(str(mask_path), mask)
    baked = np.zeros((80, 80, 3), np.uint8)
    baked[40, 40] = (0, 255, 0)  # pre-existing measurement visualization is not used
    old_overlay = tmp_path / 'old-overlay.png'
    cv2.imwrite(str(old_overlay), baked)
    record = {'detection_id': 1, 'crop_box': [20, 20, 60, 60], 'mask_path': str(mask_path), 'has_mask': True, 'kept_after_dedup': True}
    records = tmp_path / 'records.json'
    records.write_text(json.dumps([record, {**record, 'crop_box': [0, 0, 40, 40], 'kept_after_dedup': False}]))
    result = {'image_path': str(photo), 'mask_records_path': str(records), 'overlay_path': str(old_overlay),
              'model_name': 'exp03_no_os', 'measurements': [{'length_mm': 123, 'volume_mm3': 456}]}
    saved = tmp_path / 'result.json'
    saved.write_text(json.dumps(result))
    def seal(path):
        return hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns
    files = [photo, mask_path, records, old_overlay, saved]
    before = {p: seal(p) for p in files}
    image = draw_outer_contours(result)
    assert np.array_equal(image[25, 40], [0, 255, 255])  # external boundary
    assert np.array_equal(image[40, 40], [0, 0, 0])  # no centerline
    assert np.array_equal(image[34, 40], [0, 0, 0])  # no internal hole outline
    assert np.array_equal(image[5, 20], [0, 0, 0])  # rejected record not drawn
    app = FastAPI()
    app.include_router(router_module.router)
    monkeypatch.setattr(router_module, 'job_manager', SimpleNamespace(get=lambda key: SimpleNamespace(status=JobStatus.COMPLETED, result=result)))
    with TestClient(app) as client:
        assert client.get('/api/contours/overview/job').status_code == 401
        app.dependency_overrides[get_current_user] = lambda: 'reviewer'
        response = client.get('/api/contours/overview/job')
        assert response.status_code == 200
        assert response.headers['content-type'] == 'image/jpeg'
        decoded = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)
        assert decoded.shape == image.shape
        assert decoded[40, 40].max() < 10
    assert {p: seal(p) for p in files} == before
