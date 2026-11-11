import io
import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from backend.auth import get_current_user
from backend.config import settings
from backend.jobs.models import JobStatus, JobType
from backend.routers import detector_dataset as api
from backend.services import detector_dataset as service
from backend.services.detector_registry import build_registry, check_registry, sha


@pytest.fixture
def environment(tmp_path, monkeypatch):
    known = tmp_path / 'known'
    (known / 'images/train').mkdir(parents=True)
    reserved = known / 'images/train/used_plate_tile_0000.png'
    Image.new('RGB', (12, 10), 'red').save(reserved)
    script = tmp_path / 'prepare.py'
    script.write_text("TRAIN_PLATES=['reserved_train']\nVAL_PLATES=['reserved_val']\nTEST_PLATES=['reserved_test']\n")
    registry = tmp_path / 'registry.json'
    registry.write_text(json.dumps(build_registry([known], [], script)))
    for field in ('uploads_dir', 'outputs_dir'):
        directory = tmp_path / field
        directory.mkdir()
        monkeypatch.setattr(settings, field, directory)
    monkeypatch.setattr(settings, 'detector_dataset_dir', tmp_path / 'curated')
    monkeypatch.setattr(settings, 'detector_split_registry', registry)
    image_dir = settings.uploads_dir / 'photo'
    image_dir.mkdir()
    photo = image_dir / 'new_photo.png'
    Image.new('RGB', (100, 80), 'blue').save(photo)
    csv = settings.outputs_dir / 'detections.csv'
    csv.write_text('x1,y1,x2,y2,confidence\n1,2,11,12,0.9\n20,20,35,35,0.8\n50,50,70,70,0.8\n')
    meta = settings.outputs_dir / 'metadata.json'
    meta.write_text(json.dumps({'image_path': str(photo), 'image_size': [100, 80]}))
    job = SimpleNamespace(id='job', type=JobType.DETECTION, status=JobStatus.COMPLETED,
                          params={'image_path': str(photo)},
                          result={'csv_path': str(csv), 'metadata_path': str(meta), 'image_path': str(photo), 'image_stem': photo.stem})
    monkeypatch.setattr(service, 'job_manager', SimpleNamespace(get=lambda key: job if key == 'job' else None))
    app = FastAPI()
    app.include_router(api.router)
    app.dependency_overrides[get_current_user] = lambda: 'reviewer'
    with TestClient(app) as client:
        yield client, photo, reserved, registry, job


def box(key='added-a', x1=40, y1=2, x2=50, y2=12, status='added'):
    return dict(id=key, x1=x1, y1=y1, x2=x2, y2=y2, status=status, conf=1)


def payload(boxes=None, job='job'):
    return {'image_id': 'photo', 'source_job_id': job, 'source_description': 'Experiment A, plate 3',
            'boxes': boxes if boxes is not None else [box('box-0', 1, 2, 11, 12, 'accepted'),
                     box('box-1', 21, 20, 36, 35, 'accepted'), box()]}


def test_snapshot_audit_original_copy_idempotency_and_immutable_revision(environment):
    client, photo, reserved, registry, _ = environment
    before = {p: (sha(p), p.stat().st_mtime_ns) for p in (photo, reserved, registry)}
    response = client.post('/api/detector-dataset/save', json=payload())
    assert response.status_code == 200, response.text
    saved = response.json()
    first = client.get('/api/detector-dataset/versions/' + saved['version_id']).json()
    entry = first['entries'][0]
    assert entry['saved_by'] == 'reviewer'
    assert entry['change_counts'] == {'added': 1, 'corrected': 1, 'confirmed': 1}
    assert [b['id'] for b in entry['removed_detections']] == ['box-2']
    assert service.object_path(entry).read_bytes() == photo.read_bytes()
    assert not service.object_path(entry).is_symlink()
    assert client.post('/api/detector-dataset/save', json=payload()).json()['unchanged']
    changed = payload()
    changed['boxes'][1]['x2'] = 37
    second = client.post('/api/detector-dataset/save', json=changed).json()
    assert second['revision'] == 2 and second['version_id'] != saved['version_id']
    assert client.get('/api/detector-dataset/versions/' + saved['version_id']).json() == first
    assert len(service.read_version(second['version_id'])['entries']) == 1
    assert {p: (sha(p), p.stat().st_mtime_ns) for p in before} == before


@pytest.mark.parametrize('bad', [box(x2=40), box(x1=-1), box(x2=101), box(y2=81), box(x1=60, x2=50)])
def test_invalid_box_blocks_save_before_any_version(environment, bad):
    client, *_ = environment
    response = client.post('/api/detector-dataset/save', json=payload([bad]))
    assert response.status_code == 409
    assert not service.versions()


def test_duplicate_ids_coordinates_and_near_duplicate_boxes_block_save(environment):
    client, *_ = environment
    for boxes in ([box(), box()], [box('one'), box('two')], [box('one'), box('two', x1=40.1)]):
        response = client.post('/api/detector-dataset/save', json=payload(boxes))
        assert response.status_code == 409, response.text
    assert not service.versions()


def test_rejected_boxes_are_audited_not_exported_and_negative_photos_allowed(environment):
    client, *_ = environment
    result = client.post('/api/detector-dataset/save', json=payload([box('box-0', status='rejected')])).json()
    entry = service.read_version(result['version_id'])['entries'][0]
    assert entry['boxes'] == [] and len(entry['removed_detections']) == 3
    assert entry['submitted_boxes'][0]['status'] == 'rejected'


def test_source_membership_rejects_train_validation_test_and_tile_parent(environment):
    _, photo, _, registry, _ = environment
    for name in ('reserved_train.jpg', 'reserved_val.png', 'reserved_test.tif', 'used_plate.jpg'):
        with pytest.raises(ValueError, match='već je'):
            check_registry(registry, photo, name)


def test_renamed_and_reencoded_known_photo_cannot_bypass_split_check(environment):
    client, photo, reserved, registry, _ = environment
    with Image.open(reserved) as image:
        image.save(photo, compress_level=0)
    assert sha(photo) != sha(reserved)
    with pytest.raises(ValueError, match='već je'):
        check_registry(registry, photo, 'completely_renamed.png')
    assert client.post('/api/detector-dataset/save', json=payload([], None)).status_code == 409


def test_missing_or_stale_registry_fails_closed(environment):
    client, _, reserved, registry, _ = environment
    Image.new('RGB', (20, 20), 'green').save(reserved.parent / 'new_training_image.png')
    assert client.post('/api/detector-dataset/save', json=payload()).status_code == 409
    registry.unlink()
    assert client.post('/api/detector-dataset/save', json=payload()).status_code == 409
    assert not service.versions()


def test_wrong_image_job_unknown_confirmed_box_and_unknown_input_blocked(environment):
    client, _, _, _, job = environment
    assert client.post('/api/detector-dataset/save', json=payload([box('unknown', status='accepted')])).status_code == 409
    job.params['image_id'] = 'another-photo'
    assert client.post('/api/detector-dataset/save', json=payload()).status_code == 409
    assert client.post('/api/detector-dataset/save', json=payload() | {'start_training': True}).status_code == 422


def test_yolo_export_normalization_original_bytes_and_snapshot_seal(environment):
    client, photo, *_ = environment
    result = client.post('/api/detector-dataset/save', json=payload([box('manual', 10, 8, 30, 24)], None)).json()
    response = client.get('/api/detector-dataset/versions/' + result['version_id'] + '/export')
    assert response.status_code == 200, response.text
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        names = archive.namelist()
        label = next(n for n in names if n.startswith('labels/train/'))
        assert list(map(float, archive.read(label).decode().split())) == pytest.approx([0, .2, .2, .2, .2])
        assert archive.read(next(n for n in names if n.startswith('originals/'))) == photo.read_bytes()
        assert b'val: null' in archive.read('data.yaml')
        assert not any('/val/' in n or '/test/' in n for n in names)
    manifest = service.root() / 'versions' / result['version_id'] / 'manifest.json'
    manifest.write_text(manifest.read_text() + ' ')
    assert client.get('/api/detector-dataset/versions/' + result['version_id']).status_code == 409


def test_export_uses_existing_edge_aligned_tile_mapping(environment):
    client, photo, *_ = environment
    Image.new('RGB', (1500, 1300), 'navy').save(photo)
    result = client.post('/api/detector-dataset/save', json=payload([box('edge', 1400, 1200, 1490, 1290)], None)).json()
    path = service.export_version(result['version_id'])
    try:
        with zipfile.ZipFile(path) as archive:
            metadata = json.loads(archive.read('tile_metadata.json'))
            assert {(v['offset_x'], v['offset_y']) for v in metadata.values()} == {(0, 0), (220, 0), (0, 20), (220, 20)}
            assert all(v['width'] == 1280 and v['height'] == 1280 for v in metadata.values())
            assert all(0 <= float(v) <= 1 for n in archive.namelist() if n.startswith('labels/train/') for v in archive.read(n).decode().split())
    finally:
        path.unlink()


def test_auth_required_and_paths_not_accepted():
    app = FastAPI()
    app.include_router(api.router)
    with TestClient(app) as client:
        assert client.get('/api/detector-dataset').status_code == 401
        assert client.post('/api/detector-dataset/save', json=payload()).status_code == 401


def test_real_legacy_detection_contract_missing_id_is_verified_without_mutation(environment):
    client, photo, _, _, job = environment
    assert 'image_id' not in job.params
    before = json.dumps(job.__dict__, sort_keys=True)
    response = client.post('/api/detector-dataset/save', json=payload())
    assert response.status_code == 200, response.text
    entry = service.read_version(response.json()['version_id'])['entries'][0]
    verified = entry['detection']['verified_image_identity']
    assert verified['image_id'] == 'photo' and verified['filename'] == photo.name
    assert [verified['width'], verified['height']] == [100, 80]
    assert verified['sha256'] == sha(photo)
    assert entry['detection']['identity_verification'] == 'legacy_registered_path_and_detection_metadata'
    assert entry['detection']['historical_image_hash_available'] is False
    assert json.dumps(job.__dict__, sort_keys=True) == before


@pytest.mark.parametrize('conflict', ['path', 'same_name_other_id', 'metadata_dimensions', 'metadata_path', 'identity_hash', 'identity_id'])
def test_photo_provenance_conflicts_are_not_silently_repaired(environment, conflict):
    from backend.services.image_identity import capture_identity
    client, photo, _, _, job = environment
    if conflict in {'path', 'same_name_other_id'}:
        other = photo.parent.parent / 'other'
        other.mkdir()
        target = other / photo.name
        target.write_bytes(photo.read_bytes())
        if conflict == 'path':
            job.params['image_path'] = str(target)
        else:
            job.params['image_path'] = job.result['image_path'] = str(target)
    elif conflict.startswith('metadata'):
        meta = Path(job.result['metadata_path'])
        data = json.loads(meta.read_text())
        data['image_size' if conflict == 'metadata_dimensions' else 'image_path'] = [80, 100] if conflict == 'metadata_dimensions' else '/other/photo.png'
        meta.write_text(json.dumps(data))
    else:
        identity = capture_identity('photo', photo)
        identity['sha256' if conflict == 'identity_hash' else 'image_id'] = 'different'
        job.params['image_identity'] = identity
    response = client.post('/api/detector-dataset/save', json=payload())
    assert response.status_code == 409, response.text
    assert not service.versions()


def test_detection_submission_persists_identity_without_running_inference(environment, monkeypatch):
    from backend.routers import detection
    from backend.services.image_identity import capture_identity
    client, photo, *_ = environment
    calls = []
    monkeypatch.setattr(detection, 'job_manager', SimpleNamespace(submit=lambda kind, params: calls.append((kind, params)) or SimpleNamespace(id='future-job', status=JobStatus.PENDING)))
    app = FastAPI()
    app.include_router(detection.router)
    app.dependency_overrides[get_current_user] = lambda: 'reviewer'
    with TestClient(app) as client:
        response = client.post('/api/detection/run', json={'image_id': 'photo'})
    assert response.status_code == 200
    assert calls[0][1]['image_id'] == 'photo'
    assert calls[0][1]['image_identity'] == capture_identity('photo', photo)
