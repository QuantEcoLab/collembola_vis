"""Append-only detector correction snapshots and explicit YOLO exports. No training imports."""

import csv
import fcntl
import io
import json
import math
import os
import re
import shutil
import tempfile
import uuid
import zipfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from backend.config import settings
from backend.jobs.manager import job_manager
from backend.jobs.models import JobStatus, JobType
from backend.services.image import get_image_info
from backend.services.detector_registry import check_registry, sha
from backend.services.image_identity import capture_identity


class DatasetError(ValueError):
    pass


def root():
    return settings.detector_dataset_dir.resolve()


@contextmanager
def locked():
    root().mkdir(parents=True, exist_ok=True)
    with (root() / '.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def versions():
    return sorted((root() / 'versions').glob('v*/manifest.json'), reverse=True)


def read_version(version):
    if not re.fullmatch(r'v\d{6}-[a-f0-9]{12}', version):
        raise DatasetError('Neispravna verzija dataseta.')
    path = root() / 'versions' / version / 'manifest.json'
    if not path.is_file():
        raise FileNotFoundError('Verzija dataseta ne postoji.')
    data = json.loads(path.read_text())
    seal = json.loads(path.with_name('seal.json').read_text())
    if sha(path) != seal['manifest_sha256']:
        raise DatasetError('Integritet manifesta dataseta nije potvrđen.')
    return data


def baseline(image_id, job_id, image_info=None):
    if not job_id:
        return [], None
    job = job_manager.get(job_id)
    if not job or job.type != JobType.DETECTION or job.status != JobStatus.COMPLETED:
        raise DatasetError('Izvorni posao detekcije ne postoji ili nije dovršen.')
    info = image_info or get_image_info(image_id)
    if not info or info.get('image_id') != image_id:
        raise DatasetError('Fotografija nije registrirana.')
    if any(value is not None and value != image_id for value in (job.params.get('image_id'), job.result.get('image_id'))):
        raise DatasetError('Izvorni dovršeni posao detekcije ne pripada ovoj fotografiji (image_id).')
    identity = capture_identity(image_id, info['path'])
    if identity['filename'] != info['filename'] or (identity['width'], identity['height']) != (info['width'], info['height']):
        raise DatasetError('ID, naziv ili dimenzije registrirane fotografije nisu usklađeni.')
    # Legacy detection and batch routes stored image_path, not image_id. Bind the
    # registered upload path (including its image-ID directory), never just a name
    # or a guessed replacement job. Conflicting evidence always rejects the save.
    paths = [v for v in (job.params.get('image_path'), job.result.get('image_path')) if v]
    if not paths or any(str(Path(p).parent.resolve() / Path(p).name) != identity['registered_path'] for p in paths):
        raise DatasetError('Izvorni posao pokazuje na drugu registriranu fotografiju (putanja/ID/naziv).')
    metadata_path = job.result.get('metadata_path')
    metadata = None
    if metadata_path:
        try:
            metadata = json.loads(Path(metadata_path).read_text())
        except (OSError, ValueError) as exc:
            raise DatasetError('Izvorni metapodaci detekcije nisu dostupni za provjeru fotografije.') from exc
        meta_path = Path(metadata.get('image_path', ''))
        if (str(meta_path.parent.resolve() / meta_path.name) != identity['registered_path']
                or metadata.get('image_size') != [identity['width'], identity['height']]):
            raise DatasetError('Izvorni metapodaci imaju drugu fotografiju ili dimenzije.')
    recorded = [value for value in (job.params.get('image_identity'), job.result.get('image_identity')) if value]
    for value in recorded:
        if any(value.get(key) != identity[key] for key in ('image_id', 'filename', 'width', 'height', 'sha256', 'registered_path')):
            raise DatasetError('ID, naziv, dimenzije ili hash fotografije ne odgovaraju izvornom poslu detekcije.')
    if not recorded and metadata is None:
        raise DatasetError('Stariji posao nema dovoljne metapodatke za sigurnu provjeru fotografije.')
    if job.result.get('image_stem', Path(identity['filename']).stem) != Path(identity['filename']).stem:
        raise DatasetError('Naziv fotografije u izvornom rezultatu nije usklađen.')
    path = Path(job.result.get('csv_path', ''))
    if not path.is_file():
        raise DatasetError('Nedostaju izvorne detekcije; nije moguće provjeriti korisničke izmjene.')
    with path.open(newline='') as handle:
        boxes = [dict(id=f'box-{i}', **{key: float(row[key]) for key in ('x1', 'y1', 'x2', 'y2')})
                 for i, row in enumerate(csv.DictReader(handle))]
    return boxes, {'job_id': job.id, 'params': job.params, 'result': job.result,
                   'verified_image_identity': identity,
                   'identity_verification': 'captured_at_detection' if recorded else 'legacy_registered_path_and_detection_metadata',
                   'historical_image_hash_available': bool(recorded),
                   'metadata_sha256': sha(metadata_path) if metadata_path else None,
                    'detections_csv_sha256': sha(path)}


def validate_boxes(incoming, original, width, height):
    ids = set()
    final = []
    by_id = {box['id']: box for box in original}
    for box in incoming:
        if box['id'] in ids:
            raise DatasetError(f"Duplicirani ID pravokutnika: {box['id']}")
        ids.add(box['id'])
        if box['status'] == 'rejected':
            continue
        coords = [box[k] for k in ('x1', 'y1', 'x2', 'y2')]
        if not all(isinstance(v, (float, int)) and math.isfinite(v) for v in coords):
            raise DatasetError(f"Neispravne koordinate: {box['id']}")
        x1, y1, x2, y2 = coords
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise DatasetError(f"Pravokutnik {box['id']} je prazan ili izvan fotografije ({width} × {height}).")
        previous = by_id.get(box['id'])
        if previous is None and box['status'] != 'added':
            raise DatasetError(f"Pravokutnik {box['id']} nema izvornu detekciju; označite ga kao dodan.")
        action = 'added' if previous is None else 'confirmed' if all(previous[k] == box[k] for k in ('x1', 'y1', 'x2', 'y2')) else 'corrected'
        final.append({**box, 'action': action, 'original_bbox': previous})
    # An empty set is a valid fully-reviewed negative image; empty rectangles are not.
    for i, a in enumerate(final):
        for b in final[i + 1:]:
            inter = max(0, min(a['x2'], b['x2']) - max(a['x1'], b['x1'])) * max(0, min(a['y2'], b['y2']) - max(a['y1'], b['y1']))
            area_a = (a['x2'] - a['x1']) * (a['y2'] - a['y1'])
            area_b = (b['x2'] - b['x1']) * (b['y2'] - b['y1'])
            if inter / (area_a + area_b - inter) >= 0.95:
                raise DatasetError(f"Duplicirani pravokutnici (IoU ≥ 0.95): {a['id']} i {b['id']}. Uklonite jedan.")
    removed = [b for b in original if b['id'] not in {v['id'] for v in final}]
    return final, removed


def save_snapshot(request, username):
    info = get_image_info(request.image_id)
    if not info:
        raise FileNotFoundError('Fotografija nije pronađena.')
    source = Path(info['path'])
    before_hash = sha(source)
    before_stat = source.stat()
    with Image.open(source) as image:
        width, height = image.size
        exif_date = str(image.getexif().get(36867, '')) or None
    original, detection = baseline(request.image_id, request.source_job_id, info)
    if detection and detection['verified_image_identity']['sha256'] != before_hash:
        raise DatasetError('Fotografija je promijenjena tijekom provjere izvornog posla.')
    incoming = [b.model_dump() for b in request.boxes]
    final, removed = validate_boxes(incoming, original, width, height)
    try:
        membership = check_registry(settings.detector_split_registry, source, info['filename'])
    except ValueError as exc:
        raise DatasetError(str(exc)) from exc
    now = datetime.now(timezone.utc).isoformat()
    with locked():
        previous_paths = versions()
        previous = read_version(previous_paths[0].parent.name) if previous_paths else None
        entries = previous['entries'] if previous else []
        # One source per version, even when re-uploaded under another image ID/name.
        previous_entry = next((e for e in entries if e['image_sha256'] == before_hash or set(e['pixel_hashes']) & set(membership['pixel_hashes'])), None)
        entry_id = previous_entry['entry_id'] if previous_entry else uuid.uuid4().hex[:12]
        obj = root() / 'objects' / before_hash / ('original' + source.suffix.lower())
        obj.parent.mkdir(parents=True, exist_ok=True)
        if not obj.exists():
            temp = obj.with_name('.' + uuid.uuid4().hex + '.tmp')
            try:
                shutil.copyfile(source, temp)
                if sha(temp) != before_hash:
                    raise DatasetError('Fotografija je promijenjena tijekom kopiranja; spremanje prekinuto.')
                os.chmod(temp, 0o444)
                os.replace(temp, obj)
            finally:
                temp.unlink(missing_ok=True)
        if sha(obj) != before_hash or sha(source) != before_hash or source.stat().st_mtime_ns != before_stat.st_mtime_ns:
            raise DatasetError('Integritet originalne fotografije nije potvrđen.')
        entry = {'entry_id': entry_id, 'revision': previous_entry['revision'] + 1 if previous_entry else 1,
                 'image_id': request.image_id, 'filename': info['filename'], 'image_sha256': before_hash,
                 'pixel_hashes': membership['pixel_hashes'], 'width': width, 'height': height,
                 'original_object': obj.relative_to(root()).as_posix(), 'saved_at': now, 'saved_by': username,
                 'source': {'description': request.source_description, 'registered_path': str(source),
                            'resolved_path': str(source.resolve()), 'ingestion': 'server_path' if source.is_symlink() else 'upload',
                            'exif_date_original': exif_date},
                 'split_check': membership, 'detection': detection, 'original_boxes': original,
                 'submitted_boxes': incoming, 'boxes': final, 'removed_detections': removed,
                 'change_counts': {action: sum(b['action'] == action for b in final) for action in ('added', 'corrected', 'confirmed')},
                 'assignment': 'future_training_pool', 'class_id': 0}
        # Identical resubmission is idempotent; old revision remains immutable.
        if previous_entry and all(entry[k] == previous_entry[k] for k in ('boxes', 'submitted_boxes', 'source', 'image_sha256')):
            return {'version_id': previous['version_id'], 'entry_id': entry_id, 'revision': previous_entry['revision'], 'unchanged': True}
        version_id = f'v{len(previous_paths) + 1:06d}-{uuid.uuid4().hex[:12]}'
        parent = root() / 'versions'
        parent.mkdir(exist_ok=True)
        stage = Path(tempfile.mkdtemp(prefix='.pending-', dir=parent))
        try:
            data = {'schema_version': 1, 'version_id': version_id, 'parent_version': previous['version_id'] if previous else None,
                    'created_at': now, 'created_by': username, 'purpose': 'future_detector_training_only',
                    'entries': [e for e in entries if e['entry_id'] != entry_id] + [entry]}
            manifest = stage / 'manifest.json'
            manifest.write_text(json.dumps(data, indent=2, ensure_ascii=False))
            (stage / 'seal.json').write_text(json.dumps({'manifest_sha256': sha(manifest)}))
            os.rename(stage, parent / version_id)
        finally:
            if stage.exists():
                shutil.rmtree(stage)
    return {'version_id': version_id, 'entry_id': entry_id, 'revision': entry['revision'], 'unchanged': False}


def object_path(entry):
    path = (root() / entry['original_object']).resolve()
    if not path.is_relative_to(root() / 'objects') or not path.is_file() or sha(path) != entry['image_sha256']:
        raise DatasetError('Sačuvana originalna fotografija nije prošla provjeru integriteta.')
    return path


def yolo(boxes, width, height):
    return '\n'.join(f"0 {(b['x1'] + b['x2']) / (2 * width):.9f} {(b['y1'] + b['y2']) / (2 * height):.9f} {(b['x2'] - b['x1']) / width:.9f} {(b['y2'] - b['y1']) / height:.9f}" for b in boxes) + ('\n' if boxes else '')


def export_version(version):
    # Same existing tile size/overlap, edge alignment and 30% box retention.
    from scripts.create_tiled_dataset import map_roi_to_tile
    data = read_version(version)
    handle, name = tempfile.mkstemp(prefix='detector-yolo-', suffix='.zip')
    os.close(handle)
    try:
        with zipfile.ZipFile(name, 'w', zipfile.ZIP_DEFLATED) as archive:
            tile_metadata = {}
            for entry in data['entries']:
                path = object_path(entry)
                archive.write(path, 'originals/' + entry['entry_id'] + path.suffix)
                archive.writestr('original_labels/' + entry['entry_id'] + '.txt', yolo(entry['boxes'], entry['width'], entry['height']))
                with Image.open(path) as image:
                    image = image.convert('RGB')
                    # Match create_tiled_dataset edge-aligned stride=1024 coordinates.
                    xs, ys = [], []
                    for length, offsets in ((image.width, xs), (image.height, ys)):
                        start = 0
                        while True:
                            end = min(start + 1280, length)
                            offsets.append(max(0, end - 1280) if end == length else start)
                            if end == length:
                                break
                            start += 1024
                    for iy, y in enumerate(ys):
                        for ix, x in enumerate(xs):
                            tile = image.crop((x, y, min(x + 1280, image.width), min(y + 1280, image.height)))
                            labels = []
                            for b in entry['boxes']:
                                mapped = map_roi_to_tile(b['x1'], b['y1'], b['x2'] - b['x1'], b['y2'] - b['y1'], x, y, 1280)
                                if mapped:
                                    cx, cy, w, h = mapped
                                    # Existing mapping is 1280-based; small photos keep native size.
                                    labels.append(f'0 {cx * 1280 / tile.width:.9f} {cy * 1280 / tile.height:.9f} {w * 1280 / tile.width:.9f} {h * 1280 / tile.height:.9f}')
                            # Preserve background tiles too, including fully-reviewed negative photos.
                            tile_id = f"{entry['entry_id']}_tile_{iy * len(xs) + ix:04d}"
                            buf = io.BytesIO()
                            tile.save(buf, 'JPEG', quality=95)
                            archive.writestr('images/train/' + tile_id + '.jpg', buf.getvalue())
                            archive.writestr('labels/train/' + tile_id + '.txt', '\n'.join(labels) + ('\n' if labels else ''))
                            tile_metadata[tile_id] = {'entry_id': entry['entry_id'], 'source_sha256': entry['image_sha256'], 'offset_x': x, 'offset_y': y, 'width': tile.width, 'height': tile.height, 'split': 'train', 'num_annotations': len(labels)}
            archive.writestr('manifest.json', json.dumps(data, indent=2, ensure_ascii=False))
            archive.writestr('tile_metadata.json', json.dumps(tile_metadata, indent=2))
            archive.writestr('data.yaml', 'path: .\ntrain: images/train\nval: null\n# Supply an independent validation set before future training.\nnc: 1\nnames: [collembola]\n')
            archive.writestr('README.txt', 'Future detector training pool, not a trained model.\nYOLO detection class 0, normalized center-x center-y width height.\n1280 tiles, overlap 256, 30% retained-box rule; reviewed background tiles included.\nOriginal bytes and full-image labels are included separately.\nNo automatic train/val/test split. val:null must be configured with an independent validation set before training.\nDo not mix tiles from one source photograph across splits.\n')
        return Path(name)
    except BaseException:
        Path(name).unlink(missing_ok=True)
        raise
