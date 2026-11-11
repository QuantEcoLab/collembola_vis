"""Read-only split inventory. Source photographs stay grouped with their crops/tiles."""

import ast
import csv
import hashlib
import json
import os
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageOps
import yaml

PHOTO = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp', '.webp'}
TEXT = {'.json', '.csv', '.yaml', '.yml', '.txt'}
PRUNE = {'.git', 'node_modules', '__pycache__', '.venv', '.venv-ruler-ft',
         '.vscode', 'frontend', 'backend', 'docs', 'scripts', 'models',
         'integration_exp03_local', 'curated_detector', 'detector_dataset_worktree_backup'}
SPLITS = {'train': 'train', 'training': 'train', 'val': 'validation',
          'validation': 'validation', 'valid': 'validation', 'test': 'test'}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def source_id(value):
    stem = Path(str(value).replace('\\', '/')).stem
    stem = re.sub(r'^ROI_', '', stem, flags=re.I)
    stem = re.split(r'_(?:tile|det|crop|audit|ruler|c_\d|region)\w*', stem, flags=re.I)[0]
    return re.sub(r'[^a-z0-9]', '', unicodedata.normalize('NFKC', stem).casefold())


def pixel_hashes(path):
    """Native and EXIF-oriented RGB hashes, without resizing or rewriting the image."""
    Image.MAX_IMAGE_PIXELS = None
    with Image.open(path) as image:
        result = []
        views = (image, ImageOps.exif_transpose(image)) if image.getexif().get(274, 1) != 1 else (image,)
        for view in views:
            rgb = view.convert('RGB')
            digest = hashlib.sha256(f'RGB:{rgb.width}:{rgb.height}:'.encode())
            for y in range(0, rgb.height, 128):
                digest.update(rgb.crop((0, y, rgb.width, min(y + 128, rgb.height))).tobytes())
            result.append(digest.hexdigest())
        return sorted(set(result))


def inventory(roots, excluded=()):
    """Fingerprint membership evidence, including newly added datasets, not app uploads."""
    files = {}
    excluded = [Path(p).resolve() for p in excluded]
    for root in map(Path, roots):
        if not root.is_dir():
            raise ValueError(f'Nedostaje korijen provjere skupova: {root}')
        for directory, dirs, names in os.walk(root, followlinks=False):
            parent = Path(directory)
            dirs[:] = sorted(d for d in dirs if d not in PRUNE and not d.startswith(('.venv', 'detector_dataset_worktree_backup_', 'integrate_exp03_worktree_backup_'))
                             and not any((parent / d).resolve().is_relative_to(e) for e in excluded))
            for name in names:
                p = parent / name
                # Ordinary per-image UI saves are not split membership evidence;
                # retain original ROI CSVs and all dataset manifests in this folder.
                if parent.name == 'annotations' and re.fullmatch(r'[a-f0-9]{12}\.json', name):
                    continue
                if p.suffix.lower() not in PHOTO | TEXT or any(p.resolve().is_relative_to(e) for e in excluded):
                    continue
                stat = p.stat()
                files[str(p.resolve())] = [stat.st_size, stat.st_mtime_ns]
    return dict(sorted(files.items()))


def build_registry(roots, excluded, detector_script, previous=None):
    files = inventory(roots, excluded)
    memberships = {}
    image_paths = set()

    def reference(value, split, context):
        if not isinstance(value, str):
            return
        candidate = Path(value)
        candidates = [candidate, context.parent / candidate]
        for r in roots:
            candidates.append(Path(r) / candidate)
            for marker in ('data/', 'outputs/'):
                if marker in value:
                    candidates.append(Path(r) / value[value.index(marker):])
        existing = next((p for p in candidates if p.exists()), None)
        if existing and not any(existing.resolve().is_relative_to(Path(r).resolve()) for r in roots):
            raise ValueError(f'Deklarirani skup je izvan opsega registra; dodajte --root za {existing}')
        if existing and existing.is_dir():
            for p in existing.rglob('*'):
                if p.is_file() and p.suffix.lower() in PHOTO:
                    image_paths.add(str(p.resolve()))
                    add(source_id(p.name), split, context)
        elif existing and existing.suffix.lower() in PHOTO:
            image_paths.add(str(existing.resolve()))
            add(source_id(existing.name), split, context)
        elif existing and existing.suffix.lower() == '.txt':
            for line in existing.read_text().splitlines():
                if line.strip() and Path(line.strip()).suffix.lower() in PHOTO:
                    reference(line.strip(), split, existing)

    def add(identity, split, evidence):
        if identity:
            memberships.setdefault(identity, set()).add((split, str(evidence)))

    def record(row, inherited=None, evidence=''):
        if isinstance(row, list):
            for item in row:
                record(item, inherited, evidence)
        elif isinstance(row, dict):
            role = str(row.get('split', row.get('role', row.get('proposed_source_role', inherited or '')))).lower()
            split = next((v for k, v in SPLITS.items() if re.search(r'(?:^|_)' + k + r'(?:_|$)', role)), inherited)
            if split:
                for key in ('source_id', 'plate_id', 'source_image', 'image_origin', 'image_name', 'img_path'):
                    if row.get(key):
                        add(source_id(row[key]), split, evidence)
                if row.get('source_image'):
                    p = Path(row['source_image'])
                    if p.is_file():
                        image_paths.add(str(p.resolve()))
            for key, value in row.items():
                if isinstance(value, (list, dict)):
                    record(value, SPLITS.get(str(key).lower(), split), evidence)

    # The detector's explicit original-photo train/val lists remain evidence even
    # when a generated tiled directory is not present on this machine.
    tree = ast.parse(Path(detector_script).read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in {'TRAIN_PLATES', 'VAL_PLATES', 'TEST_PLATES'}:
                    for value in ast.literal_eval(node.value):
                        add(source_id(value), {'TRAIN_PLATES': 'train', 'VAL_PLATES': 'validation', 'TEST_PLATES': 'test'}[target.id], detector_script)
    for filename in files:
        path = Path(filename)
        split = next((SPLITS[p.lower()] for p in reversed(path.parts[:-1]) if p.lower() in SPLITS), None)
        if not split and any(re.search(r'(?:^|_)(?:test|holdout)(?:_|$)', part, re.I) for part in path.parts[:-1]):
            split = 'test'
        if path.suffix.lower() in PHOTO:
            if split:
                add(source_id(path.name), split, path)
                image_paths.add(filename)
            continue
        if path.suffix == '.json':
            try:
                record(json.loads(path.read_text()), split, path)
            except (json.JSONDecodeError, UnicodeError):
                raise ValueError(f'Nečitljivi metapodaci: {path}')
        elif path.suffix == '.csv':
            csv_split = SPLITS.get(path.stem.lower(), split)
            with path.open(newline='') as handle:
                for row in csv.DictReader(handle):
                    record(row, csv_split, path)
                    if csv_split and row.get('img_path'):
                        reference(row['img_path'], csv_split, path)
                    # ROI conversion defines every unlisted plate as detector test.
                    if 'plate_id' in row and 'image_name' in row:
                        sid = source_id(row['plate_id'])
                        if sid not in memberships:
                            add(sid, 'test', path)
        elif path.suffix in {'.yaml', '.yml'}:
            data = yaml.safe_load(path.read_text())
            if isinstance(data, dict):
                for key in ('train', 'val', 'validation', 'test'):
                    value = data.get(key)
                    for item in value if isinstance(value, list) else [value]:
                        if isinstance(item, str):
                            reference(str(Path(str(data.get('path') or '.')) / item), SPLITS[key], path)
        elif path.suffix == '.txt' and path.stem.lower() in SPLITS:
            reference(str(path), SPLITS[path.stem.lower()], path)
    # Locate original photos for known crop/tile sources; catches renamed or
    # losslessly re-encoded uploads through native decoded-pixel hashes.
    for filename in files:
        if Path(filename).suffix.lower() in PHOTO and source_id(filename) in memberships:
            image_paths.add(filename)
    images = []
    seen = set()
    # Reuse decoded-pixel identities only for byte-identical images from our
    # earlier inventory; re-enumerate all membership evidence and file hashes.
    cache = {row['file_sha256']: row for row in (previous or {}).get('images', [])}
    for index, filename in enumerate(sorted(image_paths)):
        file_hash = sha(filename)
        identity = source_id(filename)
        if file_hash in seen:
            continue
        seen.add(file_hash)
        cached = cache.get(file_hash)
        if cached:
            dimensions = cached['dimensions']
            pixels = cached['pixel_hashes']
        else:
            with Image.open(filename) as image:
                dimensions = list(image.size)
            pixels = pixel_hashes(filename)
        images.append({'path': filename, 'file_sha256': file_hash, 'dimensions': dimensions,
                       'pixel_hashes': pixels, 'source_id': identity})
        if index % 1000 == 0:
            print(f'Indexed split images: {index}/{len(image_paths)}', flush=True)
    return {'schema_version': 1, 'created_at': datetime.now(timezone.utc).isoformat(),
            'roots': list(map(str, roots)), 'excluded': list(map(str, excluded)), 'inventory': files,
            'detector_script': str(detector_script), 'detector_script_sha256': sha(detector_script),
            'registry_implementation_sha256': sha(Path(__file__)),
            'source_memberships': {key: [{'split': s, 'evidence': e} for s, e in sorted(value)] for key, value in memberships.items()},
            'images': images}


def check_registry(path, image_path, filename):
    if not Path(path).is_file():
        raise ValueError('Registar train/validation/test skupova nije pripremljen. Spremanje je blokirano.')
    registry = json.loads(Path(path).read_text())
    if registry.get('registry_implementation_sha256') != sha(Path(__file__)):
        raise ValueError('Ažurirana je provjera skupova; osvježite registar prije spremanja.')
    if inventory(registry['roots'], registry['excluded']) != registry['inventory'] or sha(registry['detector_script']) != registry['detector_script_sha256']:
        raise ValueError('Postojeći skupovi su promijenjeni; osvježite registar prije spremanja.')
    identity = source_id(filename)
    matches = registry['source_memberships'].get(identity, [])
    file_hash = sha(image_path)
    pixels = pixel_hashes(image_path)
    for row in registry['images']:
        if row['file_sha256'] == file_hash or set(row['pixel_hashes']) & set(pixels):
            matches = matches + [{'split': 'existing split', 'evidence': row['path']}]
    if matches:
        example = matches[0]
        raise ValueError(f"Fotografija ili njezin izvor već je u train/validation/test skupu ({example['split']}): {example['evidence']}")
    return {'registry_sha256': sha(path), 'registry_created_at': registry['created_at'],
            'roots': registry['roots'], 'checked_source_id': identity,
            'file_sha256': file_hash, 'pixel_hashes': pixels, 'membership': 'not_found_in_registered_splits'}
