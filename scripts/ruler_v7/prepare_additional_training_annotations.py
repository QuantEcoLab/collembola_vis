#!/usr/bin/env python3
"""Prepare 22 reviewed training-source targets for manual annotation, not YOLO."""

import argparse
import csv
import json
import math
import shutil
from collections import Counter
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw

from prepare_reference_annotation_pack import ASSETS, ROOT, read_csv, sha256

TRAIN_SOURCES = {'C_1_2_Collembola005', 'C_1_3_Collembola006'}
FORBIDDEN_SOURCES = {'C_3_2_Collembola011', 'C_4_1_PS_500-1000_014', 'C_5_1_PS_500-1000_017'}
SELECTION = [
    ('C_1_2_Collembola005', '505', 'BODY_PART_MISSING'),
    ('C_1_2_Collembola005', '826', 'BODY_PART_MISSING'),
    ('C_1_2_Collembola005', '862', 'MULTIPLE_PROBLEMS'),
    ('C_1_2_Collembola005', '934', 'BODY_PART_MISSING'),
    ('C_1_2_Collembola005', '1100', 'TOO_SMALL'),
    ('C_1_2_Collembola005', '1126', 'TOO_SMALL'),
    ('C_1_2_Collembola005', '1190', 'TOO_SMALL'),
    ('C_1_2_Collembola005', '1233', 'TOO_SMALL'),
    ('C_1_2_Collembola005', '1243', 'MULTIPLE_PROBLEMS'),
    ('C_1_2_Collembola005', '1257', 'BODY_PART_MISSING'),
    ('C_1_2_Collembola005', '1278', 'RULER_INCLUDED'),
    ('C_1_3_Collembola006', '173', 'MULTIPLE_PROBLEMS'),
    ('C_1_3_Collembola006', '492', 'MULTIPLE_PROBLEMS'),
    ('C_1_3_Collembola006', '562', 'MULTIPLE_PROBLEMS'),
    ('C_1_3_Collembola006', '624', 'MULTIPLE_PROBLEMS'),
    ('C_1_3_Collembola006', '687', 'TOO_SMALL'),
    ('C_1_3_Collembola006', '713', 'MULTIPLE_PROBLEMS'),
    ('C_1_3_Collembola006', '722', 'TOO_SMALL'),
    ('C_1_3_Collembola006', '729', 'TOO_SMALL'),
    ('C_1_3_Collembola006', '895', 'MULTIPLE_PROBLEMS'),
    ('C_1_3_Collembola006', '1036', 'TOO_SMALL'),
    ('C_1_3_Collembola006', '1063', 'RULER_INCLUDED'),
]


def build(assets, output, previous):
    candidates_path = ROOT / 'outputs/ruler_v7_body_error_review/annotation_candidates.csv'
    candidates = {(r['source_id'], r['detection_id']): r for r in read_csv(candidates_path)}
    cut = {(r['source_id'], r['detection_id']) for r in read_csv(candidates_path.with_name('padding_cases.csv'))}
    positions = {(r['source_id'], r['detection_id']): r for r in read_csv(ROOT / 'reports/ruler_v7/manual_review/ruler_position_review.csv')}
    sources = {r['source_id']: r['image_path'] for r in read_csv(ROOT / 'reports/ruler_v7/unseen/candidate_selection_leakage_audit.csv')}
    old = json.loads((previous / 'manifest.json').read_text())
    previous_ids = {(r['source_id'], r['detection_id']) for r in old}
    old_hashes = {str(p): sha256(p) for p in previous.rglob('*') if p.is_file()}
    ids = {(s, d) for s, d, _group in SELECTION}
    if len(ids) != 22 or ids & previous_ids or ids & cut:
        raise ValueError('Invalid, already selected or truncated IDs')
    if output.exists():
        raise ValueError(f'Refusing to overwrite {output}')
    Image.MAX_IMAGE_PIXELS = None
    manifest = []
    for source in sorted(TRAIN_SOURCES):
        with Image.open(assets / sources[source]) as full:
            for s, detection, group in SELECTION:
                if s != source:
                    continue
                row = candidates[(source, detection)]
                if row['crop_condition'] != 'COMPLETE' or positions[(source, detection)]['position_class'] != 'R':
                    raise ValueError('Target is not a complete real ruler organism')
                if 'UNASSESSABLE' in (row['v5_primary_error'], row['v7_primary_error']):
                    raise ValueError('Unclear review')
                if group not in {row['v5_primary_error'], row['v7_primary_error']}:
                    raise ValueError('Selection group must be supported by a saved model rating')
                raw_path = (assets / row['original_path']).resolve()
                if not raw_path.is_relative_to(assets.resolve()) or raw_path.parent.name != 'all_crops':
                    raise ValueError('Only genuine raw all_crops inputs may be annotated')
                with Image.open(raw_path) as raw:
                    raw.load()
                    width, height = raw.size
                    x, y = math.floor(float(row['bbox_x1'])), math.floor(float(row['bbox_y1']))
                    box = [x, y, x + width, y + height]
                    # Confirms that the input really is an unpainted source crop,
                    # not a text/contour/panel image mislabeled as an original.
                    diff = ImageChops.difference(raw.convert('RGB'), full.crop(box).convert('RGB'))
                    if diff.getbbox() is not None:
                        raise ValueError(f'Raw/source pixels differ: {source}/{detection}')
                for prior in old:
                    if prior['source_id'] != source:
                        continue
                    annotation = json.loads((previous / 'annotations' / (prior['case_id'] + '.json')).read_text())
                    origin = prior['annotation_box_source_xyxy']
                    points = [(p[0] + origin[0] - x, p[1] + origin[1] - y) for p in annotation['shapes'][0]['points']]
                    overlap = Image.new('1', (width, height))
                    ImageDraw.Draw(overlap).polygon(points, fill=1)
                    if overlap.getbbox() is not None:
                        raise ValueError(f'Target crop intersects an existing manual body: {source}/{detection}')
                case_id = f'{source}__det{int(detection):04d}'
                files = {'image': f'images/{case_id}.png', 'original': f'original_crops/{case_id}.png',
                         'context': f'context/{case_id}.png',
                         'v5': f'references/v5/{case_id}_mask.png', 'v7': f'references/v7/{case_id}_mask.png'}
                context_box = [max(0, x - 450), max(0, y - 400), min(full.width, x + width + 450), min(full.height, y + height + 400)]
                # Validate both reference masks before creating this case.
                for model in ['v5', 'v7']:
                    path = (assets / row[model + '_mask_path']).resolve()
                    if not path.is_relative_to(assets.resolve()):
                        raise ValueError('Mask outside asset root')
                    with Image.open(path) as image:
                        if image.size != (width, height):
                            raise ValueError('Raw mask dimensions do not match the original crop')
                for name in files.values():
                    (output / name).parent.mkdir(parents=True, exist_ok=True)
                for kind in ['image', 'original']:
                    shutil.copy2(raw_path, output / files[kind])
                full.crop(context_box).save(output / files['context'])
                for model in ['v5', 'v7']:
                    shutil.copy2(assets / row[model + '_mask_path'], output / files[model])
                manifest.append({
                    'case_id': case_id, 'source_id': source, 'detection_id': detection,
                    'selection_group': group,
                    'selection_reason': f"{group}; V5={row['v5_primary_error']}; V7={row['v7_primary_error']}",
                    'review_record': row, 'source_image': sources[source], 'assets_root': str(assets),
                    'original_crop_box_source_xyxy': box, 'annotation_box_source_xyxy': box,
                    'context_box_source_xyxy': context_box, 'original_dimensions': [width, height],
                    'annotation_size': [width, height], 'files': files,
                    'sha256': {kind: sha256(output / name) for kind, name in files.items()},
                    'proposed_source_role': 'TRAIN_SOURCE_ANNOTATION_ONLY', 'dataset_assignment': 'NONE_NOT_A_YOLO_DATASET',
                })
    if len({row['sha256']['original'] for row in manifest}) != 22:
        raise ValueError('Duplicate raw crop bytes in selection')
    for folder in ['annotations', 'annotation_history']:
        (output / folder).mkdir(parents=True, exist_ok=True)
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    info = {'title': '22 dodatne ruler jedinke — samo trening izvori',
            'annotation_mode': 'ORIGINAL_RAW_CROP', 'autosave_on_close': True,
            'case_count': len(manifest), 'counts_by_source': dict(Counter(r['source_id'] for r in manifest)),
            'counts_by_selection_group': dict(Counter(r['selection_group'] for r in manifest)),
            'source_allowlist': sorted(TRAIN_SOURCES), 'excluded_sources': sorted(FORBIDDEN_SOURCES),
            'source_overlap_with_validation_or_test': [], 'overlap_with_previous12': [],
            'candidate_manifest_sha256': sha256(candidates_path),
            'old_pack_file_hashes': old_hashes,
            'server_started': False, 'annotations_created': 0}
    (output / 'pack_info.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
    with (output / 'selection.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['source_id', 'detection_id', 'selection_group', 'v5_error', 'v7_error', 'original_crop_path', 'width', 'height'])
        writer.writeheader()
        for row in manifest:
            writer.writerow({'source_id': row['source_id'], 'detection_id': row['detection_id'],
                             'selection_group': row['selection_group'], 'v5_error': row['review_record']['v5_primary_error'],
                             'v7_error': row['review_record']['v7_primary_error'],
                             'original_crop_path': str(assets / row['review_record']['original_path']),
                             'width': row['annotation_size'][0], 'height': row['annotation_size'][1]})
    assert all(sha256(Path(path)) == digest for path, digest in old_hashes.items())
    print(json.dumps({k: v for k, v in info.items() if k != 'old_pack_file_hashes'}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets-root', type=Path, default=ASSETS)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs/ruler_v7_additional_train22_v1')
    parser.add_argument('--previous-pack', type=Path, default=ROOT / 'outputs/ruler_v7_reference12_v1')
    args = parser.parse_args()
    build(args.assets_root.resolve(), args.output_dir.resolve(), args.previous_pack.resolve())


if __name__ == '__main__':
    main()
