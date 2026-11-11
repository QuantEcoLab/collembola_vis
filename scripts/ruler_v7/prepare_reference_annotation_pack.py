#!/usr/bin/env python3
"""Build the explicitly selected 12-case independent annotation pack; no models."""

import argparse
import csv
import hashlib
import json
import math
import shutil
from collections import Counter
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
ASSETS = Path('/home/jana/collembola-rescue/incoming/ruler-v7-20261027T162453Z/snapshot/repo')
SELECTION = [
    ('C_1_2_Collembola005', '382', 'digit_30', 'TOO_SMALL on both models'),
    ('C_1_2_Collembola005', '750', 'ticks_crowded', 'MULTIPLE_PROBLEMS on both models'),
    ('C_1_2_Collembola005', '781', 'digit_20', 'BODY_PART_MISSING on both models'),
    ('C_1_3_Collembola006', '536', 'long_tick', 'TOO_SMALL on both models'),
    ('C_1_3_Collembola006', '620', 'short_ticks', 'MULTIPLE_PROBLEMS on both models'),
    ('C_1_3_Collembola006', '1030', 'digit_20', 'RULER_INCLUDED / MULTIPLE_PROBLEMS'),
    ('C_3_2_Collembola011', '40', 'digit_30', 'MULTIPLE_PROBLEMS on both models'),
    ('C_3_2_Collembola011', '318', 'ticks_vertical_body', 'TOO_SMALL on both models'),
    ('C_4_1_PS_500-1000_014', '328', 'digit_10', 'MULTIPLE_PROBLEMS on both models'),
    ('C_4_1_PS_500-1000_014', '473', 'long_tick', 'MULTIPLE_PROBLEMS on both models'),
    ('C_5_1_PS_500-1000_017', '315', 'ticks_horizontal_body', 'MULTIPLE_PROBLEMS on both models'),
    ('C_5_1_PS_500-1000_017', '365', 'long_tick_vertical_body', 'MULTIPLE_PROBLEMS on both models'),
]
PROPOSED_ROLES = {
    'C_1_2_Collembola005': 'TRAIN_CANDIDATE',
    'C_1_3_Collembola006': 'TRAIN_CANDIDATE',
    'C_3_2_Collembola011': 'VALIDATION_RESERVED',
    'C_4_1_PS_500-1000_014': 'FINAL_TEST_RESERVED',
    'C_5_1_PS_500-1000_017': 'FINAL_TEST_RESERVED',
}


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(candidates, assets, output):
    rows = {(r['source_id'], r['detection_id']): r for r in read_csv(candidates)}
    cut = {(r['source_id'], r['detection_id']) for r in read_csv(candidates.with_name('padding_cases.csv'))}
    sources = {r['source_id']: r['image_path'] for r in read_csv(
        ROOT / 'reports/ruler_v7/unseen/candidate_selection_leakage_audit.csv')}
    if output.exists():
        raise ValueError(f'Refusing to overwrite existing pack: {output}')
    selected = []
    for source, detection, position, reason in SELECTION:
        row = rows[(source, detection)]
        if row['crop_condition'] != 'COMPLETE' or (source, detection) in cut:
            raise ValueError(f'Excluded cut/incomplete case: {source} {detection}')
        if 'UNASSESSABLE' in (row['v5_primary_error'], row['v7_primary_error']):
            raise ValueError('Select only both-assessable cases')
        for name in ['original_path', 'v5_mask_path', 'v7_mask_path']:
            path = (assets / row[name]).resolve()
            if not path.is_relative_to(assets.resolve()) or not path.is_file():
                raise ValueError(f'Unavailable reference: {path}')
        selected.append((row, position, reason))
    Image.MAX_IMAGE_PIXELS = None
    for folder in ['images', 'original_crops', 'context', 'references/v5', 'references/v7', 'annotations', 'annotation_history']:
        (output / folder).mkdir(parents=True, exist_ok=True)
    manifest = []
    for source in PROPOSED_ROLES:
        source_path = assets / sources[source]
        with Image.open(source_path) as full:
            for row, position, reason in selected:
                if row['source_id'] != source:
                    continue
                detection = row['detection_id']
                case_id = f'{source}__det{int(detection):04d}'
                raw = assets / row['original_path']
                with Image.open(raw) as image:
                    width, height = image.size
                x, y = math.floor(float(row['bbox_x1'])), math.floor(float(row['bbox_y1']))
                annotation_box = [max(0, x - 220), max(0, y - 220), min(full.width, x + width + 220), min(full.height, y + height + 220)]
                context_box = [max(0, x - 450), max(0, y - 400), min(full.width, x + width + 450), min(full.height, y + height + 400)]
                names = {'image': f'images/{case_id}.png', 'original': f'original_crops/{case_id}.png',
                         'context': f'context/{case_id}.png',
                         'v5': f'references/v5/{case_id}_mask.png', 'v7': f'references/v7/{case_id}_mask.png'}
                shutil.copy2(raw, output / names['original'])
                full.crop(annotation_box).save(output / names['image'])
                full.crop(context_box).save(output / names['context'])
                for model in ['v5', 'v7']:
                    shutil.copy2(assets / row[model + '_mask_path'], output / names[model])
                manifest.append({
                    'case_id': case_id, 'source_id': source, 'detection_id': detection,
                    'position': position, 'selection_reason': reason,
                    'review_record': row, 'source_image': sources[source], 'assets_root': str(assets),
                    'original_crop_box_source_xyxy': [x, y, x + width, y + height],
                    'annotation_box_source_xyxy': annotation_box, 'context_box_source_xyxy': context_box,
                    'annotation_size': [annotation_box[2]-annotation_box[0], annotation_box[3]-annotation_box[1]],
                    'files': names,
                    'sha256': {key: sha256(output / name) for key, name in names.items()},
                    'proposed_source_role': PROPOSED_ROLES[source],
                    'dataset_assignment': 'NONE_REFERENCE_ONLY',
                })
    # Restore the displayed order, independent of filesystem/source iteration.
    order = {(s, d): i for i, (s, d, _p, _r) in enumerate(SELECTION)}
    manifest.sort(key=lambda r: order[(r['source_id'], r['detection_id'])])
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    split = {'status': 'PROPOSAL_ONLY_NOT_APPLIED', 'sources': PROPOSED_ROLES,
             'counts': dict(Counter(r['proposed_source_role'] for r in manifest)),
             'no_source_overlap': len(PROPOSED_ROLES) == 5,
             'note': 'Reference-only annotations; no dataset membership is changed. Final-test sources were previously visually reviewed, so this is not a blind untouched test. A fresh source-level test is needed for a confirmatory generalization claim.'}
    (output / 'proposed_source_split.json').write_text(json.dumps(split, indent=2), encoding='utf-8')
    print(json.dumps({'pack': str(output), 'cases': len(manifest), 'sources': dict(Counter(r['source_id'] for r in manifest)), 'annotations_created': 0}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidates', type=Path, default=ROOT / 'outputs/ruler_v7_body_error_review/annotation_candidates.csv')
    parser.add_argument('--assets-root', type=Path, default=ASSETS)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs/ruler_v7_reference12_v1')
    args = parser.parse_args()
    prepare(args.candidates.resolve(), args.assets_root.resolve(), args.output_dir.resolve())


if __name__ == '__main__':
    main()
