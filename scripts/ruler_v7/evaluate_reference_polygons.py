#!/usr/bin/env python3
"""Compare saved masks to independent manual polygons, without model execution."""

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from web_reference_annotation import AnnotationPack

ROOT = Path(__file__).resolve().parents[2]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare_mask(gt, raw_mask, raw_box):
    """Translate the original-size mask; never resize it to the annotation image."""
    x1, y1, x2, y2 = raw_box
    if not (0 <= x1 < x2 <= gt.shape[1] and 0 <= y1 < y2 <= gt.shape[0]):
        raise ValueError('Raw crop lies outside annotation coordinates')
    if raw_mask.shape != (y2 - y1, x2 - x1):
        raise ValueError('Raw mask dimensions do not match the recorded source crop')
    gt_area = int(gt.sum())
    if not gt_area:
        raise ValueError('Empty manual reference')
    prediction = np.zeros_like(gt)
    prediction[y1:y2, x1:x2] = raw_mask
    visible_gt = np.zeros_like(gt)
    visible_gt[y1:y2, x1:x2] = gt[y1:y2, x1:x2]
    intersection = int((gt & prediction).sum())
    union = int((gt | prediction).sum())
    pred_area = int(prediction.sum())
    raw_union = int((visible_gt | prediction).sum())
    return {
        'iou': intersection / union if union else 0,
        'dice': 2 * intersection / (gt_area + pred_area),
        'precision': intersection / pred_area if pred_area else 0,
        'recall': intersection / gt_area,
        'prediction_to_reference_area_ratio': pred_area / gt_area,
        'within_raw_crop_iou': intersection / raw_union if raw_union else None,
        'reference_outside_raw_fraction': (gt_area - int(visible_gt.sum())) / gt_area,
    }


def evaluate(pack_path, output):
    pack = AnnotationPack(pack_path)
    state = pack.state()
    if state['completed'] != 12:
        raise ValueError('All twelve independent manual contours must be saved first')
    if output.exists():
        raise ValueError(f'Refusing to overwrite previous diagnostics: {output}')
    results, provenance = [], []
    for case, annotation in zip(pack.cases, state['annotations']):
        if annotation['flags'].get('model_mask_imported') is not False or not annotation['flags'].get('manual_reference'):
            raise ValueError('Expected independent manually drawn reference')
        shape = annotation['shapes'][0]
        if shape['label'] != 'trunk':
            raise ValueError('Expected trunk/body reference class')
        metadata = annotation['reference_metadata']
        for name in ('source_id', 'detection_id', 'annotation_box_source_xyxy'):
            if metadata[name] != case[name]:
                raise ValueError(f'Annotation provenance mismatch: {name}')
        for name, path in case['files'].items():
            if sha256(pack.root / path) != case['sha256'][name]:
                raise ValueError(f'Pack asset changed: {case["case_id"]} {name}')
        reference = Image.new('1', tuple(case['annotation_size']))
        ImageDraw.Draw(reference).polygon([tuple(p) for p in shape['points']], fill=1)
        gt = np.asarray(reference, dtype=bool)
        raw, canvas = case['original_crop_box_source_xyxy'], case['annotation_box_source_xyxy']
        box = (raw[0] - canvas[0], raw[1] - canvas[1], raw[2] - canvas[0], raw[3] - canvas[1])
        annotation_path = pack.annotation_file(case)
        provenance.append({'case_id': case['case_id'], 'annotation_sha256': sha256(annotation_path),
                           'annotation_saved_at': metadata['saved_at'], 'asset_sha256': case['sha256']})
        for model in ('v5', 'v7'):
            with Image.open(pack.file(case, model)) as image:
                mask = np.asarray(image.convert('L')) > 0
            metrics = compare_mask(gt, mask, box)
            results.append({'source_id': case['source_id'], 'detection_id': case['detection_id'],
                            'model': model, 'proposed_source_role_not_applied': case['proposed_source_role'],
                            'vertices': len(shape['points']), 'reference_area_px': int(gt.sum()),
                            'reference_touches_annotation_edge': bool(gt[0].any() or gt[-1].any() or gt[:, 0].any() or gt[:, -1].any()),
                            **metrics})
    metric_names = ('iou', 'dice', 'precision', 'recall', 'prediction_to_reference_area_ratio', 'within_raw_crop_iou')
    grouped = defaultdict(list)
    for row in results:
        grouped[('ALL', row['model'])].append(row)
        grouped[(row['source_id'], row['model'])].append(row)
    aggregates = [{'source_id': source, 'model': model, 'cases': len(rows),
                   **{'mean_' + name: float(np.mean([r[name] for r in rows if r[name] is not None])) for name in metric_names}}
                  for (source, model), rows in sorted(grouped.items())]
    v5 = {r['detection_id'] + '/' + r['source_id']: r['iou'] for r in results if r['model'] == 'v5'}
    v7 = {r['detection_id'] + '/' + r['source_id']: r['iou'] for r in results if r['model'] == 'v7'}
    outside = [r['reference_outside_raw_fraction'] for r in results if r['model'] == 'v5']
    summary = {
        'scope': '12 deliberately selected ruler hard cases; not unbiased general performance, not YOLO mAP',
        'reference_count': 12, 'model_comparisons': 24,
        'rasterization': 'Pillow ImageDraw.polygon at original annotation resolution; no mask resizing',
        'alignment': 'Translate raw masks by source crop origin minus annotation image origin',
        'mean_metrics_by_model_and_source': aggregates,
        'v7_higher_iou_cases': sum(v7[key] > value for key, value in v5.items()),
        'maximum_reference_outside_raw_fraction': max(outside),
        'cases_with_more_than_5_percent_reference_outside_raw_crop': sum(value > .05 for value in outside),
        'provenance': provenance,
        'dataset_membership_changed': False, 'models_run': False,
        'note': 'Source roles are proposals only. These results do not authorize training or production changes; previously reviewed sources are not a blind untouched final test.',
    }
    output.mkdir(parents=True, exist_ok=False)
    for name, rows in [('per_case_metrics.csv', results), ('by_source_model.csv', aggregates)]:
        with (output / name).open('w', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps({key: value for key, value in summary.items() if key != 'provenance'}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pack', type=Path, default=ROOT / 'outputs/ruler_v7_reference12_v1')
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    evaluate(args.pack.resolve(), (args.output_dir or args.pack / 'reference_mask_diagnostics_v1').resolve())


if __name__ == '__main__':
    main()
