#!/usr/bin/env python3
"""Prepare/audit/export manual ruler polygons into a NEW isolated YOLO dataset.

No models are imported or executed. Export requires explicit per-case visual
decisions; crops with cut known manual neighbours cannot be silently exported.
"""

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw

from web_reference_annotation import AnnotationPack, validate_polygon

ROOT = Path(__file__).resolve().parents[2]
ROLES = {
    'C_1_2_Collembola005': 'train', 'C_1_3_Collembola006': 'train',
    'C_3_2_Collembola011': 'val', 'C_4_1_PS_500-1000_014': 'test',
    'C_5_1_PS_500-1000_017': 'test',
}
VARIANTS = {'standard': (0.16, 4), 'tight': (0.04, 2)}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, rows, fields):
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def contains(rect, points):
    l, t, r, b = rect
    return all(l <= x < r and t <= y < b for x, y in points)


def crop_box(points, size, ratio, minimum):
    xs, ys = zip(*points)
    px = max(minimum, math.ceil((max(xs) - min(xs)) * ratio))
    py = max(minimum, math.ceil((max(ys) - min(ys)) * ratio))
    return [max(0, math.floor(min(xs)) - px), max(0, math.floor(min(ys)) - py),
            min(size[0], math.ceil(max(xs)) + px + 1), min(size[1], math.ceil(max(ys)) + py + 1)]


def intersects_polygon(rect, points):
    l, t, r, b = rect
    xs, ys = zip(*points)
    if max(xs) < l or min(xs) >= r or max(ys) < t or min(ys) >= b:
        return False
    mask = Image.new('1', (r-l, b-t))
    ImageDraw.Draw(mask).polygon([(x-l, y-t) for x, y in points], fill=1)
    return mask.getbbox() is not None


def yolo_line(points, rect):
    l, t, r, b = rect
    if not contains(rect, points):
        raise ValueError('Refusing to clip a manual polygon')
    return '0 ' + ' '.join(f'{v:.9f}' for x, y in points
                          for v in ((x-l)/(r-l), (y-t)/(b-t)))


def load_references():
    cases = []
    hashes = {}
    for name in ['ruler_v7_reference12_v1', 'ruler_v7_additional_train22_v1']:
        pack = AnnotationPack(ROOT / 'outputs' / name)
        state = pack.state()
        if state['completed'] != len(pack.cases):
            raise ValueError('Manual annotation pack is incomplete')
        for case, annotation in zip(pack.cases, state['annotations']):
            path = pack.annotation_file(case)
            hashes[str(path)] = digest(path)
            origin = case['annotation_box_source_xyxy']
            points = [[x+origin[0], y+origin[1]] for x, y in annotation['shapes'][0]['points']]
            if annotation['shapes'][0]['label'] != 'trunk':
                raise ValueError('Unexpected annotation class')
            for field in ['source_id', 'detection_id', 'annotation_box_source_xyxy']:
                if annotation['reference_metadata'][field] != case[field]:
                    raise ValueError('Manual annotation provenance mismatch')
            source = Path(case['assets_root']) / case['source_image']
            cases.append({'case_id': case['case_id'], 'source_id': case['source_id'],
                          'detection_id': case['detection_id'], 'split': ROLES[case['source_id']],
                          'annotation': str(path), 'annotation_sha256': hashes[str(path)],
                          'source_image': str(source), 'points_source_xy': points})
    if len(cases) != 34 or len({c['case_id'] for c in cases}) != 34:
        raise ValueError('Expected 34 unique source/detection references')
    return cases, hashes


def draw_overlay(image, rect, included, target):
    result = image.copy()
    draw = ImageDraw.Draw(result, 'RGBA')
    for case in included:
        points = [(x-rect[0], y-rect[1]) for x, y in case['points_source_xy']]
        color = (0, 255, 120, 255) if case['case_id'] == target else (0, 220, 255, 255)
        draw.polygon(points, fill=(*color[:3], 35), outline=color)
    return result


def panel(image, size):
    result = Image.new('RGB', size, '#eeeeee')
    scale = min(size[0]/image.width, size[1]/image.height)
    enlarged = image.resize((max(1, round(image.width*scale)), max(1, round(image.height*scale))), Image.Resampling.NEAREST)
    result.paste(enlarged, ((size[0]-enlarged.width)//2, (size[1]-enlarged.height)//2))
    return result


def prepare(output):
    if output.exists():
        raise ValueError(f'Refusing to overwrite existing audit: {output}')
    Image.MAX_IMAGE_PIXELS = None
    cases, hashes = load_references()
    (output / 'audit/candidates').mkdir(parents=True)
    (output / 'audit/contact_sheets').mkdir(parents=True)
    source_hashes = {}
    for source in sorted(ROLES):
        same = [case for case in cases if case['source_id'] == source]
        path = Path(same[0]['source_image'])
        source_hashes[str(path)] = digest(path)
        with Image.open(path) as full:
            full.load()
            previews = []
            for case in same:
                validate_polygon(case['points_source_xy'], full.width, full.height)
                case['variants'] = {}
                line = Image.new('RGB', (1500, 300), 'white')
                text = ImageDraw.Draw(line)
                text.text((6, 4), case['case_id'] + ' / ' + case['split'], fill='black')
                points = case['points_source_xy']
                xs, ys = zip(*points)
                context = [max(0, math.floor(min(xs))-140), max(0, math.floor(min(ys))-140),
                           min(full.width, math.ceil(max(xs))+141), min(full.height, math.ceil(max(ys))+141)]
                context_image = full.crop(context).convert('RGB')
                draw = ImageDraw.Draw(context_image)
                draw.line([(x-context[0], y-context[1]) for x,y in points]+[(points[0][0]-context[0],points[0][1]-context[1])], fill='lime', width=1)
                line.paste(panel(context_image, (300, 260)), (0, 35))
                for variant_index, (variant, (ratio, minimum)) in enumerate(VARIANTS.items()):
                    rect = crop_box(points, full.size, ratio, minimum)
                    included = [other for other in same if contains(rect, other['points_source_xy'])]
                    partial = [other['case_id'] for other in same if other not in included and intersects_polygon(rect, other['points_source_xy'])]
                    raw = full.crop(rect).convert('RGB')
                    image_path = f'audit/candidates/{case["case_id"]}_{variant}.png'
                    overlay_path = f'audit/candidates/{case["case_id"]}_{variant}_overlay.png'
                    raw.save(output / image_path)
                    overlay = draw_overlay(raw, rect, included, case['case_id'])
                    overlay.save(output / overlay_path)
                    case['variants'][variant] = {'box_source_xyxy': rect,
                        'included_case_ids': [c['case_id'] for c in included],
                        'cut_known_manual_neighbours': partial, 'image': image_path,
                        'image_sha256': digest(output / image_path), 'overlay': overlay_path,
                        'size': list(raw.size)}
                    x = 300 + variant_index*600
                    text.text((x+5, 20), variant + (' CUT-KNOWN '+','.join(partial) if partial else ''), fill='red' if partial else 'black')
                    line.paste(panel(raw, (300, 260)), (x, 35))
                    line.paste(panel(overlay, (300, 260)), (x+300, 35))
                previews.append(line)
            for i in range(0, len(previews), 4):
                sheet = Image.new('RGB', (1500, 300*len(previews[i:i+4])), 'white')
                for j, line in enumerate(previews[i:i+4]):
                    sheet.paste(line, (0, j*300))
                sheet.save(output / f'audit/contact_sheets/{source}_{i//4+1:02d}.jpg', quality=96)
    manifest = {'status': 'AWAITING_EXPLICIT_VISUAL_DECISIONS', 'cases': cases,
                'original_annotation_hashes': hashes, 'source_image_hashes': source_hashes,
                'source_splits': ROLES, 'models_run': False}
    (output / 'candidate_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps({'audit': str(output), 'cases': len(cases), 'source_counts': dict(Counter(c['source_id'] for c in cases))}, indent=2))


def export(output, decisions_path):
    manifest = json.loads((output / 'candidate_manifest.json').read_text())
    decisions = json.loads(decisions_path.read_text())
    cases = manifest['cases']
    by_id = {c['case_id']: c for c in cases}
    if set(decisions) != set(by_id):
        raise ValueError('Every candidate requires an explicit visual decision')
    for filename, expected in {**manifest['original_annotation_hashes'], **manifest['source_image_hashes']}.items():
        if digest(Path(filename)) != expected:
            raise ValueError(f'Original input changed: {filename}')
    dataset = output / 'yolo_seg'
    if dataset.exists():
        raise ValueError('Refusing to overwrite a prior dataset')
    accepted, blocked, covered = [], [], []
    for case in cases:
        decision = decisions[case['case_id']]
        if not decision.get('reason'):
            raise ValueError('Visual review needs a reason')
        choice = decision['variant']
        if choice == 'covered':
            covered.append({'case_id': case['case_id'], 'source_id': case['source_id'],
                            'split': case['split'], 'covered_by': decision['covered_by'], 'reason': decision['reason']})
            continue
        if choice == 'blocked':
            blocked.append({'case_id': case['case_id'], 'source_id': case['source_id'], 'split': case['split'], 'reason': decision['reason']})
            continue
        if choice not in case['variants'] or not decision.get('all_visible_bodies_labeled'):
            raise ValueError('Unreviewed or unsupported variant')
        candidate = case['variants'][choice]
        if candidate['cut_known_manual_neighbours']:
            raise ValueError('Chosen variant cuts an existing manually labeled neighbour')
        if digest(output / candidate['image']) != candidate['image_sha256']:
            raise ValueError('Candidate crop changed after review')
        accepted.append((case, candidate, decision))
    for row in covered:
        owner = next((c for c in accepted if c[0]['case_id'] == row['covered_by']), None)
        if owner is None or row['case_id'] not in owner[1]['included_case_ids']:
            raise ValueError('Covered reference is not fully included in an accepted image')
    accepted_ids = {c['case_id'] for c, _candidate, _decision in accepted}
    covered_ids = {r['case_id'] for r in covered}
    for case, candidate, _decision in accepted:
        for key in candidate['included_case_ids']:
            other = by_id[key]
            if other['source_id'] != case['source_id'] or other['split'] != case['split']:
                raise ValueError('Source-level split leakage')
            if key not in accepted_ids | covered_ids:
                raise ValueError('A blocked/unreviewed manual reference cannot be included as a hidden label')
            yolo_line(other['points_source_xy'], candidate['box_source_xyxy'])
    for split in ('train', 'val', 'test'):
        (dataset / 'images' / split).mkdir(parents=True)
        (dataset / 'labels' / split).mkdir(parents=True)
    rows = []
    for case, candidate, decision in accepted:
        split = case['split']
        image_path = dataset / 'images' / split / (case['case_id']+'.png')
        # Copy exact bytes; the exported training image has no audit markings.
        image_path.write_bytes((output / candidate['image']).read_bytes())
        rect = candidate['box_source_xyxy']
        included = [by_id[k] for k in candidate['included_case_ids']]
        if any(c['source_id'] != case['source_id'] or c['split'] != split for c in included):
            raise ValueError('Source-level split leakage')
        label_path = dataset / 'labels' / split / (case['case_id']+'.txt')
        lines = [yolo_line(c['points_source_xy'], rect) for c in included]
        label_path.write_text('\n'.join(lines)+'\n', encoding='utf-8')
        for c, line in zip(included, lines):
            values = list(map(float, line.split()[1:]))
            restored = [(values[i]*(rect[2]-rect[0])+rect[0], values[i+1]*(rect[3]-rect[1])+rect[1]) for i in range(0,len(values),2)]
            if len(restored) != len(c['points_source_xy']) or any(abs(a-b) > 1e-5 for old,new in zip(c['points_source_xy'],restored) for a,b in zip(old,new)):
                raise ValueError('Export changed manual polygon geometry')
        rows.append({'case_id': case['case_id'], 'source_id': case['source_id'], 'detection_id': case['detection_id'],
                     'split': split, 'variant': decision['variant'], 'crop_source_xyxy': json.dumps(rect),
                     'included_manual_cases': ';'.join(candidate['included_case_ids']), 'polygon_count': len(lines),
                     'image': image_path.relative_to(output).as_posix(), 'label': label_path.relative_to(output).as_posix(),
                     'reason': decision['reason']})
    yaml = f'path: {dataset.as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: trunk\n'
    (dataset / 'dataset.yaml').write_text(yaml, encoding='utf-8')
    fields = ['case_id','source_id','detection_id','split','variant','crop_source_xyxy','included_manual_cases','polygon_count','image','label','reason']
    write_csv(output / 'export_manifest.csv', rows, fields)
    write_csv(output / 'blocked_cases.csv', blocked, ['case_id','source_id','split','reason'])
    write_csv(output / 'covered_cases.csv', covered, ['case_id','source_id','split','covered_by','reason'])
    summary = {'status': 'PREPARED_FOR_REVIEW_NOT_TRAINED', 'input_references': len(cases),
               'exported_images': len(rows), 'blocked_cases': len(blocked), 'covered_without_duplicate_image': len(covered),
               'images_by_split': dict(Counter(r['split'] for r in rows)),
               'polygons_by_split': {s: sum(r['polygon_count'] for r in rows if r['split']==s) for s in ROLES.values()},
               'source_splits': ROLES, 'source_overlap': [], 'geometry_roundtrip_tolerance_px': 1e-5,
               'exported_unique_manual_references': len(set(k for r in rows for k in r['included_manual_cases'].split(';'))),
               'six_old_training_ruler_scenes_are_independent_test': False,
               'test_sources_previously_visually_reviewed': True, 'training_run': False}
    (output / 'validation_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


def add_pair(output, keys):
    """Make one candidate with two existing manual labels; never pseudo-label."""
    path = output / 'candidate_manifest.json'
    manifest = json.loads(path.read_text())
    cases = manifest['cases']
    pair = [next(c for c in cases if c['case_id'] == key) for key in keys]
    if len(pair) != 2 or len({c['source_id'] for c in pair}) != 1:
        raise ValueError('Choose two manually labeled bodies from one source')
    primary = pair[0]
    same = [c for c in cases if c['source_id'] == primary['source_id']]
    Image.MAX_IMAGE_PIXELS = None
    with Image.open(primary['source_image']) as full:
        rect = crop_box(pair[0]['points_source_xy'] + pair[1]['points_source_xy'], full.size, .08, 3)
        included = [c for c in same if contains(rect, c['points_source_xy'])]
        partial = [c['case_id'] for c in same if c not in included and intersects_polygon(rect, c['points_source_xy'])]
        raw = full.crop(rect).convert('RGB')
        name = f'audit/candidates/{primary["case_id"]}_paired.png'
        overlay = f'audit/candidates/{primary["case_id"]}_paired_overlay.png'
        raw.save(output / name)
        draw_overlay(raw, rect, included, primary['case_id']).save(output / overlay)
        primary['variants']['paired'] = {'box_source_xyxy': rect, 'included_case_ids': [c['case_id'] for c in included],
            'cut_known_manual_neighbours': partial, 'image': name, 'overlay': overlay,
            'image_sha256': digest(output / name), 'size': list(raw.size)}
    path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(primary['variants']['paired'], indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs/ruler_yolo_seg_candidate_v1')
    parser.add_argument('--export-decisions', type=Path, help='Export only after explicit per-case visual review')
    parser.add_argument('--pair-cases', nargs=2, help='Add one two-body crop using two existing manual polygons')
    args = parser.parse_args()
    if args.pair_cases:
        add_pair(args.output_dir.resolve(), args.pair_cases)
    elif args.export_decisions:
        export(args.output_dir.resolve(), args.export_decisions.resolve())
    else:
        prepare(args.output_dir.resolve())


if __name__ == '__main__':
    main()
