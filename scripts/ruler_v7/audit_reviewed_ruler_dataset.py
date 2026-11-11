#!/usr/bin/env python3
"""Independently verify exported ruler data and build a read-only visual gallery."""

import argparse
import csv
import html
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw

from build_reviewed_ruler_dataset import ROOT, ROLES, digest


def read_csv(path):
    with path.open(encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))


def audit(output):
    Image.MAX_IMAGE_PIXELS = None
    manifest = json.loads((output / 'candidate_manifest.json').read_text())
    by_id = {c['case_id']: c for c in manifest['cases']}
    rows = read_csv(output / 'export_manifest.csv')
    blocked = read_csv(output / 'blocked_cases.csv')
    covered = read_csv(output / 'covered_cases.csv')
    for filename, expected in {**manifest['original_annotation_hashes'], **manifest['source_image_hashes']}.items():
        if digest(Path(filename)) != expected:
            raise ValueError(f'Original input changed: {filename}')
    statuses = [r['case_id'] for group in (rows, blocked, covered) for r in group]
    if len(statuses) != 34 or set(statuses) != set(by_id) or len(set(statuses)) != 34:
        raise ValueError('Each of the 34 references needs exactly one final disposition')
    source_splits = defaultdict(set)
    image_hashes = set()
    polygons = Counter()
    max_error = 0.0
    overlays = output / 'audit/exported'
    overlays.mkdir(exist_ok=True)
    for source in sorted(ROLES):
        source_rows = [r for r in rows if r['source_id'] == source]
        if not source_rows:
            continue
        with Image.open(by_id[source_rows[0]['case_id']]['source_image']) as full:
            for row in source_rows:
                case = by_id[row['case_id']]
                split = row['split']
                if split != ROLES[source]:
                    raise ValueError('Unexpected source split')
                source_splits[source].add(split)
                rect = json.loads(row['crop_source_xyxy'])
                image_path = output / row['image']
                label_path = output / row['label']
                with Image.open(image_path) as image:
                    image = image.convert('RGB')
                    w, h = image.size
                    if (w, h) != (rect[2]-rect[0], rect[3]-rect[1]):
                        raise ValueError('Dataset image was resized')
                    if ImageChops.difference(image, full.crop(rect).convert('RGB')).getbbox():
                        raise ValueError('Exported training image differs from unpainted source crop')
                hashed = digest(image_path)
                if hashed in image_hashes:
                    raise ValueError('Duplicate exported image bytes')
                image_hashes.add(hashed)
                ids = row['included_manual_cases'].split(';')
                lines = label_path.read_text().splitlines()
                if len(lines) != len(ids) or len(lines) != int(row['polygon_count']):
                    raise ValueError('Wrong number of polygon instances')
                draw = ImageDraw.Draw(image, 'RGBA')
                for key, line in zip(ids, lines):
                    parts = line.split()
                    values = list(map(float, parts[1:]))
                    if parts[0] != '0' or len(values) < 6 or len(values) % 2 or not all(math.isfinite(v) and 0 <= v <= 1 for v in values):
                        raise ValueError('Invalid YOLO-seg line')
                    reference = by_id[key]
                    if reference['source_id'] != source or reference['split'] != split:
                        raise ValueError('Cross-source manual label')
                    points = [[values[i]*w, values[i+1]*h] for i in range(0,len(values),2)]
                    original = reference['points_source_xy']
                    if len(points) != len(original):
                        raise ValueError('Manual vertices were simplified/dropped')
                    error = max(abs(a-b) for old,new in zip(original,points)
                                for a,b in zip(old,[new[0]+rect[0],new[1]+rect[1]]))
                    max_error = max(max_error, error)
                    if error > 1e-5:
                        raise ValueError('Manual geometry changed')
                    color = (0,255,120,255) if key==row['case_id'] else (0,220,255,255)
                    draw.polygon([tuple(p) for p in points], fill=(*color[:3],30), outline=color)
                    polygons[split] += 1
                image.save(overlays / (row['case_id']+'_yolo_overlay.png'))
    for split in ('train','val','test'):
        images = {p.stem for p in (output/'yolo_seg/images'/split).glob('*.png')}
        labels = {p.stem for p in (output/'yolo_seg/labels'/split).glob('*.txt')}
        if images != labels:
            raise ValueError('Unpaired images/labels')
    if any(len(splits)!=1 for splits in source_splits.values()):
        raise ValueError('Source appears in more than one split')
    report = {
        'references_accounted_for': 34, 'exported_images': len(rows),
        'images_by_split': dict(Counter(r['split'] for r in rows)), 'polygons_by_split': dict(polygons),
        'blocked_for_neighbour_or_boundary_review': len(blocked), 'covered_in_two_instance_crop': len(covered),
        'max_manual_vertex_roundtrip_error_px': max_error,
        'source_splits': {s: sorted(v) for s,v in source_splits.items()},
        'source_overlap': [], 'images_unresized_and_pixel_identical_to_source': True,
        'unique_exported_images': True, 'paired_images_labels': True,
        'original_manual_annotation_hashes_unchanged': True,
        'no_prediction_masks_used_as_labels': True,
        'neighbour_completeness_check': 'assistant visual inspection of all 34 contexts and both crop variants; uncertain cases blocked, not a machine proof',
        'status': 'AWAITING_USER_VISUAL_AUDIT_NOT_TRAINED',
    }
    (output / 'independent_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    cards=[]
    accepted={r['case_id']:r for r in rows}
    blocked_map={r['case_id']:r for r in blocked}
    covered_map={r['case_id']:r for r in covered}
    for key,case in by_id.items():
        escape=html.escape
        if key in accepted:
            row=accepted[key];status=case['split'];reason=row['reason']
            first=row['image'];second='audit/exported/'+key+'_yolo_overlay.png'
            heading=f'{status}: {row["polygon_count"]} poligon(a), {row["variant"]}'
        elif key in blocked_map:
            status='blocked';reason=blocked_map[key]['reason'];heading='IZDVOJENO — ne ulazi u dataset'
            first=case['variants']['standard']['overlay'];second=case['variants']['tight']['overlay']
        else:
            status='covered';owner=covered_map[key]['covered_by'];reason=covered_map[key]['reason'];heading='UKLJUČENO U ZAJEDNIČKI ISJEČAK'
            first=accepted[owner]['image'];second='audit/exported/'+owner+'_yolo_overlay.png'
        cards.append(f'<section data-status="{status}"><h2>{escape(key)}</h2><b>{escape(heading)}</b>'
                     f'<p>{escape(reason)}</p><div class="images"><a href="{escape(first)}"><img src="{escape(first)}"></a>'
                     f'<a href="{escape(second)}"><img src="{escape(second)}"></a></div>'
                     '<p>Za izvezene: lijevo originalni isječak, desno konture ponovno učitane iz YOLO labela. '
                     'Za izdvojene: standardni i uski kandidat, oba samo za audit. Zeleno = ciljni trup, cijan = drugi ručno označeni trup.</p></section>')
    page='''<!doctype html><html lang="hr"><meta charset="utf-8"><title>Ruler YOLO-seg audit</title>
<style>body{background:#111b26;color:#ecf3fa;font:16px system-ui;max-width:1500px;margin:auto;padding:20px}header{position:sticky;top:0;background:#203044;padding:12px}section{background:#1c2b3d;padding:14px;margin:16px 0;border-radius:7px}h2{font-size:19px}button{padding:9px;margin:3px}.images{display:flex;gap:20px}.images a{width:48%;height:380px;background:#080e15}.images img{width:100%;height:100%;object-fit:contain;image-rendering:pixelated}a{color:#98d6ff}pre{white-space:pre-wrap}</style>
<header><h1>Ruler YOLO-seg — kandidat za provjeru, bez treninga</h1>
<p>24 slike / 25 ručnih poligona; train 20/21, val 1/1, test 3/3. Devet slučajeva izdvojeno je za doradu; još jedna jedinka uključena je kao drugi poligon u zajedničkoj slici.</p>
<p>Izolirani dodatak, ne zamjena za osnovni trening dataset. Val/test premali su za pouzdanu ukupnu procjenu. Testne fotografije već su pregledane i nisu slijepi test.</p>
<button onclick="filter('all')">Sve 34 reference</button><button onclick="filter('train')">Train</button><button onclick="filter('val')">Val</button><button onclick="filter('test')">Test</button><button onclick="filter('blocked')">Izdvojeni</button><button onclick="filter('covered')">Zajednički isječak</button>
<p><a href="export_manifest.csv">Manifest izvoza</a> · <a href="blocked_cases.csv">Izdvojeni slučajevi</a> · <a href="independent_audit.json">Neovisne tehničke provjere</a></p></header>'''
    page+='\n'.join(cards)
    page+='<script>function filter(value){for(const s of document.querySelectorAll("section[data-status]"))s.hidden=value!=="all"&&s.dataset.status!==value;}</script></html>'
    (output / 'index.html').write_text(page,encoding='utf-8')
    print(json.dumps(report,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'outputs/ruler_yolo_seg_candidate_v1')
    args=parser.parse_args()
    audit(args.output_dir.resolve())


if __name__=='__main__':
    main()
