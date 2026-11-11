#!/usr/bin/env python3
"""Build a new, read-only-to-inputs pilot report from 12 frozen exp03 masks.

No model loading, no jobs, no application integration, no full-dataset measurement.
"""
import argparse
import csv
import hashlib
import html
import json
import math
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.measure import regionprops

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.experimental_centerline import Config, analyze
from scripts.measure_organisms import measure_organism_from_mask

# Frozen after inspecting a contact sheet of 22 candidates, before running the
# experimental geometry. These visual categories are not biological ground truth.
SAMPLE = [
    (596, 'ravna', 'Gotovo ravna, uska jedinka; bez dodira ruba.'),
    (492, 'ravna', 'Izdužena gotovo ravna jedinka; bez dodira ruba.'),
    (1, 'umjereno zakrivljena', 'Šire tijelo, blaga/umjerena zakrivljenost, dodir ruba.'),
    (4, 'umjereno zakrivljena', 'Umjerena zakrivljenost uz bočne izdanke i dodir ruba.'),
    (321, 'umjereno zakrivljena', 'Manja izdužena jedinka; bez dodira ruba.'),
    (0, 'jako zakrivljena', 'Velika C-maska s nogama/izdancima i dodirom ruba; prethodni primjer.'),
    (22, 'jako zakrivljena', 'C-maska s tankim izdancima, bez dodira ruba.'),
    (84, 'jako zakrivljena', 'Zakrivljeno tijelo s nogama i tankim spojem preko praznine; moguća petlja.'),
    (639, 'jako zakrivljena', 'Tanka zakrivljena maska: test stabilnosti izdvajanja tijela.'),
    (273, 'problematična', 'Razgranata maska s vidljivim bočnim izdancima.'),
    (675, 'problematična', 'Kratka/fragmentarna maska: upitna anatomska glavna putanja.'),
    (663, 'problematična', 'Mala U-maska s dodirom ruba.'),
]

WARNINGS = {
    'CROP_EDGE_CONTACT': 'Maska dodiruje rub cropa: tijelo može biti odrezano.',
    'MULTIPLE_COMPONENTS': 'Više odvojenih komponenti: mjeri se kandidat najveće, ostale su isključene.',
    'EMPTY_MASK': 'Prazna maska.', 'BODY_EXTRACTION_EMPTY': 'Nije izdvojeno stabilno tijelo.',
    'EXCESSIVE_BODY_REMOVAL': 'Izdvajanje tijela uklonilo je više od 35% glavne komponente.',
    'BODY_EXTRACTION_SPLIT': 'Izdvajanje tijela razdvojilo je značajne dijelove maske.',
    'BODY_TOO_THIN_FOR_STABLE_WIDTH': 'Premalo piksela širine za stabilan presjek.',
    'NO_RELIABLE_MAIN_PATH': 'Nema pouzdane otvorene glavne putanje između dva kraja.',
    'SKELETON_LOOP_AMBIGUITY': 'Petlja skeleta: nema jednoznačne anatomske putanje.',
    'DISCONNECTED_BODY_SKELETON': 'Skelet izdvojenog tijela nije povezan.',
    'MAIN_PATH_BRANCH_AMBIGUITY': 'Nakon uklanjanja grana ostalo je više od dva kraja.',
    'SMOOTHED_PATH_OUTSIDE_BODY': 'Dio zaglađene putanje izlazi iz izdvojenog tijela.',
    'BODY_ENDPOINT_UNCERTAIN': 'Nije pouzdano određena granica kraja tijela.',
    'LOCAL_WIDTH_SPIKE_POSSIBLE_APPENDAGE': 'Lokalni skok širine: mogući ostatak noge/antene.',
    'OFF_CENTER_PATH_OR_APPENDAGE_WIDTH': 'Izrazito asimetrični presjeci: pomaknuta linija ili bočni izdanak.',
    'INVALID_PERPENDICULAR_CROSS_SECTIONS': 'Neki presjeci nisu valjani; novi volumen nije izračunat.',
    'BRANCH_TERRITORY_CUT_AMBIGUOUS': 'Odvajanje područja grana razdvojilo je putanju ili tijelo.',
    'SUBSTANTIAL_BRANCH_TERRITORY_REMOVAL': 'Značajan dio otvorene maske pripadao je odbačenim granama; provjerite anatomsku granicu.',
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)


def ellipse_points(mask):
    r = regionprops(mask.astype(np.uint8))[0]
    ys, xs = np.nonzero(mask)
    points = np.column_stack([xs, ys]).astype(float)
    center = points.mean(axis=0)
    cov = (points - center).T @ (points - center) / len(points)
    values, vectors = np.linalg.eigh(cov)
    order = np.argsort(values)[::-1]
    t = np.linspace(0, 2 * np.pi, 160)
    ellipse = center + np.column_stack([np.cos(t) * r.axis_major_length / 2, np.sin(t) * r.axis_minor_length / 2]) @ vectors[:, order].T
    return ellipse, center, vectors[:, order[0]], float(r.axis_major_length)


def polyline(points, color, width=1.3, extra=''):
    text = ' '.join(f'{p[0]:.3f},{p[1]:.3f}' for p in points)
    return f'<polyline points="{text}" fill="none" stroke="{color}" stroke-width="{width}" {extra}/>'


def skeleton_svg(mask, color):
    ys, xs = np.nonzero(mask)
    segments = ' '.join(f'M{x},{y}h.7' for y, x in zip(ys, xs))
    return f'<path d="{segments}" stroke="{color}" stroke-width="1.3" fill="none"/>'


def make_svg(mask, result, key):
    h, w = mask.shape
    pad = max(h, w) * .14
    ellipse, center, axis, length = ellipse_points(mask)
    prefix = f'cases/{key}/'
    body = [f'<svg viewBox="{-pad} {-pad} {w + 2 * pad} {h + 2 * pad}" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Elipsa i eksperimentalna geometrija {key}">',
            f'<image href="{prefix}original.png" width="{w}" height="{h}"/>',
            f'<g data-layer="body"><image href="{prefix}body_overlay.png" width="{w}" height="{h}"/></g>',
            '<g data-layer="ellipse">', polyline(ellipse, '#ff9f1c', 1.6, 'stroke-dasharray="4 2"'),
            polyline([center - axis * length / 2, center + axis * length / 2], '#ff9f1c', 1.), '</g>',
            '<g data-layer="raw">', skeleton_svg(result['raw_skeleton_mask'], '#cbd5e1'), '</g>',
            '<g data-layer="removed">', skeleton_svg(result['removed_skeleton_mask'], '#ff2acb'),
            skeleton_svg(result['remaining_off_path_mask'], '#ff2acb'), '</g>']
    sections = result['cross_sections']
    body.append('<g data-layer="widths">')
    for section in sections[::max(1, len(sections) // 24)]:
        if section['left_xy'] is not None and section['right_xy'] is not None:
            body.append(polyline([section['left_xy'], section['right_xy']], '#00ddff', .9))
    body.extend(['</g>', '<g data-layer="path">'])
    if result['centerline_xy']:
        body.append(polyline(result['centerline_xy'], '#5dff50', 1.7))
        for p in (result['centerline_xy'][0], result['centerline_xy'][-1]):
            body.append(f'<circle cx="{p[0]}" cy="{p[1]}" r="2" fill="#ffff00"/>')
    body.extend(['</g>', '</svg>'])
    return ''.join(body)


def metric(value):
    return '—' if value is None else f'{value:.6f}'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job-dir', type=Path, default=ROOT / 'outputs/integration_exp03_local/app_data/web_outputs/657cb33a4398')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit('Refusing to overwrite any existing report: ' + str(args.output))
    job_dir = args.job_dir.resolve()
    job_file = job_dir / 'result.json'
    job = json.loads(job_file.read_text())
    saved = job['result']
    if saved.get('model_name') != 'exp03_no_os':
        raise ValueError('Expected existing exp03 results')
    records_file = Path(saved['mask_records_path'])
    records = {r['detection_id']: r for r in json.loads(records_file.read_text())}
    measurements = {r['detection_id']: r for r in saved['measurements']}
    calibration = saved['calibration_um_per_px']
    photo_file = Path(saved['image_path'])
    source_paths = [job_file, records_file, Path(saved['csv_path']), Path(saved['excel_path']), photo_file,
                    ROOT / 'scripts/measure_organisms.py', ROOT / 'backend/services/contours.py']
    source_paths += [Path(records[i]['mask_path']) for i, _, _ in SAMPLE]
    source_paths += list((ROOT / 'models').glob('*.pt'))
    source_paths += [ROOT / 'outputs/ruler_v7_reviewed367_ft_exp03_no_os/evaluation/fixed_candidate/best.pt']
    before = {str(p): {'sha256': sha(p), 'mtime_ns': p.stat().st_mtime_ns} for p in source_paths}
    args.output.mkdir(parents=True, exist_ok=False)
    out = args.output.resolve()
    config = Config()
    manifest = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'source_job_id': job['id'],
                'source_image': str(photo_file), 'calibration_um_per_px': calibration, 'config': asdict(config),
                'selection': [{'detection_id': i, 'category': category, 'reason': reason} for i, category, reason in SAMPLE],
                'selection_policy': '22 saved-mask crop previews inspected; 12 IDs frozen before experimental measurement. Categories are visual, not GT.',
                'real_samples': len(SAMPLE), 'source_saved_measurement_count': len(measurements),
                'metadata_scan_found_multiple_components': any(r.get('mask_components', 1) > 1 for r in measurements.values()),
                'synthetic_controls': ['multiple_components', 'closed_loop'], 'input_seal': before,
                'implementation_sha256': {name: sha(ROOT / 'scripts' / name) for name in ['experimental_centerline.py', 'compare_exp03_centerline_pilot.py']},
                'training': False, 'inference': False, 'app_integration': False, 'aggregate_policy': 'None included until explicit biological review; no aggregate length/volume statistics.'}
    write_json(out / 'pilot_manifest.json', manifest)
    Image.MAX_IMAGE_PIXELS = None
    photo = Image.open(photo_file)
    photo.load()
    cases = []
    work = []
    for i, category, reason in SAMPLE:
        record = records[i]
        if not record['kept_after_dedup'] or not record['has_mask']:
            raise ValueError('Sample must use retained saved masks')
        mask = np.array(Image.open(record['mask_path']).convert('L')) > 0
        work.append((f'det{i:05d}', mask, photo.crop(record['crop_box']).convert('RGB'), category, reason,
                     {'kind': 'saved_real_mask', 'detection_id': i, 'crop_box': record['crop_box'], 'mask_path': record['mask_path']}, measurements[i]))
    # Explicit controls are separate from all biological observations.
    y, x = np.mgrid[:200, :260]
    multi = (x - np.clip(x, 60, 180))**2 + (y - 100)**2 <= 16**2
    multi[12:21, 12:21] = True
    y, x = np.mgrid[:180, :180]
    ring = (np.hypot(x - 90, y - 90) - 50)**2 <= 8**2
    for name, mask in [('multiple_components', multi), ('closed_loop', ring)]:
        artificial = Image.fromarray(np.where(mask[..., None], np.array([130, 150, 170], dtype=np.uint8), np.array([20, 30, 45], dtype=np.uint8)).astype(np.uint8))
        work.append(('synthetic_' + name, mask, artificial, 'sintetička kontrola',
                     'NIJE STVARNA JEDINKA: provjera sigurnosnog označavanja ' + name,
                     {'kind': 'synthetic_control', 'control': name}, None))
    html_sections = []
    for key, mask, original, category, reason, source, stored in work:
        case_dir = out / 'cases' / key
        case_dir.mkdir(parents=True)
        original.save(case_dir / 'original.png')
        Image.fromarray(mask.astype(np.uint8) * 255).save(case_dir / 'mask.png')
        old = measure_organism_from_mask(mask, calibration, [0, 0, mask.shape[1], mask.shape[0]])
        if stored:
            for field, expected in [('major_axis_px', stored['length_px']), ('minor_axis_px', stored['width_px']), ('volume_um3', stored['volume_mm3'] * 1e9)]:
                if not math.isclose(old[field], expected, rel_tol=1e-10):
                    raise ValueError('Old recomputation differs from saved row: ' + key)
        result = analyze(mask, calibration, config)
        body = result['body_mask']
        Image.fromarray(body.astype(np.uint8) * 255).save(case_dir / 'body.png')
        removed = mask & ~body
        rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
        rgba[body] = [0, 180, 255, 65]
        rgba[removed] = [255, 45, 100, 150]
        Image.fromarray(rgba).save(case_dir / 'body_overlay.png')
        arrays = {k: v for k, v in result.items() if isinstance(v, np.ndarray)}
        for name, array in arrays.items():
            Image.fromarray(array.astype(np.uint8) * 255).save(case_dir / (name + '.png'))
        serial = {k: v for k, v in result.items() if not isinstance(v, np.ndarray)}
        old_metrics = {'length_px': old['major_axis_px'], 'width_px': old['minor_axis_px'],
                       'length_mm': old['length_um'] / 1000, 'volume_mm3': old['volume_um3'] / 1e9,
                       'source': 'verified_against_existing_saved_measurement' if stored else 'same_formula_on_explicit_synthetic_control'}
        item = {'case_id': key, 'category': category, 'reason': reason, 'source': source,
                'old': old_metrics, 'new': serial, 'aggregate_included': False}
        write_json(case_dir / 'result.json', item)
        fieldnames = ['s_px', 'x_px', 'y_px', 'width_px', 'delta_l_px', 'volume_px3', 'width_mm', 'delta_l_mm', 'volume_mm3']
        with (case_dir / 'cross_sections.csv').open('x', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for s in result['cross_sections']:
                row = {k: s[k] for k in fieldnames if k in s}
                row.update(width_mm=s['width_px'] * calibration / 1000 if s['width_px'] is not None else None,
                           delta_l_mm=s['delta_l_px'] * calibration / 1000,
                           volume_mm3=s['volume_px3'] * (calibration / 1000)**3 if s['volume_px3'] is not None else None)
                writer.writerow(row)
        cases.append(item)
        warn = ''.join('<li>' + html.escape(WARNINGS.get(code, code)) + f' <code>{code}</code></li>' for code in result['warnings'])
        if not warn:
            warn = '<li>Nema automatskog geometrijskog upozorenja. Anatomija tijela i krajevi još nisu potvrđeni.</li>'
        rows = ''.join(f'<tr><th>{label}</th><td>{metric(a)}</td><td>{metric(b)}</td></tr>' for label, a, b in [
            ('Duljina (px)', old_metrics['length_px'], result['length_px']),
            ('Duljina (mm)', old_metrics['length_mm'], result['length_mm']),
            ('Volumen (mm³)', old_metrics['volume_mm3'], result['volume_mm3'])])
        html_sections.append(f'''<section class="case" id="{key}" data-category="{html.escape(category)}">
<h2>{key} · {html.escape(category)}</h2><p>{html.escape(reason)}</p>
<p class="{'bad' if result['geometry_unreliable'] else 'pending'}">{'NEPOUZDANO — brojevi su samo dijagnostički kandidati' if result['geometry_unreliable'] else 'GEOMETRIJSKI KANDIDAT — čeka biološki pregled'} · IZVAN ZBIRNE STATISTIKE</p>
<ul>{warn}</ul><table><thead><tr><th>Mjera</th><th>Postojeća elipsa + cilindar</th><th>Eksperimentalna putanja + presjeci</th></tr></thead><tbody>{rows}</tbody></table>
<p>Komponente: {len(result['component_areas_px'])} · radijus otvaranja: {metric(result.get('opening_radius_px'))} px · uklonjene skeletne grane nakon izdvajanja: {len(result['removed_branches'])} · omjer putanja/tetiva: {metric(result.get('tortuosity'))}</p>
<div class="panels"><figure><figcaption>{'Izvorni crop fotografije' if stored else 'Sintetički oblik — nije fotografija'}</figcaption><img src="cases/{key}/original.png" alt="{key} original" loading="lazy"></figure>
<figure><figcaption>Izvorna binarna maska — bez promjena</figcaption><img src="cases/{key}/mask.png" alt="{key} maska" loading="lazy"></figure>
<figure><figcaption>Izdvojeno tijelo (bijelo); geometrijska pretpostavka</figcaption><img src="cases/{key}/body.png" alt="{key} tijelo" loading="lazy"></figure>
<figure class="geometry"><figcaption>Usporedba elipse, putanje, uklanjanja i lokalnih presjeka</figcaption>{make_svg(mask, result, key)}</figure></div>
<p><a href="cases/{key}/result.json">Svi međurezultati JSON</a> · <a href="cases/{key}/cross_sections.csv">Svi lokalni presjeci CSV</a> · <a href="cases/{key}/original.png" target="_blank">Izvorni crop u punoj veličini</a></p></section>''')
        print(key, 'old/new length', metric(old_metrics['length_px']), metric(result['length_px']), 'warnings', result['warnings'], flush=True)
    write_json(out / 'comparison.json', cases)
    document = '''<!doctype html><html lang="hr"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Exp03 — eksperimentalna zakrivljena duljina</title>
<style>body{font:15px/1.5 system-ui;background:#edf2f7;color:#172b45;margin:0}header,main{max-width:1320px;margin:auto;padding:24px}header{background:#fff;border-bottom:3px solid #16a394}.notice{background:#fff2cc;padding:15px;border:1px solid #d9ac38}.case{background:white;border:1px solid #bdcbd8;border-radius:8px;padding:22px;margin:24px 0;scroll-margin-top:10px}.bad{color:#a01629;font-weight:bold}.pending{color:#805b00;font-weight:bold}.panels{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px}figure{margin:0;background:#172b45;color:white;border:1px solid #4a6582;display:flex;flex-direction:column}figcaption{font-size:12px;padding:8px}figure img,figure svg{width:100%;height:330px;object-fit:contain;image-rendering:auto}figure svg{background:#101d2c}table{border-collapse:collapse;margin:15px 0;width:100%}th,td{border:1px solid #bdcbd8;padding:7px;text-align:left}code{font-size:11px;overflow-wrap:anywhere}a{color:#006b85}label{display:inline-block;margin-right:16px}nav a{display:inline-block;margin:3px 9px}details{margin:12px 0}button,select{padding:7px}.legend{padding:10px;background:#172b45;color:white}.synthetic{color:#9919ac}@media(max-width:950px){.panels{grid-template-columns:repeat(2,minmax(0,1fr))}}@media(max-width:550px){.panels{grid-template-columns:1fr}header,main{padding:12px}}</style>
<header><h1>Exp03: pilot zakrivljene središnje linije i lokalnih presjeka</h1>
<p>12 stvarnih spremljenih maski fotografije AS_MOS_Ctrl_4_004.jpg + 2 jasno označene sintetičke kontrole. Kalibracija: <b>7,931 µm/px</b>. Nije obrađen cijeli skup 659 jedinki.</p>
<div class="notice"><b>EKSPERIMENT ZA BIOLOŠKI PREGLED.</b> Nijedan rezultat nije uključen u zbirnu statistiku niti prenesen u aplikaciju. Dodir ruba, više komponenti, petlje i nejasni presjeci znače nepouzdanost. Nema zbirnog prosjeka/zbroja volumena. Izvorni rezultati i modeli ostaju nepromijenjeni.</div>
<details><summary>Kako je definirana nova metoda i što treba pregledati?</summary>
<ol><li>Najveća 8-povezana komponenta; sve ostale ostaju vidljive kao isključene.</li><li>Izotropno morfološko otvaranje radijusa 0,35 × referentni radijus (99. percentil udaljenosti do granice, najmanje 1 px). To je hipoteza odvajanja tankih nogu/antena; nije anatomski klasifikator.</li><li>Skelet tijela, iterativno uklanjanje kratkih tankih i izrazito tankih grana; geodetska glavna putanja. Preostale račve/petlje se označavaju.</li><li>Pikseli se pridružuju najbližem skeletu glavne putanje ili odbačene grane. Područja odbačenih grana isključuju se i iz maske za lokalne širine; nije uklonjena samo linija grane.</li><li>Zaglađivanje linije, produženje do prve granice očišćenog tijela, uzorkovanje približno svakih 1 px.</li><li>Na sredini svakog segmenta presjek okomit na lokalnu tangentu: do prve granice tijela s obje strane. Presjek se ne nastavlja preko praznine do drugog kraka zakrivljene jedinke.</li><li>V = Σ π(wᵢ/2)² Δlᵢ. Debljina = lokalna širina; kružni presjek je prihvaćena pretpostavka projekta.</li></ol>
<p>Za potvrdu pogledajte jesu li oba kraja na tijelu, jesu li noge i antene crveno/magenta isključene i prolaze li tirkizni presjeci samo kroz glavno tijelo. Nisu zajamčeno anatomski isključeni samo zato što geometrijski testovi prolaze. Otvaranje može skratiti tanke krajeve tijela; to treba vizualno procijeniti.</p></details>
<div class="legend">Narančasto: postojeća elipsa i glavna os · zeleno: nova putanja · žuto: krajevi · tirkizno: lokalne širine · crveno: isključeni pikseli · magenta: uklonjene/odbačene skeletne grane · sivo: izvorni skelet</div>
<p>Slojevi: <label><input type="checkbox" checked onchange="layer('ellipse',this.checked)">Elipsa</label><label><input type="checkbox" checked onchange="layer('path',this.checked)">Putanja</label><label><input type="checkbox" checked onchange="layer('widths',this.checked)">Širine</label><label><input type="checkbox" checked onchange="layer('removed',this.checked)">Uklonjeno</label><label><input type="checkbox" checked onchange="layer('body',this.checked)">Tijelo</label><label><input type="checkbox" checked onchange="layer('raw',this.checked)">Izvorni skelet</label></p>
<label>Prikaz: <select onchange="filter(this.value)"><option value="">Svi primjeri</option><option>ravna</option><option>umjereno zakrivljena</option><option>jako zakrivljena</option><option>problematična</option><option>sintetička kontrola</option></select></label>
<p><a href="pilot_manifest.json">Zamrznuti uzorak / parametri / hashovi izvora</a> · <a href="comparison.json">Usporedba JSON</a> · <a href="verification.json">Provjera očuvanosti i izračuna</a></p><nav>'''
    document += ''.join(f'<a href="#{case["case_id"]}">{case["case_id"]}</a>' for case in cases)
    document += '</nav></header><main>' + ''.join(html_sections) + '''</main><script>
function layer(name,on){document.querySelectorAll('[data-layer="'+name+'"]').forEach(n=>n.style.display=on?'':'none')}
function filter(value){document.querySelectorAll('.case').forEach(n=>n.hidden=!!value&&n.dataset.category!==value)}
</script></html>'''
    (out / 'index.html').write_text(document)
    for name, previous in before.items():
        path = Path(name)
        if sha(path) != previous['sha256'] or path.stat().st_mtime_ns != previous['mtime_ns']:
            raise RuntimeError('Source changed during pilot: ' + name)
    verified = {'status': 'READ_ONLY_INPUTS_VERIFIED', 'real_masks_measured': len(SAMPLE), 'synthetic_controls': 2,
                'source_saved_measurements': len(measurements), 'all_input_hashes_and_mtimes_unchanged': True,
                'old_method_recomputed_matches_all_12_saved_rows': True, 'aggregate_included': 0,
                'app_integrated': False, 'training_or_inference_run': False,
                'geometry_flagged_real_cases': [c['case_id'] for c in cases if c['source']['kind'] == 'saved_real_mask' and c['new']['geometry_unreliable']],
                'no_new_length_available': [c['case_id'] for c in cases if c['new']['length_px'] is None]}
    write_json(out / 'verification.json', verified)
    print(json.dumps(verified, indent=2))


if __name__ == '__main__':
    main()
