#!/usr/bin/env python3
"""Freeze and prepare one controlled ruler fine-tuning experiment, never train."""

import argparse
import csv
import hashlib
import io
import itertools
import json
import math
import pickle
import re
import shutil
import statistics
import zipfile
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path

import yaml
from PIL import Image

from build_reviewed_ruler_dataset import yolo_line
from prepare_reference_annotation_pack import ASSETS, ROOT, read_csv

TRAIN_SOURCES = {'C_1_2_Collembola005', 'C_1_3_Collembola006'}
EVAL_SOURCES = {'C_3_2_Collembola011', 'C_4_1_PS_500-1000_014', 'C_5_1_PS_500-1000_017'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Inert:
    """An inert placeholder for pickle tensor/model references; no model code."""
    def __new__(cls, *args, **kwargs):
        return object.__new__(cls)

    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        pass


class MetadataOnly(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) == ('collections', 'OrderedDict'):
            return OrderedDict
        if module in {'builtins', '__builtin__'} and name == 'set':
            return set
        # All other globals, including torch rebuild functions and model classes,
        # resolve to our inert placeholder. No torch import or storage is loaded.
        return Inert

    def persistent_load(self, pid):
        return Inert()


def checkpoint_metadata(path):
    with zipfile.ZipFile(path) as archive:
        payload = archive.read(next(n for n in archive.namelist() if n.endswith('/data.pkl')))
    values = MetadataOnly(io.BytesIO(payload)).load()
    return {key: values.get(key) for key in ('date', 'version', 'epoch', 'train_args', 'train_metrics', 'train_results')}


def base_source(stem):
    if '_ruler_' in stem:
        return stem.split('_ruler_', 1)[0]
    match = re.fullmatch(r'(.+)_det\d+', stem)
    if not match:
        raise ValueError(f'Cannot establish source ID: {stem}')
    return re.sub(r'_(?:audit_good|batch\d+|v5_manual)$', '', match.group(1))


def label_count(path):
    lines = path.read_text(encoding='utf-8').splitlines()
    if not lines or any(not line.strip() for line in lines):
        raise ValueError(f'Empty label/line: {path}')
    for line in lines:
        parts = line.split()
        if len(parts) < 7 or parts[0] != '0':
            raise ValueError(f'Invalid class/polygon: {path}')
        coords = list(map(float, parts[1:]))
        if len(coords) % 2 or not all(math.isfinite(v) and 0 <= v <= 1 for v in coords):
            raise ValueError(f'Invalid coordinates: {path}')
        if len(set(zip(coords[::2], coords[1::2]))) < 3:
            raise ValueError(f'Degenerate polygon: {path}')
        area = abs(sum(x*v-u*y for (x,y),(u,v) in zip(
            list(zip(coords[::2], coords[1::2])),
            list(zip(coords[::2], coords[1::2]))[1:]+list(zip(coords[::2], coords[1::2]))[:1]))) / 2
        if area <= 0:
            raise ValueError(f'Zero-area polygon: {path}')
    return len(lines)


def record(image, label, source, split, origin):
    polygons = label_count(label)
    with Image.open(image) as im:
        dimensions = list(im.size)
        im.verify()
    return {'image_origin': str(image), 'label_origin': str(label), 'source_id': source,
            'split': split, 'origin': origin, 'image_name': image.name, 'label_name': label.name,
            'polygons': polygons, 'image_sha256': sha(image), 'label_sha256': sha(label),
            'image_bytes': image.stat().st_size, 'label_bytes': label.stat().st_size,
            'dimensions': dimensions}


def dataset_records(base):
    result = []
    for split in ('train', 'val'):
        files = [p for p in (base/'images'/split).iterdir() if p.suffix.lower() in {'.png','.jpg','.jpeg'}]
        images = {p.stem:p for p in files}
        labels = {p.stem:p for p in (base/'labels'/split).glob('*.txt')}
        if len(images) != len(files) or images.keys() != labels.keys():
            raise ValueError(f'Collision/unpaired base dataset: {split}')
        result.extend(record(image, labels[stem], base_source(stem), 'train' if split=='train' else 'base_val', 'base_v7')
                      for stem,image in sorted(images.items()))
    return result


def approved_records(addon):
    candidate = json.loads((addon/'candidate_manifest.json').read_text())
    cases = {r['case_id']:r for r in candidate['cases']}
    blocked = {r['case_id'] for r in read_csv(addon/'blocked_cases.csv')}
    if len(blocked) != 9:
        raise ValueError('Expected the nine excluded blocked references')
    for path, expected in candidate['original_annotation_hashes'].items():
        if sha(Path(path)) != expected:
            raise ValueError(f'Approved manual annotation changed: {path}')
    rows = read_csv(addon/'export_manifest.csv')
    audit = json.loads((addon/'independent_audit.json').read_text())
    if audit['images_by_split'] != {'train':20,'val':1,'test':3} or audit['polygons_by_split'] != {'train':21,'val':1,'test':3}:
        raise ValueError('Approved audit counts changed')
    result = []
    for row in rows:
        case = cases[row['case_id']]
        included = row['included_manual_cases'].split(';')
        if blocked.intersection(included) or row['case_id'] in blocked:
            raise ValueError('Blocked reference entered an approved sample')
        variant = case['variants'][row['variant']]
        image, label = addon/row['image'], addon/row['label']
        if sha(image) != variant['image_sha256']:
            raise ValueError('Image changed since visual audit')
        expected = '\n'.join(yolo_line(cases[key]['points_source_xy'], variant['box_source_xyxy']) for key in included)+'\n'
        if label.read_bytes() != expected.encode('utf-8'):
            raise ValueError('Label changed since approved manual export')
        if row['split']=='train' and row['source_id'] not in TRAIN_SOURCES:
            raise ValueError('Non-training source in ruler training addition')
        r = record(image, label, row['source_id'], {'train':'train','val':'ruler_val','test':'ruler_test'}[row['split']], 'approved_manual_ruler')
        r['case_id'] = row['case_id']
        r['included_manual_cases'] = included
        r['frozen_image'] = str(Path(row['image']).relative_to('yolo_seg'))
        r['frozen_label'] = str(Path(row['label']).relative_to('yolo_seg'))
        result.append(r)
    if Counter(r['split'] for r in result) != {'train':20,'ruler_val':1,'ruler_test':3}:
        raise ValueError('Unexpected approved subset')
    # No additional image/label can hide alongside the approved manifest.
    actual = {str(p.relative_to(addon)) for kind in ('images','labels') for p in (addon/'yolo_seg'/kind).rglob('*') if p.is_file()}
    if actual != {row[k] for row in rows for k in ('image','label')}:
        raise ValueError('Unmanifested files in approved dataset')
    return result


def check_records(records):
    splits = sorted({r['split'] for r in records})
    for a,b in itertools.combinations(splits,2):
        left,right = [r for r in records if r['split']==a],[r for r in records if r['split']==b]
        for field in ('source_id','image_sha256','label_sha256'):
            overlap = {r[field] for r in left}&{r[field] for r in right}
            if overlap:
                raise ValueError(f'Cross-split {field} overlap: {a} / {b}: {sorted(overlap)}')
    for split in splits:
        rows = [r for r in records if r['split']==split]
        for field in ('image_name','label_name'):
            if len({r[field] for r in rows}) != len(rows):
                raise ValueError(f'Filename collision: {split}/{field}')
    if any(r['source_id'] in EVAL_SOURCES for r in records if r['split']=='train'):
        raise ValueError('Reserved source in training')
    return {'source_overlap': [], 'cross_split_image_hash_matches': [], 'cross_split_label_hash_matches': [],
            'pairs_and_labels_valid': True, 'filename_collisions': [],
            'counts': {s:{'images':sum(r['split']==s for r in records),
                         'polygons':sum(r['polygons'] for r in records if r['split']==s)} for s in splits}}


def write_csv(path, rows, fields):
    with path.open('w',encoding='utf-8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields,extrasaction='ignore')
        writer.writeheader();writer.writerows(rows)


def prepare(assets, addon, output):
    if output.exists():
        raise ValueError(f'Refusing to overwrite experiment: {output}')
    run = assets/'runs/segment/runs/segment/trunk_seg_v7_ruler_finetune'
    original = yaml.safe_load((run/'args.yaml').read_text())
    relative_data = Path(original['data'].replace('\\','/'))
    if relative_data.is_absolute() or '..' in relative_data.parts:
        raise ValueError('Unexpected recorded dataset path')
    dataset_yaml = assets/relative_data
    if dataset_yaml.parent != assets/'data/trunk_dataset_combined_v7/yolo_seg':
        raise ValueError('Unexpected recorded V7 dataset')
    base = dataset_yaml.parent
    initial = run/'weights/best.pt'
    metadata = checkpoint_metadata(initial)
    recorded_args = {k:v for k,v in original.items() if k!='save_dir'}
    if metadata['train_args'] != recorded_args:
        raise ValueError('Checkpoint/args.yaml training provenance mismatch')
    if metadata['version'] != '8.4.175':
        raise ValueError('Unexpected checkpoint runtime version')
    results = read_csv(run/'results.csv')
    if [int(row['epoch']) for row in results] != list(range(1,11)) or metadata['train_results']['epoch'] != list(range(1,11)):
        raise ValueError('Unexpected training epochs')
    for name,value in metadata['train_metrics'].items():
        if name in results[-1] and not math.isclose(float(results[-1][name]),value,abs_tol=1e-6):
            raise ValueError('Checkpoint metrics differ from training result row')
    base_rows = dataset_records(base)
    approved = approved_records(addon)
    checks = check_records(base_rows+approved)  # Must pass BEFORE creating output.
    if checks['counts'] != {'base_val':{'images':463,'polygons':463}, 'train':{'images':1934,'polygons':1939},
                            'ruler_val':{'images':1,'polygons':1}, 'ruler_test':{'images':3,'polygons':3}}:
        raise ValueError('Unexpected dataset sizes')
    for folder in ['evidence','initial','frozen_ruler_addon','combined_v7_ruler','config','manifests']:
        (output/folder).mkdir(parents=True,exist_ok=True)
    # Preserve historical records verbatim, including timestamps/Windows paths.
    for name in ['args.yaml','results.csv']:
        shutil.copy2(run/name,output/'evidence'/('original_v7_'+name))
    for name in ['export_manifest.csv','blocked_cases.csv','covered_cases.csv','independent_audit.json','audit_decisions.json']:
        shutil.copy2(addon/name,output/'evidence'/name)
    shutil.copy2(dataset_yaml,output/'evidence/original_v7_dataset.yaml')
    shutil.copy2(initial,output/'initial/v7_best.pt')
    frozen=[];combined=[]
    for r in approved:
        for kind in ('image','label'):
            destination=output/'frozen_ruler_addon'/r['frozen_'+kind]
            destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(r[kind+'_origin'],destination)
            if sha(destination)!=r[kind+'_sha256']:
                raise ValueError('Freeze copy failed hash verification')
        frozen.append({'case_id':r['case_id'],'source_id':r['source_id'],
                       'split':{'train':'train','ruler_val':'val','ruler_test':'test'}[r['split']],
                       'polygon_count':r['polygons'],'image':r['frozen_image'],'label':r['frozen_label'],
                       'image_sha256':r['image_sha256'],'label_sha256':r['label_sha256']})
    for r in base_rows+[r for r in approved if r['split']=='train']:
        split='val' if r['split']=='base_val' else 'train'
        copy=dict(r)
        for kind in ('image','label'):
            dest=output/'combined_v7_ruler'/('images' if kind=='image' else 'labels')/split/r[kind+'_name']
            dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(r[kind+'_origin'],dest)
            if sha(dest)!=r[kind+'_sha256']:
                raise ValueError('Combined copy failed hash verification')
            copy[kind]=dest.relative_to(output).as_posix()
        combined.append(copy)
    for filename,root,test in [('combined_v7_ruler/dataset.yaml',output/'combined_v7_ruler',False),
                               ('config/ruler_evaluation.yaml',output/'frozen_ruler_addon',True)]:
        content={'path':str(root),'train':'images/train','val':'images/val','names':{0:'trunk'}}
        if test:content['test']='images/test'
        (output/filename).write_text(yaml.safe_dump(content,sort_keys=False),encoding='utf-8')
    write_csv(output/'manifests/frozen_addon.csv',frozen,['case_id','source_id','split','polygon_count','image','label','image_sha256','label_sha256'])
    (output/'manifests/combined_inventory.json').write_text(json.dumps(combined,indent=2),encoding='utf-8')
    (output/'manifests/preflight.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    distribution=[]
    for split in sorted({r['split'] for r in base_rows+approved}):
        for source in sorted({r['source_id'] for r in base_rows+approved if r['split']==split}):
            same=[r for r in base_rows+approved if r['split']==split and r['source_id']==source]
            distribution.append({'split':split,'source_id':source,'images':len(same),'polygons':sum(r['polygons'] for r in same)})
    write_csv(output/'manifests/source_distribution.csv',distribution,['split','source_id','images','polygons'])
    # Freeze a non-cherry-picked contour subset: at most first+last per base-val source.
    contour=[]
    for source in sorted({r['source_id'] for r in base_rows if r['split']=='base_val'}):
        same=sorted([r for r in combined if r['split']=='base_val' and r['source_id']==source],key=lambda r:r['image_name'])
        for r in ([same[0]] if len(same)==1 else [same[0],same[-1]]):
            contour.append({'phase':'base_validation','source_id':source,'image':r['image'],'image_sha256':r['image_sha256']})
    for r in approved:
        if r['split'] in {'ruler_val','ruler_test'}:
            contour.append({'phase':'ruler_validation' if r['split']=='ruler_val' else 'final_test_only',
                            'source_id':r['source_id'],'image':'frozen_ruler_addon/'+r['frozen_image'],'image_sha256':r['image_sha256']})
    write_csv(output/'manifests/fixed_contour_examples.csv',contour,['phase','source_id','image','image_sha256'])
    config=dict(metadata['train_args'])
    overrides={'model':str(output/'initial/v7_best.pt'),'data':str(output/'combined_v7_ruler/dataset.yaml'),
               'lr0':0.0001,'resume':False,'project':str(output/'training_runs'),'name':'v7_ruler_ft_exp01'}
    config.update(overrides)
    (output/'config/train.yaml').write_text(yaml.safe_dump(config,sort_keys=False),encoding='utf-8')
    (output/'config/changes_from_original.json').write_text(json.dumps({k:{'original':metadata['train_args'][k],'proposed':v} for k,v in overrides.items()},indent=2),encoding='utf-8')
    reset_times=[];last=0
    for row in results:
        elapsed=float(row['time']);reset_times.append(elapsed-last if elapsed>=last else elapsed);last=elapsed
    evidence={'original_run':str(run),'original_dataset_yaml':str(dataset_yaml),'dataset':str(base),
              'original_checkpoint':str(initial),'checkpoint_sha256':sha(initial),'checkpoint_bytes':initial.stat().st_size,
              'frozen_checkpoint':str(output/'initial/v7_best.pt'),'ultralytics_version':metadata['version'],
              'checkpoint_date':metadata['date'],'checkpoint_epoch_field':metadata['epoch'],
              'checkpoint_epoch_note':'epoch=-1 because checkpoint was stripped; embedded results and metrics match epoch10',
              'checkpoint_train_metrics':metadata['train_metrics'],'original_training_args':metadata['train_args'],
              'base_counts':{'train_images':1914,'train_polygons':1918,'val_images':463,'val_polygons':463},
              'original_epoch_seconds':reset_times,'original_total_training_seconds_accounting_for_resume':sum(reset_times),
              'original_mean_epoch_seconds':statistics.mean(reset_times),
              'original_torch_version':'not recorded in available metadata; must record future installed version',
              'base_data_note':'Existing V7 labels preserved byte-for-byte, including their historical reviewed-label provenance. Only the new addition is independently hand-drawn; no new V5/V7 prediction is used as a label.'}
    (output/'evidence/provenance.json').write_text(json.dumps(evidence,indent=2),encoding='utf-8')
    (output/'config/runtime_requirements.txt').write_text('ultralytics==8.4.175\nPyYAML>=6\n',encoding='utf-8')
    # Seal all prepared files. Training writes only NEW run/cache files later.
    sealed={p.relative_to(output).as_posix():sha(p) for p in output.rglob('*') if p.is_file()}
    seal={'status':'PREPARED_NOT_RUN_REQUIRES_EXPLICIT_TRAIN_APPROVAL',
          'files':sealed,'data_preflight':checks,'initial_checkpoint_sha256':sha(initial),
          'required_ultralytics_version':metadata['version'],
          'run_directory':str(output/'training_runs/v7_ruler_ft_exp01'),
          'base_plus_addon_counts':checks['counts'],
          'training_executed':False,'inference_executed':False}
    (output/'experiment_seal.json').write_text(json.dumps(seal,indent=2),encoding='utf-8')
    print(json.dumps({'experiment':str(output),'counts':checks['counts'],'initial_checkpoint_sha256':sha(initial),
                      'prepared_files':len(sealed),'copied_file_bytes':sum(p.stat().st_size for p in output.rglob('*') if p.is_file()),
                      'training_executed':False,'inference_executed':False},indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets-root',type=Path,default=ASSETS)
    parser.add_argument('--addon',type=Path,default=ROOT/'outputs/ruler_yolo_seg_candidate_v1')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'outputs/ruler_v7_targeted_ft_exp01')
    args=parser.parse_args()
    prepare(args.assets_root.resolve(),args.addon.resolve(),args.output_dir.resolve())


if __name__=='__main__':
    main()
