#!/usr/bin/env python3
"""Freeze the approved 367 original-crop labels plus original V7 data; never load a model."""

import argparse
import csv
import hashlib
import json
import math
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import yaml
from PIL import Image

from prepare_controlled_review import ROOT, FIXED_SHA, sha, content_hash
from prepare_targeted_ruler_experiment import dataset_records, approved_records, TRAIN_SOURCES, EVAL_SOURCES, label_count
from web_reference_annotation import validate_polygon

DEFAULT_OUTPUT=ROOT/'outputs/ruler_v7_reviewed367_ft_exp03_no_os'
AUDIT=ROOT/'outputs/ruler_train_v1_completed_audit'
PREVIOUS=ROOT/'outputs/ruler_v7_targeted_ft_exp02_os5'


def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as handle:json.dump(value,handle,indent=2,ensure_ascii=False,allow_nan=False)


def write_yaml(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as handle:yaml.safe_dump(value,handle,sort_keys=False,allow_unicode=True)


def csv_file(path,rows,fields):
    with path.open('x',encoding='utf-8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)


def sealed_files(root,seal_name):
    value=json.loads((root/seal_name).read_text())
    for name,digest in value['files'].items():
        path=(root/name).resolve()
        if not path.is_relative_to(root.resolve()) or sha(path)!=digest:raise ValueError(f'Sealed input changed: {name}')
    return value


def encode_polygon(points,width,height):
    """Use all saved vertices, at roundtrip precision; no clipping, simplification or repair."""
    validate_polygon(points,width,height)
    coords=[coordinate for x,y in points for coordinate in (x/width,y/height)]
    text='0 '+' '.join(format(v,'.17g') for v in coords)+'\n'
    decoded=[float(v) for v in text.split()[1:]]
    if not all(math.isfinite(v) and 0<=v<=1 for v in decoded):raise ValueError('Normalized coordinates outside [0,1]')
    restored=[[x*width,y*height] for x,y in zip(decoded[::2],decoded[1::2])]
    if len(restored)!=len(points):raise ValueError('Vertex count changed')
    error=max(abs(a-b) for old,new in zip(points,restored) for a,b in zip(old,new))
    if error>1e-9:raise ValueError('YOLO conversion did not preserve coordinates')
    validate_polygon(restored,width,height)
    return text,error


def pixel_sha(path):
    with Image.open(path) as image:
        rgb=image.convert('RGB')
        return hashlib.sha256(f'RGB:{rgb.width}:{rgb.height}:'.encode()+rgb.tobytes()).hexdigest()


def duplicate_groups(rows,field):
    groups=defaultdict(list)
    for row in rows:groups[row[field]].append(row)
    return [[{'source_id':r['source_id'],'split':r['split'],'image_name':r['image_name']} for r in group]
            for group in groups.values() if len(group)>1]


def inventory_checks(rows):
    for row in rows:
        if 'pixel_sha256' not in row:row['pixel_sha256']=pixel_sha(Path(row['image_origin']))
    image_dupes=duplicate_groups(rows,'image_sha256');pixel_dupes=duplicate_groups(rows,'pixel_sha256')
    if image_dupes or pixel_dupes:raise ValueError('Duplicate images/pixels found: '+json.dumps({'bytes':image_dupes,'pixels':pixel_dupes}))
    splits=sorted({r['split'] for r in rows});overlap=[]
    for i,left in enumerate(splits):
        for right in splits[i+1:]:
            shared={r['source_id'] for r in rows if r['split']==left}&{r['source_id'] for r in rows if r['split']==right}
            if shared:overlap.append({'left':left,'right':right,'sources':sorted(shared)})
    if overlap:raise ValueError('Source leakage: '+json.dumps(overlap))
    if any(r['source_id'] in EVAL_SOURCES for r in rows if r['split']=='train'):raise ValueError('Ruler evaluation source entered training')
    for split in splits:
        group=[r for r in rows if r['split']==split]
        for key in ['image_name','label_name']:
            if len({r[key] for r in group})!=len(group):raise ValueError(f'Filename collision: {split}/{key}')
        if len({Path(r['image_name']).stem for r in group})!=len(group):raise ValueError('Stem collision')
    return {'counts':{split:{'images':sum(r['split']==split for r in rows),
        'labels':sum(r['split']==split for r in rows),'polygons':sum(r['polygons'] for r in rows if r['split']==split)} for split in splits},
        'duplicate_image_bytes':[],'duplicate_decoded_pixels':[],'source_overlap':[],
        'filename_collisions':[],'reserved_sources_in_train':[],
        'source_sets':{split:sorted({r['source_id'] for r in rows if r['split']==split}) for split in splits}}


def inputs(audit):
    sealed_files(audit,'report_seal.json');sealed_files(audit/'backup','seal.json')
    verification=json.loads((audit/'verification.json').read_text())
    if sha(audit/'verification.json')!=(audit/'verification.json.sha256').read_text().split()[0]:raise ValueError('Audit verification changed')
    if verification['report_seal_sha256']!=sha(audit/'report_seal.json'):raise ValueError('Audit seal mismatch')
    manifest=json.loads((audit/'training_candidates_manifest.json').read_text());items=manifest['items']
    if manifest['count']!=367 or len(items)!=367:raise ValueError('Exactly 367 approved samples are required')
    if Counter(r['category'] for r in items)!={'ACCEPTED_UNCHANGED':357,'MANUALLY_CORRECTED':10}:raise ValueError('Approved decision distribution changed')
    if len({r['case_id'] for r in items})!=367 or len({(r['source_id'],r['detection_id']) for r in items})!=367:raise ValueError('Duplicate approved identity')
    blocked=json.loads((audit/'blocked_positive_decisions.json').read_text())['items']
    with (audit/'excluded_decisions_3_4_5.csv').open(newline='') as handle:excluded=list(csv.DictReader(handle))
    if len(blocked)!=36 or len(excluded)!=164 or any(int(r['decision'])!=4 for r in excluded):raise ValueError('Final exclusion set changed')
    forbidden={r['case_id'] for r in blocked+excluded}
    if forbidden&{r['case_id'] for r in items}:raise ValueError('An excluded case entered the approved manifest')
    addons=[]
    for item in items:
        if item['role']!='train' or item['source_id'] not in TRAIN_SOURCES or item['decision'] not in (1,2):raise ValueError('Forbidden addon source/decision')
        if not item['training_candidate'] or not item['geometry_valid'] or item['errors']:raise ValueError('Unapproved geometry')
        image=Path(item['image_path']);artifact=Path(item['label_artifact_path'])
        if sha(image)!=item['expected_image_sha256'] or sha(artifact)!=item['label_artifact_sha256']:raise ValueError('Approved source changed')
        label=json.loads(artifact.read_text())['payload']['label']
        if content_hash(label)!=item['label_sha256'] or label['polygons']!=item['polygons']:raise ValueError('Final polygon changed')
        if len(item['polygons'])!=1:raise ValueError('Exactly one approved trunk polygon per new sample')
        with Image.open(image) as opened:
            if opened.size!=(item['width'],item['height']):raise ValueError('Image dimensions changed')
        text,error=encode_polygon(item['polygons'][0],item['width'],item['height'])
        stem='review367__'+item['case_id']
        addons.append({'case_id':item['case_id'],'source_id':item['source_id'],'detection_id':item['detection_id'],
            'decision':item['decision'],'category':item['category'],'source_revision':item['revision'],
            'source_label_sha256':item['label_sha256'],'source_event_sha256':item['event_sha256'],
            'label_origin':str(artifact),'image_origin':str(image),'split':'train','origin':'approved_original_crop367',
            'image_name':stem+image.suffix.lower(),'label_name':stem+'.txt','dimensions':[item['width'],item['height']],
            'image_sha256':sha(image),'label_sha256':hashlib.sha256(text.encode()).hexdigest(),
            'polygons':1,'vertices':len(item['polygons'][0]),'max_roundtrip_error_px':error,
            'touches_crop_edge':item['touches_crop_edge'],'yolo_text':text})
    provenance_path=ROOT/'outputs/ruler_v7_targeted_ft_exp01/evidence/provenance.json'
    provenance=json.loads(provenance_path.read_text());base=Path(provenance['dataset']);base_rows=dataset_records(base)
    if Counter(r['split'] for r in base_rows)!={'train':1914,'base_val':463}:raise ValueError('Original base split counts changed')
    if sum(r['polygons'] for r in base_rows if r['split']=='train')!=1918:raise ValueError('Original base polygons changed')
    previous_manifest=ROOT/'outputs/ruler_v7_targeted_ft_exp01/manifests/combined_inventory.json'
    previous_seal=json.loads((ROOT/'outputs/ruler_v7_targeted_ft_exp01/experiment_seal.json').read_text())
    if sha(previous_manifest)!=previous_seal['files']['manifests/combined_inventory.json']:raise ValueError('Base reference inventory changed')
    known={(r['split'],r['image_name']):r for r in json.loads(previous_manifest.read_text()) if r['origin']=='base_v7'}
    for row in base_rows:
        reference=known[(row['split'],row['image_name'])]
        for key in ['image_sha256','label_sha256','source_id','polygons','dimensions']:
            if row[key]!=reference[key]:raise ValueError(f'Base input changed: {row["image_name"]}/{key}')
    evaluation=[r for r in approved_records(ROOT/'outputs/ruler_yolo_seg_candidate_v1') if r['split'] in {'ruler_val','ruler_test'}]
    checkpoint=PREVIOUS/'training_runs/v7_ruler_ft_exp02_os5/weights/best.pt'
    if sha(checkpoint)!=FIXED_SHA:raise ValueError('Wrong fixed starting checkpoint')
    checks=inventory_checks(base_rows+addons+evaluation)
    if checks['counts']!={'base_val':{'images':463,'labels':463,'polygons':463},
        'ruler_test':{'images':3,'labels':3,'polygons':3},'ruler_val':{'images':1,'labels':1,'polygons':1},
        'train':{'images':2281,'labels':2281,'polygons':2285}}:raise ValueError('Unexpected combined inventory')
    return {'audit':audit,'base':base,'base_rows':base_rows,'addons':addons,'evaluation':evaluation,
        'checkpoint':checkpoint,'checks':checks,'blocked':blocked,'excluded':excluded,
        'audit_verification':verification,'base_reference_manifest':previous_manifest,'base_provenance':provenance_path}


def clean_record(row):return {k:v for k,v in row.items() if k!='yolo_text'}


def copy_file(source,destination,expected):
    destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists():raise ValueError('Output collision: '+str(destination))
    shutil.copy2(source,destination)
    if sha(destination)!=expected:raise ValueError('Copy hash mismatch: '+str(destination))


def verify(output):
    seal=sealed_files(output,'experiment_seal.json')
    external=json.loads((output/'evidence/external_input_hashes.json').read_text())
    for path,digest in external.items():
        if sha(Path(path))!=digest:raise ValueError('Original input changed: '+path)
    inventory=json.loads((output/'manifests/combined_inventory.json').read_text())
    addon=json.loads((output/'manifests/frozen_addon.json').read_text())
    if len(addon)!=367 or len(inventory)!=2744:raise ValueError('Wrong frozen data size')
    if Counter(r['decision'] for r in addon)!={1:357,2:10}:raise ValueError('Forbidden addon decision')
    for rows in [addon,inventory,json.loads((output/'manifests/ruler_evaluation.json').read_text())]:
        for row in rows:
            image,label=output/row['image'],output/row['label']
            if sha(image)!=row['image_sha256'] or sha(label)!=row['label_sha256']:raise ValueError('Frozen pair changed')
            if label_count(label)!=row['polygons']:raise ValueError('Frozen polygon count mismatch')
    actual={p.relative_to(output).as_posix() for folder in ['frozen_train_addon','combined_v7_reviewed367','frozen_ruler_evaluation']
        for kind in ['images','labels'] for p in (output/folder/kind).rglob('*') if p.is_file()}
    listed={r[k] for r in addon+inventory+json.loads((output/'manifests/ruler_evaluation.json').read_text()) for k in ['image','label']}
    if actual!=listed:raise ValueError('Unmanifested or missing dataset files')
    train_list=(output/'combined_v7_reviewed367/train_images.txt').read_text().splitlines()
    if len(train_list)!=2281 or len(set(train_list))!=2281:raise ValueError('Repeated or missing training paths')
    train=yaml.safe_load((output/'config/train.yaml').read_text())
    if train['resume'] is not False or train['fraction']!=1. or train['model']!=str(output/'initial/exp02_os5_best.pt'):raise ValueError('Unexpected starting mode')
    if train['data']!=str(output/'combined_v7_reviewed367/dataset.yaml'):raise ValueError('Unexpected training dataset')
    if sha(output/'initial/exp02_os5_best.pt')!=FIXED_SHA:raise ValueError('Wrong starting model')
    if (output/'training_runs').exists():raise ValueError('Training run directory already exists')
    return seal


def prepare(audit,output):
    if output.exists():raise ValueError('Refusing to overwrite experiment: '+str(output))
    value=inputs(audit)
    for folder in ['evidence','initial','manifests','config','frozen_train_addon','combined_v7_reviewed367','frozen_ruler_evaluation']:
        (output/folder).mkdir(parents=True,exist_ok=True)
    external={str(audit/'report_seal.json'):sha(audit/'report_seal.json'),str(audit/'backup/review.sqlite3'):sha(audit/'backup/review.sqlite3'),
        str(value['checkpoint']):FIXED_SHA,str(value['base_reference_manifest']):sha(value['base_reference_manifest']),
        str(audit/'verification.json'):sha(audit/'verification.json'),str(audit/'backup/seal.json'):sha(audit/'backup/seal.json')}
    source_db=Path(value['audit_verification'].get('source_review',ROOT/'outputs/ruler_exp02_os5_manual_validation_v1'))/'review.sqlite3'
    if sha(source_db)!=value['audit_verification']['source_database_sha256']:raise ValueError('Original review database changed')
    external[str(source_db)]=sha(source_db)
    for name in ['training_candidates_manifest.json','blocked_positive_decisions.json','excluded_decisions_3_4_5.csv','summary.json','report_seal.json','verification.json']:
        copy_file(audit/name,output/'evidence'/name,sha(audit/name));external[str(audit/name)]=sha(audit/name)
    for name in ['train.yaml','ruler_evaluation.yaml','evaluation_plan.json']:
        copy_file(PREVIOUS/'config'/name,output/'evidence'/('previous_'+name),sha(PREVIOUS/'config'/name))
        external[str(PREVIOUS/'config'/name)]=sha(PREVIOUS/'config'/name)
    copy_file(value['base']/'dataset.yaml',output/'evidence/original_base_dataset.yaml',sha(value['base']/'dataset.yaml'))
    external[str(value['base']/'dataset.yaml')]=sha(value['base']/'dataset.yaml')
    copy_file(value['checkpoint'],output/'initial/exp02_os5_best.pt',FIXED_SHA)
    for recipe in [Path(__file__).resolve(),ROOT/'scripts/ruler_v7/run_reviewed367_experiment.py']:
        copy_file(recipe,output/'evidence'/recipe.name,sha(recipe));external[str(recipe)]=sha(recipe)
    frozen=[];combined=[];evaluation=[]
    for row in value['addons']:
        r=clean_record(row)
        for kind,folder in [('image','images'),('label','labels')]:
            path=output/'frozen_train_addon'/folder/'train'/row[kind+'_name'];path.parent.mkdir(parents=True,exist_ok=True)
            if kind=='image':copy_file(row['image_origin'],path,row['image_sha256'])
            else:
                with path.open('x') as handle:handle.write(row['yolo_text'])
                if sha(path)!=row['label_sha256']:raise ValueError('YOLO serialization changed')
            r[kind]=path.relative_to(output).as_posix()
        frozen.append(r)
        external[row['image_origin']]=row['image_sha256'];external[row['label_origin']]=sha(Path(row['label_origin']))
    for row in value['base_rows']+[dict(r,image_origin=str(output/r['image']),label_origin=str(output/r['label'])) for r in frozen]:
        r=clean_record(row);split='val' if r['split']=='base_val' else 'train'
        for kind,folder in [('image','images'),('label','labels')]:
            path=output/'combined_v7_reviewed367'/folder/split/r[kind+'_name']
            copy_file(r[kind+'_origin'],path,r[kind+'_sha256']);r[kind]=path.relative_to(output).as_posix()
        combined.append(r)
        if row['origin']=='base_v7':
            external[row['image_origin']]=row['image_sha256'];external[row['label_origin']]=row['label_sha256']
    for row in value['evaluation']:
        r=clean_record(row);split='val' if row['split']=='ruler_val' else 'test'
        for kind,folder in [('image','images'),('label','labels')]:
            path=output/'frozen_ruler_evaluation'/folder/split/row[kind+'_name']
            copy_file(row[kind+'_origin'],path,row[kind+'_sha256']);r[kind]=path.relative_to(output).as_posix()
            external[row[kind+'_origin']]=row[kind+'_sha256']
        evaluation.append(r)
    train_images=[str(output/r['image']) for r in combined if r['split']=='train']
    with (output/'combined_v7_reviewed367/train_images.txt').open('x') as handle:handle.write('\n'.join(train_images)+'\n')
    write_yaml(output/'combined_v7_reviewed367/dataset.yaml',{'path':str(output/'combined_v7_reviewed367'),
        'train':'train_images.txt','val':'images/val','names':{0:'trunk'}})
    write_yaml(output/'config/ruler_evaluation.yaml',{'path':str(output/'frozen_ruler_evaluation'),
        'train':None,'val':'images/val','test':'images/test','names':{0:'trunk'}})
    config=yaml.safe_load((PREVIOUS/'config/train.yaml').read_text())
    overrides={'model':str(output/'initial/exp02_os5_best.pt'),'data':str(output/'combined_v7_reviewed367/dataset.yaml'),
        'project':str(output/'training_runs'),'name':'v7_reviewed367_ft_exp03_no_os','resume':False,'exist_ok':False,
        'epochs':10,'patience':10,'batch':16,'imgsz':320,'optimizer':'AdamW','lr0':.00005,'lrf':.1,
        'warmup_epochs':3.,'warmup_bias_lr':0.,'device':'cpu','workers':0,'amp':False,'cache':False,
        'seed':0,'deterministic':True,'fraction':1.,'mosaic':0.,'close_mosaic':0,'translate':.05,'scale':.1,
        'hsv_h':0.,'hsv_s':.2,'hsv_v':.15,'save_period':1}
    config.update(overrides);write_yaml(output/'config/train.yaml',config)
    write_json(output/'config/changes_from_exp02.json',{k:{'previous':yaml.safe_load((PREVIOUS/'config/train.yaml').read_text()).get(k),'proposed':v} for k,v in overrides.items()})
    plan={'status':'PREPARED_NOT_EXECUTED','checkpoint_selection':'Ultralytics best.pt on unchanged base validation only',
        'base_validation':{'images':463,'polygons':463,'data':str(output/'combined_v7_reviewed367/dataset.yaml'),'split':'val'},
        'ruler_validation':{'images':1,'polygons':1,'source_ids':['C_3_2_Collembola011'],'data':str(output/'config/ruler_evaluation.yaml'),'split':'val','evaluation_only':True},
        'ruler_test':{'images':3,'polygons':3,'source_ids':sorted(EVAL_SOURCES-{'C_3_2_Collembola011'}),'data':str(output/'config/ruler_evaluation.yaml'),'split':'test','evaluation_only':True},
        'reference_checkpoint_sha256':FIXED_SHA,'gate':'Base mask mAP50-95 drop <=0.01 absolute versus fixed exp02_os5 in same runtime; separate ruler metrics diagnostic only',
        'test_selection_forbidden':True,'production_replacement_authorized':False}
    write_json(output/'config/evaluation_plan.json',plan)
    write_json(output/'manifests/frozen_addon.json',frozen);write_json(output/'manifests/combined_inventory.json',combined)
    write_json(output/'manifests/ruler_evaluation.json',evaluation);write_json(output/'manifests/preflight.json',value['checks'])
    fields=['case_id','source_id','detection_id','decision','category','source_revision','image','label','dimensions','image_sha256','label_sha256','vertices','max_roundtrip_error_px']
    csv_file(output/'manifests/frozen_addon.csv',frozen,fields)
    distribution=[]
    for origin,rows in [('base',value['base_rows']),('addon',frozen),('combined',combined),('evaluation_only',evaluation)]:
        for split,source_id in sorted({(r['split'],r['source_id']) for r in rows}):
            group=[r for r in rows if r['split']==split and r['source_id']==source_id]
            distribution.append({'dataset':origin,'split':split,'source_id':source_id,'images':len(group),'labels':len(group),'polygons':sum(r['polygons'] for r in group)})
    csv_file(output/'manifests/source_distribution.csv',distribution,['dataset','split','source_id','images','labels','polygons'])
    write_json(output/'evidence/external_input_hashes.json',external)
    runner=ROOT/'scripts/ruler_v7/run_reviewed367_experiment.py'
    command=f'.venv-ruler-ft/bin/python -B {runner.relative_to(ROOT)} --experiment {output.relative_to(ROOT)} --train'
    summary={'status':'PREPARED_NOT_RUN','output':str(output),'base_dataset':str(value['base']),
        'base_train':{'images':1914,'labels':1914,'polygons':1918},'addon':{'images':367,'labels':367,'polygons':367},
        'combined_train':{'images':2281,'labels':2281,'polygons':2285},'base_validation':{'images':463,'labels':463,'polygons':463},
        'separate_ruler_validation':{'images':1,'labels':1,'polygons':1},'separate_ruler_test':{'images':3,'labels':3,'polygons':3},
        'addon_decisions':dict(Counter(r['category'] for r in frozen)),
        'addon_sources':dict(Counter(r['source_id'] for r in frozen)),'source_sets':value['checks']['source_sets'],
        'excluded_blocked_positive':36,'excluded_decision4':164,'previous20_not_appended':True,
        'oversampling':False,'training_list_entries':2281,'unique_training_images':2281,
        'initial_checkpoint_origin':str(value['checkpoint']),'initial_checkpoint':str(output/'initial/exp02_os5_best.pt'),
        'initial_checkpoint_sha256':FIXED_SHA,'required_ultralytics_version':'8.4.175','torch_threads':4,
        'max_addon_coordinate_roundtrip_error_px':max(r['max_roundtrip_error_px'] for r in frozen),
        'base_validation_byte_identical':True,'new_training_run_directory':str(output/'training_runs'/config['name']),
        'training_command':command,'all_parameters':config,'training_started':False,'evaluation_started':False,'inference_started':False}
    write_json(output/'summary.json',summary)
    report=['# Approved original-crop 367-sample experiment — PREPARED, NOT RUN','',
        '| Set | Images | Label files | Polygons |','|---|---:|---:|---:|',
        '| Original base train | 1914 | 1914 | 1918 |','| Frozen approved addition | 367 | 367 | 367 |',
        '| Combined train, no oversampling | 2281 | 2281 | 2285 |','| Unchanged base validation | 463 | 463 | 463 |',
        '| Separate ruler validation | 1 | 1 | 1 |','| Separate ruler test | 3 | 3 | 3 |','',
        '367 samples are the exact approved manifest: 357 decision 1 + 10 decision 2. The 36 blocked positives',
        'and 164 decision-4 samples are permanently excluded from this experiment. No previous 20-sample addon',
        'or 5x repetitions are appended to the original base train. Each combined image occurs once.',
        'Original PNG bytes are copied unchanged. YOLO coordinates use 17 significant digits, all saved vertices,',
        'class 0 (trunk), no padding, clipping, simplification, normalization repair, or new pseudo-label inference.',
        '', '## Source-level split','```json',json.dumps(summary['source_sets'],indent=2),'```',
        '', '## Starting checkpoint',str(value['checkpoint']),f'SHA-256: `{FIXED_SHA}`','',
        '## All explicit training parameters','```yaml',yaml.safe_dump(config,sort_keys=False).rstrip(),'```','',
        'Other library defaults are frozen by Ultralytics 8.4.175. CPU, 4 torch threads. Optimizer is reset',
        '(resume=false), LR is halved versus exp02, warmup bias LR starts at zero, mild color/scale/translation',
        'augmentation, mosaic/mixup/copy-paste off. Base validation selects best.pt; ruler val/test are separate',
        'evaluation-only sets and never part of training or the base validation directory.',
        '', '## Exact future training command — not executed','```bash',command,'```',
        '', '## Outputs and seals','- `frozen_train_addon/`: exactly 367 existing PNG copies and 367 new YOLO-seg labels.',
        '- `combined_v7_reviewed367/`: new independent dataset with original base train/val plus approved addon.',
        '- `manifests/frozen_addon.json` and `.csv`: identities, decisions, dimensions, paths, hashes.',
        '- `manifests/source_distribution.csv`: full source-level counts.',
        '- `config/train.yaml`: complete explicit configuration.',
        '- `config/evaluation_plan.json`: future evaluation plan only.',
        '- `experiment_seal.json`: prepared-file hashes and immutable external input hashes.',
        '- Training output (does not exist yet): `'+summary['new_training_run_directory']+'`.']
    (output/'README.md').write_text('\n'.join(report)+'\n')
    seal={'status':'PREPARED_NOT_RUN','files':{p.relative_to(output).as_posix():sha(p) for p in output.rglob('*') if p.is_file()},
        'initial_checkpoint_sha256':FIXED_SHA,'required_ultralytics_version':'8.4.175',
        'run_directory':summary['new_training_run_directory'],'counts':value['checks']['counts'],
        'training_executed':False,'evaluation_executed':False,'inference_executed':False}
    write_json(output/'experiment_seal.json',seal)
    (output/'experiment_seal.json.sha256').write_text(sha(output/'experiment_seal.json')+'  experiment_seal.json\n')
    verify(output)
    print(json.dumps({k:v for k,v in summary.items() if k!='all_parameters'},indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit',type=Path,default=AUDIT)
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    action=parser.add_mutually_exclusive_group()
    action.add_argument('--preflight',action='store_true',help='Read-only inventory/label check, no output created')
    action.add_argument('--verify',action='store_true',help='Verify an already prepared experiment, no model imports')
    args=parser.parse_args()
    if args.preflight:
        value=inputs(args.audit.resolve());print(json.dumps(value['checks'],indent=2))
    elif args.verify:
        seal=verify(args.output.resolve());print(json.dumps({'sealed_files_verified':len(seal['files']),'counts':seal['counts'],'training_started':False},indent=2))
    else:prepare(args.audit.resolve(),args.output.resolve())


if __name__=='__main__':main()
