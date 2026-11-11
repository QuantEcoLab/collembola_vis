#!/usr/bin/env python3
"""Prepare native-pixel v2 review crops, frozen inference and lossless history transfer."""

import argparse
import copy
import csv
import json
import math
import shutil
import sqlite3
from collections import Counter
from contextlib import closing
from pathlib import Path

import numpy as np
from PIL import Image

from prepare_controlled_review import (
    ROOT, FIXED_SHA, INITIAL_SHA, ROLES, sha, content_hash, write_new, legacy_helpers,
)
from web_controlled_ruler_review import Catalogue, ReviewStore, POSITIVE
from web_reference_annotation import validate_polygon


def expanded_box(case, padding=220):
    x,y=case['source_origin'];w,h=case['width'],case['height'];iw,ih=case['source_dimensions']
    return [max(0,x-padding),max(0,y-padding),min(iw,x+w+padding),min(ih,y+h+padding)]


def translate(polygons, offset):
    dx,dy=offset
    return [[[x+dx,y+dy] for x,y in polygon] for polygon in polygons]


def sqlite_rows(path):
    with closing(sqlite3.connect(f'file:{path}?mode=ro',uri=True)) as con:
        con.row_factory=sqlite3.Row
        if con.execute('PRAGMA integrity_check').fetchone()[0]!='ok':raise ValueError('Invalid snapshot database')
        return [dict(row) for row in con.execute('SELECT * FROM events ORDER BY id')]


def verify_files(root, seal):
    for name,digest in seal.items():
        if sha(root/name)!=digest:raise ValueError(f'Changed input: {name}')


def prepare(v1, backup, output):
    if output.exists():raise ValueError(f'Refusing to overwrite {output}')
    old=Catalogue(v1)
    verify_files(backup,{'cases.json':sha(v1/'cases.json'),'config.json':sha(v1/'config.json')})
    saved=sqlite_rows(backup/'review.sqlite3');live=sqlite_rows(v1/'review.sqlite3')
    if saved!=live[:len(saved)]:raise ValueError('An existing v1 event changed after backup')
    selection=json.loads((ROOT/'outputs/ruler_v7_targeted_ft_exp02_os5/evaluation/candidate_selection.json').read_text())
    initial=Path(selection['initial_checkpoint']);candidate=Path(old.config['candidate_checkpoint'])
    if sha(initial)!=INITIAL_SHA or sha(candidate)!=FIXED_SHA:raise ValueError('Wrong approved model')
    for folder in ['crops','frozen_inputs','logs','predictions/initial_v7','predictions/exp02_os5','annotation_revisions']:
        (output/folder).mkdir(parents=True,exist_ok=True)
    shutil.copytree(backup,output/'parent_snapshot')
    helper=ROOT/'scripts/ruler_v7/run_final_pipeline_comparison.py'
    config={**old.config,'review_version':'v2_expanded_220px','parent_review':str(v1),
            'parent_backup':str(backup),'parent_database_sha256':sha(backup/'review.sqlite3'),
            'initial_checkpoint':str(initial),'padding_px':220,'crop_or_padding_changed':True,
            'imgsz_policy':'max(640, ceil(max(native_width,native_height)/32)*32)',
            'target_selection':'Maximum mask IoU with original detector crop rectangle; confidence breaks ties; no overlap means no selected target',
            'legacy_postprocessing_source':str(helper),'legacy_postprocessing_source_sha256':sha(helper)}
    external={str(initial):INITIAL_SHA,str(candidate):FIXED_SHA,str(helper):sha(helper)}
    cases=[];inventory=[];Image.MAX_IMAGE_PIXELS=None
    for source in ROLES:
        originals=[c for c in old.cases if c['source_id']==source]
        source_path=Path(originals[0]['source_image']);external[str(source_path)]=sha(source_path)
        with Image.open(source_path) as image:
            full=image.convert('RGB')
            for previous in originals:
                case=copy.deepcopy(previous);box=expanded_box(case);left,top,right,bottom=box
                x,y=case['source_origin'];offset=[x-left,y-top]
                crop=full.crop(box)
                with Image.open(case['original']) as raw:
                    raw_pixels=np.asarray(raw.convert('RGB'))
                restored=np.asarray(crop.crop((offset[0],offset[1],offset[0]+case['width'],offset[1]+case['height'])))
                if not np.array_equal(raw_pixels,restored):raise ValueError(f'Parent crop pixels do not match source: {case["case_id"]}')
                name='crops/'+case['case_id']+'.png';crop.save(output/name,compress_level=2)
                external[case['original']]=case['original_sha256']
                case.update(original=str(output/name),original_sha256=sha(output/name),
                    width=crop.width,height=crop.height,source_origin=[left,top],initial_mask=None,initial_mask_sha256='')
                case['crop_frame']={'version':'v2','padding_px':220,'old_source_origin':[x,y],
                    'old_dimensions':[previous['width'],previous['height']],
                    'old_box_in_new':[offset[0],offset[1],offset[0]+previous['width'],offset[1]+previous['height']],
                    'offset':offset,'parent_crop':previous['original'],'parent_crop_sha256':previous['original_sha256'],
                    'parent_pixels_exact':True,'limited_by_full_image_edge':box!=[x-220,y-220,x+previous['width']+220,y+previous['height']+220]}
                ref=case.get('human_reference')
                if ref:
                    external[ref['path']]=ref['sha256']
                    annotation=json.loads(Path(ref['path']).read_text());origin=ref['original_annotation_box_source']
                    projected=[[[px+origin[0]-left,py+origin[1]-top] for px,py in shape['points']]
                               for shape in annotation['shapes'] if shape['label']=='trunk']
                    safe=bool(projected) and all(0<=px<crop.width and 0<=py<crop.height for p in projected for px,py in p)
                    ref.update(polygons=projected if safe else [],usable_losslessly=safe,same_frame=origin==box,
                        inverse_error_px=0,mapping_note='Exact translation from original human frame to expanded v2 crop; no clipping/scaling' if safe else 'Human frame extends beyond v2 crop; not imported')
                cases.append(case)
                inventory.append({'case_id':case['case_id'],'source_id':source,'detection_id':case['detection_id'],
                    'role':case['role'],'old_x':x,'old_y':y,'new_x':left,'new_y':top,
                    'width':crop.width,'height':crop.height,'crop_sha256':case['original_sha256'],'old_pixels_exact':True})
        print(f'Expanded source {source}: {len(originals)} native-pixel crops',flush=True)
    cases.sort(key=lambda c:c['index']);inventory.sort(key=lambda c:next(x['index'] for x in cases if x['case_id']==c['case_id']))
    with (output/'frozen_inputs/inventory.csv').open('x',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(inventory[0]));writer.writeheader();writer.writerows(inventory)
    config['inventory_sha256']=sha(output/'frozen_inputs/inventory.csv')
    write_new(output/'config.json',config);write_new(output/'prepared_cases.json',cases)
    write_new(output/'frozen_inputs/migration_evidence.json',{
        'C_1_2_Collembola005__det0185':'User explicitly reports in the v2 request that detection 185 has its head cut off by the old crop'})
    write_new(output/'frozen_inputs/external_input_hashes.json',external)
    write_new(output/'preparation_seal.json',{'files':{p.relative_to(output).as_posix():sha(p)
        for p in output.rglob('*') if p.is_file() and not p.is_relative_to(output/'logs')}})
    print(json.dumps({'output':str(output),'cases':len(cases),'roles':dict(Counter(c['role'] for c in cases)),
                      'padding_px':220,'full_resolution_pixel_match':len(cases)},indent=2),flush=True)


def select_target(result, shape, anchor, functions):
    """Associate to the unchanged detection identity, rather than a higher-score neighbour."""
    if result.masks is None or result.boxes is None:return None,None,[]
    x1,y1,x2,y2=anchor;anchor_area=(x2-x1)*(y2-y1);options=[];best=None
    for i in range(len(result.boxes)):
        mask,confidence=functions['result_mask'](result[i],shape)
        if mask is None:continue
        area=int((mask>0).sum());intersection=int((mask[y1:y2,x1:x2]>0).sum())
        overlap=intersection/(area+anchor_area-intersection) if intersection else 0
        options.append({'instance':i,'anchor_mask_iou':overlap,'confidence':confidence,'intersection_px':intersection})
        score=(overlap,confidence or 0)
        if intersection and (best is None or score>best[0]):best=(score,mask,confidence,i)
    return (best[1],best[2],options) if best else (None,None,options)


def infer(output):
    import cv2
    import torch
    import ultralytics
    from ultralytics import YOLO
    if (output/'prediction_seal.json').exists():raise ValueError('Refusing to overwrite completed inference')
    verify_files(output,json.loads((output/'preparation_seal.json').read_text())['files'])
    config=json.loads((output/'config.json').read_text());cases=json.loads((output/'prepared_cases.json').read_text())
    for path,digest in json.loads((output/'frozen_inputs/external_input_hashes.json').read_text()).items():
        if sha(Path(path))!=digest:raise ValueError('External source changed')
    if ultralytics.__version__!='8.4.175':raise ValueError('Unexpected Ultralytics version')
    functions=legacy_helpers(Path(config['legacy_postprocessing_source']));torch.set_num_threads(4)
    all_records={}
    for model,checkpoint,digest in [('initial_v7',config['initial_checkpoint'],INITIAL_SHA),('exp02_os5',config['candidate_checkpoint'],FIXED_SHA)]:
        if sha(Path(checkpoint))!=digest:raise ValueError('Checkpoint changed')
        network=YOLO(checkpoint)
        if network.task!='segment':raise ValueError('Expected segmentation model')
        records=[]
        for number,case in enumerate(cases,1):
            record_path=output/'predictions'/model/(case['case_id']+'.json')
            if record_path.exists():
                record=json.loads(record_path.read_text())
                if record['crop_sha256']!=case['original_sha256'] or record['checkpoint_sha256']!=digest or sha(output/record['mask'])!=record['mask_sha256']:
                    raise ValueError('Partial inference cache mismatch')
                records.append(record);continue
            pixels=cv2.imread(case['original']);size=max(640,math.ceil(max(case['width'],case['height'])/32)*32)
            result=network.predict(pixels,imgsz=size,conf=config['conf'],device='cpu',verbose=False)[0]
            mask,confidence,options=select_target(result,pixels.shape[:2],case['crop_frame']['old_box_in_new'],functions)
            contour=functions['contour_from_mask'](mask)
            proposal=[contour[:,0,:].tolist()] if contour is not None and len(contour)>=3 else []
            binary=np.zeros(pixels.shape[:2],dtype=np.uint8) if mask is None else mask
            components=cv2.connectedComponents((binary>0).astype(np.uint8))[0]-1
            _,hierarchy=cv2.findContours((binary>0).astype(np.uint8),cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)
            holes=sum(int(r[3]>=0) for r in hierarchy[0]) if hierarchy is not None else 0
            name=f'predictions/{model}/{case["case_id"]}_mask.png'
            with (output/name).open('xb') as handle:Image.fromarray(binary).save(handle,format='PNG')
            record={'case_id':case['case_id'],'source_id':case['source_id'],'detection_id':case['detection_id'],'role':case['role'],
                'checkpoint_sha256':digest,'crop_sha256':case['original_sha256'],'mask':name,'mask_sha256':sha(output/name),
                'confidence':confidence,'has_mask':mask is not None,'proposal':proposal,'proposal_sha256':content_hash(proposal),
                'connected_components':components,'holes':holes,'touches_crop_edge':bool(binary[0].any() or binary[-1].any() or binary[:,0].any() or binary[:,-1].any()),
                'imgsz':size,'instance_candidates':options,'target_selection':config['target_selection'],
                'postprocessing':'Frozen legacy result_mask rasterization per instance, original-detection anchor selection, largest external editable contour; full selected mask retained'}
            write_new(record_path,record);records.append(record)
            if number%50==0:print(f'{model}: {number}/{len(cases)}',flush=True)
        all_records[model]=records
        write_new(output/(model+'_manifest.json'),records)
        if sha(Path(checkpoint))!=digest:raise ValueError('Checkpoint changed during inference')
    initial={r['case_id']:r for r in all_records['initial_v7']}
    for case in cases:
        pred=initial[case['case_id']]
        case.update(initial_mask=str(output/pred['mask']),initial_mask_sha256=pred['mask_sha256'],initial_has_prediction=pred['has_mask'])
    write_new(output/'cases.json',cases)
    write_new(output/'prediction_manifest.json',all_records['exp02_os5'])
    write_new(output/'prediction_seal.json',{'files':{p.relative_to(output).as_posix():sha(p)
        for p in (output/'predictions').rglob('*') if p.is_file()},
        'manifest_sha256':sha(output/'prediction_manifest.json'),'initial_manifest_sha256':sha(output/'initial_v7_manifest.json'),
        'checkpoint_sha256':FIXED_SHA,'initial_checkpoint_sha256':INITIAL_SHA,
        'counts':{m:{'success':len(r),'with_mask':sum(p['has_mask'] for p in r)} for m,r in all_records.items()},
        'runtime':{'torch':torch.__version__,'ultralytics':ultralytics.__version__}})
    write_new(output/'frozen_inputs/bundle_seal.json',{'files':{p.relative_to(output).as_posix():sha(p) for p in [
        output/'cases.json',output/'config.json',output/'prepared_cases.json',output/'preparation_seal.json',
        output/'initial_v7_manifest.json',output/'exp02_os5_manifest.json',output/'frozen_inputs/inventory.csv',
        output/'frozen_inputs/external_input_hashes.json',output/'frozen_inputs/migration_evidence.json',*(output/'crops').glob('*.png'),*(output/'parent_snapshot').rglob('*.json'),
        output/'parent_snapshot/review.sqlite3']}})
    print(json.dumps(json.loads((output/'prediction_seal.json').read_text())['counts'],indent=2),flush=True)


def migration_action(state, previous, truncated, user_evidence=None):
    category=state['category'];notes=state['notes'].casefold()
    explicit=any(word in notes for word in ['premali crop','odrezan','odreziv','truncated','cropped','cut off','cut-off'])
    evidence=[]
    if explicit:evidence.append('Explicit saved crop-truncation note: '+state['notes'])
    if (previous['source_id'],previous['detection_id']) in truncated:evidence.append('Previously reviewed TRUNCATED in padding_cases.csv')
    if user_evidence and previous['case_id'] in user_evidence:evidence.append(user_evidence[previous['case_id']])
    if category=='AMBIGUOUS_OR_TRUNCATED' and evidence:return 'REOPEN_CROP_TRUNCATION',evidence
    if category=='UNLABELED_NEIGHBOUR':return 'REOPEN_NEIGHBOUR_CONTEXT',['Expanded context requires a new neighbour review']
    if category=='PENDING':return 'PRESERVE_PENDING_DRAFT',[]
    return 'TRANSFER_DECISION',evidence


def migrate(output):
    if (output/'review.sqlite3').exists():raise ValueError('Refusing to overwrite v2 review database')
    catalogue=Catalogue(output);config=catalogue.config
    old_cases=json.loads((output/'parent_snapshot/cases.json').read_text())
    old_by_id={c['case_id']:c for c in old_cases}
    # A read-only legacy store handle: initialization is intentionally bypassed.
    previous=ReviewStore.__new__(ReviewStore);previous.database=output/'parent_snapshot/review.sqlite3'
    def readonly_connection():return sqlite3.connect(f'file:{previous.database}?mode=ro',uri=True)
    parent_rows=sqlite_rows(previous.database)
    snapshots=[]
    with closing(readonly_connection()) as con:
        con.row_factory=sqlite3.Row
        for case in old_cases:
            state=previous.state(case,con)
            if state['revision']:snapshots.append((state['revision'],case,state))
    store=ReviewStore(catalogue)
    with store.connection() as con:
        con.executemany('INSERT INTO events (id,case_id,kind,created_at,payload,previous_hash,event_hash,artifact,artifact_sha256) VALUES (?,?,?,?,?,?,?,?,?)',
            [tuple(row[k] for k in ['id','case_id','kind','created_at','payload','previous_hash','event_hash','artifact','artifact_sha256']) for row in parent_rows])
        con.execute("INSERT INTO metadata VALUES ('parent_input_fingerprint',?)",(json.loads((output/'parent_snapshot/backup_report.json').read_text())['metadata'][0][1],))
    for row in parent_rows:shutil.copy2(output/'parent_snapshot'/row['artifact'],output/row['artifact'])
    store.verify_history()
    padding_path=ROOT/'outputs/ruler_v7_body_error_review/padding_cases.csv'
    with padding_path.open(newline='') as handle:truncated={(r['source_id'],r['detection_id']) for r in csv.DictReader(handle) if r['crop_condition']=='TRUNCATED'}
    user_evidence=json.loads((output/'frozen_inputs/migration_evidence.json').read_text())
    entries=[];case_by_id={c['case_id']:c for c in catalogue.cases}
    for _,old_case,state in sorted(snapshots,key=lambda value:value[0]):
        case=case_by_id[old_case['case_id']];offset=case['crop_frame']['offset']
        action,evidence=migration_action(state,old_case,truncated,user_evidence)
        migrated=copy.deepcopy(state)
        for key in ['revision','training_candidate']:migrated.pop(key,None)
        migrated['polygons']=translate(state['polygons'],offset)
        migrated['crop_migration']={'parent_case_revision':state['revision'],'parent_label_hash':state['final_label_hash'],
            'action':action,'offset':offset,'evidence':evidence}
        migrated['expanded_context_review_required']=state['category'] in POSITIVE
        for polygon in migrated['polygons']:
            if state['export_geometry_valid']:validate_polygon(polygon,case['width'],case['height'])
        inverse=translate(migrated['polygons'],[-v for v in offset])
        # Floating coordinates are never rounded; roundtrip tolerance only handles binary float arithmetic.
        errors=[abs(a-b) for op,np_ in zip(state['polygons'],inverse) for old,new in zip(op,np_) for a,b in zip(old,new)]
        error=max(errors,default=0)
        if error>1e-9:raise ValueError('Non-lossless polygon translation')
        if action.startswith('REOPEN_'):
            migrated.update(category='PENDING',has_draft=True,locked=False,final_label_hash=None,export_geometry_valid=False)
            if case['role']!='train':migrated['model_exposed']=True
        # Preserve old GT timestamps in copied events; this derived event has its own timestamp.
        label={'source_id':case['source_id'],'detection_id':case['detection_id'],'role':case['role'],
            'category':migrated['category'],'polygons':migrated['polygons'],'notes':migrated['notes'],
            'checks':migrated['checks'],'reference_id':migrated['reference_id'],'label_source':migrated['label_source'],
            'export_geometry_valid':migrated['export_geometry_valid'],'dimensions':[case['width'],case['height']],
            'source_origin':case['source_origin'],'coordinate_system':'ORIGINAL_CROP_PIXELS','label':'trunk'}
        label_hash=content_hash(label)
        if action=='TRANSFER_DECISION':migrated['final_label_hash']=label_hash
        # Old negative decisions retain their unused manual draft in the translated revision too.
        last_decision=next((json.loads(r['payload']) for r in reversed(parent_rows) if r['case_id']==case['case_id'] and r['kind']=='DECISION'),None)
        unused=translate(last_decision.get('unused_draft_polygons',[]),offset) if last_decision else []
        payload={'state':migrated,'label':label,'label_hash':label_hash,'unused_draft_polygons':unused,
            'parent_database_sha256':config['parent_database_sha256'],'input_fingerprint':catalogue.fingerprint,
            'reason':action,'offset':offset,'parent_revision':state['revision'],'inverse_error_px':error}
        store.append(case,'MIGRATE_CROP',payload,store.state(case)['revision'])
        entries.append({'case_id':case['case_id'],'source_id':case['source_id'],'detection_id':case['detection_id'],
            'old_category':state['category'],'new_category':migrated['category'],'action':action,
            'evidence':evidence,'offset':offset,'parent_revision':state['revision'],'inverse_error_px':error})
    report={'parent_backup':config['parent_backup'],'parent_events_preserved':len(parent_rows),
        'original_decisions':dict(Counter(e['old_category'] for e in entries if e['old_category']!='PENDING')),
        'transferred_decisions':sum(e['action']=='TRANSFER_DECISION' for e in entries),
        'reopened_pending':sum(e['action'].startswith('REOPEN_') for e in entries),
        'reopened_truncation':sum(e['action']=='REOPEN_CROP_TRUNCATION' for e in entries),
        'neighbours_for_recheck':sum(e['action']=='REOPEN_NEIGHBOUR_CONTEXT' for e in entries),
        'pending_drafts_preserved':sum(e['action']=='PRESERVE_PENDING_DRAFT' for e in entries),'entries':entries}
    write_new(output/'migration_report.json',report)
    verify(output)
    print(json.dumps({k:v for k,v in report.items() if k!='entries'},indent=2),flush=True)


def verify(output):
    catalogue=Catalogue(output);store=ReviewStore(catalogue)
    parent=sqlite_rows(output/'parent_snapshot/review.sqlite3');current=sqlite_rows(output/'review.sqlite3')
    if current[:len(parent)]!=parent:raise ValueError('Original event rows changed')
    for row in parent:
        if (output/row['artifact']).read_bytes()!=(output/'parent_snapshot'/row['artifact']).read_bytes():raise ValueError('Original artifact changed')
    live=Path(catalogue.config['parent_review'])
    live_rows=sqlite_rows(live/'review.sqlite3')
    if live_rows[:len(parent)]!=parent:raise ValueError('Original v1 event changed')
    for name in ['cases.json','config.json']:
        if sha(live/name)!=sha(output/'parent_snapshot'/name):raise ValueError('Original v1 configuration changed')
    for row in parent:
        if sha(live/row['artifact'])!=row['artifact_sha256']:raise ValueError('Original v1 artifact changed')
    roles=Counter(c['role'] for c in catalogue.cases)
    if roles!={'train':567,'val':85,'test':127}:raise ValueError('Source role/count mismatch')
    by_parent={c['case_id']:c for c in json.loads((output/'parent_snapshot/cases.json').read_text())}
    legacy=ReviewStore.__new__(ReviewStore)
    migration_events={r['case_id']:json.loads(r['payload']) for r in current if r['kind']=='MIGRATE_CROP'}
    checked_transfers=0;supplemental_ids=set()
    with closing(sqlite3.connect(f'file:{output/"parent_snapshot/review.sqlite3"}?mode=ro',uri=True)) as con:
        con.row_factory=sqlite3.Row
        for case in catalogue.cases:
            if case['case_id'] not in migration_events:continue
            payload=migration_events[case['case_id']];migrated=payload['state']
            if payload.get('source_snapshot'):
                snapshot=output/payload['source_snapshot']
                if sha(snapshot/'review.sqlite3')!=payload['source_snapshot_database_sha256']:raise ValueError('Supplemental snapshot changed')
                with closing(sqlite3.connect(f'file:{snapshot/"review.sqlite3"}?mode=ro',uri=True)) as extra:
                    extra.row_factory=sqlite3.Row
                    old_state=legacy.state(by_parent[case['case_id']],extra)
                for original_row in payload['original_parent_events']:
                    supplemental_ids.add(original_row['id'])
                    if sha(snapshot/original_row['artifact'])!=original_row['artifact_sha256']:raise ValueError('Supplemental artifact changed')
                    if next(r for r in live_rows if r['id']==original_row['id'])!=original_row:raise ValueError('Supplemental original event changed')
            else:old_state=legacy.state(by_parent[case['case_id']],con)
            expected=translate(old_state['polygons'],case['crop_frame']['offset'])
            if migrated['polygons']!=expected:raise ValueError('Migrated contour is not a pure translation')
            for key in ['notes','checks','reference_id','first_blind_label_hash']:
                if migrated[key]!=old_state[key]:raise ValueError('Original annotation metadata changed during migration')
            if old_state['model_exposed'] and not migrated['model_exposed']:raise ValueError('Prior model exposure lost')
            if payload['reason']=='TRANSFER_DECISION' and migrated['category']!=old_state['category']:
                raise ValueError('Transferred category changed')
            checked_transfers+=1
    for case in catalogue.cases:
        old=by_parent[case['case_id']]
        if any(case[k]!=old[k] for k in ['source_id','detection_id','role','index']):raise ValueError('Identity changed')
        for model in ['initial_v7','exp02_os5']:
            record=json.loads((output/'predictions'/model/(case['case_id']+'.json')).read_text())
            if record['crop_sha256']!=case['original_sha256']:raise ValueError('Prediction used wrong crop')
            with Image.open(output/record['mask']) as im:
                if im.size!=(case['width'],case['height']):raise ValueError('Wrong mask geometry')
    report={'cases':len(catalogue.cases),'roles':dict(roles),'parent_event_rows_identical':len(parent),
        'parent_artifacts_identical':len(parent),'v1_events_now':len(live_rows),
        'migrated_states_verified':checked_transfers,
        'supplemental_original_events_preserved':len(supplemental_ids),
        'total_original_events_preserved':len(parent)+len(supplemental_ids),
        'v1_unimported_events':len(live_rows)-len(parent)-len(supplemental_ids),
        'v1_additional_events_since_snapshot':len(live_rows)-len(parent),
        'categories':dict(Counter(store.state(c)['category'] for c in catalogue.cases)),
        'prediction_counts':json.loads((output/'prediction_seal.json').read_text())['counts']}
    (output/'logs/verification.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)


def supplement(output,backup):
    """Append late v1 decisions using a new immutable source snapshot, without rebasing v2."""
    catalogue=Catalogue(output);store=ReviewStore(catalogue)
    base=sqlite_rows(output/'parent_snapshot/review.sqlite3');latest=sqlite_rows(backup/'review.sqlite3')
    if latest[:len(base)]!=base:raise ValueError('Supplement diverges from preserved v1 history')
    tail=latest[len(base):]
    if not tail:raise ValueError('No later v1 history to import')
    affected={r['case_id'] for r in tail};existing=sqlite_rows(output/'review.sqlite3')
    if any(r['id']>len(base) and r['case_id'] in affected for r in existing):
        raise ValueError('A late-source case already has v2 edits; explicit conflict resolution required')
    snapshot_name='parent_supplements/'+backup.name
    if (output/snapshot_name).exists():raise ValueError('Supplement already exists')
    shutil.copytree(backup,output/snapshot_name)
    parent_cases={c['case_id']:c for c in json.loads((backup/'cases.json').read_text())}
    legacy=ReviewStore.__new__(ReviewStore);entries=[]
    evidence=json.loads((output/'frozen_inputs/migration_evidence.json').read_text())
    with (ROOT/'outputs/ruler_v7_body_error_review/padding_cases.csv').open(newline='') as handle:
        truncated={(r['source_id'],r['detection_id']) for r in csv.DictReader(handle) if r['crop_condition']=='TRUNCATED'}
    with closing(sqlite3.connect(f'file:{backup/"review.sqlite3"}?mode=ro',uri=True)) as con:
        con.row_factory=sqlite3.Row
        for case in catalogue.cases:
            if case['case_id'] not in affected:continue
            old=legacy.state(parent_cases[case['case_id']],con);offset=case['crop_frame']['offset']
            action,reasons=migration_action(old,parent_cases[case['case_id']],truncated,evidence)
            state=copy.deepcopy(old)
            for key in ['revision','training_candidate']:state.pop(key,None)
            state['polygons']=translate(old['polygons'],offset)
            state['crop_migration']={'parent_case_revision':old['revision'],'parent_label_hash':old['final_label_hash'],
                'action':action,'offset':offset,'evidence':reasons}
            state['expanded_context_review_required']=old['category'] in POSITIVE
            if action.startswith('REOPEN_'):
                state.update(category='PENDING',locked=False,has_draft=True,final_label_hash=None,export_geometry_valid=False)
                if case['role']!='train':state['model_exposed']=True
            label={key:state[key] for key in ['category','polygons','notes','checks','reference_id','label_source','export_geometry_valid']}
            label.update(source_id=case['source_id'],detection_id=case['detection_id'],role=case['role'],label='trunk',
                dimensions=[case['width'],case['height']],source_origin=case['source_origin'],coordinate_system='ORIGINAL_CROP_PIXELS')
            digest=content_hash(label)
            if action=='TRANSFER_DECISION':state['final_label_hash']=digest
            original_events=[r for r in tail if r['case_id']==case['case_id']]
            decision=next((json.loads(r['payload']) for r in reversed(latest) if r['case_id']==case['case_id'] and r['kind']=='DECISION'),{})
            payload={'state':state,'label':label,'label_hash':digest,'reason':action,'offset':offset,
                'parent_revision':old['revision'],'original_parent_events':original_events,
                'source_snapshot':snapshot_name,'source_snapshot_database_sha256':sha(backup/'review.sqlite3'),
                'input_fingerprint':catalogue.fingerprint,'unused_draft_polygons':translate(decision.get('unused_draft_polygons',[]),offset)}
            store.append(case,'MIGRATE_CROP',payload,store.state(case)['revision'])
            entries.append({'case_id':case['case_id'],'action':action,'old_category':old['category'],'new_category':state['category']})
    report={'backup':str(backup),'original_events_preserved_in_supplement':len(tail),'entries':entries,
        'additional_transferred_decisions':sum(e['action']=='TRANSFER_DECISION' for e in entries),
        'additional_reopened_pending':sum(e['action'].startswith('REOPEN_') for e in entries)}
    write_new(output/'supplement_report.json',report)
    verify(output)
    print(json.dumps(report,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['prepare','infer','migrate','supplement','verify'],required=True)
    parser.add_argument('--v1',type=Path,default=ROOT/'outputs/ruler_exp02_os5_manual_validation_v1')
    parser.add_argument('--backup',type=Path)
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/ruler_exp02_os5_manual_validation_v2')
    args=parser.parse_args();output=args.output.resolve()
    if args.stage=='prepare':
        if args.backup is None:parser.error('--backup is required for prepare')
        prepare(args.v1.resolve(),args.backup.resolve(),output)
    elif args.stage=='supplement':
        if args.backup is None:parser.error('--backup is required for supplement')
        supplement(output,args.backup.resolve())
    else:{'infer':infer,'migrate':migrate,'verify':verify}[args.stage](output)


if __name__=='__main__':main()
