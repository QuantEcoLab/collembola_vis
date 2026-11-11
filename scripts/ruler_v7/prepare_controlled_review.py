#!/usr/bin/env python3
"""Freeze the approved 779-case inventory and run only the fixed segmentation model."""

import argparse
import ast
import csv
import hashlib
import json
import math
import os
import shutil
from collections import Counter
from pathlib import Path

from PIL import Image

ROOT=Path(__file__).resolve().parents[2]
FIXED_SHA='9540b6478a8d77fcf6f2aa7db5c639c812e3c91953baa57c58670903221fc958'
INITIAL_SHA='a46e37b85efe2794d1c5d72cabbbb6ef0aab5d5c72cd22d9264faeea46098883'
ROLES={'C_1_2_Collembola005':'train','C_1_3_Collembola006':'train',
       'C_3_2_Collembola011':'val','C_4_1_PS_500-1000_014':'test','C_5_1_PS_500-1000_017':'test'}


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False)


def content_hash(value):return hashlib.sha256(canonical(value).encode()).hexdigest()


def write_new(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as handle:
        json.dump(value,handle,indent=2,ensure_ascii=False)


def prepare(output,inventory,summary_path):
    if output.exists():raise ValueError(f'Refusing to overwrite {output}')
    summary=json.loads(summary_path.read_text())
    checkpoint=Path(summary['candidate_checkpoint'])
    assert sha(checkpoint)==summary['candidate_sha256']==FIXED_SHA
    assert sha(Path(summary['canonical_index']))==summary['canonical_index_sha256']
    # Persist exact original inventory BEFORE any inference/annotation activity.
    frozen=output/'frozen_inputs';frozen.mkdir(parents=True)
    shutil.copy2(inventory,frozen/'inventory.csv')
    shutil.copy2(summary_path,frozen/'inventory_summary.json')
    write_new(frozen/'inventory_hashes.json',{'inventory.csv':sha(inventory),'inventory_summary.json':sha(summary_path)})
    (frozen/'inventory.csv.sha256').write_text(sha(inventory)+'  inventory.csv\n')
    with inventory.open(newline='',encoding='utf-8') as handle:rows=list(csv.DictReader(handle))
    assert len(rows)==779 and len({(r['source_id'],r['detection_id']) for r in rows})==779
    assert Counter(r['role'] for r in rows)=={'train':567,'val':85,'test':127}
    audit=json.loads((ROOT/'outputs/ruler_yolo_seg_candidate_v1/candidate_manifest.json').read_text())
    external={};source_paths={}
    for name,digest in audit['source_image_hashes'].items():
        assert sha(Path(name))==digest
        external[name]=digest
    with (ROOT/'reports/ruler_v7/unseen/candidate_selection_leakage_audit.csv').open(newline='') as f:
        for row in csv.DictReader(f):source_paths[row['source_id']]=row['image_path']
    human={}
    for folder in ['ruler_v7_reference12_v1','ruler_v7_additional_train22_v1']:
        pack=ROOT/'outputs'/folder
        manifest_path=pack/'manifest.json';external[str(manifest_path)]=sha(manifest_path)
        for row in json.loads(manifest_path.read_text()):
            path=pack/'annotations'/(row['case_id']+'.json')
            annotation=json.loads(path.read_text())
            assert annotation['flags']['manual_reference'] and not annotation['flags']['model_mask_imported']
            assert annotation['shapes'][0]['label']=='trunk'
            external[str(path)]=sha(path)
            human[(row['source_id'],row['detection_id'])]=(row,annotation,path)
    cases=[]
    Image.MAX_IMAGE_PIXELS=None
    for inventory_row,row in enumerate(rows):
        source,detection=row['source_id'],row['detection_id'];role=ROLES[source]
        assert row['role']==role
        original=Path(row['original_crop']);assert sha(original)==row['crop_sha256']
        external[str(original)]=row['crop_sha256']
        with Image.open(original) as im:assert im.size==(int(row['width']),int(row['height']))
        # Coordinate provenance comes from the unchanged canonical raw index.
        cases.append({'case_id':f'{source}__det{int(detection):04d}','source_id':source,
                      'detection_id':detection,'role':role,'inventory_row':inventory_row,
                      'original':str(original),'original_sha256':row['crop_sha256'],
                      'width':int(row['width']),'height':int(row['height']),
                      'initial_mask':row['initial_v7_mask'] or None,
                      'initial_mask_sha256':row['initial_v7_mask_sha256'],
                      'legacy_position':row['legacy_position'],'legacy_crop_condition':row['legacy_crop_condition'],
                      'legacy_blocked_warning':row['legacy_blocked_neighbour_warning']})
    with Path(summary['canonical_index']).open(newline='') as f:
        raw={(r['source_id'],str(int(r['detection_id']))):r for r in csv.DictReader(f)}
    refs=Counter()
    for case in cases:
        record=raw[(case['source_id'],case['detection_id'])]
        x,y=math.floor(float(record['bbox_x1'])),math.floor(float(record['bbox_y1']))
        case['source_origin']=[x,y]
        backup=Path('/home/jana/collembola-rescue/incoming/ruler-v7-20261027T162453Z/snapshot/repo')
        source=backup/source_paths[case['source_id']]
        case['source_image']=str(source);case['source_image_sha256']=external[str(source)]
        with Image.open(source) as im:case['source_dimensions']=list(im.size)
        if case['initial_mask']:
            p=Path(case['initial_mask']);assert sha(p)==case['initial_mask_sha256'];external[str(p)]=sha(p)
            with Image.open(p) as im:assert im.size==(case['width'],case['height'])
        match=human.get((case['source_id'],case['detection_id']))
        case['human_reference']=None
        if match:
            origin,annotation,path=match
            points=annotation['shapes'][0]['points'];box=origin['annotation_box_source_xyxy']
            assert annotation['imageWidth']==box[2]-box[0] and annotation['imageHeight']==box[3]-box[1]
            assert annotation['reference_metadata']['source_id']==case['source_id']
            assert annotation['reference_metadata']['detection_id']==case['detection_id']
            projected=[[px+box[0]-x,py+box[1]-y] for px,py in points]
            safe=all(0<=px<case['width'] and 0<=py<case['height'] for px,py in projected)
            inverse=max(abs(a-b) for old,new in zip(points,projected)
                        for a,b in zip(old,[new[0]+x-box[0],new[1]+y-box[1]]))
            safe=safe and inverse<1e-9
            ref={'id':case['case_id']+'_human','path':str(path),'sha256':external[str(path)],
                 'original_annotation_size':[annotation['imageWidth'],annotation['imageHeight']],
                 'original_annotation_box_source':box,'same_frame':box==[x,y,x+case['width'],y+case['height']],
                 'usable_losslessly':safe,'inverse_error_px':inverse,
                 'polygons':[projected] if safe else [],
                 'mapping_note':'exact integer translation, no clipping/scaling' if safe else 'Expanded-frame human geometry is not fully contained in this original crop; not imported or clipped'}
            case['human_reference']=ref;refs['total']+=1;refs['usable']+=safe;refs['unusable']+=not safe
    cases.sort(key=lambda c:({'train':0,'val':1,'test':2}[c['role']],c['source_id'],int(c['detection_id'])))
    for i,case in enumerate(cases):case['index']=i
    helper=ROOT/'scripts/ruler_v7/run_final_pipeline_comparison.py'
    config={'count':779,'roles':ROLES,'candidate_checkpoint':str(checkpoint),'candidate_sha256':FIXED_SHA,
            'initial_checkpoint_sha256':INITIAL_SHA,'imgsz':320,'conf':.15,'device':'cpu',
            'torch_threads':4,'inventory_sha256':sha(frozen/'inventory.csv'),
            'legacy_postprocessing_source':str(helper),'legacy_postprocessing_source_sha256':sha(helper),
            'detector_run':False,'dedup_run':False,'crop_or_padding_changed':False,
            'blind_policy':'Server denies predictions and model-based history for val/test until a manual decision is locked. Revisions after unlock are marked model-exposed.'}
    write_new(output/'cases.json',cases);write_new(output/'config.json',config)
    write_new(frozen/'external_input_hashes.json',external)
    write_new(frozen/'bundle_seal.json',{'files':{p.relative_to(output).as_posix():sha(p)
        for p in output.rglob('*') if p.is_file()},'human_references':dict(refs)})
    for folder in ['predictions/exp02_os5','logs','annotation_revisions']:(output/folder).mkdir(parents=True,exist_ok=True)
    print(json.dumps({'frozen_cases':779,'roles':dict(Counter(c['role'] for c in cases)),
                      'human_references':dict(refs),'inventory_sha256':config['inventory_sha256']},indent=2))


def legacy_helpers(path):
    import cv2
    import numpy as np
    from typing import Any
    tree=ast.parse(path.read_text())
    names={'normalize_mask','result_mask','contour_from_mask'}
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert len(nodes)==3
    namespace={'np':np,'cv2':cv2,'Any':Any}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace)
    return namespace


def infer(output):
    import cv2
    import numpy as np
    import torch
    import ultralytics
    from ultralytics import YOLO
    config=json.loads((output/'config.json').read_text());cases=json.loads((output/'cases.json').read_text())
    seal=json.loads((output/'frozen_inputs/bundle_seal.json').read_text())
    for name,digest in seal['files'].items():assert sha(output/name)==digest
    for name,digest in json.loads((output/'frozen_inputs/external_input_hashes.json').read_text()).items():assert sha(Path(name))==digest
    checkpoint=Path(config['candidate_checkpoint']);assert sha(checkpoint)==FIXED_SHA==config['candidate_sha256']
    assert ultralytics.__version__=='8.4.175' and torch.version.cuda is None
    helper_path=Path(config['legacy_postprocessing_source']);assert sha(helper_path)==config['legacy_postprocessing_source_sha256']
    functions=legacy_helpers(helper_path);torch.set_num_threads(4)
    network=YOLO(checkpoint);assert network.task=='segment'
    records=[]
    for i,case in enumerate(cases):
        key=case['case_id'];record_path=output/'predictions/exp02_os5'/(key+'.json')
        if record_path.exists():
            record=json.loads(record_path.read_text())
            assert record['checkpoint_sha256']==FIXED_SHA and record['crop_sha256']==case['original_sha256']
            assert sha(output/record['mask'])==record['mask_sha256']
            records.append(record);continue
        original=cv2.imread(case['original']);assert original.shape[:2]==(case['height'],case['width'])
        result=network.predict(original,imgsz=320,conf=.15,device='cpu',verbose=False)[0]
        mask,confidence=functions['result_mask'](result,original.shape[:2])
        contour=functions['contour_from_mask'](mask)
        proposal=[contour[:,0,:].tolist()] if contour is not None and len(contour)>=3 else []
        binary=np.zeros(original.shape[:2],dtype=np.uint8) if mask is None else mask
        components=cv2.connectedComponents((binary>0).astype(np.uint8))[0]-1
        contours,hierarchy=cv2.findContours((binary>0).astype(np.uint8),cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)
        holes=sum(int(row[3]>=0) for row in hierarchy[0]) if hierarchy is not None else 0
        name='predictions/exp02_os5/'+key+'_mask.png'
        with (output/name).open('xb') as handle:Image.fromarray(binary).save(handle,format='PNG')
        record={'case_id':key,'source_id':case['source_id'],'detection_id':case['detection_id'],'role':case['role'],
                'checkpoint_sha256':FIXED_SHA,'crop_sha256':case['original_sha256'],'mask':name,'mask_sha256':sha(output/name),
                'confidence':confidence,'has_mask':mask is not None,'proposal':proposal,'proposal_sha256':content_hash(proposal),
                'connected_components':components,'holes':holes,'touches_crop_edge':bool(binary[0].any() or binary[-1].any() or binary[:,0].any() or binary[:,-1].any()),
                'postprocessing':'Exact legacy result_mask + contour_from_mask; highest-confidence instance, unchanged clipping/rasterization; largest external editable outline. Full raw selected mask is retained.'}
        write_new(record_path,record);records.append(record)
        if (i+1)%50==0:print(f'Predicted {i+1}/779',flush=True)
    assert len(records)==779 and sha(checkpoint)==FIXED_SHA
    write_new(output/'prediction_manifest.json',records)
    write_new(output/'prediction_seal.json',{'files':{p.relative_to(output).as_posix():sha(p)
        for p in (output/'predictions/exp02_os5').rglob('*') if p.is_file()},
        'manifest_sha256':sha(output/'prediction_manifest.json'),'checkpoint_sha256':FIXED_SHA,
        'successful_predictions':len(records),'with_mask':sum(r['has_mask'] for r in records),
        'no_mask':sum(not r['has_mask'] for r in records),'runtime':{'torch':torch.__version__,'ultralytics':ultralytics.__version__}})
    print(json.dumps({'successful_predictions':len(records),'with_mask':sum(r['has_mask'] for r in records),
                      'roles':dict(Counter(r['role'] for r in records)),'detector_run':False,'dedup_run':False},indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/ruler_exp02_os5_manual_validation_v1')
    parser.add_argument('--infer',action='store_true')
    args=parser.parse_args();output=args.output.resolve()
    if args.infer:infer(output)
    else:prepare(output,Path('/tmp/opencode/ruler_exp02_review_inventory.csv'),Path('/tmp/opencode/ruler_exp02_review_inventory_summary.json'))


if __name__=='__main__':main()
