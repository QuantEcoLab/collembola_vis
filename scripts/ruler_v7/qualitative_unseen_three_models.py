#!/usr/bin/env python3
"""One frozen photograph, one legacy detector pass, three unchanged segmentation/dedup passes."""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import shutil
import sys
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
INTAKE=ROOT/'outputs/qualitative_unseen_archive_selection_v1'
OUTPUT=ROOT/'outputs/qualitative_C_5_1_Collembola016_three_models_v1'
SNAPSHOT=Path('/home/jana/collembola-rescue/incoming/ruler-v7-20261027T162453Z/snapshot/repo')
IMAGE_SHA='a6e50925d27175825072462122a53a9a1191fed212c35dadcb3b39e8b79b1bc1'
V7_SHA='a46e37b85efe2794d1c5d72cabbbb6ef0aab5d5c72cd22d9264faeea46098883'
EXP02_SHA='9540b6478a8d77fcf6f2aa7db5c639c812e3c91953baa57c58670903221fc958'
EXP03_SHA='3fe97dd43f387815f8ee947eb5d02a6b0a7afadc1be59942a6b4a1f49f9aa0da'
DEDUP_SHA='b272fa46a1a2966a907ee85306310e40f3f54647d04999a9a6e5ba5e2d52830c'


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda:handle.read(8*1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def write_json(path,value,exclusive=True):
    with path.open('x' if exclusive else 'w',encoding='utf-8') as handle:
        json.dump(value,handle,indent=2,ensure_ascii=False,allow_nan=False)


def csv_file(path,rows,fields):
    with path.open('x',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)


def verify_seal(root,name):
    seal=json.loads((root/name).read_text())
    for relative,digest in seal['files'].items():
        if sha(root/relative)!=digest:raise ValueError('Sealed input changed: '+relative)
    return seal


def runtime(output):
    sys.path.insert(0,str(output/'runtime_dependencies'))
    os.environ['YOLO_CONFIG_DIR']=str(output/'runtime_settings');os.environ['YOLO_AUTOINSTALL']='false'
    import run_final_pipeline_comparison as pipeline
    versions={name:importlib.metadata.version(package) for name,package in [
        ('ultralytics','ultralytics'),('torch','torch'),('numpy','numpy'),('opencv','opencv-python'),
        ('pandas','pandas'),('joblib','joblib'),('scikit-learn','scikit-learn'),('scipy','scipy'),('threadpoolctl','threadpoolctl'),
        ('cloudpickle','cloudpickle'),('narwhals','narwhals'),('Pillow','Pillow')]}
    if versions['ultralytics']!='8.4.175' or versions['scikit-learn']!='1.9.0':raise ValueError('Runtime differs from legacy model metadata')
    return pipeline,versions


def tile_origins(width,height,params):
    rows=[];y=0;size=params['tile_size'];stride=params['stride']
    while y<height:
        x=0;y_end=0
        while x<width:
            right=min(x+size,width);bottom=min(y+size,height)
            left=max(0,right-size) if right==width else x
            top=max(0,bottom-size) if bottom==height else y
            rows.append([left,top,right,bottom]);y_end=bottom
            if right==width:break
            x+=stride
        if y_end==height:break
        y+=stride
    return rows


def crop_bounds(det,width,height):
    return [max(0,math.floor(det['x1'])),max(0,math.floor(det['y1'])),
            min(width,math.ceil(det['x2'])),min(height,math.ceil(det['y2']))]


def array_sha(array):
    return hashlib.sha256(f'{array.dtype}:{array.shape}:'.encode()+array.tobytes()).hexdigest()


def mask_iou(a,b):
    intersection=int((a&b).sum());union=int((a|b).sum())
    return intersection/union if union else 1.


def confidence_stats(values):
    import numpy as np
    values=[v for v in values if v is not None]
    return {'count':len(values),'minimum':min(values) if values else None,'median':float(np.median(values)) if values else None,
            'mean':float(np.mean(values)) if values else None,'maximum':max(values) if values else None}


def prepare(output):
    if (output/'config.json').exists():raise ValueError('Prepared run already exists')
    verify_seal(INTAKE,'intake_seal.json');verify_seal(INTAKE,'verification_seal.json')
    selection=json.loads((INTAKE/'selection_report.json').read_text());photo=selection['photo']
    if photo['source_id']!='C_5_1_Collembola016' or sha(photo['frozen_path'])!=IMAGE_SHA:raise ValueError('Wrong authorized photograph')
    pipeline,versions=runtime(output)
    helper=ROOT/'scripts/ruler_v7/run_final_pipeline_comparison.py'
    old_config=json.loads((ROOT/'outputs/ruler_exp02_os5_manual_validation_v1/config.json').read_text())
    if sha(helper)!=old_config['legacy_postprocessing_source_sha256']:raise ValueError('Legacy pipeline implementation changed')
    expected={'tile_size':1280,'overlap':256,'stride':1024,'detector_conf':.6,'bbox_nms_iou':.5,
        'seg_conf':.15,'seg_imgsz':320,'dedup_threshold':.5,'device':'cpu','envelope_mode':False,'mask_iou_dedup':False,'restricted_v5_cleanup':False}
    if pipeline.PARAMS!=expected:raise ValueError('Legacy parameters changed')
    production=SNAPSHOT/'runs/segment/runs/segment/trunk_seg_v7_ruler_finetune/weights/best.pt'
    models={
        'production_v7':{'path':str(production),'sha256':V7_SHA,'label':'Početni / production V7'},
        'exp02_os5':{'path':str(ROOT/'outputs/ruler_v7_targeted_ft_exp02_os5/training_runs/v7_ruler_ft_exp02_os5/weights/best.pt'),'sha256':EXP02_SHA,'label':'Fiksirani exp02_os5'},
        'exp03_no_os':{'path':str(ROOT/'outputs/ruler_v7_reviewed367_ft_exp03_no_os/evaluation/fixed_candidate/best.pt'),'sha256':EXP03_SHA,'label':'Fiksirani exp03_no_os'}}
    for model in models.values():
        if sha(model['path'])!=model['sha256']:raise ValueError('Segmentation checkpoint changed')
    if sha(ROOT/'outputs/ruler_v7_targeted_ft_exp02_os5/initial/v7_best.pt')!=V7_SHA:raise ValueError('Production V7 frozen reference mismatch')
    if sha(pipeline.DEDUP_MODEL)!=DEDUP_SHA:raise ValueError('Dedup classifier changed')
    payload=pipeline.joblib.load(pipeline.DEDUP_MODEL)
    if not all(feature in payload['features'] for feature in pipeline.FEATURES):raise ValueError('Dedup feature schema changed')
    for folder in ['input','shared/crops','context','previews','evidence','model_runs']:(output/folder).mkdir(parents=True,exist_ok=True)
    frozen=output/'input'/photo['filename']
    if frozen.exists():raise ValueError('Input copy already exists')
    shutil.copy2(photo['frozen_path'],frozen)
    config={'status':'PREPARED_NOT_INFERRED','source_id':photo['source_id'],'input':str(frozen),'image_sha256':IMAGE_SHA,
        'image_dimensions':photo['dimensions'],'models':models,
        'detector':{'path':str(pipeline.DETECTOR),'sha256':sha(pipeline.DETECTOR)},
        'dedup':{'path':str(pipeline.DEDUP_MODEL),'sha256':DEDUP_SHA,'features':payload['features'],
            'classifier_type':type(payload['model']).__name__,'sklearn_version':'1.9.0'},
        'parameters':pipeline.PARAMS,'bbox_padding_px':0,'runtime':versions,'torch_threads':4,
        'legacy_pipeline_source':str(helper),'legacy_pipeline_sha256':sha(helper),
        'script_sha256':sha(Path(__file__)),'input_intake_seal_sha256':sha(INTAKE/'intake_seal.json'),
        'inference_policy':'Exactly one legacy run_detector_once call; unchanged segment_one_model/result_mask/classifier dedup for each fixed model.',
        'difference_order':'descending (dedup-status disagreement, mask-presence disagreement, 1 - minimum pairwise mask IoU, mask area range), then ascending detection_id; diagnostic only, no GT winner',
        'legacy_render_note':'Final full-resolution overlays use the unchanged legacy renderer (largest external contour); crop mask viewer preserves every pixel/component of the stored selected mask.'}
    write_json(output/'config.json',config)
    wheels={p.name:sha(p) for p in (output/'runtime_wheels').glob('*.whl')}
    write_json(output/'evidence/runtime_wheels.json',wheels)
    shutil.copy2(helper,output/'evidence/legacy_pipeline.py')
    shutil.copy2(Path(__file__),output/'evidence/qualitative_inference_recipe.py')
    write_json(output/'input_seal.json',{'files':{p.relative_to(output).as_posix():sha(p) for p in [
        frozen,output/'config.json',output/'evidence/runtime_wheels.json',output/'evidence/legacy_pipeline.py',output/'evidence/qualitative_inference_recipe.py']},
        'external_files':{str(pipeline.DETECTOR):sha(pipeline.DETECTOR),str(pipeline.DEDUP_MODEL):DEDUP_SHA,str(helper):sha(helper),
            photo['frozen_path']:IMAGE_SHA,**{m['path']:m['sha256'] for m in models.values()}}})
    print(json.dumps(config,indent=2,ensure_ascii=False),flush=True)


def verify_inputs(output):
    seal=verify_seal(output,'input_seal.json')
    for path,digest in seal['external_files'].items():
        if sha(path)!=digest:raise ValueError('Original/model input changed: '+path)
    if sha(Path(__file__))!=json.loads((output/'config.json').read_text())['script_sha256']:raise ValueError('Inference recipe changed')
    return json.loads((output/'config.json').read_text())


def infer(output):
    config=verify_inputs(output)
    with (output/'single_inference_launch.lock').open('x') as handle:handle.write(datetime.now(timezone.utc).isoformat())
    pipeline,versions=runtime(output)
    if versions!=config['runtime'] or pipeline.PARAMS!=config['parameters']:raise ValueError('Runtime/parameters drifted')
    import cv2
    import numpy as np
    import torch
    from PIL import Image
    torch.set_num_threads(4)
    state={'state':'RUNNING','source_id':config['source_id'],'detector_passes':0,'started_at_utc':datetime.now(timezone.utc).isoformat()}
    write_json(output/'inference_job.json',state)
    class DetectorRecorder:
        def __init__(self,network):self.network=network;self.calls=[]
        def __call__(self,*args,**kwargs):
            result=self.network(*args,**kwargs)
            captured=[]
            for value in result:
                boxes=value.boxes;xy=boxes.xyxy.cpu().numpy()
                captured.append({'xyxy':xy.tolist(),'box_dtype':xy.dtype.str,'confidence':boxes.conf.cpu().numpy().tolist(),
                    'classes':boxes.cls.cpu().numpy().tolist()})
            self.calls.append({'call':len(self.calls),'tile_size':list(args[0].size),'kwargs':kwargs,'results':captured})
            return result
    detector=DetectorRecorder(pipeline.YOLO(config['detector']['path']))
    if detector.network.task!='detect':raise ValueError('Wrong detector task')
    detections,detector_metadata=pipeline.run_detector_once(Path(config['input']),detector,output)
    state['detector_passes']=1
    if not detections:raise ValueError('No detections; preserving run rather than changing thresholds')
    tile_boxes=tile_origins(*config['image_dimensions'],config['parameters'])
    if len(tile_boxes)!=len(detector.calls):raise ValueError('Detector call count/tile geometry mismatch')
    before=[]
    for tile_id,(box,call) in enumerate(zip(tile_boxes,detector.calls)):
        if call['tile_size']!=[box[2]-box[0],box[3]-box[1]]:raise ValueError('Tile dimensions mismatch')
        for value in call['results']:
            local=np.array(value['xyxy'],dtype=np.dtype(value['box_dtype']))
            for i,(x1,y1,x2,y2) in enumerate(local):
                before.append({'x1':float(x1+box[0]),'y1':float(y1+box[1]),'x2':float(x2+box[0]),'y2':float(y2+box[1]),
                    'conf':float(value['confidence'][i]),'class':int(value['classes'][i]),'tile_id':tile_id})
    replay=pipeline.nms_global(before)
    for i,row in enumerate(replay):row['detection_id']=i
    if replay!=detections:raise ValueError('Once-only detector/NMS record cannot be reproduced')
    effective_detector={k:str(v) if isinstance(v,Path) else v for k,v in vars(detector.network.predictor.args).items()}
    write_json(output/'shared/detections.json',detections)
    write_json(output/'shared/detector_trace.json',{'tile_boxes':tile_boxes,'calls':detector.calls,'before_nms':before,'effective_parameters':effective_detector})
    detections_digest=sha(output/'shared/detections.json');del detector
    Image.MAX_IMAGE_PIXELS=None
    with Image.open(config['input']) as image:full=pipeline.cv2.cvtColor(np.array(image.convert('RGB')),cv2.COLOR_RGB2BGR)
    height,width=full.shape[:2];shared=[]
    for det in detections:
        box=crop_bounds(det,width,height);x1,y1,x2,y2=box;crop=full[y1:y2,x1:x2]
        if not crop.size:raise ValueError('Invalid detector crop; no silent removal')
        path=output/'shared/crops'/f'{config["source_id"]}_det{det["detection_id"]:04d}.png'
        if not cv2.imwrite(str(path),crop):raise OSError('Crop write failed')
        shared.append({'detection_id':det['detection_id'],'bbox':[det[k] for k in ['x1','y1','x2','y2']],
            'crop_box':box,'width':x2-x1,'height':y2-y1,'image':path.relative_to(output).as_posix(),
            'image_sha256':sha(path),'model_input_array_sha256':array_sha(crop),'detector_confidence':det['conf']})
    write_json(output/'shared/crop_manifest.json',shared)
    all_records={};summaries={};effectives={}
    for tag,model in config['models'].items():
        print('SEGMENTATION_AND_DEDUP '+tag,flush=True)
        records=[]
        class SegmentRecorder:
            def __init__(self):
                self.network=pipeline.YOLO(model['path'])
                if self.network.task!='segment':raise ValueError('Wrong segmentation task')
            def predict(self,crop,**kwargs):
                index=len(records);expected=shared[index]
                if array_sha(crop)!=expected['model_input_array_sha256']:raise ValueError('Model received a different crop')
                result=self.network.predict(crop,**kwargs)
                mask,confidence=pipeline.result_mask(result[0],crop.shape[:2])
                binary=np.zeros(crop.shape[:2],dtype=np.uint8) if mask is None else mask
                record={'detection_id':expected['detection_id'],'has_mask':mask is not None,'confidence':confidence,
                    'mask_area_px':int(np.count_nonzero(binary)),'components':int(cv2.connectedComponents((binary>0).astype(np.uint8))[0]-1),
                    'touches_crop_edge':bool(binary[0].any() or binary[-1].any() or binary[:,0].any() or binary[:,-1].any()),
                    'input_array_sha256':array_sha(crop),'mask_array_sha256':array_sha(binary),'predict_kwargs':kwargs,
                    'returned_instances':len(result[0].boxes) if result[0].boxes is not None else 0}
                records.append(record);return result
        network=SegmentRecorder();payload=pipeline.joblib.load(config['dedup']['path'])
        kept,legacy_summary=pipeline.segment_one_model(config['source_id'],tag,network,full,detections,output/'model_runs',payload)
        if len(records)!=len(detections) or sha(output/'shared/detections.json')!=detections_digest:raise ValueError('Detection count/state drift')
        kept_ids=set(legacy_summary['kept_detection_ids'])
        effective={k:str(v) if isinstance(v,Path) else v for k,v in vars(network.network.predictor.args).items()}
        effectives[tag]=effective
        for row,crop in zip(records,shared):
            stem=f'{config["source_id"]}_det{row["detection_id"]:04d}';base=output/'model_runs'/tag
            actual_crop=base/'all_crops'/(stem+'.png')
            if sha(actual_crop)!=crop['image_sha256']:raise ValueError('Saved model crops are not byte-identical')
            mask_path=base/'all_masks'/(stem+'_mask.png')
            if row['has_mask']:
                binary=cv2.imread(str(mask_path),cv2.IMREAD_GRAYSCALE)
                if array_sha(binary)!=row['mask_array_sha256']:raise ValueError('Stored mask differs from legacy result')
            else:
                (base/'empty_display_masks').mkdir(exist_ok=True);mask_path=base/'empty_display_masks'/(stem+'_empty.png')
                binary=np.zeros((crop['height'],crop['width']),dtype=np.uint8);cv2.imwrite(str(mask_path),binary)
            row.update(mask=mask_path.relative_to(output).as_posix(),mask_sha256=sha(mask_path),
                kept_after_dedup=row['detection_id'] in kept_ids,removed_as_duplicate=row['detection_id'] not in kept_ids)
        model_summary={'label':model['label'],'all_detections':len(detections),'no_mask':sum(not r['has_mask'] for r in records),
            'duplicates_removed':len(detections)-len(kept_ids),'final_boxes':len(kept_ids),
            'final_boxes_with_mask':sum(r['has_mask'] and r['kept_after_dedup'] for r in records),
            'segmentation_confidence':confidence_stats([r['confidence'] for r in records]),
            'edge_touching_masks':sum(r['touches_crop_edge'] for r in records),'multi_component_masks':sum(r['components']>1 for r in records),
            'legacy_summary':legacy_summary,'full_overlay':str((ROOT/legacy_summary['full_overlay']).relative_to(output))}
        write_json(output/'model_runs'/tag/'raw_records.json',records)
        all_records[tag]=records;summaries[tag]=model_summary
        del kept,network,payload
    relevant=['imgsz','conf','iou','max_det','retina_masks','augment','agnostic_nms','classes','half','device','rect','batch']
    baseline={key:effectives['production_v7'][key] for key in relevant}
    if any({key:args[key] for key in relevant}!=baseline for args in effectives.values()):raise ValueError('Effective segmentation parameters differ')
    write_json(output/'evidence/effective_predictor_arguments.json',{'detector':effective_detector,'segmentation':effectives,'identical_segmentation_parameters':baseline})
    comparison=[]
    for index,crop in enumerate(shared):
        models={tag:records[index] for tag,records in all_records.items()};tags=list(models);ious={}
        masks={tag:cv2.imread(str(output/row['mask']),cv2.IMREAD_GRAYSCALE)>0 for tag,row in models.items()}
        for i,a in enumerate(tags):
            for b in tags[i+1:]:ious[a+'__'+b]=mask_iou(masks[a],masks[b])
        x1,y1,x2,y2=crop['crop_box'];cw=min(width,max(640,4*(x2-x1)));ch=min(height,max(640,4*(y2-y1)))
        left=max(0,min(int((x1+x2-cw)/2),width-cw));top=max(0,min(int((y1+y2-ch)/2),height-ch))
        context=full[top:top+ch,left:left+cw].copy()
        cv2.rectangle(context,(x1-left,y1-top),(x2-left,y2-top),(0,0,255),2)
        path=output/'context'/f'det{crop["detection_id"]:04d}.jpg';cv2.imwrite(str(path),context,[cv2.IMWRITE_JPEG_QUALITY,92])
        areas=[r['mask_area_px'] for r in models.values()]
        comparison.append({**crop,'models':models,'pairwise_mask_iou':ious,'minimum_mask_iou':min(ious.values()),
            'dedup_disagreement':len({r['kept_after_dedup'] for r in models.values()})>1,
            'mask_presence_disagreement':len({r['has_mask'] for r in models.values()})>1,'mask_area_range':max(areas)-min(areas),
            'context':path.relative_to(output).as_posix(),'context_box':[left,top,left+cw,top+ch]})
    ranked=sorted(comparison,key=lambda r:(-int(r['dedup_disagreement']),-int(r['mask_presence_disagreement']),r['minimum_mask_iou'],-r['mask_area_range'],r['detection_id']))
    write_json(output/'comparison_manifest.json',comparison)
    differences=[{k:r[k] for k in ['detection_id','dedup_disagreement','mask_presence_disagreement','minimum_mask_iou','mask_area_range']} for r in ranked]
    write_json(output/'largest_differences.json',differences)
    csv_file(output/'largest_differences.csv',differences,list(differences[0]))
    for tag,path in [('original',Path(config['input'])),*[(tag,output/s['full_overlay']) for tag,s in summaries.items()]]:
        with Image.open(path) as image:
            if list(image.size)!=config['image_dimensions']:raise ValueError('Full-resolution output dimensions changed')
            preview=image.copy();preview.thumbnail((1500,1500));preview.save(output/'previews'/(tag+'.jpg'),quality=94)
    summary={'status':'COMPLETED','qualitative_only':True,'ground_truth_available':False,'training_started':False,
        'source_id':config['source_id'],'image':Path(config['input']).relative_to(output).as_posix(),
        'image_sha256':IMAGE_SHA,'dimensions':config['image_dimensions'],'detector_passes':1,'detector_tile_predict_calls':len(tile_boxes),
        'detections_before_bbox_nms':detector_metadata['detections_before_bbox_nms'],'detections_after_bbox_nms':len(detections),
        'detector_confidence':confidence_stats([r['conf'] for r in detections]),'shared_detections_sha256':detections_digest,
        'shared_crop_manifest_sha256':sha(output/'shared/crop_manifest.json'),'models':summaries,
        'model_hashes':{tag:model['sha256'] for tag,model in config['models'].items()},'runtime':versions,
        'all_crops_and_detections_identical':True,'parameters_unchanged':True,'dedup_classifier_sha256':DEDUP_SHA,
        'mask_presence_disagreements':sum(r['mask_presence_disagreement'] for r in comparison),
        'dedup_disagreements':sum(r['dedup_disagreement'] for r in comparison),'largest_difference_ids':[r['detection_id'] for r in ranked[:50]],
        'blind_status':'Consumed by this qualitative model comparison/visual review; not a blind test for future claims',
        'legacy_render_note':config['legacy_render_note']}
    write_json(output/'summary.json',summary)
    write_json(output/'qualitative_exposure.json',{'source_id':config['source_id'],'image_sha256':IMAGE_SHA,
        'comparison_generated_at_utc':datetime.now(timezone.utc).isoformat(),'subsequent_blind_test_claim_allowed':False,
        'manual_gt_created':False,'model_selection_or_threshold_tuning_performed':False})
    verify_inputs(output)
    write_json(output/'result_seal.json',{'files':{p.relative_to(output).as_posix():sha(p) for folder in ['input','shared','model_runs','context','previews','evidence']
        for p in (output/folder).rglob('*') if p.is_file()},'records':{name:sha(output/name) for name in ['summary.json','comparison_manifest.json','largest_differences.json','largest_differences.csv','qualitative_exposure.json']}})
    state.update(state='COMPLETED',finished_at_utc=datetime.now(timezone.utc).isoformat(),detector_tile_predict_calls=len(tile_boxes),
        segmentation_predict_calls={tag:len(records) for tag,records in all_records.items()},inputs_unchanged=True)
    write_json(output/'inference_job.json',state,exclusive=False)
    print(json.dumps(summary,indent=2,ensure_ascii=False),flush=True)


def render(output):
    verify_inputs(output);seal=verify_seal(output,'result_seal.json')
    for name,digest in seal['records'].items():
        if sha(output/name)!=digest:raise ValueError('Result metadata changed')
    template=Path(__file__).with_name('web_unseen_three_models.html')
    shutil.copy2(template,output/'index.html')
    summary=json.loads((output/'summary.json').read_text());config=json.loads((output/'config.json').read_text())
    report=['# Kvalitativna usporedba tri fiksirana modela — bez GT-a','',
        'Fotografija: `'+config['source_id']+'`; SHA-256: `'+IMAGE_SHA+'`.',
        'Detector je izvršen jednom. Sve tri segmentacije koriste iste post-NMS detekcije i identične bbox cropove.',
        'Pragovi, legacy rasterizacija maske, odabir highest-confidence instance i classifier dedup nisu mijenjani.',
        '', '| Model | Detekcije | Bez maske | Odbačeni duplikati | Konačni bboxovi | Konačni s maskom |',
        '|---|---:|---:|---:|---:|---:|']
    for tag,model in summary['models'].items():report.append(f'| {tag} | {model["all_detections"]} | {model["no_mask"]} | {model["duplicates_removed"]} | {model["final_boxes"]} | {model["final_boxes_with_mask"]} |')
    report+=['','## Provenancija','```json',json.dumps(config,indent=2,ensure_ascii=False),'```','',
        'Ovo je isključivo kvalitativna vizualna provjera bez ručnih GT oznaka. Razlike maski nisu mjera točnosti.',
        'Fotografija nakon ove usporedbe/pregleda više nije slijepi test. Nema izbora pobjednika prema izgledu,',
        'promjene parametara, treninga, GT-a ni promjene production modela.',
        '', 'Full-resolution overlayi koriste izvorni legacy renderer najveće vanjske konture. Pojedinačne',
        'binarne maske i pregled cropova zadržavaju sve komponente odabrane legacy maske.',
        'Dedup odluke mogu se razlikovati jer isti classifier prima različite mask-feature vrijednosti; svi',
        'post-NMS detection ID-jevi ostaju vidljivi u usporedbi, uključujući odbačene duplikate.',
        '', 'Naredbe i potpuni stdout/stderr nalaze se u `logs/`; hashovi i efektivni argumenti u `config.json` i `evidence/`.']
    (output/'REPORT.md').write_text('\n'.join(report)+'\n')
    write_json(output/'presentation_seal.json',{'files':{'index.html':sha(output/'index.html'),'REPORT.md':sha(output/'REPORT.md')},'template_sha256':sha(template)},exclusive=not (output/'presentation_seal.json').exists())


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=OUTPUT)
    parser.add_argument('--stage',choices=['prepare','infer','render','check'],required=True);args=parser.parse_args();output=args.output.resolve()
    if args.stage=='prepare':prepare(output)
    elif args.stage=='check':print(json.dumps(verify_inputs(output),indent=2,ensure_ascii=False))
    elif args.stage=='infer':infer(output)
    else:render(output)


if __name__=='__main__':main()
