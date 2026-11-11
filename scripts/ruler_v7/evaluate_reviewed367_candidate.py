#!/usr/bin/env python3
"""Lock the base-validation-selected exp03 checkpoint, then compare it to fixed exp02_os5."""

import argparse
import csv
import json
import math
import os
import shutil
import sys
from datetime import datetime,timezone
from pathlib import Path

import numpy as np
import yaml
from PIL import Image,ImageDraw,ImageFont

from prepare_reviewed367_experiment import ROOT,DEFAULT_OUTPUT,FIXED_SHA,sealed_files,sha
from prepare_targeted_ruler_experiment import checkpoint_metadata
from evaluate_targeted_ruler_candidate import (
    METRIC_KEYS,plain,read_csv,csv_file,pixel_metrics,reference_masks,validator_class,contour_panel,
)


def write_once(path,value):
    if path.exists():
        if json.loads(path.read_text())!=value:raise ValueError('Immutable evaluation evidence differs: '+str(path))
    else:
        with path.open('x') as handle:json.dump(value,handle,indent=2,allow_nan=False)


def input_integrity(experiment):
    seal=sealed_files(experiment,'experiment_seal.json')
    for path,digest in json.loads((experiment/'evidence/external_input_hashes.json').read_text()).items():
        if sha(Path(path))!=digest:raise ValueError('Original input changed: '+path)
    return seal


def lock_candidate(experiment,output):
    seal=input_integrity(experiment)
    job=json.loads((experiment/'logs/training_job.json').read_text())
    if job['state']!='COMPLETED' or job['returncode']!=0 or job['training_launches']!=1:
        raise ValueError('The single authorized training run has not completed successfully')
    run=Path(seal['run_directory']);results=read_csv(run/'results.csv')
    if [int(r['epoch']) for r in results]!=list(range(1,11)):raise ValueError('Expected exactly ten epochs')
    approved=yaml.safe_load((experiment/'config/train.yaml').read_text());actual=yaml.safe_load((run/'args.yaml').read_text())
    if any(actual.get(k)!=v for k,v in approved.items()):raise ValueError('Training configuration drift')
    initial=experiment/'initial/exp02_os5_best.pt';trainer_best=run/'weights/best.pt'
    fixed=output/'fixed_candidate';fixed.mkdir(exist_ok=True);candidate=fixed/'best.pt'
    if candidate.exists():
        if sha(candidate)!=sha(trainer_best):raise ValueError('Previously fixed candidate differs from trainer best.pt')
    else:
        with trainer_best.open('rb') as source,candidate.open('xb') as destination:shutil.copyfileobj(source,destination)
    if sha(initial)!=FIXED_SHA:raise ValueError('Wrong initial exp02 checkpoint')
    meta=checkpoint_metadata(candidate);metrics=meta['train_metrics']
    matches=[int(row['epoch']) for row in results if all(math.isclose(float(row[k]),float(metrics[k]),abs_tol=2e-5) for k in METRIC_KEYS)]
    if not matches:raise ValueError('Cannot tie trainer-selected best.pt to the base-validation epoch metrics')
    fitnesses={int(row['epoch']):float(row['metrics/mAP50-95(B)'])+float(row['metrics/mAP50-95(M)']) for row in results}
    maximum=max(fitnesses.values())
    if not any(abs(fitnesses[epoch]-maximum)<=4e-5 for epoch in matches):
        raise ValueError('best.pt is not consistent with maximum Ultralytics base-validation fitness')
    selection={'checkpoint':str(candidate),'trainer_best_checkpoint':str(trainer_best),'sha256':sha(candidate),'initial_checkpoint':str(initial),
        'initial_sha256':sha(initial),'matching_training_epochs':matches,'base_validation_images':463,
        'criterion':'Unmodified Ultralytics best.pt selected solely on base validation fitness (box mAP50-95 + mask mAP50-95)',
        'base_epoch_fitness':fitnesses,'stored_train_metrics':plain(metrics),
        'training_config_sha256':sha(experiment/'config/train.yaml'),'results_csv_sha256':sha(run/'results.csv'),
        'all_epoch_checkpoint_sha256':{f'epoch{i}.pt':sha(run/'weights'/f'epoch{i}.pt') for i in range(10)},
        'test_used_for_selection':False,'ruler_val_used_for_selection':False}
    selection_path=output/'candidate_selection.json'
    selection['selected_at_utc']=json.loads(selection_path.read_text())['selected_at_utc'] if selection_path.exists() else datetime.now(timezone.utc).isoformat()
    # Convert integer JSON keys to their stable serialized representation for resume checks.
    selection=json.loads(json.dumps(selection))
    write_once(output/'candidate_selection.json',selection)
    return initial,candidate,selection


def evaluate(experiment):
    os.environ['YOLO_CONFIG_DIR']=str(experiment/'runtime_settings');os.environ['YOLO_AUTOINSTALL']='false'
    output=experiment/'evaluation';output.mkdir(exist_ok=True)
    if (output/'evaluation_seal.json').exists():
        sealed_files(output,'evaluation_seal.json');input_integrity(experiment)
        completed=json.loads((output/'summary.json').read_text())
        if sha(Path(completed['candidate_selection']['checkpoint']))!=completed['candidate_selection']['sha256']:
            raise ValueError('Previously fixed checkpoint changed')
        print(json.dumps(completed,indent=2),flush=True);return
    initial,candidate,selection=lock_candidate(experiment,output)
    parameters={'imgsz':320,'batch':16,'device':'cpu','workers':0,'rect':False,'augment':False,
        'conf':None,'iou':.7,'max_det':300,'overlap_mask':True,'mask_ratio':4,'save_json':False,'plots':True,'half':False}
    prediction_parameters={'imgsz':320,'conf':.15,'iou':.7,'max_det':300,'retina_masks':False,
                           'device':'cpu','verbose':False,'save':False,'half':False}
    inventory=json.loads((experiment/'manifests/combined_inventory.json').read_text())
    evaluation=json.loads((experiment/'manifests/ruler_evaluation.json').read_text())
    base=[r for r in inventory if r['split']=='base_val'];examples=[]
    for source in sorted({r['source_id'] for r in base}):
        rows=sorted([r for r in base if r['source_id']==source],key=lambda r:r['image_name'])
        examples.extend([dict(r,phase='base_validation') for r in ([rows[0]] if len(rows)==1 else [rows[0],rows[-1]])])
    examples.extend(dict(r,phase='ruler_validation' if r['split']=='ruler_val' else 'final_test_only') for r in evaluation)
    protocol={'validation_parameters':parameters,'prediction_parameters':prediction_parameters,'seed':0,'torch_threads':4,
        'candidate_selection_sha256':sha(output/'candidate_selection.json'),'initial_checkpoint_sha256':FIXED_SHA,
        'candidate_checkpoint_sha256':selection['sha256'],'examples':examples,
        'instance_protocol':'Separate instance arrays; best one-to-one target IoU and missing/extra instances at IoU0.5. Union metrics are an additional diagnostic penalizing extras, never a proposal or label.',
        'thresholds_or_crop_tuning_after_test':False,'same_bytes_and_parameters_for_both_models':True}
    write_once(output/'evaluation_protocol.json',protocol)
    from reviewed367_runtime_guard import install
    install(experiment)
    immutable={candidate.resolve(),(output/'candidate_selection.json').resolve(),(output/'evaluation_protocol.json').resolve()}
    def prevent_evidence_write(event,args):
        targets=[]
        if event=='open' and args[2]&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND):targets=[args[0]]
        elif event in {'os.remove','os.chmod','os.truncate'}:targets=[args[0]]
        elif event in {'os.rename','os.replace'}:targets=list(args[:2])
        for target in targets:
            if isinstance(target,(str,bytes,os.PathLike)) and Path(os.fsdecode(target)).resolve() in immutable:
                raise PermissionError('Fixed evaluation evidence is immutable')
    sys.addaudithook(prevent_evidence_write)
    import torch
    import ultralytics
    from ultralytics import YOLO
    from ultralytics.utils import ops
    from ultralytics.utils.torch_utils import init_seeds
    if ultralytics.__version__!='8.4.175' or torch.__version__!='2.15.0+cpu':raise ValueError('Unexpected evaluation runtime')
    torch.set_num_threads(4)
    write_once(output/'environment.json',{'torch':torch.__version__,'ultralytics':ultralytics.__version__,
        'python':os.sys.version,'device':'cpu','torch_threads':4})
    source_map={r['image_name']:r['source_id'] for r in inventory+evaluation}
    checkpoints={'initial_exp02_os5':initial,'candidate':candidate};table=[];gallery=[];pixels=[]
    def checkpoint_integrity():
        if sha(initial)!=selection['initial_sha256'] or sha(candidate)!=selection['sha256']:raise ValueError('Fixed checkpoint changed')
    def validate(phase,data,split,expected):
        for tag,checkpoint in checkpoints.items():
            checkpoint_integrity();folder=output/f'{phase}_{tag}';record_path=folder/'metrics_record.json'
            if record_path.exists():
                record=json.loads(record_path.read_text())
                if record['checkpoint_sha256']!=sha(checkpoint) or record['protocol_sha256']!=sha(output/'evaluation_protocol.json'):
                    raise ValueError('Cached evaluation identity mismatch')
            else:
                folder.mkdir(exist_ok=True);init_seeds(0,deterministic=True)
                print(f'EVALUATION {phase} {tag}',flush=True)
                metrics=YOLO(checkpoint).val(validator=validator_class(folder,source_map),data=str(data),split=split,
                    **parameters,project=str(output/'ultralytics'),name=f'{phase}_{tag}',exist_ok=False)
                captured=json.loads((folder/'per_image_validation_stats.json').read_text())
                if len(captured)!=expected:raise ValueError('Wrong evaluation image count')
                record={'phase':phase,'model':tag,'images':expected,'checkpoint_sha256':sha(checkpoint),
                    'protocol_sha256':sha(output/'evaluation_protocol.json'),**plain(metrics.results_dict)}
                write_once(record_path,record)
            table.append(record)
    validate('base_val',experiment/'combined_v7_reviewed367/dataset.yaml','val',463)
    validate('ruler_val',experiment/'config/ruler_evaluation.yaml','val',1)
    predictions={}
    for phase in ['base_validation','ruler_validation','final_test_only']:
        selected=[r for r in examples if r['phase']==phase]
        if phase=='final_test_only':
            checkpoint_integrity()
            gate={'candidate_fixed_sha256':selection['sha256'],'initial_fixed_sha256':FIXED_SHA,
                'candidate_selection_sha256':sha(output/'candidate_selection.json'),'protocol_sha256':sha(output/'evaluation_protocol.json'),
                'base_and_ruler_validation_completed':True,'test_used_for_selection':False}
            if not (output/'final_test_gate.json').exists():
                gate['opened_at_utc']=datetime.now(timezone.utc).isoformat();write_once(output/'final_test_gate.json',gate)
            else:
                previous=json.loads((output/'final_test_gate.json').read_text())
                if any(previous[k]!=v for k,v in gate.items()):raise ValueError('Test gate changed')
            validate('ruler_test',experiment/'config/ruler_evaluation.yaml','test',3)
        for tag,checkpoint in checkpoints.items():
            network=None;init_seeds(0,deterministic=True)
            for example in selected:
                image=experiment/example['image'];label=experiment/example['label'];key=image.stem
                if sha(image)!=example['image_sha256'] or sha(label)!=example['label_sha256']:raise ValueError('Evaluation pair changed')
                with Image.open(image) as im:shape=(im.height,im.width)
                gt=reference_masks(label,shape);directory=output/'predictions'/tag;directory.mkdir(parents=True,exist_ok=True)
                stored=directory/(key+'.npz')
                if stored.exists():
                    with np.load(stored,allow_pickle=False) as cache:pred=cache['masks'];scores=cache['confidence']
                else:
                    if network is None:network=YOLO(checkpoint)
                    result=network.predict(str(image),**prediction_parameters)[0]
                    if result.masks is None:pred=np.zeros((0,*shape),dtype=bool);scores=np.zeros(0)
                    else:
                        pred=(ops.scale_masks(result.masks.data[None].float(),shape,padding=True)[0]>.5).cpu().numpy()
                        scores=result.boxes.conf.cpu().numpy()
                    np.savez_compressed(stored,masks=pred,confidence=scores)
                if pred.shape[1:]!=shape:raise ValueError('Incorrect original-coordinate mask mapping')
                predictions[(tag,key)]=pred
                if phase!='base_validation':
                    if len(gt)!=1:raise ValueError('Ruler protocol expects one manual reference per image')
                    ious=[pixel_metrics(gt[0],mask)['iou'] for mask in pred];matched=int(any(v>=.5 for v in ious))
                    union=np.any(pred,axis=0) if len(pred) else np.zeros(shape,dtype=bool)
                    pixels.append({'phase':phase,'source_id':example['source_id'],'case_id':example['case_id'],
                        'model':tag,'predicted_instances':len(pred),'missing_gt_instances_iou50':1-matched,
                        'extra_predicted_instances_iou50':len(pred)-matched,'best_instance_iou':max(ious,default=0.),
                        **pixel_metrics(gt[0],union)})
            del network
        for example in selected:
            image=experiment/example['image'];label=experiment/example['label'];key=image.stem
            with Image.open(image) as im:original=im.convert('RGB')
            gt=reference_masks(label,(original.height,original.width));sheet=Image.new('RGB',(1520,470),'white')
            panels=[([], 'Original'),(gt,'GT'),(predictions[('initial_exp02_os5',key)],'Initial exp02_os5'),(predictions[('candidate',key)],'New candidate')]
            for i,(masks,title) in enumerate(panels):sheet.paste(contour_panel(original,masks,title),(i*380,40))
            ImageDraw.Draw(sheet).text((12,8),phase+' / '+key,fill='black',font=ImageFont.truetype('DejaVuSans.ttf',18))
            destination=output/'contours'/(key+'.png');destination.parent.mkdir(exist_ok=True);sheet.save(destination)
            gallery.append({'phase':phase,'case_id':key,'image':destination.relative_to(output).as_posix()})
    csv_file(output/'mask_box_metrics.csv',table);csv_file(output/'ruler_pixel_metrics.csv',pixels)
    deltas=[]
    for phase in ['base_val','ruler_val','ruler_test']:
        before=next(r for r in table if r['phase']==phase and r['model']=='initial_exp02_os5')
        after=next(r for r in table if r['phase']==phase and r['model']=='candidate')
        deltas.append({'phase':phase,**{k:after[k]-before[k] for k in METRIC_KEYS}})
    csv_file(output/'mask_box_deltas.csv',deltas)
    aggregates=[]
    for phase in ['ruler_validation','final_test_only']:
        for tag in checkpoints:
            rows=[r for r in pixels if r['phase']==phase and r['model']==tag]
            aggregates.append({'phase':phase,'model':tag,'images':len(rows),**{k:float(np.mean([r[k] for r in rows]))
                for k in ['iou','dice','pixel_recall','pixel_precision','best_instance_iou']}})
    csv_file(output/'ruler_pixel_summary.csv',aggregates)
    base_delta=next(r for r in deltas if r['phase']=='base_val')['metrics/mAP50-95(M)']
    initial_sources=json.loads((output/'base_val_initial_exp02_os5/per_source_metrics.json').read_text())
    candidate_sources={r['source_id']:r for r in json.loads((output/'base_val_candidate/per_source_metrics.json').read_text())}
    regional=[{'source_id':r['source_id'],'images':r['images'],**{k:candidate_sources[r['source_id']][k]-r[k] for k in METRIC_KEYS}} for r in initial_sources]
    csv_file(output/'base_per_source_changes.csv',regional)
    checkpoint_integrity();input_integrity(experiment)
    summary={'status':'COMPLETED','candidate_selection':selection,'mask_box_metrics':table,'deltas':deltas,
        'base_mask_map50_95_delta':base_delta,'base_retention_gate_pass':base_delta>=-.01,
        'ruler_pixel_summary':aggregates,'base_per_source_changes':regional,'fixed_examples':gallery,
        'protocol_sha256':sha(output/'evaluation_protocol.json'),'training_variants':1,'epochs':10,
        'production_changed':False,'original_and_prepared_inputs_unchanged':True,
        'test_limitations':'One fixed ruler validation image and three previously inspected source-separated test images; diagnostic, not a fresh blind test.'}
    write_once(output/'summary.json',summary)
    page='<!doctype html><meta charset="utf-8"><title>exp03 reviewed367 — results</title><style>body{background:#182332;color:white;font:16px system-ui;max-width:1600px;margin:auto;padding:20px}a{color:#8bd4ff}img{width:100%}pre{white-space:pre-wrap}</style><h1>Fixed exp02_os5 / reviewed367 candidate</h1>'
    page+='<p>Original / GT / initial exp02_os5 / fixed new candidate. Identical input bytes, runtime and parameters.</p><p><a href="summary.json">Summary</a> · <a href="mask_box_metrics.csv">Metrics</a> · <a href="ruler_pixel_metrics.csv">Ruler instance/pixel metrics</a></p>'
    page+=f'<p>Base mask mAP50-95 delta: {base_delta:+.6f}; retention gate: {"PASS" if base_delta>=-.01 else "FAIL"}</p>'
    for item in gallery:page+=f'<h2>{item["phase"]} / {item["case_id"]}</h2><a href="{item["image"]}"><img src="{item["image"]}"></a>'
    (output/'index.html').write_text(page)
    artifacts={p.relative_to(output).as_posix():sha(p) for p in output.rglob('*') if p.is_file()}
    write_once(output/'evaluation_seal.json',{'files':artifacts,'candidate_sha256':selection['sha256'],'initial_sha256':FIXED_SHA})
    print(json.dumps(summary,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--experiment',type=Path,default=DEFAULT_OUTPUT)
    args=parser.parse_args();evaluate(args.experiment.resolve())


if __name__=='__main__':main()
