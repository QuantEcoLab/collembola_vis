#!/usr/bin/env python3
"""Evaluate the completed approved exp02_os5, never train or deploy a model."""

import argparse
import csv
import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parents[2]
METRIC_KEYS=['metrics/precision(B)','metrics/recall(B)','metrics/mAP50(B)','metrics/mAP50-95(B)',
             'metrics/precision(M)','metrics/recall(M)','metrics/mAP50(M)','metrics/mAP50-95(M)']


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def plain(value):
    if isinstance(value,dict):return {k:plain(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [plain(v) for v in value]
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    return value


def read_csv(path):
    with path.open(newline='',encoding='utf-8') as handle:return list(csv.DictReader(handle))


def csv_file(path,rows):
    with path.open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def pixel_metrics(gt,pred):
    intersection=int((gt&pred).sum());ga=int(gt.sum());pa=int(pred.sum());union=ga+pa-intersection
    return {'iou':intersection/union if union else 1.0,
            'dice':2*intersection/(ga+pa) if ga+pa else 1.0,
            'pixel_recall':intersection/ga if ga else 0.0,
            'pixel_precision':intersection/pa if pa else 0.0,
            'area_ratio':pa/ga if ga else 0.0,'reference_area_px':ga,'prediction_area_px':pa}


def reference_masks(label,shape):
    h,w=shape;result=[]
    for line in label.read_text().splitlines():
        parts=line.split();assert parts[0]=='0'
        coords=list(map(float,parts[1:]));image=Image.new('1',(w,h))
        ImageDraw.Draw(image).polygon([(coords[i]*w,coords[i+1]*h) for i in range(0,len(coords),2)],fill=1)
        result.append(np.asarray(image,dtype=bool))
    return np.stack(result)


def validator_class(directory,source_map):
    from ultralytics.models.yolo.segment import SegmentationValidator
    from ultralytics.utils.metrics import SegmentMetrics
    class CapturingValidator(SegmentationValidator):
        def init_metrics(self,model):
            super().init_metrics(model);self.captured=[]

        def update_metrics(self,preds,batch):
            update=self.metrics.update_stats
            def capture(stat):
                self.captured.append({k:v.copy() if isinstance(v,np.ndarray) else v for k,v in stat.items()})
                update(stat)
            self.metrics.update_stats=capture
            try:super().update_metrics(preds,batch)
            finally:self.metrics.update_stats=update

        def get_stats(self):
            result=super().get_stats()
            grouped=defaultdict(list)
            for row in self.captured:grouped[source_map[row['im_name']]].append(row)
            regional=[]
            for source,rows in sorted(grouped.items()):
                metrics=SegmentMetrics(names=self.names)
                for row in rows:metrics.update_stats(row)
                metrics.process(plot=False)
                regional.append({'source_id':source,'images':len(rows),**plain(metrics.results_dict)})
            (directory/'per_source_metrics.json').write_text(json.dumps(regional,indent=2))
            (directory/'per_image_validation_stats.json').write_text(json.dumps(plain(self.captured)))
            (directory/'validated_metrics.json').write_text(json.dumps(plain(result),indent=2))
            return result
    return CapturingValidator


def contour_panel(image,masks,title):
    import cv2
    panel=Image.new('RGB',(380,430),'#edf0f3');w,h=image.size
    scale=min(350/w,350/h);size=(max(1,round(w*scale)),max(1,round(h*scale)))
    enlarged=image.resize(size,Image.Resampling.LANCZOS)
    draw=ImageDraw.Draw(enlarged)
    for mask in masks:
        contours,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
        for contour in contours:
            points=[(float(p[0][0])*scale,float(p[0][1])*scale) for p in contour]
            if len(points)>1:draw.line(points+[points[0]],fill='#ffd700',width=2)
    panel.paste(enlarged,((380-size[0])//2,45+(370-size[1])//2))
    ImageDraw.Draw(panel).text((12,10),title,fill='black',font=ImageFont.truetype('DejaVuSans.ttf',18))
    return panel


def evaluate(experiment):
    os.environ['YOLO_CONFIG_DIR']=str(experiment/'runtime_settings')
    os.environ['YOLO_AUTOINSTALL']='false'
    from run_targeted_ruler_experiment import verify
    seal,train_cfg,runtime=verify(experiment)
    job=json.loads((experiment/'logs/training_job.json').read_text())
    if job['state']!='COMPLETED' or job['returncode']!=0:raise ValueError('Approved training must finish successfully first')
    if runtime['ultralytics']!='8.4.175':raise ValueError('Unexpected runtime version')
    run=Path(seal['run_directory']);initial=experiment/'initial/v7_best.pt';candidate=run/'weights/best.pt'
    if not candidate.exists():raise ValueError('No trainer-selected best checkpoint')
    results=read_csv(run/'results.csv')
    if [int(row['epoch']) for row in results]!=list(range(1,11)):raise ValueError('Expected ten completed epochs')
    actual=yaml.safe_load((run/'args.yaml').read_text())
    for key in ['epochs','imgsz','batch','optimizer','lr0','device','close_mosaic','resume']:
        if actual[key]!=train_cfg[key]:raise ValueError(f'Training changed approved parameter: {key}')
    import torch
    import ultralytics
    from ultralytics import YOLO
    from ultralytics.utils import ops
    from ultralytics.utils.torch_utils import init_seeds
    torch.set_num_threads(4);init_seeds(0,deterministic=True)
    output=experiment/'evaluation'
    output.mkdir(exist_ok=True)
    from prepare_targeted_ruler_experiment import checkpoint_metadata
    meta=checkpoint_metadata(candidate)
    stored_metrics=meta['train_metrics']
    matching=[int(r['epoch']) for r in results if all(
        math.isclose(float(r[k]),float(stored_metrics[k]),abs_tol=2e-5) for k in METRIC_KEYS)]
    if not matching:raise ValueError('Cannot match trainer-selected checkpoint metrics to the completed epoch history')
    selection={'checkpoint':str(candidate),'sha256':sha(candidate),'last_sha256':sha(run/'weights/last.pt'),
               'initial_checkpoint':str(initial),'initial_sha256':sha(initial),
               'matching_training_epochs':matching,'checkpoint_recorded_train_metrics':plain(stored_metrics),
               'criterion':'Unmodified Ultralytics best.pt selected on base validation; no ruler-test checkpoint selection',
               'torch':torch.__version__,'ultralytics':ultralytics.__version__}
    selection_path=output/'candidate_selection.json'
    if selection_path.exists() and json.loads(selection_path.read_text())!=selection:raise ValueError('Candidate selection changed between evaluation phases')
    selection_path.write_text(json.dumps(selection,indent=2))
    plan=json.loads((experiment/'config/evaluation_plan.json').read_text())
    inventory=json.loads((experiment/'manifests/combined_inventory.json').read_text())
    source_map={r['image_name']:r['source_id'] for r in inventory}
    frozen=read_csv(experiment/'manifests/frozen_addon.csv')
    source_map.update({Path(r['image']).name:r['source_id'] for r in frozen})
    source_by_stem={Path(name).stem:source for name,source in source_map.items()}
    checkpoints={'initial_v7':initial,'candidate':candidate}
    table=[]
    def validation_phase(phase,data,split,expected_images):
        for tag,checkpoint in checkpoints.items():
            folder=output/f'{phase}_{tag}'
            cached=folder/'metrics_record.json'
            if cached.exists():
                record=json.loads(cached.read_text());assert record['checkpoint_sha256']==sha(checkpoint)
            else:
                folder.mkdir(exist_ok=True)
                print(f'EVALUATION {phase} {tag}',flush=True)
                metrics=YOLO(checkpoint).val(validator=validator_class(folder,source_map),
                    data=str(experiment/data),split=split,**plan['validation_parameters'],
                    project=str(output/'ultralytics'),name=f'{phase}_{tag}',exist_ok=False)
                captured=json.loads((folder/'per_image_validation_stats.json').read_text())
                if len(captured)!=expected_images:raise ValueError('Evaluation image count differs from the frozen split')
                record={'phase':phase,'model':tag,'images':len(captured),'checkpoint_sha256':sha(checkpoint),**plain(metrics.results_dict)}
                cached.write_text(json.dumps(record,indent=2))
            table.append(record)
    validation_phase('base_val','combined_v7_ruler/dataset.yaml','val',463)
    validation_phase('ruler_val','config/ruler_evaluation.yaml','val',1)
    examples=read_csv(experiment/plan['contour_protocol']['example_manifest'])
    pixel_rows=[];gallery=[]
    for phase in ['base_validation','ruler_validation','final_test_only']:
        selected=[e for e in examples if e['phase']==phase]
        if phase=='final_test_only':
            assert sha(candidate)==selection['sha256']
            (output/'final_test_gate.json').write_text(json.dumps({'candidate_sha256_fixed_before_test':sha(candidate),
                'initial_sha256':sha(initial),'config_sha256':sha(experiment/'config/train.yaml'),
                'base_and_ruler_validation_completed_before_test':True,
                'test_used_for_training_or_checkpoint_selection':False},indent=2))
            validation_phase('ruler_test','config/ruler_evaluation.yaml','test',3)
        predictions={}
        for tag,checkpoint in checkpoints.items():
            network=None
            for example in selected:
                image=experiment/example['image'];assert sha(image)==example['image_sha256']
                key=image.stem
                label=Path(str(image).replace('/images/','/labels/')).with_suffix('.txt')
                with Image.open(image) as original:shape=(original.height,original.width)
                gt=reference_masks(label,shape);gt_union=np.any(gt,axis=0)
                prediction_dir=output/'predictions'/tag;prediction_dir.mkdir(parents=True,exist_ok=True)
                stored=prediction_dir/(key+'.npz')
                if stored.exists():
                    cache=np.load(stored,allow_pickle=False);pred=cache['masks'];scores=cache['conf']
                else:
                    if network is None:network=YOLO(checkpoint)
                    result=network.predict(str(image),imgsz=320,conf=0.15,iou=0.7,max_det=300,
                                           retina_masks=False,device='cpu',verbose=False,save=False)[0]
                    if result.masks is None:
                        pred=np.zeros((0,*shape),dtype=bool);scores=np.zeros(0)
                    else:
                        pred=(ops.scale_masks(result.masks.data[None].float(),shape,padding=True)[0]>.5).cpu().numpy()
                        scores=result.boxes.conf.cpu().numpy()
                    np.savez_compressed(stored,masks=pred,conf=scores)
                assert pred.shape[1:]==shape
                predictions[(tag,key)]=pred
                union=np.any(pred,axis=0) if len(pred) else np.zeros(shape,dtype=bool)
                if phase!='base_validation':
                    # All four frozen ruler evaluation images have exactly one
                    # manual body. Report union scores (penalize extras) AND best
                    # instance overlap, without hiding false positives.
                    assert len(gt)==1
                    instance_scores=[pixel_metrics(gt[0],p)['iou'] for p in pred]
                    matched=int(any(score>=.5 for score in instance_scores))
                    pixel_rows.append({'phase':phase,'source_id':source_by_stem[key],'case_id':key,'model':tag,
                                       'predicted_instances':len(pred),'missing_gt_instances_iou50':1-matched,
                                       'extra_predicted_instances_iou50':len(pred)-matched,
                                       'best_instance_iou':max(instance_scores,default=0.0),
                                       **pixel_metrics(gt_union,union)})
            del network
        for example in selected:
            image=experiment/example['image'];key=image.stem
            label=Path(str(image).replace('/images/','/labels/')).with_suffix('.txt')
            with Image.open(image) as opened:original=opened.convert('RGB')
            gt=reference_masks(label,(original.height,original.width))
            combined=Image.new('RGB',(1520,470),'white')
            for i,(masks,title) in enumerate([( [],'Original'),(gt,'Ručna / postojeća GT oznaka'),
                                            (predictions[('initial_v7',key)],'Početni V7'+(' — nema maske' if not len(predictions[('initial_v7',key)]) else '')),
                                            (predictions[('candidate',key)],'Novi kandidat'+(' — nema maske' if not len(predictions[('candidate',key)]) else ''))]):
                combined.paste(contour_panel(original,masks,title),(i*380,40))
            ImageDraw.Draw(combined).text((12,8),f'{phase} / {key}',fill='black',font=ImageFont.truetype('DejaVuSans.ttf',18))
            destination=output/'contours'/(key+'_side_by_side.png');destination.parent.mkdir(exist_ok=True)
            combined.save(destination);gallery.append({'phase':phase,'case_id':key,'image':destination.relative_to(output).as_posix()})
    csv_file(output/'mask_box_metrics.csv',table)
    deltas=[]
    for phase in ['base_val','ruler_val','ruler_test']:
        before=next(r for r in table if r['phase']==phase and r['model']=='initial_v7')
        after=next(r for r in table if r['phase']==phase and r['model']=='candidate')
        deltas.append({'phase':phase,**{key:after[key]-before[key] for key in METRIC_KEYS}})
    csv_file(output/'mask_box_deltas.csv',deltas)
    csv_file(output/'ruler_pixel_metrics.csv',pixel_rows)
    pixel_aggregate=[]
    for phase in ['ruler_validation','final_test_only']:
        for tag in checkpoints:
            rows=[r for r in pixel_rows if r['phase']==phase and r['model']==tag]
            pixel_aggregate.append({'phase':phase,'model':tag,'images':len(rows),**{
                key:float(np.mean([r[key] for r in rows])) for key in ['iou','dice','pixel_recall','pixel_precision','area_ratio']}})
    csv_file(output/'ruler_pixel_summary.csv',pixel_aggregate)
    base_delta=next(r for r in deltas if r['phase']=='base_val')['metrics/mAP50-95(M)']
    summary={'status':'COMPLETED','candidate_selection':selection,'mask_box_metrics':table,'deltas':deltas,
             'base_mask_map50_95_drop_exceeds_0_01':base_delta < -.01,
             'base_mask_map50_95_delta':base_delta,'ruler_pixel_summary':pixel_aggregate,
             'fixed_contour_examples':gallery,'pixel_mask_mapping':'Same retina_masks=False predictions; ops.scale_masks to original input dimensions, threshold0.5; union penalizes extra masks; also best-instance IoU',
             'contour_predict_parameters':{'imgsz':320,'conf':0.15,'iou':0.7,'max_det':300,'retina_masks':False,'device':'cpu'},
             'test_limitations':'Three fixed source-separated previously visually reviewed cases, not a blind untouched test; ruler val has one image',
             'production_changed':False,'control_1x_executed':False}
    (output/'summary.json').write_text(json.dumps(summary,indent=2))
    page='<!doctype html><meta charset="utf-8"><title>V7 ruler fine-tuning — rezultati</title><style>body{font:16px system-ui;background:#182332;color:white;padding:20px}img{width:100%;max-width:1520px}a{color:#9ce1ff}</style><h1>Početni V7 / kandidat — fiksirani primjeri</h1>'
    page+='<p>Original, GT, početni V7 i kandidat. Jednaki ulazi, konf.0.15, imgsz320 i prikaz; nema detector/dedup promjena.</p>'
    page+='<p><a href="summary.json">Sažetak</a> · <a href="mask_box_metrics.csv">Mask/box metrike</a> · <a href="ruler_pixel_metrics.csv">Ruler pixel metrike</a></p>'
    if base_delta < -.01:page+='<h2>UPOZORENJE: osnovni mask mAP50–95 pao je više od 0.01!</h2>'
    for item in gallery:page+=f'<h2>{item["phase"]} / {item["case_id"]}</h2><a href="{item["image"]}"><img src="{item["image"]}"></a>'
    (output/'index.html').write_text(page,encoding='utf-8')
    verify(experiment)
    assert sha(candidate)==selection['sha256'] and sha(initial)==selection['initial_sha256']
    print(json.dumps(summary,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment',type=Path,default=ROOT/'outputs/ruler_v7_targeted_ft_exp02_os5')
    args=parser.parse_args();evaluate(args.experiment.resolve())


if __name__=='__main__':
    main()
