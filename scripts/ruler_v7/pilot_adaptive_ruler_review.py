#!/usr/bin/env python3
"""Seven-case v3 padding pilot. No review database, label migration, or full-catalogue mode."""

import argparse
import html
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from prepare_controlled_review import ROOT, INITIAL_SHA, FIXED_SHA, sha, write_new
from web_controlled_ruler_review import Catalogue

SOURCE_A='C_1_2_Collembola005'
SOURCE_B='C_1_3_Collembola006'
SELECTION=[(SOURCE_A,'43','User-requested / saved note: premali crop'),
           (SOURCE_A,'185','User-requested: head cut off; neighbour merging in v2'),
           (SOURCE_A,'219','User-requested: neighbour merging in v2'),
           (SOURCE_A,'65','Saved note: premali crop'),
           (SOURCE_A,'1259','Earlier human diagnosis: TRUNCATED'),
           (SOURCE_B,'3','Earlier human diagnosis: TRUNCATED'),
           (SOURCE_B,'72','Earlier human diagnosis: TRUNCATED')]
VARIANTS={'old':0,'p10':.10,'p20':.20,'p30':.30,'v2_220':None}
RULES={'minimum_confidence':.25,'minimum_anchor_iou':.20,'minimum_mask_inside_anchor':.60,
       'minimum_anchor_coverage':.15,'maximum_bbox_area_ratio':2.,'runner_up_iou_ratio':.85,
       'require_centroid_inside':True,'require_bbox_centre_inside':True,
       'reject_multiple_components':True,'reject_crop_edge_contact':True}


def crop_geometry(case,variant):
    width,height=case['width'],case['height'];x,y=case['source_origin'];iw,ih=case['source_dimensions']
    fraction=VARIANTS[variant]
    pad=220 if fraction is None else (0 if fraction==0 else min(64,max(8,math.ceil(fraction*max(width,height)))))
    box=[max(0,x-pad),max(0,y-pad),min(iw,x+width+pad),min(ih,y+height+pad)]
    anchor=[x-box[0],y-box[1],x-box[0]+width,y-box[1]+height]
    return box,anchor,pad


def select_instance(instances,anchor,rules=RULES):
    """Keep masks separate; require meaningful target overlap and central association."""
    x1,y1,x2,y2=anchor;anchor_area=(x2-x1)*(y2-y1);ranked=[];diagnostics=[]
    for index,instance in enumerate(instances):
        mask=np.asarray(instance['mask'],dtype=bool);ys,xs=np.nonzero(mask);fail=[]
        if not len(xs):
            diagnostics.append({'instance':index,'eligible':False,'reasons':['EMPTY_MASK']});continue
        area=len(xs);intersection=int(mask[y1:y2,x1:x2].sum());union=anchor_area+area-intersection
        iou=intersection/union;inside=intersection/area;coverage=intersection/anchor_area
        centre=[float(xs.mean()),float(ys.mean())]
        bbox=[int(xs.min()),int(ys.min()),int(xs.max())+1,int(ys.max())+1]
        bbox_centre=[(bbox[0]+bbox[2])/2,(bbox[1]+bbox[3])/2]
        in_anchor=lambda p:x1<=p[0]<x2 and y1<=p[1]<y2
        ratio=(bbox[2]-bbox[0])*(bbox[3]-bbox[1])/anchor_area
        if instance['confidence']<rules['minimum_confidence']:fail.append('LOW_CONFIDENCE')
        if not in_anchor(centre):fail.append('CENTROID_OUTSIDE_DETECTION')
        if not in_anchor(bbox_centre):fail.append('BBOX_CENTRE_OUTSIDE_DETECTION')
        if iou<rules['minimum_anchor_iou']:fail.append('WEAK_ANCHOR_IOU')
        if inside<rules['minimum_mask_inside_anchor']:fail.append('EXCESS_MASK_OUTSIDE_DETECTION')
        if coverage<rules['minimum_anchor_coverage']:fail.append('INSUFFICIENT_DETECTION_COVERAGE')
        if ratio>rules['maximum_bbox_area_ratio']:fail.append('EXCESS_MASK_EXTENT')
        if instance['components']!=1:fail.append('MULTIPLE_COMPONENTS_IN_SINGLE_INSTANCE')
        if instance['touches_crop_edge']:fail.append('MASK_TOUCHES_CROP_EDGE')
        record={'instance':index,'confidence':instance['confidence'],'area_px':area,'intersection_px':intersection,
                'anchor_iou':iou,'mask_inside_anchor':inside,'anchor_coverage':coverage,
                'centroid':centre,'bbox':bbox,'bbox_centre':bbox_centre,'bbox_area_ratio':ratio,
                'components':instance['components'],'touches_crop_edge':instance['touches_crop_edge'],
                'eligible':not fail,'reasons':fail}
        diagnostics.append(record)
        if not fail:ranked.append((iou,intersection,instance['confidence'],index))
    ranked.sort(reverse=True)
    if not ranked:return None,diagnostics,'NO_RELIABLY_ASSOCIATED_INSTANCE'
    if len(ranked)>1 and ranked[1][0]>=rules['runner_up_iou_ratio']*ranked[0][0]:
        return None,diagnostics,'AMBIGUOUS_COMPETING_TARGET_INSTANCES'
    return ranked[0][-1],diagnostics,'SINGLE_INSTANCE_PROPOSAL_REQUIRES_VISUAL_REVIEW'


def verify_seal(output,name):
    seal=json.loads((output/name).read_text())
    if seal.get('manifest_sha256') and sha(output/'manifest.json')!=seal['manifest_sha256']:
        raise ValueError('Prediction manifest changed')
    for path,digest in seal['files'].items():
        if sha(Path(path) if Path(path).is_absolute() else output/path)!=digest:
            raise ValueError(f'Frozen evidence changed: {path}')


def prepare(output,backup):
    if output.exists():raise ValueError('Refusing to overwrite pilot')
    v1=Catalogue(ROOT/'outputs/ruler_exp02_os5_manual_validation_v1')
    by_id={(c['source_id'],c['detection_id']):c for c in v1.cases}
    cases=[dict(by_id[(s,d)],pilot_reason=reason) for s,d,reason in SELECTION]
    if len(cases)!=7 or any(c['role']!='train' for c in cases):raise ValueError('Pilot must contain exactly seven specified TRAIN cases')
    for folder in ['crops','context','predictions','comparisons','logs','frozen_inputs']:(output/folder).mkdir(parents=True,exist_ok=True)
    selection=json.loads((ROOT/'outputs/ruler_v7_targeted_ft_exp02_os5/evaluation/candidate_selection.json').read_text())
    models={'initial_v7':{'checkpoint':selection['initial_checkpoint'],'sha256':INITIAL_SHA},
            'exp02_os5':{'checkpoint':v1.config['candidate_checkpoint'],'sha256':FIXED_SHA}}
    for model in models.values():
        if sha(Path(model['checkpoint']))!=model['sha256']:raise ValueError('Checkpoint identity changed')
    v2=ROOT/'outputs/ruler_exp02_os5_manual_validation_v2'
    config={'scope':'SEVEN_CASE_PILOT_ONLY','cases':len(cases),'models':models,'imgsz':320,'conf':.15,
        'retina_masks':True,'device':'cpu','variants':VARIANTS,'padding_rule':'ceil(fraction * max(old_width,old_height)), clamped to [8,64] px per side',
        'association_rules':RULES,'backup_v2':str(backup),'v2_database_sha256':sha(v2/'review.sqlite3'),
        'full_catalogue_allowed':False,'training':False,'dataset_export':False,
        'mask_extraction':'Raw per-instance retina mask in native crop coordinates; never masks.xy rasterization, union, or largest-instance selection',
        'v2_note':'Archived v2 inference used >=640 despite inherited config imgsz=320. This pilot reruns every padding at actual imgsz=320 for a controlled comparison.'}
    external={str(v2/'review.sqlite3'):config['v2_database_sha256'],str(backup/'review.sqlite3'):sha(backup/'review.sqlite3')}
    for model in models.values():external[model['checkpoint']]=model['sha256']
    Image.MAX_IMAGE_PIXELS=None;variants=[]
    for source in [SOURCE_A,SOURCE_B]:
        selected=[c for c in cases if c['source_id']==source];source_path=Path(selected[0]['source_image'])
        external[str(source_path)]=sha(source_path)
        with Image.open(source_path) as image:
            full=image.convert('RGB')
            for case in selected:
                key=case['case_id'];x,y=case['source_origin'];w,h=case['width'],case['height']
                with Image.open(case['original']) as raw:old=np.asarray(raw.convert('RGB'))
                for variant in VARIANTS:
                    box,anchor,pad=crop_geometry(case,variant);crop=full.crop(box)
                    if not np.array_equal(np.asarray(crop.crop(anchor)),old):raise ValueError('Original pixels changed')
                    name=f'crops/{key}__{variant}.png';crop.save(output/name)
                    variants.append({'case_id':key,'source_id':source,'detection_id':case['detection_id'],'role':case['role'],
                        'variant':variant,'padding_px':pad,'box_source':box,'anchor':anchor,'image':name,
                        'image_sha256':sha(output/name),'width':crop.width,'height':crop.height,
                        'old_bbox_fraction':w*h/(crop.width*crop.height),'effective_old_bbox_at_320':[w*320/max(crop.size),h*320/max(crop.size)]})
                context_box=[max(0,x-400),max(0,y-350),min(full.width,x+w+400),min(full.height,y+h+350)]
                context=full.crop(context_box);draw=ImageDraw.Draw(context)
                draw.rectangle((x-context_box[0],y-context_box[1],x+w-context_box[0],y+h-context_box[1]),outline='#ffcc44',width=2)
                context.thumbnail((1000,850));context.save(output/f'context/{key}.jpg',quality=95)
    write_new(output/'pilot_cases.json',cases);write_new(output/'config.json',config);write_new(output/'variants.json',variants)
    sealed={p.relative_to(output).as_posix():sha(p) for p in output.rglob('*') if p.is_file()}
    sealed.update(external);write_new(output/'frozen_inputs/preparation_seal.json',{'files':sealed})
    print(json.dumps({'pilot_cases':7,'variants':len(variants),'model_runs_planned':len(variants)*2,'output':str(output)},indent=2))


def infer(output):
    import cv2
    import torch
    import ultralytics
    from ultralytics import YOLO
    if (output/'manifest.json').exists():raise ValueError('Pilot inference already complete')
    verify_seal(output,'frozen_inputs/preparation_seal.json')
    config=json.loads((output/'config.json').read_text());variants=json.loads((output/'variants.json').read_text())
    if len(variants)!=35 or config['scope']!='SEVEN_CASE_PILOT_ONLY':raise ValueError('Pilot scope changed')
    if ultralytics.__version__!='8.4.175':raise ValueError('Unexpected runtime')
    torch.set_num_threads(4);records=[]
    for model,info in config['models'].items():
        network=YOLO(info['checkpoint'])
        if network.task!='segment':raise ValueError('Expected segment model')
        for variant in variants:
            key=variant['case_id'];name=variant['variant'];folder=output/'predictions'/model/name;folder.mkdir(parents=True,exist_ok=True)
            record_path=folder/(key+'.json')
            if record_path.exists():raise ValueError('Partial inference exists; use a new pilot directory')
            original=cv2.imread(str(output/variant['image']))
            result=network.predict(original,imgsz=320,conf=.15,retina_masks=True,device='cpu',verbose=False)[0]
            instances=[];saved=[]
            if result.masks is not None:
                for i,raw in enumerate(result.masks.data.cpu().numpy()):
                    mask=raw>.5
                    if mask.shape!=original.shape[:2]:raise ValueError('Retina mask not in native coordinates')
                    components=cv2.connectedComponents(mask.astype(np.uint8))[0]-1
                    edge=bool(mask[0].any() or mask[-1].any() or mask[:,0].any() or mask[:,-1].any())
                    confidence=float(result.boxes.conf[i].item())
                    path=folder/f'{key}__instance{i:02d}.png';Image.fromarray(mask.astype(np.uint8)*255).save(path)
                    instance={'mask':mask,'confidence':confidence,'components':components,'touches_crop_edge':edge}
                    instances.append(instance)
                    contours,hierarchy=cv2.findContours(mask.astype(np.uint8),cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)
                    saved.append({'instance':i,'mask':path.relative_to(output).as_posix(),'mask_sha256':sha(path),
                        'confidence':confidence,'components':components,'touches_crop_edge':edge,
                        'contours':[c[:,0,:].tolist() for c in contours],
                        'hierarchy':hierarchy[0].tolist() if hierarchy is not None else []})
            selected,diagnostics,reason=select_instance(instances,variant['anchor'],config['association_rules'])
            proposal=[]
            if selected is not None:
                info_selected=saved[selected]
                external=[p for p,h in zip(info_selected['contours'],info_selected['hierarchy']) if h[3]<0]
                if len(external)!=1:raise ValueError('Selected mask is not a single connected instance')
                proposal=external
            record={**variant,'model':model,'checkpoint_sha256':info['sha256'],'imgsz':320,'retina_masks':True,
                'instances':saved,'target_instance':selected,'association':diagnostics,'status':'PENDING',
                'proposal':proposal,'reason':reason,'no_union':True,'mask_source':'raw native retina mask per instance'}
            write_new(record_path,record);records.append(record)
            print(f'{model} {key} {name}: target={selected}, instances={len(saved)}, {reason}',flush=True)
        if sha(Path(info['checkpoint']))!=info['sha256']:raise ValueError('Checkpoint changed')
    write_new(output/'manifest.json',records)
    write_new(output/'frozen_inputs/prediction_seal.json',{'files':{p.relative_to(output).as_posix():sha(p)
        for p in (output/'predictions').rglob('*') if p.is_file()},'manifest_sha256':sha(output/'manifest.json'),
        'runtime':{'torch':torch.__version__,'ultralytics':ultralytics.__version__}})
    print(json.dumps({'runs':len(records),'association_proposals':sum(r['target_instance'] is not None for r in records),'all_statuses':'PENDING'},indent=2))


def font(size):
    return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',size)


def render(output):
    import cv2
    verify_seal(output,'frozen_inputs/preparation_seal.json');verify_seal(output,'frozen_inputs/prediction_seal.json')
    rows=json.loads((output/'manifest.json').read_text());cases=json.loads((output/'pilot_cases.json').read_text())
    assessment_path=output/'visual_assessment.json'
    assessment=json.loads(assessment_path.read_text()) if assessment_path.exists() else {'all_779_processing_allowed':False,'cases':{}}
    by_key={(r['case_id'],r['variant'],r['model']):r for r in rows}
    shades=['#e69bff','#ffaa77','#78bfff','#ff7198','#e7d36a','#afc2d3']
    def panel(record,model):
        original=Image.open(output/record['image']).convert('RGB');draw=ImageDraw.Draw(original)
        if model:
            for instance in record['instances']:
                selected=instance['instance']==record['target_instance'];color='#46ff95' if selected else shades[instance['instance']%len(shades)]
                for contour,h in zip(instance['contours'],instance['hierarchy']):
                    if len(contour)>2:draw.line([tuple(p) for p in contour]+[tuple(contour[0])],fill=color,width=2 if selected else 1)
                ys,xs=np.nonzero(np.asarray(Image.open(output/instance['mask'])))
                if len(xs):draw.text((float(xs.mean()),float(ys.mean())),str(instance['instance']),fill=color,font=font(11),stroke_width=1,stroke_fill='black')
        draw.rectangle(tuple(record['anchor']),outline='#ffd94c',width=1)
        scale=min(290/original.width,290/original.height)
        original=original.resize((round(original.width*scale),round(original.height*scale)),Image.Resampling.LANCZOS)
        canvas=Image.new('RGB',(310,360),'#142131');canvas.paste(original,((310-original.width)//2,48+(290-original.height)//2))
        draw=ImageDraw.Draw(canvas);title=f'{record["variant"]} | pad {record["padding_px"]}px | {record["width"]}x{record["height"]}'
        draw.text((8,4),title,font=font(12),fill='white')
        if model:
            selected=record['target_instance'];label=f'{model}: '+(f'candidate #{selected} / PENDING' if selected is not None else 'NO TARGET / PENDING')
            draw.text((8,22),label,font=font(12),fill='#7fffb4' if selected is not None else '#ffbc98')
        else:
            w,h=record['effective_old_bbox_at_320'];draw.text((8,22),f'old bbox @320: {w:.0f} x {h:.0f}px',font=font(12),fill='#e0d67b')
        return canvas
    Image.MAX_IMAGE_PIXELS=None
    for source in {c['source_id'] for c in cases}:
        subset=[c for c in cases if c['source_id']==source]
        with Image.open(subset[0]['source_image']) as full:
            overview=full.convert('RGB');width,height=overview.size;overview.thumbnail((1800,1800))
        draw=ImageDraw.Draw(overview)
        for case in subset:
            x,y=case['source_origin'];w,h=case['width'],case['height']
            box=[x*overview.width/width,y*overview.height/height,(x+w)*overview.width/width,(y+h)*overview.height/height]
            draw.rectangle(box,outline='#ffdc4c',width=3)
            draw.text((box[0],box[1]-20),case['detection_id'],font=font(17),fill='#ffdc4c',stroke_width=2,stroke_fill='black')
        overview.save(output/f'context/{source}_full.jpg',quality=95)
    sections=[]
    for case in cases:
        key=case['case_id'];sheet=Image.new('RGB',(1550,1140),'#0e1825');draw=ImageDraw.Draw(sheet)
        draw.text((12,8),f'{key} | imgsz=320 | separate instances; yellow=old bbox; green=target proposal',font=font(17),fill='white')
        draw.text((12,34),'Rows: clean crop / initial V7 / exp02_os5. Every result remains PENDING; geometry association is not GT.',font=font(14),fill='#bed0df')
        for col,variant in enumerate(VARIANTS):
            base=by_key[(key,variant,'initial_v7')]
            sheet.paste(panel(base,None),(col*310,60))
            for row,model in enumerate(['initial_v7','exp02_os5'],1):sheet.paste(panel(by_key[(key,variant,model)],model),(col*310,60+row*360))
        name=f'comparisons/{key}.jpg';sheet.save(output/name,quality=96)
        details=[]
        for variant in VARIANTS:
            for model in ['initial_v7','exp02_os5']:
                record=by_key[(key,variant,model)]
                details.append({'variant':variant,'model':model,'target':record['target_instance'],'reason':record['reason'],
                    'instance_count':len(record['instances']),'association':record['association']})
        note=assessment.get('cases',{}).get(key,{}).get('finding','Vizualna provjera još nije potvrđena.')
        sections.append(f'<section id="{key}"><h2>{html.escape(key)}</h2><p>{html.escape(case["pilot_reason"])}</p><p class="warning">{html.escape(note)}</p>'
            f'<a href="{name}"><img src="{name}" alt="padding comparison"></a><details><summary>Širi kontekst — nije ulaz modela</summary>'
            f'<img class="context" src="context/{key}.jpg"><p><a href="context/{case["source_id"]}_full.jpg">Cijela fotografija — označeni pilot-cropovi, samo kontekst</a></p></details><details><summary>Podaci svake instance i razlozi odbijanja</summary>'
            f'<pre>{html.escape(json.dumps(details,indent=2))}</pre></details></section>')
    page='''<!doctype html><html lang="hr"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>V3 pilot — adaptivni padding</title>
<style>body{margin:0;background:#101b28;color:#edf4fc;font:16px system-ui}header{padding:18px;background:#22364c;position:sticky;top:0}main{padding:16px;max-width:1800px;margin:auto}section{padding:14px;margin:20px 0;background:#1b2c3e;border-radius:8px}img{width:100%;height:auto}.context{max-width:1000px}a{color:#8fceff}pre{white-space:pre-wrap;overflow-wrap:anywhere}.warning{color:#ffd49a}details{margin:12px 0}</style>
<header><strong>V3 PILOT — 7 slučajeva, 5 paddinga, oba fiksirana modela, stvarni imgsz=320</strong><br>Read-only usporedba. Nema baze odluka niti pregleda svih 779.</header><main>
<p>Stupci: stari crop · 10% · 20% · 30% dulje stranice (8–64 px) · v2 referenca +220 px. Svi ponovno obrađeni pri istom imgsz=320.</p>
<p>Svaka maska je zasebna instanca. Zelena = strogo povezan ciljni prijedlog; ostale boje = odvojene neprihvaćene instance; žuto = izvorni detector crop. Klikni usporedbu za punu veličinu.</p>
<p class="warning">Svi slučajevi ostaju PENDING. Zelena je samo geometrijski kandidat, ne potvrđena oznaka. Povezanost s bboxom ne dokazuje cijelo tijelo ni odsutnost susjeda. Bez pouzdane povezanosti nema ciljne konture. Pilot još nije prošao provjeru svih problematičnih primjera: obrada svih 779 je zaustavljena.</p>'''+ '<nav>'+' · '.join(f'<a href="#{c["case_id"]}">{html.escape(c["source_id"])} / {c["detection_id"]}</a>' for c in cases)+'</nav>'+''.join(sections)+'</main></html>'
    (output/'index.html').write_text(page)
    summary={'scope':'PILOT_ONLY','cases':len(cases),'runs':len(rows),'association_rules':RULES,
        'proposals_by_variant_model':{f'{v}/{m}':sum(r['target_instance'] is not None for r in rows if r['variant']==v and r['model']==m)
            for v in VARIANTS for m in ['initial_v7','exp02_os5']},
        'visual_body_complete_confirmed':False,'visual_neighbour_exclusion_confirmed':False,
        'all_779_processing_allowed':False,'all_review_statuses':'PENDING'}
    (output/'pilot_summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',required=True,choices=['prepare','infer','render'])
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/ruler_adaptive_padding_v3_pilot')
    parser.add_argument('--backup-v2',type=Path)
    args=parser.parse_args();output=args.output.resolve()
    if args.stage=='prepare':
        if args.backup_v2 is None:parser.error('--backup-v2 is required')
        prepare(output,args.backup_v2.resolve())
    else:{'infer':infer,'render':render}[args.stage](output)


if __name__=='__main__':main()
