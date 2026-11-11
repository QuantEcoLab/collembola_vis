#!/usr/bin/env python3
"""Finalize already completed legacy predictions after metadata-only failure. Never import a model."""

import json
from datetime import datetime,timezone
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from qualitative_unseen_three_models import (
    ROOT,OUTPUT,IMAGE_SHA,DEDUP_SHA,sha,write_json,csv_file,array_sha,mask_iou,confidence_stats,verify_inputs,
)


def main():
    output=OUTPUT;config=verify_inputs(output)
    if (output/'summary.json').exists():raise ValueError('Completed result exists; refusing to overwrite')
    with (output/'single_saved_result_finalization.lock').open('x') as handle:handle.write(datetime.now(timezone.utc).isoformat())
    error_log=output/'logs/inference.stderr.log';error=error_log.read_text()
    if "KeyError: 'half'" not in error:raise ValueError('This recovery applies only to the recorded metadata failure')
    shared=json.loads((output/'shared/crop_manifest.json').read_text());detections=json.loads((output/'shared/detections.json').read_text())
    trace=json.loads((output/'shared/detector_trace.json').read_text())
    metadata=json.loads((output/'detector'/f'{config["source_id"]}_detector_metadata.json').read_text())
    if len(shared)!=len(detections) or len(trace['calls'])!=metadata['num_tiles']:raise ValueError('Incomplete shared detector results')
    all_records={};summaries={};argument_sets=[]
    for tag,model in config['models'].items():
        base=output/'model_runs'/tag
        records=json.loads((base/'raw_records.json').read_text());legacy=json.loads((base/'summary.json').read_text())
        if len(records)!=len(shared) or legacy['segmented_rows']!=len(shared):raise ValueError('A model pass did not finish')
        if {r['detection_id'] for r in records}!={r['detection_id'] for r in shared}:raise ValueError('Different detection ID set')
        kept=set(legacy['kept_detection_ids'])
        for row,crop in zip(records,shared):
            if row['detection_id']!=crop['detection_id'] or row['input_array_sha256']!=crop['model_input_array_sha256']:
                raise ValueError('Different model input crop')
            expected=output/crop['image'];stem=expected.stem
            if sha(base/'all_crops'/(stem+'.png'))!=crop['image_sha256']:raise ValueError('Saved crops are not identical')
            if sha(output/row['mask'])!=row['mask_sha256']:raise ValueError('Saved mask hash changed')
            binary=cv2.imread(str(output/row['mask']),cv2.IMREAD_GRAYSCALE)
            if binary.shape!=(crop['height'],crop['width']) or array_sha(binary)!=row['mask_array_sha256']:
                raise ValueError('Saved mask array changed')
            if row['mask_area_px']!=int(np.count_nonzero(binary)):raise ValueError('Area mismatch')
            if row['kept_after_dedup']!=(row['detection_id'] in kept):raise ValueError('Dedup status changed')
            argument_sets.append(json.dumps(row['predict_kwargs'],sort_keys=True))
        summaries[tag]={'label':model['label'],'all_detections':len(detections),'no_mask':sum(not r['has_mask'] for r in records),
            'duplicates_removed':len(detections)-len(kept),'final_boxes':len(kept),
            'final_boxes_with_mask':sum(r['has_mask'] and r['kept_after_dedup'] for r in records),
            'segmentation_confidence':confidence_stats([r['confidence'] for r in records]),
            'edge_touching_masks':sum(r['touches_crop_edge'] for r in records),
            'multi_component_masks':sum(r['components']>1 for r in records),
            'legacy_summary':legacy,'full_overlay':str((ROOT/legacy['full_overlay']).relative_to(output))}
        all_records[tag]=records
    if len(set(argument_sets))!=1:raise ValueError('Explicit segmentation arguments differ')
    write_json(output/'evidence/effective_predictor_arguments.json',{
        'detector':trace['effective_parameters'],'all_3240_segmentation_calls_identical_kwargs':json.loads(argument_sets[0]),
        'runtime':config['runtime'],'parameter_defaults_source':'Same unchanged Ultralytics 8.4.175 legacy predict implementation for all models',
        'metadata_capture_error':'args.half is absent in this runtime. It was not passed or changed. Model execution and legacy dedup had already finished.',
        'unsupported_argument_was_not_substituted_or_retuned':True})
    Image.MAX_IMAGE_PIXELS=None
    with Image.open(config['input']) as image:full=cv2.cvtColor(np.array(image.convert('RGB')),cv2.COLOR_RGB2BGR)
    height,width=full.shape[:2];comparison=[]
    for index,crop in enumerate(shared):
        models={tag:records[index] for tag,records in all_records.items()};tags=list(models);ious={}
        masks={tag:cv2.imread(str(output/row['mask']),cv2.IMREAD_GRAYSCALE)>0 for tag,row in models.items()}
        for i,a in enumerate(tags):
            for b in tags[i+1:]:ious[a+'__'+b]=mask_iou(masks[a],masks[b])
        x1,y1,x2,y2=crop['crop_box'];cw=min(width,max(640,4*(x2-x1)));ch=min(height,max(640,4*(y2-y1)))
        left=max(0,min(int((x1+x2-cw)/2),width-cw));top=max(0,min(int((y1+y2-ch)/2),height-ch))
        context=full[top:top+ch,left:left+cw].copy();cv2.rectangle(context,(x1-left,y1-top),(x2-left,y2-top),(0,0,255),2)
        path=output/'context'/f'det{crop["detection_id"]:04d}.jpg'
        if not cv2.imwrite(str(path),context,[cv2.IMWRITE_JPEG_QUALITY,92]):raise OSError('Context write failed')
        areas=[r['mask_area_px'] for r in models.values()]
        comparison.append({**crop,'models':models,'pairwise_mask_iou':ious,'minimum_mask_iou':min(ious.values()),
            'dedup_disagreement':len({r['kept_after_dedup'] for r in models.values()})>1,
            'mask_presence_disagreement':len({r['has_mask'] for r in models.values()})>1,'mask_area_range':max(areas)-min(areas),
            'context':path.relative_to(output).as_posix(),'context_box':[left,top,left+cw,top+ch]})
    ranked=sorted(comparison,key=lambda r:(-int(r['dedup_disagreement']),-int(r['mask_presence_disagreement']),r['minimum_mask_iou'],-r['mask_area_range'],r['detection_id']))
    write_json(output/'comparison_manifest.json',comparison)
    differences=[{k:r[k] for k in ['detection_id','dedup_disagreement','mask_presence_disagreement','minimum_mask_iou','mask_area_range']} for r in ranked]
    write_json(output/'largest_differences.json',differences);csv_file(output/'largest_differences.csv',differences,list(differences[0]))
    for tag,path in [('original',Path(config['input'])),*[(tag,output/s['full_overlay']) for tag,s in summaries.items()]]:
        with Image.open(path) as image:
            if list(image.size)!=config['image_dimensions']:raise ValueError('Overlay is not full resolution')
            preview=image.copy();preview.thumbnail((1500,1500));preview.save(output/'previews'/(tag+'.jpg'),quality=94)
    recovery={'reason':'KeyError half during metadata comparison after every neural/legacy dedup pass completed',
        'original_driver_returncode':json.loads((output/'logs/run_job.json').read_text())['returncode'],
        'original_stderr_sha256':sha(error_log),'finalizer_source_sha256':sha(Path(__file__)),
        'detector_reruns':0,'segmentation_reruns':0,'classifier_reruns':0,
        'method':'Reused and hash-verified existing detector trace, shared PNGs, all 3240 raw mask records and three original legacy dedup summaries. Only context, statistics and presentation generated.'}
    write_json(output/'logs/postprocessing_recovery.json',recovery)
    summary={'status':'COMPLETED','qualitative_only':True,'ground_truth_available':False,'training_started':False,
        'source_id':config['source_id'],'image':Path(config['input']).relative_to(output).as_posix(),'image_sha256':IMAGE_SHA,
        'dimensions':config['image_dimensions'],'detector_passes':1,'detector_tile_predict_calls':len(trace['calls']),
        'detections_before_bbox_nms':metadata['detections_before_bbox_nms'],'detections_after_bbox_nms':len(detections),
        'detector_confidence':confidence_stats([r['conf'] for r in detections]),'shared_detections_sha256':sha(output/'shared/detections.json'),
        'shared_crop_manifest_sha256':sha(output/'shared/crop_manifest.json'),'models':summaries,
        'model_hashes':{tag:model['sha256'] for tag,model in config['models'].items()},'runtime':config['runtime'],
        'all_crops_and_detections_identical':True,'parameters_unchanged':True,'dedup_classifier_sha256':DEDUP_SHA,
        'mask_presence_disagreements':sum(r['mask_presence_disagreement'] for r in comparison),
        'dedup_disagreements':sum(r['dedup_disagreement'] for r in comparison),
        'largest_difference_ids':[r['detection_id'] for r in ranked[:50]],
        'blind_status':'Consumed by this qualitative model comparison/visual review; not a blind test for future claims',
        'legacy_render_note':config['legacy_render_note'],'postprocess_recovery':recovery}
    write_json(output/'summary.json',summary)
    write_json(output/'qualitative_exposure.json',{'source_id':config['source_id'],'image_sha256':IMAGE_SHA,
        'comparison_generated_at_utc':datetime.now(timezone.utc).isoformat(),'subsequent_blind_test_claim_allowed':False,
        'manual_gt_created':False,'model_selection_or_threshold_tuning_performed':False})
    verify_inputs(output)
    write_json(output/'result_seal.json',{'files':{p.relative_to(output).as_posix():sha(p) for folder in ['input','shared','model_runs','context','previews','evidence']
        for p in (output/folder).rglob('*') if p.is_file()},'records':{name:sha(output/name) for name in ['summary.json','comparison_manifest.json','largest_differences.json','largest_differences.csv','qualitative_exposure.json']}})
    write_json(output/'completion_status.json',{'state':'COMPLETED_FROM_EXISTING_PREDICTIONS','detector_passes':1,
        'detector_tile_predict_calls':len(trace['calls']),'segmentation_predict_calls':{tag:len(rows) for tag,rows in all_records.items()},
        'postprocessing_only_recovery':True,'model_input_and_hash_verification_passed':True})
    print(json.dumps(summary,indent=2,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
