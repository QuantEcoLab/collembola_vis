#!/usr/bin/env python3
"""Make an independent 5x ruler sampling variant; never install or train."""

import argparse
import ast
import copy
import json
import math
import shutil
from collections import Counter
from pathlib import Path
from types import MethodType, SimpleNamespace

import yaml

from prepare_targeted_ruler_experiment import ROOT, check_records, sha, write_csv, read_csv
from run_targeted_ruler_experiment import verify


def exposure(base, ruler, factor, epochs=10, batch=16):
    slots=ruler*factor
    total=base+slots
    def probability(size):
        return 1-math.prod((total-slots-i)/(total-i) for i in range(size))
    full,remainder=divmod(total,batch)
    expected_batches=full*probability(batch)+(probability(remainder) if remainder else 0)
    return {'base_slots_per_epoch':base,'unique_ruler_images':ruler,'ruler_slots_per_epoch':slots,
            'total_slots_per_epoch':total,'ruler_share_percent':100*slots/total,
            'epochs':epochs,'batch':batch,'batches_per_epoch':math.ceil(total/batch),
            'expected_batches_with_ruler_per_epoch':expected_batches,
            'expected_batches_with_ruler_over_run':expected_batches*epochs,
            'ruler_visits_over_run':slots*epochs,'visits_per_unique_ruler':factor*epochs,
            'assumptions':'single-process CPU, shuffle=True, fraction=1, drop_last=False, all epochs completed; visits are not optimizer steps or guaranteed surviving post-augmentation masks'}


def source_audit(source_root, provenance_path):
    provenance=json.loads(provenance_path.read_text())
    if provenance['version']!='8.4.175' or provenance['installed'] or provenance['imported']:
        raise ValueError('Expected uninstalled exact-version source evidence')
    for name,digest in provenance['source_hashes'].items():
        if sha(source_root/name)!=digest:
            raise ValueError('Upstream source hash mismatch')
    trainer_path=source_root/'ultralytics/engine/trainer.py'
    dataset_path=source_root/'ultralytics/data/dataset.py'
    trainer=ast.parse(trainer_path.read_text())
    dataset=ast.parse(dataset_path.read_text())
    segment=ast.parse((source_root/'ultralytics/models/yolo/segment/train.py').read_text())
    segment_class=next(n for n in segment.body if isinstance(n,ast.ClassDef) and n.name=='SegmentationTrainer')
    segment_bases=[ast.unparse(n) for n in segment_class.bases]
    if not any(name.endswith('DetectionTrainer') for name in segment_bases):
        raise ValueError('Segmentation does not use the checked detection trainer')
    if any(isinstance(n,ast.FunctionDef) and n.name in {'get_dataloader','_do_train','_close_dataloader_mosaic'} for n in segment_class.body):
        raise ValueError('Segmentation overrides the checked loader/mosaic behavior')
    base=next(n for n in trainer.body if isinstance(n,ast.ClassDef) and n.name=='BaseTrainer')
    ds_class=next(n for n in dataset.body if isinstance(n,ast.ClassDef) and n.name=='YOLODataset')
    gate=next(n for n in ast.walk(base) if isinstance(n,ast.If)
              and 'not mosaic_closed' in ast.unparse(n.test)
              and 'self.args.close_mosaic' in ast.unparse(n.test))
    fn_close=next(n for n in base.body if isinstance(n,ast.FunctionDef) and n.name=='_close_dataloader_mosaic')
    fn_dataset=next(n for n in ds_class.body if isinstance(n,ast.FunctionDef) and n.name=='close_mosaic')
    namespace={'LOGGER':SimpleNamespace(info=lambda text:None),'copy':copy.copy}
    module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),fn_close,fn_dataset],type_ignores=[])
    # Only the two actual setter functions run against inert namespaces, not a
    # trainer, dataset, transform pipeline, torch, or model instance.
    exec(compile(ast.fix_missing_locations(module),'upstream_mosaic_setters','exec'),namespace)
    hyp=SimpleNamespace(close_mosaic=10,mosaic=1.0,copy_paste=0.0,mixup=0.0,cutmix=0.0)
    ds=SimpleNamespace(mosaic=True,build_transforms=lambda hyp:vars(hyp).copy())
    ds.close_mosaic=MethodType(namespace['close_mosaic'],ds)
    fake=SimpleNamespace(epochs=10,args=hyp,train_loader=SimpleNamespace(dataset=ds))
    closed=False;active=[];close_epoch=None
    condition=compile(ast.Expression(gate.test),'upstream_mosaic_gate','eval')
    for epoch in range(10):
        if eval(condition,{},dict(self=fake,mosaic_closed=closed,epoch=epoch)):
            namespace['_close_dataloader_mosaic'](fake);closed=True;close_epoch=epoch
        if not closed:active.append(epoch+1)
    if close_epoch!=0 or active or ds.transforms['mosaic']!=0.0:
        raise ValueError('Mosaic finding differs from expected exact-version behavior')
    snippets={}
    ranges={
        'ultralytics/engine/trainer.py':[(486,508),(515,535),(1143,1149)],
        'ultralytics/data/dataset.py':[(386,396)],
        'ultralytics/models/yolo/detect/train.py':[(90,104)],
        'ultralytics/models/yolo/segment/train.py':[(1,25)],
        'ultralytics/data/build.py':[(280,300),(380,415)],
        'ultralytics/data/base.py':[(434,449)],
        'ultralytics/data/augment.py':[(1150,1171),(1512,1524),(2870,2911)],
    }
    for name,regions in ranges.items():
        lines=(source_root/name).read_text().splitlines()
        snippets[name]=[{'first_line':a,'last_line':b,'text':'\n'.join(lines[a-1:b])} for a,b in regions]
    return {**provenance,'gate':ast.unparse(gate.test),'gate_line':gate.lineno,
            'segmentation_trainer_bases':segment_bases,
            'zero_based_close_epoch':close_epoch,'mosaic_active_human_epochs':active,
            'mosaic_effective_probability_all_epochs':0.0,
            'mixup_cutmix_copy_paste_effective_probability':0.0,
            'runtime_execution':'AST condition and two real setter bodies on inert namespaces only; no imports/installation/model execution',
            'source_excerpts':snippets,
            'correction_to_prior_control_readme':'8.4.175 uses >=, not ==; epochs5/close_mosaic10 would also close from epoch0. The old frozen README was not edited.'}


def prepare(control,output,source_root,source_provenance):
    if output.exists():raise ValueError(f'Refusing to overwrite {output}')
    parent_seal,control_config,_runtime=verify(control)
    if Path(parent_seal['run_directory']).exists():raise ValueError('Control has already been run; investigate first')
    control_hashes={p.relative_to(control).as_posix():sha(p) for p in control.rglob('*') if p.is_file()}
    audit=source_audit(source_root,source_provenance)
    records=json.loads((control/'manifests/combined_inventory.json').read_text())
    ruler=[r for r in records if r['origin']=='approved_manual_ruler' and r['split']=='train']
    if len(ruler)!=20 or sum(r['polygons'] for r in ruler)!=21:
        raise ValueError('Expected exactly the approved 20/21 manual train addition')
    eval_rows=read_csv(control/'manifests/frozen_addon.csv')
    shutil.copytree(control,output,ignore=shutil.ignore_patterns('experiment_seal.json','README.md','resource_estimates.json'))
    ledger=[]
    def add_ledger(row,repeat):
        ledger.append({'case_id':row['case_id'],'repeat_index':repeat,'total_factor':5,
                       'source_id':row['source_id'],'split':'train','polygon_count':row['polygons'],
                       'image':row['image'],'label':row['label'],
                       'canonical_image':'frozen_ruler_addon/'+row['frozen_image'],
                       'canonical_label':'frozen_ruler_addon/'+row['frozen_label'],
                       'image_sha256':row['image_sha256'],'label_sha256':row['label_sha256']})
    for row in ruler:
        row['oversampling_repeat']=1
        add_ledger(row,1)
        for repeat in range(2,6):
            replica=copy.deepcopy(row)
            replica['oversampling_repeat']=repeat
            for kind in ['image','label']:
                original=Path(row[kind]);name=f'{original.stem}__os5_rep{repeat:02d}{original.suffix}'
                destination=original.with_name(name)
                shutil.copy2(control/original,output/destination)
                if sha(output/destination)!=row[kind+'_sha256']:
                    raise ValueError('Oversampling changed image/label bytes')
                replica[kind]=destination.as_posix();replica[kind+'_name']=name
            records.append(replica);add_ledger(replica,repeat)
    separate=[]
    for row in eval_rows:
        if row['split']=='train':continue
        separate.append({'source_id':row['source_id'],'split':'ruler_val' if row['split']=='val' else 'ruler_test',
                         'image_sha256':row['image_sha256'],'label_sha256':row['label_sha256'],
                         'image_name':Path(row['image']).name,'label_name':Path(row['label']).name,
                         'polygons':int(row['polygon_count'])})
    checks=check_records(records+separate)
    expected={'train':{'images':2014,'polygons':2023},'base_val':{'images':463,'polygons':463},
              'ruler_val':{'images':1,'polygons':1},'ruler_test':{'images':3,'polygons':3}}
    if checks['counts']!=expected:raise ValueError('Unexpected oversampled counts')
    if Counter(r['case_id'] for r in ledger)!=Counter({r['case_id']:5 for r in ruler}):
        raise ValueError('Unequal ruler multiplicity')
    if len({r['image_sha256'] for r in separate})!=4:
        raise ValueError('Duplicate targeted evaluation images')
    for row in records:
        for kind in ['image','label']:
            if sha(output/row[kind])!=row[kind+'_sha256']:raise ValueError('Copied sample changed')
    (output/'manifests/combined_inventory.json').write_text(json.dumps(records,indent=2))
    (output/'manifests/preflight.json').write_text(json.dumps(checks,indent=2))
    write_csv(output/'manifests/ruler_oversampling.csv',ledger,list(ledger[0]))
    distribution=[]
    for split in sorted({r['split'] for r in records+separate}):
        for source in sorted({r['source_id'] for r in records+separate if r['split']==split}):
            same=[r for r in records+separate if r['split']==split and r['source_id']==source]
            unique={r['image_sha256']:r for r in same}
            distribution.append({'split':split,'source_id':source,'samples':len(same),
                                 'polygon_visits':sum(r['polygons'] for r in same),
                                 'unique_images':len(unique),'unique_polygons':sum(r['polygons'] for r in unique.values())})
    write_csv(output/'manifests/source_distribution.csv',distribution,['split','source_id','samples','polygon_visits','unique_images','unique_polygons'])
    config=copy.deepcopy(control_config)
    config.update(model=str(output/'initial/v7_best.pt'),data=str(output/'combined_v7_ruler/dataset.yaml'),
                  project=str(output/'training_runs'),name='v7_ruler_ft_exp02_os5')
    for key in config:
        if key not in {'model','data','project','name'} and config[key]!=control_config[key]:
            raise ValueError('Oversampling must not change training/augmentation hyperparameters')
    (output/'config/train.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
    for filename,root,test in [('combined_v7_ruler/dataset.yaml',output/'combined_v7_ruler',False),
                               ('config/ruler_evaluation.yaml',output/'frozen_ruler_addon',True)]:
        values={'path':str(root),'train':'images/train','val':'images/val','names':{0:'trunk'}}
        if test:values['test']='images/test'
        (output/filename).write_text(yaml.safe_dump(values,sort_keys=False))
    provenance=json.loads((output/'evidence/provenance.json').read_text())
    provenance['frozen_checkpoint']=str(output/'initial/v7_best.pt')
    (output/'evidence/provenance.json').write_text(json.dumps(provenance,indent=2))
    diff={k:{'original':provenance['original_training_args'][k],'proposed':v} for k,v in config.items()
          if provenance['original_training_args'][k]!=v}
    (output/'config/changes_from_original.json').write_text(json.dumps(diff,indent=2))
    evaluation=json.loads((output/'config/evaluation_plan.json').read_text())
    evaluation['candidate_checkpoint']='training_runs/v7_ruler_ft_exp02_os5/weights/best.pt'
    evaluation['oversampling_applies_to_training_only']=True
    (output/'config/evaluation_plan.json').write_text(json.dumps(evaluation,indent=2))
    (output/'evidence/ultralytics_8_4_175_source_audit.json').write_text(json.dumps(audit,indent=2))
    (output/'evidence/oversampling_preparation_code.json').write_text(json.dumps({Path(__file__).name:sha(Path(__file__))},indent=2))
    report={'factor':5,'definition':'five TOTAL entries per approved ruler image (one original plus four copies), not six',
            'control':str(control),'control_seal_sha256':sha(control/'experiment_seal.json'),
            'base_unique_images':1914,'unique_ruler_images':20,'unique_ruler_polygons':21,
            'control_exposure':exposure(1914,20,1),'recommended_exposure':exposure(1914,20,5),
            'per_epoch_polygon_visits':{'control':1939,'oversampled':2023,'ruler_control':21,'ruler_oversampled':105},
            'sample_count_increase_percent':100*(2014/1934-1),
            'evaluation_unique_images':{'base_val':463,'ruler_val':1,'ruler_test':3},
            'augmentations_changed':False,'mosaic_active_epochs':[],
            'sampling_order':'physical unique filenames; unweighted shuffled sampling of indices with seed0 and workers0; each copy index visited once per epoch',
            'augmentation_policy':'fresh random transform draws per __getitem__, never reset by filename; deterministic across runs with identical runtime/seed; no guarantee every rendered copy differs or every object survives augmentation',
            'blocked_cases_included':[],'models_run':False,'environment_installed':False}
    (output/'oversampling_design.json').write_text(json.dumps(report,indent=2))
    seal=copy.deepcopy(parent_seal)
    seal.update(run_directory=str(output/'training_runs/v7_ruler_ft_exp02_os5'),
                base_plus_addon_counts=checks['counts'],data_preflight=checks,
                oversampling_factor=5,control_experiment=str(control))
    seal.pop('preparation_checks',None)
    seal['preparation_checks']={'combined_image_pairs':2477,'combined_polygons':2486,'ruler_sampling_entries':100,
        'ruler_unique_images':20,'ruler_unique_polygons':21,'blocked_reference_count_excluded':9,
        'evaluation_oversampling':False,'augmentation_hyperparameters_unchanged':True}
    seal['files']={p.relative_to(output).as_posix():sha(p) for p in output.rglob('*') if p.is_file()}
    (output/'experiment_seal.json').write_text(json.dumps(seal,indent=2))
    now={p.relative_to(control).as_posix():sha(p) for p in control.rglob('*') if p.is_file()}
    if now!=control_hashes:raise ValueError('Control experiment changed')
    print(json.dumps({'recommended':str(output),'counts':checks['counts'],'ledger_rows':len(ledger),
                      'control_unchanged':True,'training_started':False,'inference_started':False},indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--control',type=Path,default=ROOT/'outputs/ruler_v7_targeted_ft_exp01')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'outputs/ruler_v7_targeted_ft_exp02_os5')
    parser.add_argument('--source-root',type=Path,default=Path('/tmp/opencode/ultralytics_8_4_175_source'))
    parser.add_argument('--source-provenance',type=Path,default=Path('/tmp/opencode/ultralytics_8_4_175_source_provenance.json'))
    args=parser.parse_args()
    prepare(args.control.resolve(),args.output_dir.resolve(),args.source_root.resolve(),args.source_provenance.resolve())


if __name__=='__main__':
    main()
