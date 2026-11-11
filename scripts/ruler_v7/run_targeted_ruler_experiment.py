#!/usr/bin/env python3
"""Verify a sealed experiment, or train ONLY with the explicit --train flag."""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path

import yaml

ROOT=Path(__file__).resolve().parents[2]


def verify(experiment):
    seal=json.loads((experiment/'experiment_seal.json').read_text())
    for relative,expected in seal['files'].items():
        path=(experiment/relative).resolve()
        if not path.is_relative_to(experiment.resolve()) or not path.is_file():
            raise ValueError(f'Missing/outside experiment file: {relative}')
        if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
            raise ValueError(f'Frozen experiment input changed: {relative}')
    config=yaml.safe_load((experiment/'config/train.yaml').read_text())
    if config['resume'] is not False or config['imgsz']!=320 or config['device']!='cpu':
        raise ValueError('Unexpected experiment mode/preprocessing/device')
    if Path(config['model']).resolve()!=experiment/'initial/v7_best.pt':
        raise ValueError('Unexpected starting checkpoint')
    if Path(config['data']).resolve()!=experiment/'combined_v7_ruler/dataset.yaml':
        raise ValueError('Unexpected training data')
    runtime={}
    for name in ['torch','ultralytics']:
        runtime[name]=importlib.metadata.version(name) if importlib.util.find_spec(name) else None
    return seal,config,runtime


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment',type=Path,default=ROOT/'outputs/ruler_v7_targeted_ft_exp01')
    action=parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--check',action='store_true',help='Read-only seal/config/runtime check; no model imports')
    action.add_argument('--train',action='store_true',help='Execute approved training, only after explicit user approval')
    args=parser.parse_args()
    experiment=args.experiment.resolve()
    seal,config,runtime=verify(experiment)
    if args.check:
        print(json.dumps({'sealed_files_verified':len(seal['files']),
                          'initial_checkpoint_sha256':seal['initial_checkpoint_sha256'],
                          'counts':seal['base_plus_addon_counts'],'runtime_installed':runtime,
                          'required_ultralytics_version':seal['required_ultralytics_version'],
                          'training_started':False,'inference_started':False},indent=2))
        return
    if not all(runtime.values()):
        raise SystemExit('Install the isolated CPU runtime first; no training started.')
    if runtime['ultralytics']!=seal['required_ultralytics_version']:
        raise SystemExit('Ultralytics version differs from recorded V7; no training started.')
    run=Path(seal['run_directory'])
    if run.exists():
        raise SystemExit('Refusing to reuse or overwrite an existing run.')
    # Lazy imports: neither --check nor dataset preparation loads a model.
    import torch
    from ultralytics import YOLO
    torch.set_num_threads(min(4,os.cpu_count() or 1))
    (experiment/'training_runtime.json').write_text(json.dumps({
        'torch':torch.__version__,'ultralytics':runtime['ultralytics'],
        'torch_threads':torch.get_num_threads(),'device':config['device']},indent=2))
    YOLO(config['model']).train(**config)


if __name__=='__main__':
    main()
