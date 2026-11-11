#!/usr/bin/env python3
"""Verify the exact approved 367-sample experiment; model imports only under --train."""

import argparse
import importlib.metadata
import json
from pathlib import Path

import yaml

from prepare_reviewed367_experiment import DEFAULT_OUTPUT, verify


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment',type=Path,default=DEFAULT_OUTPUT)
    action=parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--check',action='store_true')
    action.add_argument('--train',action='store_true')
    args=parser.parse_args();experiment=args.experiment.resolve();seal=verify(experiment)
    versions={package:importlib.metadata.version(package) for package in ['torch','ultralytics']}
    if versions['ultralytics']!='8.4.175':raise SystemExit('Expected the frozen Ultralytics 8.4.175 runtime')
    config=yaml.safe_load((experiment/'config/train.yaml').read_text())
    if args.check:
        print(json.dumps({'sealed_files_verified':len(seal['files']),'counts':seal['counts'],
            'runtime':versions,'initial_checkpoint_sha256':seal['initial_checkpoint_sha256'],
            'training_started':False,'evaluation_started':False,'inference_started':False},indent=2));return
    import torch
    from ultralytics import YOLO
    torch.set_num_threads(4)
    with (experiment/'training_runtime.json').open('x') as handle:
        json.dump({**versions,'torch_threads':4,'device':config['device']},handle,indent=2)
    YOLO(config['model']).train(**config)


if __name__=='__main__':main()
