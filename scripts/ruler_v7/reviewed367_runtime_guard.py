"""Process-local cache relocation and write guards; no training parameter overrides."""

import hashlib
import json
import os
import sys
from pathlib import Path


def cache_path(path,root):
    key=hashlib.sha256(str(Path(path).resolve()).encode()).hexdigest()
    return root/(key+'.cache')


def install(experiment):
    experiment=experiment.resolve()
    cache=experiment/'runtime_cache';cache.mkdir(exist_ok=True)
    seal=json.loads((experiment/'experiment_seal.json').read_text())
    protected={(experiment/name).resolve() for name in seal['files']}
    protected.add(experiment/'experiment_seal.json')
    protected.update(Path(name).resolve() for name in json.loads((experiment/'evidence/external_input_hashes.json').read_text()))
    roots=[experiment/name for name in ['combined_v7_reviewed367','frozen_train_addon','frozen_ruler_evaluation']]
    def check(path):
        if not isinstance(path,(str,bytes,os.PathLike)):return
        resolved=Path(os.fsdecode(path)).resolve()
        if resolved in protected or any(resolved.is_relative_to(root) for root in roots):
            raise PermissionError('Frozen experiment/source path is read-only: '+str(resolved))
    def audit(event,args):
        if event=='open':
            path,mode,flags=args
            if flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND):check(path)
        elif event in {'os.remove','os.rmdir','os.mkdir','os.chmod','os.truncate'}:check(args[0])
        elif event in {'os.rename','os.replace'}:
            check(args[0]);check(args[1])
    sys.addaudithook(audit)
    import ultralytics.data.utils as utils
    import ultralytics.data.dataset as dataset
    original_load=utils.load_dataset_cache_file;original_save=utils.save_dataset_cache_file
    def load(path):return original_load(cache_path(path,cache))
    def save(prefix,path,value,version):
        target=cache_path(path,cache)
        with (experiment/'logs/cache_redirects.jsonl').open('a') as handle:
            handle.write(json.dumps({'requested':str(path),'actual':str(target)})+'\n')
        return original_save(prefix,target,value,version)
    utils.load_dataset_cache_file=dataset.load_dataset_cache_file=load
    utils.save_dataset_cache_file=dataset.save_dataset_cache_file=save
