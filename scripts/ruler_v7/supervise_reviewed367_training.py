#!/usr/bin/env python3
"""Run the one authorized sealed experiment once; retain complete child stdout/stderr."""

import csv
import json
import os
import selectors
import subprocess
import sys
from datetime import datetime,timezone
from pathlib import Path

import yaml

from prepare_reviewed367_experiment import ROOT, DEFAULT_OUTPUT, FIXED_SHA, sealed_files, sha


def now():return datetime.now(timezone.utc).isoformat()


def state(path,value):
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2));temporary.replace(path)


def capture(command,prefix,environment):
    with prefix.with_suffix('.stdout.log').open('xb') as stdout, prefix.with_suffix('.stderr.log').open('xb') as stderr, prefix.with_suffix('.combined.log').open('xb') as combined:
        child=subprocess.Popen(command,cwd=ROOT,env=environment,stdout=subprocess.PIPE,stderr=subprocess.PIPE,bufsize=0)
        selector=selectors.DefaultSelector();selector.register(child.stdout,selectors.EVENT_READ,stdout);selector.register(child.stderr,selectors.EVENT_READ,stderr)
        print(json.dumps({'pid':child.pid,'command':command,'log_prefix':str(prefix)}),flush=True)
        while selector.get_map():
            for key,_ in selector.select(timeout=5):
                chunk=os.read(key.fileobj.fileno(),65536)
                if chunk:
                    key.data.write(chunk);key.data.flush();combined.write(chunk);combined.flush()
                else:selector.unregister(key.fileobj);key.fileobj.close()
        selector.close();code=child.wait()
        for handle in [stdout,stderr,combined]:handle.flush();os.fsync(handle.fileno())
    return code


def main():
    experiment=DEFAULT_OUTPUT.resolve();logs=experiment/'logs';logs.mkdir(exist_ok=True)
    job_path=logs/'training_job.json'
    with (logs/'single_training_launch.lock').open('x') as handle:handle.write(now())
    job={'state':'CHECKING','started_at':now(),'experiment':str(experiment),'training_launches':0}
    state(job_path,job)
    environment=dict(os.environ,PYTHONUNBUFFERED='1',YOLO_AUTOINSTALL='false',YOLO_CONFIG_DIR=str(experiment/'runtime_settings'))
    python=str(ROOT/'.venv-ruler-ft/bin/python');runner=str(ROOT/'scripts/ruler_v7/run_reviewed367_experiment.py')
    check=[python,'-u','-B',runner,'--experiment',str(experiment),'--check']
    try:
        code=capture(check,logs/'pretrain_check',environment)
        if code:raise RuntimeError('Pretraining --check failed; no training launched')
        checked=json.loads((logs/'pretrain_check.stdout.log').read_text())
        expected={'base_val':{'images':463,'labels':463,'polygons':463},'train':{'images':2281,'labels':2281,'polygons':2285},
                  'ruler_val':{'images':1,'labels':1,'polygons':1},'ruler_test':{'images':3,'labels':3,'polygons':3}}
        if checked['counts']!=expected or checked['initial_checkpoint_sha256']!=FIXED_SHA or checked['sealed_files_verified']!=6258:
            raise RuntimeError('Checked counts/hash differ from the approved experiment')
        config=yaml.safe_load((experiment/'config/train.yaml').read_text())
        if any(config[k]!=v for k,v in {'epochs':10,'save_period':1,'fraction':1.,'resume':False,'batch':16,'imgsz':320,'lr0':.00005}.items()):
            raise RuntimeError('Approved training parameters differ')
        support=experiment/'runtime_support';support.mkdir(exist_ok=True)
        startup=support/'sitecustomize.py'
        with startup.open('x') as handle:
            handle.write('import os,sys,traceback\ntry:\n sys.path.insert(0,'+repr(str(ROOT/'scripts/ruler_v7'))+')\n from reviewed367_runtime_guard import install\n from pathlib import Path\n install(Path('+repr(str(experiment))+'))\nexcept BaseException:\n traceback.print_exc()\n os._exit(78)\n')
        environment['PYTHONPATH']=str(support)+(os.pathsep+environment['PYTHONPATH'] if environment.get('PYTHONPATH') else '')
        command=[python,'-u','-B',runner,'--experiment',str(experiment),'--train']
        job.update(state='RUNNING',check_returncode=0,training_launches=1,training_command=command,
            train_config_sha256=sha(experiment/'config/train.yaml'),seal_sha256=sha(experiment/'experiment_seal.json'),
            runtime_guard_sha256=sha(ROOT/'scripts/ruler_v7/reviewed367_runtime_guard.py'),stdout=str(logs/'training.stdout.log'),stderr=str(logs/'training.stderr.log'))
        state(job_path,job)
        code=capture(command,logs/'training',environment)
        job.update(returncode=code,finished_at=now())
        if code:raise RuntimeError(f'Authorized training exited {code}; automatic retry is forbidden')
        run=Path(json.loads((experiment/'experiment_seal.json').read_text())['run_directory'])
        with (run/'results.csv').open(newline='') as handle:results=list(csv.DictReader(handle))
        if [int(r['epoch']) for r in results]!=list(range(1,11)):raise RuntimeError('Expected exactly ten completed epochs')
        for i in range(10):
            if not (run/'weights'/f'epoch{i}.pt').is_file():raise RuntimeError(f'Missing epoch checkpoint {i}')
        actual=yaml.safe_load((run/'args.yaml').read_text())
        differences={k:{'approved':v,'actual':actual.get(k)} for k,v in config.items() if actual.get(k)!=v}
        if differences:raise RuntimeError('Training parameters drifted: '+json.dumps(differences))
        sealed_files(experiment,'experiment_seal.json')
        for path,digest in json.loads((experiment/'evidence/external_input_hashes.json').read_text()).items():
            if sha(Path(path))!=digest:raise RuntimeError('Original input changed: '+path)
        artifacts={p.relative_to(experiment).as_posix():sha(p) for p in run.rglob('*') if p.is_file()}
        with (logs/'training_artifact_hashes.json').open('x') as handle:json.dump(artifacts,handle,indent=2)
        job.update(state='COMPLETED',epochs_completed=10,epoch_checkpoints=10,original_and_prepared_inputs_unchanged=True,
            stdout_sha256=sha(logs/'training.stdout.log'),stderr_sha256=sha(logs/'training.stderr.log'),combined_sha256=sha(logs/'training.combined.log'))
    except BaseException as exc:
        job.update(state='FAILED',error=str(exc),finished_at=now());state(job_path,job);raise
    state(job_path,job);print(json.dumps(job,indent=2),flush=True)


if __name__=='__main__':main()
