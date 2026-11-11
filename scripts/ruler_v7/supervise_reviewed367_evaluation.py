#!/usr/bin/env python3
"""Wait for the one authorized run; launch one fixed-protocol paired evaluation."""

import json
import os
import time

from prepare_reviewed367_experiment import ROOT,DEFAULT_OUTPUT,sha
from supervise_reviewed367_training import capture,state,now


def main():
    experiment=DEFAULT_OUTPUT.resolve();logs=experiment/'logs';job_path=logs/'evaluation_job.json'
    with (logs/'single_evaluation_launch.lock').open('x') as handle:handle.write(now())
    job={'state':'WAITING_FOR_TRAINING','started_at':now(),'evaluation_launches':0,'experiment':str(experiment)}
    state(job_path,job)
    try:
        while True:
            training=json.loads((logs/'training_job.json').read_text())
            if training['state']=='FAILED':raise RuntimeError('Training failed; no evaluation will be started')
            if training['state']=='COMPLETED':break
            time.sleep(15)
        if training['returncode']!=0 or training['epochs_completed']!=10 or training['training_launches']!=1:
            raise RuntimeError('Unexpected training completion identity')
        recipe=ROOT/'scripts/ruler_v7/evaluate_reviewed367_candidate.py'
        with (logs/'evaluation_code.json').open('x') as handle:
            json.dump({str(path):sha(path) for path in [recipe,ROOT/'scripts/ruler_v7/evaluate_targeted_ruler_candidate.py',
                ROOT/'scripts/ruler_v7/reviewed367_runtime_guard.py']},handle,indent=2)
        command=[str(ROOT/'.venv-ruler-ft/bin/python'),'-u','-B',str(recipe),'--experiment',str(experiment)]
        environment=dict(os.environ,PYTHONUNBUFFERED='1',YOLO_AUTOINSTALL='false',YOLO_CONFIG_DIR=str(experiment/'runtime_settings'))
        job.update(state='RUNNING',evaluation_launches=1,command=command);state(job_path,job)
        code=capture(command,logs/'evaluation',environment);job.update(returncode=code,finished_at=now())
        if code:raise RuntimeError(f'Fixed-protocol evaluation failed with exit {code}; no automatic retry')
        summary=json.loads((experiment/'evaluation/summary.json').read_text())
        job.update(state='COMPLETED',candidate_sha256=summary['candidate_selection']['sha256'],
            base_retention_gate_pass=summary['base_retention_gate_pass'],
            stdout_sha256=sha(logs/'evaluation.stdout.log'),stderr_sha256=sha(logs/'evaluation.stderr.log'))
    except BaseException as exc:
        job.update(state='FAILED',error=str(exc),finished_at=now());state(job_path,job);raise
    state(job_path,job);print(json.dumps(job,indent=2),flush=True)


if __name__=='__main__':main()
