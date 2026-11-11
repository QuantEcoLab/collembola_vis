"""Synthetic tests for pre-test checkpoint fixation, without model loading or evaluation."""

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

import evaluate_reviewed367_candidate as evaluation


class EvaluationLockTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir='/tmp/opencode',prefix='checkpoint-lock-test-')
        self.root=Path(self.temp.name);self.run=self.root/'run';self.output=self.root/'evaluation'
        for name in ['logs','initial','config','run/weights','evaluation']:(self.root/name).mkdir(parents=True,exist_ok=True)
        (self.root/'initial/exp02_os5_best.pt').write_bytes(b'fixed initial')
        (self.run/'weights/best.pt').write_bytes(b'selected epoch eight')
        for i in range(10):(self.run/'weights'/f'epoch{i}.pt').write_bytes(str(i).encode())
        config={'epochs':10,'batch':16,'imgsz':320,'resume':False}
        for path in [self.root/'config/train.yaml',self.run/'args.yaml']:path.write_text(yaml.safe_dump(config))
        (self.root/'logs/training_job.json').write_text(json.dumps({'state':'COMPLETED','returncode':0,'training_launches':1}))
        self.rows=[]
        for epoch in range(1,11):
            metrics={key:.8-.001*abs(epoch-8) for key in evaluation.METRIC_KEYS}
            self.rows.append({'epoch':epoch,**metrics})
        with (self.run/'results.csv').open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(self.rows[0]));writer.writeheader();writer.writerows(self.rows)
        self.patches=[patch.object(evaluation,'input_integrity',return_value={'run_directory':str(self.run)}),
            patch.object(evaluation,'FIXED_SHA',evaluation.sha(self.root/'initial/exp02_os5_best.pt'))]
        for item in self.patches:item.start()

    def tearDown(self):
        for item in reversed(self.patches):item.stop()
        self.temp.cleanup()

    def test_best_base_epoch_is_fixed_as_an_independent_copy_before_test(self):
        with patch.object(evaluation,'checkpoint_metadata',return_value={'train_metrics':self.rows[7]}):
            initial,candidate,selection=evaluation.lock_candidate(self.root,self.output)
        self.assertEqual(selection['matching_training_epochs'],[8])
        self.assertEqual(candidate.read_bytes(),(self.run/'weights/best.pt').read_bytes())
        self.assertNotEqual(candidate,self.run/'weights/best.pt')
        self.assertFalse(selection['test_used_for_selection'])
        self.assertFalse((self.output/'final_test_gate.json').exists())

    def test_inferior_epoch_cannot_be_substituted_for_best(self):
        with patch.object(evaluation,'checkpoint_metadata',return_value={'train_metrics':self.rows[1]}):
            with self.assertRaisesRegex(ValueError,'maximum Ultralytics base-validation fitness'):
                evaluation.lock_candidate(self.root,self.output)
        self.assertFalse((self.output/'candidate_selection.json').exists())

    def test_fixed_checkpoint_cannot_be_replaced_on_resume(self):
        with patch.object(evaluation,'checkpoint_metadata',return_value={'train_metrics':self.rows[7]}):
            evaluation.lock_candidate(self.root,self.output)
            (self.run/'weights/best.pt').write_bytes(b'different candidate')
            with self.assertRaisesRegex(ValueError,'Previously fixed candidate differs'):
                evaluation.lock_candidate(self.root,self.output)

    def test_training_failure_blocks_candidate_selection(self):
        (self.root/'logs/training_job.json').write_text(json.dumps({'state':'FAILED','returncode':1,'training_launches':1}))
        with self.assertRaisesRegex(ValueError,'has not completed successfully'):
            evaluation.lock_candidate(self.root,self.output)


if __name__=='__main__':unittest.main()
