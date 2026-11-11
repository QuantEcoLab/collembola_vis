"""Preparation tests: fail on split leakage/invalid labels, with no training."""

import io
import hashlib
import json
import pickle
import sys
import tempfile
import unittest
from pathlib import Path

from prepare_targeted_ruler_experiment import MetadataOnly, base_source, check_records, label_count
from run_targeted_ruler_experiment import verify


class PreparationTests(unittest.TestCase):
    def test_source_identity_merges_batch_and_audit_aliases(self):
        self.assertEqual(base_source('K1_Fe2O3001_audit_good_det0400'),'K1_Fe2O3001')
        self.assertEqual(base_source('C_5_2_HDPE017_batch1_det0033'),'C_5_2_HDPE017')
        self.assertEqual(base_source('C_4_3_PS_500034_ruler_0_20__selection_x__inst_001'),'C_4_3_PS_500034')
        with self.assertRaises(ValueError):base_source('unknown')

    def test_cross_split_sources_images_or_label_hashes_block_preparation(self):
        first=dict(source_id='source_a',image_sha256='image_a',label_sha256='label_a',
                   image_name='a.png',label_name='a.txt',polygons=1,split='train')
        second=dict(source_id='source_b',image_sha256='image_b',label_sha256='label_b',
                    image_name='b.png',label_name='b.txt',polygons=1,split='base_val')
        self.assertEqual(check_records([first,second])['source_overlap'],[])
        for field in ['source_id','image_sha256','label_sha256']:
            bad={**second,field:first[field]}
            with self.assertRaises(ValueError):check_records([first,bad])

    def test_invalid_and_empty_labels_are_rejected(self):
        with tempfile.TemporaryDirectory(dir='/tmp/opencode') as folder:
            path=Path(folder)/'label.txt'
            path.write_text('0 0.1 0.1 0.9 0.1 0.5 0.9\n')
            self.assertEqual(label_count(path),1)
            for content in ['', '0 0.1 0.1 1.2 0.1 0.5 0.9\n',
                            '0 0.1 0.1 0.2 0.2 0.3 0.3\n',
                            '0 0.1 0.1 nan 0.1 0.5 0.9\n']:
                path.write_text(content)
                with self.assertRaises(ValueError):label_count(path)

    def test_pickle_globals_are_inert_not_executed(self):
        class Dangerous:
            def __reduce__(self):return (eval, ('1/0',))
        payload=pickle.dumps({'version':'8.4.175','model':Dangerous()})
        meta=MetadataOnly(io.BytesIO(payload)).load()
        self.assertEqual(meta['version'],'8.4.175')

    def test_seal_detects_mutation_without_importing_model_runtime(self):
        import yaml
        with tempfile.TemporaryDirectory(dir='/tmp/opencode') as folder:
            root=Path(folder)
            (root/'config').mkdir();(root/'initial').mkdir();(root/'combined_v7_ruler').mkdir()
            (root/'initial/v7_best.pt').write_bytes(b'fake checkpoint, not loaded')
            config={'resume':False,'imgsz':320,'device':'cpu','model':str(root/'initial/v7_best.pt'),
                    'data':str(root/'combined_v7_ruler/dataset.yaml')}
            (root/'config/train.yaml').write_text(yaml.safe_dump(config))
            seal={'files':{p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in root.rglob('*') if p.is_file()}}
            (root/'experiment_seal.json').write_text(json.dumps(seal))
            before=set(sys.modules)
            verify(root)
            self.assertNotIn('torch',set(sys.modules)-before)
            self.assertNotIn('ultralytics',set(sys.modules)-before)
            (root/'initial/v7_best.pt').write_bytes(b'changed')
            with self.assertRaises(ValueError):verify(root)


if __name__=='__main__':
    unittest.main()
