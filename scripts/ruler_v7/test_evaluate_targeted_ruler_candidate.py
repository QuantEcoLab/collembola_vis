"""Synthetic metric tests; no real model forward or training."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from evaluate_targeted_ruler_candidate import pixel_metrics,validator_class


class EvaluationTests(unittest.TestCase):
    def test_same_area_does_not_hide_wrong_shape(self):
        gt=np.zeros((20,20),dtype=bool);gt[5:15,5:15]=True
        pred=np.zeros_like(gt);pred[5:15,10:20]=True
        score=pixel_metrics(gt,pred)
        self.assertEqual(score['area_ratio'],1)
        self.assertEqual(score['pixel_recall'],.5)
        self.assertEqual(score['pixel_precision'],.5)
        self.assertAlmostEqual(score['iou'],1/3)
        self.assertEqual(score['dice'],.5)

    def test_empty_mask_and_excess_pixels_are_penalized(self):
        gt=np.zeros((20,20),dtype=bool);gt[5:15,5:15]=True
        empty=pixel_metrics(gt,np.zeros_like(gt))
        self.assertEqual(empty['iou'],0)
        self.assertEqual(empty['pixel_recall'],0)
        self.assertEqual(empty['area_ratio'],0)
        excess=pixel_metrics(gt,np.ones_like(gt))
        self.assertEqual(excess['pixel_recall'],1)
        self.assertEqual(excess['pixel_precision'],.25)
        self.assertEqual(excess['iou'],.25)

    def test_capture_metrics_produces_correct_source_groups_without_forward(self):
        from ultralytics.utils.metrics import SegmentMetrics
        with tempfile.TemporaryDirectory(dir='/tmp/opencode') as folder:
            root=Path(folder);klass=validator_class(root,{'one.png':'source_a','two.png':'source_b'})
            obj=klass.__new__(klass)
            obj.names={0:'trunk'};obj.metrics=SegmentMetrics(names=obj.names)
            obj.args=SimpleNamespace(plots=False,save_json=False,task='segment')
            obj.save_dir=root
            obj.captured=[]
            for name in ['one.png','two.png']:
                row={'tp':np.ones((1,10),dtype=bool),'tp_m':np.ones((1,10),dtype=bool),
                     'conf':np.array([.9]),'pred_cls':np.array([0]),'target_cls':np.array([0]),
                     'target_img':np.array([0]),'im_name':name}
                obj.captured.append(row);obj.metrics.update_stats(row)
            result=obj.get_stats()
            self.assertGreater(result['metrics/mAP50-95(M)'],.9)
            source=json.loads((root/'per_source_metrics.json').read_text())
            self.assertEqual({r['source_id'] for r in source},{'source_a','source_b'})
            self.assertTrue(all(r['images']==1 for r in source))
            self.assertEqual(len(json.loads((root/'per_image_validation_stats.json').read_text())),2)


if __name__=='__main__':
    unittest.main()
