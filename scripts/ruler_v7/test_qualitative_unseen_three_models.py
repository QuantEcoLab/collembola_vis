"""Geometry/identity checks for the one-pass legacy comparison; no inference."""

import unittest
import numpy as np

from qualitative_unseen_three_models import tile_origins,crop_bounds,array_sha,mask_iou


class QualitativeComparisonTests(unittest.TestCase):
    def test_legacy_edge_tile_order_is_preserved_even_for_one_pixel_remainder(self):
        rows=tile_origins(2305,2305,{'tile_size':1280,'stride':1024})
        self.assertEqual(len(rows),9)
        self.assertEqual(rows[0],[0,0,1280,1280])
        self.assertEqual(rows[2],[1025,0,2305,1280])
        self.assertEqual(rows[-1],[1025,1025,2305,2305])

    def test_crops_use_legacy_floor_ceil_clipping_without_padding(self):
        box=crop_bounds({'x1':-0.2,'y1':15.7,'x2':101.1,'y2':29.01},100,100)
        self.assertEqual(box,[0,15,100,30])

    def test_array_identity_includes_shape_dtype_and_pixel_order(self):
        a=np.arange(18,dtype=np.uint8).reshape(2,3,3)
        self.assertEqual(array_sha(a),array_sha(a.copy()))
        self.assertNotEqual(array_sha(a),array_sha(a[:,:,::-1]))
        self.assertNotEqual(array_sha(a),array_sha(a.reshape(3,2,3)))

    def test_disagreement_iou_handles_missing_masks_without_claiming_accuracy(self):
        empty=np.zeros((8,8),dtype=bool);full=np.ones((8,8),dtype=bool);half=full.copy();half[:4]=False
        self.assertEqual(mask_iou(empty,empty),1.)
        self.assertEqual(mask_iou(empty,full),0.)
        self.assertEqual(mask_iou(half,full),.5)
        self.assertEqual(mask_iou(full,half),.5)


if __name__=='__main__':unittest.main()
