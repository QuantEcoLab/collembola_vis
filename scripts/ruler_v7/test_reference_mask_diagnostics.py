"""Check coordinate alignment and framing/segmentation separation."""
import unittest
import numpy as np

from evaluate_reference_polygons import compare_mask


class DiagnosticsTests(unittest.TestCase):
    def test_translated_original_size_mask_is_perfect_without_resizing(self):
        gt = np.zeros((20, 30), dtype=bool)
        gt[8:12, 13:17] = True
        raw = np.zeros((10, 10), dtype=bool)
        raw[3:7, 3:7] = True
        result = compare_mask(gt, raw, (10, 5, 20, 15))
        self.assertEqual(result['iou'], 1)
        self.assertEqual(result['recall'], 1)
        self.assertEqual(result['reference_outside_raw_fraction'], 0)
        with self.assertRaises(ValueError):
            compare_mask(gt, raw, (10, 5, 21, 15))

    def test_crop_loss_is_not_mistaken_for_inside_crop_segmentation_error(self):
        gt = np.zeros((20, 30), dtype=bool)
        gt[8:12, 13:21] = True
        raw = np.zeros((10, 7), dtype=bool)
        raw[3:7, 3:7] = True
        result = compare_mask(gt, raw, (10, 5, 17, 15))
        self.assertEqual(result['reference_outside_raw_fraction'], .5)
        self.assertEqual(result['within_raw_crop_iou'], 1)
        self.assertEqual(result['iou'], .5)
        self.assertEqual(result['recall'], .5)


if __name__ == '__main__':
    unittest.main()
