"""Sampling arithmetic and exact-version source-logic checks, no runtime install."""

import unittest
from pathlib import Path

from prepare_oversampled_ruler_experiment import exposure, source_audit


class OversamplingTests(unittest.TestCase):
    def test_five_means_five_total_visits_not_six(self):
        control=exposure(1914,20,1)
        new=exposure(1914,20,5)
        self.assertEqual(control['ruler_visits_over_run'],200)
        self.assertEqual(new['ruler_visits_over_run'],1000)
        self.assertEqual(new['visits_per_unique_ruler'],50)
        self.assertEqual(new['base_slots_per_epoch'],1914)
        self.assertEqual(new['ruler_slots_per_epoch'],100)
        self.assertEqual(new['total_slots_per_epoch'],2014)
        self.assertEqual(control['batches_per_epoch'],121)
        self.assertEqual(new['batches_per_epoch'],126)
        self.assertAlmostEqual(new['ruler_share_percent'],4.9652432969)

    def test_expected_batches_are_not_image_count_divided_by_batch_size(self):
        result=exposure(1914,20,1)
        self.assertAlmostEqual(result['expected_batches_with_ruler_per_epoch'],18.5892222393)
        self.assertGreater(result['expected_batches_with_ruler_per_epoch'],20/16)
        self.assertLessEqual(result['expected_batches_with_ruler_per_epoch'],20)

    def test_exact_8_4_175_mosaic_gate_and_setters_without_model_imports(self):
        root=Path('/tmp/opencode/ultralytics_8_4_175_source')
        provenance=Path('/tmp/opencode/ultralytics_8_4_175_source_provenance.json')
        if not provenance.exists():
            self.skipTest('Requires verified exact-version source archive, not installation')
        result=source_audit(root,provenance)
        self.assertEqual(result['version'],'8.4.175')
        self.assertEqual(result['zero_based_close_epoch'],0)
        self.assertEqual(result['mosaic_active_human_epochs'],[])
        self.assertEqual(result['mosaic_effective_probability_all_epochs'],0)
        self.assertIn('>=',result['gate'])
        self.assertFalse(result['installed'])
        self.assertFalse(result['imported'])


if __name__=='__main__':
    unittest.main()
