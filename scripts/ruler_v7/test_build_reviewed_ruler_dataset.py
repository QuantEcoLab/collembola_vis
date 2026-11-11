"""Meaningful checks for lossless crop export, neighbour labels and split safety."""

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from build_reviewed_ruler_dataset import contains, crop_box, digest, export, intersects_polygon, yolo_line


class DatasetTests(unittest.TestCase):
    def test_crop_preserves_all_subpixel_vertices_and_label_roundtrip(self):
        points = [[10.25, 20.75], [29.9, 20.6], [25.2, 48.4], [11.1, 47.9]]
        box = crop_box(points, (100, 100), .16, 4)
        self.assertTrue(contains(box, points))
        values = list(map(float, yolo_line(points, box).split()[1:]))
        restored = [[values[i]*(box[2]-box[0])+box[0], values[i+1]*(box[3]-box[1])+box[1]]
                    for i in range(0, len(values), 2)]
        for old, new in zip(points, restored):
            for a, b in zip(old, new):
                self.assertLess(abs(a-b), 1e-5)
        with self.assertRaises(ValueError):
            yolo_line(points, [12, 22, 26, 45])

    def test_bbox_overlap_without_polygon_overlap_is_not_a_cut(self):
        triangle = [[0, 0], [10, 0], [0, 10]]
        self.assertFalse(intersects_polygon([8, 8, 12, 12], triangle))
        self.assertTrue(intersects_polygon([0, 0, 4, 4], triangle))

    def fixture(self):
        temp = tempfile.TemporaryDirectory(dir='/tmp/opencode')
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        image = root / 'paired.png'
        Image.new('RGB', (30, 30), 'gray').save(image)
        cases = [dict(case_id=name, source_id='C_1_2_Collembola005', split='train', detection_id=str(i),
                      points_source_xy=[[x, 2], [x+5, 2], [x+5, 8], [x, 8]], variants={})
                 for i, (name, x) in enumerate([('a', 2), ('b', 15)])]
        cases[0]['variants']['paired'] = dict(box_source_xyxy=[0, 0, 30, 30], included_case_ids=['a', 'b'],
            cut_known_manual_neighbours=[], image='paired.png', image_sha256=digest(image))
        manifest = dict(cases=cases, original_annotation_hashes={}, source_image_hashes={})
        decisions = dict(a=dict(variant='paired', all_visible_bodies_labeled=True, reason='Two existing manual bodies'),
                         b=dict(variant='covered', covered_by='a', reason='Second full manual instance in same image'))
        (root / 'candidate_manifest.json').write_text(json.dumps(manifest))
        path = root / 'decisions.json'
        path.write_text(json.dumps(decisions))
        return root, path, manifest

    def test_paired_export_keeps_two_labels_and_no_duplicate_image(self):
        root, decisions, _manifest = self.fixture()
        export(root, decisions)
        self.assertEqual(len(list((root / 'yolo_seg/images/train').glob('*.png'))), 1)
        self.assertEqual(len((root / 'yolo_seg/labels/train/a.txt').read_text().splitlines()), 2)
        self.assertEqual((root / 'yolo_seg/images/train/a.png').read_bytes(), (root / 'paired.png').read_bytes())
        self.assertEqual(json.loads((root / 'validation_summary.json').read_text())['exported_unique_manual_references'], 2)

    def test_cut_known_neighbour_blocks_export_before_dataset_creation(self):
        root, decisions, manifest = self.fixture()
        manifest['cases'][0]['variants']['paired']['cut_known_manual_neighbours'] = ['b']
        (root / 'candidate_manifest.json').write_text(json.dumps(manifest))
        with self.assertRaises(ValueError):
            export(root, decisions)
        self.assertFalse((root / 'yolo_seg').exists())

    def test_cross_source_label_cannot_enter_training_image(self):
        root, decisions, manifest = self.fixture()
        manifest['cases'][1]['source_id'] = 'C_3_2_Collembola011'
        manifest['cases'][1]['split'] = 'val'
        (root / 'candidate_manifest.json').write_text(json.dumps(manifest))
        with self.assertRaises(ValueError):
            export(root, decisions)
        self.assertFalse((root / 'yolo_seg').exists())


if __name__ == '__main__':
    unittest.main()
