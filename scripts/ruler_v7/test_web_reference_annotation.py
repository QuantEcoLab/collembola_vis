"""Test independent annotation geometry and persistence without real labels."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from web_reference_annotation import AnnotationPack, Handler, validate_polygon


class AnnotationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir='/tmp/opencode')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for folder in ['images', 'annotations', 'annotation_history']:
            (self.root / folder).mkdir()
        Image.new('RGB', (100, 100), 'gray').save(self.root / 'images/clean.png')
        rows = [dict(case_id=f'case_{i}', source_id='source', detection_id=str(i),
                     annotation_size=[100, 100], annotation_box_source_xyxy=[200, 300, 300, 400],
                     files={'image': 'images/clean.png'}) for i in range(12)]
        (self.root / 'manifest.json').write_text(json.dumps(rows))
        self.pack = AnnotationPack(self.root)

    def test_empty_start_does_not_create_labels(self):
        state = self.pack.state()
        self.assertEqual(state['completed'], 0)
        self.assertEqual(state['next_pending'], 0)
        self.assertEqual(list((self.root / 'annotations').iterdir()), [])

    def test_manual_points_saved_exactly_and_resumed(self):
        points = [[10.5, 10], [80, 10], [80, 80], [10, 80]]
        original = (self.root / 'images/clean.png').read_bytes()
        result = self.pack.save(0, points, 'manual note')
        self.assertEqual(result['shapes'][0]['points'], points)
        self.assertFalse(result['flags']['model_mask_imported'])
        self.assertEqual(result['reference_metadata']['annotation_box_source_xyxy'], [200, 300, 300, 400])
        self.assertEqual((self.root / 'images/clean.png').read_bytes(), original)
        resumed = AnnotationPack(self.root).state()
        self.assertEqual(resumed['completed'], 1)
        self.assertEqual(resumed['next_pending'], 1)
        image_path = self.root / 'annotations' / result['imagePath']
        self.assertEqual(image_path.resolve(), (self.root / 'images/clean.png').resolve())

    def test_revision_history_is_not_overwritten(self):
        points = [[10, 10], [80, 10], [80, 80], [10, 80]]
        first = self.pack.save(0, points)
        history = list((self.root / 'annotation_history').iterdir())
        first_bytes = history[0].read_bytes()
        second = self.pack.save(0, [[11, 11], [79, 11], [79, 79], [11, 79]])
        self.assertEqual(len(list((self.root / 'annotation_history').iterdir())), 2)
        self.assertEqual(history[0].read_bytes(), first_bytes)
        self.assertNotEqual(first['shapes'][0]['points'], second['shapes'][0]['points'])

    def test_invalid_geometry_is_rejected_without_labels(self):
        bad = [[], [[1, 1], [2, 2]], [[1, 1], [100, 1], [1, 40]],
               [[1, 1], [2, float('nan')], [3, 3]],
               [[10, 10], [80, 80], [80, 10], [10, 80]],
               [[10, 10], [80, 10], [80, 80], [80, 10], [10, 80]]]
        for points in bad:
            with self.assertRaises(ValueError):
                self.pack.save(0, points)
        self.assertEqual(list((self.root / 'annotations').iterdir()), [])

    def test_area_and_no_smoothing(self):
        self.assertEqual(validate_polygon([[10, 10], [80, 10], [80, 80], [10, 80]], 100, 100), 4900)

    def test_repeated_closing_and_consecutive_vertices_save_without_reshaping(self):
        unique = [[10, 10], [80, 10], [80, 80], [10, 80]]
        duplicate = [[10, 10], [10, 10], [80, 10], [80, 80], [80, 80], [10, 80], [10, 10]]
        before = json.loads(json.dumps(duplicate))
        result = self.pack.save(0, duplicate)
        self.assertEqual(result['shapes'][0]['points'], unique)
        self.assertEqual(result['reference_metadata']['area_px'], 4900)
        self.assertEqual(result['reference_metadata']['redundant_vertices_removed'], 3)
        self.assertEqual(duplicate, before)
        self.assertEqual(self.pack.state()['completed'], 1)

    def test_variable_pack_size_and_last_case_route_without_server(self):
        path = self.root / 'manifest.json'
        rows = json.loads(path.read_text())
        rows.extend({**rows[0], 'case_id': f'case_{i}', 'detection_id': str(i)} for i in range(12, 22))
        path.write_text(json.dumps(rows))
        (self.root / 'pack_info.json').write_text(json.dumps({
            'case_count': 22, 'annotation_mode': 'ORIGINAL_RAW_CROP', 'autosave_on_close': True,
            'source_allowlist': ['source']}))
        pack = AnnotationPack(self.root)
        self.assertEqual(len(pack.state()['cases']), 22)
        self.assertTrue(pack.state()['options']['autosave_on_close'])
        pack.save(21, [[10, 10], [80, 10], [80, 80], [10, 80]])
        restored = AnnotationPack(self.root).state()
        self.assertEqual(restored['completed'], 1)
        self.assertIsNotNone(restored['annotations'][21])
        handler = Handler.__new__(Handler)
        handler.server = SimpleNamespace(pack=pack)
        handler.path = '/asset?index=21&kind=image'
        responses = []
        handler.body = lambda content, *args: responses.append(content)
        handler.json = lambda value, *args: self.fail(str(value))
        handler.do_GET()
        self.assertEqual(responses[0], (self.root / 'images/clean.png').read_bytes())

    def test_legacy_pack_keeps_explicit_save_and_source_allowlist_is_enforced(self):
        self.assertFalse(self.pack.state()['options']['autosave_on_close'])
        (self.root / 'pack_info.json').write_text(json.dumps({'source_allowlist': ['other_source']}))
        with self.assertRaises(ValueError):
            AnnotationPack(self.root)


if __name__ == '__main__':
    unittest.main()
