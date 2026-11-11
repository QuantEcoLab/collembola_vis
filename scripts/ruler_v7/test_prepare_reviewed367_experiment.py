"""Preparation checks for exact approved-label conversion and leakage guards."""

import copy
import tempfile
import unittest
from pathlib import Path

from prepare_reviewed367_experiment import encode_polygon, inventory_checks, prepare


class Reviewed367PreparationTests(unittest.TestCase):
    def test_yolo_roundtrip_preserves_vertices_including_image_border(self):
        points=[[0,0],[136.1234567890123,0],[136.1234567890123,90.9876543210987],[0,90.9876543210987]]
        before=copy.deepcopy(points);text,error=encode_polygon(points,137,92)
        values=list(map(float,text.split()[1:]))
        self.assertEqual(text.split()[0],'0')
        self.assertEqual(len(values),2*len(points))
        self.assertTrue(all(0<=value<=1 for value in values))
        self.assertLess(error,1e-9)
        self.assertEqual(points,before)

    def test_invalid_polygons_are_rejected_not_repaired_or_clipped(self):
        for points in ([[0,0],[9,0],[9,9],[0,0]],[[0,0],[10,0],[9,9]],[[1,1],[8,8],[1,8],[8,1]]):
            with self.assertRaises(ValueError):encode_polygon(points,10,10)

    def row(self,source,split,image_hash,pixel_hash,name):
        return dict(source_id=source,split=split,image_sha256=image_hash,pixel_sha256=pixel_hash,
                    image_name=name+'.png',label_name=name+'.txt',polygons=1)

    def test_different_encodings_of_identical_pixels_are_still_duplicates(self):
        rows=[self.row('train_source','train','png_hash','same_pixels','a'),
              self.row('other_train','train','jpeg_hash','same_pixels','b')]
        with self.assertRaisesRegex(ValueError,'Duplicate images/pixels'):inventory_checks(rows)

    def test_source_leakage_is_rejected_even_for_distinct_images(self):
        rows=[self.row('same_source','train','image_a','pixels_a','a'),
              self.row('same_source','base_val','image_b','pixels_b','b')]
        with self.assertRaisesRegex(ValueError,'Source leakage'):inventory_checks(rows)

    def test_existing_experiment_is_never_overwritten(self):
        with tempfile.TemporaryDirectory(dir='/tmp/opencode') as directory:
            path=Path(directory);sentinel=path/'existing.txt';sentinel.write_text('preserve')
            with self.assertRaisesRegex(ValueError,'Refusing to overwrite'):prepare(path/'unused_audit',path)
            self.assertEqual(sentinel.read_text(),'preserve')


if __name__=='__main__':unittest.main()
