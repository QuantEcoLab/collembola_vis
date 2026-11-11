"""Strict, non-repairing audit of saved original-crop polygons."""

import copy
import unittest

from audit_completed_train_review import polygon_audit


class FinalTrainPolygonAuditTests(unittest.TestCase):
    def test_valid_border_polygon_is_flagged_without_modification(self):
        polygons=[[[0,0],[9,0],[9,9],[0,9]]];before=copy.deepcopy(polygons)
        records,errors=polygon_audit(polygons,10,10)
        self.assertFalse(errors)
        self.assertTrue(records[0]['valid'])
        self.assertEqual(records[0]['edge_sides'],['left','top','right','bottom'])
        self.assertEqual(polygons,before)

    def test_repeated_closing_vertex_is_not_silently_normalized(self):
        polygon=[[[1,1],[8,1],[8,8],[1,8],[1,1]]]
        records,errors=polygon_audit(polygon,10,10)
        self.assertTrue(errors)
        self.assertEqual(records[0]['vertex_count'],5)
        self.assertEqual(records[0]['repeated_vertices'],1)
        self.assertEqual(polygon[0][0],polygon[0][-1])

    def test_self_crossing_and_nonfinite_or_outside_vertices_are_invalid(self):
        for polygon in ([[1,1],[8,8],[1,8],[8,1]],[[1,1],[10,1],[8,8]],[[1,1],[float('nan'),1],[8,8]]):
            records,errors=polygon_audit([polygon],10,10)
            self.assertTrue(errors)
            self.assertFalse(records[0]['valid'])

    def test_all_polygons_are_audited_and_incomplete_polygon_is_not_dropped(self):
        records,errors=polygon_audit([[[1,1],[8,1],[8,8],[1,8]],[[2,2],[3,3]]],10,10)
        self.assertEqual(len(records),2)
        self.assertTrue(records[0]['valid'])
        self.assertFalse(records[1]['valid'])
        self.assertEqual(records[1]['vertex_count'],2)
        self.assertTrue(errors)


if __name__=='__main__':unittest.main()
