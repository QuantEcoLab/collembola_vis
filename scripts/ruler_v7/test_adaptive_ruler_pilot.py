"""Meaningful selection regressions for the isolated v3 pilot."""

import unittest
import numpy as np

from pilot_adaptive_ruler_review import crop_geometry, select_instance


def instance(box,confidence=.8,components=1,edge=False):
    mask=np.zeros((60,60),dtype=bool)
    x1,y1,x2,y2=box;mask[y1:y2,x1:x2]=True
    return dict(mask=mask,confidence=confidence,components=components,touches_crop_edge=edge)


class AdaptivePilotTests(unittest.TestCase):
    def test_padding_is_moderate_adaptive_and_clipped_only_to_photo_edge(self):
        case=dict(width=135,height=169,source_origin=[1000,1000],source_dimensions=[10000,10000])
        boxes=[crop_geometry(case,v) for v in ['p10','p20','p30']]
        self.assertEqual([b[2] for b in boxes],[17,34,51])
        self.assertEqual(boxes[1][1],[34,34,169,203])
        case['source_origin']=[0,5]
        self.assertEqual(crop_geometry(case,'p20')[0],[0,0,169,208])
        case.update(width=1000,height=2000)
        self.assertEqual(crop_geometry(case,'p30')[2],64)

    def test_target_is_not_largest_or_highest_confidence_neighbour(self):
        masks=[instance([1,1,50,50],.999),instance([22,22,38,38],.6)]
        originals=[m['mask'].copy() for m in masks]
        index,records,_=select_instance(masks,[20,20,40,40])
        self.assertEqual(index,1)
        self.assertIn('EXCESS_MASK_OUTSIDE_DETECTION',records[0]['reasons'])
        for item,original in zip(masks,originals):self.assertTrue(np.array_equal(item['mask'],original))

    def test_overlap_without_central_association_is_rejected(self):
        selected,records,_=select_instance([instance([15,20,33,40])],[30,20,50,40])
        self.assertIsNone(selected)
        self.assertIn('CENTROID_OUTSIDE_DETECTION',records[0]['reasons'])

    def test_one_merged_or_disconnected_instance_is_not_silently_fixed(self):
        merged=instance([20,20,50,40])
        merged['mask'][40:50,40:50]=True
        index,records,_=select_instance([merged],[20,20,40,40])
        self.assertIsNone(index)
        self.assertIn('EXCESS_MASK_OUTSIDE_DETECTION',records[0]['reasons'])
        disconnected=instance([22,22,38,38],components=2)
        index,records,_=select_instance([disconnected],[20,20,40,40])
        self.assertIsNone(index)
        self.assertIn('MULTIPLE_COMPONENTS_IN_SINGLE_INSTANCE',records[0]['reasons'])

    def test_equally_plausible_instances_have_no_automatic_target(self):
        masks=[instance([21,21,37,37]),instance([22,22,38,38])]
        selected,_,reason=select_instance(masks,[20,20,40,40])
        self.assertIsNone(selected)
        self.assertEqual(reason,'AMBIGUOUS_COMPETING_TARGET_INSTANCES')

    def test_edge_contact_and_empty_result_keep_pending_without_contour(self):
        selected,records,_=select_instance([instance([22,22,38,38],edge=True)],[20,20,40,40])
        self.assertIsNone(selected)
        self.assertIn('MASK_TOUCHES_CROP_EDGE',records[0]['reasons'])
        selected,records,reason=select_instance([],[20,20,40,40])
        self.assertIsNone(selected)
        self.assertEqual(records,[])
        self.assertEqual(reason,'NO_RELIABLY_ASSOCIATED_INSTANCE')


if __name__=='__main__':unittest.main()
