"""Checks for expanded geometry, target identity, and append-only migrated decisions."""

import copy
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from prepare_controlled_review import content_hash
from prepare_expanded_controlled_review import expanded_box, translate, migration_action, select_target, supplement, sqlite_rows
from test_web_controlled_ruler_review import fixture, POLYGON
from web_controlled_ruler_review import ReviewStore


class ExpandedReviewTests(unittest.TestCase):
    def test_late_snapshot_appends_without_rebasing_existing_history(self):
        with tempfile.TemporaryDirectory(dir='/tmp/opencode',prefix='late-review-test-') as directory:
            root=Path(directory);old_root=root/'v1';old_root.mkdir();old=fixture(old_root);old_store=ReviewStore(old)
            new_root=root/'v2';new_root.mkdir();new=fixture(new_root);ReviewStore(new)
            for case in new.cases:case['crop_frame']={'offset':[0,0]}
            (new_root/'parent_snapshot').mkdir()
            shutil.copy2(old_root/'review.sqlite3',new_root/'parent_snapshot/review.sqlite3')
            (new_root/'frozen_inputs/migration_evidence.json').write_text('{}')
            old_store.decide(0,dict(revision=0,decision=3,notes='late original',polygons=POLYGON))
            backup=root/'late_backup';shutil.copytree(old_root,backup)
            original=sqlite_rows(backup/'review.sqlite3')[0]
            with patch('prepare_expanded_controlled_review.Catalogue',return_value=new), patch('prepare_expanded_controlled_review.verify'):
                supplement(new_root,backup)
                with self.assertRaises(ValueError):supplement(new_root,backup)
            import json
            event=sqlite_rows(new_root/'review.sqlite3')[0];payload=json.loads(event['payload'])
            self.assertEqual(payload['original_parent_events'],[original])
            self.assertEqual(sqlite_rows(new_root/payload['source_snapshot']/'review.sqlite3'),[original])
            self.assertEqual((new_root/payload['source_snapshot']/original['artifact']).read_bytes(),(old_root/original['artifact']).read_bytes())
            store=ReviewStore(new)
            self.assertEqual(store.state(new.case(0))['notes'],'late original')
            self.assertEqual(store.state(new.case(0))['category'],'FALSE_DETECTION')
            store.reopen_last(0,store.last_decision())
            self.assertEqual(store.state(new.case(0))['polygons'],POLYGON)

    def test_padding_and_translation_preserve_original_geometry(self):
        case=dict(source_origin=[100,50],width=135,height=169,source_dimensions=[1000,900])
        self.assertEqual(expanded_box(case),[0,0,455,439])
        polygons=[[[.125,2.5],[20.5,3.25],[9.75,12.875]]]
        shifted=translate(polygons,[100,50])
        self.assertEqual(translate(shifted,[-100,-50]),polygons)
        self.assertEqual(polygons[0][0],[.125,2.5])
        self.assertEqual(shifted[0][0],[100.125,52.5])

    def test_only_evidenced_truncation_reopens_ambiguous_decision(self):
        case=dict(source_id='source',detection_id='185',case_id='source__det0185')
        state=dict(category='AMBIGUOUS_OR_TRUNCATED',notes='')
        self.assertEqual(migration_action(state,case,set())[0],'TRANSFER_DECISION')
        self.assertEqual(migration_action(state,case,{('source','185')})[0],'REOPEN_CROP_TRUNCATION')
        self.assertEqual(migration_action(state,case,set(),{case['case_id']:'User reports cut-off head'})[0],'REOPEN_CROP_TRUNCATION')
        state['notes']='premali crop'
        self.assertEqual(migration_action(state,case,set())[0],'REOPEN_CROP_TRUNCATION')
        for category in ['ACCEPTED_UNCHANGED','MANUALLY_CORRECTED','FALSE_DETECTION']:
            state['category']=category
            self.assertEqual(migration_action(state,case,{('source','185')})[0],'TRANSFER_DECISION')
        state['category']='UNLABELED_NEIGHBOUR'
        self.assertEqual(migration_action(state,case,set())[0],'REOPEN_NEIGHBOUR_CONTEXT')

    def test_target_anchor_beats_higher_confidence_neighbour(self):
        target=np.zeros((12,12),dtype=np.uint8);target[3:9,3:9]=255
        neighbour=np.zeros((12,12),dtype=np.uint8);neighbour[:3,:3]=255
        class Result:
            masks=True
            boxes=[0,1]
            def __getitem__(self,index):return [(neighbour,.999),(target,.6)][index]
        functions={'result_mask':lambda value,shape:value}
        mask,confidence,options=select_target(Result(),target.shape,[4,4,8,8],functions)
        self.assertTrue(np.array_equal(mask,target))
        self.assertEqual(confidence,.6)
        self.assertEqual(options[0]['anchor_mask_iou'],0)
        mask,_,_=select_target(Result(),target.shape,[10,10,12,12],functions)
        self.assertIsNone(mask)

    def test_migration_preserves_history_and_undo_uses_shifted_contour(self):
        with tempfile.TemporaryDirectory(dir='/tmp/opencode',prefix='expanded-review-test-') as directory:
            catalogue=fixture(Path(directory));store=ReviewStore(catalogue);case=catalogue.case(0)
            state=store.decide(0,dict(revision=0,decision=1,fast_review=True,polygons=POLYGON,notes='original note'))
            with store.connection() as con:
                original_rows=[tuple(r) for r in con.execute('SELECT * FROM events ORDER BY id')]
            artifact=next((catalogue.root/'annotation_revisions').glob('*.json'));original_bytes=artifact.read_bytes()
            migrated=copy.deepcopy(state);migrated.pop('revision');migrated.pop('training_candidate')
            migrated['polygons']=translate(state['polygons'],[1,2])
            migrated['expanded_context_review_required']=True
            migrated['crop_migration']={'offset':[1,2],'action':'TRANSFER_DECISION'}
            label={key:migrated[key] for key in ['category','polygons','notes','checks','reference_id','label_source']}
            digest=content_hash(label);migrated['final_label_hash']=digest
            store.append(case,'MIGRATE_CROP',dict(state=migrated,label=label,label_hash=digest),state['revision'])
            current=store.state(case)
            self.assertEqual(current['category'],'ACCEPTED_UNCHANGED')
            self.assertEqual(current['polygons'],translate(POLYGON,[1,2]))
            self.assertFalse(current['training_candidate'])
            self.assertEqual(current['notes'],'original note')
            last=store.last_decision()
            reopened=store.reopen_last(0,last)
            self.assertEqual(reopened['category'],'PENDING')
            self.assertEqual(reopened['polygons'],translate(POLYGON,[1,2]))
            self.assertEqual(artifact.read_bytes(),original_bytes)
            with store.connection() as con:
                self.assertEqual([tuple(r) for r in con.execute('SELECT * FROM events ORDER BY id')][:len(original_rows)],original_rows)
            store.verify_history()


if __name__=='__main__':unittest.main()
