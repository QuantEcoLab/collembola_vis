"""Isolated synthetic checks for the controlled-review trust boundary."""

import copy
import json
import sqlite3
import tempfile
import threading
import unittest
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from PIL import Image

from prepare_controlled_review import FIXED_SHA, ROLES, content_hash, sha
from web_controlled_ruler_review import App, Catalogue, ReviewStore, Server

POLYGON = [[[2, 2], [15, 2], [15, 15], [2, 15]]]


def write_json(path, value):
    path.write_text(json.dumps(value))


def fixture(root):
    (root / 'frozen_inputs').mkdir()
    Image.new('RGB', (20, 20), 'gray').save(root / 'original.png')
    Image.new('L', (20, 20), 255).save(root / 'mask.png')
    write_json(root / 'human.json', {'manual': True})
    cases, predictions = [], []
    for i, (source, role) in enumerate([(s, r) for s, r in ROLES.items() if r != 'test'] +
                                         [('C_4_1_PS_500-1000_014', 'test')]):
        case = dict(index=i, case_id=str(i), source_id=source, role=role, detection_id=i,
                    width=20, height=20, source_origin=[0, 0], source_dimensions=[20, 20],
                    original=str(root / 'original.png'), original_sha256=sha(root / 'original.png'),
                    initial_mask=str(root / 'mask.png'), initial_mask_sha256=sha(root / 'mask.png'),
                    source_image=str(root / 'original.png'), source_image_sha256=sha(root / 'original.png'),
                    legacy_position='LEGACY_SECRET', legacy_crop_condition='SECRET', legacy_blocked_warning='SECRET',
                    human_reference=dict(id='human', path=str(root / 'human.json'), sha256=sha(root / 'human.json'),
                        original_annotation_size=[20, 20], same_frame=True, usable_losslessly=True,
                        mapping_note='Same original frame', polygons=POLYGON))
        cases.append(case)
        predictions.append(dict(case_id=str(i), checkpoint_sha256=FIXED_SHA,
            crop_sha256=case['original_sha256'], mask='mask.png', mask_sha256=sha(root / 'mask.png'),
            proposal=POLYGON, proposal_sha256=content_hash(POLYGON), confidence=.987654,
            has_mask=True, connected_components=1, holes=0, touches_crop_edge=False))
    write_json(root / 'cases.json', cases)
    write_json(root / 'config.json', dict(count=len(cases), candidate_sha256=FIXED_SHA, inventory_sha256='synthetic'))
    write_json(root / 'frozen_inputs/external_input_hashes.json', {})
    write_json(root / 'frozen_inputs/bundle_seal.json', {'files': {p:sha(root/p) for p in ['cases.json', 'config.json']}})
    write_json(root / 'prediction_manifest.json', predictions)
    write_json(root / 'prediction_seal.json', dict(manifest_sha256=sha(root/'prediction_manifest.json'),
        files={'mask.png':sha(root/'mask.png')}))
    return Catalogue(root)


class ControlledReviewTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='controlled-review-', dir='/tmp/opencode')
        self.root = Path(self.tmp.name)
        self.catalogue = fixture(self.root)
        self.app = App(self.catalogue)
        self.store = self.app.store
        self.val = next(c['index'] for c in self.catalogue.cases if c['role'] == 'val')
        self.test = next(c['index'] for c in self.catalogue.cases if c['role'] == 'test')

    def tearDown(self):
        self.tmp.cleanup()

    def data(self, index=0, **kwargs):
        case = self.catalogue.case(index)
        return dict(index=index, revision=self.store.state(case)['revision'],
                    polygons=copy.deepcopy(POLYGON), notes='human inspection',
                    checks=dict(whole_body=True, no_unlabeled_neighbour=True), **kwargs)

    def test_resume_conflict_and_immutable_history(self):
        old = self.data()
        saved = self.store.draft(0, old)
        with self.assertRaises(RuntimeError):
            self.store.draft(0, old)
        restored = ReviewStore(self.catalogue).state(self.catalogue.case(0))
        self.assertEqual(restored, saved)
        self.assertTrue(restored['has_draft'])
        with self.store.connection() as con:
            with self.assertRaises(sqlite3.IntegrityError):
                con.execute('DELETE FROM events')
            with self.assertRaises(sqlite3.IntegrityError):
                con.execute("UPDATE events SET kind='DECISION'")
        artifact = next((self.root/'annotation_revisions').glob('*.json'))
        artifact.write_text('{}')
        with self.assertRaisesRegex(ValueError, 'artifact changed'):
            ReviewStore(self.catalogue)

    def test_train_positive_checks_and_unchanged(self):
        data = self.data(decision=1)
        data['checks']['whole_body'] = False
        with self.assertRaises(ValueError):
            self.store.decide(0, data)
        with self.assertRaisesRegex(ValueError, 'unchanged'):
            self.store.decide(0, self.data(decision=2))
        state = self.store.decide(0, self.data(decision=1))
        self.assertTrue(state['training_candidate'])
        state = self.store.draft(0, self.data())
        self.assertFalse(state['training_candidate'])
        for number in (3, 4, 5):
            state = self.store.decide(0, self.data(decision=number))
            self.assertFalse(state['training_candidate'])

    def test_blind_lock_unlock_and_exposure_provenance(self):
        for i in (self.val, self.test):
            before = self.app.detail(i)
            self.assertNotIn('predictions', before)
            self.assertNotIn('legacy_warnings', before)
            self.assertNotIn('metrics', before)
            self.assertNotIn('LEGACY_SECRET', json.dumps(before))
            with self.assertRaises(PermissionError):
                self.store.decide(i, self.data(i, decision=1))
            state = self.store.decide(i, self.data(i, decision=2))
            first = state['first_blind_label_hash']
            self.assertTrue(state['locked'])
            self.assertFalse(state['training_candidate'])
            self.assertIn('metrics', self.app.detail(i))
            with self.assertRaises(PermissionError):
                self.store.draft(i, self.data(i))
            with self.assertRaises(ValueError):
                self.store.unlock(i, self.data(i, reason=''))
            self.store.unlock(i, self.data(i, reason='Correct a misplaced vertex'))
            self.assertNotIn('predictions', self.app.detail(i))
            state = self.store.decide(i, self.data(i, decision=2))
            self.assertEqual(first, state['first_blind_label_hash'])
            with self.store.connection() as con:
                row = con.execute("SELECT payload FROM events WHERE kind='DECISION' AND case_id=? ORDER BY id DESC", (str(i),)).fetchone()
                self.assertTrue(json.loads(row[0])['revision_after_model_exposure'])

    def test_geometry_and_reference_validation(self):
        for polygon in ([[1,1],[15,15],[1,15],[15,1]], [[-1,2],[15,2],[15,15]], [[1,1],[1,1],[2,2]]):
            data = self.data(self.val, decision=2)
            data['polygons'] = [polygon]
            with self.assertRaises(ValueError):
                self.store.decide(self.val, data)
        case = self.catalogue.case(self.val)
        case['human_reference']['usable_losslessly'] = False
        with self.assertRaisesRegex(ValueError, 'losslessly'):
            self.store.draft(self.val, self.data(self.val, reference_id='human'))

    def test_http_prediction_assets_denied_before_lock_and_after_unlock(self):
        server = Server(('127.0.0.1', 0), self.app)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f'http://127.0.0.1:{server.server_port}'
        def get(path):
            try:
                with urlopen(base+path) as r:return r.status, r.read()
            except HTTPError as e:return e.code, e.read()
        try:
            for i in (self.val, self.test):
                for kind in ('candidate', 'initial'):
                    self.assertEqual(get(f'/asset?index={i}&kind={kind}')[0], 403)
                self.assertEqual(get(f'/asset?index={i}&kind=original')[0], 200)
                self.assertNotIn(b'confidence', get(f'/api/case?index={i}')[1])
                data = self.data(i, decision=3)
                req = Request(base+'/api/decision', json.dumps(data).encode(), {'Content-Type':'application/json'})
                with urlopen(req) as r:self.assertTrue(json.loads(r.read())['state']['locked'])
                self.assertEqual(get(f'/asset?index={i}&kind=candidate')[0], 200)
                self.store.unlock(i, self.data(i, reason='Recheck negative category'))
                self.assertEqual(get(f'/asset?index={i}&kind=candidate')[0], 403)
            self.assertEqual(get('/')[0], 200)
            self.assertEqual(get('/asset?index=0&kind=candidate')[0], 200)
        finally:
            server.shutdown();server.server_close();thread.join()

    def test_frozen_prediction_tampering_rejected(self):
        (self.root/'mask.png').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'prediction changed'):
            Catalogue(self.root)

    def test_fast_review_explicitly_persists_positive_confirmations(self):
        for number in (1, 2):
            data = self.data(decision=number, fast_review=True)
            data['checks'] = {'whole_body': False, 'no_unlabeled_neighbour': False}
            if number == 2:data['polygons'][0][0] = [3, 3]
            state = self.store.decide(0, data)
            self.assertEqual(state['checks'], {'whole_body': True, 'no_unlabeled_neighbour': True})
            self.assertTrue(state['training_candidate'])
            with self.store.connection() as con:
                payload = json.loads(con.execute('SELECT payload FROM events ORDER BY id DESC LIMIT 1').fetchone()[0])
            self.assertEqual(payload['confirmation_source'], 'FAST_REVIEW_EXPLICIT_DECISION')
            self.assertTrue(payload['fast_review'])
        with self.assertRaises(PermissionError):
            self.store.decide(self.val, self.data(self.val, decision=1, fast_review=True))

    def test_reopen_last_is_append_only_and_restores_draft(self):
        self.store.decide(0, self.data(decision=1))
        negative = self.data(1, decision=4)
        negative['notes'] = 'Keep this exact note'
        self.store.decide(1, negative)
        with self.store.connection() as con:
            before = [tuple(r) for r in con.execute('SELECT * FROM events ORDER BY id')]
        artifacts = {p: p.read_bytes() for p in (self.root/'annotation_revisions').glob('*.json')}
        last = self.store.last_decision()
        self.assertEqual(last['index'], 1)
        state = self.store.reopen_last(1, last)
        self.assertEqual(state['category'], 'PENDING')
        self.assertEqual(state['notes'], negative['notes'])
        self.assertEqual(state['polygons'], negative['polygons'])
        self.assertFalse(state['training_candidate'])
        self.assertEqual(self.store.last_decision()['index'], 0)
        with self.store.connection() as con:
            after = [tuple(r) for r in con.execute('SELECT * FROM events ORDER BY id')]
        self.assertEqual(after[:-1], before)
        for p, data in artifacts.items():self.assertEqual(p.read_bytes(), data)
        self.assertEqual(ReviewStore(self.catalogue).state(self.catalogue.case(1)), state)
        with self.assertRaises(RuntimeError):self.store.reopen_last(1, last)

    def test_reopen_rejects_stale_last_decision_even_with_current_case_revision(self):
        self.store.decide(0, self.data(decision=1))
        old = self.store.last_decision()
        self.store.decide(1, self.data(1, decision=3))
        with self.assertRaisesRegex(RuntimeError, 'Last saved decision changed'):
            self.store.reopen_last(0, old)
        self.assertEqual(self.store.state(self.catalogue.case(0))['category'], 'ACCEPTED_UNCHANGED')

    def test_reopen_blind_label_preserves_first_hash_and_denies_predictions(self):
        state = self.store.decide(self.val, self.data(self.val, decision=2, fast_review=True))
        first_hash = state['first_blind_label_hash']
        state = self.store.reopen_last(self.val, self.store.last_decision())
        self.assertTrue(state['model_exposed'])
        self.assertEqual(state['first_blind_label_hash'], first_hash)
        self.assertFalse(state['locked'])
        self.assertNotIn('predictions', self.app.detail(self.val))
        self.assertNotIn('metrics', self.app.detail(self.val))
        self.store.decide(self.val, self.data(self.val, decision=2, fast_review=True))
        with self.store.connection() as con:
            payload = json.loads(con.execute('SELECT payload FROM events ORDER BY id DESC LIMIT 1').fetchone()[0])
        self.assertTrue(payload['revision_after_model_exposure'])


if __name__ == '__main__':
    unittest.main()
