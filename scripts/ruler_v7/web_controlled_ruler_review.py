#!/usr/bin/env python3
"""Human-controlled train review and server-enforced blind val/test annotation."""

import argparse
import copy
import io
import json
import os
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
from PIL import Image, ImageDraw

from prepare_controlled_review import ROOT, FIXED_SHA, ROLES, sha, canonical, content_hash
from web_reference_annotation import normalize_polygon_vertices, validate_polygon

CATEGORIES={1:'ACCEPTED_UNCHANGED',2:'MANUALLY_CORRECTED',3:'FALSE_DETECTION',
            4:'AMBIGUOUS_OR_TRUNCATED',5:'UNLABELED_NEIGHBOUR'}
POSITIVE={'ACCEPTED_UNCHANGED','MANUALLY_CORRECTED'}


def polygon_mask(polygons,width,height):
    image=Image.new('1',(width,height))
    draw=ImageDraw.Draw(image)
    for polygon in polygons:draw.polygon([tuple(p) for p in polygon],fill=1)
    return np.asarray(image,dtype=bool)


class Catalogue:
    def __init__(self,root):
        self.root=root.resolve()
        self.config=json.loads((root/'config.json').read_text())
        self.cases=json.loads((root/'cases.json').read_text())
        seal=json.loads((root/'frozen_inputs/bundle_seal.json').read_text())
        for name,digest in seal['files'].items():
            if sha(root/name)!=digest:raise ValueError('Frozen inventory/case configuration changed')
        for name,digest in json.loads((root/'frozen_inputs/external_input_hashes.json').read_text()).items():
            if sha(Path(name))!=digest:raise ValueError(f'Original input changed: {name}')
        prediction_seal=json.loads((root/'prediction_seal.json').read_text())
        if sha(root/'prediction_manifest.json')!=prediction_seal['manifest_sha256']:
            raise ValueError('Prediction manifest changed')
        for name,digest in prediction_seal['files'].items():
            if sha(root/name)!=digest:raise ValueError('Frozen prediction changed')
        self.predictions=json.loads((root/'prediction_manifest.json').read_text())
        self.by_id={p['case_id']:p for p in self.predictions}
        if len(self.cases)!=self.config['count'] or len(self.by_id)!=len(self.cases):
            raise ValueError('Incomplete inference/catalogue')
        for case in self.cases:
            if ROLES[case['source_id']]!=case['role']:raise ValueError('Unexpected source role')
            pred=self.by_id[case['case_id']]
            if pred['checkpoint_sha256']!=self.config['candidate_sha256'] or pred['crop_sha256']!=case['original_sha256']:
                raise ValueError('Prediction provenance mismatch')
        self.fingerprint=content_hash({'inventory':self.config['inventory_sha256'],
            'cases':sha(root/'cases.json'),'predictions':prediction_seal['manifest_sha256'],
            'checkpoint':self.config['candidate_sha256']})

    def case(self,index):
        if type(index) is not int or not 0<=index<len(self.cases):raise ValueError('Invalid case index')
        return self.cases[index]

    def prediction(self,case):return self.by_id[case['case_id']]

    def initial_mask(self,case):
        if not case['initial_mask']:return np.zeros((case['height'],case['width']),dtype=bool)
        with Image.open(case['initial_mask']) as image:return np.asarray(image.convert('L'))>0

    def candidate_mask(self,case):
        with Image.open(self.root/self.prediction(case)['mask']) as image:return np.asarray(image.convert('L'))>0

    def context(self,case):
        x,y=case['source_origin'];w,h=case['width'],case['height'];iw,ih=case['source_dimensions']
        cw,ch=min(iw,max(1000,4*w)),min(ih,max(850,4*h))
        left=max(0,min(int(x+w/2-cw/2),iw-cw));top=max(0,min(int(y+h/2-ch/2),ih-ch))
        return {'box':[left,top,left+cw,top+ch], 'crop_box':[x-left,y-top,x-left+w,y-top+h]}

    def verify_case(self,case):
        if sha(Path(case['original']))!=case['original_sha256']:raise ValueError('Original crop changed')
        if case['initial_mask'] and sha(Path(case['initial_mask']))!=case['initial_mask_sha256']:
            raise ValueError('Initial prediction changed')
        pred=self.prediction(case)
        if sha(self.root/pred['mask'])!=pred['mask_sha256']:raise ValueError('Candidate prediction changed')


class ReviewStore:
    def __init__(self,catalogue):
        self.catalogue=catalogue;self.root=catalogue.root
        self.lock=threading.RLock();self.database=self.root/'review.sqlite3'
        (self.root/'annotation_revisions').mkdir(exist_ok=True)
        with self.connection() as con:
            con.execute('CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY,value TEXT NOT NULL)')
            con.execute('CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY,case_id TEXT NOT NULL,kind TEXT NOT NULL,'
                        'created_at TEXT NOT NULL,payload TEXT NOT NULL,previous_hash TEXT NOT NULL,event_hash TEXT UNIQUE NOT NULL,'
                        'artifact TEXT NOT NULL,artifact_sha256 TEXT NOT NULL)')
            con.execute("CREATE TRIGGER IF NOT EXISTS no_update BEFORE UPDATE ON events BEGIN SELECT RAISE(ABORT,'append-only events'); END")
            con.execute("CREATE TRIGGER IF NOT EXISTS no_delete BEFORE DELETE ON events BEGIN SELECT RAISE(ABORT,'append-only events'); END")
            con.execute("INSERT OR IGNORE INTO metadata VALUES ('input_fingerprint',?)",(catalogue.fingerprint,))
            if con.execute("SELECT value FROM metadata WHERE key='input_fingerprint'").fetchone()[0]!=catalogue.fingerprint:
                raise ValueError('Review belongs to different frozen inputs')
        self.verify_history()

    @contextmanager
    def connection(self):
        con=sqlite3.connect(self.database,timeout=15);con.row_factory=sqlite3.Row
        try:
            with con:yield con
        finally:con.close()

    def verify_history(self):
        previous=''
        with self.connection() as con:
            for row in con.execute('SELECT * FROM events ORDER BY id'):
                value={'case_id':row['case_id'],'kind':row['kind'],'created_at':row['created_at'],
                       'payload':json.loads(row['payload']),'previous_hash':previous}
                if row['previous_hash']!=previous or content_hash(value)!=row['event_hash']:
                    raise ValueError('History hash chain is invalid')
                if sha(self.root/row['artifact'])!=row['artifact_sha256']:
                    raise ValueError('History artifact changed')
                previous=row['event_hash']

    def state(self,case,con=None):
        if con is None:
            with self.connection() as conn:return self.state(case,conn)
        state={'category':'PENDING','polygons':[],'notes':'','checks':{},'locked':False,
               'revision':0,'has_draft':False,'model_exposed':False,'first_blind_label_hash':None,
                'final_label_hash':None,'reference_id':None,'label_source':None,'export_geometry_valid':False}
        for row in con.execute('SELECT * FROM events WHERE case_id=? ORDER BY id',(case['case_id'],)):
            payload=json.loads(row['payload']);state['revision']=row['id']
            if row['kind']=='DRAFT':
                state.update(category='PENDING',polygons=payload['polygons'],notes=payload['notes'],
                             checks=payload['checks'],reference_id=payload.get('reference_id'),has_draft=True,
                             final_label_hash=None,export_geometry_valid=False)
            elif row['kind']=='DECISION':
                label=payload['label']
                state.update(category=label['category'],polygons=label['polygons'],notes=label['notes'],checks=label['checks'],
                    reference_id=label['reference_id'],label_source=label['label_source'],has_draft=False,
                    locked=case['role']!='train',final_label_hash=payload['label_hash'],
                    export_geometry_valid=label['export_geometry_valid'])
                state['expanded_context_review_required']=False
                if case['role']!='train':
                    state['model_exposed']=True
                    if state['first_blind_label_hash'] is None:state['first_blind_label_hash']=payload['label_hash']
            elif row['kind']=='UNLOCK':
                state.update(locked=False,category='PENDING',has_draft=True,final_label_hash=None,
                              export_geometry_valid=False,model_exposed=True)
            elif row['kind']=='REOPEN':
                state.update(payload['draft'])
                state.update(locked=False,category='PENDING',has_draft=True,final_label_hash=None,
                             export_geometry_valid=False)
                if case['role']!='train':state['model_exposed']=True
            elif row['kind']=='MIGRATE_CROP':
                state.update(copy.deepcopy(payload['state']))
                state['revision']=row['id']
        state['training_candidate']=case['role']=='train' and case['source_id'] in {'C_1_2_Collembola005','C_1_3_Collembola006'} and state['category'] in POSITIVE and not state['has_draft'] and state['export_geometry_valid'] and not state.get('expanded_context_review_required',False) and all(state['checks'].get(k) is True for k in ['whole_body','no_unlabeled_neighbour'])
        return state

    def append(self,case,kind,payload,expected_revision):
        with self.lock,self.connection() as con:
            con.execute('BEGIN IMMEDIATE')
            state=self.state(case,con)
            if expected_revision!=state['revision']:raise RuntimeError('Revision conflict; your draft is preserved in the browser. Reload case state before retrying.')
            if kind in {'DRAFT','DECISION'} and case['role']!='train' and state['locked']:
                raise PermissionError('Blind GT is locked. Explicit unlock with a reason is required.')
            if kind=='UNLOCK' and (case['role']=='train' or not state['locked']):raise ValueError('Only locked val/test GT can be unlocked')
            if kind=='REOPEN':
                latest=self.last_decision(con)
                if not latest or latest['case_id']!=case['case_id'] or latest['decision_revision']!=payload['decision_revision']:
                    raise RuntimeError('Last saved decision changed; reload the review before undoing it')
            previous=con.execute('SELECT event_hash FROM events ORDER BY id DESC LIMIT 1').fetchone()
            value={'case_id':case['case_id'],'kind':kind,'created_at':datetime.now(timezone.utc).isoformat(timespec='microseconds'),
                   'payload':payload,'previous_hash':previous[0] if previous else ''}
            digest=content_hash(value);path='annotation_revisions/'+digest+'.json'
            with (self.root/path).open('x',encoding='utf-8') as handle:
                handle.write(canonical({**value,'event_hash':digest}));handle.flush();os.fsync(handle.fileno())
            con.execute('INSERT INTO events (case_id,kind,created_at,payload,previous_hash,event_hash,artifact,artifact_sha256) VALUES (?,?,?,?,?,?,?,?)',
                (case['case_id'],kind,value['created_at'],canonical(payload),value['previous_hash'],digest,path,sha(self.root/path)))
        return self.state(case)

    def draft(self,index,data):
        case=self.catalogue.case(index)
        polygons=data.get('polygons',[]);self.check_draft(polygons,case)
        payload={'polygons':polygons,'notes':self.notes(data),'checks':self.checks(data),
                 'reference_id':self.reference_id(case,data)}
        return self.append(case,'DRAFT',payload,self.revision(data))

    @staticmethod
    def check_draft(polygons,case):
        if not isinstance(polygons,list) or len(polygons)>20:raise ValueError('At most twenty manual contours')
        for polygon in polygons:
            if not isinstance(polygon,list) or len(polygon)>5000:raise ValueError('Too many vertices')
            for p in polygon:
                if not isinstance(p,list) or len(p)!=2 or any(type(v) not in (int,float) or not np.isfinite(v) for v in p):raise ValueError('Invalid vertex')
                if not (0<=p[0]<case['width'] and 0<=p[1]<case['height']):raise ValueError('Vertex outside original crop')

    @staticmethod
    def notes(data):
        notes=data.get('notes','')
        if not isinstance(notes,str) or len(notes)>10000:raise ValueError('Invalid notes')
        return notes

    @staticmethod
    def checks(data):
        checks=data.get('checks',{})
        if not isinstance(checks,dict):raise ValueError('Invalid body/neighbour checks')
        return {key:checks.get(key) is True for key in ['whole_body','no_unlabeled_neighbour']}

    @staticmethod
    def revision(data):
        revision=data.get('revision')
        if type(revision) is not int or revision<0:raise ValueError('Expected explicit revision number')
        return revision

    @staticmethod
    def reference_id(case,data):
        value=data.get('reference_id')
        if value is None:return None
        ref=case.get('human_reference')
        if not ref or not ref['usable_losslessly'] or value!=ref['id']:raise ValueError('Human reference cannot be mapped losslessly')
        if sha(Path(ref['path']))!=ref['sha256']:raise ValueError('Original human reference changed')
        return value

    def decide(self,index,data):
        case=self.catalogue.case(index);number=data.get('decision')
        if type(number) is not int or number not in CATEGORIES:raise ValueError('Choose decision1–5')
        if case['role']!='train' and number==1:raise PermissionError('Val/test must be manually annotated or confirmed, never accepted from a model proposal')
        self.catalogue.verify_case(case)
        pred=self.catalogue.prediction(case)
        fast=data.get('fast_review',False)
        if type(fast) is not bool:raise ValueError('Expected boolean fast_review')
        checks=self.checks(data)
        if fast and number in (1,2):checks={'whole_body':True,'no_unlabeled_neighbour':True}
        polygons=[];label_source='NO_RELIABLE_POSITIVE_GT';export_valid=False
        ref_id=self.reference_id(case,data)
        if number in (1,2):
            if not all(checks.values()):raise ValueError('Confirm whole body and absence of unlabelled neighbours before accepting GT')
            if number==1:
                polygons=copy.deepcopy(pred['proposal'])
                if not polygons:raise ValueError('No proposed contour; draw manually with2 or classify3/4/5')
                self.check_draft(polygons,case)
                label_source='HUMAN_ACCEPTED_MODEL_OUTLINE'
                try:
                    for p in polygons:validate_polygon(p,case['width'],case['height'])
                    export_valid=True
                except ValueError:export_valid=False
            else:
                self.check_draft(data.get('polygons',[]),case)
                polygons=[normalize_polygon_vertices(p) for p in data.get('polygons',[]) if p]
                if not polygons:raise ValueError('Draw/confirm at least one human contour')
                for p in polygons:validate_polygon(p,case['width'],case['height'])
                export_valid=True
                if case['role']=='train' and content_hash(polygons)==pred['proposal_sha256']:
                    raise ValueError('Contour is unchanged; use decision1, or actually correct it before2')
                label_source='HUMAN_CORRECTED_MODEL_PROPOSAL' if case['role']=='train' else 'HUMAN_GT_BLIND_IN_THIS_TOOL'
                if ref_id:label_source='HUMAN_CONFIRMED_OR_EDITED_PREVIOUS_MANUAL_REFERENCE'
        label={'source_id':case['source_id'],'detection_id':case['detection_id'],'role':case['role'],
            'category':CATEGORIES[number],'polygons':polygons,'label':'trunk',
            'coordinate_system':'ORIGINAL_CROP_PIXELS','dimensions':[case['width'],case['height']],
            'source_origin':case['source_origin'],'notes':self.notes(data),'checks':checks,
            'reference_id':ref_id,'label_source':label_source,'export_geometry_valid':export_valid}
        previous=self.state(case)
        payload={'label':label,'label_hash':content_hash(label),'input_fingerprint':self.catalogue.fingerprint,
            'source_crop_sha256':case['original_sha256'],'full_source_sha256':case['source_image_sha256'],
            'initial_prediction_sha256':case['initial_mask_sha256'],'candidate_mask_sha256':pred['mask_sha256'],
            'candidate_proposal_sha256':pred['proposal_sha256'],'checkpoint_sha256':self.catalogue.config['candidate_sha256'],
            'locked_after_save':case['role']!='train','revision_after_model_exposure':previous['model_exposed'],
            'unused_draft_polygons':data.get('polygons',[]) if number not in (1,2) else [],
            'fast_review':fast,'confirmation_source':'FAST_REVIEW_EXPLICIT_DECISION' if fast and number in (1,2) else 'MANUAL_CHECKBOXES'}
        return self.append(case,'DECISION',payload,self.revision(data))

    def last_decision(self,con=None):
        """Most recently saved decision still active, including across page reloads."""
        if con is None:
            with self.connection() as conn:return self.last_decision(conn)
        by_id={case['case_id']:case for case in self.catalogue.cases};seen=set()
        for row in con.execute("SELECT * FROM events WHERE kind IN ('DECISION','MIGRATE_CROP') ORDER BY id DESC"):
            if row['case_id'] in seen:continue
            seen.add(row['case_id']);case=by_id[row['case_id']];state=self.state(case,con)
            if state['category']=='PENDING' or state['has_draft']:continue
            return {'index':case['index'],'case_id':case['case_id'],'source_id':case['source_id'],
                    'detection_id':case['detection_id'],'category':state['category'],
                    'revision':state['revision'],'decision_revision':row['id']}
        return None

    def reopen_last(self,index,data):
        """Append a pending revision; never edit or delete the saved decision."""
        case=self.catalogue.case(index);decision_revision=data.get('decision_revision')
        if type(decision_revision) is not int:raise ValueError('Expected decision_revision')
        with self.connection() as con:
            row=con.execute("SELECT * FROM events WHERE id=? AND case_id=? AND kind IN ('DECISION','MIGRATE_CROP')",
                            (decision_revision,case['case_id'])).fetchone()
        if row is None:raise ValueError('Saved decision not found')
        previous=json.loads(row['payload']);label=previous['label']
        draft={key:copy.deepcopy(label[key]) for key in ['notes','checks','reference_id','label_source']}
        draft['polygons']=copy.deepcopy(label['polygons'] if label['category'] in POSITIVE else previous.get('unused_draft_polygons',[]))
        return self.append(case,'REOPEN',{'reason':'USER_UNDO_LAST_DECISION',
            'decision_revision':decision_revision,'previous_label_hash':previous['label_hash'],
            'draft':draft,'revision_after_model_exposure':case['role']!='train'},self.revision(data))

    def unlock(self,index,data):
        case=self.catalogue.case(index);reason=data.get('reason','')
        if not isinstance(reason,str) or len(reason.strip())<3:raise ValueError('Unlock requires a recorded explanatory reason')
        previous=self.state(case)
        return self.append(case,'UNLOCK',{'reason':reason.strip(),'previous_label_hash':previous['final_label_hash'],
            'first_blind_label_hash':previous['first_blind_label_hash'],'subsequent_revision_is_model_exposed':True},self.revision(data))


def metrics_for(catalogue,case,state):
    if state['category'] not in POSITIVE:return {'applicable':False,'reason':'No complete reliable positive GT for this category'}
    gt=polygon_mask(state['polygons'],case['width'],case['height']);ga=int(gt.sum());result={'applicable':True}
    for name,pred in [('initial_v7',catalogue.initial_mask(case)),('exp02_os5',catalogue.candidate_mask(case))]:
        inter=int((gt&pred).sum());pa=int(pred.sum());union=ga+pa-inter
        result[name]={'iou':inter/union if union else 1,'dice':2*inter/(ga+pa) if ga+pa else 1,
                      'pixel_recall':inter/ga if ga else 0,'pixel_precision':inter/pa if pa else 0,
                      'area_ratio':pa/ga if ga else 0}
    return result


class App:
    def __init__(self,catalogue):self.catalogue=catalogue;self.store=ReviewStore(catalogue)

    def can_reveal(self,case):return case['role']=='train' or self.store.state(case)['locked']

    def catalogue_public(self):
        rows=[];pending={};counts={role:{'total':0,'done':0} for role in ['train','val','test']}
        with self.store.connection() as con:
            for case in self.catalogue.cases:
                state=self.store.state(case,con);role=case['role'];counts[role]['total']+=1
                done=state['category']!='PENDING' and not state['has_draft']
                counts[role]['done']+=done
                if not done and role not in pending:pending[role]=case['index']
                rows.append({'index':case['index'],'source_id':case['source_id'],'detection_id':case['detection_id'],
                             'role':role,'category':state['category'],'locked':state['locked'],
                             'has_human_reference':bool(case.get('human_reference'))})
            latest=self.store.last_decision(con)
        return {'cases':rows,'progress':counts,'next_pending':pending,'last_decision':latest,
                'review_version':self.catalogue.config.get('review_version','v1')}

    def detail(self,index):
        case=self.catalogue.case(index);state=self.store.state(case)
        public={key:case[key] for key in ['index','case_id','source_id','detection_id','role','width','height']}
        public.update(state=state,context=self.catalogue.context(case),
                      original_sha256=case['original_sha256'],source_image_sha256=case['source_image_sha256'],
                      blind_before_lock=case['role']!='train' and not state['locked'])
        if 'crop_frame' in case:public['crop_frame']=case['crop_frame']
        ref=case.get('human_reference')
        public['human_reference']=None if not ref else {k:ref[k] for k in ['id','sha256','original_annotation_size','same_frame','usable_losslessly','mapping_note','polygons']}
        if self.can_reveal(case):
            pred=self.catalogue.prediction(case)
            public['predictions']={key:pred[key] for key in ['proposal','proposal_sha256','confidence','has_mask','connected_components','holes','touches_crop_edge','mask_sha256']}
            public['initial_has_mask']=case.get('initial_has_prediction',bool(case['initial_mask']))
            if case['role']=='train':
                public['legacy_warnings']={k:case[k] for k in ['legacy_position','legacy_crop_condition','legacy_blocked_warning']}
            else:public['metrics']=metrics_for(self.catalogue,case,state)
        return public


class Server(ThreadingHTTPServer):
    daemon_threads=True
    def __init__(self,address,app):
        self.app=app;self.context_lock=threading.Lock();super().__init__(address,Handler)

    @lru_cache(maxsize=8)
    def context_image(self,index):
        case=self.app.catalogue.case(index);box=self.app.catalogue.context(case)['box']
        Image.MAX_IMAGE_PIXELS=None
        with self.context_lock,Image.open(case['source_image']) as image:part=image.crop(box).convert('RGB')
        part.thumbnail((1200,1000),Image.Resampling.LANCZOS);buffer=io.BytesIO();part.save(buffer,format='JPEG',quality=94)
        return buffer.getvalue()


class Handler(BaseHTTPRequestHandler):
    def send(self,body,mime='application/json; charset=utf-8',status=200):
        self.send_response(status);self.send_header('Content-Type',mime);self.send_header('Content-Length',str(len(body)))
        self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff');self.end_headers()
        try:self.wfile.write(body)
        except (BrokenPipeError,ConnectionResetError):pass

    def json(self,value,status=200):self.send(json.dumps(value,ensure_ascii=False).encode(),status=status)

    def do_GET(self):
        try:
            parsed=urlparse(self.path);query=parse_qs(parsed.query);app=self.server.app
            if parsed.path=='/':return self.send(Path(__file__).with_suffix('.html').read_bytes(),'text/html; charset=utf-8')
            if parsed.path=='/api/catalogue':return self.json(app.catalogue_public())
            index=int(query.get('index',['-1'])[0]);case=app.catalogue.case(index)
            if parsed.path=='/api/case':return self.json(app.detail(index))
            if parsed.path=='/api/history':
                with app.store.connection() as con:
                    rows=[]
                    for r in con.execute('SELECT * FROM events WHERE case_id=? ORDER BY id',(case['case_id'],)):
                        payload=json.loads(r['payload'])
                        item={'revision':r['id'],'kind':r['kind'],'created_at':r['created_at'],'event_hash':r['event_hash'],
                              'reason':payload.get('reason','')}
                        if payload.get('original_parent_events'):
                            item['original_v1_revisions']=[{k:original[k] for k in ['id','kind','created_at','event_hash','case_id']}
                                for original in payload['original_parent_events']]
                        rows.append(item)
                return self.json(rows)
            if parsed.path=='/asset':
                kind=query.get('kind',[''])[0]
                if kind=='original':return self.send(Path(case['original']).read_bytes(),'image/png')
                if kind=='context':return self.send(self.server.context_image(index),'image/jpeg')
                if kind in {'initial','candidate'}:
                    if not app.can_reveal(case):raise PermissionError('Blind mode: predictions are unavailable before locked human annotation')
                    if kind=='candidate':return self.send((app.catalogue.root/app.catalogue.prediction(case)['mask']).read_bytes(),'image/png')
                    if case['initial_mask']:return self.send(Path(case['initial_mask']).read_bytes(),'image/png')
                    buffer=io.BytesIO();Image.new('L',(case['width'],case['height'])).save(buffer,format='PNG')
                    return self.send(buffer.getvalue(),'image/png')
                raise ValueError('Unsupported asset')
            return self.json({'error':'Not found'},404)
        except PermissionError as exc:self.json({'error':str(exc)},403)
        except (ValueError,KeyError,IndexError) as exc:self.json({'error':str(exc)},400)
        except OSError as exc:self.json({'error':str(exc)},500)

    def do_POST(self):
        try:
            if self.path not in {'/api/draft','/api/decision','/api/unlock','/api/reopen-last'}:return self.json({'error':'Not found'},404)
            length=int(self.headers.get('Content-Length','0'))
            if self.headers.get('Content-Type')!='application/json' or not 0<length<2000000:raise ValueError('Expected small JSON request')
            data=json.loads(self.rfile.read(length))
            if not isinstance(data,dict):raise ValueError('Expected an object')
            app=self.server.app;index=data['index']
            fn={'/api/draft':app.store.draft,'/api/decision':app.store.decide,'/api/unlock':app.store.unlock,
                '/api/reopen-last':app.store.reopen_last}[self.path]
            fn(index,data)
            self.json(app.detail(index))
        except PermissionError as exc:self.json({'error':str(exc)},403)
        except RuntimeError as exc:self.json({'error':str(exc)},409)
        except (ValueError,KeyError,TypeError) as exc:self.json({'error':str(exc)},400)
        except (OSError,sqlite3.Error) as exc:self.json({'error':str(exc)},500)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/ruler_exp02_os5_manual_validation_v1')
    parser.add_argument('--port',type=int,default=8770)
    args=parser.parse_args();catalogue=Catalogue(args.output.resolve())
    if catalogue.config['candidate_sha256']!=FIXED_SHA:raise ValueError('Only the approved fixed candidate is allowed')
    app=App(catalogue);server=Server(('127.0.0.1',args.port),app)
    print(f'Controlled ruler review: http://127.0.0.1:{server.server_port} ; default TRAIN; val/test blind until lock',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()


if __name__=='__main__':main()
