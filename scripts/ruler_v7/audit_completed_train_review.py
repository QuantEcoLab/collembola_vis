#!/usr/bin/env python3
"""Seal and audit completed original-crop TRAIN decisions; emit manifests, not a dataset."""

import argparse
import csv
import json
import math
import shutil
import sqlite3
from collections import Counter, defaultdict
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from prepare_controlled_review import ROOT, ROLES, sha, content_hash, canonical
from web_controlled_ruler_review import Catalogue, ReviewStore, CATEGORIES, POSITIVE
from web_reference_annotation import validate_polygon

TRAIN_SOURCES={'C_1_2_Collembola005','C_1_3_Collembola006'}


def read_connection(path):
    return sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True)


def read_snapshot(root):
    """Use one read transaction; no ReviewStore initialization or writable DB connection."""
    cases=json.loads((root/'cases.json').read_text())
    config=json.loads((root/'config.json').read_text())
    if config.get('crop_or_padding_changed') or config.get('review_version','v1')!='v1':
        raise ValueError('Only the original unpadded v1 review is allowed')
    if len(cases)!=779 or len({c['case_id'] for c in cases})!=779:raise ValueError('Wrong case catalogue')
    for case in cases:
        if ROLES[case['source_id']]!=case['role']:raise ValueError('Source role changed')
    with closing(read_connection(root/'review.sqlite3')) as con:
        con.row_factory=sqlite3.Row;con.execute('BEGIN')
        if con.execute('PRAGMA integrity_check').fetchone()[0]!='ok':raise ValueError('SQLite integrity failure')
        events=[dict(r) for r in con.execute('SELECT * FROM events ORDER BY id')]
        metadata=[dict(r) for r in con.execute('SELECT * FROM metadata ORDER BY key')]
        store=ReviewStore.__new__(ReviewStore)
        states={case['case_id']:store.state(case,con) for case in cases}
    previous='';case_ids={c['case_id'] for c in cases}
    for event in events:
        if event['case_id'] not in case_ids:raise ValueError('Unknown event identity')
        if event['kind'] not in {'DRAFT','DECISION','REOPEN','UNLOCK'}:raise ValueError('Unexpected event kind in original v1')
        value={key:event[key] for key in ['case_id','kind','created_at','previous_hash']}
        value['payload']=json.loads(event['payload'])
        if event['previous_hash']!=previous or content_hash(value)!=event['event_hash']:raise ValueError('Broken event chain')
        artifact=root/event['artifact']
        if sha(artifact)!=event['artifact_sha256']:raise ValueError('Changed history artifact')
        if json.loads(artifact.read_text())!={**value,'event_hash':event['event_hash']}:raise ValueError('Artifact does not match database event')
        previous=event['event_hash']
    progress={role:{'total':0,'done':0,'pending':0} for role in ['train','val','test']}
    counts=Counter();by_source={}
    for case in cases:
        state=states[case['case_id']];done=state['category']!='PENDING' and not state['has_draft']
        p=progress[case['role']];p['total']+=1;p['done']+=done;p['pending']+=not done
        if case['role']=='train':
            counts[state['category']]+=1
            by_source.setdefault(case['source_id'],Counter())[state['category']]+=1
    if progress['train']!={'total':567,'done':567,'pending':0}:raise ValueError(f'TRAIN is not complete: {progress["train"]}')
    if set(by_source)!=TRAIN_SOURCES:raise ValueError('Wrong TRAIN source allowlist')
    summary={'progress':progress,'train_categories':{category:counts[category] for category in CATEGORIES.values()},
        'train_by_source':{s:{category:c[category] for category in CATEGORIES.values()} for s,c in sorted(by_source.items())},
        'history_events':len(events),'last_event_id':events[-1]['id'],'history_head_sha256':previous,
        'event_rows_sha256':content_hash(events),'metadata_sha256':content_hash(metadata),
        'database_sha256':sha(root/'review.sqlite3')}
    return cases,states,events,metadata,summary


def write_json(path,value):
    with path.open('x',encoding='utf-8') as handle:json.dump(value,handle,indent=2,ensure_ascii=False,allow_nan=False)


def write_csv(path,rows,fields):
    with path.open('x',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def polygon_audit(polygons,width,height):
    """Audit the exact saved vertices. Never normalize, clip, round, or repair them."""
    records=[];errors=[]
    if not isinstance(polygons,list) or not polygons:return [],['NO_FINAL_POLYGONS']
    for i,polygon in enumerate(polygons):
        count=len(polygon) if isinstance(polygon,list) else 0
        record={'polygon_index':i,'vertex_count':count,'valid':False,'errors':[],
            'finite_coordinates':False,'coordinates_in_bounds':False,'edge_sides':[],'area_px2':None,
            'distinct_vertices':0,'repeated_vertices':0}
        valid_points=isinstance(polygon,list) and all(isinstance(p,list) and len(p)==2 and
            all(type(v) in (int,float) and math.isfinite(v) for v in p) for p in polygon)
        record['finite_coordinates']=valid_points and count>0
        if valid_points and count:
            xs=[p[0] for p in polygon];ys=[p[1] for p in polygon]
            record['coordinates_in_bounds']=all(0<=x<width and 0<=y<height for x,y in polygon)
            record['bounds_xyxy']=[min(xs),min(ys),max(xs),max(ys)]
            record['distinct_vertices']=len({tuple(p) for p in polygon})
            record['repeated_vertices']=count-record['distinct_vertices']
            record['area_px2']=abs(sum(a[0]*b[1]-b[0]*a[1] for a,b in zip(polygon,polygon[1:]+polygon[:1])))/2
            eps=1e-6
            record['edge_sides']=[name for name,flag in [('left',min(xs)<=eps),('top',min(ys)<=eps),
                ('right',max(xs)>=width-1-eps),('bottom',max(ys)>=height-1-eps)] if flag] if record['coordinates_in_bounds'] else []
        try:
            validate_polygon(polygon,width,height)
            if count>5000:raise ValueError('More than the review editor limit of 5000 vertices')
            record['valid']=True
        except (ValueError,TypeError,IndexError) as exc:
            record['errors'].append(str(exc));errors.append(f'polygon {i}: {exc}')
        records.append(record)
    return records,errors


def audit(source,output):
    cases,states,events,metadata,preflight=read_snapshot(source)
    if output.exists():raise ValueError(f'Refusing to overwrite {output}')
    output.mkdir(parents=True);backup=output/'backup';backup.mkdir()
    with closing(read_connection(source/'review.sqlite3')) as src,closing(sqlite3.connect(backup/'review.sqlite3')) as dest:
        src.backup(dest)
    # Check the snapshot boundary again before committing the audit to this backup.
    for name in ['cases.json','config.json']:
        shutil.copy2(source/name,backup/name)
    shutil.copytree(source/'annotation_revisions',backup/'annotation_revisions')
    for name in ['frozen_inputs/bundle_seal.json','frozen_inputs/inventory.csv','prediction_seal.json']:
        path=backup/name;path.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source/name,path)
    backup_cases,backup_states,backup_events,backup_metadata,summary=read_snapshot(backup)
    if backup_events!=events or backup_metadata!=metadata or backup_cases!=cases or backup_states!=states:
        raise ValueError('Source changed while backing up; no source data changed, audit must use a fresh snapshot')
    backup_hashes={p.relative_to(backup).as_posix():sha(p) for p in backup.rglob('*') if p.is_file()}
    write_json(backup/'seal.json',{'algorithm':'SHA-256','files':backup_hashes,
        'database_sha256':sha(backup/'review.sqlite3'),'event_rows_sha256':content_hash(events),
        'history_head_sha256':summary['history_head_sha256'],'source_database_sha256_before':preflight['database_sha256']})
    (backup/'review.sqlite3.sha256').write_text(sha(backup/'review.sqlite3')+'  review.sqlite3\n')
    (backup/'seal.json.sha256').write_text(sha(backup/'seal.json')+'  seal.json\n')
    # Catalogue only validates existing seals and hashes. It does not run a model or initialize a store.
    catalogue=Catalogue(source)
    if next(m['value'] for m in metadata if m['key']=='input_fingerprint')!=catalogue.fingerprint:
        raise ValueError('Review fingerprint does not match the frozen original crops')
    by_revision={r['id']:r for r in events};candidates=[];positives=[];audit_rows=[];excluded=[];polygon_rows=[]
    problems=[];aggregate=Counter();vertices=[]
    for case in cases:
        if case['role']!='train':continue
        state=states[case['case_id']];category=state['category'];number=next(n for n,c in CATEGORIES.items() if c==category)
        event=by_revision[state['revision']]
        if event['kind']!='DECISION':raise ValueError('Completed TRAIN state is not a final DECISION')
        payload=json.loads(event['payload']);label=payload['label']
        if payload['label_hash']!=content_hash(label) or payload['label_hash']!=state['final_label_hash']:
            raise ValueError('Final label hash mismatch')
        for key in ['source_id','detection_id','role']:
            if label[key]!=case[key]:raise ValueError(f'Final label identity mismatch: {case["case_id"]}')
        base={'case_id':case['case_id'],'source_id':case['source_id'],'detection_id':case['detection_id'],
            'role':'train','decision':number,'category':category,'revision':event['id'],'created_at':event['created_at'],
            'event_sha256':event['event_hash'],'label_sha256':payload['label_hash'],
            'image_path':case['original'],'expected_image_sha256':case['original_sha256'],
            'label_artifact_path':str(backup/event['artifact']),'label_json_pointer':'/payload/label',
            'label_artifact_sha256':event['artifact_sha256'],'width':case['width'],'height':case['height'],
            'source_origin':case['source_origin'],'notes':state['notes'],'checks':state['checks']}
        if category not in POSITIVE:
            excluded.append({k:base[k] for k in ['case_id','source_id','detection_id','decision','category','revision']})
            continue
        errors=[];image_exists=Path(case['original']).is_file();artifact_exists=(backup/event['artifact']).is_file()
        actual_image_sha=None;image_dimensions=None
        if image_exists:
            actual_image_sha=sha(Path(case['original']))
            try:
                with Image.open(case['original']) as image:
                    image_dimensions=list(image.size);image.verify()
            except OSError as exc:errors.append('IMAGE_UNREADABLE: '+str(exc))
        else:errors.append('IMAGE_MISSING')
        if not artifact_exists:errors.append('FINAL_LABEL_ARTIFACT_MISSING')
        if actual_image_sha!=case['original_sha256']:errors.append('IMAGE_HASH_MISMATCH')
        if image_dimensions!=[case['width'],case['height']]:errors.append('IMAGE_DIMENSIONS_MISMATCH')
        if label.get('dimensions')!=[case['width'],case['height']]:errors.append('LABEL_DIMENSIONS_MISMATCH')
        if label.get('coordinate_system')!='ORIGINAL_CROP_PIXELS':errors.append('WRONG_COORDINATE_SYSTEM')
        if label.get('source_origin')!=case['source_origin']:errors.append('SOURCE_ORIGIN_MISMATCH')
        if payload.get('source_crop_sha256')!=case['original_sha256']:errors.append('LABEL_IMAGE_PROVENANCE_MISMATCH')
        if label.get('label')!='trunk':errors.append('UNEXPECTED_CLASS')
        polygon_details,geometry_errors=polygon_audit(state['polygons'],case['width'],case['height'])
        errors.extend(geometry_errors)
        if not all(state['checks'].get(k) is True for k in ['whole_body','no_unlabeled_neighbour']):errors.append('MISSING_POSITIVE_CONFIRMATIONS')
        if not state['export_geometry_valid']:errors.append('SAVED_EXPORT_GEOMETRY_INVALID')
        touches_edge=any(p['edge_sides'] for p in polygon_details)
        eligible=not errors
        audit_record={**base,'polygons':state['polygons'],'polygon_audit':polygon_details,
            'geometry_valid':not geometry_errors,'image_exists':image_exists,'final_label_artifact_exists':artifact_exists,
            'image_label_pair_exists':image_exists and artifact_exists,'actual_image_sha256':actual_image_sha,
            'image_dimensions':image_dimensions,'touches_crop_edge':touches_edge,'errors':errors,
            'training_candidate':eligible,'edge_policy':'Flag only; no automatic change to human decision or exclusion of otherwise valid positives'}
        positives.append(audit_record)
        if eligible:candidates.append(audit_record)
        else:problems.append(audit_record)
        aggregate['positive_cases']+=1;aggregate['polygons']+=len(polygon_details)
        aggregate['valid_geometry_cases']+=not geometry_errors;aggregate['invalid_geometry_cases']+=bool(geometry_errors)
        aggregate['edge_touching_cases']+=touches_edge;aggregate['image_label_pairs_exist']+=image_exists and artifact_exists
        aggregate['eligible_candidates']+=eligible;aggregate['blocked_positive_cases']+=not eligible
        aggregate['nonfinite_coordinate_cases']+=any(not p['finite_coordinates'] for p in polygon_details)
        aggregate['out_of_bounds_cases']+=any(p['finite_coordinates'] and not p['coordinates_in_bounds'] for p in polygon_details)
        aggregate['repeated_vertex_cases']+=any(p['repeated_vertices'] for p in polygon_details)
        audit_rows.append({'case_id':case['case_id'],'source_id':case['source_id'],'detection_id':case['detection_id'],
            'decision':number,'category':category,'revision':event['id'],'polygon_count':len(polygon_details),
            'vertex_count':sum(p['vertex_count'] for p in polygon_details),'geometry_valid':not geometry_errors,
            'touches_crop_edge':touches_edge,'image_label_pair_exists':image_exists and artifact_exists,
            'training_candidate':eligible,'errors':' | '.join(errors)})
        for polygon in polygon_details:
            vertices.append(polygon['vertex_count'])
            polygon_rows.append({'case_id':case['case_id'],**{k:polygon[k] for k in ['polygon_index','vertex_count','valid','finite_coordinates','coordinates_in_bounds','distinct_vertices','repeated_vertices','area_px2']},
                'edge_sides':','.join(polygon['edge_sides']),'errors':' | '.join(polygon['errors'])})
        if aggregate['positive_cases']%50==0:print(f'Audited {aggregate["positive_cases"]} positive labels',flush=True)
    if len(candidates)+len(problems)+len(excluded)!=567:raise ValueError('TRAIN accounting failure')
    if any(r['decision'] not in (1,2) or r['source_id'] not in TRAIN_SOURCES for r in candidates):raise ValueError('Forbidden training candidate')
    counts_rows=[]
    for source_id,counts in [('ALL_TRAIN',summary['train_categories']),*summary['train_by_source'].items()]:
        counts_rows.append({'source_id':source_id,**{str(n):counts[c] for n,c in CATEGORIES.items()},'total':sum(counts.values())})
    write_csv(output/'decision_counts.csv',counts_rows,['source_id','1','2','3','4','5','total'])
    write_csv(output/'positive_polygon_audit.csv',audit_rows,list(audit_rows[0]))
    write_csv(output/'polygon_details.csv',polygon_rows,list(polygon_rows[0]))
    write_csv(output/'excluded_decisions_3_4_5.csv',excluded,['case_id','source_id','detection_id','decision','category','revision'])
    common={'purpose':'TRAIN_CANDIDATE_MANIFEST_ONLY_NOT_A_DATASET','source_review':str(source),'backup_database':str(backup/'review.sqlite3'),
        'backup_database_sha256':sha(backup/'review.sqlite3'),'source_roles':ROLES,'training_source_allowlist':sorted(TRAIN_SOURCES),
        'category_allowlist':['ACCEPTED_UNCHANGED','MANUALLY_CORRECTED'],'categories_3_4_5_forbidden':True,
        'image_label_pair_definition':'Original existing crop PNG + copied immutable final event JSON at /payload/label; no YOLO label files generated'}
    write_json(output/'positive_decisions_manifest.json',{**common,'count':len(positives),'items':positives})
    write_json(output/'training_candidates_manifest.json',{**common,'count':len(candidates),'items':candidates})
    write_json(output/'blocked_positive_decisions.json',{'count':len(problems),'items':problems})
    final_cases,final_states,final_events,final_metadata,final_summary=read_snapshot(source)
    if final_events!=events or final_metadata!=metadata or final_states!=states or final_cases!=cases:
        raise ValueError('Source review changed during audit; snapshot remains valid but report is not the current source state')
    if sha(source/'review.sqlite3')!=preflight['database_sha256']:raise ValueError('Original database bytes changed during audit')
    summary.update(output=str(output),source_review=str(source),backup=str(backup),backup_database_sha256=sha(backup/'review.sqlite3'),
        backup_seal_sha256=sha(backup/'seal.json'),audit=dict(aggregate),
        vertices={'minimum':min(vertices,default=0),'maximum':max(vertices,default=0),'total':sum(vertices)},
        candidates_by_category=dict(Counter(r['category'] for r in candidates)),
        blocked_positives_by_category=dict(Counter(r['category'] for r in problems)),
        original_database_bytes_unchanged=True,original_event_rows_and_artifacts_unchanged=True,
        edge_touch_is_diagnostic_not_a_new_rejection=True)
    write_json(output/'summary.json',summary)
    report=['# Završni audit izvornog TRAIN pregleda','',
        f'567/567 dovršenih TRAIN slučajeva; 0 TRAIN PENDING. VAL/TEST ostaju izvan manifesta.',
        f'Backup: `{backup}`',f'SHA-256 baze: `{summary["backup_database_sha256"]}`','',
        '| Source | 1 accepted | 2 corrected | 3 false | 4 ambiguous | 5 neighbour | Ukupno |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for row in counts_rows:report.append('| '+' | '.join(str(row[k]) for k in ['source_id','1','2','3','4','5','total'])+' |')
    report+=['','## Provjera pozitivnih oznaka',f'- Odluke 1/2: {len(positives)}.',
        f'- Geometrijski valjani slučajevi: {aggregate["valid_geometry_cases"]}; nevaljani: {aggregate["invalid_geometry_cases"]}.',
        f'- Image–label parovi postoje: {aggregate["image_label_pairs_exist"]}/{len(positives)}.',
        f'- Dodir ruba: {aggregate["edge_touching_cases"]} slučajeva (dijagnostička oznaka, ne promjena ljudske odluke).',
        f'- Broj poligona: {aggregate["polygons"]}; vrhova: {sum(vertices)}; raspon po poligonu: {min(vertices,default=0)}–{max(vertices,default=0)}.',
        f'- Kandidati koji prolaze sve provjere: {len(candidates)}; pozitivni blokirani provjerama: {len(problems)}.',
        '', 'Koordinate se provjeravaju bez promjene: 0 ≤ x < width, 0 ≤ y < height. Dodir ruba znači',
        'x ≤ 1e-6, y ≤ 1e-6, x ≥ width−1−1e-6 ili y ≥ height−1−1e-6 u izvornom pikselnom sustavu.',
        '', 'Image–label par u ovom izvještaju znači postojeća originalna PNG slika i nepromjenjivi',
        'finalni JSON događaj, s oznakom na `/payload/label`. Nisu generirane nove YOLO label datoteke.',
        '', '## Datoteke', '- `decision_counts.csv`: sažetak ukupno i po source ID-ju.',
        '- `positive_polygon_audit.csv` i `polygon_details.csv`: provjere svake odluke 1/2 i svakog poligona.',
        '- `positive_decisions_manifest.json`: sve konačne TRAIN odluke 1/2, uključujući dijagnostiku.',
        '- `training_candidates_manifest.json`: samo TRAIN odluke 1/2 koje prolaze provjere.',
        '- `blocked_positive_decisions.json`: pozitivne odluke koje ne prolaze provjere; izvornici nisu popravljani.',
        '- `excluded_decisions_3_4_5.csv`: odluke koje su potpuno isključene iz kandidata.',
        '- `summary.json`: strojno čitljiv izvještaj i hashovi.',
        '- `backup/seal.json`, `backup/review.sqlite3.sha256`: zapečaćen backup baze i povijesti.',
        '', 'Izvorna baza, poligoni, bilješke, timestampovi i povijest ostali su identični.',
        'Manifest nije dataset. Nije izvršena inferenca ili trening.']
    (output/'REPORT.md').write_text('\n'.join(report)+'\n')
    report_hashes={p.relative_to(output).as_posix():sha(p) for p in output.iterdir() if p.is_file()}
    write_json(output/'report_seal.json',{'algorithm':'SHA-256','files':report_hashes,'backup_seal_sha256':sha(backup/'seal.json')})
    (output/'report_seal.json.sha256').write_text(sha(output/'report_seal.json')+'  report_seal.json\n')
    print(json.dumps(summary,indent=2,ensure_ascii=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=ROOT/'outputs/ruler_exp02_os5_manual_validation_v1')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--check',action='store_true',help='Read-only completion/history preflight; write nothing')
    args=parser.parse_args();source=args.source.resolve()
    if args.check:
        print(json.dumps(read_snapshot(source)[-1],indent=2,ensure_ascii=False));return
    output=args.output or ROOT/'outputs'/('ruler_train_v1_completed_audit_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    audit(source,output.resolve())


if __name__=='__main__':main()
