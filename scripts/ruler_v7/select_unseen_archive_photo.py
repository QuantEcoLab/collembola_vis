#!/usr/bin/env python3
"""Select one archived native-resolution photograph after a bounded historical-use audit. No models."""

import argparse
import csv
import hashlib
import io
import json
import os
import re
import sqlite3
import unicodedata
import zipfile
import zlib
import gzip
import bz2
import lzma
from collections import Counter,defaultdict
from contextlib import closing
from datetime import datetime,timezone
from pathlib import Path,PurePosixPath

from PIL import Image,ImageOps

ROOT=Path(__file__).resolve().parents[2]
SNAPSHOT=Path('/home/jana/collembola-rescue/incoming/ruler-v7-20261027T162453Z/snapshot/repo')
ARCHIVES=ROOT/'data/unseen_candidate_archives/collembole_za_trening'
PHOTO_EXT={'.jpg','.jpeg','.png','.tif','.tiff','.bmp','.webp','.gif'}
TEXT_EXT={'.json','.jsonl','.csv','.yaml','.yml','.txt','.md','.log','.py','.toml','.ini','.xml','.html','.htm','.js','.ts','.ipynb'}
PRUNE={'.git','.venv','.venv-ruler-ft','node_modules','__pycache__','.cache','.vscode'}
HEX=re.compile(r'(?<![0-9a-fA-F])([0-9a-fA-F]{64})(?![0-9a-fA-F])')
IMAGE_PATH=re.compile(r'(?:[A-Za-z]:[\\/]|/home/|/mnt/|/media/|/tmp/opencode/|data/|outputs/)[^\r\n"<>|,;]*?\.(?:jpe?g|png|tiff?|bmp|webp)',re.I)


def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda:handle.read(8*1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def canonical_id(text):
    return re.sub('[^a-z0-9]','',unicodedata.normalize('NFKC',text).casefold())


def pixel_digests(source):
    """Hash native decoded RGB pixels in stripes; also compare EXIF-oriented representations."""
    def digest(image):
        rgb=image if image.mode=='RGB' else image.convert('RGB')
        raw=hashlib.sha256();canonical=hashlib.sha256(f'RGB:{rgb.width}:{rgb.height}:'.encode())
        for y in range(0,rgb.height,128):
            block=rgb.crop((0,y,rgb.width,min(y+128,rgb.height))).tobytes()
            raw.update(block);canonical.update(block)
        return {'dimensions':[rgb.width,rgb.height],'pixel_bytes_sha256':raw.hexdigest(),'pixel_sha256':canonical.hexdigest()}
    with Image.open(source) as image:
        image.load();orientation=int(image.getexif().get(274,1) or 1)
        native=digest(image)
        oriented=digest(ImageOps.exif_transpose(image)) if orientation!=1 else dict(native)
        return {'native':native,'exif_oriented':oriented,'exif_orientation':orientation,'decoded_mode':'RGB',
            'definition':'pixel_bytes_sha256=SHA256(native RGB pixel bytes); pixel_sha256=SHA256(ASCII RGB:width:height: + same bytes). No resizing.'}


def write_json(path,value):
    with path.open('x',encoding='utf-8') as handle:json.dump(value,handle,indent=2,ensure_ascii=False)


def walk_files(root,excluded):
    for directory,dirs,files in os.walk(root,followlinks=False):
        parent=Path(directory)
        dirs[:]=sorted(d for d in dirs if d not in PRUNE and not any((parent/d).resolve().is_relative_to(x) for x in excluded))
        for name in sorted(files):
            path=parent/name
            if not any(path.resolve().is_relative_to(x) for x in excluded):yield path


def archive_inventory(folder):
    inventory=[];photos=[]
    for path in sorted(folder.glob('*.zip')):
        with zipfile.ZipFile(path) as archive:
            members=[{'name':member.filename,'bytes':member.file_size,'compressed_bytes':member.compress_size,
                'crc32':f'{member.CRC:08x}','encrypted':bool(member.flag_bits&1),'compression':member.compress_type}
                for member in archive.infolist() if not member.is_dir()]
        record={'archive':str(path),'sha256':sha(path),'archive_bytes':path.stat().st_size,'members':members}
        inventory.append(record)
        for member in members:
            filename=PurePosixPath(member['name'])
            if filename.suffix.lower() in PHOTO_EXT:
                stem=filename.stem
                source_id=re.sub(r'\s*\(\d+\)$','',stem)
                photos.append({**member,'archive':str(path),'archive_sha256':record['sha256'],
                    'filename':filename.name,'source_id':source_id,'canonical_source_id':canonical_id(source_id)})
    if not photos:raise ValueError('No archive photographs found')
    return inventory,photos


def run(folder,output,roots):
    if output.exists():raise ValueError('Refusing to overwrite '+str(output))
    started=datetime.now(timezone.utc).isoformat();Image.MAX_IMAGE_PIXELS=None
    archives,candidates=archive_inventory(folder)
    output.mkdir(parents=True);(output/'evidence').mkdir();(output/'input').mkdir()
    write_json(output/'archive_inventory.json',archives)
    excluded=[folder.resolve(),output.resolve()]
    files=[];seen_paths=set()
    for root in roots:
        for path in walk_files(root,excluded):
            resolved=path.resolve()
            if resolved not in seen_paths:files.append(path);seen_paths.add(resolved)
    images=[];metadata=[];databases=[];checkpoints=[];caches=[]
    hits=defaultdict(list);hit_counts=Counter();needles={r['canonical_source_id'] for r in candidates}
    recorded_hashes=defaultdict(list);metadata_errors=[];extra_images=set();metadata_cache={};evidence=[]
    def scan_ids(text,path,kind):
        normalized=canonical_id(text)
        for needle in needles:
            if needle in normalized:
                hit_counts[needle]+=1
                if len(hits[needle])<20:hits[needle].append({'path':str(path),'kind':kind})
    for path in files:
        scan_ids(str(path),path,'historical_filename_or_directory')
        suffix=path.suffix.lower()
        if suffix in PHOTO_EXT:images.append(path)
        elif suffix in TEXT_EXT:
            # Numeric YOLO label contents have no image identities or SHA records; their filenames were checked.
            if suffix=='.txt' and 'labels' in path.parts:continue
            metadata.append(path)
        elif suffix in {'.sqlite3','.db','.sqlite'}:databases.append(path)
        elif suffix=='.pt':checkpoints.append(path)
        elif suffix in {'.cache','.pkl','.pickle','.joblib'}:caches.append(path)
    def resolve_image(value,context):
        value=value.replace('\\\\','/').replace('\\','/').strip()
        guesses=[Path(value),context.parent/value,ROOT/value,SNAPSHOT/value]
        match=re.search(r'(?:^|/)(data|outputs|runs)/',value)
        if match:
            relative=value[match.start():].lstrip('/')
            guesses.extend([ROOT/relative,SNAPSHOT/relative])
        for guess in guesses:
            try:
                if guess.is_file() and guess.suffix.lower() in PHOTO_EXT:
                    resolved=guess.resolve()
                    if not any(resolved.is_relative_to(x) for x in excluded) and resolved not in seen_paths:
                        extra_images.add(resolved)
                    return
            except (OSError,ValueError):pass
    def scan_text(text,path,kind,digest=None,record_evidence=True):
        scan_ids(text,path,kind)
        for match in HEX.finditer(text):
            value=match.group(1).lower()
            if len(recorded_hashes[value])<5:recorded_hashes[value].append(str(path))
        for match in IMAGE_PATH.finditer(text):resolve_image(match.group(0),path)
        if record_evidence:evidence.append({'path':str(path),'kind':kind,'sha256':digest or sha(path)})
    print(json.dumps({'historical_files':len(files),'images':len(images),'text_records':len(metadata),
        'databases':len(databases),'checkpoints':len(checkpoints),'binary_caches':len(caches),'archive_photo_members':len(candidates)}),flush=True)
    for number,path in enumerate(metadata,1):
        try:
            raw=path.read_bytes();digest=hashlib.sha256(raw).hexdigest()
            if digest not in metadata_cache:
                text=raw.decode('utf-8',errors='replace');scan_text(text,path,'historical_text_record',digest)
                metadata_cache[digest]=True
            else:evidence.append({'path':str(path),'kind':'duplicate_text_record','sha256':digest})
        except (OSError,ValueError) as exc:metadata_errors.append({'path':str(path),'error':str(exc)})
        if number%3000==0:print(f'Scanned text records {number}/{len(metadata)}',flush=True)
    for path in databases:
        try:
            database_digest=sha(path)
            with closing(sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True)) as con:
                con.execute('BEGIN')
                tables=[r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]
                for table in tables:
                    quoted='"'+table.replace('"','""')+'"'
                    for row in con.execute('SELECT * FROM '+quoted):
                        text='\n'.join(value.decode('utf-8',errors='replace') if isinstance(value,bytes) else str(value) for value in row if value is not None)
                        scan_text(text,path,'sqlite_table:'+table,database_digest,record_evidence=False)
            evidence.append({'path':str(path),'kind':'readonly_sqlite_all_tables','sha256':database_digest})
        except (OSError,sqlite3.Error) as exc:metadata_errors.append({'path':str(path),'error':str(exc)})
    for path in checkpoints:
        try:
            if zipfile.is_zipfile(path):
                with zipfile.ZipFile(path) as archive:
                    for name in archive.namelist():
                        if name.endswith('/data.pkl') or name=='data.pkl':
                            raw=archive.read(name);scan_text(raw.decode('latin1'),path,'checkpoint_metadata_strings',sha(path))
            else:scan_text(path.read_bytes().decode('latin1'),path,'legacy_checkpoint_strings')
        except (OSError,zipfile.BadZipFile) as exc:metadata_errors.append({'path':str(path),'error':str(exc)})
    for path in caches:
        try:
            raw=path.read_bytes()
            if raw[:2]==b'\x1f\x8b':raw=gzip.decompress(raw)
            elif raw[:3]==b'BZh':raw=bz2.decompress(raw)
            elif raw[:6]==b'\xfd7zXZ\x00':raw=lzma.decompress(raw)
            elif len(raw)>2 and raw[0]==0x78:
                try:raw=zlib.decompress(raw)
                except zlib.error:pass
            scan_text(raw.decode('latin1'),path,'cache_or_pickle_strings_no_unpickle')
        except (OSError,ValueError,lzma.LZMAError) as exc:metadata_errors.append({'path':str(path),'error':str(exc)})
    images.extend(sorted(extra_images));write_json(output/'evidence/metadata_errors.json',metadata_errors)
    with (output/'evidence/historical_records.jsonl').open('x') as handle:
        for row in evidence:handle.write(json.dumps(row,ensure_ascii=False)+'\n')
    if metadata_errors:raise RuntimeError('Historical records could not all be read; see metadata_errors.json. No photo extracted.')
    remaining=[];rejected=[]
    for candidate in candidates:
        if hit_counts[candidate['canonical_source_id']]:
            rejected.append({**candidate,'reason':'SOURCE_ID_ALREADY_RECORDED','matches':hit_counts[candidate['canonical_source_id']],
                'evidence_examples':hits[candidate['canonical_source_id']]})
        elif candidate['encrypted'] or candidate['bytes']>100_000_000 or re.search(r'_(?:det\d+|crop|mask|overlay|thumb|x)$',Path(candidate['filename']).stem,re.I):
            rejected.append({**candidate,'reason':'NOT_PREFERRED_ORIGINAL_PHOTO_MEMBER'})
        else:remaining.append(candidate)
    write_json(output/'evidence/source_id_rejections.json',rejected)
    print(json.dumps({'source_id_rejected':sum(r['reason']=='SOURCE_ID_ALREADY_RECORDED' for r in rejected),
        'not_preferred':sum(r['reason']!='SOURCE_ID_ALREADY_RECORDED' for r in rejected),'source_new_candidates':len(remaining),
        'new_candidate_names':[c['filename'] for c in remaining]}),flush=True)
    if not remaining:raise RuntimeError('No wholly new source ID found; no photo extracted')
    # Header-only inventory first. Differing native dimensions prove a canonical pixel digest cannot match.
    image_rows=[];image_errors=[];byte_groups=defaultdict(list);dimensions=defaultdict(list)
    for number,path in enumerate(images,1):
        try:
            digest=sha(path)
            with Image.open(path) as image:
                width,height=image.size;orientation=int(image.getexif().get(274,1) or 1)
            row={'path':str(path),'file_sha256':digest,'width':width,'height':height,'exif_orientation':orientation}
            image_rows.append(row);byte_groups[digest].append(row);dimensions[(width,height)].append(row)
        except (OSError,ValueError,Image.DecompressionBombError) as exc:image_errors.append({'path':str(path),'error':str(exc)})
        if number%5000==0:print(f'Indexed image files {number}/{len(images)}',flush=True)
    write_json(output/'evidence/image_errors.json',image_errors)
    with (output/'evidence/historical_images.csv').open('x',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=['path','file_sha256','width','height','exif_orientation']);writer.writeheader();writer.writerows(image_rows)
    if image_errors:raise RuntimeError('Historical images could not all be indexed; no photo extracted')
    pixel_cache={};pixel_evidence=[];attempts=[];selected=None;selected_bytes=None
    for candidate in sorted(remaining,key=lambda c:(c['archive'],c['name'])):
        with zipfile.ZipFile(candidate['archive']) as archive:
            with archive.open(candidate['name']) as stream:
                with Image.open(stream) as im:width,height=im.size
            if min(width,height)<8000:
                attempts.append({**candidate,'dimensions':[width,height],'rejected':'NOT_FULL_RESOLUTION_8K_MINIMUM'});continue
            raw=archive.read(candidate['name'])
        file_digest=hashlib.sha256(raw).hexdigest()
        if file_digest in byte_groups or file_digest in recorded_hashes:
            attempts.append({**candidate,'dimensions':[width,height],'file_sha256':file_digest,'rejected':'FILE_HASH_ALREADY_RECORDED'});continue
        candidate_pixels=pixel_digests(io.BytesIO(raw))
        pixel_hashes={candidate_pixels[k][field] for k in ['native','exif_oriented'] for field in ['pixel_bytes_sha256','pixel_sha256']}
        recorded_matches={value:recorded_hashes[value] for value in pixel_hashes if value in recorded_hashes}
        if recorded_matches:
            attempts.append({**candidate,'dimensions':[width,height],'file_sha256':file_digest,'rejected':'PIXEL_HASH_IN_HISTORICAL_RECORD','matches':recorded_matches});continue
        comparable=[row for row in image_rows if (row['width'],row['height']) in {(width,height),(height,width)}]
        comparisons=[];match=None
        for row in comparable:
            key=row['file_sha256']
            if key not in pixel_cache:
                value=pixel_digests(Path(row['path']));pixel_cache[key]=value
                pixel_evidence.append({'representative_image':row['path'],'file_sha256':key,**value})
            value=pixel_cache[key]
            historical_hashes={value[k][field] for k in ['native','exif_oriented'] for field in ['pixel_bytes_sha256','pixel_sha256']}
            comparisons.append(row['path'])
            if pixel_hashes&historical_hashes:
                match={'path':row['path'],'matching_hashes':sorted(pixel_hashes&historical_hashes)};break
        if match:
            attempts.append({**candidate,'dimensions':[width,height],'file_sha256':file_digest,'rejected':'DECODED_PIXEL_HASH_ALREADY_PRESENT','match':match});continue
        selected={**candidate,'dimensions':[width,height],'file_sha256':file_digest,'pixels':candidate_pixels,
            'source_id_matches':0,'file_hash_matches':0,'recorded_pixel_hash_matches':0,'computed_pixel_hash_matches':0,
            'same_or_transposed_dimensions_compared':len(comparisons),'comparable_image_paths':comparisons}
        selected_bytes=raw;break
    write_json(output/'evidence/pixel_hash_inventory.json',pixel_evidence)
    write_json(output/'evidence/candidate_hash_rejections.json',attempts)
    if selected is None:raise RuntimeError('No candidate passed all identity/hash checks; no photo extracted')
    # Recheck membership against exactly the historical baseline gathered before this new intake output existed.
    if selected['file_sha256'] in byte_groups or hit_counts[selected['canonical_source_id']]:raise ValueError('Novelty assertion failed')
    frozen=output/'input'/selected['filename']
    with frozen.open('xb') as handle:handle.write(selected_bytes)
    if sha(frozen)!=selected['file_sha256'] or pixel_digests(frozen)!=selected['pixels']:raise ValueError('Freeze copy mismatch')
    selected['frozen_path']=str(frozen);selected['frozen_image_bytes']=frozen.stat().st_size
    coverage={'roots':[str(root) for root in roots],'excluded_paths':[str(p) for p in excluded],
        'exclusion_rationale':'Newly uploaded candidate archives and this new intake output are not historical usage. Environment, git internals and package caches are excluded.',
        'historical_file_paths_checked':len(files),'historical_images_checked_by_file_sha256':len(image_rows),
        'distinct_historical_image_file_hashes':len(byte_groups),'historical_text_records':len(metadata),
        'historical_database_files':len(databases),'historical_checkpoint_files':len(checkpoints),'historical_binary_caches':len(caches),
        'historical_metadata_hash_values':len(recorded_hashes),'additional_referenced_images':len(extra_images),
        'same_or_transposed_dimensions_checked':selected['same_or_transposed_dimensions_compared'],
        'distinct_full_pixel_decodes':len(pixel_cache),'other_dimensions_excluded_without_decoding':len(image_rows)-selected['same_or_transposed_dimensions_compared'],
        'dimension_exclusion_basis':'Canonical decoded-pixel hash includes native RGB dimensions; native/transposed dimensions are compared, including EXIF rotation.',
        'source_id_comparison':'Case-insensitive NFKC, removing punctuation, underscores, spaces and hyphens; tested against all historical paths, text and SQLite cells.',
        'metadata_errors':len(metadata_errors),'image_errors':len(image_errors),
        'archive_files_listed':len(archives),'archive_photographs_listed':len(candidates),'photographs_extracted_to_disk':1,
        'source_new_candidates':len(remaining),'archive_candidate_images_read_in_memory':len(attempts)+1}
    report={'status':'CONFIRMED_NEW_WITHIN_ALL_AVAILABLE_PROJECT_AND_SNAPSHOT_EVIDENCE','started_at_utc':started,
        'completed_at_utc':datetime.now(timezone.utc).isoformat(),'photo':selected,'coverage':coverage,
        'inference_started':False,'detector_started':False,'training_started':False,'manual_gt_created':False,
        'qualitative_only':True,'future_visual_review_consumes_blind_status':True,
        'claim_scope':'No source-ID, encoded-file SHA-256 or decoded-RGB pixel hash match in the available project, restored historical snapshot, their datasets, model metadata, reviews, evaluations and recorded references. This is an evidence-bounded claim, not a claim about unavailable external systems.'}
    write_json(output/'selection_report.json',report)
    lines=['# Potvrđeni novi qualitative input — inferenca nije pokrenuta','',
        f'- Naziv: `{selected["filename"]}`',f'- Source ID: `{selected["source_id"]}`',
        f'- Dimenzije: {width} × {height} px.',f'- Arhiva: `{selected["archive"]}`',f'- Član arhive: `{selected["name"]}`',
        f'- Zamrznuta datoteka: `{frozen}`',f'- SHA-256 datoteke: `{selected["file_sha256"]}`',
        f'- SHA-256 dekodiranih RGB bajtova: `{selected["pixels"]["native"]["pixel_bytes_sha256"]}`',
        f'- Kanonski pixel SHA-256 (RGB + dimenzije): `{selected["pixels"]["native"]["pixel_sha256"]}`','',
        '## Dokaz i obuhvat', '```json',json.dumps(coverage,indent=2,ensure_ascii=False),'```','',
        'Svi provjereni source-ID/file/pixel hash rezultati za izabranu fotografiju imaju 0 podudaranja.',
        'Puni popisi povijesnih datoteka, hashova, uspoređenih slika i odbijenih arhivskih kandidata nalaze se u `evidence/`.',
        'Iz arhiva je na disk izdvojena točno jedna fotografija. Nema masovnog raspakiravanja, modelskog učitavanja, inference ili ručnih GT oznaka.',
        '', 'Provjera pokriva dostupni repozitorij i verificirani vraćeni povijesni snapshot; ne tvrdi ništa o nedostupnim vanjskim evidencijama.',
        'Ovo je ulaz za kvalitativnu vizualnu provjeru bez GT-a. Nakon budućeg vizualnog pregleda fotografija se više ne smije nazivati slijepim testom.']
    with (output/'REPORT.md').open('x') as handle:handle.write('\n'.join(lines)+'\n')
    files_seal={p.relative_to(output).as_posix():sha(p) for p in output.rglob('*') if p.is_file()}
    write_json(output/'intake_seal.json',{'algorithm':'SHA-256','files':files_seal,'inference_started':False})
    (output/'intake_seal.json.sha256').write_text(sha(output/'intake_seal.json')+'  intake_seal.json\n')
    print(json.dumps({'status':report['status'],'photo':selected,'coverage':coverage,'output':str(output)},indent=2,ensure_ascii=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archives',type=Path,default=ARCHIVES)
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/qualitative_unseen_archive_selection_v1')
    args=parser.parse_args();run(args.archives.resolve(),args.output.resolve(),[ROOT,SNAPSHOT])


if __name__=='__main__':main()
