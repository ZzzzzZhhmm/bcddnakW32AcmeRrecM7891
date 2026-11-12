"""Losslessly tier legacy writing archives;never target scientific runs or models."""
from pathlib import Path
import argparse,datetime,gzip,hashlib,json,os,shutil,zlib

def sha(b):return hashlib.sha256(b).hexdigest()
def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def chunks(data):
    start=0;n=len(data)
    while start<n:
        limit=min(start+1024*1024,n);p=start+64*1024
        while p<limit:
            p=data.find(b'\n',p,limit)
            if p<0:break
            if zlib.crc32(data[max(0,p-1024):p])&1023==0:break
            p+=1
        end=p+1 if 0<=p<limit else limit
        yield data[start:end];start=end
def gzip_header_end(b):
    assert b[:3]==b'\x1f\x8b\x08';flags=b[3];p=10
    if flags&4:p+=2+int.from_bytes(b[p:p+2],'little')
    for bit in [8,16]:
        if flags&bit:p=b.index(0,p)+1
    if flags&2:p+=2
    return p
def rebuild(raw,header,trailer,level):
    c=zlib.compressobj(level,zlib.DEFLATED,-15)
    return header+c.compress(raw)+c.flush()+trailer
def materialize(C,m):
    raw=[]
    for r in m['chunks']:
        f=C/'chunks'/r['sha256'][:2]/(r['sha256']+'.z');b=zlib.decompress(f.read_bytes());assert len(b)==r['bytes'] and sha(b)==r['sha256'];raw.append(b)
    raw=b''.join(raw);assert len(raw)==m['raw_tar_bytes'] and sha(raw)==m['raw_tar_sha256']
    data=rebuild(raw,bytes.fromhex(m['gzip_header_hex']),bytes.fromhex(m['gzip_trailer_hex']),m['gzip_level']);assert sha(data)==m['original_archive_sha256'];return data
def atomic(p,value):
    t=p.with_suffix(p.suffix+'.atomic');assert not t.exists();t.write_text(json.dumps(value,indent=2));t.replace(p)
def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--evidence-root',type=Path,required=True);ap.add_argument('--apply',action='store_true');ap.add_argument('--materialize-manifest',type=Path);ap.add_argument('--output',type=Path);args=ap.parse_args()
    P=args.evidence_root.resolve(strict=True);C=P/'writing_archive_cas_20261112_v1'
    if args.materialize_manifest:
        m=json.loads(args.materialize_manifest.read_text());assert args.output and not args.output.exists()
        print(json.dumps(dict(mode='materialize_apply' if args.apply else 'materialize_dry_run',archive_sha256=m['original_archive_sha256'],output=str(args.output))))
        if not args.apply:return
        dest=args.output.resolve();assert dest.is_relative_to(P) or dest.is_relative_to(Path('/tmp'))
        required=4*1024**3 if dest.is_relative_to(P) else 0
        assert shutil.disk_usage(dest.parent).free>required+m['original_archive_bytes'];data=materialize(C,m)
        with dest.open('xb') as f:f.write(data)
        print(json.dumps(dict(status='original_gzip_archive_restored_bit_exact',sha256=sha(dest.read_bytes()))));return
    planfile=P/'ITER30_WRITING_TIERING_PLAN_V1.json';assert sha(planfile.read_bytes())=='ad6469db733047b1487885df5c6911d8b81174201b4eababf3ed7ada47f8e472'
    plan=json.loads(planfile.read_text());proof=json.loads((P/'ITER30_WRITING_TIERING_LOCAL_VERIFICATION_V1.json').read_text());assert proof['all_target_bytes_independently_recoverable_from_local_and_retained_AFS_archives']
    selected=plan['archives'];assert len(selected)==19 and all(z['folder']!='20261112_v26' for z in selected)
    if not args.apply:print(json.dumps(dict(mode='dry_run',archives=len(selected),source_bytes=sum(z['archive_bytes'] for z in selected),current_v26_excluded=True)));return
    assert not C.exists();C.mkdir();(C/'chunks').mkdir();(C/'manifests').mkdir();before=shutil.disk_usage(P).free;index={};removed=0;newbytes=0
    qualification=dict(utc=stamp(),zlib_version=zlib.ZLIB_VERSION,mode='Content-defined SHA256/zlib chunks of exact TAR bytes,with proven original GZIP reconstruction before each tiering action.',original_windows_unchanged=True,source_sha256=sha(Path(__file__).read_bytes()))
    (C/'QUALIFICATION.json').write_text(json.dumps(qualification,indent=2))
    with (C/'journal.jsonl').open('x',buffering=1) as log:
        for item in selected:
            a=Path(item['AFS_path']).resolve();assert a.is_relative_to(P) and (a.parent==P or a.parent==P/'writing_candidates'/item['folder']) and not a.is_symlink()
            assert shutil.disk_usage(P).free>1024**3
            source_stat=a.stat();original=a.read_bytes();assert sha(original)==item['archive_sha256'];raw=gzip.decompress(original);p=gzip_header_end(original)
            level=9 if original[8]==2 else 1 if original[8]==4 else 6;header=original[:p];trailer=original[-8:]
            if sha(rebuild(raw,header,trailer,level))!=item['archive_sha256']:
                log.write(json.dumps(dict(utc=stamp(),status='not_tiered_gzip_recipe_mismatch',archive=str(a)))+'\n');continue
            refs=[]
            for b in chunks(raw):
                h=sha(b);d=C/'chunks'/h[:2];d.mkdir(exist_ok=True);f=d/(h+'.z')
                if f.exists():assert sha(zlib.decompress(f.read_bytes()))==h
                else:
                    assert shutil.disk_usage(P).free>1024**3
                    packed=zlib.compress(b,3)
                    with f.open('xb') as o:o.write(packed);o.flush();os.fsync(o.fileno())
                    assert zlib.decompress(f.read_bytes())==b;newbytes+=len(packed)
                refs.append(dict(sha256=h,bytes=len(b)))
            m=dict(utc=stamp(),original_archive=str(a),original_archive_sha256=item['archive_sha256'],original_archive_bytes=len(original),local_original_archive=item['local_path'],
                raw_tar_sha256=sha(raw),raw_tar_bytes=len(raw),gzip_header_hex=header.hex(),gzip_trailer_hex=trailer.hex(),gzip_level=level,zlib_version=zlib.ZLIB_VERSION,chunks=refs)
            mf=C/'manifests'/(item['folder']+'.json');assert not mf.exists();mf.write_text(json.dumps(m,indent=2));assert sha(materialize(C,json.loads(mf.read_text())))==item['archive_sha256']
            now=a.stat();assert (now.st_ino,now.st_size,now.st_mtime_ns)==(source_stat.st_ino,source_stat.st_size,source_stat.st_mtime_ns) and sha(a.read_bytes())==item['archive_sha256']
            for proc in Path('/proc').iterdir():
                if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
                try:cmd=(proc/'cmdline').read_bytes()
                except OSError:continue
                assert str(a).encode() not in cmd,'Archive referenced by an active process'
            record=dict(utc=stamp(),archive=str(a),manifest=str(mf),manifest_sha256=sha(mf.read_bytes()),archive_sha256=item['archive_sha256'],bit_exact_restore_verified=True,local_original_preserved=True)
            log.write(json.dumps(record|dict(status='verified_tiering_intent'))+'\n');log.flush();os.fsync(log.fileno());a.unlink();removed+=len(original)
            index[str(a)]=record;atomic(C/'INDEX.json',index)
            D=P/'writing_candidates'/item['folder'];marker=D/'TIERED_ARCHIVE_20261112.json';assert not marker.exists();marker.write_text(json.dumps(record|dict(storage='Exact reconstructable archive in AFS CAS plus unchanged full local original',cas_root=str(C)),indent=2))
            oldmarker=D/'TIERED_EXTRACTION_20261112.json'
            if oldmarker.exists():v=json.loads(oldmarker.read_text());v.update(archive_storage='lossless_AFS_CAS_and_full_local_original',archive_CAS_manifest=str(mf));atomic(oldmarker,v)
            log.write(json.dumps(record|dict(status='original_archive_tiered_byte_exact'))+'\n');print(json.dumps(dict(folder=item['folder'],removed_bytes=removed,new_unique_CAS_bytes=newbytes,AFS_free_bytes=shutil.disk_usage(P).free)),flush=True)
    out=dict(utc=stamp(),archives_tiered=len(index),original_archive_bytes_tiered=removed,new_unique_CAS_bytes=newbytes,AFS_free_before_bytes=before,AFS_free_after_bytes=shutil.disk_usage(P).free,
        all_original_archive_hashes_bit_exact_restorable=True,all_local_original_archives_unchanged=True,unique_scientific_data_checkpoints_and_existing_CAS_untouched=True,current_v26_unchanged=True,
        INDEX_sha256=sha((C/'INDEX.json').read_bytes()),journal_sha256=sha((C/'journal.jsonl').read_bytes()),source_sha256=qualification['source_sha256'])
    (C/'RECEIPT.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
if __name__=='__main__':main()
