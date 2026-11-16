"""Stream Kaggle-safe evidence ZIPs and rebuild every file checksum."""
from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile
from .common import rows
from .portable import POLICY, SAFE, path_map, rewrite_payload, safe_component

def _json_bytes(value):
    return (json.dumps(value,indent=2,ensure_ascii=False)+'\n').encode('utf8')

def _build(names,load,samples,metadata,archive,prefix):
    archive=Path(archive);archive.parent.mkdir(parents=True,exist_ok=True);prefix=safe_component(prefix)
    mapping=path_map(names);changed={old:new for old,new in mapping.items() if old!=new}
    files={};partial=archive.with_name(archive.name+'.partial')
    with zipfile.ZipFile(partial,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        def emit(name,data):
            z.writestr(prefix+'/'+name,data);files[name]=hashlib.sha256(data).hexdigest()
        for old,new in mapping.items():emit(new,rewrite_payload(load(old),old,mapping))
        emit('samples.jsonl',''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in samples).encode('utf8'))
        emit('filename_map.json',_json_bytes(dict(policy=POLICY,renamed_files=changed)))
        result=dict(metadata,schema=2,count=len(samples),path_policy=POLICY,renamed_file_count=len(changed),
            task_counts=dict(Counter(r['paper_task'] for r in samples)),files=files)
        z.writestr(prefix+'/bundle.json',_json_bytes(result))
    with zipfile.ZipFile(partial) as z:
        if z.testzip() is not None:raise ValueError('Corrupt trace archive')
        assert all(SAFE.fullmatch(part) for n in z.namelist() for part in PurePosixPath(n).parts)
    partial.replace(archive)
    return dict(rows=len(samples),renamed_files=len(changed),archive=str(archive),
                task_counts=result['task_counts'],mib=archive.stat().st_size/2**20)

def package(source,destination):
    source,destination=Path(source),Path(destination)
    archive=destination if destination.suffix=='.zip' else destination.with_suffix('.zip')
    prefix=destination.stem if destination.suffix=='.zip' else destination.name
    originals={str(r['sample_id']):r for r in rows(source/'run/predictions_apc_vis.jsonl')}
    files={};samples=[]
    for row in rows(source/'samples.jsonl'):
        sid=str(row['sample_id']);trace=source/'run/trace/apc_vis'/sid
        if not sid.isdecimal():raise ValueError(sid)
        if not trace.is_dir():continue
        required=['input.png','scene.json','depth.npy','get_objects_of_interest.json','apc_visual_prompt.png']
        if any(not (trace/name).is_file() for name in required):raise ValueError('Incomplete original trace: '+sid)
        for path in sorted(trace.rglob('*')):
            if path.is_file():files[f'original/{sid}/'+path.relative_to(trace).as_posix()]=path
        samples.append(dict(row,image_path=f'original/{sid}/input.png',trace_path=f'original/{sid}',
            historical_prediction=originals[sid].get('prediction'),historical_status=originals[sid].get('status'),
            historical_correct=originals[sid].get('prediction')==row['answer']))
    if not samples:raise ValueError('No complete original traces')
    return _build(files,lambda name:files[name].read_bytes(),samples,
        dict(sampling='Diagnostic selection enriched for baseline errors; not population accuracy.'),archive,prefix)

def repair_zip(source,destination):
    source,destination=Path(source),Path(destination)
    if source.resolve()==destination.resolve():raise ValueError('Repair to a separate archive first')
    with zipfile.ZipFile(source) as z:
        bundles=[n for n in z.namelist() if PurePosixPath(n).name=='bundle.json']
        if len(bundles)!=1:raise ValueError('Expected one trace bundle')
        prefix=str(PurePosixPath(bundles[0]).parent)
        metadata=json.loads(z.read(bundles[0]));samples=[json.loads(line) for line in
            z.read(prefix+'/samples.jsonl').decode('utf8').splitlines() if line.strip()]
        names=[name for name in metadata['files'] if name not in ('samples.jsonl','filename_map.json')]
        def load(name):
            data=z.read(prefix+'/'+name)
            if hashlib.sha256(data).hexdigest()!=metadata['files'][name]:raise ValueError('Original bundle checksum mismatch: '+name)
            return data
        return _build(names,load,samples,metadata,destination,prefix)
