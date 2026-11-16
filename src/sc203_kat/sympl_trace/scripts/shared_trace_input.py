"""Read the shared sc203-kat APC bundle as a read-only directory or ZIP."""
import argparse
import hashlib
import io
import json
import stat
import zipfile
from pathlib import Path, PurePosixPath
import numpy as np
from PIL import Image
from trace_runtime import atomic_json

def safe_name(name):
    p=PurePosixPath(name)
    if not name or p.is_absolute() or '\\' in name or ':' in name or '..' in p.parts:
        raise ValueError('Unsafe trace bundle path: '+name)
    return p.as_posix()

def pixel_sha(image):
    image=image.convert('RGB')
    return hashlib.sha256(str(image.size).encode()+image.tobytes()).hexdigest()

class TraceBundle:
    def __init__(self,path):
        self.path=Path(path).resolve();self.zip=None;self.prefix='';self._names=None
        if self.path.is_file():
            self.zip=zipfile.ZipFile(self.path)
            names=set(self.zip.namelist())
            roots=[n[:-len('bundle.json')] for n in names if (n=='bundle.json' or n.endswith('/bundle.json')) and n[:-len('bundle.json')]+'samples.jsonl' in names]
            if len(roots)!=1:raise ValueError('ZIP must contain exactly one trace bundle root: '+str(roots))
            self.prefix=roots[0]
            for info in self.zip.infolist():
                safe_name(info.filename.rstrip('/'))
                if stat.S_ISLNK(info.external_attr>>16):raise ValueError('Symlinks are not accepted in trace ZIP')
        self.metadata=json.loads(self.read('bundle.json'))
        self.metadata_sha=hashlib.sha256(self.read('bundle.json')).hexdigest()
        self.rows=[json.loads(x) for x in self.read('samples.jsonl').decode('utf8').splitlines()]
        if self.metadata.get('count')!=120 or len(self.rows)!=120:
            raise ValueError('Expected the shared 120-sample diagnostic trace cohort')

    def close(self):
        if self.zip:self.zip.close()
    def __enter__(self):return self
    def __exit__(self,*args):self.close()

    def read(self,name):
        name=safe_name(name)
        if self.zip:return self.zip.read(self.prefix+name)
        target=(self.path/name).resolve();target.relative_to(self.path)
        if not target.is_file():raise FileNotFoundError(target)
        return target.read_bytes()

    def names(self):
        if self._names is None:
            self._names=([n[len(self.prefix):] for n in self.zip.namelist() if n.startswith(self.prefix) and not n.endswith('/')]
                if self.zip else [p.relative_to(self.path).as_posix() for p in self.path.rglob('*') if p.is_file()])
        return self._names

    def validate(self,expected=None,verify_all=True):
        hashes=self.metadata.get('files',{})
        for row in self.rows:
            safe_name(row['image_path']);safe_name(row['trace_path'])
            if row['image_path'] not in hashes:raise ValueError('Image absent from bundle checksums: '+row['image_path'])
        checks=hashes if verify_all else {r['image_path']:hashes[r['image_path']] for r in self.rows}
        for name,digest in checks.items():
            if hashlib.sha256(self.read(name)).hexdigest()!=digest:
                raise ValueError('Trace input checksum mismatch: '+name)
        expected={str(r['sample_id']):r for r in expected} if expected is not None else None
        if len({str(r['sample_id']) for r in self.rows})!=120:raise ValueError('Duplicate sample IDs')
        counts={task:sum(r['paper_task']==task for r in self.rows) for task in ('left_right','visibility','facing')}
        if counts!={'left_right':40,'visibility':40,'facing':40}:raise ValueError('Task counts differ: '+str(counts))
        if expected is not None and {str(r['sample_id']) for r in self.rows}!=set(expected):raise ValueError('Cohort IDs differ from notebook')
        pixels={}
        for row in self.rows:
            sid=str(row['sample_id']);im=Image.open(io.BytesIO(self.read(row['image_path']))).convert('RGB')
            pixels[sid]=pixel_sha(im)
            if expected is not None:
                ref=expected[sid]
                for key in ('question_id','source_question','source_choices','question','choices','answer','paper_task'):
                    if row[key]!=ref[key]:raise ValueError('Cohort metadata mismatch: '+sid+' / '+key)
                if pixels[sid]!=ref['pixel_sha256']:raise ValueError('Cohort RGB mismatch: '+sid)
            if row['answer'] not in row['choices']:raise ValueError('Invalid answer: '+sid)
        return dict(status='PASS',source=str(self.path),format='zip' if self.zip else 'directory',
            bundle_sha256=self.metadata_sha,rows=120,unique_RGB_images=len(set(pixels.values())),
            task_counts=counts,checksummed_files=len(checks),pixel_sha256=pixels)

    def image_path(self,row,work):
        if not self.zip:
            p=(self.path/safe_name(row['image_path'])).resolve();p.relative_to(self.path)
            return p
        p=Path(work)/'shared_inputs'/(str(row['sample_id'])+'.png');p.parent.mkdir(parents=True,exist_ok=True)
        data=self.read(row['image_path'])
        if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest()!=hashlib.sha256(data).hexdigest():
            tmp=p.with_suffix('.tmp');tmp.write_bytes(data);tmp.replace(p)
        return p

    def apc_record(self,row,out,create_previews=True):
        sid=str(row['sample_id']);folder=row['trace_path'].rstrip('/')+'/'
        names=set(self.names())
        record=dict(prediction=row.get('historical_prediction'),correct=row.get('historical_correct'),
                    status=row.get('historical_status'),previews=[],selected_boxes={},orientation_angles=[])
        if record['prediction'] is None or record['correct'] is None:raise ValueError('Missing APC baseline result: '+sid)
        if record['correct']!=(record['prediction']==row['answer']):raise ValueError('APC correctness mismatch: '+sid)
        if folder+'scene.json' in names:record['scene']=json.loads(self.read(folder+'scene.json'))
        if folder+'reference_viewer.json' in names:record['reference_viewer']=json.loads(self.read(folder+'reference_viewer.json'))
        for name in sorted(n for n in names if n.startswith(folder) and '/' not in n[len(folder):]):
            stem=PurePosixPath(name).name
            if stem.startswith('object_') and stem.endswith('.name'):
                box=name[:-len('.name')]+'_selected_box.npy'
                if box in names:record['selected_boxes'][self.read(name).decode().strip()]=np.load(io.BytesIO(self.read(box)),allow_pickle=False).tolist()
            if stem.startswith('orientation_angles_') and stem.endswith('.json'):
                record['orientation_angles'].append(json.loads(self.read(name)))
            if create_previews and stem.endswith('.png') and stem not in ('input.png','visual_prompt_temp.png'):
                dest='apc_previews/'+sid+'/'+PurePosixPath(stem).stem+'.jpg'
                p=Path(out)/dest;p.parent.mkdir(parents=True,exist_ok=True)
                image=Image.open(io.BytesIO(self.read(name))).convert('RGB');image.thumbnail((600,450))
                image.save(p,format='JPEG',quality=70)
                record['previews'].append(dict(path=dest,label=stem+' (APC thumbnail; original in shared dataset)'))
        return record

def find_bundle(value=None,input_root='/kaggle/input'):
    root=Path(value) if value else Path(input_root)
    if root.is_file():
        with TraceBundle(root):pass
        return root.resolve()
    if (root/'bundle.json').is_file() and (root/'samples.jsonl').is_file() and (root/'original').is_dir():
        candidates=[root]
    else:
        candidates=[p.parent for p in root.rglob('bundle.json') if (p.parent/'samples.jsonl').is_file() and (p.parent/'original').is_dir()]
        candidates+=list(root.rglob('sc203_apc_trace120.zip'))
    distinct={}
    for path in sorted(candidates,key=lambda p:(p.is_file(),str(p))):
        with TraceBundle(path) as bundle:distinct.setdefault(bundle.metadata_sha,path.resolve())
    if len(distinct)!=1:
        raise ValueError('Add the existing sc203-apc-trace120 Kaggle Dataset, or set TRACE_INPUT to its ZIP/extracted folder. Found: '+str(list(map(str,distinct.values()))))
    return next(iter(distinct.values()))

def preflight(cfg):
    work=Path(cfg['work']);out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=True)
    expected=[json.loads(x) for x in (work/'paired_manifest.jsonl').read_text(encoding='utf8').splitlines()]
    source=None
    try:
        source=find_bundle(cfg.get('trace_input'))
        with TraceBundle(source) as bundle:report=bundle.validate(expected,verify_all=True)
    except Exception as exc:
        atomic_json(out/'input_validation.json',dict(status='FAIL',source=str(source),error=repr(exc)));raise
    atomic_json(out/'input_validation.json',report)
    print('Shared APC dataset preflight PASS:',report['rows'],'rows,',report['checksummed_files'],'checksummed files; source:',source,flush=True)
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('config');args=parser.parse_args()
    preflight(json.loads(Path(args.config).read_text()))
