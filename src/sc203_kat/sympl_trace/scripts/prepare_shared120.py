"""Prepare SymPL from the same read-only APC trace input used by sc203-kat."""
import hashlib,json,sys
from pathlib import Path
from shared_trace_input import find_bundle,TraceBundle
from trace_runtime import atomic_json

def main(config_path):
    cfg=json.loads(Path(config_path).read_text());work=Path(cfg['work']);out=Path(cfg['output'])
    out.mkdir(parents=True,exist_ok=True)
    expected=[json.loads(x) for x in (work/'paired_manifest.jsonl').read_text(encoding='utf8').splitlines()]
    source=find_bundle(cfg.get('trace_input'))
    with TraceBundle(source) as bundle:
        report=bundle.validate(expected,verify_all=False)
        atomic_json(out/'input_validation.json',report)
        by_id={str(r['sample_id']):r for r in bundle.rows}
        rows=[]
        for ref in expected:
            original=by_id[str(ref['sample_id'])]
            row=dict(ref)
            row['apc']=bundle.apc_record(original,out,create_previews=False)
            rows.append(row)
        if cfg.get('smoke_only'):
            selected=[]
            for task in ('left_right','visibility','facing'):
                group=[r for r in rows if r['paper_task']==task]
                good=[r for r in group if r['apc']['correct']];bad=[r for r in group if not r['apc']['correct']]
                first=good[0] if good else group[0]
                second=next((r for r in bad if r['sample_id']!=first['sample_id']),None)
                if second is None:second=next(r for r in group if r['sample_id']!=first['sample_id'])
                selected.extend([first,second])
            rows=selected
        for row in rows:
            row['image_path']=str(bundle.image_path(by_id[str(row['sample_id'])],work))
            row['apc']=bundle.apc_record(by_id[str(row['sample_id'])],out,create_previews=True)
        identity={k:cfg[k] for k in ('runtime_sha','model_id','model_revision','data_revision',
            'category_policy','quantization','max_pixels','max_new_tokens','smoke_only','seed')}
        identity['trace_bundle_sha256']=bundle.metadata_sha
        identity['trace_samples_sha256']=hashlib.sha256(bundle.read('samples.jsonl')).hexdigest()
        identity['rows']=[{k:r[k] for k in ('sample_id','question','choices','pixel_sha256')} for r in rows]
    fingerprint=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    if (out/'manifest.json').exists():
        assert json.loads((out/'manifest.json').read_text())['fingerprint']==fingerprint,'Incompatible resume: use a new output folder'
    atomic_json(out/'manifest.json',dict(fingerprint=fingerprint,identity=identity,selected_rows=len(rows)))
    (out/'samples.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf8')
    print('Shared APC trace input:',source,'; selected:',len(rows),'(historical APC results imported; SymPL perception recomputed)',flush=True)

if __name__=='__main__':main(sys.argv[1])
