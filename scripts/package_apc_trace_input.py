"""Package existing evidence (not source code) for a Kaggle Input dataset."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import sys
import zipfile
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from sc203_kat.ablation.common import read, rows, sha, write

def package(source, destination):
    source, destination = Path(source), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    originals = {str(r['sample_id']):r for r in rows(source/'run/predictions_apc_vis.jsonl')}
    manifest = []
    for row in rows(source/'samples.jsonl'):
        sid = str(row['sample_id']); trace = source/'run/trace/apc_vis'/sid
        if not sid.isdecimal(): raise ValueError(sid)
        if not trace.is_dir(): continue
        required = ['input.png','scene.json','depth.npy','get_objects_of_interest.json','apc_visual_prompt.png']
        if any(not (trace/name).is_file() for name in required):
            raise ValueError('Incomplete original trace: ' + sid)
        target = destination/'original'/sid
        shutil.copytree(trace, target, dirs_exist_ok=True)
        item = dict(row, image_path=f'original/{sid}/input.png', trace_path=f'original/{sid}',
                    historical_prediction=originals[sid].get('prediction'),
                    historical_status=originals[sid].get('status'),
                    historical_correct=originals[sid].get('prediction') == row['answer'])
        manifest.append(item)
    if not manifest: raise ValueError('No complete traces found')
    (destination/'samples.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in manifest),encoding='utf8')
    files = {p.relative_to(destination).as_posix():sha(p) for p in sorted(destination.rglob('*'))
             if p.is_file() and p.name != 'bundle.json'}
    write(destination/'bundle.json', dict(schema=1, count=len(manifest),
        task_counts=dict(Counter(r['paper_task'] for r in manifest)), files=files,
        sampling='Diagnostic selection enriched for baseline errors; not population accuracy.'))
    archive = destination.with_suffix('.zip')
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(destination.rglob('*')):
            if p.is_file(): z.write(p, destination.name+'/'+p.relative_to(destination).as_posix())
    print(json.dumps(dict(rows=len(manifest),task_counts=dict(Counter(r['paper_task'] for r in manifest)),
                         archive=str(archive),mib=archive.stat().st_size/2**20),indent=2))

if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();package(a.source,a.output)
