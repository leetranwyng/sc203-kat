"""Run staged, resumable APC perception ablations on a fixed diagnostic cohort."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from .common import read,write,rows,sha,inside,VARIANTS

def find_bundle(value):
    if value:
        path=Path(value)
        if (path/'bundle.json').is_file():return path
        matches=list(path.rglob('bundle.json'))
    else:matches=list(Path('/kaggle/input').rglob('bundle.json'))
    matches=[p for p in matches if (p.parent/'samples.jsonl').is_file() and (p.parent/'original').is_dir()]
    if len(matches)!=1:raise ValueError('Set TRACE_INPUT to the extracted sc203-apc-trace120 folder; found '+str(matches))
    return matches[0].parent

def prepare(cfg,root):
    bundle=find_bundle(cfg.get('trace_input'));out=Path(cfg['output_dir']);out.mkdir(parents=True,exist_ok=True)
    metadata=read(bundle/'bundle.json')
    for name,digest in metadata['files'].items():
        if sha(inside(bundle,name))!=digest:raise ValueError('Trace input checksum mismatch: '+name)
    items=rows(bundle/'samples.jsonl')
    if cfg.get('sample_ids'):
        selected=set(map(str,cfg['sample_ids']));items=[r for r in items if str(r['sample_id']) in selected]
        if {str(r['sample_id']) for r in items}!=selected:raise ValueError('Requested IDs absent from trace bundle')
    if not items:raise ValueError('Empty trace cohort')
    if not cfg['variants'] or any(v not in VARIANTS for v in cfg['variants']):raise ValueError('Unknown/empty variants')
    if len(cfg['variants'])!=len(set(cfg['variants'])):raise ValueError('Duplicate variants')
    if len(items)!=len({str(r['sample_id']) for r in items}):raise ValueError('Duplicate sample IDs')
    pins=read(root/'configs/ablations/upstreams.json')
    cfg['pins']=pins
    source_digest=hashlib.sha256()
    for p in sorted((root/'src/sc203_kat/ablation').glob('*.py')):source_digest.update(p.read_bytes())
    for base in (root/'profiles/apc_vis',root/'configs/ablations'):
        for p in sorted(base.rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts:source_digest.update(p.read_bytes())
    inputs={k:sha(v) for k,v in cfg.get('weight_inputs',{}).items() if v}
    identity=dict(bundle_sha=sha(bundle/'bundle.json'),sample_ids=[r['sample_id'] for r in items],
        variants=cfg['variants'],pins=pins,source_sha=source_digest.hexdigest(),weight_inputs=inputs,
        sam3_threshold=cfg['sam3_threshold'],oneref_min_iou=cfg['oneref_min_iou'],seed=cfg['seed'],
        model_input=cfg.get('model_input'))
    resume=cfg.get('resume_input')
    if resume:
        source=Path(resume)
        if read(source/'identity.json')!=identity:raise ValueError('Resume identity differs; do not mix ablations')
        if not (out/'identity.json').exists():shutil.copytree(source,out,dirs_exist_ok=True)
    if (out/'identity.json').exists() and read(out/'identity.json')!=identity:
        raise ValueError('Output belongs to another cohort/config/code; choose a new output_dir')
    write(out/'identity.json',identity)
    write(out/'ablation_config.json',cfg)
    for row in items:
        source=inside(bundle,row['trace_path']);dest=out/'original'/str(row['sample_id'])
        shutil.copytree(source,dest,dirs_exist_ok=True)
        row['trace_path']=str(dest);row['image_path']=str(dest/'input.png')
    (out/'samples.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in items),encoding='utf8')
    cfg['needed_models']=[k for k in ('sam3','oneref','oa2') if any(k in v for v in cfg['variants'])]
    cfg['need_oa1']=any(VARIANTS[v][0]!='original' for v in cfg['variants'])
    cfg['need_sam1']=any(VARIANTS[v][0]=='oneref' for v in cfg['variants'])
    kinds=list(dict.fromkeys(VARIANTS[v][0] for v in cfg['variants']))
    for kind in ('sam3','oneref'):
        jobs=[]
        if kind in cfg['needed_models']:
            for row in items:
                names=read(Path(row['trace_path'])/'get_objects_of_interest.json')['result'][0]
                for idx,name in enumerate(names):
                    key=f'{row["sample_id"]}/object_{idx}'
                    jobs.append(dict(key=key,image=row['image_path'],text=name,
                                     output=str(out/'perception'/kind/key)))
        write(out/f'jobs_{kind}.json',jobs)
    write(out/'protocol.json',dict(design='frozen-upstream controlled trace replay',
        controls=['object list','reference viewer','Depth Pro map','APC focal length and unprojection',
                  'Qwen2.5-VL-7B FP16 weights/prompts','renderer'],
        selection='OneRef REC full image + original object description; maximum IoU over fixed top-5 candidates',
        sam3='replace proposals AND masks; retain APC crop refinement unless OneRef also selected',
        orientation='OA2 argmax including roll; save distributions and symmetry; no symmetry oracle',
        gold_usage='scoring only, never supplied to component adapters',
        sampling=metadata['sampling'],front_behind_visibility='kept separate; no question rewriting',
        failure_scoring='error and invalid count as wrong; unattempted cases reported separately'))
    return items

def run(cfg,root):
    if sys.platform!='linux':raise RuntimeError('Run on Kaggle Linux T4 x2')
    items=prepare(cfg,root);out=Path(cfg['output_dir']);work=Path(cfg['runtime_dir']);work.mkdir(parents=True,exist_ok=True)
    config=work/'ablation.json';write(config,cfg)
    from .setup import setup_apc,setup_extra
    context=setup_apc(cfg,root)
    cfg['repos'],pythons=setup_extra(cfg,root,context);write(config,cfg)
    env=dict(os.environ,**context['env']);env['HF_HUB_OFFLINE']='0';env['TRANSFORMERS_OFFLINE']='0'
    def launch(label,py,module,args,extra=False):
        if shutil.disk_usage(out).free<cfg['disk_min_free_gib']*2**30:raise RuntimeError('Low disk; checkpoint retained')
        child=env.copy()
        if extra:
            child['PYTHONPATH']=str(root/'src')
            child['CUDA_VISIBLE_DEVICES']=context['env']['CUDA_VISIBLE_DEVICES'].split(',')[0]
            child.pop('CUDA_HOME',None);child.pop('CUDA_PATH',None)
            # Do not inject the APC CUDA 12.4 toolkit into cu118/cu126 workers.
            child['LD_LIBRARY_PATH']=os.environ.get('LD_LIBRARY_PATH','')
        command=[py,'-u','-m',module,*map(str,args)]
        log=out/'logs'/f'{label}.log';log.parent.mkdir(exist_ok=True)
        write(out/'state.json',dict(stage=label,status='running',updated=time.time()))
        with log.open('a',encoding='utf8') as f:
            process=subprocess.Popen(command,env=child,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            started=time.monotonic();offset=0
            while process.poll() is None:
                time.sleep(5)
                with log.open(encoding='utf8',errors='replace') as reader:
                    reader.seek(offset);chunk=reader.read();offset=reader.tell()
                if chunk:print(chunk[-12000:],end='',flush=True)
                if time.monotonic()-started>cfg['phase_timeout_seconds']:
                    import signal
                    os.killpg(process.pid,signal.SIGTERM);process.wait(timeout=30)
                    raise TimeoutError(label+' exceeded phase timeout; outputs retained')
            with log.open(encoding='utf8',errors='replace') as reader:
                reader.seek(offset);print(reader.read()[-12000:],end='',flush=True)
        if process.returncode:raise RuntimeError(label+' failed; inspect '+str(log))
    failures=[];fatal=None
    try:
        launch('assets',context['py'],'sc203_kat.ablation.assets',[config])
        cfg=read(config)
        for kind in ('sam3','oneref'):
            if kind in cfg['needed_models']:
                try:launch(kind,pythons[kind],'sc203_kat.ablation.perception',[config,kind],extra=True)
                except Exception as exc:failures.append(dict(stage=kind,error=str(exc)))
        launch('geometry',context['py'],'sc203_kat.ablation.apc_worker',[config,'geometry'])
        if 'oa2' in cfg['needed_models']:
            jobs=[]
            kinds=list(dict.fromkeys(VARIANTS[v][0] for v in cfg['variants'] if VARIANTS[v][1]=='v2'))
            for kind in kinds:
                for row in items:
                    g=read(out/'geometry'/kind/row['sample_id']/'result.json')
                    if g['status']!='ok':continue
                    for obj in g['objects']:
                        key=f'{kind}/{row["sample_id"]}/{obj["instance_id"]}'
                        jobs.append(dict(key=key,image=row['image_path'],bbox=obj['selected_box'],text=obj['name'],
                                         output=str(out/'perception/oa2'/key)))
            write(out/'jobs_oa2.json',jobs)
            try:launch('oa2',pythons['oa2'],'sc203_kat.ablation.perception',[config,'oa2'],extra=True)
            except Exception as exc:failures.append(dict(stage='oa2',error=str(exc)))
        launch('answers',context['py'],'sc203_kat.ablation.apc_worker',[config,'answers'])
    except BaseException as exc:
        fatal=dict(error_type=type(exc).__name__,message=str(exc))
        raise
    finally:
        from .report import build
        build(out)
        write(out/'state.json',dict(status='failed' if fatal else 'finished_with_component_errors' if failures else 'finished',
                                   failures=failures,fatal=fatal))
    if failures:raise RuntimeError('Some auxiliary processes failed; report includes errors as wrong. See state.json/logs.')
    if cfg['mode']=='sanity' and not read(out/'summary.json')['technical_pass']:
        raise RuntimeError('Sanity has an error/invalid/missing render. Inspect index.html and logs before trace120.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True)
    a=p.parse_args();run(read(a.config),Path(__file__).resolve().parents[3])
