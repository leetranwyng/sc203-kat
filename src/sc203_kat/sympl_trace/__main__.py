"""Isolated trace command: python -m sc203_kat.sympl_trace --config config.json."""
import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import traceback
import zipfile
from pathlib import Path

from .instrumentation import instrument

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parents[2]
PROFILE = ROOT/'profiles/sympl'


def digest_sources():
    digest = hashlib.sha256()
    for base in (PACKAGE, PROFILE):
        for p in sorted(base.rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc':
                digest.update(p.relative_to(ROOT).as_posix().encode())
                digest.update(p.read_bytes())
    return digest.hexdigest()


def stage(name, ns):
    p = PROFILE/name
    print('[sympl-trace] shared profile:', name, flush=True)
    ns['__file__'] = str(p)
    exec(compile(p.read_text(encoding='utf8'), str(p), 'exec'), ns)


def archive_output(out):
    """Archive partial evidence even when inference or report generation failed."""
    out = Path(out)
    target = out.parent/(out.name+'_downloads')
    target.mkdir(parents=True, exist_ok=True)
    for p in target.glob('traces_part_*.zip'):
        p.unlink()
    archive = None
    size = part = 0
    try:
        for p in sorted(out.rglob('*')):
            if not p.is_file(): continue
            if archive is None or size+p.stat().st_size > 64*2**20:
                if archive: archive.close()
                part += 1
                size = 0
                archive = zipfile.ZipFile(target/f'traces_part_{part:03d}.zip','w',zipfile.ZIP_DEFLATED)
            archive.write(p,p.relative_to(out))
            size += p.stat().st_size
    finally:
        if archive: archive.close()
    print('Download evidence:', target, flush=True)


def report(cfg, python, env, worker_exit, sanity=False):
    """Record analysis errors separately; never replace the worker exception."""
    out = Path(cfg['output'])
    config = out/'run_config.json'
    config.write_text(json.dumps(cfg,indent=2),encoding='utf8')
    scripts = Path(cfg['work'])/'scripts'
    command = [python,str(scripts/'analyze120.py'),str(config)]
    result = subprocess.run(command,cwd=cfg['repo'],env=env,check=False)
    state = dict(worker_exit=worker_exit,analysis_exit=result.returncode)
    if sanity:
        gate = subprocess.run([python,str(scripts/'sanity120.py'),str(out),str(worker_exit)],
                              cwd=cfg['repo'],env=env,check=False)
        state['sanity_exit'] = gate.returncode
    (out/'execution_status.json').write_text(json.dumps(state,indent=2),encoding='utf8')
    return state


def worker(cfg, ns):
    out = Path(cfg['output']);out.mkdir(parents=True,exist_ok=True)
    config = out/'run_config.json'
    config.write_text(json.dumps(cfg,indent=2),encoding='utf8')
    python, env = ns['PY'], ns['ENV']
    scripts = Path(cfg['work'])/'scripts'
    subprocess.run([python,str(scripts/'prepare_shared120.py'),str(config)],
                   cwd=cfg['repo'],env=env,check=True)
    with (out/'worker.log').open('a',encoding='utf8') as log:
        child = subprocess.Popen([python,'-u',str(scripts/'run120.py'),str(config)],
            cwd=cfg['repo'],env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
            text=True,encoding='utf8',errors='replace')
        try:
            for line in child.stdout:
                print(line,end='',flush=True);log.write(line);log.flush()
            rc = child.wait()
        finally:
            if child.poll() is None:
                child.terminate()
                try:child.wait(timeout=30)
                except subprocess.TimeoutExpired:child.kill();child.wait()
    state = report(cfg,python,env,rc,sanity=cfg['smoke_only'])
    return not any(state.values())


def run(config):
    if platform.system() != 'Linux':
        raise RuntimeError('Inference targets Kaggle Linux with T4 x2.')
    mode = config.get('mode','sanity')
    if mode not in ('sanity','full'): raise ValueError('mode must be sanity or full')
    out=Path(config.get('output_dir',f'/kaggle/working/sc203-kat/sympl_trace_{mode}'))
    work=Path(config.get('runtime_dir','/kaggle/temp/sc203-kat/sympl_trace'))
    options = dict(config.get('method_options') or {})
    options = {
                   **dict(category_policy='dataset_mapping',quantization=None,max_pixels=None,
                   max_new_tokens=1024,retry_errors=False,min_free_gib=1.5,max_errors=5,seed=42),**options}
    if options['category_policy'] not in ('dataset_mapping','model'):raise ValueError('Invalid category_policy')
    if options['quantization'] not in (None,'4bit'):raise ValueError('Invalid quantization')
    from sc203_kat.runtime.kaggle import ensure_submodule,source_metadata
    ensure_submodule(ROOT,'sympl')
    ns=dict(__name__='__sympl_trace__',PROFILE_DIR=PROFILE,REPOSITORY_ROOT=ROOT,
        UPSTREAM_KEY='sympl',RUNTIME_DIR=str(work),OUTPUT_DIR=str(out),METHOD='sympl',
        SCOPE='paper_tasks_visibility',SANITY=mode=='sanity',METHOD_OPTIONS={},
        MODEL_INPUT=config.get('model_input'),DATASET_INPUT=None,RESUME_INPUT=config.get('resume_input'))
    stage('01_prepare.py',ns)
    shutil.copytree(PACKAGE/'scripts',work/'scripts',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copy2(PACKAGE/'paired_manifest.jsonl',work/'paired_manifest.jsonl')
    instrument(work/'repo')
    cfg=ns['CFG']
    cfg.update(options,trace_input=config.get('trace_input'),smoke_only=mode=='sanity',
               runtime_sha=digest_sources(),repo_commit=source_metadata(ROOT,'sympl')['commit'])
    ns['CONFIG'].write_text(json.dumps(cfg),encoding='utf8')
    (out/'source_checkout.json').write_text(json.dumps(source_metadata(ROOT,'sympl'),indent=2),encoding='utf8')
    sys.path.insert(0,str(work/'scripts'))
    from shared_trace_input import preflight
    try:
        preflight(cfg)
        stage('02_environment.py',ns)
        if cfg['quantization']=='4bit':
            subprocess.run([str(ns['UV']),'pip','install','--python',ns['PY'],'bitsandbytes==0.48.2'],env=ns['ENV'],check=True)
        stage('03_native_assets.py',ns)
        if cfg['model_sha']!=cfg['model_revision']:raise ValueError('Model checkpoint revision mismatch')
        ns['ENV'].update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
        (out/'environment.txt').write_text(subprocess.check_output([str(ns['UV']),'pip','freeze','--python',ns['PY']],env=ns['ENV'],text=True),encoding='utf8')
        if mode=='full' and config.get('run_smoke_first',True):
            smoke=out/'smoke';smoke.mkdir(exist_ok=True)
            if not worker(dict(cfg,output=str(smoke),smoke_only=True,sanity=True),ns):
                raise RuntimeError('Sanity failed; full 120 did not start. See smoke/worker.log and sanity_report.json.')
        if not worker(cfg,ns):
            raise RuntimeError('Trace run failed. Inspect worker.log, execution_status.json and per-sample result.json.')
    except BaseException:
        (out/'launcher_error.txt').write_text(traceback.format_exc(),encoding='utf8')
        raise
    finally:
        archive_output(out)


def recover(output):
    """Repair gallery and expose root errors from an old run, without loading models."""
    out=Path(output)
    cfg=json.loads((out/'run_config.json').read_text(encoding='utf8'))
    sys.path[:0]=[str(PACKAGE/'scripts'),str(PROFILE/'overlay/scripts')]
    for p in sorted((out/'traces').glob('*/result.json')):
        r=json.loads(p.read_text(encoding='utf8'))
        if r.get('status')=='error':
            print(r.get('sample_id'),r.get('failed_stage'),r.get('error'),flush=True)
            print(r.get('traceback','No traceback saved'),flush=True)
    try:
        from analyze120 import export
        export(out,cfg['repo'])
    finally:archive_output(out)


def main():
    parser=argparse.ArgumentParser()
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--config')
    group.add_argument('--recover-output')
    args=parser.parse_args()
    if args.recover_output:recover(args.recover_output)
    else:run(json.loads(Path(args.config).read_text(encoding='utf8')))


if __name__=='__main__':main()
