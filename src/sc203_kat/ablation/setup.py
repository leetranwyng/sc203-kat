"""Reuse the successful APC setup; auxiliary environments never alter it."""
import json
import os
from pathlib import Path
import subprocess
import sys
import shutil
from .common import read,write

def setup_apc(cfg,root):
    from sc203_kat.runtime.kaggle import ensure_submodule,source_metadata
    ensure_submodule(root,'apc_vis')
    profile=root/'profiles/apc_vis';work=Path(cfg['runtime_dir']);out=Path(cfg['output_dir'])
    # Resume the precise native-source commits from the previous session, too.
    prior=out/'runtime_audit/source_lock.json'
    if prior.is_file() and not (work/'source_lock.json').exists():
        shutil.copy2(prior,work/'source_lock.json')
    ns=dict(__name__='__apc_ablation_setup__',PROFILE_DIR=profile,REPOSITORY_ROOT=root,
            UPSTREAM_KEY='apc_vlm',RUNTIME_DIR=str(work),OUTPUT_DIR=str(out),METHOD_OPTIONS={},
            SANITY=False,MODEL_INPUT=cfg.get('model_input'),DATASET_INPUT=None,RESUME_INPUT=None)
    def stage(name):
        p=profile/name;print('[setup]',name,flush=True)
        exec(compile(p.read_text(encoding='utf8'),str(p),'exec'),ns)
    stage('01_prepare.py')
    method=read(profile/'method_defaults.json')
    method.update(work=str(work),repo=str(ns['REPO']),output=str(out),asset_input=cfg.get('model_input'),
                  sanity=False,run_kind='full',asset_root=str(work/'apc_assets'),resume_input=None,
                  ablation_assets={k:cfg[k] for k in ('pins','need_sam1','need_oa1')})
    ns.update(CFG=method,OUT=out,CONFIG=work/'config.json')
    write(work/'config.json',method)
    stage('03_environment.py')
    ns['ENV']['PYTHONPATH']=str(root/'src')+os.pathsep+ns['ENV']['PYTHONPATH']
    # Replace only the copied runtime asset entry point; tracked APC profile is untouched.
    (ns['SCRIPTS']/'apc_prepare_assets.py').write_text(
        'from sc203_kat.ablation.apc_assets import run\nimport sys\nrun(*sys.argv[1:])\n',encoding='utf8')
    stage('04_native_assets.py')
    # Trace patching usually happens in the data stage; this replay does not download the TSV.
    subprocess.run([ns['PY'],str(ns['SCRIPTS']/'apc_trace_patches.py'),str(ns['REPO']),
                    str(work/'trace_patches.json')],env=ns['ENV'],check=True)
    env=ns['ENV'];env['PYTHONPATH']=str(root/'src')+os.pathsep+env['PYTHONPATH']
    keys=['PYTHONPATH','CUDA_VISIBLE_DEVICES','CUDA_HOME','CUDA_PATH','PATH','CPATH','LIBRARY_PATH',
          'LD_LIBRARY_PATH','DISPLAY','LIBGL_ALWAYS_SOFTWARE','MESA_GL_VERSION_OVERRIDE','MPLBACKEND',
          'HF_HOME','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','PYTHONNOUSERSITE']
    context=dict(py=ns['PY'],repo=str(ns['REPO']),scripts=str(ns['SCRIPTS']),cfg=method,
                 uv=str(ns['UV']),asset_root=read(work/'assets_ready.json')['root'],
                 env={k:env[k] for k in keys if k in env})
    write(work/'apc_context.json',context)
    write(out/'source_checkout.json',source_metadata(root,'apc_vis'))
    audit=out/'runtime_audit';audit.mkdir(exist_ok=True)
    for name in ('source_lock.json','patches.json','trace_patches.json','gpu_check.txt','config.json'):
        if (work/name).is_file():shutil.copy2(work/name,audit/name)
    freeze=subprocess.check_output([str(ns['UV']),'pip','freeze','--python',ns['PY']],env=ns['ENV'],text=True)
    (audit/'apc_environment.txt').write_text(freeze,encoding='utf8')
    return context

def setup_extra(cfg,root,context):
    work=Path(cfg['runtime_dir']);uv=context['uv'];env=os.environ.copy()
    for k in ('PYTHONHOME','PYTHONPATH','VIRTUAL_ENV'):env.pop(k,None)
    env.update(UV_PYTHON_INSTALL_DIR=str(work/'python_runtimes'),UV_CACHE_DIR=str(work/'uv_cache'),
               UV_NO_PROGRESS='1',UV_LINK_MODE='copy',PYTHONNOUSERSITE='1')
    repos={};pythons={}
    for kind in cfg['needed_models']:
        pin=cfg['pins'][kind];repo=work/'extra_repos'/kind;repo.parent.mkdir(parents=True,exist_ok=True)
        if not (repo/'.git').exists():
            subprocess.run(['git','clone','--no-checkout','--filter=blob:none','https://github.com/'+pin['repo']+'.git',str(repo)],check=True)
        else:
            origin=subprocess.check_output(['git','-C',str(repo),'remote','get-url','origin'],text=True).strip()
            if origin!='https://github.com/'+pin['repo']+'.git':raise RuntimeError('Auxiliary checkout has an unexpected origin')
            subprocess.run(['git','-C',str(repo),'diff','--exit-code','HEAD'],stdout=subprocess.DEVNULL,check=True)
        subprocess.run(['git','-C',str(repo),'fetch','--depth','1','origin',pin['sha']],check=True)
        subprocess.run(['git','-C',str(repo),'checkout','--detach',pin['sha']],check=True)
        version={'sam3':'3.12','oa2':'3.11','oneref':'3.10'}[kind]
        venv=work/'extra_envs'/kind;py=str(venv/'bin/python')
        subprocess.run([uv,'python','install',version],env=env,check=True)
        # uv repairs an interrupted environment explicitly; never rely on pyvenv.cfg alone.
        if not Path(py).exists():
            if venv.exists():
                backup=venv.with_name(venv.name+'_broken_'+str(__import__('time').time_ns()))
                venv.rename(backup)
            subprocess.run([uv,'venv','--python',version,'--managed-python',str(venv)],env=env,check=True)
        lock=venv/'sc203_environment.json'
        spec=read(root/'configs/ablations/environments.json')[kind]
        if not lock.exists() or read(lock)!=spec:
            def install(packages,extra=()):
                subprocess.run([uv,'pip','install','--python',py,'--no-cache',*extra,*packages],env=env,check=True)
            install(spec['torch'],['--index-url',spec['index']])
            install(spec['packages'])
            if kind=='sam3':install(['--no-deps','-e',str(repo)])
            if kind=='oneref':
                # torchscale declares an older timm pin. Both are installed at upstream versions.
                install(['torchscale==0.3.0'],['--no-deps'])
            write(lock,spec)
        freeze=subprocess.check_output([uv,'pip','freeze','--python',py],env=env,text=True)
        target=Path(cfg['output_dir'])/'environments';target.mkdir(exist_ok=True)
        (target/f'{kind}.txt').write_text(freeze,encoding='utf8')
        repos[kind]=str(repo);pythons[kind]=py
    return repos,pythons
