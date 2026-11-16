"""Repair a live ablation session and retry only failed OneRef-dependent work."""
import json
import os
from pathlib import Path
import subprocess
import time

def recover(runtime='/kaggle/temp/apc_ablation_runtime/sanity',env=None,run=subprocess.run):
    work=Path(runtime)
    cfg=json.loads((work/'ablation.json').read_text())
    context=json.loads((work/'apc_context.json').read_text())
    out=Path(cfg['output_dir']);audit=out/'runtime_audit';audit.mkdir(exist_ok=True)
    selected=[v for v in cfg['variants'] if 'oneref' in v]
    if not selected:raise ValueError('This run has no OneRef branch')
    base=dict(os.environ if env is None else env)
    main=dict(base,**context['env'],HF_HUB_OFFLINE='0',TRANSFORMERS_OFFLINE='0')
    auxiliary=main.copy()
    # Keep the old source/fingerprint; only repair packaging support.
    auxiliary['PYTHONPATH']=context['env']['PYTHONPATH']
    auxiliary['CUDA_VISIBLE_DEVICES']=context['env']['CUDA_VISIBLE_DEVICES'].split(',')[0]
    auxiliary['LD_LIBRARY_PATH']=base.get('LD_LIBRARY_PATH','')
    for key in ('CUDA_HOME','CUDA_PATH','UV_SYSTEM_PYTHON','PYTHONHOME','VIRTUAL_ENV'):
        auxiliary.pop(key,None)
    py=str(work/'extra_envs/oneref/bin/python')
    if not Path(py).is_file():raise FileNotFoundError('Keep the original Kaggle session active: '+py)
    print('Repairing the legacy OneRef packaging and spaCy CLI dependencies...',flush=True)
    run([context['uv'],'pip','uninstall','--python',py,'typer-slim','typer'],env=auxiliary,check=True)
    run([context['uv'],'pip','install','--python',py,'--no-cache','setuptools==75.8.0','click==8.1.8','typer==0.9.0','weasel==0.3.4'],
        env=auxiliary,check=True)
    run([py,'-c','import pkg_resources; from torchmetrics import Metric; from models.OneRef_model import beit3_base_patch16_384_grounding; print("OneRef imports OK")'],
        env=auxiliary,cwd=cfg.get('repos',{}).get('oneref',str(work/'extra_repos/oneref')),check=True)
    freeze=run([context['uv'],'pip','freeze','--python',py],env=auxiliary,check=True,capture_output=True,text=True)
    (audit/'oneref_recovery_environment.txt').write_text(freeze.stdout,encoding='utf8')
    if not cfg.get('weights') or not cfg.get('repos'):
        # A setup probe stopped before setup_extra returned. Continue the original
        # orchestrator with the completed native setup supplied from its checkpoint.
        code='''import json,sys
from pathlib import Path
import sc203_kat
from sc203_kat.ablation import run as runner, setup
work=Path(sys.argv[1]);context=json.loads((work/'apc_context.json').read_text())
cfg=json.loads((work/'ablation.json').read_text());cfg['retry_errors']=True
setup.setup_apc=lambda cfg,root:context
runner.run(cfg,Path(sc203_kat.__file__).resolve().parents[2])
'''
        print('Resuming the interrupted setup; completed native compilation is reused.',flush=True)
        run([context['py'],'-u','-c',code,str(work)],env=main,check=True)
        return
    retry=dict(cfg,retry_errors=True,variants=selected)
    retry_file=work/'oneref_recovery_config.json'
    retry_file.write_text(json.dumps(retry,indent=2),encoding='utf8')
    (audit/'oneref_recovery.json').write_text(json.dumps(dict(started=time.time(),
        setuptools='75.8.0',variants=selected,original_source=context['env']['PYTHONPATH'],
        note='Preserve completed branches and baseline; no setup/model downloads are rerun.'),indent=2),encoding='utf8')
    for pyexe,module,args,child in [
        (py,'sc203_kat.ablation.perception',[str(retry_file),'oneref'],auxiliary),
        (context['py'],'sc203_kat.ablation.apc_worker',[str(retry_file),'geometry'],main),
        (context['py'],'sc203_kat.ablation.apc_worker',[str(retry_file),'answers'],main),
    ]:
        print('Retrying',module,args[-1],flush=True)
        run([pyexe,'-u','-m',module,*args],env=child,check=True)
    # The original identity still lists all branches: the report includes previous SAM3/OA2 results.
    command='from sc203_kat.ablation.report import build; import sys; build(sys.argv[1])'
    run([context['py'],'-c',command,str(out)],env=main,check=True)
    summary=json.loads((out/'summary.json').read_text())
    if not summary['technical_pass']:raise RuntimeError('OneRef still has an error/invalid output. Inspect index.html and result.json.')
    old_state=out/'state.json'
    if old_state.exists():
        (audit/'state_before_oneref_recovery.json').write_text(old_state.read_text(),encoding='utf8')
    old_state.write_text(json.dumps(dict(status='finished',failures=[],recovered_component='oneref'),indent=2),encoding='utf8')
    print('Sanity technical PASS. Report:',out/'index.html',flush=True)

if __name__=='__main__':recover()
