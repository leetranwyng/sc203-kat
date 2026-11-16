"""Real OneRef CPU import/build/forward check, without pretrained weights."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import shutil
import sys
import tempfile
import urllib.request

ROOT=Path(__file__).resolve().parents[1]

def main(install=False):
    spec=json.loads((ROOT/'configs/ablations/environments.json').read_text())['oneref']
    pins=json.loads((ROOT/'configs/ablations/upstreams.json').read_text())['oneref']
    if install:
        uv=[shutil.which('uv'),'pip']
        if uv[0] is None:raise RuntimeError('Install uv before using --install')
        subprocess.run([*uv,'install','--python',sys.executable,'--index-url',
            'https://download.pytorch.org/whl/cpu',*spec['torch']],check=True)
        subprocess.run([*uv,'uninstall','--python',sys.executable,'typer-slim','typer'],check=True)
        subprocess.run([*uv,'install','--python',sys.executable,*spec['packages']],check=True)
        subprocess.run([*uv,'install','--python',sys.executable,'--no-deps','torchscale==0.3.0'],check=True)
    folder=Path(tempfile.gettempdir())/'sc203-oneref-cpu';folder.mkdir(exist_ok=True)
    repo=folder/'OneRef'
    if not (repo/'.git').exists():subprocess.run(['git','clone','--no-checkout','--filter=blob:none',
        'https://github.com/'+pins['repo']+'.git',str(repo)],check=True)
    subprocess.run(['git','-C',str(repo),'fetch','--depth','1','origin',pins['sha']],check=True)
    subprocess.run(['git','-C',str(repo),'checkout','--detach',pins['sha']],check=True)
    tokenizer=folder/'beit3.spm'
    if not tokenizer.exists():
        url=f'https://huggingface.co/{pins["model_repo"]}/resolve/{pins["model_sha"]}/beit3_checkpoints/beit3.spm'
        with urllib.request.urlopen(url,timeout=120) as response:tokenizer.write_bytes(response.read())
    sys.path[:0]=[str(repo),str(ROOT/'src')]
    os.chdir(repo)
    import pkg_resources
    from torchmetrics import Metric
    import click,typer,spacy,torch
    from sc203_kat.ablation.perception import oneref_architecture
    torch.set_num_threads(2)
    torch.manual_seed(42)
    model=oneref_architecture(str(tokenizer)).eval()
    with torch.inference_mode():
        output=model(image=torch.zeros(1,3,384,384),img_mask=torch.zeros(1,384,384,dtype=torch.int),
                     text=['the red chair'],training=False)
    box=output[0]
    assert tuple(box.shape)==(1,4) and torch.isfinite(box).all()
    assert ((box>=0)&(box<=1)).all()
    print('PASS: actual OneRef imports, production model constructor and CPU forward.',flush=True)
    print('Random initialization only; pretrained checkpoint/GPU accuracy not evaluated.',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--install',action='store_true');main(p.parse_args().install)
