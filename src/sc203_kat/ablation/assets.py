"""Download only the pinned auxiliary weights, optionally use Kaggle Input paths."""
import json
import os
from pathlib import Path
import shutil
import sys
import zipfile
from .common import read, write, sha

def run(config):
    from huggingface_hub import hf_hub_download
    cfg=read(config);pins=cfg['pins'];weights={};audit={}
    folder=Path(cfg['runtime_dir'])/'extra_weights';folder.mkdir(parents=True,exist_ok=True)
    def download(kind,filename):
        return hf_hub_download(repo_id=pins[kind]['model_repo'],revision=pins[kind]['model_sha'],
                               filename=filename,cache_dir=str(folder/'hf'))
    for kind in cfg['needed_models']:
        local=cfg.get('weight_inputs',{}).get(kind)
        if local:
            p=Path(local)
            if not p.is_file():raise FileNotFoundError(f'{kind}: {local}')
        elif kind=='oneref':
            p=folder/'oneref_base_mixup.pth'
            if not p.is_file():
                from huggingface_hub import hf_hub_url
                from .remote_zip import extract_mixup
                url=hf_hub_url(pins[kind]['model_repo'],pins[kind]['filename'],revision=pins[kind]['model_sha'])
                extract_mixup(url,p,os.environ.get('HF_TOKEN'))
        else:p=Path(download(kind,pins[kind]['filename']))
        weights[kind]=str(p)
        audit[kind]=dict(path=str(p),sha256=sha(p),source='user_input' if local else 'pinned_HF',
                         repo=pins[kind]['model_repo'],revision=pins[kind]['model_sha'])
    if 'oneref' in cfg['needed_models']:
        weights['oneref_tokenizer']=cfg.get('weight_inputs',{}).get('oneref_tokenizer') or download('oneref','beit3_checkpoints/beit3.spm')
    cfg['weights']=weights
    write(config,cfg);write(Path(cfg['output_dir'])/'weights_lock.json',audit)

if __name__=='__main__':run(sys.argv[1])
