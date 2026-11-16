"""Fetch only live APC assets used by the controlled perception study."""
import sys
from pathlib import Path
from .common import read,write,sha

def run(config,destination):
    from huggingface_hub import hf_hub_download,snapshot_download
    cfg=read(config);spec=cfg['ablation_assets'];root=Path(cfg.get('asset_input') or cfg['asset_root'])
    if not cfg.get('asset_input'):root.mkdir(parents=True,exist_ok=True)
    pins=spec['pins'];audit={}
    wanted={'qwen_model':(cfg['model_id'],cfg['model_revision'])}
    if spec['need_oa1']:wanted['dinov2_large']=(pins['dinov2']['model_repo'],pins['dinov2']['model_sha'])
    for name,(repo,rev) in wanted.items():
        p=root/name
        if cfg.get('asset_input'):
            if not (p/'config.json').exists() or not list(p.glob('*.safetensors')):
                raise FileNotFoundError('APC asset_input missing '+str(p))
            manifest=read(root/'assets_manifest.json')
            if name=='qwen_model' and manifest.get('snapshots',{}).get(name,{}).get('sha')!=rev:
                raise ValueError('APC bundle Qwen revision differs from historical configuration')
        else:
            snapshot_download(repo,revision=rev,local_dir=str(p),
                allow_patterns=['*.json','*.safetensors','*.txt','*.model','*.jinja'])
        audit[name]=dict(repo=repo,sha=rev)
    ckpt=root/'checkpoints'
    if not cfg.get('asset_input'):ckpt.mkdir(exist_ok=True)
    for name,key,needed in [('sam_vit_h_4b8939.pth','sam1',spec['need_sam1']),
                             ('orient_dino_weight.pt','oa1',spec['need_oa1'])]:
        if not needed:continue
        p=ckpt/name;pin=pins[key]
        if not cfg.get('asset_input'):
            source=Path(hf_hub_download(pin['model_repo'],pin['filename'],revision=pin['model_sha'],
                                      local_dir=str(root/key)))
            if not p.exists():p.symlink_to(source.resolve())
        if not p.is_file():raise FileNotFoundError(str(p))
        audit[name]=dict(repo=pin['model_repo'],revision=pin['model_sha'],sha256=sha(p))
    value=dict(root=str(root.resolve()),model_id=cfg['model_id'],snapshots=audit,
               unused_models_not_downloaded=['GroundingDINO','DepthPro'])
    write(destination,value)
    write(Path(cfg['output'])/'apc_assets_lock.json',value)

if __name__=='__main__':run(*sys.argv[1:])
