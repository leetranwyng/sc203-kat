"""Download once or reuse a read-only Kaggle asset input; never duplicate weight caches."""
import hashlib,json,os,shutil,sys,time,urllib.request
from http.client import HTTPException
from pathlib import Path
from huggingface_hub import HfApi,hf_hub_download,snapshot_download
cfg=json.loads(Path(sys.argv[1]).read_text());root=Path(cfg['asset_input'] or cfg['asset_root']);root.mkdir(parents=True,exist_ok=True)
manifest_file=root/'assets_manifest.json';old=json.loads(manifest_file.read_text()) if manifest_file.exists() else {}
if cfg['asset_input']:
    required=['qwen_model/config.json','qwen_model/preprocessor_config.json','dinov2_large/config.json','bert_base_uncased/config.json',
       'checkpoints/sam_vit_h_4b8939.pth','checkpoints/groundingdino_swint_ogc.pth','checkpoints/depth_pro.pt','checkpoints/orient_dino_weight.pt']
    missing=[s for s in required if not (root/s).is_file()]
    for model in ('qwen_model','dinov2_large','bert_base_uncased'):
        if not list((root/model).glob('*.safetensors')):missing.append(model+'/*.safetensors')
    if missing or not old:raise RuntimeError('ASSET_INPUT must be the prepared asset bundle root with assets_manifest.json. Missing '+str(missing))
    if old.get('model_id')!=cfg['model_id']:raise RuntimeError('Wrong Qwen checkpoint asset bundle.')
    Path(sys.argv[2]).write_text(json.dumps(dict(old,root=str(root.resolve())),indent=2),encoding='utf-8')
    print('Using existing asset bundle:',root);raise SystemExit(0)
api=HfApi();snapshots={}
models={'qwen_model':cfg['model_id'],'dinov2_large':'facebook/dinov2-large','bert_base_uncased':'google-bert/bert-base-uncased'}
for name,repo in models.items():
    previous=old.get('snapshots',{}).get(name,{})
    revision=previous.get('sha') if previous.get('repo')==repo else cfg['model_revision'] if name=='qwen_model' else 'main'
    info=api.model_info(repo,revision=revision,files_metadata=True)
    wanted=[f for f in info.siblings if '/' not in f.rfilename and (f.rfilename.endswith(('.json','.safetensors','.txt','.model','.jinja')))]
    remaining=sum(f.size or 0 for f in wanted if not (root/name/f.rfilename).is_file() or (root/name/f.rfilename).stat().st_size!=f.size)
    if shutil.disk_usage(root).free<remaining+3*2**30:raise RuntimeError('Asset storage lacks space. Use a prepared Kaggle ASSET_INPUT; need '+str(round(remaining/2**30,2))+' GiB + 3 GiB reserve for '+repo)
    print('Download/cache',repo,'SHA',info.sha,'remaining GiB',round(remaining/2**30,2),flush=True)
    for f in sorted(wanted,key=lambda f:f.rfilename.endswith('.safetensors')):
        hf_hub_download(repo,f.rfilename,revision=info.sha,local_dir=str(root/name))
        if f.size and (root/name/f.rfilename).stat().st_size!=f.size:raise RuntimeError('Truncated asset: '+f.rfilename)
    if not list((root/name).glob('*.safetensors')):raise RuntimeError('No safetensors weights in '+repo+'; stop before model initialization.')
    snapshots[name]=dict(repo=repo,sha=info.sha)
ckpt=root/'checkpoints';ckpt.mkdir(exist_ok=True)
urls={
 'sam_vit_h_4b8939.pth':'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth',
 'groundingdino_swint_ogc.pth':'https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth',
 'depth_pro.pt':'https://ml-site.cdn-apple.com/models/depth-pro/depth_pro.pt'}
def download_checkpoint(url,target):
    part=target.with_suffix('.part')
    for attempt in range(1,7):
        try:
            offset=part.stat().st_size if part.exists() else 0
            headers={'Accept-Encoding':'identity'}
            if offset:headers['Range']=f'bytes={offset}-'
            print(f'Downloading {target.name}: attempt {attempt}/6, resume {offset/2**20:.0f} MiB',flush=True)
            with urllib.request.urlopen(urllib.request.Request(url,headers=headers),timeout=120) as response:
                resumed=response.status==206
                if resumed:
                    content_range=response.headers.get('Content-Range','')
                    if not content_range.startswith(f'bytes {offset}-'):raise OSError('Unexpected Content-Range: '+content_range)
                    total=int(content_range.split('/')[-1])
                else:
                    offset=0 # Server ignored Range: overwrite, never append a full response.
                    total=int(response.headers.get('Content-Length',0))
                if total and shutil.disk_usage(target.parent).free<total-offset+3*2**30:
                    raise RuntimeError('Not enough disk for '+target.name+'; use ASSET_INPUT.')
                with part.open('ab' if resumed else 'wb') as f:
                    while True:
                        block=response.read(1024*1024)
                        if not block:break
                        f.write(block)
                size=part.stat().st_size
                if size<1000000 or total and size!=total:raise OSError(f'Truncated download: {size}/{total} bytes')
            part.replace(target)
            return
        except (OSError,HTTPException,ValueError) as exc:
            if attempt==6:raise RuntimeError(f'Failed downloading {target.name}; partial file kept at {part}: {exc}') from exc
            print(f'{target.name}: {type(exc).__name__}: {exc}; retrying...',flush=True)
            time.sleep(min(5*attempt,30))

checks={}
for name,url in urls.items():
    target=ckpt/name;expected=old.get('files',{}).get(name,{})
    valid=target.is_file() and target.stat().st_size==expected.get('bytes')
    # Older runs wrote the manifest only after all downloads. Reuse their complete files.
    if target.is_file() and not expected:
        try:
            with urllib.request.urlopen(urllib.request.Request(url,method='HEAD'),timeout=60) as response:
                valid=target.stat().st_size==int(response.headers.get('Content-Length',0)) and target.stat().st_size>1000000
        except OSError:pass
    if not valid:
        download_checkpoint(url,target)
    else:print('Reusing',name,flush=True)
    digest=hashlib.sha256()
    with target.open('rb') as f:
        for block in iter(lambda:f.read(16*1024*1024),b''):digest.update(block)
    if valid and expected.get('sha256') and digest.hexdigest()!=expected['sha256']:raise RuntimeError('Asset checksum mismatch '+name)
    checks[name]=dict(url=url,bytes=target.stat().st_size,sha256=digest.hexdigest())
    # Save completed checkpoint metadata even if a later download fails.
    old.update(model_id=cfg['model_id'],snapshots=snapshots,files=checks)
    manifest_file.write_text(json.dumps(old,indent=2),encoding='utf-8')
ori_revision=old.get('orientation_sha','main');ori_info=api.model_info('Viglong/Orient-Anything',revision=ori_revision)
ori_path=hf_hub_download('Viglong/Orient-Anything','croplargeEX2/dino_weight.pt',revision=ori_info.sha,local_dir=str(root/'orientation_download'))
# Symlink rather than make another full checkpoint copy.
target=ckpt/'orient_dino_weight.pt'
if target.is_symlink() and target.resolve()!=Path(ori_path).resolve():target.unlink()
if not target.exists():target.symlink_to(Path(ori_path).resolve())
manifest=dict(model_id=cfg['model_id'],snapshots=snapshots,orientation_sha=ori_info.sha,files=checks,root=str(root.resolve()))
manifest_file.write_text(json.dumps(manifest,indent=2),encoding='utf-8');Path(sys.argv[2]).write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print('Assets ready:',root,flush=True)
