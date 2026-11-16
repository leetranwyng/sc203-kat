import json,hashlib,sys
from pathlib import Path
CONFIG=Path(sys.argv[1]); CFG=json.loads(CONFIG.read_text()); OUT=Path(CFG['output']); WORK=Path(CFG['work']); MODEL_INPUT=CFG['asset_input']
from huggingface_hub import HfApi,snapshot_download
old=OUT/'manifest.json'
rev=json.loads(old.read_text())['identity']['model_sha'] if old.exists() else CFG['model_revision']
if MODEL_INPUT:
    model=Path(MODEL_INPUT)
    h=hashlib.sha256()
    for p in sorted(p for p in model.iterdir() if p.suffix in ('.json','.txt','.safetensors','.jinja')):
        h.update(p.name.encode())
        with p.open('rb') as f:
            for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    sha='local-'+h.hexdigest()
else:
    sha=HfApi().model_info(CFG['model_id'],revision=rev).sha
    model=Path(snapshot_download(CFG['model_id'],revision=sha,cache_dir=str(WORK/'weights'),allow_patterns=['*.json','*.txt','*.jinja','*.safetensors']))
assert (model/'config.json').is_file() and list(model.glob('*.safetensors'))
CFG.update(model_sha=sha,weights=str(model))

CONFIG.write_text(json.dumps(CFG))