from pathlib import Path
import os,sys,json,subprocess,time,hashlib,shutil,base64,io,zipfile
WORK=Path(RUNTIME_DIR);WORK.mkdir(parents=True,exist_ok=True)
token=os.environ.get('HF_TOKEN')
if not token:
    try:
        from kaggle_secrets import UserSecretsClient
        token=UserSecretsClient().get_secret('HF_token')
    except Exception:
        token=None
if token:os.environ['HF_TOKEN']=token.strip()
else:print('HF_token chưa đặt; dùng quyền truy cập public. Có thể thêm Secret HF_token nếu bị giới hạn tải.')
del token
os.environ['TOKENIZERS_PARALLELISM']='false';os.environ['HF_HUB_DISABLE_TELEMETRY']='1';os.environ['HF_HUB_DISABLE_XET']='1'
SNAPSHOT_SHA='9059ed359baf89b49675ed3e0407f843b3ff28de78e5cae1c870adf79f44a339'
REPO=WORK/('repo_'+SNAPSHOT_SHA[:12]);REPO.mkdir(exist_ok=True)
upstream=REPOSITORY_ROOT/'third_party'/UPSTREAM_KEY
if upstream.is_dir() and any(upstream.iterdir()):
    shutil.copytree(upstream,REPO,dirs_exist_ok=True,ignore=shutil.ignore_patterns('.git','__pycache__','*.pyc'))
shutil.copytree(PROFILE_DIR/'overlay/repo',REPO,dirs_exist_ok=True)
SCRIPTS=WORK/'scripts';SCRIPTS.mkdir(exist_ok=True)
(SCRIPTS/'apc_common.py').write_text((PROFILE_DIR/"sources/apc_common.py").read_text(encoding="utf8"),encoding="utf8")
(SCRIPTS/'apc_kaggle_patches.py').write_text((PROFILE_DIR/"sources/apc_kaggle_patches.py").read_text(encoding="utf8"),encoding="utf8")
(SCRIPTS/'apc_install_native.py').write_text((PROFILE_DIR/"sources/apc_install_native.py").read_text(encoding="utf8"),encoding="utf8")
(SCRIPTS/'apc_prepare_assets.py').write_text((PROFILE_DIR/"sources/apc_prepare_assets.py").read_text(encoding="utf8"),encoding="utf8")
(SCRIPTS/'apc_qwen_adapter.py').write_text((PROFILE_DIR/"sources/apc_qwen_adapter.py").read_text(encoding="utf8"),encoding="utf8")
(SCRIPTS/'apc_experiment.py').write_text((PROFILE_DIR/"sources/apc_experiment.py").read_text(encoding="utf8"),encoding="utf8")
(SCRIPTS/'apc_prepare_data.py').write_text((PROFILE_DIR/"sources/apc_prepare_data.py").read_text(encoding="utf8"),encoding="utf8")
(SCRIPTS/'apc_kaggle_memory.py').write_text((PROFILE_DIR/"sources/apc_kaggle_memory.py").read_text(encoding="utf8"),encoding="utf8")
(SCRIPTS/'apc_qwen_preflight.py').write_text((PROFILE_DIR/"sources/apc_qwen_preflight.py").read_text(encoding="utf8"),encoding="utf8")
(SCRIPTS/'apc_kaggle_guardrails.py').write_text((PROFILE_DIR/"sources/apc_kaggle_guardrails.py").read_text(encoding="utf8"),encoding="utf8")
(SCRIPTS/'apc_trace_patches.py').write_text((PROFILE_DIR/"sources/apc_trace_patches.py").read_text(encoding="utf8"),encoding="utf8")
(SCRIPTS/'apc_worker.py').write_text((PROFILE_DIR/"sources/apc_worker.py").read_text(encoding="utf8"),encoding="utf8")
(SCRIPTS/'apc_launcher.py').write_text((PROFILE_DIR/"sources/apc_launcher.py").read_text(encoding="utf8"),encoding="utf8")
(SCRIPTS/'apc_v4_worker.py').write_text((PROFILE_DIR/"sources/apc_v4_worker.py").read_text(encoding="utf8"),encoding="utf8")
(SCRIPTS/'apc_v4_launcher.py').write_text((PROFILE_DIR/"sources/apc_v4_launcher.py").read_text(encoding="utf8"),encoding="utf8")
(SCRIPTS/'apc_full_reference.py').write_text((PROFILE_DIR/"sources/apc_full_reference.py").read_text(encoding="utf8"),encoding="utf8")
(SCRIPTS/'apc_full_checkpoint.py').write_text((PROFILE_DIR/"sources/apc_full_checkpoint.py").read_text(encoding="utf8"),encoding="utf8")

sys.path.insert(0,str(SCRIPTS))
for module in ('apc_launcher','apc_v4_launcher','apc_full_checkpoint','apc_kaggle_guardrails'):
    sys.modules.pop(module,None)
from apc_kaggle_guardrails import setup_stage,run_logged
from apc_full_checkpoint import prepare_resume,export_full
from apc_launcher import show_downloads
print('Runtime v6 written: viewer matching and sample-level error handling.')
