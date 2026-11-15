import json, os, re, shutil, subprocess, sys, time
from pathlib import Path

STARTED = time.time()
assert re.fullmatch(r'[A-Za-z0-9_-]+', RUN_NAME), 'RUN_NAME: letters, digits, underscore or hyphen'
assert PROTOCOL in {'original', 'visibility', 'both'}
assert MAX_IMAGE_TOKENS >= 256 and MAX_NEW_TOKENS > 0
assert SESSION_BUDGET_MINUTES > 0 and MAX_ROWS_THIS_SESSION >= 0
TEMP_ROOT = Path(RUNTIME_DIR)
WORK = TEMP_ROOT / MODE
OUT = Path(OUTPUT_DIR)
for path in (TEMP_ROOT, WORK, OUT):
    path.mkdir(parents=True, exist_ok=True)
gpu = subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total','--format=csv,noheader'], text=True)
devices = gpu.strip().splitlines()
if len(devices) != 2 or not all('T4' in line for line in devices):
    raise RuntimeError('Select GPU T4 ×2 on Kaggle. Detected: ' + repr(devices))
print(gpu.strip())
env = dict(os.environ, PYTHONNOUSERSITE='1', UV_PYTHON_INSTALL_DIR=str(TEMP_ROOT/'python'),
           UV_CACHE_DIR=str(TEMP_ROOT/'uv_cache'), HF_HOME=str(TEMP_ROOT/'hf'),
           HF_HUB_DISABLE_XET='1', HF_HUB_DOWNLOAD_TIMEOUT='180',
           TOKENIZERS_PARALLELISM='false', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
try:
    from kaggle_secrets import UserSecretsClient
    secret = env.get('HF_TOKEN') or UserSecretsClient().get_secret('HF_token')
    if secret:
        env['HF_TOKEN'] = secret.strip()
except Exception:
    print('No HF_token secret. The official model and data are public.')
finally:
    if 'secret' in globals():
        del secret
tools = TEMP_ROOT/'uv_bootstrap'
uv = tools/'bin/uv'
venv = TEMP_ROOT/'venv_py311_cu124_v1'
PY = venv/'bin/python'
ready = venv/'sc203_ready'
if not uv.is_file():
    subprocess.run([sys.executable,'-m','pip','install','--no-cache-dir','--quiet',
                    '--target',str(tools),'uv==0.9.5'], check=True)
if not (ready.is_file() and PY.is_file()):
    if shutil.disk_usage(TEMP_ROOT).free < 8*2**30:
        raise OSError('Need at least 8 GiB temporary space for the isolated runtime.')
    subprocess.run([str(uv),'python','install','3.11.14'], env=env, check=True)
    if not PY.is_file():
        subprocess.run([str(uv),'venv','--clear','--managed-python','--python','3.11.14',str(venv)],
                       env=env, check=True)
    subprocess.run([str(uv),'pip','install','--no-cache','--python',str(PY),
                    '--index-url','https://download.pytorch.org/whl/cu124',
                    'torch==2.5.1+cu124','torchvision==0.20.1+cu124'], env=env, check=True)
    subprocess.run([str(uv),'pip','install','--no-cache','--python',str(PY),
                    'transformers==4.57.1','accelerate==1.11.0','huggingface_hub==0.36.0',
                    'qwen-vl-utils==0.0.14','pillow==11.3.0'], env=env, check=True)
    subprocess.run([str(PY),'-c',
                    'from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor; '
                    'from qwen_vl_utils import process_vision_info; import torch; print(torch.__version__)'],
                   env=env,check=True)
    ready.touch()
CONFIG = WORK/'config.json'
CONFIG.write_text(json.dumps(dict(mode=MODE,protocol=PROTOCOL,out=str(OUT),work=str(WORK),
    model_input=MODEL_INPUT,dataset_input=DATASET_INPUT,resume_input=RESUME_INPUT,
    thinking=THINKING,max_new_tokens=MAX_NEW_TOKENS,max_image_tokens=MAX_IMAGE_TOKENS,
    deadline=STARTED+60*SESSION_BUDGET_MINUTES,max_rows_this_session=MAX_ROWS_THIS_SESSION)),encoding='utf8')
