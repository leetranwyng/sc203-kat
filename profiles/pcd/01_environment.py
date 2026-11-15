from pathlib import Path
import os, subprocess, sys
WORK=Path(RUNTIME_DIR)
WORK.mkdir(parents=True,exist_ok=True)
TOOLS=Path('/kaggle/temp/pcd7b_tools')
TOOLS.mkdir(parents=True,exist_ok=True)
subprocess.run([sys.executable,'-m','pip','install','--quiet','--target',str(TOOLS),'uv==0.9.5'],check=True)
UV=str(TOOLS/'bin/uv')
ENV=dict(os.environ,UV_PYTHON_INSTALL_DIR='/kaggle/temp/pcd7b_python',UV_CACHE_DIR='/kaggle/temp/pcd7b_uv',
         HF_HOME='/kaggle/temp/pcd7b_hf',HF_HUB_DISABLE_XET='1',TOKENIZERS_PARALLELISM='false')
from kaggle_secrets import UserSecretsClient
try:
    if not ENV.get('HF_TOKEN'):
        ENV['HF_TOKEN']=UserSecretsClient().get_secret('HF_token')
except Exception:
    print('HF_token unavailable; downloading public checkpoint without authentication.')
VENV=Path('/kaggle/temp/pcd7b_venv')
PY=str(VENV/'bin/python')
if not Path(PY).exists():
    subprocess.run([UV,'venv','--clear','--python','3.11','--managed-python',str(VENV)],env=ENV,check=True)
subprocess.run([PY,'-c','import sys; assert sys.version_info[:2]==(3,11)'],env=ENV,check=True)
subprocess.run([UV,'pip','install','--python',PY,'torch==2.5.1','torchvision==0.20.1',
                '--index-url','https://download.pytorch.org/whl/cu124'],env=ENV,check=True)
subprocess.run([UV,'pip','install','--python',PY,'transformers==4.51.3','accelerate==1.6.0',
    'huggingface_hub==0.30.2','tokenizers==0.21.1','safetensors>=0.4.3,<1',
    'pandas>=2.2,<3','Pillow>=10,<12','tqdm>=4.66,<5','openpyxl>=3.1,<4'],env=ENV,check=True)
