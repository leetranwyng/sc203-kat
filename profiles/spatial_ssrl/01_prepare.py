import os,json,sys,time,shutil,hashlib,subprocess,io,base64,zipfile
from pathlib import Path
WORK=Path(RUNTIME_DIR); WORK.mkdir(parents=True,exist_ok=True)
REPO=WORK/'repo'; SCRIPTS=WORK/'scripts'; SCRIPTS.mkdir(exist_ok=True); REPO.mkdir(exist_ok=True)
OUT=Path(OUTPUT_DIR)
if RESUME_INPUT and not (OUT/'manifest.json').exists():
    src=Path(RESUME_INPUT)
    assert (src/'manifest.json').is_file(), 'Choose the previous output folder containing manifest.json'
    shutil.copytree(src,OUT,dirs_exist_ok=True)
OUT.mkdir(parents=True,exist_ok=True)
try:
    from kaggle_secrets import UserSecretsClient
    token=os.environ.get('HF_TOKEN') or UserSecretsClient().get_secret('HF_token')
    if token:os.environ['HF_TOKEN']=token.strip()
except Exception:pass
os.environ['HF_HUB_DISABLE_XET']='1'
os.environ['TOKENIZERS_PARALLELISM']='false'
CONFIG=WORK/'config.json'

upstream=REPOSITORY_ROOT/'third_party'/UPSTREAM_KEY
if upstream.is_dir() and any(upstream.iterdir()):
    shutil.copytree(upstream,REPO,dirs_exist_ok=True,ignore=shutil.ignore_patterns('.git','__pycache__','*.pyc'))
shutil.copytree(PROFILE_DIR/'overlay',WORK,dirs_exist_ok=True)
CFG={'method': 'ssrl', 'sanity': False, 'model_id': 'internlm/Spatial-SSRL-7B', 'model_revision': '4f7440e12904173b26deae933c83f40f6ad4f4c5', 'dataset_id': 'ccvl/3DSRBench', 'data_revision': '5c90e422b68d0484d50a3ff06b3096ea7d288026', 'data_file': '3dsrbench_v1_vlmevalkit_circular.tsv', 'repo_commit': '5b27b7b314e65dadccca6e065b8a4dc039534021', 'max_tokens': 4096, 'format_prompt': 'You FIRST think about the reasoning process as an internal monologue and then provide the final answer. The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \\boxed{}.', 'runtime_sha': '776911867ff9b08896ef8be314dad0a056217926e55ae0427d2ef1ebbb455b76', 'limit': None}
CFG.update(METHOD_OPTIONS)
CFG['sanity']=SANITY
CFG.update(work=str(WORK),output=str(OUT),repo=str(REPO),scope=SCOPE,asset_root=str(WORK/'assets'),asset_input=MODEL_INPUT,dataset_input=DATASET_INPUT)
if METHOD=='sympl':assert SCOPE=='paper_tasks_visibility'
assert SCOPE in ('paper_tasks_original','paper_tasks_visibility')
CONFIG.write_text(json.dumps(CFG))
