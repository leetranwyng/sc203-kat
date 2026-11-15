"""Generate ten thin notebooks. Inference source is never embedded in these notebooks."""
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
METHODS = ("apc_vis", "pcd", "sympl", "vst", "spatial_ssrl")

def cell(kind, source):
    value = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
    if kind == "code":
        value.update(execution_count=None, outputs=[])
    return value

BOOTSTRAP = r'''import hashlib, json, os, subprocess, sys, zipfile
from pathlib import Path

CODE_ROOT = Path('/kaggle/temp/sc203-kat')
if KAT_INPUT:
    bundle = Path(KAT_INPUT)
    if bundle.is_dir():
        candidates = [bundle] + [p.parent for p in bundle.rglob('pyproject.toml')]
        CODE_ROOT = next((p for p in candidates if (p/'src/sc203_kat').is_dir()), None)
        if CODE_ROOT is None:
            raise FileNotFoundError('KAT_INPUT directory must contain an extracted sc203-kat bundle')
    else:
        # Check paths before extracting code from Kaggle Input.
        code_digest = hashlib.sha256(bundle.read_bytes()).hexdigest()[:16]
        CODE_ROOT = CODE_ROOT.parent / ('sc203-kat-' + code_digest)
        CODE_ROOT.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(bundle) as archive:
            for info in archive.infolist():
                name = info.filename.replace('\\', '/')
                destination = (CODE_ROOT/name).resolve()
                if not destination.is_relative_to(CODE_ROOT.resolve()) or ':' in name:
                    raise ValueError('Unsafe code bundle path: '+name)
                if (info.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError('Code bundle cannot contain symlinks')
            archive.extractall(CODE_ROOT)
        manifest_path = CODE_ROOT/'bundle_manifest.json'
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
            for name, expected in manifest['files_sha256'].items():
                path = (CODE_ROOT/name).resolve()
                if not path.is_relative_to(CODE_ROOT.resolve()) or not path.is_file():
                    raise ValueError('Invalid bundle manifest path: '+name)
                if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                    raise ValueError('Code bundle checksum mismatch: '+name)
else:
    if not REPO_URL:
        raise ValueError('Set KAT_INPUT to the code bundle in Kaggle Input, or set REPO_URL after pushing the local repo')
    if not (CODE_ROOT/'.git').is_dir():
        subprocess.run(['git','clone','--no-checkout',REPO_URL,str(CODE_ROOT)],check=True)
    subprocess.run(['git','-C',str(CODE_ROOT),'checkout','--detach',CODE_REF],check=True)

ENV = dict(os.environ, SC203_KAT_ROOT=str(CODE_ROOT), PYTHONPATH=str(CODE_ROOT/'src'))
if HF_SECRET_NAME:
    try:
        from kaggle_secrets import UserSecretsClient
        value = UserSecretsClient().get_secret(HF_SECRET_NAME)
        if value: ENV['HF_TOKEN'] = value.strip()
        del value
    except Exception:
        print('HF secret unavailable; using public/local model files.')
CONFIG_PATH = Path('/kaggle/temp/sc203-kat-run.json')
CONFIG_PATH.write_text(json.dumps(RUN_CONFIG,ensure_ascii=False,indent=2),encoding='utf-8')
print('Code:', CODE_ROOT, '| Method:', RUN_CONFIG['method'], '| Mode:', RUN_CONFIG['mode'])
'''
RUN = '''subprocess.run([sys.executable, '-m', 'sc203_kat', 'run', '--config', str(CONFIG_PATH)],
               cwd=CODE_ROOT, env=ENV, check=True)
'''
VIEW = '''output = Path(RUN_CONFIG['output_dir'])
print('Results:', output)
for path in sorted(output.glob('*')):
    if path.is_file(): print(path.name, f'{path.stat().st_size / 2**20:.2f} MiB')
'''

def main():
    for method in METHODS:
        for mode in ('sanity','full'):
            conf = {"method":method,"mode":mode,"run_name":method+'_'+mode,"model_input":None,"dataset_input":None,"resume_input":None,"output_dir":'/kaggle/working/sc203-kat/'+method+'_'+mode,"runtime_dir":'/kaggle/temp/sc203-kat-runtime/'+method,"method_options":{}}
            path = ROOT/'configs/experiments'/f'{method}_{mode}.json'
            path.write_text(json.dumps(conf,indent=2)+'\n',encoding='utf-8')
            config = "# Set the code bundle Input path, or your future GitHub URL.\nKAT_INPUT = None\nREPO_URL = ''\nCODE_REF = 'v0.1.0'\nHF_SECRET_NAME = 'HF_token'\nRUN_CONFIG = " + repr(conf) + '\n'
            nb = {"nbformat":4,"nbformat_minor":5,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python"}},"cells":[cell('markdown',f'# SC203-KAT · {method} · {mode}\nSelect GPU T4 ×2. Add a packaged code bundle as Kaggle Input, or set the GitHub URL and pinned release. Model weights and resume checkpoints can be added through Input. GPU inference requires the dependencies installed by the method profile.\n'),cell('code',config),cell('code',BOOTSTRAP),cell('code',RUN),cell('code',VIEW)]}
            dest=ROOT/'notebooks'/method/f'{mode}.ipynb';dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(nb,indent=1,ensure_ascii=False)+'\n',encoding='utf-8')
    print('Generated 10 notebooks')
if __name__ == '__main__': main()
