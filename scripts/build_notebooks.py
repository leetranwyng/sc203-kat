"""Generate ten GitHub-only Kaggle notebooks from experiment configs."""
import argparse
import hashlib
import json
import pprint
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
METHODS = ('apc_vis', 'pcd', 'sympl', 'vst', 'spatial_ssrl')
REPO_URL = 'https://github.com/leetranwyng/sc203-kat.git'
DEFAULT_REF = 'v0.2.0'

def cell(kind, source):
    value = {'cell_type': kind, 'id': hashlib.sha256((kind + source).encode()).hexdigest()[:12], 'metadata': {}, 'source': source.splitlines(keepends=True)}
    if kind == 'code': value.update(execution_count=None, outputs=[])
    return value

BOOTSTRAP = r'''import hashlib, json, os, subprocess, sys
from pathlib import Path

if not CODE_REF or CODE_REF.startswith('-') or any(c.isspace() for c in CODE_REF):
    raise ValueError('CODE_REF must be a Git tag, branch, or full commit.')
if REPO_URL != 'https://github.com/leetranwyng/sc203-kat.git':
    raise ValueError('Use the public leetranwyng/sc203-kat repository.')
GIT_ENV = dict(os.environ, GIT_TERMINAL_PROMPT='0', GCM_INTERACTIVE='Never')
cache = Path(os.environ.get('SC203_CODE_CACHE', '/kaggle/temp/sc203-kat-code'))
cache.mkdir(parents=True, exist_ok=True)
key = hashlib.sha256((REPO_URL + '\n' + CODE_REF).encode()).hexdigest()[:16]
CODE_ROOT = cache / key

def git(*args):
    result = subprocess.run(['git', *args], env=GIT_ENV, text=True,
                            capture_output=True, timeout=300)
    if result.returncode: raise RuntimeError((result.stderr or result.stdout).strip())
    return result.stdout.strip()

if not (CODE_ROOT / '.git').exists():
    if CODE_ROOT.exists() and any(CODE_ROOT.iterdir()):
        raise RuntimeError('Code cache is not empty; choose a different SC203_CODE_CACHE.')
    git('clone', '--no-checkout', '--no-tags', '--depth', '1', REPO_URL, str(CODE_ROOT))
else:
    if git('-C', str(CODE_ROOT), 'remote', 'get-url', 'origin') != REPO_URL:
        raise RuntimeError('Cached source origin differs from REPO_URL.')
    if git('-C', str(CODE_ROOT), 'status', '--porcelain', '--untracked-files=normal'):
        raise RuntimeError('Cached source has local changes; use a new code cache.')
git('-C', str(CODE_ROOT), 'fetch', '--depth', '1', '--no-tags', 'origin', CODE_REF)
commit = git('-C', str(CODE_ROOT), 'rev-parse', '--verify', 'FETCH_HEAD^{commit}')
git('-C', str(CODE_ROOT), 'checkout', '--detach', commit)

ENV = dict(GIT_ENV, SC203_KAT_ROOT=str(CODE_ROOT), PYTHONPATH=str(CODE_ROOT / 'src'))
if HF_SECRET_NAME and not ENV.get('HF_TOKEN'):
    try:
        from kaggle_secrets import UserSecretsClient
        value = UserSecretsClient().get_secret(HF_SECRET_NAME)
        if value: ENV['HF_TOKEN'] = value.strip()
        del value
    except Exception:
        print('HF secret unavailable; using public/local model files.')
config_dir = Path(os.environ.get('SC203_CONFIG_DIR', '/kaggle/temp/sc203-kat-configs'))
config_dir.mkdir(parents=True, exist_ok=True)
CONFIG_PATH = config_dir / (RUN_CONFIG['method'] + '_' + RUN_CONFIG['mode'] + '.json')
CONFIG_PATH.write_text(json.dumps(RUN_CONFIG, ensure_ascii=False, indent=2), encoding='utf-8')
prepare = (
    'import json,sys; from pathlib import Path; '
    'from sc203_kat.runtime.kaggle import ensure_submodule,source_metadata; '
    'root=Path(sys.argv[1]); method=sys.argv[2]; '
    'ensure_submodule(root,method); metadata=source_metadata(root,method); '
    'print(json.dumps(metadata,indent=2))'
)
subprocess.run([sys.executable, '-c', prepare, str(CODE_ROOT), RUN_CONFIG['method']], cwd=CODE_ROOT, env=ENV, check=True)
print('Code:', commit, '| Method:', RUN_CONFIG['method'], '| Mode:', RUN_CONFIG['mode'])
'''
RUN = '''subprocess.run([sys.executable, '-m', 'sc203_kat', 'run', '--config', str(CONFIG_PATH)],
               cwd=CODE_ROOT, env=ENV, check=True)
'''
VIEW = '''output = Path(RUN_CONFIG['output_dir'])
print('Results:', output)
for path in sorted(output.glob('*')):
    if path.is_file(): print(path.name, f'{path.stat().st_size / 2**20:.2f} MiB')
'''

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ref', default=DEFAULT_REF, help='Git tag, branch, or full commit to run')
    args = parser.parse_args(argv)
    for method in METHODS:
        for mode in ('sanity', 'full'):
            config_path = ROOT / 'configs/experiments' / f'{method}_{mode}.json'
            conf = json.loads(config_path.read_text(encoding='utf-8-sig'))
            if conf.get('method') != method or conf.get('mode') != mode:
                raise ValueError(f'Config method/mode mismatch: {config_path}')
            config = ('# Source is cloned anonymously from the public repository.\n'
                      f'REPO_URL = {REPO_URL!r}\nCODE_REF = {args.ref!r}\n'
                      "HF_SECRET_NAME = 'HF_token'  # Optional Hugging Face secret\n"
                      'RUN_CONFIG = ' + pprint.pformat(conf, sort_dicts=False, width=100) + '\n')
            intro = (f'# SC203-KAT · {method} · {mode}\n'
                     'Enable Internet and select GPU T4 ×2. Run All: source and the required paper '
                     'submodule are cloned from public GitHub at pinned commits. Model weights, '
                     'dataset files and resume checkpoints can be supplied through Kaggle Input. '
                     'HF_token is optional for public Hugging Face resources.\n')
            nb = {'nbformat': 4, 'nbformat_minor': 5,
                  'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                               'language_info': {'name': 'python'}},
                  'cells': [cell('markdown', intro), cell('code', config), cell('code', BOOTSTRAP),
                            cell('code', RUN), cell('code', VIEW)]}
            dest = ROOT / 'notebooks' / method / f'{mode}.ipynb'
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
    print('Generated 10 GitHub-only notebooks; experiment configs preserved')

if __name__ == '__main__': main()
