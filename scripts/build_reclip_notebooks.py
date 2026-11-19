"""Generate separate Kaggle full and sanity notebooks for both ReCLIP replays."""
import argparse
import json
import pprint
from pathlib import Path

from build_notebooks import BOOTSTRAP, REPO_URL, cell

ROOT = Path(__file__).resolve().parents[1]
SANITY_IDS = ['5832', '4798', '8268']
DATASET = 'ltrnnguyn/sc203-apc-trace-for-ablation'


def build(ref):
    directory = ROOT/'notebooks/reclip'
    directory.mkdir(parents=True,exist_ok=True)
    for method in ('apc_vis','sympl'):
        label = 'APC-Vis' if method=='apc_vis' else 'SymPL'
        for mode in ('sanity','full'):
            stem=f'{method}_reclip_{mode}'
            config=dict(method=method,mode=mode,selector='both' if method=='apc_vis' else 'paired',
                        sample_ids=SANITY_IDS.copy() if mode=='sanity' else None,seed=42,
                        trace_input=None,model_input=None,grounding_labels=None,
                        runtime_dir=f'/kaggle/temp/sc203_reclip/{stem}',
                        output_dir=f'/kaggle/working/{stem}')
            if method=='sympl':
                config['sympl_trace']=None
            settings=(f'REPO_URL = {REPO_URL!r}\nCODE_REF = {ref!r}\nHF_SECRET_NAME = "HF_token"\n'
                      'RUN_CONFIG = '+pprint.pformat(config,sort_dicts=False,width=98)+'\n')
            intro=(f'# {label} + ReCLIP | {mode}\n\n'
                   f'Kaggle settings: **GPU T4 ×2**, **Internet On**. Add Dataset **{DATASET}**. '
                   'The notebook finds the existing trace bundle in Kaggle Input. Set trace_input only if needed. '
                   'HF_token is optional for Hugging Face downloads.\n\n'
                   + ('**Three questions**, one per task. Run this before the full notebook. '
                      'Sanity passes when both selectors produce predictions for all three questions and at least one object has multiple ranked proposals.\n\n'
                      if mode=='sanity' else '**120 trace questions**, 40 per task. Use the corresponding sanity notebook first.\n\n')
                   + 'Runs both selectors on the same frozen inputs and proposal order. Original box selection is replayed in the same environment. '
                     'The historical answer is saved separately to measure baseline drift. '
                     'Only ReCLIP isolated proposal scoring is enabled: crop and blur, one OpenAI CLIP ViT-B/32 backbone, no relation resolver.\n\n'
                   + ('Historical SymPL calls and depth maps are fetched from the public saved Kaggle output. '
                      'You can optionally set sympl_trace to a local copy of that output if the public API is unavailable.\n\n' if method=='sympl' else '')
                   + f'Open gallery.html in {config["output_dir"]}. Download the full output directory with its images to retain their links. '
                     'The diagnostic cohort is not the full benchmark. Grounding accuracy requires confirmed box annotations.\n')
            prepare="""output = Path(RUN_CONFIG['output_dir'])
output.mkdir(parents=True, exist_ok=True)
if RUN_CONFIG['method'] == 'sympl' and not RUN_CONFIG.get('sympl_trace'):
    history = Path(RUN_CONFIG['runtime_dir']) / 'sympl_history'
    command = [sys.executable, '-u', '-m', 'sc203_kat.reclip.fetch_sympl', str(history)]
    if RUN_CONFIG['sample_ids']:
        command += ['--sample-ids', *RUN_CONFIG['sample_ids']]
    subprocess.run(command, cwd=CODE_ROOT, env=ENV, check=True)
    RUN_CONFIG['sympl_trace'] = str(history)
CONFIG_PATH.write_text(json.dumps(RUN_CONFIG, indent=2), encoding='utf-8')
"""
            run="""module = 'sc203_kat.reclip.apc' if RUN_CONFIG['method'] == 'apc_vis' else 'sc203_kat.reclip.sympl'
arguments = ['--config', str(CONFIG_PATH)] if RUN_CONFIG['method'] == 'apc_vis' else [str(CONFIG_PATH)]
worker = subprocess.run([sys.executable, '-u', '-m', module, *arguments], cwd=CODE_ROOT, env=ENV)
report = None
if (output / 'replay_predictions.jsonl').is_file():
    report = subprocess.run([sys.executable, '-u', '-m', 'sc203_kat.reclip.report', '--config', str(CONFIG_PATH)],
                            cwd=CODE_ROOT, env=ENV)
worker.check_returncode()
if report is None:
    raise RuntimeError(f'Runner finished without replay predictions. Inspect {output}.')
report.check_returncode()
"""
            view="""from IPython.display import FileLink, display
for name in ('gallery.html', 'summary.json', 'paired_predictions.csv',
             'grounding_decisions.jsonl', 'grounding_disagreements.jsonl'):
    path = output / name
    if path.is_file():
        display(FileLink(str(path)))
"""
            notebook=dict(nbformat=4,nbformat_minor=5,
                          metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),
                                        language_info=dict(name='python')),
                          cells=[cell('markdown',intro),cell('code',settings),cell('code',BOOTSTRAP),
                                 cell('code',prepare),cell('code',run),cell('code',view)])
            (directory/(stem+'.ipynb')).write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ref',default='v0.4.0-reclip')
    build(parser.parse_args().ref)
