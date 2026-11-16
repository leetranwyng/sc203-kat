"""Three thin GitHub-backed notebooks for controlled APC trace interventions."""
import argparse
import json
import pprint
from pathlib import Path
from build_notebooks import BOOTSTRAP,cell,REPO_URL
ROOT=Path(__file__).resolve().parents[1]

def main(ref='v0.3.0-ablation.1'):
    for study, variants in [('sanity',['sam3','oneref','oa2']),('01_sam3_oneref',['sam3','oneref']),('02_orientanything_v2',['oa2'])]:
        mode='sanity' if study=='sanity' else 'full'
        config=dict(method='apc_vis',mode=mode,study=study,trace_input=None,model_input=None,resume_input=None,
            output_dir=f'/kaggle/working/apc_ablation_{study}',runtime_dir=f'/kaggle/temp/apc_ablation_runtime/{study}',
            variants=variants,sample_ids=['5832'] if mode=='sanity' else None,
            weight_inputs=dict(sam3=None,oneref=None,oneref_tokenizer=None,oa2=None),
            sam3_threshold=.5,oneref_min_iou=0.0,seed=42,sample_timeout_seconds=900,
            phase_timeout_seconds=21600,disk_min_free_gib=3.0,retry_errors=False)
        intro='''# APC-Vis 7B · perception ablation · MODE

Kaggle: **GPU T4 ×2**, Internet ON, Secret **HF_token** with SAM 3 access.
Add the `sc203-apc-trace120` dataset prepared alongside these notebooks. Set `trace_input`
only if auto-detection finds several bundles. Code is cloned from public sc203-kat.

This is **controlled replay**, not a new full-dataset baseline: preserve the historical
object descriptions, selected viewer and Depth Pro map. Recompute downstream geometry
when a selected box/mask changes; rerender and query the same Qwen 7B in FP16.

Baseline results are imported from the existing trace bundle; baseline inference is never run.
Sanity runs all three branches. Full notebook 01 runs SAM 3 + OneRef; notebook 02 runs OA V2. Optional combinations: `sam3_oa2`, `oneref_oa2`, `sam3_oneref`,
`sam3_oneref_oa2`. Use a new output directory for
a changed experiment. Sanity uses one image; each full notebook uses all supplied diagnostic traces for its branches.

OneRef sees the original object description and full image, not the requested spatial
answer. It ranks the same top-5 proposals by overlap with its predicted REC box; it cannot
recover an instance missing from those proposals. No hidden fallback to object 0.

Runtime environments/weights stay in `/kaggle/temp`; evidence stays in `/kaggle/working`.
Optional `weight_inputs` avoid auxiliary downloads. `model_input` is the existing APC
asset bundle, not just Qwen weights. To resume, add the previous Output as Input and set
`resume_input` to its `apc_ablation_MODE` folder. Only identical code/config/cohort resume.
'''.replace('MODE',study)
        text=f"REPO_URL = {REPO_URL!r}\nCODE_REF = {ref!r}\nHF_SECRET_NAME = 'HF_token'\nRUN_CONFIG = "+pprint.pformat(config,sort_dicts=False,width=100)+'\n'
        run="""# This orchestration process does not load ML models in the notebook kernel.
subprocess.run([sys.executable, '-u', '-m', 'sc203_kat.ablation.run', '--config', str(CONFIG_PATH)],
               env=ENV, cwd=CODE_ROOT, check=True)
"""
        view="""from IPython.display import FileLink, display
output = Path(RUN_CONFIG['output_dir'])
if (output / 'summary.json').exists():
    summary = json.loads((output / 'summary.json').read_text())
    import pandas as pd
    display(pd.DataFrame(summary['by_task']))
    display(pd.DataFrame(summary['pairs']))
    display(FileLink(str(output / 'index.html')))
    print('Download the complete output folder to keep the HTML images and raw traces.')
    print('Notebook finished != sanity PASS: inspect error status and each adapter output before the full notebooks.')
else:
    print('No summary yet. Inspect the setup cell and logs; rerun after fixing the reported failure.')
"""
        export="""# Optional: archive RESULTS only; source code always comes from GitHub.
# Keep false to avoid storing a second copy when Kaggle disk is low.
EXPORT_RESULT_ZIP = False
if EXPORT_RESULT_ZIP:
    import shutil
    if shutil.disk_usage(output).free < 2 * sum(p.stat().st_size for p in output.rglob('*') if p.is_file()):
        raise RuntimeError('Not enough disk to duplicate evidence as ZIP; download the output folder instead.')
    archive = shutil.make_archive(str(output), 'zip', root_dir=output.parent, base_dir=output.name)
    display(FileLink(archive))
"""
        nb=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),
                 language_info=dict(name='python')),cells=[cell('markdown',intro),cell('code',text),cell('code',BOOTSTRAP),
                 cell('code',run),cell('code',view),cell('code',export)])
        dest=ROOT/'notebooks/apc_ablation'/f'{study}.ipynb';dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_text(json.dumps(nb,indent=1,ensure_ascii=False)+'\n',encoding='utf8')
        conf=ROOT/'configs/ablations/experiments'/f'{study}.json';conf.parent.mkdir(parents=True,exist_ok=True);conf.write_text(json.dumps(config,indent=2)+'\n',encoding='utf8')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--ref',default='v0.3.0-ablation.1');main(p.parse_args().ref)
