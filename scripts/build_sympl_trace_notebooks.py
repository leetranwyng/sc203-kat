"""Build thin launchers using the standard sc203-kat Git bootstrap."""
import argparse
import ast
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def cell(source,markdown=False):
    if not markdown:ast.parse(source)
    return dict(cell_type='markdown' if markdown else 'code',metadata={},source=source.splitlines(True),
                **({} if markdown else dict(outputs=[],execution_count=None)))

def build(ref):
    base=json.loads((ROOT/'notebooks/sympl/sanity.ipynb').read_text(encoding='utf8'))
    bootstrap=''.join(base['cells'][2]['source'])
    folder=ROOT/'notebooks/sympl';folder.mkdir(exist_ok=True)
    filenames = {'sanity':'SymPL_7B_APC120_sanity_kaggle.ipynb',
                 'full':'SymPL_7B_APC120_stage_traces_kaggle.ipynb',
                 'recover':'SymPL_7B_APC120_recover_kaggle.ipynb'}
    for mode in ('sanity','full','recover'):
        cfg=dict(method='sympl',mode='sanity' if mode=='recover' else mode,trace_input=None,model_input=None,
            resume_input=None,run_smoke_first=True,
            runtime_dir='/kaggle/temp/sc203-kat/sympl_trace',
            output_dir=f'/kaggle/working/sc203-kat/sympl_trace_{mode}',
            method_options=dict(category_policy='dataset_mapping',max_new_tokens=1024,max_pixels=None,quantization=None,retry_errors=False))
        settings=f"REPO_URL = 'https://github.com/leetranwyng/sc203-kat.git'\nCODE_REF = {ref!r}\nHF_SECRET_NAME = 'HF_token'\nRUN_CONFIG = {cfg!r}\n"
        header=f'# SymPL trace120 — {mode}\n\nKaggle T4 x2, Internet ON. Add Input: dataset APC trace120 đã có.\n'
        header+='Notebook chỉ cấu hình và gọi code từ sc203-kat; không nhúng runtime. Các luồng model cũ không đổi.\n'
        header+='Sanity chạy 6 mẫu (2/task). PASS là kiểm tra kỹ thuật, không phải 6 câu đều đúng. Full mặc định qua sanity trước 120 mẫu.\n'
        command="subprocess.run([sys.executable, '-m', 'sc203_kat.sympl_trace', '--config', str(CONFIG_PATH)], cwd=CODE_ROOT, env=ENV, check=True)\n"
        boot=bootstrap
        if mode=='recover':
            header='# Phục hồi trace SymPL cũ\n\nChạy trong session còn thư mục output cũ, hoặc mount output và sửa OLD_OUTPUT. Không tải model hay chạy inference.\n'
            settings+="OLD_OUTPUT = '/kaggle/working/sympl_trace120_sanity_v2'\n"
            # Recovery needs no upstream submodule, native tools, or model assets.
            boot=boot[:boot.index('prepare = (')]
            command="subprocess.run([sys.executable, '-m', 'sc203_kat.sympl_trace', '--recover-output', OLD_OUTPUT], cwd=CODE_ROOT, env=ENV, check=True)\n"
        display="from IPython.display import display, FileLink\noutput = Path("+('OLD_OUTPUT' if mode=='recover' else "RUN_CONFIG['output_dir']")+")\nfor p in sorted(output.rglob('sanity_report.json')): print(p, p.read_text())\nfor p in sorted(output.parent.joinpath(output.name+'_downloads').glob('*.zip')): display(FileLink(str(p)))\n"
        nb=dict(nbformat=4,nbformat_minor=5,metadata=base['metadata'],cells=[cell(header,True),cell(settings),cell(boot),cell(command),cell(display)])
        for i,c in enumerate(nb['cells']):c['id']=f'trace-{mode}-{i}'
        (folder/filenames[mode]).write_text(json.dumps(nb,ensure_ascii=False,indent=1),encoding='utf8')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--ref',default='fix/sympl-trace120')
    build(parser.parse_args().ref)
