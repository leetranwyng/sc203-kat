"""Build the three APC-Vis reruns from the Kaggle pull response (or its .ipynb)."""
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
from pprint import pformat

HERE = Path(__file__).resolve().parent
ORIENTATION = 'apc/vision_modules/orientation.py'
CONVERSION = '        direction_vector = self.orientation_to_direction(azimuth_deg, polar_deg)'
SPLITS = (
    ('sanity', 'Sanity · một ảnh', ('visibility',), 1, '4798'),
    ('full_part1_left_right_visibility', 'Full phần 1 · Left/Right và Visibility',
     ('left_right', 'visibility'), 1386, None),
    ('full_part2_facing', 'Full phần 2 · Object Facing', ('facing',), 692, None),
)

SELECT_TASKS = """    if cfg.get('paper_tasks'):
        tasks = set(cfg['paper_tasks'])
        rows = [r for r in rows if r.get('paper_task') in tasks]
    if cfg.get('sanity_sample_id') is not None:
        rows = [r for r in rows if str(r['sample_id']) == str(cfg['sanity_sample_id'])]
        if len(rows) != 1:
            raise ValueError('Expected exactly one sanity sample: '+str(cfg['sanity_sample_id']))
"""

VERIFY_SANITY = """# Verify a real APC result, not only successful model initialization.
launch(smoke=True)
folder = OUT/'smoke'
records = [json.loads(line) for line in (folder/'predictions_apc_vis.jsonl').read_text(encoding='utf-8').splitlines()]
record = records[-1]
sample = json.loads((OUT/'samples.jsonl').read_text(encoding='utf-8').splitlines()[0])
assert record['sample_id'] == sample['sample_id']
assert record['status'] == 'ok' and record['prediction'] in sample['choices'], record
trace = folder/'trace'/'apc_vis'/sample['sample_id']
assert (trace/'scene.json').is_file(), 'Missing transformed scene'
assert (trace/'apc_visual_prompt.png').is_file(), 'Missing APC render'
import math
orientation_files = sorted(trace.glob('orientation_angles_*.json'))
assert orientation_files, 'Missing Orient Anything trace'
for file in orientation_files:
    vector = json.loads(file.read_text(encoding='utf-8')).get('direction_vector')
    assert isinstance(vector, list) and len(vector) == 3 and all(math.isfinite(v) for v in vector), file
print('Sanity PASS', {'sample_id': sample['sample_id'], 'prediction': record['prediction'],
                      'gold': sample['answer'], 'orientation_records': len(orientation_files)})
"""


def source(notebook, index):
    return ''.join(notebook['cells'][index]['source'])


def set_source(notebook, index, text):
    notebook['cells'][index]['source'] = text.splitlines(keepends=True)


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f'Expected one occurrence of {old!r}')
    return text.replace(old, new, 1)


def patch_cells(notebook):
    patch_cell = source(notebook, 9)
    assignment = ast.parse(patch_cell).body[0]
    patches = ast.literal_eval(assignment.value)
    bad_anchor, bad_replacement = patches[ORIENTATION][0]
    assert bad_anchor == '        return direction_vector'
    assert "record['direction_vector']" in bad_replacement
    # These variables are local to run_orientation_estimation. Keep the original
    # raw-angle JSON and overwrite it with the vector once conversion succeeds.
    write_vector = bad_replacement.removesuffix('\n'+bad_anchor)
    patches[ORIENTATION][0] = (CONVERSION, CONVERSION+'\n'+write_vector)
    selection = "    if cfg.get('limit') is not None:"
    patches["scripts/apc_common.py"] = [(selection, SELECT_TASKS+selection)]

    # Descriptions in the manifests and summary should reflect the only active mode.
    old_protocol = 'Qwen2.5-VL-7B direct/APC-Vis'
    if old_protocol in source(notebook, 1):
        for name in ('apc_score.py', 'apc_prepare_data.py'):
            bootstrap = source(notebook, 1)
            node = next((node.value for node in ast.parse(bootstrap).body
                         if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                         and isinstance(node.value.func, ast.Attribute)
                         and node.value.func.attr == 'write_text'
                         and name in (ast.get_source_segment(bootstrap, node.value.func.value) or '')), None)
            if node and isinstance(node.args[0], ast.Constant) and old_protocol in node.args[0].value:
                patches['scripts/'+name] = [(old_protocol, 'Qwen2.5-VL-7B APC-Vis')]

    updated = replace_once(patch_cell, ast.get_source_segment(patch_cell, assignment.value), repr(patches))
    set_source(notebook, 9, updated)
    return hashlib.sha256((source(notebook, 1)+updated).encode()).hexdigest()


def build(base, key, title, tasks, expected, sanity_id):
    notebook = copy.deepcopy(base)
    runtime_sha = patch_cells(notebook)
    config_cell = source(notebook, 3)
    cfg_ast = ast.parse(config_cell).body[0]
    cfg = ast.literal_eval(cfg_ast.value)
    assert cfg['experiment'] == 'apc_qwen7b_kaggle_dispatch_v3'
    assert cfg['model_revision'] == 'cc594898137f460bfe9f0759e9844b3ce807cfb5'
    cfg.update(modes=['apc_vis'], experiment=f'apc_qwen7b_{key}_record_fix_v1',
               paper_tasks=list(tasks), sanity_sample_id=sanity_id, runtime_sha=runtime_sha)
    # Keep dataset preparation in benchmark mode, including its labels.
    # The worker uses --smoke for its actual one-image execution.
    assert cfg['sanity'] is False
    config_cell = replace_once(config_cell, ast.get_source_segment(config_cell, cfg_ast.value),
                               pformat(cfg, sort_dicts=False, width=112))
    config_cell = replace_once(config_cell,
        "assert CFG['modes'] and len(set(CFG['modes']))==len(CFG['modes']) and set(CFG['modes'])<={'direct','apc_vis'}",
        "assert CFG['modes'] == ['apc_vis'], 'APC-Vis only'")
    set_source(notebook, 3, config_cell)
    prepare = source(notebook, 9)
    prepare += f"""
prepared = json.loads((OUT/'manifest.json').read_text(encoding='utf-8'))
if CFG['limit'] is None:
    assert prepared['selected_rows'] == {expected}, prepared['selected_rows']
"""
    set_source(notebook, 9, prepare)

    how = ('Chạy APC-Vis trên mẫu 4798. PASS yêu cầu đáp án hợp lệ, vector hữu hạn, scene và ảnh render.'
           if sanity_id else
           f'Chạy APC-Vis trên {expected:,} lượt. Các hoán vị đáp án của mỗi nhóm nằm cùng một notebook.')
    set_source(notebook, 0, f'''# APC-Vis · Qwen2.5-VL-7B · {title}

Kaggle **GPU T4 ×2**, Internet và Secret **HF_token**. Chọn **Run All**.

{how}

Cấu hình model, phân bổ GPU và trace giữ theo lượt Kaggle dispatch v3.
Output có thư mục và tên ZIP riêng cho lượt này.
''')
    set_source(notebook, 2, f"""## Cấu hình

Chỉ chạy mode `apc_vis`. Phạm vi của notebook là {', '.join(tasks)}.
""")
    set_source(notebook, 12, '## Kiểm tra một ảnh' if sanity_id else '## Chạy các subtasks của phần này')
    set_source(notebook, 13, VERIFY_SANITY if sanity_id else 'launch(smoke=False)\n')
    if sanity_id:
        export = source(notebook, 15)
        scorer = 'subprocess.run([PY,str(SCRIPTS/"apc_score.py"),str(OUT)],env=ENV,cwd=REPO,check=True)'
        set_source(notebook, 15, replace_once(export, scorer,
            '# Sanity results are in smoke/predictions_apc_vis.jsonl.'))
        set_source(notebook, 14, '## Tải kết quả sanity và trace')
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            cell.update(outputs=[], execution_count=None)
            ast.parse(''.join(cell['source']))
    return notebook


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('kaggle_source', type=Path)
    args = parser.parse_args()
    original = json.loads(args.kaggle_source.read_text(encoding='utf-8-sig'))
    if 'blob' in original:
        original = json.loads(original['blob']['sourceNullable'])
    for spec in SPLITS:
        path = HERE/f'APC_Qwen7B_{spec[0]}_fixed_kaggle.ipynb'
        notebook = build(original, *spec)
        path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False)+'\n', encoding='utf8')
        print(path.name)


if __name__ == '__main__':
    main()
