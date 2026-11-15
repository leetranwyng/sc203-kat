"""Generate fresh-run APC notebooks from the original executed v5 parts."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

from build_apc_v6_resume import NEW_POLICY, OLD_POLICY, V5, patch_guardrails, patch_reference, swap


PATCH = {
    'apc_full_reference.py': patch_reference,
    'apc_kaggle_guardrails.py': patch_guardrails,
}


def build(path, part):
    notebook = json.loads(path.read_text(encoding='utf-8-sig'))

    def source(index):
        return ''.join(notebook['cells'][index]['source'])

    def set_source(index, text):
        notebook['cells'][index]['source'] = text.splitlines(keepends=True)

    original = source(1)
    bootstrap = original
    patched = {}
    for node in ast.parse(original).body:
        if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
            continue
        call = node.value
        if not isinstance(call.func, ast.Attribute) or call.func.attr != 'write_text':
            continue
        target = ast.get_source_segment(original, call.func.value)
        name = next((name for name in PATCH if target == f"SCRIPTS/'{name}'"), None)
        if name is None:
            continue
        replacement = PATCH[name](ast.literal_eval(call.args[0]))
        compile(replacement, name, 'exec')
        patched[name] = replacement
        bootstrap = swap(bootstrap, ast.get_source_segment(original, call.args[0]), repr(replacement))
    assert set(patched) == set(PATCH)
    runtime = hashlib.sha256(''.join(patched[name] for name in sorted(patched)).encode()).hexdigest()
    bootstrap = swap(bootstrap, 'from apc_full_checkpoint import prepare_resume,export_full',
                     'from apc_full_checkpoint import export_full')
    bootstrap = swap(bootstrap, 'Runtime v5 written: strict reference viewer, bounded traces, portable full checkpoint.',
                     'Runtime v6 written: viewer matching and sample-level error handling.')
    set_source(1, bootstrap)

    config = source(3)
    config = swap(config, f'apc_qwen7b_full_part{part}_v5', f'apc_qwen7b_full_part{part}_v6_fresh')
    config = swap(config, V5, runtime)
    config = swap(config, OLD_POLICY, NEW_POLICY)
    config = swap(config, '# To continue a previous v5 run, attach checkpoint.zip as Kaggle Input,\n'
                          '# then replace resume_input=None above with its real /kaggle/input/... path.',
                  '# A fresh output folder for every Run All. No checkpoint is imported.')
    config = swap(config, "assert Path(CFG['experiment']).name==CFG['experiment']",
                  """assert CFG['resume_input'] is None, 'Fresh-run notebook does not import a checkpoint.'
from uuid import uuid4
CFG['experiment'] += '_' + uuid4().hex[:12]""")
    config = swap(config,
                  "OUT=Path(CFG['output']);OUT.mkdir(parents=True,exist_ok=True)\n"
                  "with setup_stage('resume_configuration',OUT,CFG):\n    prepare_resume(CFG,WORK)",
                  "OUT=Path(CFG['output']);OUT.mkdir(parents=True,exist_ok=False)")
    set_source(3, config)

    title, count = ('Left/Right + Visibility', 1386) if part == 1 else ('Object Facing', 692)
    set_source(0, f'''# APC-Vis 7B · Part {part} · v6 chạy từ đầu

Chọn **GPU T4 ×2**, bật **Internet** và bấm **Run All**.
Phạm vi gồm {title}, **{count:,} lượt**, với đủ hoán vị đáp án.
Mỗi lần Run All tạo thư mục kết quả mới. Chỉ chạy APC-Vis.

Khớp viewer theo tên chuẩn hóa và yêu cầu khớp duy nhất.
Lỗi viewer và lỗi parse đáp án theo mẫu được ghi lại rồi worker tiếp tục.
''')
    set_source(2, '## Cấu hình\n\nGiữ cấu hình Qwen 7B, T4 ×2 và benchmark của v5. Không cần gắn checkpoint.')
    set_source(10, '''## Giám sát full và bảo vệ output

Lỗi viewer và đáp án không parse được được lưu theo từng mẫu. Lỗi hệ thống vẫn dừng.
Giữ ngân sách worker 7 giờ, ngưỡng đĩa và chính sách trace của v5.
''')
    set_source(11, '''from apc_launcher import launch as launch_full

def launch():
    if (OUT/'run/predictions_apc_vis.jsonl').exists():
        raise RuntimeError('This output already contains a run. Use Run All for a fresh run.')
    return launch_full(CFG,PY,SCRIPTS,ENV)
''')
    set_source(12, '''## Chạy full từ mẫu đầu tiên

Nếu đến giới hạn 7 giờ trước khi thử hết, trạng thái là PAUSED và output phản ánh phần đã chạy.
''')
    set_source(13, swap(source(13),
                        "    print('Save checkpoint.zip, attach it as Kaggle Input in the next session, set resume_input, and Run All.')",
                        "    print('Reached the worker time budget. Results so far are in Output.')"))
    set_source(14, '''## Lưu kết quả

`full_report.json` ghi coverage, lỗi và accuracy.
`results.zip`, `traces.zip` và `checkpoint.zip` lưu kết quả của phiên này.
Khi tổng hợp hai part, cộng số đúng và mẫu số.
''')
    set_source(15, swap(source(15),
                       "print('Keep checkpoint.zip until this part reaches complete=True. Never combine the two part accuracies by a plain mean.')",
                       "print('Download results and traces from this run. Check coverage before comparing accuracy.')"))
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            ast.parse(''.join(cell['source']))
            cell.update(outputs=[], execution_count=None)
    out = Path(__file__).resolve().parent / f'APC_Qwen7B_part{part}_v6_fresh_kaggle.ipynb'
    out.write_text(json.dumps(notebook, indent=1, ensure_ascii=False)+'\n', encoding='utf-8')
    print(out.name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('part1', type=Path)
    parser.add_argument('part2', type=Path)
    args = parser.parse_args()
    for part, path in enumerate((args.part1, args.part2), 1):
        build(path, part)
