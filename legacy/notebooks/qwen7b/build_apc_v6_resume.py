"""Build APC v6 resume notebooks from the two executed v5 .ipynb files."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

V5 = 'e4c4c99dd3bebb9694b12baaf29959d59e5fa55ec07c313c2e14e0781010d084'
OLD_POLICY = 'canonical_scene_key_and_explicit_question_subject_v1'
NEW_POLICY = 'unique_scene_key_with_quote_and_person_number_normalization_v2'
VIEWER_KEY = r'''
def viewer_key(value):
    # Only matching keys change. Keep the scene label used to index the geometry.
    value = canonical(value).replace('"', '').replace('`', '')
    value = re.sub(r"(?<!\w)'|'(?!\w)", "", value)
    value = re.sub(r"\bwomen\b", "woman", value)
    return re.sub(r"\s+", " ", value).strip()
'''
MIGRATION = '''
# The only protocol migration supported here is from this specific v5 runner.
V5_RUNTIME = "__V5__"
V5_VIEWER = "__OLD__"
V6_VIEWER = "__NEW__"


def compatible_resume_config(old, config):
    before = {k:v for k,v in old.items() if k not in OPERATIONAL_KEYS}
    after = {k:v for k,v in config.items() if k not in OPERATIONAL_KEYS}
    if before == after:
        return "same"
    if (before.get("runtime_sha") != V5_RUNTIME or
            before.get("reference_viewer_policy") != V5_VIEWER or
            after.get("reference_viewer_policy") != V6_VIEWER):
        raise RuntimeError("Unsupported checkpoint version")
    after.update(runtime_sha=V5_RUNTIME, reference_viewer_policy=V5_VIEWER)
    if before != after:
        raise RuntimeError("Resume protocol differs: check part, dataset, model and settings")
    return "v5"


def find_checkpoint(input_root, config):
    paths = []
    for path in Path(input_root).rglob("checkpoint.zip"):
        try:
            with zipfile.ZipFile(path) as archive:
                info = archive.getinfo("manifest.json")
                if info.file_size > 4*2**20:
                    continue
                previous = json.loads(archive.read(info))
            compatible_resume_config(previous["config"], config)
        except (OSError, ValueError, RuntimeError, KeyError, zipfile.BadZipFile):
            continue
        paths.append(path)
    if len(paths) != 1:
        raise RuntimeError(f"Expected one matching checkpoint.zip, found {len(paths)}. "
                           "Attach the correct Kaggle Input and set RESUME_CHECKPOINT explicitly.")
    return str(paths[0])


def import_resume_records(source, predictions, manifest, rows):
    source, predictions = Path(source), Path(predictions)
    old = json.loads((source/"manifest.json").read_text(encoding="utf-8"))
    kind = compatible_resume_config(old["config"], manifest["config"])
    if kind == "v5":
        # Rebuild the OLD fingerprint with the newly prepared rows/assets and the
        # two old code identifiers. All other identity fields must still match.
        config = dict(manifest["config"], runtime_sha=V5_RUNTIME, reference_viewer_policy=V5_VIEWER)
        expected = protocol_fingerprint(dict(manifest, config=config), rows)
    else:
        expected = manifest["fingerprint"]
    if old["fingerprint"] != expected or protocol_fingerprint(old, rows) != expected:
        raise RuntimeError("Resume fingerprint mismatch: dataset rows, assets or source differ")
    previous = read_records(source/"run/predictions_apc_vis.jsonl")
    if not previous:
        raise RuntimeError("Checkpoint does not contain any predictions")
    allowed = {str(row["sample_id"]):row for row in rows}
    kept, failed = [], []
    for sid, record in previous.items():
        row = allowed.get(sid)
        if (row is None or record.get("fingerprint") != expected or record.get("mode") != "apc_vis"
                or record.get("paper_task") != row["paper_task"]):
            raise RuntimeError("Invalid checkpoint record: " + sid)
        if successful(record) and record.get("prediction") not in row["choices"]:
            raise RuntimeError("Invalid option in checkpoint record: " + sid)
        if kind == "v5" and not successful(record):
            failed.append(record)
            continue
        if kind == "v5":
            record = dict(record, fingerprint=manifest["fingerprint"],
                          resumed_from_fingerprint=expected, resumed_from_runtime_sha=V5_RUNTIME)
        kept.append(record)
    if kind == "v5":
        report_dir = predictions.parent.parent
        atomic_json(report_dir/"resume_migration.json", dict(
            origin_fingerprint=expected, destination_fingerprint=manifest["fingerprint"],
            kept=len(kept), previous_failures=failed, pending=len(rows)-len(kept)))
    temp = predictions.with_suffix(".jsonl.tmp")
    with temp.open("w", encoding="utf-8") as stream:
        for record in kept:
            stream.write(json.dumps(record, ensure_ascii=False)+"\\n")
    temp.replace(predictions)
    print(f"Resume {kind}: loaded {len(kept)} records; {len(rows)-len(kept)} pending", flush=True)
'''.replace('__V5__', V5).replace('__OLD__', OLD_POLICY).replace('__NEW__', NEW_POLICY)


def swap(text, old, new):
    if text.count(old) != 1:
        raise ValueError('Unexpected v5 source: '+old[:110])
    return text.replace(old, new, 1)


def patch_reference(s):
    s = swap(s, '\ndef parse_objects(text):', VIEWER_KEY+'\n\ndef parse_objects(text):')
    start = s.index('    def unique(label, include_camera=False):')
    end = s.index('    # An explicit, uniquely', start)
    s = s[:start]+'''    def matching(label, include_camera=False):
        key = viewer_key(label) if label is not None else ""
        return [obj for obj in scene if key and (include_camera or viewer_key(obj) != "camera")
                and viewer_key(obj) == key]

    expected_keys, proposed_keys = matching(subject), matching(candidate)
    expected = expected_keys[0] if len(expected_keys) == 1 else None
    proposed = proposed_keys[0] if len(proposed_keys) == 1 else None
'''+s[end:]
    s = swap(s, '    elif proposed is not None:', '''    elif len(expected_keys) > 1:
        viewer, strategy = None, "ambiguous_question_subject"
    elif proposed is not None:''')
    s = swap(s, 'viewer, strategy = unique("camera", include_camera=True), "explicit_camera_question"',
             '''camera_keys = matching("camera", include_camera=True)
        viewer, strategy = (camera_keys[0] if len(camera_keys) == 1 else None), "explicit_camera_question"''')
    s = swap(s, 'available_objects=list(scene), resolved_viewer=viewer, strategy=strategy)',
             '''available_objects=list(scene), subject_matches=expected_keys, candidate_matches=proposed_keys,
                 resolved_viewer=viewer, strategy=strategy)''')
    return s.replace(OLD_POLICY, NEW_POLICY)


def patch_guardrails(s):
    s = swap(s, 'import hashlib', 'import errno\nimport hashlib')
    s = swap(s, '        if record["status"] == "ok":', '''        # Local failures are still persisted/scored, but do not stop full runs.
        if self.config["run_kind"] == "full" and record.get("error_type") in (
                "ReferenceViewerError", "InvalidAnswer"):
            self.consecutive = 0
            return None
        if record["status"] == "ok":''')
    return swap(s, '    text = str(exc).lower()', '''    if isinstance(exc, MemoryError) or (isinstance(exc, OSError) and exc.errno in
            (errno.ENOSPC, errno.EDQUOT, errno.EIO)):
        return True
    text = str(exc).lower()''')


def patch_checkpoint(s):
    s = swap(s, '    OPERATIONAL_KEYS, atomic_json, export_safely, make_report, read_records,',
             '    OPERATIONAL_KEYS, atomic_json, export_safely, make_report, protocol_fingerprint, read_records,')
    start = s.index('    old_config = ')
    end = s.index('    if not (source/', start)
    s = s[:start]+'    compatible_resume_config(manifest["config"], config)\n'+s[end:]
    return swap(s, '\ndef select_samples(rows, config):', MIGRATION+'\n\ndef select_samples(rows, config):')


def patch_worker(s):
    start = s.index('    if config.get("resume_input") and not predictions.exists():')
    end = s.index('    records = read_records(predictions', start)
    return s[:start]+'''    if config.get("resume_input") and not predictions.exists():
        from apc_full_checkpoint import import_resume_records
        import_resume_records(config["resume_input"], predictions, manifest, rows)
'''+s[end:]


PATCH = {'apc_full_reference.py':patch_reference, 'apc_kaggle_guardrails.py':patch_guardrails,
         'apc_full_checkpoint.py':patch_checkpoint, 'apc_v4_worker.py':patch_worker}


def build(path, part):
    nb = json.loads(path.read_text(encoding='utf-8-sig'))
    def cell(i): return ''.join(nb['cells'][i]['source'])
    def set_cell(i,s): nb['cells'][i]['source'] = s.splitlines(keepends=True)
    bootstrap = cell(1)
    updated = {}
    for node in ast.parse(bootstrap).body:
        if not isinstance(node,ast.Expr) or not isinstance(node.value,ast.Call):
            continue
        call = node.value
        if not isinstance(call.func,ast.Attribute) or call.func.attr!='write_text':
            continue
        target = ast.get_source_segment(cell(1),call.func.value)
        name = next((k for k in PATCH if target==f"SCRIPTS/'{k}'"), None)
        if name:
            updated[name] = (ast.get_source_segment(cell(1),call.args[0]),PATCH[name](ast.literal_eval(call.args[0])))
    assert set(updated) == set(PATCH)
    for old, new in updated.values():
        compile(new,'<embedded APC resume module>','exec')
        bootstrap = swap(bootstrap,old,repr(new))
    runtime = hashlib.sha256(''.join(new for _,new in updated.values()).encode()).hexdigest()
    bootstrap = swap(bootstrap, 'from apc_full_checkpoint import prepare_resume,export_full',
                     'from apc_full_checkpoint import find_checkpoint,prepare_resume,export_full')
    bootstrap = swap(bootstrap, 'Runtime v5 written: strict reference viewer, bounded traces, portable full checkpoint.',
                     'Runtime v6 written: normalized viewer labels and v5 checkpoint migration.')
    set_cell(1,bootstrap)
    config = cell(3)
    assert f"'experiment': 'apc_qwen7b_full_part{part}_v5'" in config and V5 in config
    config = config.replace(f'apc_qwen7b_full_part{part}_v5',f'apc_qwen7b_full_part{part}_v6_resume')
    config = config.replace(V5,runtime).replace(OLD_POLICY,NEW_POLICY)
    config = swap(config,'# To continue a previous v5 run, attach checkpoint.zip as Kaggle Input,\n'
                   '# then replace resume_input=None above with its real /kaggle/input/... path.',
                   '# None finds the one matching checkpoint.zip among Kaggle inputs.\n'
                   '# Set an explicit path if multiple checkpoints from this part are attached.')
    config = swap(config,'    prepare_resume(CFG,WORK)',
                   '    CFG["resume_input"] = RESUME_CHECKPOINT or find_checkpoint("/kaggle/input", CFG)\n'
                   '    prepare_resume(CFG,WORK)')
    set_cell(3,'RESUME_CHECKPOINT = None  # Or "/kaggle/input/.../checkpoint.zip"\n\n'+config)
    set_cell(0,f'''# APC-Vis 7B · Part {part} · Resume v6

Chọn GPU **T4 ×2**, bật Internet, và gắn **checkpoint.zip của chính Part {part}** vào Kaggle Input. Sau đó Run All.
Notebook tự chọn checkpoint khi chỉ có một bản khớp. Nếu có nhiều bản, điền đường dẫn vào `RESUME_CHECKPOINT` trong cell cấu hình.

Checkpoint v5 được kiểm tra trước khi nhập. Dự đoán hợp lệ được giữ lại. Mẫu lỗi v5 được thử lại cùng các mẫu chưa chạy. Lỗi viewer theo mẫu được lưu và chấm như cũ, rồi worker tiếp tục. Lỗi hệ thống vẫn dừng.
Nếu hết ngân sách phiên, dùng checkpoint.zip mới để chạy tiếp bằng chính notebook này. Giữ results.zip và traces.zip của các phiên để phân tích.
''')
    set_cell(2,'## Chọn checkpoint cho lượt resume\n\nGiữ cấu hình T4 ×2, Qwen 7B và phạm vi benchmark của v5. Chỉ chạy APC-Vis.')
    for c in nb['cells']:
        if c['cell_type']=='code':
            ast.parse(''.join(c['source']))
            c.update(outputs=[],execution_count=None)
    target = Path(__file__).resolve().parent/f'APC_Qwen7B_part{part}_v6_resume_kaggle.ipynb'
    target.write_text(json.dumps(nb,indent=1,ensure_ascii=False)+'\n',encoding='utf-8')
    print(target.name)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('part1',type=Path)
    parser.add_argument('part2',type=Path)
    args=parser.parse_args()
    for part,path in enumerate((args.part1,args.part2),1):
        build(path,part)
