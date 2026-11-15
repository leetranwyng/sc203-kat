"""Audit stored replay traces; labels describe observations, not causal fixes."""
import collections
import json
import re
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def norm(s):
    return s.strip().lower().replace("'", "")

def aggregate(rows, key):
    out = {}
    for label in sorted({r[key] for r in rows}):
        subset = [r for r in rows if r[key] == label]
        out[label] = {
            'rows': len(subset), 'wrong': sum(r['wrong'] for r in subset),
            'groups_with_wrong': len({r['group'] for r in subset if r['wrong']}),
            'wrong_ids': [r['id'] for r in subset if r['wrong']],
        }
    return out

with zipfile.ZipFile(ROOT / 'apc_step_c_replay.zip') as z:
    source = [json.loads(l) for l in z.read('apc_vis_predictions.jsonl').decode().splitlines()]
    apc = []
    for r in source:
        sid = r['sample_id']
        m = re.search(r"If I stand at the (.*?)'s position", r['question']) or re.search(r'Which object is the (.*?) facing towards', r['question'])
        assert m, sid
        expected = norm(m[1])
        scene_path = f'full/trace/apc_vis/{sid}/scene.json'
        scene = json.loads(z.read(scene_path)) if scene_path in z.namelist() else {}
        actual = norm(scene.get('viewer', 'unknown'))
        mismatch = actual != expected
        trace = r.get('traceback', '')
        if 'No objects to render' in trace:
            bucket = 'missing_reference_detection_render_error'
        elif 'color_dict[obj_idx]' in trace:
            bucket = 'color_palette_overflow'
        elif r['status'] != 'ok':
            bucket = 'other_pipeline_error'
        elif mismatch and r['paper_task'] == 'facing':
            # Facing can be answered from another frame; mismatch is not proof of error.
            bucket = 'facing_viewer_review'
        elif mismatch:
            bucket = 'viewer_camera_mismatch' if actual == 'camera' else 'viewer_other_object_mismatch'
        else:
            bucket = 'viewer_matches_root_cause_unresolved'
        calls = [c for c in r.get('vlm_calls', []) if 'extract the **perspective**' in c.get('prompt', '')]
        apc.append(dict(id=sid, group=r['circular_group'], task=r['paper_task'],
            wrong=r['prediction'] != r['answer'], status=r['status'], bucket=bucket,
            expected_viewer=expected, actual_viewer=actual, viewer_mismatch=mismatch,
            reference_present_in_scene=expected in {norm(k) for k in scene.get('camera', {})},
            viewer_raw=[c['raw_output'] for c in calls]))

with zipfile.ZipFile(ROOT / 'pcd_step_c_replay_results.zip') as z:
    source = json.loads(z.read('pcd_step_c_replay/predictions.json'))
    pcd = []
    for r in source:
        other = r['diagnostic_contrastive_prediction'] if r['branch'] == 'camera_direct' else r['diagnostic_camera_prediction']
        pcd.append(dict(id=r['qid'], group=r['base_qid'], task=r['category'],
            wrong=r['prediction'] != r['answer'], branch=r['branch'],
            other_branch_correct=other == r['answer'],
            branches_disagree=r['diagnostic_camera_prediction'] != r['diagnostic_contrastive_prediction'],
            extraction_fallback=r['extraction_fallback'], answer_parse_fallback=r['answer_parse_fallback']))

summary = {'scope': 'Selected replay only; rows include option permutations. Observed mismatches do not prove causal recovery.',
    'apc_by_failure': aggregate(apc, 'bucket'),
    'apc_by_task': {task: aggregate([r for r in apc if r['task'] == task], 'bucket') for task in sorted({r['task'] for r in apc})},
    'pcd_by_branch': aggregate(pcd, 'branch'),
    'pcd_wrong_other_correct': sum(r['wrong'] and r['other_branch_correct'] for r in pcd),
    'pcd_disagreements': sum(r['branches_disagree'] for r in pcd),
    'apc_rows': apc, 'pcd_rows': pcd}
(ROOT / 'failure_case_counts.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({k:v for k,v in summary.items() if not k.endswith('_rows')}, ensure_ascii=False, indent=2))
print('APC mismatch scene presence', collections.Counter((r['reference_present_in_scene'], r['wrong']) for r in apc if r['viewer_mismatch'] and r['status']=='ok'))
print('PCD fallbacks', sum(r['extraction_fallback'] for r in pcd), sum(r['answer_parse_fallback'] for r in pcd))
