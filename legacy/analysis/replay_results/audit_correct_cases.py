"""Audit successful answers without treating answer accuracy as pipeline validity."""
import collections
import io
import json
import re
import zipfile
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
def norm(s):
    return re.sub(r'\s+', ' ', s.lower().replace('"', '').replace("'", '').replace(' that is ', ' ')).strip()

out = {'scope': 'Selected replay, option permutations counted separately; no 3D ground truth available.', 'apc': [], 'pcd': []}
with zipfile.ZipFile(ROOT/'apc_step_c_replay.zip') as z:
    rows = [json.loads(l) for l in z.read('apc_vis_predictions.jsonl').decode().splitlines()]
    groups = collections.defaultdict(list)
    for r in rows: groups[r['circular_group']].append(r)
    for r in rows:
        if r['prediction'] != r['answer']: continue
        sid = r['sample_id']; pre = f'full/trace/apc_vis/{sid}/'
        scene = json.loads(z.read(pre+'scene.json'))
        calls = r['vlm_calls']; flags=[]; evidence={}
        q = r['question']
        m = re.search(r"If I stand at the (.*?)'s position", q) or re.search(r'Which object is the (.*?) facing towards', q)
        ref = m[1]; viewer=scene['viewer']; camera=scene['camera']
        if norm(ref) != norm(viewer):
            flags.append('explicit_viewer_mismatch' if r['paper_task']!='facing' else 'facing_viewer_differs_not_proven_error')
        evidence.update(reference=ref, viewer=viewer)
        expected=[ref]
        if r['paper_task']=='facing': expected += list(r['choices'].values())
        else:
            m=re.search(r'facing, is the (.*?) (?:visible or not|on the left or right of me)\?',q)
            assert m, sid
            expected.append(m[1])
            if norm(viewer)==norm(m[1]): flags.append('target_selected_as_viewer_excluded_from_render')
        scene_names={norm(k) for k in camera}
        aliases={'lime green car':'limingreen car', 'shepherd':'person'}
        absent=[v for v in expected if norm(v) not in scene_names and aliases.get(norm(v)) not in scene_names]
        if absent: flags.append('required_entity_absent_from_scene');evidence['absent_entities']=absent
        detectors=[json.loads(l) for l in z.read(pre+'detector.jsonl').decode().splitlines()]
        missed=[d['category'] for d in detectors if not d['boxes']]
        if missed: flags.append('zero_detection_candidates');evidence['zero_detection']=missed
        if any('real' in k and 'locations' in k and 'orientations' in k for k in camera): flags.append('boilerplate_extracted_as_object')
        if sid=='6098': flags.append('extra_objects_beyond_question');evidence['extra_objects']=['drywall','hammer','paper towel roll','sponge','sink']
        raw_view=next(c['raw_output'] for c in calls if 'extract the **perspective**' in c.get('prompt',''))
        parsed=re.findall(r'\+\+(.*?)\+\+', raw_view)
        parsed=(parsed[-1] if parsed else raw_view).strip().lower().replace("'",'')
        if parsed not in camera and viewer=='camera': flags.append('viewer_parser_fallback_to_camera')
        evidence['viewer_raw']=raw_view
        integrity=[]
        for name,obj in camera.items():
            pos=np.asarray(obj['position']);ori=np.asarray(obj['orientation'])
            if not np.isfinite(pos).all() or not np.isfinite(ori).all(): integrity.append(f'{name}: nonfinite scene')
            if abs(np.linalg.norm(ori)-1)>0.01: integrity.append(f'{name}: orientation not unit')
        depth=np.load(io.BytesIO(z.read(pre+'depth.npy')),allow_pickle=False)
        if not np.isfinite(depth).all() or (depth<=0).any(): integrity.append('invalid depth values')
        im=Image.open(io.BytesIO(z.read(pre+'input.png')));w,h=im.size
        for n in z.namelist():
            if not n.startswith(pre) or not n.endswith('_selected_box.npy'):continue
            box=np.load(io.BytesIO(z.read(n)),allow_pickle=False)
            if not np.isfinite(box).all() or box[2]<=box[0] or box[3]<=box[1]:integrity.append(n+': invalid box')
            maskpath=n.replace('_selected_box.npy','_mask.png')
            mask=np.asarray(Image.open(io.BytesIO(z.read(maskpath))))
            if mask.shape!=(h,w) or not np.any(mask):integrity.append(maskpath+': invalid/empty mask')
        if integrity: flags.append('numeric_or_shape_integrity_issue')
        evidence['integrity_issues']=integrity
        refinement=[]
        for c in calls:
            if 'Please return its index.' not in c.get('prompt',''):continue
            category=re.search(r"fits the description: '(.*?)'",c['prompt'])[1]
            namepath=next(n for n in z.namelist() if n.startswith(pre) and n.endswith('.name') and z.read(n).decode()==category)
            boxes=np.load(io.BytesIO(z.read(namepath[:-5]+'_boxes.npy')),allow_pickle=False)
            digits=[i for i in range(len(boxes)) if str(i) in c['raw_output']]
            if len(digits)!=1: refinement.append(dict(category=category, matched_indices=digits, raw=c['raw_output']))
        if refinement: flags.append('refinement_index_ambiguous_or_fallback');evidence['refinement_review']=refinement
        if any(not x['matched_indices'] for x in refinement): flags.append('refinement_default_index_zero')
        if not all(x['prediction']==x['answer'] for x in groups[r['circular_group']]): flags.append('paired_permutation_has_wrong_answer')
        if q.count('"')%2:flags.append('unbalanced_quote_in_source_question')
        # Manually reviewed semantic contradictions, not generic keyword inference.
        if sid in ['8388','8560','8841']: flags.append('answer_changes_meaning_between_stages')
        if sid in ['5416','5417']:flags.append('translation_mentions_wrong_entity')
        evidence['last_four_outputs']=[c['raw_output'] for c in calls[-4:]]
        out['apc'].append(dict(id=sid,group=r['circular_group'],task=r['paper_task'],flags=flags,evidence=evidence))

with zipfile.ZipFile(ROOT/'pcd_step_c_replay_results.zip') as z:
    rows=json.loads(z.read('pcd_step_c_replay/predictions.json'))
    cfg=json.loads(z.read('pcd_step_c_replay/config.json'));alpha=cfg['alpha']
    groups=collections.defaultdict(list)
    for r in rows:groups[r['base_qid']].append(r)
    for r in rows:
        if r['prediction']!=r['answer']:continue
        flags=[];checks={}
        m=re.search(r"If I stand at the (.*?)'s position facing where it is facing, is the (.*?) (?:in front of me or behind me|on the left or right of me)\?",r['question'])
        checks['extraction_matches_question']=bool(m) and [norm(x) for x in m.groups()]==[norm(r['ref_obj']),norm(r['target_obj'])]
        if not checks['extraction_matches_question']:flags.append('extraction_mismatch')
        expected_same=r['viewpoint_raw'].lower().startswith('yes' if r['category']=='orientation_on_the_left' else 'no')
        checks['branch_matches_implemented_rule']=expected_same==r['is_same_viewpoint'] and r['branch']==('camera_direct' if expected_same else 'pcd')
        checks['score_arithmetic_matches']=all(abs(s['orig']-alpha*s['cam']-s['contrastive'])<1e-6 for s in r['scores'].values())
        checks['prediction_matches_branch']=r['prediction']==r['diagnostic_camera_prediction' if expected_same else 'diagnostic_contrastive_prediction']
        checks['no_parse_fallback']=not(r['extraction_fallback'] or r['answer_parse_fallback'])
        if not all(checks.values()):flags.append('implementation_consistency_issue')
        if r['diagnostic_camera_prediction']!=r['diagnostic_contrastive_prediction']:flags.append('other_branch_wrong_not_evidence_of_gating_error')
        if not all(x['prediction']==x['answer'] for x in groups[r['base_qid']]): flags.append('paired_permutation_has_wrong_answer')
        out['pcd'].append(dict(id=r['qid'],group=r['base_qid'],task=r['category'],flags=flags,checks=checks,
            ref_obj=r['ref_obj'],target_obj=r['target_obj'],branch=r['branch'],viewpoint_raw=r['viewpoint_raw']))

out['counts']={model:dict(collections.Counter(f for r in out[model] for f in r['flags'])) for model in ['apc','pcd']}
(ROOT/'correct_case_audit.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(out['counts'],indent=2))
for model in ['apc','pcd']:
 print(model,'n',len(out[model]))
 print('tasks',dict(collections.Counter(r['task'] for r in out[model])))
