"""Verify paper-task filtering/rewrite against the real 3DSRBench metadata."""
import collections,copy,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from apc_common import choose_samples
source=Path(r'E:\SC203\SenseNova-SI\results\sensenovasi3b_failedrun\samples.txt')
original=[json.loads(x) for x in source.read_text(encoding='utf-8-sig').splitlines()]
before=copy.deepcopy(original)
selected=choose_samples(original,dict(scope='paper_tasks_visibility',limit=None))
assert original==before,'Source was mutated'
counts=collections.Counter(r['paper_task'] for r in selected)
groups=collections.defaultdict(list)
for r in selected:groups[r['circular_group']].append(r)
assert len(selected)==2078 and len(groups)==1039
assert counts=={'visibility':688,'left_right':698,'facing':692}
for r in selected:
 o=next(x for x in original if x['sample_id']==r['sample_id'])
 assert r['answer']==o['answer'] and r['image_path']==o['image_path'] and r['circular_group']==o['circular_group']
 assert r['source_question']==o['question'] and r['source_choices']==o['choices']
 if r['paper_task']=='visibility':
  assert r['question'].endswith('visible or not?')
  assert r['choices'][r['answer']]==('yes' if o['choices'][o['answer']]=='in front of' else 'no')
 else:assert r['question']==o['question'] and r['choices']==o['choices']
for v in groups.values():
 assert len(v)==2 and len({r['question'] for r in v})==1 and sorted(v[0]['choices'].values())==sorted(v[1]['choices'].values())
small=choose_samples(original,dict(scope='paper_tasks_visibility',limit=1));assert len(small)==2
example=next(r for r in selected if r['paper_task']=='visibility')
report=dict(total_rows=len(selected),total_groups=len(groups),counts=dict(counts),example=example,
 note='Verified deterministic recast from released original TSV metadata. This is not a claim of verbatim equivalence to an unreleased author evaluator.')
Path('notebooks/paper_task_selection_verified.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print('PASS real data: 2078 rows / 1039 groups; all 688 visibility rows rewritten; option order, ground-truth letter, image and group preserved; source immutable; complete-group limits.')
print(json.dumps(example,ensure_ascii=False,indent=2))
