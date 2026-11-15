"""Score both paired modes with a full denominator; never drop failed APC samples."""
import collections,json,statistics,sys
from pathlib import Path
from apc_common import read_jsonl
out=Path(sys.argv[1]);m=json.loads((out/'manifest.json').read_text());samples=[json.loads(x) for x in (out/'samples.jsonl').read_text().splitlines()]
truth={r['sample_id']:r for r in samples};groups=collections.defaultdict(list)
for r in samples:groups[r['circular_group']].append(r)
metrics={}
for mode in m['config']['modes']:
 records=read_jsonl(out/'full'/f'predictions_{mode}.jsonl')
 assert set(records).issubset(truth)
 for r in records.values():assert r['fingerprint']==m['fingerprint']
 def hit(sid):
  p=records.get(sid,{});return p.get('status')=='ok' and p.get('prediction')==truth[sid]['answer']
 merged=[{**t,**records.get(sid,dict(status='missing',prediction=None)),'correct':bool(hit(sid))} for sid,t in truth.items()]
 grouped=[dict(circular_group=k,category=v[0]['category'],paper_task=v[0].get('paper_task',v[0]['category']),correct=all(hit(r['sample_id']) for r in v)) for k,v in groups.items()]
 successful=[r for r in records.values() if r['status']=='ok'];invalid=[r for r in successful if r.get('prediction') is None]
 mean=lambda xs:statistics.mean(xs) if xs else None
 metric=dict(expected_rows=len(samples),attempted_rows=len(records),successful_rows=len(successful),missing_rows=len(samples)-len(records),
  pipeline_errors=sum(r['status']!='ok' for r in records.values()),invalid_outputs=len(invalid),
  invalid_by_reason=dict(collections.Counter(r.get('output_issue','unparsed_answer') for r in invalid)),
  row_accuracy=mean([r['correct'] for r in merged]),circular_groups=len(grouped),circular_accuracy=mean([r['correct'] for r in grouped]),
  circular_by_subtype={k:mean([r['correct'] for r in grouped if r['category']==k]) for k in sorted({r['category'] for r in grouped})},
  circular_by_paper_task={k:mean([r['correct'] for r in grouped if r['paper_task']==k]) for k in sorted({r['paper_task'] for r in grouped})},
  selected_subtasks=sorted({r['category'] for r in samples}),scope=m['config'].get('scope'),
  coverage=len(successful)/len(samples),all_rows_attempted=len(records)==len(samples),complete=len(successful)==len(samples),
  mean_seconds=mean([r['seconds'] for r in successful]),parse_policy=m['config']['parse_policy'],
  protocol='Paper task families with visibility recast from front/behind; Qwen2.5-VL-3B direct/APC-Vis, no LLM judge; not exact reproduction of paper Table 1')
 for name,values in [('predictions',merged),('circular_groups',grouped),('invalid_outputs',[r for r in merged if r['status']=='ok' and r.get('prediction') is None]),('pipeline_errors',[r for r in merged if r['status']!='ok'])]:
  (out/(mode+'_'+name+'.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in values),encoding='utf-8')
 metrics[mode]=metric
(out/'summary.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8');print(json.dumps(metrics,indent=2))

if set(('direct','apc_vis')).issubset(metrics):
    a=read_jsonl(out/'full/predictions_direct.jsonl');b=read_jsonl(out/'full/predictions_apc_vis.jsonl')
    paired=collections.Counter()
    for sid,t in truth.items():
        ha=a.get(sid,{}).get('status')=='ok' and a.get(sid,{}).get('prediction')==t['answer']
        hb=b.get(sid,{}).get('status')=='ok' and b.get(sid,{}).get('prediction')==t['answer']
        paired['both_correct' if ha and hb else 'direct_only' if ha else 'apc_only' if hb else 'both_wrong']+=1
    comparison=dict(paired_rows=dict(paired),row_accuracy_delta_pp=100*(metrics['apc_vis']['row_accuracy']-metrics['direct']['row_accuracy']),
        circular_accuracy_delta_pp=100*(metrics['apc_vis']['circular_accuracy']-metrics['direct']['circular_accuracy']),
        circular_delta_by_paper_task_pp={k:100*(metrics['apc_vis']['circular_by_paper_task'][k]-metrics['direct']['circular_by_paper_task'][k]) for k in metrics['direct']['circular_by_paper_task']})
    (out/'paired_comparison.json').write_text(json.dumps(comparison,indent=2),encoding='utf-8')
