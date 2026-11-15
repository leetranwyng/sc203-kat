"""Recompute scores from immutable predictions; compare only identical inputs."""
import collections, hashlib, itertools, json, pathlib, re, sys, zipfile
ROOT=pathlib.Path(__file__).resolve().parent
OLD=ROOT.parent/'results_qwen7b'
sys.path.insert(0,str(OLD))
from build_analysis import reparse_sympl, pcd_label, sympl_label
MODELS=['APC-Vis','PCD','SymPL','VST','Spatial-SSRL']
TASKS=['left_right','front_behind','visibility','facing']
def read(p):
 p=pathlib.Path(p)
 return [json.loads(s) for s in p.read_text(encoding='utf8').splitlines() if s.strip()] if p.suffix=='.jsonl' else json.loads(p.read_text(encoding='utf8'))
def save(name,data):
 (ROOT/name).write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf8')
def pct(a,b):return round(100*a/b,4) if b else None
def detail_status(r):
 if r is None:return 'Missing'
 if r.get('status','ok')!='ok':return {'error':'Error','method_failure':'Failure','invalid':'Invalid'}.get(r['status'],r['status'])
 if r.get('prediction') not in r['choices']:return 'Invalid'
 return 'Correct' if r['prediction']==r['answer'] else 'Wrong'
def status(r):
 state=detail_status(r)
 return state if state in ['Correct','Missing'] else 'Wrong'
expected={};records={};canonical={};checks=collections.Counter()
def key(r):return (r.get('task',r.get('paper_task')),str(r['sample_id']))
def identity(r):
 h=r.get('image_sha256') or pathlib.PurePosixPath(r['image_path']).stem
 return (r['question_id'],r['question'],r['choices'],r['answer'],r['circular_group'],h)
def add(model,samples,preds):
 expected.setdefault(model,{})
 records.setdefault(model,{})
 for s in samples:
  k=key(s)
  assert k not in expected[model],(model,k,'duplicate sample')
  if k in canonical:assert identity(s)==identity(canonical[k]),(model,k,'input mismatch')
  else:canonical[k]=s
  expected[model][k]=s
 for r in preds:
  k=key(r);assert k in expected[model] and k not in records[model],(model,k)
  assert identity(r)==identity(expected[model][k]),(model,k,'prediction input mismatch')
  if 'correct' in r:assert bool(r['correct'])==(status(r)=='Correct'),(model,k,'saved correctness mismatch')
  records[model][k]=r;checks['verified_predictions']+=1
samples=read(ROOT/'raw_final/apc/results/samples.jsonl')+read(ROOT/'raw/apc2/extracted/samples.jsonl')
rich={str(r['sample_id']):r for path in [ROOT/'raw_final/apc/results/run/predictions_apc_vis.jsonl',ROOT/'raw/apc2/run/predictions_apc_vis.jsonl'] for r in read(path)}
preds=read(ROOT/'raw_final/apc/benchmark_results/predictions.jsonl')
preds=[dict(r,**{k:v for k,v in rich.get(str(r['sample_id']),{}).items() if k not in r}) for r in preds]
add('APC-Vis',samples,preds)
pcd=read(OLD/'raw/pcd_predictions.json')
pcd=[dict(r,sample_id=r['row_id'],question_id=r['qid'],circular_group=r['base_qid'],choices={k:r[k] for k in 'ABCD' if r.get(k)},paper_task='left_right' if r['category']=='orientation_on_the_left' else 'front_behind') for r in pcd]
add('PCD',pcd,pcd)
sympl=[reparse_sympl(r) for r in read(OLD/'raw/sympl_predictions.jsonl')]
add('SymPL',read(OLD/'raw/sympl_samples.jsonl'),sympl)
add('VST',read(ROOT/'raw_final/vst/checkpoint/samples.jsonl'),read(ROOT/'raw_final/vst/checkpoint/predictions.jsonl'))
add('Spatial-SSRL',read(ROOT/'raw/ssrl/samples.jsonl'),read(ROOT/'raw/ssrl/predictions.jsonl'))

def groups(keys):
 out=collections.defaultdict(list)
 for k in keys:out[(k[0],canonical[k]['circular_group'])].append(k)
 return out
def stats(model,task,keys):
 rr=records[model];gg=groups(keys);assert all(len(v)==2 for v in gg.values())
 count=collections.Counter(status(rr.get(k)) for k in keys)
 done=[ks for ks in gg.values() if all(k in rr for k in ks)]
 good=sum(all(status(rr.get(k))=='Correct' for k in ks) for ks in done)
 valid=[ks for ks in done if all(detail_status(rr[k]) in ['Correct','Wrong'] for k in ks)]
 inconsistent=sum(len({rr[k]['choices'][rr[k]['prediction']] for k in ks})!=1 for ks in valid)
 sec=[rr[k]['seconds'] for k in keys if k in rr and isinstance(rr[k].get('seconds'),(float,int))]
 attempted=len(keys)-count['Missing']
 return dict(model=model,task=task,expected=len(keys),attempted=attempted,correct=count['Correct'],wrong=count['Wrong'],invalid=count['Invalid'],error=count['Error'],method_failure=count['Failure'],missing=count['Missing'],coverage_pct=pct(attempted,len(keys)),accuracy_attempted_pct=pct(count['Correct'],attempted),accuracy_expected_pct=pct(count['Correct'],len(keys)),expected_groups=len(gg),completed_groups=len(done),correct_groups=good,circular_completed_pct=pct(good,len(done)),circular_expected_pct=pct(good,len(gg)),valid_groups=len(valid),option_inconsistent_groups=inconsistent,mean_seconds=round(sum(sec)/len(sec),3) if sec else None)
accuracy=[]
for m in MODELS:
 for task in TASKS:
  keys=[k for k in expected[m] if k[0]==task]
  if keys:accuracy.append(stats(m,task,keys))
 accuracy.append(stats(m,'overall_own_scope',list(expected[m])))
save('accuracy.json',accuracy)

pairs=[];patterns=[];common_scores=[];case_lists=[]
for task in TASKS:
 available=[m for m in MODELS if any(k[0]==task for k in expected[m])]
 common=set.intersection(*[{k for k in records[m] if k[0]==task} for m in available])
 complete_keys=[k for ks in groups(common).values() if len(ks)==2 for k in ks]
 for m in available:
  if complete_keys:common_scores.append(stats(m,task,complete_keys))
 c=collections.Counter(tuple(status(records[m][k]) for m in available) for k in common)
 for st,n in sorted(c.items()):patterns.append(dict(task=task,models=available,states=list(st),rows=n))
 for k in sorted(common):
  case_lists.append(dict(task=task,id=k[1],states={m:status(records[m][k]) for m in available},all_correct=all(status(records[m][k])=='Correct' for m in available),all_wrong=all(status(records[m][k])=='Wrong' for m in available)))
 for a,b in itertools.combinations(available,2):
  shared={k for k in records[a] if k[0]==task}&records[b].keys()
  for unit in ['rows','circular_groups']:
   counts=collections.Counter()
   if unit=='rows':
    states=[(status(records[a][k]),status(records[b][k])) for k in shared]
   else:
    states=[]
    for ks in groups(shared).values():
     if len(ks)!=2:continue
     def gs(m):
      ss=[status(records[m][k]) for k in ks]
      return 'Correct' if all(s=='Correct' for s in ss) else 'Wrong' if all(s in ['Correct','Wrong'] for s in ss) else 'Technical'
     states.append((gs(a),gs(b)))
   for sa,sb in states:
    counts['both_correct' if sa==sb=='Correct' else 'both_wrong' if sa==sb=='Wrong' else 'a_only_correct' if sa=='Correct' and sb=='Wrong' else 'b_only_correct' if sb=='Correct' and sa=='Wrong' else 'technical']+=1
   pairs.append(dict(task=task,a=a,b=b,unit=unit,n=len(states),**{x:counts[x] for x in ['both_correct','a_only_correct','b_only_correct','both_wrong','technical']}))
save('pairwise.json',pairs);save('patterns.json',patterns);save('common_scores.json',common_scores);save('case_lists.json',case_lists)

trace_ids=set();trace_files={}
replay=read(ROOT/'raw_final/apc/replay_comparison.json')
replay_cases={str(r['sample_id']):r for r in replay['cases']}
for folder in (ROOT/'raw_final/apc/replay/run/trace/apc_vis').iterdir():
 if folder.is_dir():
  trace_ids.add(folder.name)
  trace_files[folder.name]=[p.relative_to(ROOT).as_posix() for p in sorted(folder.rglob('*')) if p.is_file()]
diagnostics={
 'apc_trace_samples':len(trace_ids),
 'replay_summary':replay['by_task'],
 'replay_prediction_changes':sum(r['prediction_changed'] for r in replay['cases']),
 'apc_trace_ids':sorted(trace_ids),
 'apc_errors':[{k:r.get(k) for k in ['sample_id','status','message','failed_stage','raw_output']} for r in records['APC-Vis'].values() if detail_status(r) in ['Error','Invalid']],
 'apc_viewer_strategies':dict(collections.Counter(r.get('reference_viewer_strategy','unavailable') for r in records['APC-Vis'].values())),
 'pcd_failure_labels':dict(collections.Counter(pcd_label(r) for r in pcd)),
 'sympl_failure_labels':dict(collections.Counter(sympl_label(r) for r in sympl)),
 'sympl_reparsed_rows':sum(bool(r.get('reparse_rule')) for r in sympl),
 'vst_capped':sum(bool(r.get('capped')) for r in records['VST'].values()),
 'ssrl_capped':sum(any(g.get('hit_token_limit') for g in r.get('generation',[])) for r in records['Spatial-SSRL'].values()),
 'input_checks':dict(checks),
}
save('diagnostics.json',diagnostics)
cases=[]
for k,s in sorted(canonical.items()):
 h=identity(s)[-1];mdata={}
 for m in MODELS:
  if k not in expected[m]:continue
  r=records[m].get(k)
  mdata[m]={'state':status(r),'original_state':detail_status(r)}
  if r:
   # Preserve full original records in separate JSON files; load on demand in the viewer.
   name=f'records/{m}/{k[0]}_{k[1]}.json';p=ROOT/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding='utf8')
   mdata[m].update(prediction=r.get('prediction'),raw=r.get('raw_output',r.get('camera_raw')),seconds=r.get('seconds'),record=name)
   if m=='PCD':mdata[m].update(label=pcd_label(r),branch=r['branch'],camera=r['diagnostic_camera_prediction'],contrastive=r['diagnostic_contrastive_prediction'])
   if m=='SymPL':mdata[m]['label']=sympl_label(r)
   if m=='APC-Vis':mdata[m].update(label=r.get('message') or 'Root cause not established',trace=trace_files.get(k[1],[]),replay=replay_cases.get(k[1]))
 cases.append(dict(id=k[1],task=k[0],question=s['question'],choices=s['choices'],answer=s['answer'],group=s['circular_group'],image=f'../results_qwen7b/media/{h}.jpg',models=mdata))
save('cases.json',cases)
payload=dict(accuracy=accuracy,pairs=pairs,patterns=patterns,common_scores=common_scores,diagnostics=diagnostics,cases=cases)
(ROOT/'data.js').write_text('window.AUDIT='+json.dumps(payload,ensure_ascii=False).replace('<','\\u003c')+';',encoding='utf8')
save('missing_rows.json',[dict(model=m,task=k[0],sample_id=k[1],question_id=s['question_id'],circular_group=s['circular_group']) for m,ss in expected.items() for k,s in ss.items() if k not in records[m]])
for r in accuracy:print(r['model'],r['task'],f"{r['correct']}/{r['attempted']}/{r['expected']}",r['accuracy_attempted_pct'],r['circular_completed_pct'])
print('COMMON',[(r['model'],r['task'],r['attempted'],r['accuracy_attempted_pct']) for r in common_scores])
print('DIAGNOSTICS',diagnostics)
