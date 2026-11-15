import ast, base64, collections, importlib.util, json, pathlib, sys, tempfile, zipfile
from unittest.mock import patch
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[1];sys.path.insert(0,str(HERE))
from completion_helpers import prepare_replay,finish_replay,merge_benchmark
from apc_kaggle_guardrails import pending_rows,protocol_fingerprint,read_records
def rows(p):return [json.loads(l) for l in pathlib.Path(p).read_text(encoding='utf8').splitlines() if l.strip()]
def nb(name):return json.loads((HERE/name).read_text(encoding='utf8'))
def literal_assignment(source,name):
 return next(ast.literal_eval(n.value) for n in ast.parse(source).body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id==name)
a=nb('APC_Vis_7B_complete_and_120_traces_Kaggle.ipynb')
v=nb('VST_7B_RL_complete_missing_Kaggle.ipynb')
embedded=0
for notebook in [a,v]:
 for c in notebook['cells']:
  if c['cell_type']!='code':continue
  source=''.join(c['source']);source=source.split('\n',1)[1] if source.startswith('%%writefile') else source
  tree=ast.parse(source)
  for n in ast.walk(tree):
   if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='write_text' and n.args:
    try:text=ast.literal_eval(n.args[0])
    except (ValueError,TypeError):continue
    if isinstance(text,str) and ('def ' in text or 'import ' in text):compile(text,'embedded','exec');embedded+=1
config=literal_assignment(''.join(a['cells'][3]['source']),'CFG')
old=json.loads((ROOT/'results_five7b/raw/apc1/manifest.json').read_text())
samples=rows(ROOT/'results_five7b/raw/apc1/extracted/samples.jsonl')
assert protocol_fingerprint(dict(old,config=config),samples)==old['fingerprint']
saved=read_records(ROOT/'results_five7b/raw/apc1/run/predictions_apc_vis.jsonl')
pending=pending_rows(samples,saved,old['fingerprint'],False)
assert collections.Counter(r['paper_task'] for r in pending)==dict(left_right=192,visibility=183)
selection=json.loads((HERE/'apc_trace_selection.json').read_text())
assert {r['sample_id'] for r in selection}>={'5700','5778','5832'}
for task in ['left_right','visibility','facing']:
 ss=[r for r in selection if r['paper_task']==task]
 assert len(ss)==40 and len({r['image_sha256'] for r in ss})==40
 assert sum(r['prior_correct'] for r in ss)==12
v_saved=rows(ROOT/'results_five7b/raw/vst/predictions.jsonl')
v_samples=rows(ROOT/'results_five7b/raw/vst/extracted/samples.jsonl')
keys={r['sample_key'] for r in v_saved}
assert collections.Counter(r['task'] for r in v_samples if r['sample_key'] not in keys)==dict(left_right=177,facing=690)
spec=importlib.util.spec_from_file_location('vst_runner',ROOT/'notebooks/vst7b/runner.py');vm=importlib.util.module_from_spec(spec);spec.loader.exec_module(vm)
with tempfile.TemporaryDirectory(prefix='completion_test_') as d:
 root=pathlib.Path(d)
 manifest=json.loads((ROOT/'results_five7b/raw/vst/manifest.json').read_text())
 restored=vm.restore(root/'vst',None,manifest) if False else None
 (root/'vst').mkdir()
 restored=vm.restore(root/'vst',ROOT/'results_five7b/raw/vst/checkpoint.zip',manifest)
 assert len(restored)==1899
 wrong=dict(manifest,fingerprint='changed')
 try:vm.restore(root/'vst',None,wrong)
 except ValueError:pass
 else:raise AssertionError('Fingerprint change must fail')
 ref=root/'reference.zip'
 with zipfile.ZipFile(ref,'w') as z:
  z.writestr('trace_selection.json',json.dumps(selection))
  for name in ['samples','predictions']:
   path=ROOT/f'results_five7b/raw/apc2/extracted/{"apc_vis_predictions" if name=="predictions" else "samples"}.jsonl'
   z.writestr('part2_'+name+'.jsonl',path.read_bytes())
 part1=root/'run1/part1';part1.mkdir(parents=True)
 (part1/'apc_vis_predictions.jsonl').write_bytes((ROOT/'results_five7b/raw/apc1/extracted/apc_vis_predictions.jsonl').read_bytes())
 merged=merge_benchmark(part1,ref,root/'combined');assert not merged['complete']
 assert merged['left_right']['missing']==192 and merged['visibility']['missing']==183
 work=root/'work';work.mkdir()
 all_samples=samples+rows(ROOT/'results_five7b/raw/apc2/extracted/samples.jsonl')
 def fake_prepare(command,**kwargs):
  cfg=json.loads(pathlib.Path(command[-1]).read_text());prep=pathlib.Path(cfg['output'])
  (prep/'samples.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in all_samples))
  (prep/'manifest.json').write_text(json.dumps(dict(old,config=cfg)))
 with patch('completion_helpers.subprocess.run',side_effect=fake_prepare):
  cfg=prepare_replay(config,part1,ref,work,HERE,sys.executable,{},None)
  out=pathlib.Path(cfg['output']);ms=json.loads((out/'manifest.json').read_text())
  selected=rows(out/'samples.jsonl');assert len(selected)==120
  run=out/'run';run.mkdir();journal=run/'predictions_apc_vis.jsonl'
  with journal.open('w') as f:
   for r in selected[:30]:
    trace=run/'trace/apc_vis'/r['sample_id'];trace.mkdir(parents=True);(trace/'input.png').write_bytes(b'placeholder')
    f.write(json.dumps(dict(sample_id=r['sample_id'],paper_task=r['paper_task'],mode='apc_vis',
      fingerprint=ms['fingerprint'],status='ok',prediction=r['answer'],trace_path=str(trace.relative_to(run)).replace('\\','/')))+'\n')
  report=finish_replay(out,ref);assert not report['complete'] and sum(x['trace_saved'] for x in report['cases'])==30
  resume=out/'trace_resume.zip';assert resume.is_file()
  part_b=root/'run2/part1';part_b.mkdir(parents=True)
  cfg2=prepare_replay(config,part_b,ref,work,HERE,sys.executable,{},resume)
  out2=pathlib.Path(cfg2['output']);records=read_records(out2/'run/predictions_apc_vis.jsonl')
  assert len(records)==30
  rest=pending_rows(rows(out2/'samples.jsonl'),records,json.loads((out2/'manifest.json').read_text())['fingerprint'],False)
  assert len(rest)==90
print('PASS: notebook syntax,',embedded,'embedded modules, exact resume fingerprints, VST 867/APC 375 pending, 120 stratified traces, partial trace archive/resume, benchmark merge.')
