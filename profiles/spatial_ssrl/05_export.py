from collections import defaultdict
rows=[json.loads(p.read_text()) for p in sorted(OUT.glob('traces/*/result.json'))]
samples=[json.loads(l) for l in (OUT/'samples.jsonl').read_text().splitlines()]
by_id={r['sample_id']:r for r in rows}
assert set(by_id)=={s['sample_id'] for s in samples},'Incomplete; resume before scoring'
def metrics(values):
    groups=defaultdict(list)
    for r in values:groups[r['circular_group']].append(r['correct'])
    return dict(rows=len(values),row_accuracy=sum(r['correct'] for r in values)/len(values),
                groups=len(groups),circular_accuracy=sum(all(g) for g in groups.values())/len(groups),
                invalid_or_error=sum(r['status']!='ok' for r in values))
result=dict(overall=metrics(rows),by_category={k:metrics([r for r in rows if r['category']==k]) for k in sorted({r['category'] for r in rows})})
if not SANITY:
    result['scope']=SCOPE
    result['by_task']={task:metrics([r for r in rows if r['paper_task']==task])
                       for task in sorted({r['paper_task'] for r in rows})}
(OUT/'summary.json').write_text(json.dumps(result,indent=2))
with (OUT/'predictions.jsonl').open('w',encoding='utf8') as f:
    for r in rows:f.write(json.dumps(r,ensure_ascii=False)+'\n')
print(json.dumps(result,indent=2),OUT)

diagnostics={
    'rows':len(rows),
    'invalid_or_error':sum(r['status']!='ok' for r in rows),
    'generation_limit_hits':sum(any(c.get('hit_token_limit',False) for c in r.get('generation',[])) for r in rows),
}
(OUT/'diagnostics.json').write_text(json.dumps(diagnostics,indent=2),encoding='utf8')
archive=OUT.parent/(OUT.name+'_trace.zip')
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as bundle:
    for path in OUT.rglob('*'):
        if path.is_file():bundle.write(path,path.relative_to(OUT))
from IPython.display import FileLink,display
display(FileLink(str(archive)))

if SANITY:
    r=rows[0]; print('Prediction:',r['prediction'],'Expected:',r['answer'],'Status:',r['status'])
    assert r['status']=='ok', 'Technical sanity failed; inspect traces/'+r['sample_id']+'/calls.jsonl and result.json'
