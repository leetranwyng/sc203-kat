"""Explicit health gate; benchmark accuracy is not a condition for PASS."""
import hashlib,json
from pathlib import Path
from trace_runtime import atomic_json
from analyze120 import read_events,refs

def check(root,worker_exit=0,expected_count=6):
    root=Path(root)
    def read(name,default):
        p=root/name
        return json.loads(p.read_text(encoding='utf8')) if p.exists() else default
    samples=[json.loads(x) for x in (root/'samples.jsonl').read_text(encoding='utf8').splitlines()] if (root/'samples.jsonl').exists() else []
    expected_ids={r['sample_id'] for r in samples}
    results=[json.loads(p.read_text(encoding='utf8')) for p in sorted((root/'traces').glob('*/result.json'))]
    actual_ids={r['sample_id'] for r in results}
    validation=read('trace_validation.json',{})
    failures=read('failure_cases.json',[])
    checks=dict(worker_exit_zero=worker_exit==0,selected_six=len(samples)==expected_count,
        exactly_expected_results=len(results)==expected_count and actual_ids==expected_ids,
        two_per_task=all(sum(r['paper_task']==task for r in samples)==2 for task in ('left_right','visibility','facing')),
        input_validation_pass=read('input_validation.json',{}).get('status')=='PASS',
        every_status_ok=bool(results) and all(r['status']=='ok' for r in results),
        every_trace_closed=bool(results) and all(r.get('trace_closed') for r in results),
        diagnostics_complete=len(failures)==expected_count and all(r.get('root_cause')!='analysis_error' for r in failures),
        all_required_stages=len(validation)==expected_count and all(not v.get('missing_stages',['absent']) for v in validation.values()),
        all_artifacts_present=len(validation)==expected_count and all(not v.get('missing_artifacts',['absent']) for v in validation.values()),
        calls_terminated=len(validation)==expected_count and all(not v.get('unterminated_calls',['absent']) for v in validation.values()),
        renderer_replays_exactly=len(validation)==expected_count and all(v.get('renderer_replay_exact') is True for v in validation.values()),
        gallery_created=(root/'gallery.html').is_file())
    paths=set()
    for row in results:
        for event in read_events(root/row['attempt_path']):paths.update(refs(event))
    corrupt=[]
    for rel in sorted(paths):
        p=root/rel
        if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest()!=p.stem:corrupt.append(rel)
    checks['artifact_content_hashes_match']=bool(paths) and not corrupt
    report=dict(status='PASS' if all(checks.values()) else 'FAIL',checks=checks,
        failed_checks=[k for k,v in checks.items() if not v],corrupt_artifacts=corrupt,
        completed=len(results),correct=sum(bool(r.get('correct')) for r in results),
        note='PASS validates the inference/trace pipeline, not benchmark accuracy or upstream ground truth.',
        warnings={r['sample_id']:r.get('flags',[]) for r in failures if r.get('flags')})
    atomic_json(root/'sanity_report.json',report)
    return report

if __name__=='__main__':
    import sys
    result=check(sys.argv[1],worker_exit=int(sys.argv[2]))
    print(json.dumps(result,indent=2))
    raise SystemExit(0 if result['status']=='PASS' else 1)
