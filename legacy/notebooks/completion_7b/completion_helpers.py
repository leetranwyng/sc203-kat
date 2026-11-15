"""Bookkeeping for a frozen benchmark continuation and independent diagnostic replay."""
import collections, hashlib, json, pathlib, shutil, subprocess, zipfile
from apc_kaggle_guardrails import atomic_json, protocol_fingerprint, read_records

def jsonl(path):
    return [json.loads(x) for x in pathlib.Path(path).read_text(encoding='utf8').splitlines() if x.strip()]

def load_reference(path):
    with zipfile.ZipFile(path) as z:
        return {name:json.loads(z.read(name)) if name.endswith('.json') else
                [json.loads(x) for x in z.read(name).decode('utf8').splitlines() if x.strip()]
                for name in z.namelist()}

def outcome(row):
    if row.get('status')!='ok':return row.get('status','missing')
    return 'correct' if row.get('prediction')==row['answer'] else 'wrong'

def archive_files(folder,target,names):
    folder,target=pathlib.Path(folder),pathlib.Path(target)
    names=[pathlib.Path(n) for n in names if (folder/n).is_file()]
    needed=sum((folder/n).stat().st_size for n in names)+256*2**20
    if shutil.disk_usage(folder).free < needed+int(1.5*2**30):
        print('ZIP deferred: keep raw Output files; insufficient archive reserve:',target)
        return False
    temp=target.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED) as z:
        for n in names:z.write(folder/n,n.as_posix())
    temp.replace(target)
    return True

def merge_benchmark(part1,reference,out):
    part1,out=pathlib.Path(part1),pathlib.Path(out);out.mkdir(parents=True,exist_ok=True)
    ref=load_reference(reference)
    rows=jsonl(part1/'apc_vis_predictions.jsonl')+ref['part2_predictions.jsonl']
    assert len(rows)==2078 and len({r['sample_id'] for r in rows})==2078
    with (out/'predictions.jsonl').open('w',encoding='utf8') as f:
        for r in rows:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    report={}
    for task in ['left_right','visibility','facing']:
        rr=[r for r in rows if r['paper_task']==task];groups=collections.defaultdict(list)
        for r in rr:groups[r['circular_group']].append(r)
        assert all(len(g)==2 for g in groups.values())
        count=collections.Counter(outcome(r) for r in rr)
        report[task]=dict(expected=len(rr),attempted=len(rr)-count['missing'],missing=count['missing'],
            statuses=dict(count),row_accuracy=sum(outcome(r)=='correct' for r in rr)/len(rr),
            circular_accuracy=sum(all(outcome(r)=='correct' for r in g) for g in groups.values())/len(groups))
    report['complete']=all(r.get('status')!='missing' for r in rows)
    report['note']='Frozen part 2 + resumed part 1; replay is excluded. Errors remain in denominators.'
    atomic_json(out/'summary.json',report)
    archive_files(out,out/'benchmark_results.zip',['predictions.jsonl','summary.json'])
    print('Combined benchmark: complete=',report['complete'],flush=True)
    return report

def prepare_replay(config,part1,reference,work,scripts,python,env,resume_input=None):
    work,part1=pathlib.Path(work),pathlib.Path(part1)
    ref=load_reference(reference);selection=ref['trace_selection.json']
    assert len(selection)==120 and len({r['sample_id'] for r in selection})==120
    prep=work/'trace_prepare';prep.mkdir(exist_ok=True)
    trace_out=part1.parent/'trace_replay_120';trace_out.mkdir(exist_ok=True)
    cfg=dict(config,experiment='apc_trace_replay_120_v2',output=str(prep),resume_input=None,
        paper_tasks=['left_right','visibility','facing'],expected_rows=2078,expected_circular_groups=1039,
        expected_task_counts=dict(left_right=698,visibility=688,facing=692),
        trace_keep_per_task=40,trace_max_total_mib=2048,trace_max_sample_mib=128,
        run_timeout_seconds=10800,retry_failed=False)
    path=work/'trace_prepare_config.json';path.write_text(json.dumps(cfg),encoding='utf8')
    subprocess.run([str(python),str(pathlib.Path(scripts)/'apc_prepare_data.py'),str(path)],env=env,cwd=config['repo'],check=True)
    manifest=json.loads((prep/'manifest.json').read_text())
    available={r['sample_id']:r for r in jsonl(prep/'samples.jsonl')}
    rows=[available[s['sample_id']] for s in selection]
    for r,s in zip(rows,selection):assert r['image_sha256']==s['image_sha256'] and r['paper_task']==s['paper_task']
    cfg.update(output=str(trace_out),expected_rows=120,expected_circular_groups=120,
               expected_task_counts=dict(left_right=40,visibility=40,facing=40))
    manifest.update(config=cfg,selected_rows=120,circular_groups=120,tasks=cfg['expected_task_counts'],
        protocol='Diagnostic replay selected from previous answers: 28 wrong + 12 correct per task; NOT benchmark accuracy')
    manifest['fingerprint']=protocol_fingerprint(manifest,rows)
    if resume_input and not (trace_out/'run/predictions_apc_vis.jsonl').exists():
        with zipfile.ZipFile(resume_input) as z:
            previous=json.loads(z.read('manifest.json'))
            if previous['fingerprint']!=manifest['fingerprint']:raise ValueError('Trace resume fingerprint mismatch')
            infos=z.infolist()
            if sum(i.file_size for i in infos)>3*2**30:raise ValueError('Unexpectedly large trace archive')
            if shutil.disk_usage(trace_out).free < sum(i.file_size for i in infos)+int(1.5*2**30):raise OSError('Not enough disk for trace resume')
            for i in infos:
                pp=pathlib.PurePosixPath(i.filename)
                if pp.is_absolute() or '..' in pp.parts or '\\' in i.filename:raise ValueError('Unsafe ZIP member')
                if i.is_dir():continue
                target=trace_out.joinpath(*pp.parts)
                target.resolve().relative_to(trace_out.resolve())
                target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(z.read(i))
    previous=trace_out/'manifest.json'
    if previous.exists() and json.loads(previous.read_text())['fingerprint']!=manifest['fingerprint']:
        raise ValueError('Existing replay output has another fingerprint')
    atomic_json(previous,manifest)
    atomic_json(trace_out/'config.json',cfg)
    atomic_json(trace_out/'selection.json',selection)
    (trace_out/'samples.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf8')
    # A portable trace resume must contain the trace, not only the answer.
    journal=trace_out/'run/predictions_apc_vis.jsonl'
    saved=read_records(journal,repair=True)
    valid=[];dropped=[]
    for r in saved.values():
        path=r.get('trace_path')
        target=(trace_out/'run'/path) if path else None
        if target is not None:
            target.resolve().relative_to((trace_out/'run/trace').resolve())
        if target is not None and target.is_dir() and any(target.iterdir()):valid.append(r)
        else:dropped.append(r)
    if dropped:
        with (trace_out/'previous_missing_trace_attempts.jsonl').open('a',encoding='utf8') as f:
            for r in dropped:f.write(json.dumps(r)+'\n')
        tmp=journal.with_suffix('.tmp');tmp.write_text(''.join(json.dumps(r)+'\n' for r in valid),encoding='utf8');tmp.replace(journal)
    print('Replay: selected 120 distinct task/image cases, retained prior trace records:',len(valid),flush=True)
    return cfg

def finish_replay(out,reference):
    out=pathlib.Path(out);ref=load_reference(reference)
    selection=ref['trace_selection.json'];saved=read_records(out/'run/predictions_apc_vis.jsonl',repair=True)
    sample_index={r['sample_id']:r for r in jsonl(out/'samples.jsonl')}
    result=[]
    for s in selection:
        r=saved.get(s['sample_id'],{});row=sample_index[s['sample_id']]
        path=out/'run'/r['trace_path'] if r.get('trace_path') else None
        exists=path is not None and path.is_dir() and any(path.iterdir())
        result.append(dict(s,replay_status=r.get('status','missing'),replay_prediction=r.get('prediction'),
            replay_correct=r.get('status')=='ok' and r.get('prediction')==row['answer'],
            prediction_changed=bool(r) and r.get('prediction')!=s['prior_prediction'],
            trace_saved=exists,trace_path=r.get('trace_path'),
            failed_stage=r.get('failed_stage'),message=r.get('message')))
    by_task={}
    for task in ['left_right','visibility','facing']:
        rr=[r for r in result if r['paper_task']==task]
        by_task[task]=dict(selected=len(rr),trace_saved=sum(r['trace_saved'] for r in rr),
            replay_correct=sum(r['replay_correct'] for r in rr),
            replay_wrong=sum(r['replay_status']=='ok' and not r['replay_correct'] for r in rr),
            invalid_or_error=sum(r['replay_status'] in ['invalid','error'] for r in rr),
            missing=sum(r['replay_status']=='missing' for r in rr))
    report=dict(complete=all(r['trace_saved'] for r in result),by_task=by_task,cases=result,
        note='28 wrong + 12 correct selected using PRIOR output per task. Replayed labels may change. No benchmark predictions overwritten.')
    atomic_json(out/'replay_comparison.json',report)
    # One selected row per image is not a full circular pair. Do not present
    # the generic worker's single-row group score as benchmark CircularEval.
    for filename in ['full_report.json','summary.json']:
        path=out/filename
        if path.is_file():
            value=json.loads(path.read_text())
            value['protocol']='Diagnostic replay; purposefully selected previous failures and controls'
            value['accuracy_note']='Selected sample only, excluded from the 2078-row benchmark. CircularEval not applicable.'
            value['circular_accuracy']=None
            for metric in value.get('by_task',{}).values():metric['circular_accuracy']=None
            atomic_json(path,value)
    result_names=[p.relative_to(out) for p in out.rglob('*') if p.is_file() and
        'trace' not in p.relative_to(out).parts and p.suffix in ['.json','.jsonl','.txt','.log']]
    archive_files(out,out/'results.zip',result_names)
    names=[p.relative_to(out) for p in out.rglob('*') if p.is_file() and
        (p.suffix in ['.json','.jsonl'] or 'trace' in p.relative_to(out).parts)]
    archive_files(out,out/'trace_resume.zip',names)
    print(json.dumps(dict(complete=report['complete'],by_task=by_task),indent=2),flush=True)
    return report
