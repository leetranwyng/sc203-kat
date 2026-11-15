"""Build the local 7B audit from the verified Kaggle artifacts."""
import base64
import hashlib
import os
import csv
import json
import re
import shutil
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from zipfile import ZipFile, is_zipfile

HERE = Path(__file__).resolve().parent
RAW = HERE / 'raw'
TRACE_ARCHIVES = Path.home()/'.codex/visualizations/2026/11/09/01a1be47-0aa8-7e81-ab50-6197ff4f7a8a'
TASKS = {'orientation_on_the_left': 'Left/Right',
         'orientation_in_front_of': 'Visibility',
         'multi_object_facing': 'Object Facing'}
MODELS = ('APC-Vis', 'PCD', 'SymPL')


def load(name):
    path = RAW / name
    if path.suffix == '.jsonl':
        return [json.loads(line) for line in path.read_text(encoding='utf8').splitlines() if line.strip()]
    return json.loads(path.read_text(encoding='utf8'))


def by_id(records, key='sample_id'):
    index = {str(r[key]): r for r in records}
    assert len(index) == len(records), key
    return index


def pcd_options(row):
    return {letter: row[letter] for letter in 'ABCD' if row.get(letter)}


def csv_export(name, rows):
    if not rows:
        return
    with (HERE / name).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def outcome(row, prediction_key='prediction'):
    status = row.get('status', 'ok')
    if status != 'ok':
        return {'error': 'Error', 'method_failure': 'Failure', 'invalid': 'Invalid'}.get(status, 'Unavailable')
    if row.get(prediction_key) not in ('A','B','C','D'):
        return 'Invalid'
    return 'Correct' if row[prediction_key] == row['answer'] else 'Wrong'


def grouped_status(statuses):
    return next((s for s in ('Unavailable','Error','Failure','Invalid','Wrong') if s in statuses), 'Correct')


def group_rows(records, group_key):
    groups = defaultdict(list)
    for row in records:
        groups[row[group_key]].append(row)
    assert all(len(group) == 2 for group in groups.values())
    return groups


def accuracy(model, records, group_key, pred_key='prediction', pcd=False):
    result = []
    for category in ('Overall', *TASKS):
        subset = records if category == 'Overall' else [r for r in records if r['category'] == category]
        if not subset:
            continue
        counts = Counter(outcome(r, pred_key) for r in subset)
        groups = group_rows(subset, group_key)
        circular = Counter(grouped_status([outcome(r, pred_key) for r in g]) for g in groups.values())
        label = 'Front/Behind' if pcd and category == 'orientation_in_front_of' else TASKS.get(category, category)
        result.append(dict(method=model, task=label, rows=len(subset), correct=counts['Correct'],
                           wrong=counts['Wrong'], invalid=counts['Invalid'], method_failure=counts['Failure'],
                           pipeline_error=counts['Error'], row_accuracy_pct=round(100*counts['Correct']/len(subset),4),
                           groups=len(groups), circular_correct=circular['Correct'],
                           circular_accuracy_pct=round(100*circular['Correct']/len(groups),4)))
    return result


def pcd_label(row):
    chosen = row['branch']
    alt = row['diagnostic_camera_prediction'] if chosen == 'pcd' else row['diagnostic_contrastive_prediction']
    if outcome(row) == 'Correct':
        return 'Final answer correct'
    if alt == row['answer']:
        return f'Branch routing | {chosen} wrong, alternative correct'
    if alt in pcd_options(row):
        return 'Both PCD branches wrong | stage unresolved'
    return 'Other branch unavailable'


def normalize_sympl_object(text):
    name = re.sub(r"""['’"“”]""", '', text.casefold())
    name = ' '.join(name.split()).rstrip('.')
    return {'megazine': 'magazine'}.get(name, name)


def reparse_sympl(row):
    """Recover a unique named answer from an invalid Object Facing output."""
    if row['status'] != 'invalid' or row['paper_task'] != 'facing':
        return row
    name = normalize_sympl_object(row.get('raw_output') or '')
    matches = [
        letter for letter, choice in row['choices'].items()
        if name and normalize_sympl_object(choice) == name
    ]
    if len(matches) != 1:
        return row
    letter = matches[0]
    return {**row, 'prediction': letter, 'status': 'ok', 'correct': letter == row['answer'],
            'original_prediction': row.get('prediction'), 'original_status': row['status'],
            'reparse_rule': 'Unique normalized Object Facing option'}


def sympl_label(row):
    if row.get('reparse_rule'):
        return 'Object Facing | answer recovered by normalization'
    if outcome(row) == 'Correct':
        return 'Final answer correct'
    raw = (row.get('raw_output') or '').lower()
    if 'reference viewer not in the list' in raw:
        return 'Reference viewer missing from object list'
    if 'category detection fail' in raw:
        return 'Reasoning category parser'
    if row['status'] == 'method_failure':
        return 'Method stopped | stage unresolved'
    if row['status'] == 'invalid':
        if row['paper_task'] == 'left_right':
            return 'Left/Right | object name instead of direction'
        return 'Final answer parser'
    if row['status'] == 'error':
        return 'Pipeline exception'
    return 'Wrong final answer | stage unresolved'


def apc_label(row):
    if row.get('error_type') == 'NameError' and "name 'record' is not defined" in row.get('message',''):
        assert 'orientation_to_direction' in row['traceback']
        return 'Orient Anything | trace instrumentation NameError'
    return row.get('failed_stage') or 'Unknown stage'


def check_match(sample, row, pcd=False):
    if pcd:
        assert (sample['question_id'], sample['source_question'], sample['source_choices'],
                sample['answer'], Path(sample['image_path']).stem) == (
                row['qid'], row['question'], pcd_options(row), row['answer'], row['image_sha256'])
    else:
        for field in ('question_id','question','choices','answer','circular_group','category'):
            assert sample[field] == row[field], (sample['sample_id'],field)
        assert Path(sample['image_path']).stem == Path(row['image_path']).stem


def patterns(cases, get_states):
    output = []
    for names in (tuple(MODELS), *combinations(MODELS, 2), ('PCD','PCD baseline'), ('SymPL','APC baseline')):
        for task in ('Left/Right','Visibility','Object Facing','Front/Behind'):
            cohort = [r for r in cases if all(r['tasks'].get(name)==task for name in names)]
            if not cohort:
                continue
            for level in ('Row','CircularEval'):
                entries = cohort if level == 'Row' else group_rows(cohort,'group').values()
                counts = Counter()
                for value in entries:
                    rows = [value] if level == 'Row' else value
                    states = tuple(grouped_status([get_states(r)[m] for r in rows]) for m in names)
                    counts[states] += 1
                for states, count in sorted(counts.items()):
                    output.append(dict(models=' + '.join(names), task=task, unit=level,
                                       pattern=' / '.join(states), count=count, denominator=sum(counts.values())))
    return output


def copy_member(archive, member, destination):
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix+'.part')
    with archive.open(member) as read, partial.open('wb') as write:
        shutil.copyfileobj(read, write)
    partial.replace(destination)


def extract_media(cases):
    image_names = {r['image'] for r in cases}
    pcd_path = RAW/'pcd_full_trace.zip'
    if pcd_path.is_file() and is_zipfile(pcd_path):
        with ZipFile(pcd_path) as bundle:
            for name in bundle.namelist():
                if name.startswith('images/') and Path(name).stem in image_names:
                    copy_member(bundle,name,HERE/'media'/f'{Path(name).stem}.jpg')
    paired = HERE.parent/'replay_results'/'paired_gallery'/'images'
    for name in image_names:
        source = paired/f'{name}.jpg'
        target = HERE/'media'/source.name
        if source.is_file() and not target.exists():
            shutil.copyfile(source,target)
    sympl_path = RAW/'sympl_full_trace.zip'
    if not sympl_path.is_file():
        sympl_path = Path(os.environ.get('SC203_SYMPL_TRACE', TRACE_ARCHIVES/'sympl_full_trace.zip'))
    if sympl_path.is_file() and is_zipfile(sympl_path):
        with ZipFile(sympl_path) as bundle:
            files = set(bundle.namelist())
            files_by_case = defaultdict(list)
            for member in files:
                if member.startswith('traces/'):
                    files_by_case[member.split('/')[1]].append(member)
            for case in cases:
                row = case['sympl']
                if not row:
                    continue
                artifact = row.get('original_image')
                if artifact in files:
                    copy_member(bundle,artifact,HERE/'media'/f"{case['image']}.jpg")
                calls = f"traces/{case['id']}/calls.jsonl"
                if calls in files:
                    target = HERE/'sympl_trace'/f"{case['id']}.jsonl"
                    copy_member(bundle,calls,target)
                    case['symplTrace'] = target.relative_to(HERE).as_posix()
                    case['symplTools'] = []
                    images = {}
                    for line in bundle.read(calls).splitlines():
                        call = json.loads(line)
                        tool = call.get('component', 'Trace')
                        phase = call.get('phase')
                        if phase in ('end','error','detector_candidates') and tool != 'process_messages':
                            item = {k:call[k] for k in ('component','phase','seconds','result','error','boxes','scores','phrases') if k in call}
                            result = item.get('result')
                            if result is not None and len(json.dumps(result,ensure_ascii=False)) > 5500:
                                item['result'] = 'Large result. See the original JSONL trace.'
                            case['symplTools'].append(item)
                        if phase == 'end' and tool in ('symbolic_layout_prompting','do_abstraction','do_projection_2d','do_bipartition','do_localization'):
                            for asset in re.findall(r'artifacts/[a-f0-9]+\.png',line.decode()):
                                if asset in files:
                                    images.setdefault(asset,tool)
                    for member in files_by_case.get(case['id'],()):
                        if member.lower().endswith('.png') and ('layout' in member.lower() or 'projection' in member.lower()):
                            images.setdefault(member,Path(member).stem)
                    case['symplImages'] = [{'asset':asset,'stage':stage} for asset,stage in list(images.items())[:8]]
                    for item in case['symplImages']:
                        copy_member(bundle,item['asset'],HERE/item['asset'])
    apc_path = Path(os.environ.get('SC203_APC_TRACE', str(
        TRACE_ARCHIVES/'apc_qwen7b_full_trace.zip')))
    if apc_path.is_file() and is_zipfile(apc_path):
        with ZipFile(apc_path) as bundle:
            files_by_case = defaultdict(list)
            for member in bundle.namelist():
                match = re.search(r'(?:^|/)trace/apc_vis/(\d+)/(.+)$',member)
                if match:
                    files_by_case[match[1]].append((member,match[2]))
            for case in cases:
                case['apcVision'] = {}
                case['apcImages'] = []
                for member, name in files_by_case[case['id']]:
                    if name.startswith('orientation_angles_') and name.endswith('.json'):
                        case['apcVision'][name] = json.loads(bundle.read(member))
                    elif name == 'detector.jsonl':
                        case['apcVision']['detector'] = [json.loads(line) for line in bundle.read(member).splitlines()]
                    elif name.endswith('.png') and (name.startswith('orientation_input_') or
                            (name.startswith('object_') and name.endswith('_mask.png'))):
                        path = f"apc_trace/{case['id']}/{name}"
                        copy_member(bundle,member,HERE/path)
                        case['apcImages'].append({'asset':path,'stage':name})
                    elif name.startswith('detection_candidates_') and name.endswith('.png'):
                        import io
                        from PIL import Image
                        safe = re.sub(r'[<>:"/\\|?*]', '_', Path(name).stem)
                        path = f"apc_trace/{case['id']}/{safe}.jpg"
                        target = HERE/path
                        if not target.exists():
                            target.parent.mkdir(parents=True,exist_ok=True)
                            with Image.open(io.BytesIO(bundle.read(member))) as im:
                                im.thumbnail((720,480))
                                im.convert('RGB').save(target,quality=84)
                        case['apcImages'].append({'asset':path,'stage':Path(name).stem})
    # Direct file downloads remain usable even when a full ZIP is still transferring.
    for case in cases:
        trace = HERE/'sympl_trace'/f"{case['id']}.jsonl"
        if trace.is_file() and not case.get('symplTrace'):
            case['symplTrace'] = trace.relative_to(HERE).as_posix()
            case['symplTools'] = []
            images = {}
            for line in trace.read_text(encoding='utf8').splitlines():
                entry = json.loads(line)
                stage = entry.get('component','Trace')
                if entry.get('phase') in ('end','error','detector_candidates') and stage != 'process_messages':
                    item = {k:entry[k] for k in ('component','phase','seconds','result','error','boxes','scores','phrases') if k in entry}
                    if len(json.dumps(item,ensure_ascii=False)) < 10000:
                        case['symplTools'].append(item)
                for asset in re.findall(r'artifacts/[a-f0-9]+\.png',line):
                    if (HERE/asset).exists():
                        images.setdefault(asset,stage)
            case['symplImages'] = [{'asset':k,'stage':v} for k,v in list(images.items())[:8]]
        folder = HERE/'apc_trace'/case['id']
        if folder.is_dir() and not case.get('apcVision'):
            vision = {}
            for file in folder.glob('*.json'):
                vision[file.name] = json.loads(file.read_text(encoding='utf8'))
            detector = folder/'detector.jsonl'
            if detector.is_file():
                vision['detector'] = [json.loads(l) for l in detector.read_text(encoding='utf8').splitlines()]
            case['apcVision'] = vision
            case['apcImages'] = [{'asset':file.relative_to(HERE).as_posix(),'stage':file.stem}
                                 for file in sorted(folder.glob('*.png'))]
    # Reuse the pinned public benchmark cache when an output ZIP omits the input.
    missing = {h for h in image_names if not (HERE/'media'/f'{h}.jpg').exists()}
    tsv = Path(os.environ.get('SC203_BENCHMARK_TSV', str(Path.home()/'AppData/Local/Temp/sc203_compare_images.tsv')))
    if missing and tsv.is_file():
        csv.field_size_limit(2**31-1)
        with tsv.open(encoding='utf-8-sig', newline='') as stream:
            for row in csv.DictReader(stream, delimiter='\t'):
                value = row['image'].strip()
                if value.isdigit():
                    continue
                blob = base64.b64decode(value.split(',')[-1])
                digest = hashlib.sha256(blob).hexdigest()
                if digest in missing:
                    (HERE/'media'/f'{digest}.jpg').write_bytes(blob)
                    missing.remove(digest)
                    if not missing:
                        break
    from PIL import Image
    dimensions = {}
    for case in cases:
        image = case['image']
        if image not in dimensions:
            path = HERE/'media'/f'{image}.jpg'
            if path.is_file():
                with Image.open(path) as photo:
                    dimensions[image] = list(photo.size)
            else:
                dimensions[image] = None
        case['imageSize'] = dimensions[image]
        case['hasImage'] = bool(case['imageSize'])


def refine_sympl_stage(case):
    tools = {x['component']:x.get('result') for x in case.get('symplTools', []) if x['phase']=='end'}
    if case['labels']['SymPL'] == 'Method stopped | stage unresolved' and tools.get('symbolic_layout_prompting') == 'fail':
        case['labels']['SymPL'] = 'Symbolic layout prompting returned fail'
    if case['tasks']['SymPL'] != 'Left/Right' or case['states']['SymPL'] != 'Wrong':
        return
    abstraction, layout = tools.get('do_abstraction'),tools.get('do_bipartition')
    if not isinstance(abstraction,list) or len(abstraction)<6 or not isinstance(layout,list) or len(layout)!=3:
        return
    names, (partitions, x, _) = abstraction[5], layout
    target = re.search(r'facing where it is facing, is the (.+?) on the left or right',case['question'],re.I)
    if not target or len(names)!=1 or names[0].strip().casefold()!=target[1].strip().casefold():
        return
    if len(x)!=1 or len(partitions)!=1 or partitions[0].get('type')!='rectangle':
        return
    boundary = partitions[0]['coords'][1][0]
    if abs(x[0]-boundary)<1:
        return
    side = 'left' if x[0] < boundary else 'right'
    gold = case['choices'][case['answer']].strip().lower().split()[-1]
    if gold not in ('left','right'):
        return
    case['symplLayout'] = {'target':names[0], 'x':round(x[0],3), 'boundary':boundary, 'side':side, 'gold':gold}
    case['labels']['SymPL'] = ('Symbolic layout already disagrees with gold' if side!=gold
                               else 'Final answer disagrees with correct symbolic layout')


def main():
    manifest = load('apc_manifest.json')
    pcd_config = load('pcd_config.json')
    sympl_manifest = load('sympl_manifest.json')
    checkpoint = manifest['config']['model_revision']
    assert manifest['config']['model_id'] == 'Qwen/Qwen2.5-VL-7B-Instruct'
    assert checkpoint == pcd_config['model_revision'] == sympl_manifest['identity']['model_sha']
    assert manifest['data_sha'] == pcd_config['dataset_revision'] == sympl_manifest['identity']['dataset_sha']

    samples = by_id(load('apc_samples.jsonl'))
    apc = by_id(load('apc_apc_vis_predictions.jsonl'))
    direct = by_id(load('apc_direct_predictions.jsonl'))
    pcd = by_id(load('pcd_predictions_gpu0.jsonl'),'row_id')
    sympl_original = by_id(load('sympl_predictions.jsonl'))
    sympl = {sid: reparse_sympl(row) for sid, row in sympl_original.items()}
    assert len(samples)==len(apc)==len(direct)==len(sympl)==2078 and len(pcd)==1386
    assert all(r['fingerprint']==manifest['fingerprint'] for r in (*apc.values(),*direct.values()))
    assert all(r['fingerprint']==sympl_manifest['fingerprint'] for r in sympl.values())
    cases = []
    for sid, sample in samples.items():
        a, d, s, p = apc[sid],direct[sid],sympl[sid],pcd.get(sid)
        for row in (a,d,s):
            check_match(sample,row)
        if p:
            check_match(sample,p,True)
        task = TASKS[sample['category']]
        tasks = {'APC-Vis':task,'SymPL':task,'APC baseline':task}
        states = {'APC-Vis':outcome(a),'SymPL':outcome(s),'APC baseline':outcome(d)}
        labels = {'APC-Vis':apc_label(a),'SymPL':sympl_label(s)}
        if p:
            tasks['PCD'] = tasks['PCD baseline'] = 'Front/Behind' if task=='Visibility' else task
            states.update(PCD=outcome(p))
            states['PCD baseline']=outcome(p,'baseline_prediction')
            labels['PCD']=pcd_label(p)
        cases.append(dict(id=sid,qid=sample['question_id'],group=sample['circular_group'],question=sample['question'],
                          choices=sample['choices'],answer=sample['answer'],image=Path(sample['image_path']).stem,
                          tasks=tasks,states=states,labels=labels,
                          apc={k:a.get(k) for k in ('message','failed_stage','vlm_calls','traceback')},
                          direct=d['prediction'],pcd=p,sympl={k:s.get(k) for k in ('raw_output','prediction','status',
                              'error','traceback','generation','original_image','reparse_rule',
                              'original_prediction','original_status')}))
    assert sum(r['states']['PCD']=='Correct' for r in cases if r['pcd']) == 1012
    assert sum(r['states']['APC-Vis']=='Error' for r in cases) == 2078
    for row in pcd.values():
        assert row['hit'] == (outcome(row) == 'Correct')
        assert row['baseline_hit'] == (outcome(row,'baseline_prediction') == 'Correct')
        assert row['prediction'] == row[
            'diagnostic_contrastive_prediction' if row['branch']=='pcd' else 'diagnostic_camera_prediction']
    stats = []
    for name,rows,key,pred,protocol in [
        ('APC baseline',direct,'circular_group','prediction',False),
        ('APC-Vis',apc,'circular_group','prediction',False),
        ('PCD baseline',pcd,'base_qid','baseline_prediction',True),
        ('PCD',pcd,'base_qid','prediction',True),
        ('SymPL',sympl,'circular_group','prediction',False),
    ]:
        stats.extend(accuracy(name,list(rows.values()),key,pred,protocol))
    coverage = patterns(cases,lambda r:r['states'])
    extract_media(cases)
    for case in cases:
        refine_sympl_stage(case)
    labels = Counter((model,row['tasks'][model],label) for row in cases for model,label in row['labels'].items())
    breakdown = [dict(method=m,task=t,stage=stage,count=count) for (m,t,stage),count in sorted(labels.items())]
    csv_export('sympl_reparse.csv', [
        dict(id=sid, task=TASKS[before['category']], raw_output=before['raw_output'],
             choices=json.dumps(before['choices'], ensure_ascii=False),
             original_status=before['status'], original_prediction=before['prediction'],
             rescored_status=after['status'], rescored_prediction=after['prediction'],
             gold=before['answer'], correct=outcome(after)=='Correct',
             rule=after.get('reparse_rule', 'Object name does not specify a side'
                            if before['paper_task']=='left_right' else 'No unique normalized choice'))
        for sid, before in sympl_original.items() if before['status'] == 'invalid'
        for after in [sympl[sid]]
    ])
    csv_export('accuracy.csv',stats)
    csv_export('overlap.csv',coverage)
    csv_export('stages.csv',breakdown)
    csv_export('joined_rows.csv',[dict(id=r['id'],qid=r['qid'],group=r['group'],task=r['tasks']['APC-Vis'],
               apc=r['states']['APC-Vis'],sympl=r['states']['SymPL'],pcd=r['states'].get('PCD','Unavailable'),
               pcd_task=r['tasks'].get('PCD',''),pcd_stage=r['labels'].get('PCD',''),
               sympl_stage=r['labels']['SymPL'],apc_stage=r['labels']['APC-Vis']) for r in cases])
    data = dict(accuracy=stats, overlap=coverage, stages=breakdown, cases=cases, checkpoint=checkpoint,
                dataset=manifest['data_sha'])
    (HERE/'data.js').write_text('window.RUN = '+json.dumps(data,ensure_ascii=False).replace('<','\\u003c')+';\n',
                              encoding='utf8')
    print('Joined:',len(cases),'rows;',sum(r['hasImage'] for r in cases),'with images.')
    print('Tool traces:',sum(bool(r.get('apcVision')) for r in cases),'APC,',
          sum(bool(r.get('symplTrace')) for r in cases),'SymPL; row trace prompts available for all.')
    for r in stats:
        if r['task']=='Overall':
            print(r)


if __name__ == '__main__':
    main()
