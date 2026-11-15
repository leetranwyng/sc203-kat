import base64,csv,hashlib,io,json,os,re,sys,shutil
from pathlib import Path
from PIL import Image
from huggingface_hub import HfApi,hf_hub_download
from apc_common import atomic_json,choose_samples
cfg=json.loads(Path(sys.argv[1]).read_text());work=Path(cfg['work']);out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=True)
assets=json.loads((work/'assets_ready.json').read_text())
if cfg['sanity']:
    source=Path(cfg['sanity_image']) if cfg['sanity_image'] else Path(cfg['repo'])/'demo/sample_image_man.jpg'
    with Image.open(source) as im:im.verify()
    items=[dict(sample_id='sanity',question_id='repo_demo',question=cfg['sanity_question'],choices=cfg['sanity_choices'],answer=None,category='orientation_on_the_left',paper_task='left_right',circular_group='repo_demo',image_path=str(source))]
    data_sha=None;all_count=1;columns=[]
else:
    old=json.loads((out/'manifest.json').read_text()) if (out/'manifest.json').exists() else None
    revision=old['data_sha'] if old else cfg['data_revision']
    if cfg.get('dataset_input'):
        tsv=Path(cfg['dataset_input']);data_sha='local:'+hashlib.sha256(tsv.read_bytes()).hexdigest()
    else:
        api=HfApi();info=api.dataset_info(cfg['dataset_id'],revision=revision,files_metadata=True);data_sha=info.sha
        entry=next(f for f in info.siblings if f.rfilename==cfg['data_file'])
        if shutil.disk_usage(work).free<(entry.size or 0)+1.5*2**30:raise RuntimeError('Need TSV size + 1.5 GiB for images; attach dataset_input or free workspace.')
        tsv=hf_hub_download(cfg['dataset_id'],cfg['data_file'],repo_type='dataset',revision=data_sha)
    images_dir = work / 'data/images'
    images_dir.mkdir(parents=True, exist_ok=True)
    csv.field_size_limit(2**31 - 1)
    items, refs, unresolved = [], {}, []

    def missing(x):
        return x is None or str(x).strip().lower() in ('', 'nan', 'none', 'null')

    with open(tsv, encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f, delimiter='\t')
        columns = reader.fieldnames
        required = {'index', 'qid', 'question', 'image', 'answer', 'category'}
        if not required.issubset(columns or []):
            raise ValueError(f'TSV schema changed: {columns}')
        print('TSV columns:', columns, flush=True)
        for rownum, row in enumerate(reader):
            sid = str(row['index'])
            qid = str(row['qid'])
            value = str(row['image']).strip()
            choices = {k: row[k] for k in 'ABCD' if not missing(row.get(k))}
            answer = row['answer'].strip().upper()
            if len(choices) < 2 or answer not in choices:
                raise ValueError(f'Invalid choices/answer: {sid}')
            item = dict(sample_id=sid, question_id=qid, question=row['question'],
                        choices=choices, answer=answer, category=row['category'],
                        circular_group=re.sub(r'-[0-3]$', '', qid), image_path=None)
            # VLMEvalKit may store an index reference instead of repeating base64.
            if value in refs:
                item['image_path'] = refs[value]
            elif re.fullmatch(r'\d+', value or ''):
                unresolved.append((item, value))
            else:
                payload = value.split(',', 1)[1] if value.startswith('data:image/') else value
                raw = base64.b64decode(payload, validate=True)
                target = images_dir / (hashlib.sha256(raw).hexdigest() + '.img')
                if not target.is_file() or target.stat().st_size != len(raw):
                    with Image.open(io.BytesIO(raw)) as image:
                        image.verify()
                    tmp = target.with_suffix('.tmp')
                    tmp.write_bytes(raw)
                    tmp.replace(target)
                item['image_path'] = str(target)
            if item['image_path']:
                refs[sid] = item['image_path']
            items.append(item)
            if (rownum + 1) % 1000 == 0:
                print('Prepared rows:', rownum + 1, flush=True)

    while unresolved:
        pending = []
        for item, key in unresolved:
            if key in refs:
                item['image_path'] = refs[key]
                refs[item['sample_id']] = refs[key]
            else:
                pending.append((item, key))
        if len(pending) == len(unresolved):
            raise ValueError(f'Unresolved image references: {[k for _, k in pending[:5]]}')
        unresolved = pending
    assert len({r['sample_id'] for r in items}) == len(items), 'Duplicate sample IDs'
    assert all(Path(r['image_path']).is_file() for r in items)

    all_count=len(items);items=choose_samples(items,cfg)
    if not items:raise ValueError('No selected rows')
manifest=dict(config=cfg,model_sha=assets['snapshots']['qwen_model']['sha'],data_sha=data_sha,
    selected_rows=len(items),total_dataset_rows=all_count,circular_groups=len({r['circular_group'] for r in items}),
    source_lock=json.loads((work/'source_lock.json').read_text()),patches=json.loads((work/'patches.json').read_text()),assets=assets,
    protocol='Paper task families: left/right, visibility recast from front/behind, facing; 3B direct vs APC-Vis; exact-choice without LLM judge; not exact reproduction of 7B Table 1')
text=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in items)
manifest['fingerprint']=hashlib.sha256((json.dumps(manifest,sort_keys=True)+text).encode()).hexdigest()
old_path=out/'manifest.json'
if old_path.exists() and json.loads(old_path.read_text())['fingerprint']!=manifest['fingerprint']:
    raise RuntimeError('Existing experiment has another fingerprint; choose a new experiment name, never rewrite old predictions.')
(out/'samples.jsonl').write_text(text,encoding='utf-8');atomic_json(old_path,manifest)
print('Prepared',len(items),'rows /',manifest['circular_groups'],'groups; model SHA',manifest['model_sha'],flush=True)
