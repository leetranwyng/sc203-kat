import base64,csv,hashlib,io,json,os,re,sys,shutil
from pathlib import Path
from PIL import Image
from huggingface_hub import HfApi,hf_hub_download
from apc_common import atomic_json,choose_samples
cfg=json.loads(Path(sys.argv[1]).read_text());work=Path(cfg['work']);out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=True)
if cfg['sanity']:
    row=json.loads((work/'sample.json').read_text())
    row['image_path']=str(work/'sanity.png')
    items=choose_samples([row],cfg)
    data_sha=cfg['data_revision']
else:
    if cfg['dataset_input']:
        tsv=Path(cfg['dataset_input'])
        h=hashlib.sha256()
        with tsv.open('rb') as f:
            for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
        data_sha=h.hexdigest()
    else:
        data_sha=cfg['data_revision']
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

identity={k:cfg[k] for k in ('method','model_id','model_sha','dataset_id','scope','sanity','max_tokens','runtime_sha','format_prompt')}
identity['dataset_sha']=data_sha
if cfg['method']=='sympl':
    assets=json.loads((out/'assets_ready.json').read_text())
    identity['vision_versions']={k:assets.get(k) for k in ('snapshots','orientation_sha','files')}
    identity['native_commits']=json.loads((out/'source_lock.json').read_text())
h=hashlib.sha256(json.dumps(identity,sort_keys=True).encode())
assert items,'No samples'
for r in items:
    h.update(json.dumps({k:v for k,v in r.items() if k!='image_path'},sort_keys=True).encode())
fp=h.hexdigest()
path=out/'manifest.json'
if path.exists() and json.loads(path.read_text())['fingerprint']!=fp:raise ValueError('Incompatible resume data/config/model')
atomic_json(path,dict(identity=identity,fingerprint=fp,selected_rows=len(items),repo_commit=cfg['repo_commit']))
(out/'samples.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in items),encoding='utf8')
print('Prepared',len(items),'rows',flush=True)
