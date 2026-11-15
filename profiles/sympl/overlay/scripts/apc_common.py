"""Shared dataset/parser/state helpers; no model imports needed for offline validation."""
import hashlib,json,os,re,time,unicodedata
from pathlib import Path
POLICY='strict_letter_or_unique_exact_choice_v1'
def parse_option(text,choices):
    def norm(x):return re.sub(r'\s+',' ',unicodedata.normalize('NFKC',x).strip()).casefold()
    match=re.fullmatch(r'\s*(?:(?:the\s+)?answer\s*(?:is|:)\s*)?[\(\[]?([A-D])[\)\]]?[.!]?\s*',text,re.I)
    strict=match.group(1).upper() if match and match.group(1).upper() in choices else None
    exact=[k for k,v in choices.items() if norm(text) in (norm(v),norm(f'{k}. {v}'),norm(f'{k}) {v}'))]
    return strict if strict else exact[0] if len(exact)==1 else None

def atomic_json(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(path)

def read_jsonl(path,repair=False):
    path=Path(path);rows={}
    if not path.exists():return rows
    with path.open('rb+' if repair else 'rb') as f:
        while True:
            start=f.tell();line=f.readline()
            if not line:break
            try:r=json.loads(line)
            except (ValueError,UnicodeDecodeError):
                if repair and not line.endswith(b'\n') and not f.read():f.truncate(start);break
                raise RuntimeError('Corrupt JSONL '+str(path)+' byte '+str(start)) from None
            rows[r['sample_id']]=r
            if repair and not line.endswith(b'\n'):f.seek(0,2);f.write(b'\n')
    return rows

PAPER_TASKS={'orientation_on_the_left':'left_right','orientation_in_front_of':'visibility','multi_object_facing':'facing'}

def paper_sample(row):
    row=dict(row)
    row['paper_task']=PAPER_TASKS[row['category']]
    row['source_question']=row['question'];row['source_choices']=dict(row['choices'])
    row['question_transform']='unchanged'
    if row['category']=='orientation_in_front_of':
        match=re.fullmatch(r'(.*?facing where it is facing, )is (.+?) in front of me or behind me\?',row['question'],re.I)
        if not match:raise ValueError('Unrecognized front/behind template for '+row['sample_id'])
        mapping={'in front of':'yes','behind':'no'}
        if set(row['choices'].values())!=set(mapping):raise ValueError('Unexpected front/behind options')
        row['question']=match.group(1)+'is '+match.group(2)+' visible or not?'
        row['choices']={key:mapping[value] for key,value in row['choices'].items()}
        row['question_transform']='front_behind_to_visibility_yes_no_v1'
        # Option positions and correct option letter are preserved, including each permutation.
    return row

def choose_samples(rows,cfg):
    scope=cfg.get('scope')
    if scope in ('paper_tasks_original','paper_tasks_visibility'):
        rows=[r for r in rows if r['category'] in PAPER_TASKS]
        if scope=='paper_tasks_visibility':rows=[paper_sample(r) for r in rows]
    elif scope!='all_original':raise ValueError('Unsupported scope')
    if cfg.get('limit') is not None:
        groups=set(r['circular_group'] for r in rows[:cfg['limit']]);rows=[r for r in rows if r['circular_group'] in groups]
    return rows
