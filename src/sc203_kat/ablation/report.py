"""Portable HTML and paired tables; labels describe evidence, not assumed causes."""
import csv
import html
import itertools
import json
from pathlib import Path
from .common import read,write,rows,VARIANTS

def summarize(items, records, variants):
    summaries=[];pairs=[]
    for task in dict.fromkeys(r['paper_task'] for r in items):
        cohort=[r for r in items if r['paper_task']==task]
        for variant in variants:
            rr=[records.get((variant,str(r['sample_id']))) for r in cohort]
            attempted=sum(r is not None for r in rr)
            correct=sum(bool(r and r.get('correct')) for r in rr)
            summaries.append(dict(task=task,variant=variant,n=len(cohort),attempted=attempted,
                correct=correct,wrong=attempted-correct,unattempted=len(cohort)-attempted,
                accuracy=correct/len(cohort) if attempted==len(cohort) else None,
                attempted_accuracy=correct/attempted if attempted else None))
        for a,b in itertools.combinations(variants,2):
            pair=dict(task=task,a=a,b=b,both_correct=0,a_only=0,b_only=0,both_wrong=0,unpaired=0)
            for row in cohort:
                x,y=records.get((a,str(row['sample_id']))),records.get((b,str(row['sample_id'])))
                if x is None or y is None:pair['unpaired']+=1;continue
                goodx,goody=bool(x.get('correct')),bool(y.get('correct'))
                pair['both_correct' if goodx and goody else 'a_only' if goodx else 'b_only' if goody else 'both_wrong']+=1
            pairs.append(pair)
    return summaries,pairs

def build(out):
    out=Path(out);items=rows(out/'samples.jsonl');active=read(out/'identity.json')['variants']
    variants=['baseline']+active;records={};predictions=[]
    for row in items:
        # Import historical predictions only: never launch a baseline worker.
        records[('baseline',str(row['sample_id']))]=dict(status=row.get('historical_status','ok'),
            prediction=row.get('historical_prediction'),seconds=0,source='historical_APC_trace',
            correct=row.get('historical_status')=='ok' and row.get('historical_prediction')==row['answer'])
    for variant in active:
        for row in items:
            p=out/'answers'/variant/str(row['sample_id'])/'result.json'
            if p.exists():
                value=read(p);value['correct']=value.get('status')=='ok' and value.get('prediction')==row['answer']
                records[(variant,str(row['sample_id']))]=value
                predictions.append(dict(value,answer=row['answer'],question=row['question'],choices=row['choices']))
    (out/'predictions.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in predictions),encoding='utf8')
    table,pairs=summarize(items,records,variants)
    fresh=[r for (variant,sid),r in records.items() if variant!='baseline']
    technical_pass=len(fresh)==len(items)*len(active) and all(r['status']=='ok' for r in fresh)
    technical_pass=technical_pass and all((out/r['attempt']/'apc_visual_prompt.png').is_file() for r in fresh)
    write(out/'summary.json',dict(by_task=table,pairs=pairs,technical_pass=technical_pass,
        baseline_source='historical saved result; never rerun',
        note='Diagnostic enriched trace cohort. Wrong includes invalid/error. No full-dataset accuracy claim.',
        circular_eval='Not reported: selected traces do not guarantee complete option permutations.'))
    for name,data in [('accuracy.csv',table),('paired_outcomes.csv',pairs)]:
        if data:
            with (out/name).open('w',newline='',encoding='utf8') as f:
                w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    esc=lambda v:html.escape(str(v),quote=True)
    def link(p,label=None):
        p=Path(p)
        return f'<a href="{esc(p.relative_to(out).as_posix())}">{esc(label or p.name)}</a>'
    def img(p,label):
        p=Path(p)
        return f'<figure><img loading="lazy" src="{esc(p.relative_to(out).as_posix())}"><figcaption>{esc(label)}</figcaption></figure>' if p.exists() else ''
    def details(p,title):
        return f'<details><summary>{esc(title)}</summary><pre>{esc(p.read_text(encoding="utf8"))}</pre></details>'
    parts=['<!doctype html><meta charset="utf-8"><title>APC perception ablations</title>',
        '<style>body{font:16px system-ui;margin:24px;background:#f5f7fa;color:#162538}table{border-collapse:collapse;background:white}td,th{padding:9px;border:1px solid #ccd4df} .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:16px}.panel,article{background:white;border:1px solid #d5dce5;border-radius:10px;padding:14px;margin:12px 0}figure{margin:10px 0}img{max-width:100%;max-height:480px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:13px monospace}.correct{color:#087c48}.wrong{color:#bd2b33}select,input{padding:8px;margin:8px}summary{cursor:pointer;padding:8px}nav{position:sticky;top:0;background:#e9f0f8;padding:8px;z-index:1}</style>',
        '<h1>APC-Vis: controlled perception ablations</h1><p>Same saved objects, viewer and depth. Qwen 7B FP16 and renderer unchanged. '
        'Baseline is imported from existing results and consumes no inference. '
        'This is the selected diagnostic cohort, enriched for errors. It is not a full 3DSRBench evaluation. '
        'Invalid and error count as wrong. Arrows show the 2D projection of an estimated 3D direction, not orientation ground truth.</p>',
        '<nav>Task <select id="task"><option value="">All</option>'+''.join(f'<option>{esc(t)}</option>' for t in sorted({r['paper_task'] for r in items}))+
        '</select> Outcome <select id="outcome"><option value="">All</option><option value="improved">Any correction</option><option value="regressed">Any regression</option><option value="disagree">Any variant disagreement</option></select> ID/question <input id="search" placeholder="Search"><span id="count"></span></nav>',
        '<h2>Accuracy</h2><table><tr><th>Variant</th><th>Task</th><th>Correct</th><th>Wrong</th><th>Accuracy</th></tr>']
    for r in table:
        acc=f'{100*r["accuracy"]:.2f}%' if r['accuracy'] is not None else f'Partial ({r["attempted"]}/{r["n"]})'
        parts.append(f'<tr><td>{r["variant"]}</td><td>{r["task"]}</td><td>{r["correct"]}</td><td>{r["wrong"]}</td><td>{acc}</td></tr>')
    parts.append('</table><p>'+link(out/'accuracy.csv')+' · '+link(out/'paired_outcomes.csv')+' · '+link(out/'predictions.jsonl')+' · '+link(out/'summary.json')+'</p>')
    for row in items:
        sid=str(row['sample_id']);base=records.get(('baseline',sid));tags=[]
        for variant in variants:
            rec=records.get((variant,sid))
            if base and rec and rec['correct']!=base['correct']:
                tags+=['disagree','improved' if rec['correct'] else 'regressed']
        parts.append(f'<article data-task="{esc(row["paper_task"])}" data-tags="{esc(" ".join(tags))}" data-search="{esc((sid+" "+row["question"]).lower())}"><h2>{esc(row["paper_task"])} · {sid}</h2><p>{esc(row["question"])}</p><p>{esc(row["choices"])} · Gold: {esc(row["answer"])} · Historical APC: {esc(row.get("historical_prediction"))}</p>')
        parts.append(img(out/'original'/sid/'input.png','Input image'))
        parts.append('<details><summary>Frozen depth preview</summary>'+img(out/'original'/sid/'depth_preview.png','Depth Pro original (relative preview scale)')+'</details>')
        parts.append('<details><summary>Original trace and frozen inputs</summary>')
        for p in sorted((out/'original'/sid).iterdir()):
            if p.is_file():parts.append(link(p)+' · ')
        parts.append('</details><div class="grid">')
        for variant in variants:
            rec=records.get((variant,sid));kind,_=VARIANTS.get(variant,('original','v1'))
            parts.append(f'<section class="panel"><h3>{esc(variant)}</h3>')
            if rec:
                parts.append(f'<p class="{"correct" if rec["correct"] else "wrong"}">{"Correct" if rec["correct"] else "Wrong"} · {esc(rec.get("prediction"))} · {esc(rec["status"])} · {rec.get("seconds",0):.1f}s</p>')
                folder=out/rec['attempt'] if variant!='baseline' else out/'original'/sid
                if variant=='baseline':parts.append('<p>Historical result · no replay</p>')
                parts.append(img(folder/'boxes_orientation.png','Selected objects and projected orientation'))
                parts.append(img(folder/'apc_visual_prompt.png','APC rendered view'))
                if variant!='baseline':parts.append(details(out/'answers'/variant/sid/'result.json','Answer, timing and errors'))
                for name in ('scene.json','intervention.json'):
                    if (folder/name).exists():parts.append(details(folder/name,name))
                for p in sorted((folder/'vlm').glob('*.json')):parts.append(details(p,'Qwen: '+p.name))
            else:parts.append('<p>Not attempted</p>')
            gp=out/'geometry'/kind/sid/'result.json'
            if gp.exists():
                g=read(gp);gf=out/g['attempt'] if g.get('attempt') else gp.parent
                parts.append(details(gp,'Geometry status and object coordinates'))
                for p in sorted(gf.glob('object_*_crop.png')):parts.append(img(p,p.name))
                for p in sorted(gf.glob('object_*_mask.png')):parts.append(img(p,p.name))
                for p in sorted(gf.glob('object_*_selection.json')):parts.append(details(p,p.name))
                for p in sorted(gf.glob('detection_candidates*.png')):parts.append(img(p,'APC selection candidates'))
                for p in sorted((gf/'vlm').glob('*.json')):parts.append(details(p,'Selection VLM: '+p.name))
            parts.append('<details><summary>All component files</summary>')
            for component in ('sam3','oneref','oa2'):
                parent=out/'perception'/component
                target=parent/kind/sid if component=='oa2' else parent/sid
                for pattern in ('*/candidates.png','*/grounding.png'):
                    for p in sorted(target.glob(pattern)):parts.append(img(p,component+' '+p.parent.name))
                for p in sorted(target.rglob('*')):
                    if p.is_file():parts.append(link(p,component+'/'+p.parent.name+'/'+p.name)+'<br>')
            parts.append('</details></section>')
        parts.append('</div></article>')
    parts.append('''<script>function filter(){let n=0;document.querySelectorAll('article').forEach(e=>{let ok=(!task.value||e.dataset.task===task.value)&&(!outcome.value||e.dataset.tags.includes(outcome.value))&&e.dataset.search.includes(search.value.toLowerCase());e.hidden=!ok;if(ok)n++;});count.textContent=n+' cases';}['task','outcome','search'].forEach(id=>document.getElementById(id).addEventListener('input',filter));filter();</script>''')
    (out/'index.html').write_text('\n'.join(parts),encoding='utf8')
    return table
