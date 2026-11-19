"""Export comparable paired results and a browser-based grounding review."""
import argparse
from collections import Counter, defaultdict
import csv
import html
import json
import os
from pathlib import Path
from urllib.parse import quote

TASKS = ('left_right', 'visibility', 'facing')
OK = {'ok', 'success', 'complete'}


def read_rows(path):
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def csv_file(path, rows, fields):
    with path.open('w', encoding='utf-8-sig', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def escaped(value):
    return html.escape(str(value))


def valid(branch):
    return branch.get('status') in OK and branch.get('prediction') is not None


def chosen(decision, branch):
    return decision.get(branch + '_box')


def changed(decision):
    base, new = chosen(decision, 'baseline'), chosen(decision, 'reclip')
    return base is not None and new is not None and not same_box(base, new)


def branch_correct(branch, answer):
    return valid(branch) and str(branch['prediction']).upper() == str(answer).upper()


def status(branch):
    return branch.get('status', 'unattempted')


def same_box(a, b):
    return a is not None and b is not None and all(abs(float(x)-float(y)) < 1e-6 for x,y in zip(a,b))


def iou(a, b):
    if a is None or b is None:
        return 0.0
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    area = max(x2-x1, 0)*max(y2-y1, 0)
    union = (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-area
    return area/union if union > 0 else 0.0


def grounding_review(out, decisions, labels_path):
    labels = {}
    if labels_path:
        labels = {
            (str(r['sample_id']), int(r['object_index'])): r['box_xyxy']
            for r in read_rows(Path(labels_path)) if r.get('confirmed') is True
        }
    reviewed = []
    disagreement = []
    for decision in decisions:
        a, b = chosen(decision,'baseline'), chosen(decision,'reclip')
        key = (str(decision['sample_id']), int(decision['object_index']))
        target = labels.get(key)
        if target is not None:
            overlaps = [iou(box,target) for box in decision.get('proposals_xyxy',[])]
            reviewed.append({'recall': bool(overlaps and max(overlaps) >= .5),
                             'baseline': iou(a,target) >= .5, 'reclip': iou(b,target) >= .5})
        if not same_box(a,b) and (a is not None or b is not None):
            disagreement.append(dict(decision, review={'confirmed': False, 'box_xyxy': None, 'notes': ''}))
    path = out/'grounding_disagreements.jsonl'
    path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in disagreement), encoding='utf-8')
    result = {'confirmed_objects':len(reviewed),'disagreements_for_review':len(disagreement),
              'iou_threshold':.5, 'recall_at_k':None, 'baseline_selection_accuracy':None,
              'reclip_selection_accuracy':None}
    if reviewed:
        result.update(recall_at_k=sum(r['recall'] for r in reviewed)/len(reviewed),
                      baseline_selection_accuracy=sum(r['baseline'] for r in reviewed)/len(reviewed),
                      reclip_selection_accuracy=sum(r['reclip'] for r in reviewed)/len(reviewed))
    return result


def resource(out, value):
    if not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = out/path
    if not path.is_file():
        return None
    return quote(Path(os.path.relpath(path,out)).as_posix(),safe='/')


def illustration(out, file, label):
    url = resource(out,file)
    if not url:
        return '<p>Image unavailable for this branch.</p>'
    return f'<figure><a href="{escaped(url)}"><img loading="lazy" src="{escaped(url)}" alt=""></a><figcaption>{escaped(label)}</figcaption></figure>'


def overlay(out, image, boxes, selected, label):
    from PIL import Image
    url = resource(out,image)
    if not url:
        return '<p>Input image unavailable.</p>'
    path = Path(image)
    if not path.is_absolute():
        path = out/path
    with Image.open(path) as im:
        width, height = im.size
    overlays = []
    for index, box in enumerate(boxes):
        if len(box) != 4:
            continue
        x1,y1,x2,y2 = map(float,box)
        if not (x2 > x1 and y2 > y1):
            continue
        css = 'left:%.5f%%;top:%.5f%%;width:%.5f%%;height:%.5f%%' % (
            100*x1/width,100*y1/height,100*(x2-x1)/width,100*(y2-y1)/height)
        marked = ' picked' if same_box(box,selected) else ''
        overlays.append(f'<span class="box{marked}" style="{css}"><b>{index}</b></span>')
    if selected is None:
        label += ' • no selection'
    return (f'<figure><div class="overlay"><img loading="lazy" src="{escaped(url)}" alt="">'
            + ''.join(overlays) + f'</div><figcaption>{escaped(label)}</figcaption></figure>')


def build_gallery(out, predictions, decisions, results, summary, method):
    css = """
body{font:15px/1.55 system-ui;background:#eef2f7;color:#172b44;margin:24px}
header,article{background:#fff;border:1px solid #d6deeb;border-radius:10px;padding:18px;margin:16px auto;max-width:1400px}
nav{position:sticky;top:0;z-index:3;background:#dde8f5;padding:10px}
select,input{padding:8px;margin:4px}table{border-collapse:collapse}td,th{padding:7px;border:1px solid #d0d9e5}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,320px),1fr));gap:14px}
figure{margin:6px 0}img{max-width:100%;max-height:460px}.overlay{position:relative;width:fit-content;line-height:0}
.overlay img{display:block;max-height:none}.box{position:absolute;box-sizing:border-box;border:2px dashed #54ddff;color:white}
.box b{background:#14283f;line-height:1.2;padding:1px 4px}.box.picked{border:3px solid #ffdb37}
pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f4f6fa;padding:10px}details{margin:10px 0}summary{cursor:pointer}
"""
    parts = ['<!doctype html><html lang="vi"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>ReCLIP paired review</title>',
             '<style>'+css+'</style><header><h1>'+escaped(method)+' · ReCLIP instance selection</h1>',
             '<p>So sánh với baseline chạy lại cùng môi trường trên tập trace chẩn đoán. Box viền vàng là lựa chọn của selector. Mỗi số là ID proposal gốc.</p>',
             '<p><a href="summary.json">Summary</a> · <a href="paired_predictions.csv">Paired predictions</a> · <a href="grounding_decisions.jsonl">Grounding decisions</a> · <a href="grounding_disagreements.jsonl">Cases to annotate</a></p>',
             '<table><tr><th>Task</th><th>Paired n</th><th>Baseline</th><th>ReCLIP</th><th>Sai → đúng</th><th>Đúng → sai</th><th>Net</th></tr>']
    for task,g in summary['by_task'].items():
        parts.append(f"<tr><td>{task}</td><td>{g['paired_n']}</td><td>{g['paired_baseline_correct']}</td><td>{g['paired_reclip_correct']}</td><td>{g['wrong_to_right']}</td><td>{g['right_to_wrong']}</td><td>{g['net_gain']}</td></tr>")
    parts.append('</table></header><nav><input id="query" placeholder="Sample ID / expression"><select id="task"><option value="">All tasks</option>'
                 + ''.join('<option>'+t+'</option>' for t in TASKS)
                 + '</select><select id="outcome"><option value="">All cases</option><option value="box">Changed box</option><option value="fixed">Sai → đúng</option><option value="hurt">Đúng → sai</option><option value="error">Error / invalid</option></select><span id="count"></span></nav>')
    for result in sorted(results,key=lambda r:(not(r['fixed'] or r['hurt']),r['task'],r['sample_id'])):
        sample_id = result['sample_id']
        row = predictions[sample_id]
        flags = ' '.join(k for k in ('box','fixed','hurt','error') if result[k])
        parts.append(f'<article data-task="{result["task"]}" data-flags="{flags}"><h2>{escaped(sample_id)} · {result["task"]}</h2><p>{escaped(row.get("question",""))}</p>')
        parts.append('<pre>'+escaped(f"Choices: {row.get('choices',{})}\nGold: {result['gold']} | Historical: {result['historical_prediction']}\nBaseline: {result['baseline_prediction']} ({result['baseline_status']}) | ReCLIP: {result['reclip_prediction']} ({result['reclip_status']})")+'</pre>')
        if result['answer_changed'] or result['error']:
            parts.append('<div class="grid">')
            for name in ('baseline','reclip'):
                branch = row.get(name) or {}
                parts.append(illustration(out,branch.get('render') or branch.get('layout'),name+' render / symbolic layout'))
            parts.append('</div>')
        if not decisions.get(sample_id):
            parts.append(illustration(out,row.get('image'),'Original input'))
        for decision in decisions.get(sample_id,[]):
            expression = decision.get('expression','')
            scoring = decision.get('scoring') or {}
            boxes = decision.get('proposals_xyxy',[])
            image = decision.get('image') or row.get('image')
            parts.append('<details'+(' open' if changed(decision) else '')+'><summary>'+escaped(f"{decision.get('role','object')} · {expression} · {scoring.get('status',decision.get('input_status','unscored'))}")+'</summary><div class="grid">')
            for branch in ('baseline','reclip'):
                parts.append(overlay(out,image,boxes,chosen(decision,branch),branch))
            parts.append('</div><p>Top-2 margin: '+escaped(scoring.get('top2_margin'))+'</p>')
            parts.append('<table><tr><th>ID</th><th>Crop</th><th>Blur</th><th>Combined</th><th>Rank</th><th>Status</th></tr>')
            for item in scoring.get('proposals',[]):
                parts.append('<tr>'+''.join('<td>'+escaped(item.get(k,''))+'</td>' for k in ('index','crop_score','blur_score','combined_score','rank','status'))+'</tr>')
            parts.append('</table></details>')
        parts.append('<details><summary>Saved predictions and traces</summary>')
        for name in ('baseline', 'reclip'):
            branch = row.get(name) or {}
            for key in ('stages', 'vlm_calls'):
                value = (branch.get('metadata') or {}).get(key)
                if isinstance(value, str) and (url := resource(out,value)):
                    parts.append(f'<a href="{escaped(url)}">{name} {key}</a> ')
        parts.append('<pre>'+escaped(json.dumps(row,ensure_ascii=False,indent=2))+'</pre></details></article>')
    parts.append("""<script>
const controls = ['query','task','outcome'].map(id=>document.getElementById(id));
function update() {
  const [query,task,outcome] = controls.map(e=>e.value.trim().toLowerCase());
  let count=0;
  for (const article of document.querySelectorAll('article')) {
    const match=(!task || article.dataset.task===task) &&
      (!outcome || article.dataset.flags.split(' ').includes(outcome)) &&
      (!query || article.textContent.toLowerCase().includes(query));
    article.hidden=!match;
    if (match) count++;
  }
  document.getElementById('count').textContent=count+' cases';
}
for (const control of controls) control.addEventListener('input',update);
update();
</script></html>""")
    (out/'gallery.html').write_text(''.join(parts),encoding='utf-8')


def export(config):
    out = Path(config['output_dir'])
    rows = read_rows(out/'replay_predictions.jsonl')
    decisions = read_rows(out/'grounding_decisions.jsonl')
    if not rows:
        raise ValueError('No paired replay results found')
    by_id = {str(r['sample_id']):r for r in rows}
    if len(by_id) != len(rows):
        raise ValueError('Duplicate sample IDs in replay predictions')
    grouped = defaultdict(list)
    seen = set()
    for d in decisions:
        sample_id = str(d['sample_id'])
        key = (sample_id,d['object_index'])
        if key in seen:
            raise ValueError(f'Duplicate grounding row {key}')
        seen.add(key)
        grouped[sample_id].append(d)

    results = []
    for sample_id,row in by_id.items():
        baseline,reclip = (row.get(name) or {} for name in ('baseline','reclip'))
        gold = row['answer']
        a,b = branch_correct(baseline,gold),branch_correct(reclip,gold)
        pair = all(status(branch) not in ('unattempted','disabled','not_run') for branch in (baseline,reclip))
        historical = row.get('historical_prediction')
        historical_correct = row.get('historical_correct')
        if historical_correct is None and historical is not None:
            historical_correct = str(historical).upper() == str(gold).upper()
        scored = [d.get('scoring') or {} for d in grouped[sample_id]]
        ran = any(s.get('selected_index') is not None for s in scored)
        ranked = any(sum(p.get('rank') is not None for p in s.get('proposals',[])) > 1 for s in scored)
        results.append(dict(sample_id=sample_id,task=row['task'],gold=gold,historical_prediction=historical,
            historical_correct=historical_correct,baseline_prediction=baseline.get('prediction'),
            baseline_status=status(baseline),reclip_prediction=reclip.get('prediction'),reclip_status=status(reclip),
            paired=pair,baseline_correct=a,reclip_correct=b,box=any(changed(d) for d in grouped[sample_id]),
            answer_changed=valid(baseline) and valid(reclip) and baseline['prediction']!=reclip['prediction'],
            fixed=pair and not a and b,hurt=pair and a and not b,
            error=any(status(x) not in OK|{'disabled','unattempted','not_run'} for x in (baseline,reclip)),
            scorer_selected=ran,scorer_ranked_multiple=ranked,
            baseline_selector_reached=bool(baseline.get('selector_reached')),
            reclip_selector_reached=bool(reclip.get('selector_reached'))))
    csv_file(out/'paired_predictions.csv',results,list(results[0]))
    summary = {'method':config['method'],'mode':config['mode'],'expected_n':len(config['sample_ids']) if config.get('sample_ids') else 120,
               'cohort_n':len(results),'by_task':{},
               'comparison_note':'Diagnostic subset. Primary comparison uses the same-environment baseline. Invalid/error count as wrong. No full-benchmark or CircularEval claim.'}
    for task in (*TASKS,'overall'):
        group = [r for r in results if r['task']==task or task=='overall']
        pair = [r for r in group if r['paired']]
        n=len(pair)
        a,b = (sum(r[branch+'_correct'] for r in pair) for branch in ('baseline','reclip'))
        historical=[r for r in group if r['historical_correct'] is not None]
        history_compare=[r for r in historical if r['baseline_status'] not in ('disabled','unattempted','not_run')]
        summary['by_task'][task] = dict(
            recorded_n=len(group),paired_n=n,paired_baseline_correct=a,paired_reclip_correct=b,
            baseline_accuracy=a/n if n else None,reclip_accuracy=b/n if n else None,
            wrong_to_right=sum(r['fixed'] for r in pair),right_to_wrong=sum(r['hurt'] for r in pair),
            net_gain=b-a,changed_box_samples=sum(r['box'] for r in group),
            changed_valid_answers=sum(r['answer_changed'] for r in group),
            baseline_selector_reached_samples=sum(r['baseline_selector_reached'] for r in group),
            reclip_selector_reached_samples=sum(r['reclip_selector_reached'] for r in group),
            scorer_selected_samples=sum(r['scorer_selected'] for r in group),
            scorer_ranked_multiple_samples=sum(r['scorer_ranked_multiple'] for r in group),
            baseline_statuses=dict(Counter(r['baseline_status'] for r in group)),
            reclip_statuses=dict(Counter(r['reclip_status'] for r in group)),
            historical_n=len(historical),historical_correct=sum(r['historical_correct'] for r in historical),
            baseline_vs_history_n=len(history_compare),
            baseline_vs_history_correctness_changed=sum(r['baseline_correct']!=r['historical_correct'] for r in history_compare),
            baseline_vs_history_prediction_changed=sum(r['baseline_prediction']!=r['historical_prediction'] for r in history_compare))
    summary['grounding'] = grounding_review(out,decisions,config.get('grounding_labels'))
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    build_gallery(out,by_id,grouped,results,summary,config['method'])
    print(json.dumps(summary['by_task']['overall'],indent=2),flush=True)
    if config['mode']=='sanity':
        if (len(results)!=summary['expected_n'] or {r['task'] for r in results}!=set(TASKS)
                or not any(r['scorer_ranked_multiple'] and r['reclip_selector_reached'] for r in results)
                or any(not r['paired'] or r['baseline_status'] not in OK or r['reclip_status'] not in OK for r in results)):
            raise RuntimeError('Sanity did not complete all three tasks with both branches and at least one multi-proposal rerank. Inspect gallery.html and runner logs.')
        print('Sanity PASS: both branches, three tasks, multi-proposal reranking',flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    args = parser.parse_args()
    export(json.loads(args.config.read_text(encoding='utf-8')))


if __name__ == '__main__':
    main()
