"""Dependency-free HTML report from normalized prediction records.

``render_report(records, output_path)`` writes one offline HTML file. Image and
record links are preserved as supplied; callers creating portable exports should
copy media and provide relative paths (see import_existing.import_directory).
"""
from __future__ import annotations

import json
from pathlib import Path

from sc203_kat.evaluation.scoring import evaluate, normalize_record, pairwise


def render_report(records, output_path):
    rows = [normalize_record(row) for row in records]
    summaries, pairs = evaluate(rows), pairwise(rows)
    payload = json.dumps(dict(rows=rows, summaries=summaries, pairs=pairs), ensure_ascii=False).replace("<", "\\u003c")
    template = r'''<!doctype html><html lang="vi"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SC203-KAT · Kết quả</title><style>
body{font:15px/1.6 system-ui;background:#f3f6fa;color:#183042;margin:0}main{max-width:1450px;margin:auto;padding:28px}table{border-collapse:collapse;width:100%;background:white;font-size:14px}th,td{padding:9px 12px;border-bottom:1px solid #dfe6ec;text-align:left}th{background:#e0eaf3}section{overflow:auto;margin:20px 0}select,input,button{font:inherit;padding:8px;border:1px solid #b6c7d5;border-radius:5px;background:white}.controls{display:flex;gap:10px;flex-wrap:wrap}.case{background:white;padding:20px;margin:20px 0;border:1px solid #dbe4ec;border-radius:10px}.photo{max-width:650px;max-height:440px;width:100%;object-fit:contain;object-position:left}.models{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:10px}.model{padding:12px;border:1px solid #dce5ed;border-radius:7px}.Correct{color:#087346}.Wrong{color:#b13127}pre{white-space:pre-wrap;max-height:420px;overflow:auto;font-size:13px}a{color:#12658e}small{color:#587184}
</style><main><h1>SC203-KAT · Kết quả</h1><p>Fail, invalid và error đều tính là Sai. Front/Behind và Visibility là hai protocol riêng.</p>
<h2>Accuracy theo task</h2><section id="scores"></section>
<h2>Đúng/sai chung và riêng</h2><div class="controls"><select id="pairtask"></select><select id="unit"><option value="rows">Lượt câu hỏi</option><option value="circular_groups">CircularEval</option></select></div><section id="pairs"></section>
<h2>Quan sát từng câu</h2><div class="controls"><select id="task"></select><select id="filter"><option value="any">Tất cả</option><option value="allwrong">Các model cùng sai</option><option value="disagree">Có đúng và có sai</option><option value="allcorrect">Các model cùng đúng</option></select><input id="query" placeholder="ID hoặc câu hỏi"><button id="prev">←</button><button id="next">→</button><span id="count"></span></div><div id="cases"></div>
</main><script>const D=__PAYLOAD__;
const T={left_right:'Left/Right',front_behind:'Front/Behind',visibility:'Visibility',facing:'Facing'},es=x=>String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])),pct=x=>x==null?'—':Number(x).toFixed(2)+'%';
const table=(id,heads,rows)=>document.getElementById(id).innerHTML='<table><thead><tr>'+heads.map(x=>'<th>'+es(x)+'</th>').join('')+'</tr></thead><tbody>'+rows.map(r=>'<tr>'+r.map(x=>'<td>'+es(x)+'</td>').join('')+'</tr>').join('')+'</tbody></table>';
table('scores',['Model','Task','Số câu','Đúng','Sai','Accuracy','CircularEval'],D.summaries.map(r=>[r.model,T[r.task]||r.task,r.rows,r.correct,r.wrong,pct(r.accuracy_pct),pct(r.circular_accuracy_pct)]));
const tasks=[...new Set(D.rows.map(r=>r.task))];for(const id of ['task','pairtask'])document.getElementById(id).innerHTML=tasks.map(t=>'<option value="'+es(t)+'">'+es(T[t]||t)+'</option>').join('');
function showpairs(){table('pairs',['Model A','Model B','N','Cùng đúng','Chỉ A đúng','Chỉ B đúng','Cùng sai'],D.pairs.filter(r=>r.task===pairtask.value&&r.unit===unit.value).map(r=>[r.a,r.b,r.n,r.both_correct,r.a_only_correct,r.b_only_correct,r.both_wrong]))}pairtask.onchange=unit.onchange=showpairs;showpairs();
const grouped=new Map();for(const r of D.rows){const k=r.task+':'+r.sample_id;if(!grouped.has(k))grouped.set(k,[]);grouped.get(k).push(r)}let page=0;
function show(){let caseset=[...grouped.values()].filter(rs=>rs[0].task===task.value&&(!query.value||(rs[0].sample_id+' '+rs[0].question+' '+rs[0].circular_group).toLowerCase().includes(query.value.toLowerCase())));caseset=caseset.filter(rs=>filter.value==='allwrong'?rs.every(r=>!r.correct):filter.value==='allcorrect'?rs.every(r=>r.correct):filter.value==='disagree'?rs.some(r=>r.correct)&&rs.some(r=>!r.correct):true);page=Math.min(page,Math.max(0,Math.ceil(caseset.length/8)-1));count.textContent=caseset.length+' câu · Trang '+(page+1);document.getElementById('cases').innerHTML=caseset.slice(page*8,page*8+8).map(rs=>{let c=rs[0];return '<article class="case"><h3>#'+es(c.sample_id)+' · '+es(c.circular_group)+'</h3>'+(c.image?'<img class="photo" loading="lazy" src="'+es(c.image)+'">':'')+'<p>'+es(c.question)+'</p><p>'+Object.entries(c.choices).map(([k,v])=>es(k+'. '+v)).join(' · ')+'<br><b>Đáp án: '+es(c.answer)+'</b></p><div class="models">'+rs.map(r=>'<div class="model"><b>'+es(r.model)+'</b> <b class="'+r.outcome+'">'+(r.correct?'Đúng':'Sai')+'</b><p>Prediction: '+es(r.prediction||'—')+'</p>'+(r.raw_output?'<details><summary>Output</summary><pre>'+es(r.raw_output)+'</pre></details>':'')+(r.record_path?'<p><a target="_blank" href="'+es(r.record_path)+'">JSON / trace gốc</a></p>':'')+(r.trace_viewer?'<p><a target="_blank" href="'+es(r.trace_viewer)+'">Mở trace từng bước</a></p>':'')+'</div>').join('')+'</div></article>'}).join('')}
task.onchange=filter.onchange=query.oninput=()=>{page=0;show()};prev.onclick=()=>{page=Math.max(0,page-1);show()};next.onclick=()=>{page++;show()};show();
</script></html>'''
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(template.replace("__PAYLOAD__", payload), encoding="utf8")
    return target
