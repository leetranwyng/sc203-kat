"""Trace integrity, exact renderer replay and conservative failure triage. CPU only."""
import ast, collections, csv, hashlib, html, io, json, math, re, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from trace_runtime import atomic_json, unpack, artifact_ref
from apc_common import parse_option

REQUIRED=('get_objects_of_interest','get_reference_viewer','estimate_scene_geometry','run_depth_intrinsic_estimation',
          'run_detection','run_detection_refinement','unproject_to_3D_from_bbox','run_orientation_estimation',
          'top_view_position','do_perspective_change','do_projection_2d','do_abstraction','do_bipartition',
          'do_localization','symbolic_layout_prompting')

def refs(value):
    if isinstance(value,dict):
        for k,v in value.items():
            if k in ('image','array') and isinstance(v,str) and v.startswith('artifacts/'):yield v
            else:yield from refs(v)
    elif isinstance(value,list):
        for v in value:yield from refs(v)

def read_events(folder):
    path=Path(folder)/'calls.jsonl'
    return [json.loads(s) for s in path.read_text(encoding='utf8').splitlines()] if path.exists() else []

def missing_refs(root,folder):
    events=read_events(folder)
    if not events:return ['calls.jsonl']
    return sorted({s for e in events for s in refs(e) if not (Path(root)/s).is_file()})

def renderer(repo):
    # Execute only the shipped renderer and half-plane helper, without importing any vision model.
    ns=dict(np=np,Image=Image,ImageDraw=ImageDraw,math=math,Dict=dict,List=list)
    for file,names in [('utils.py',('_line_rect_intersections','_fill_halfplane')),('localization.py',('render_final_image',))]:
        tree=ast.parse((Path(repo)/'sympl'/file).read_text(encoding='utf8'))
        for node in tree.body:
            if isinstance(node,ast.FunctionDef) and node.name in names:
                node.decorator_list=[]
                exec(compile(ast.Module(body=[node],type_ignores=[]),file,'exec'),ns)
    return ns['render_final_image']

def strict_symbol(text,colors):
    text=str(text).strip().lower().strip(' .\n\t\"\'')
    for color in colors:
        if text in (color, color+' area', 'the '+color+' area', color+' dot', 'the '+color+' dot'):return color
    return None

def layout_decision(u,v,colors,mapping,background,row):
    regions=[];margins=[];warnings=[]
    for x,y in zip(u,v):
        if not np.isfinite([x,y]).all():regions.append('nonfinite');margins.append(None);continue
        ix,iy=int(round(float(x))),int(round(float(y)))
        if not (0<=ix<512 and 0<=iy<512):regions.append('outside');margins.append(None);continue
        rgb=background.getpixel((ix,iy));regions.append('yellow' if rgb==(255,255,0) else 'black' if rgb==(0,0,0) else 'unknown')
        patch=np.asarray(background)[max(0,iy-10):min(512,iy+11),max(0,ix-10):min(512,ix+11)]
        mixed=bool(np.any(np.any(patch!=rgb,axis=-1)))
        margins.append(mixed)
        if mixed:warnings.append('dot_overlaps_partition_boundary')
    names=[k for k in mapping if k!='ref'];expected=None
    if len(names)==1 and row['paper_task'] in ('left_right','visibility') and regions[0] in ('yellow','black'):
        text=('on the left' if regions[0]=='yellow' else 'on the right') if row['paper_task']=='left_right' else ('yes' if regions[0]=='yellow' else 'no')
        expected=parse_option(text,row['choices'])
    elif len(names)==2 and regions.count('yellow')==1 and all(c in ('yellow','black') for c in regions):
        expected=parse_option(names[regions.index('yellow')],row['choices'])
    else:warnings.append('layout_has_no_unique_answer')
    return dict(regions=regions,expected_option=expected,boundary_overlap=margins,warnings=warnings)

def analyze_one(root,row,repo=None):
    root=Path(root);folder=root/row['attempt_path'];events=read_events(folder)
    config_path=root/'run_config.json'
    cfg=json.loads(config_path.read_text()) if config_path.exists() else {}
    repo=repo or cfg.get('repo')
    starts={e['id']:e for e in events if e.get('phase')=='start'}
    ends=collections.defaultdict(list)
    for e in events:
        if e.get('phase')=='end':ends[e['component']].append(e)
    checks=dict(missing_artifacts=missing_refs(root,folder),
                missing_stages=[s for s in REQUIRED if s not in ends],
                unterminated_calls=[i for i in starts if not any(e.get('id')==i and e.get('phase') in ('end','error') for e in events)])
    diag=dict(sample_id=row['sample_id'],task=row['paper_task'],apc_correct=row['apc']['correct'],
              sympl_correct=row['correct'],status=row['status'],flags=[],evidence=[],integrity=checks,
              root_cause='unreviewed',certainty='unreviewed',previews=[])
    previews=folder/'previews';previews.mkdir(exist_ok=True)
    selections=[unpack(e['values'],root) for e in events if e.get('phase')=='snapshot' and e.get('component')=='refinement_selection']
    diag['refinement_selection']=selections
    if any(s['fallback_used'] for s in selections):diag['flags'].append('detection_refinement_used_fallback')
    def preview(img,name,label):
        p=previews/(name+'.png');img.save(p)
        diag['previews'].append(dict(path=p.relative_to(root).as_posix(),label=label))
    image=Image.open(folder/'input.png').convert('RGB') if (folder/'input.png').exists() else None
    if image:preview(image,'input','Original RGB input')
    if ends['run_depth_intrinsic_estimation']:
        depth,K=unpack(ends['run_depth_intrinsic_estimation'][-1]['result'],root)
        depth=np.asarray(depth);valid=depth[np.isfinite(depth)&(depth>0)]
        if valid.size:
            lo,hi=np.percentile(valid,[2,98]);scale=np.clip((depth-lo)/(hi-lo+1e-8),0,1)
            from matplotlib import colormaps
            rgb=(colormaps['viridis'](scale)[...,:3]*255).astype('uint8')
            preview(Image.fromarray(rgb),'depth',f'Depth preview: {lo:.3f}–{hi:.3f} (2–98 percentile; exact depth in NPZ)')
    for i,e in enumerate(ends['run_detection_refinement']):
        inp=unpack(starts[e['id']]['inputs'],root);box=unpack(e['result'],root)[0]
        if image:
            im=image.copy();ImageDraw.Draw(im).rectangle(box,outline='red',width=3)
            preview(im,f'box_{i}','Selected bbox: '+str(inp.get('category')))
    # Diagnostic depth masks are captured before the code discards rejected pixels.
    bands=[e for e in events if e.get('phase')=='snapshot' and e.get('component')=='depth_band']
    for i,e in enumerate(bands):
        data=unpack(e['values'],root);xs=np.asarray(data['xs']);ys=np.asarray(data['ys']);keep=np.asarray(data['keep'],dtype=bool)
        ratio=float(keep.mean()) if keep.size else 0
        diag.setdefault('depth_keep_ratios',[]).append(ratio)
        if ratio<.3:diag['flags'].append('low_depth_support_requires_review')
        if image:
            arr=np.asarray(image).copy();xs=xs.astype(int);ys=ys.astype(int)
            arr[ys,xs]=(arr[ys,xs]*.35+np.where(keep[:,None],[0,255,80],[255,0,80])*.65).astype('uint8')
            preview(Image.fromarray(arr),f'depth_support_{i}',f'Depth selection {i}: kept {ratio:.1%}; green kept, pink rejected')
        hist=np.asarray(data['hist']);edges=np.asarray(data['bin_edges'])
        chart=Image.new('RGB',(640,220),'white');d=ImageDraw.Draw(chart)
        if hist.size and hist.max():
            for j,h in enumerate(hist):
                x=20+j*600/len(hist);d.rectangle((x,190-float(h)/hist.max()*160,x+max(1,600/len(hist)-1),190),fill='#456caa')
        d.text((15,5),f"z_mode={data['z_mode']:.4f}, band={data['band_param']}, retained={ratio:.1%}",fill='black')
        d.text((15,198),f'depth range: {edges[0]:.3f} ... {edges[-1]:.3f}',fill='black')
        preview(chart,f'hist_{i}','Depth histogram (exact counts/edges in snapshot)')
    if ends['estimate_scene_geometry']:
        scene=unpack(ends['estimate_scene_geometry'][-1]['result'],root)
        if scene:
            diag['scene_camera']=scene
            # A camera X/Z plot, explicitly labelled rather than presented as a GT reconstruction.
            objects={k:v for k,v in scene.items() if k!='camera'}
            pts=[np.asarray(v['position']) for v in objects.values()]
            if pts and np.isfinite(pts).all():
                import matplotlib;matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                fig,ax=plt.subplots(figsize=(6,4))
                for name,obj in objects.items():
                    p=np.asarray(obj['position']);ax.scatter(p[0],p[2]);ax.text(p[0],p[2],name)
                    if obj['orientation'] is not None:
                        direction=np.asarray(obj['orientation']);ax.quiver(p[0],p[2],direction[0],direction[2],angles='xy',scale_units='xy',scale=1)
                ax.set(xlabel='Camera X',ylabel='Camera Z',title='Estimated camera X/Z; arrow = facing vector');ax.axis('equal')
                b=io.BytesIO();fig.savefig(b,format='png',bbox_inches='tight');plt.close(fig);b.seek(0)
                preview(Image.open(b).copy(),'geometry','Estimated positions and viewer orientation (not ground truth)')
    if ends['do_abstraction'] and ends['do_bipartition'] and ends['do_localization'] and repo:
        ab=unpack(ends['do_abstraction'][-1]['result'],root);mapping=ab[6]
        specs,u,v=unpack(ends['do_bipartition'][-1]['result'],root)
        actual=unpack(ends['do_localization'][-1]['result'],root)
        render=renderer(repo)
        replay=render(np.asarray(u),np.asarray(v),np.asarray(ab[4]),specs)
        matches=actual.size==replay.size and np.array_equal(np.asarray(actual),np.asarray(replay))
        checks['renderer_replay_exact']=matches
        if not matches:diag['flags'].append('renderer_replay_mismatch')
        preview(actual,'layout','Exact symbolic image sent to final VLM')
        background=render(np.array([]),np.array([]),np.empty((0,3)),specs)
        decision=layout_decision(u,v,ab[4],mapping,background,row)
        diag['layout']=decision;diag['mapping']=mapping;diag['flags']+=decision['warnings']
        final_start=starts.get(ends['symbolic_layout_prompting'][-1]['id']) if ends['symbolic_layout_prompting'] else None
        direct=[e for e in events if e.get('phase')=='start' and e['component']=='process_messages' and final_start and e.get('parent')==final_start['id']]
        final_raw=None
        if direct:
            qe=next((e for e in events if e.get('id')==direct[-1]['id'] and e.get('phase')=='end'),None)
            if qe:final_raw=qe['result']
        diag['final_vlm_raw']=final_raw
        one=len(mapping)==2;symbol=strict_symbol(final_raw,['yellow','black'] if one else [mapping[k] for k in mapping if k!='ref'])
        read_option=None
        if symbol and one:
            text=('on the left' if symbol=='yellow' else 'on the right') if row['paper_task']=='left_right' else ('yes' if symbol=='yellow' else 'no')
            read_option=parse_option(text,row['choices'])
        elif symbol:
            names=[k for k in mapping if k!='ref' and mapping[k]==symbol]
            if len(names)==1:read_option=parse_option(names[0],row['choices'])
        diag['strict_read_option']=read_option
        expected=decision['expected_option'];clear=matches and not any(decision['boundary_overlap']) and expected is not None
        if clear:
            diag['evidence'].append('Exact renderer replay; target region has a unique answer away from boundary.')
            if expected!=row['answer']:diag['flags'].append('layout_disagrees_with_gt_upstream_review')
            if read_option and read_option!=expected:diag['flags'].append('vlm_reading_disagrees_with_layout')
            if read_option and row['prediction']!=read_option:diag['flags'].append('parser_mapping_disagreement')
            if row['correct'] and expected!=row['answer']:diag['flags'].append('correct_answer_with_compensating_errors')
        if not symbol:diag['flags'].append('ambiguous_final_response_requires_review')
    if row['status']=='error':diag.update(root_cause='runtime_error',certainty='observed',failed_stage=row.get('failed_stage'))
    elif row['status']=='method_failure':
        message=str(row.get('raw_output',''))
        stage=next((v for k,v in [('refered objects','get_objects_of_interest'),('reference viewer not in','get_reference_viewer'),
                    ('reasoning category','select_category_from_question'),('object detection','estimate_scene_geometry'),
                    ('only reference','do_projection_2d'),('cannot find reference','do_projection_2d')] if k in message),None)
        diag.update(root_cause='pipeline_failure',certainty='observed',failed_stage=stage,failure_message=message)
    elif 'parser_mapping_disagreement' in diag['flags']:diag.update(root_cause='parser_or_mapping',certainty='observed_disagreement')
    elif 'vlm_reading_disagrees_with_layout' in diag['flags']:diag.update(root_cause='vlm_layout_reading',certainty='observed_disagreement')
    elif 'layout_disagrees_with_gt_upstream_review' in diag['flags']:diag.update(root_cause='upstream_geometry_or_partition',certainty='requires_review')
    elif row['correct']:diag.update(root_cause='no_detected_failure',certainty='automatic_checks_only')
    # Numpy arrays in scene are already decoded small lists from trace storage.
    atomic_json(folder/'diagnosis.json',diag)
    return diag

def export(root,repo):
    root=Path(root);results=[];diagnoses=[]
    for p in sorted((root/'traces').glob('*/result.json')):
        r=json.loads(p.read_text(encoding='utf8'));results.append(r)
        try:diagnoses.append(analyze_one(root,r,repo))
        except Exception as exc:
            diagnoses.append(dict(sample_id=r['sample_id'],root_cause='analysis_error',certainty='unreviewed',error=repr(exc),flags=[],previews=[]))
    (root/'predictions.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in results),encoding='utf8')
    atomic_json(root/'failure_cases.json',diagnoses)
    expected=[json.loads(x) for x in (root/'samples.jsonl').read_text(encoding='utf8').splitlines()]
    summary=dict(expected=len(expected),completed=len(results),correct=sum(r['correct'] for r in results),
        accuracy_on_completed=sum(r['correct'] for r in results)/len(results) if results else None,
        statuses=dict(collections.Counter(r['status'] for r in results)),
        triage=dict(collections.Counter(d['root_cause'] for d in diagnoses)),
        missing_samples=sorted({r['sample_id'] for r in expected}-{r['sample_id'] for r in results}),
        note='Selected diagnostic set, not full benchmark accuracy. No full CircularEval from one row/group.')
    summary['by_task']={task:dict(completed=len(group),correct=sum(r['correct'] for r in group),
                                  accuracy=sum(r['correct'] for r in group)/len(group))
        for task in ('left_right','visibility','facing') if (group:=[r for r in results if r['paper_task']==task])}
    summary['paired_outcomes_APC_SymPL']=dict(collections.Counter(
        ('1' if r['apc']['correct'] else '0')+('1' if r['correct'] else '0') for r in results))
    atomic_json(root/'summary.json',summary)
    validation={d['sample_id']:d.get('integrity',{}) for d in diagnoses}
    atomic_json(root/'trace_validation.json',validation)
    with (root/'failure_cases.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f);w.writerow(['sample_id','task','APC_correct','SymPL_correct','root_cause','certainty','flags'])
        for r,d in zip(results,diagnoses):w.writerow([r['sample_id'],r['paper_task'],r['apc']['correct'],r['correct'],d['root_cause'],d['certainty'],';'.join(d['flags'])])
    build_gallery(root,results,diagnoses,summary)
    return summary

def build_gallery(root,rows,diagnoses,summary):
    cards=[]
    for r,d in zip(rows,diagnoses):
        folder=root/r['attempt_path'];events=read_events(folder)
        trace_html=[]
        for e in events:
            # Exact raw inputs/outputs remain readable, with image and array links.
            def show(v):
                if artifact_ref(v,'image'):return '<a href="'+html.escape(v['image'])+'"><img loading="lazy" src="'+html.escape(v['image'])+'"></a>'
                if artifact_ref(v,'array'):return '<a href="'+html.escape(v['array'])+'">NPZ '+html.escape(str(v.get('shape')))+' '+html.escape(str(v.get('dtype')))+'</a>'
                if isinstance(v,dict):return ''.join('<div><b>'+html.escape(str(k))+':</b> '+show(x)+'</div>' for k,x in v.items())
                if isinstance(v,list):return '<pre>'+html.escape(json.dumps(v,ensure_ascii=False,indent=1))+'</pre>' if not any(isinstance(x,dict) for x in v) else ''.join(show(x) for x in v)
                return '<span>'+html.escape(str(v))+'</span>'
            trace_html.append('<details><summary>'+html.escape(f"{e.get('id','')} {e['phase']} · {e.get('component','')} parent={e.get('parent','')}")+'</summary>'+show(e)+'</details>')
        previews=''.join('<figure><a href="'+p['path']+'"><img loading="lazy" src="'+p['path']+'"></a><figcaption>'+html.escape(p['label'])+'</figcaption></figure>' for p in d.get('previews',[]))
        apc=''.join('<figure><img loading="lazy" src="'+p['path']+'"><figcaption>'+html.escape(p['label'])+'</figcaption></figure>' for p in r['apc'].get('previews',[]))
        pair=('1' if r['apc']['correct'] else '0')+('1' if r['correct'] else '0')
        review=dict(sample_id=r['sample_id'],entity_correct=None,viewer_correct=None,box_correct=None,
            depth_correct=None,orientation_correct=None,transform_correct=None,partition_correct=None,
            vlm_reading_correct=None,parser_correct=None,root_cause=d['root_cause'],certainty='unreviewed',evidence='',notes='')
        cards.append(f'''<article data-task="{r['paper_task']}" data-pair="{pair}" data-cause="{d['root_cause']}">
<h2>{html.escape(r['sample_id'])} · {r['paper_task']} · APC/SymPL {pair}</h2>
<p>{html.escape(r['source_question'])}</p><p>SymPL input: {html.escape(r['question'])}</p>
<pre>Choices: {html.escape(str(r['choices']))}\nGT {r['answer']} | APC {r['apc']['prediction']} | SymPL {r['prediction']}\nRaw SymPL: {html.escape(str(r['raw_output']))}</pre>
<p><b>Triage: {d['root_cause']} ({d['certainty']})</b> — {html.escape(', '.join(d['flags']))}</p>
<div class="columns"><section><h3>APC-Vis saved trace</h3>{apc}<details><summary>APC scene / results</summary><pre>{html.escape(json.dumps(r['apc'],ensure_ascii=False,indent=2))}</pre></details></section>
<section><h3>SymPL</h3>{previews}</section></div>
<details><summary>Diagnosis and evidence</summary><pre>{html.escape(json.dumps(d,ensure_ascii=False,indent=2))}</pre></details>
<details><summary>Full trace: inputs, outputs, tensors, VLM prompts</summary>{''.join(trace_html)}</details>
<details><summary>Manual review — edit JSON then download</summary><textarea>{html.escape(json.dumps(review,ensure_ascii=False,indent=2))}</textarea><button onclick="downloadReview(this)">Download review JSON</button></details>
</article>''')
    page='''<!doctype html><html lang="vi"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>SymPL × APC — 120 cases</title>
<style>body{font:16px/1.5 system-ui;background:#edf1f7;color:#18283e;margin:0;padding:24px}header,article{background:white;padding:24px;margin:auto auto 24px;max-width:1400px;border-radius:12px}.columns{display:grid;grid-template-columns:1fr 1fr;gap:24px}img{max-width:100%;max-height:440px}figure{margin:16px 0}figcaption{color:#456;font-size:14px}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f4f6fa;padding:12px}details{border:1px solid #ccd5e4;padding:10px;margin:8px 0}summary{cursor:pointer}select,input,button{padding:9px;margin:5px}textarea{width:95%;height:330px;font-family:monospace}@media(max-width:800px){.columns{display:block}}</style>
<header><h1>SymPL × APC-Vis — paired stage inspection</h1><p>Automatic triage is evidence, not ground-truth annotation of the upstream cause. Pink/green masks and X/Z diagrams are diagnostic overlays. “Exact symbolic image” is the real VLM input.</p>
<select id="task"><option value="">All tasks</option><option>left_right</option><option>visibility</option><option>facing</option></select>
<select id="pair"><option value="">All outcomes</option><option value="10">APC correct / SymPL wrong</option><option value="01">APC wrong / SymPL correct</option><option value="00">Both wrong</option><option value="11">Both correct</option></select>
<input id="search" placeholder="ID, error, stage, triage…"><span id="count"></span><details><summary>Run summary</summary><pre>SUMMARY</pre></details></header>CARDS
<script>function filter(){let n=0;for(const a of document.querySelectorAll('article')){a.hidden=(task.value&&a.dataset.task!==task.value)||(pair.value&&a.dataset.pair!==pair.value)||(search.value&&!a.textContent.toLowerCase().includes(search.value.toLowerCase()));if(!a.hidden)n++;}document.getElementById('count').textContent=n+' cases';}for(const id of ['task','pair','search'])document.getElementById(id).addEventListener('input',filter);filter();function downloadReview(b){let s=b.parentNode.querySelector('textarea').value;let j;try{j=JSON.parse(s)}catch(e){alert(e);return}let u=URL.createObjectURL(new Blob([JSON.stringify(j,null,2)],{type:'application/json'}));let a=document.createElement('a');a.href=u;a.download='review_'+j.sample_id+'.json';a.click();setTimeout(()=>URL.revokeObjectURL(u),1000);}</script></html>'''
    (root/'gallery.html').write_text(page.replace('SUMMARY',html.escape(json.dumps(summary,indent=2))).replace('CARDS',''.join(cards)),encoding='utf8')

if __name__=='__main__':
    cfg=json.loads(Path(sys.argv[1]).read_text());print(json.dumps(export(cfg['output'],cfg['repo']),indent=2))
