from pathlib import Path
import json,csv,base64,hashlib,io,re,zipfile,collections,sys
from PIL import Image
from gallery_traces import load_traces
root=Path(__file__).resolve().parents[1];out=root/'replay_results'/'paired_gallery';out.mkdir(exist_ok=True); pics=out/'images';pics.mkdir(exist_ok=True)
def read(p,n,lines=False):
 with zipfile.ZipFile(root/p) as z:
  data=z.read(n)
 return [json.loads(l) for l in data.splitlines()] if lines else json.loads(data)
A={str(r['sample_id']):r for r in read('apc_3dsrbench_full_results.zip','apc_vis_predictions.jsonl',1)}
P=read('pcd_3dsrbench_full_results.zip','pcd_3dsrbench_full/predictions.json')
apc_traces,pcd_traces=load_traces(root,out)
rows=[];groups=collections.defaultdict(list);assert len(set(str(r['row_id']) for r in P))==len(P)
for p in P:
 sid=str(p['row_id']);a=A[sid];opts={k:p[k] for k in 'ABCD' if p[k]}
 assert a['source_question']==p['question'] and a['source_choices']==opts and a['question_id']==p['qid'] and a['answer']==p['answer'] and Path(a['image_path']).stem==p['image_sha256'] and a['circular_group']==p['base_qid']
 ac=a['prediction']==a['answer'];pc=p['prediction']==p['answer'];assert ac==a['correct'] and pc==bool(p['hit'])
 key=('11' if pc else '10') if ac else ('01' if pc else '00')
 calls=[c['raw_output'] for c in a.get('vlm_calls',[])];viewer=[m for c in calls for m in re.findall(r'\+\+([^+\n]+)\+\+',c)]
 r=dict(id=sid,qid=p['qid'],g=p['base_qid'],t=a['paper_task'],k=key,img=p['image_sha256'],aq=a['question'],pq=p['question'],ao=a['choices'],po=opts,ap=a['prediction'],pp=p['prediction'],gt=p['answer'],branch=p['branch'],ref=p['ref_obj'],target=p['target_obj'],rewrite=p['rewrite_question'],view=p['viewpoint_raw'],scores=p['scores'],cam=p['camera_raw'],calls=calls,status=a['status'],error=a.get('traceback',''),v=viewer)
 r['apcTrace']=apc_traces.get(sid)
 diag=pcd_traces.get(sid)
 if diag:
  assert diag['qid']==p['qid'] and diag['image_sha256']==p['image_sha256'] and diag['answer']==p['answer']
  r['dual']={k:diag[k] for k in ('diagnostic_camera_prediction','diagnostic_camera_raw','diagnostic_contrastive_prediction','scores','branch','prediction','viewpoint_prompt','viewpoint_raw','camera_prompt','original_prompt')}
 rows.append(r);groups[r['g']].append(r)
for group in groups.values():
 assert len(group)==2
 for r in group:r['circ']=''.join('1' if all(x['k'][i]=='1' for x in group) else '0' for i in (0,1))
stats=collections.Counter((r['t'],r['k'],r['branch']) for r in rows); print(stats);print('Circular',collections.Counter((g[0]['t'],g[0]['circ']) for g in groups.values()))
print('APC / PCD / both traces:',sum(bool(r['apcTrace']) for r in rows),sum('dual' in r for r in rows),sum(bool(r['apcTrace']) and 'dual' in r for r in rows))
needed={r['img'] for r in rows}
remaining=needed-{p.stem for p in pics.glob("*.jpg")}
def save(h,b):
 if h not in remaining:return
 im=Image.open(io.BytesIO(b)).convert('RGB');im.thumbnail((960,800));im.save(pics/(h+'.jpg'),quality=87);remaining.remove(h)
with zipfile.ZipFile(root/'replay_results/pcd_step_c_replay_results.zip') as z:
 for n in z.namelist():
  if n.endswith('.img'):save(Path(n).stem,z.read(n))
if len(sys.argv)>1:
 csv.field_size_limit(2147483647)
 with open(sys.argv[1],encoding='utf-8-sig',newline='') as f:
  for row in csv.DictReader(f,delimiter='\t'):
   v=row['image'].strip()
   if not v.isdigit():
    b=base64.b64decode(v.split(',')[-1]);save(hashlib.sha256(b).hexdigest(),b)
   if not remaining:break
print('images',len(list(pics.glob('*.jpg'))),'required',len(needed))
template=(Path(__file__).parent/'paired_gallery.template').read_text(encoding='utf8')
(out/'index.html').write_text(template.replace('DATA_HERE',json.dumps(rows,ensure_ascii=False).replace('<','\\u003c')),encoding='utf8')
