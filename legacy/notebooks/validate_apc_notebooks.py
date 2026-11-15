"""Offline regression checks for APC notebook preparation, scoring and compatibility patches."""
import ast,base64,collections,contextlib,hashlib,io,json,math,shutil,sys,tempfile,types,zipfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];HERE=ROOT/'notebooks';sys.path.insert(0,str(HERE))
from apc_common import parse_option,choose_samples,read_jsonl
from apc_kaggle_patches import patch_repo
for value,expected in [('B','B'),('B. yes','B'),('yes','B'),('!!!!!!!!',None),('',None),('A or B',None),('B. no',None)]:assert parse_option(value,{'A':'no','B':'yes'})==expected
assert parse_option('same',{'A':'same','B':'same'}) is None
with tempfile.TemporaryDirectory() as temp:
 root=Path(temp);repo=root/'repo';shutil.copytree(ROOT/'apc',repo/'apc')
 external=repo/'apc/vision_modules/src'
 for name,text in {
  'orient_anything/paths.py':'DINO_LARGE="facebook/dinov2-large"\n',
  'orient_anything/vision_tower.py':'class Embed:\n def forward(self, pixel_values, bool_masked_pos: Optional[torch.Tensor] = None) -> torch.Tensor:\n  embeddings = self.dropout(embeddings)\n  return embeddings\n',
  'GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py':'text_encoder_type="bert-base-uncased"\n'}.items():
  p=external/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text)
 assets=root/'assets';changes=patch_repo(repo,assets)
 first={x['file']:(repo/x['file']).read_bytes() for x in changes}
 changes2=patch_repo(repo,assets)
 assert first=={x['file']:(repo/x['file']).read_bytes() for x in changes2},'Patching is not idempotent'
 detection=(repo/'apc/vision_modules/detection.py').read_text()
 assert 'device=str(self.device)' in detection and detection.count('device=str(self.device)')==2
 assert 'from .src.orient_anything.render import render, Model' not in '\n'.join((repo/'apc/vision_modules/orientation.py').read_text().splitlines()[:25])
 # Exercise the renderer's branches without a display/GPU, using the real patched method.
 renderer=(repo/'apc/renderer.py').read_text();tree=ast.parse(renderer)
 func=next(x for x in ast.walk(tree) if isinstance(x,ast.FunctionDef) and x.name=='render_visual_prompt')
 class Scene:
  def __init__(self):self.camera=types.SimpleNamespace(fov=None)
  def add_geometry(self,x):pass
  def save_image(self):return b'fixture'
 ns=dict(np=np,math=math,Dict=__import__('typing').Dict,Any=__import__('typing').Any,trimesh=types.SimpleNamespace(Scene=Scene))
 exec(compile(ast.Module(body=[func],type_ignores=[]),'renderer','exec'),ns)
 captured=[]
 class Self:
  draw_grid=lambda *a,**kw:None
  def make_cube(self,pos,ori,**kw):captured.append((list(pos),list(ori),kw['color']));return None
  add_wireframe=lambda *a:Image.new('RGB',(4,4))
 scene={'behind':dict(position=np.array([2.,1.,3.]),orientation=np.array([1,0,0]),color=['red',[1,0,0]]),
        'front':dict(position=np.array([0.,0.,-2.]),orientation=np.array([0,0,1]),color=['blue',[0,0,1]])}
 ns['render_visual_prompt'](Self(),scene,'camera',trace_save_dir=temp,render_whole_scene=False,visualize_trace=False)
 assert len(captured)==1 and captured[0][1]==[0,0,1] and captured[0][2][0][:3]==[0,0,1]
 captured.clear()
 ns['render_visual_prompt'](Self(),{'front':scene['front']},'camera',trace_save_dir=temp,render_whole_scene=True,visualize_trace=False)
 assert len(captured)==1,'all-front whole-scene branch failed'
 # Verify vectorized projection gives the same points as the old loops.
 depth=(repo/'apc/vision_modules/depth.py').read_text();start=depth.index('        ys, xs =');end=depth.index('        # Compute final 3D coordinates',start)
 segment=depth[start:end]
 def mode(v,keepdims=True):
  unique,count=np.unique(v,return_counts=True);return np.array([unique[np.argmax(count)]]),None
 for seed in range(10):
  rng=np.random.default_rng(seed);d=rng.choice([2.,2.1,2.5,3.],size=(4,5)).astype('float32');mask=rng.random((4,5))>.3
  K=np.array([[20.,0,2],[0,20.,2],[0,0,1.]])
  scope=dict(depth_unproj=d,segment_mask=mask,K=K,np=np,mode=mode)
  exec('\n'.join(x[8:] if x.startswith('        ') else x for x in segment.splitlines()),scope)
  ys,xs=np.where(mask);values=d[ys,xs];md=mode(values)[0][0];points=[]
  for y,x in zip(ys,xs):
   z=d[y,x]
   if z>=md-.1*md and z<=md+.1*md:points.append([z*(x-K[0,2])/K[0,0],-z*(y-K[1,2])/K[1,1],-z])
  np.testing.assert_allclose(scope['points_filtered'],np.array(points),atol=1e-12)
 print('PASS patches: idempotent, correct vision GPU, lazy debug import, renderer orientation/color alignment/all-front/zero-XY branches, vectorized geometry equivalence.')

# Execute actual data preparation/scoring with local tiny images and mocked HF metadata.
with tempfile.TemporaryDirectory() as temp:
 root=Path(temp);work=root/'work';work.mkdir();repo=root/'repo';(repo/'demo').mkdir(parents=True)
 image=root/'fixture.jpg';Image.new('RGB',(6,4)).save(image);raw=image.read_bytes()
 shutil.copyfile(image,repo/'demo/sample_image_man.jpg')
 for name,obj in [('source_lock.json',{'src':'fixture'}),('patches.json',[]),('assets_ready.json',{'snapshots':{'qwen_model':{'sha':'model-sha'}}})]:(work/name).write_text(json.dumps(obj))
 import csv
 tsv=root/'test.tsv'
 with tsv.open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=['index','qid','question','A','B','answer','category','image'],delimiter='\t');writer.writeheader()
  for i,(qid,im) in enumerate([('q',base64.b64encode(raw).decode()),('q-1','0'),('r','0'),('r-1','2')]):
   writer.writerow(dict(index=i,qid=qid,question='Which?',A='no',B='yes',answer='B',category='orientation_on_the_left',image=im))
 cfg=dict(work=str(work),repo=str(repo),output=str(work/'full'),sanity=False,dataset_input=str(tsv),data_revision='pinned',scope='all_original',limit=None,modes=['direct','apc_vis'],parse_policy='strict_letter_or_unique_exact_choice_v1')
 path=root/'cfg.json';path.write_text(json.dumps(cfg))
 fake=types.ModuleType('huggingface_hub');fake.HfApi=lambda:None;fake.hf_hub_download=lambda *a,**k:(_ for _ in ()).throw(AssertionError('Unexpected download'))
 def prepare():
  with patch.dict(sys.modules,{'huggingface_hub':fake}),patch.object(sys,'argv',['prepare',str(path)]),contextlib.redirect_stdout(io.StringIO()):exec(compile((HERE/'apc_prepare_data.py').read_text(),'data','exec'),{'__name__':'__main__'})
 prepare();prepare()
 out=Path(cfg['output']);manifest=json.loads((out/'manifest.json').read_text());samples=[json.loads(x) for x in (out/'samples.jsonl').read_text().splitlines()]
 assert len(samples)==4 and manifest['circular_groups']==2 and len(list((work/'data/images').glob('*.img')))==1
 (out/'full').mkdir()
 for m in cfg['modes']:
  records=[dict(sample_id=r['sample_id'],fingerprint=manifest['fingerprint'],mode=m,status='ok',prediction='B',seconds=1) for r in samples]
  if m=='apc_vis':records[0].update(prediction=None,output_issue='degenerate_exclamation');records[2].update(status='error',prediction=None)
  (out/'full'/f'predictions_{m}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
 with patch.object(sys,'argv',['score',str(out)]),contextlib.redirect_stdout(io.StringIO()):exec(compile((HERE/'apc_score.py').read_text(),'score','exec'),{})
 summary=json.loads((out/'summary.json').read_text());assert summary['direct']['row_accuracy']==1 and summary['apc_vis']['row_accuracy']==.5
 assert summary['apc_vis']['circular_accuracy']==0 and summary['apc_vis']['pipeline_errors']==1 and summary['apc_vis']['invalid_outputs']==1
 assert summary['apc_vis']['all_rows_attempted'] and not summary['apc_vis']['complete']
 cfg.update(sanity=True,output=str(work/'sanity'),sanity_image=None,sanity_question='Which?',sanity_choices={'A':'left','B':'right'})
 path.write_text(json.dumps(cfg));prepare();assert len((work/'sanity/samples.jsonl').read_text().splitlines())==1
 print('PASS data/scoring: base64 reference images, image dedup, immutable fingerprint resume, exact circular groups, paired modes, failures/invalid in denominator, one-image sanity without dataset download.')
for path in HERE.glob('apc*.ipynb'):
 n=json.loads(path.read_text(encoding='utf-8'))
 for c in n['cells']:
  if c['cell_type']=='code':
   ast.parse(''.join(c['source']));assert not c['outputs']
 text=json.dumps(n)
 assert 'do_sample' in text and 'apc_vis' in text and 'Xvfb' in text
print('PASS notebook syntax, clean outputs, embedded source bundle and both full/sanity artifacts.')
# Actual worker control flow with fake GPU/pipeline, including a sample error and invalid output.
import random,signal,traceback
experiment_source=(HERE/'apc_experiment.py').read_text(encoding='utf-8-sig')
functions=[n for n in ast.parse(experiment_source).body if isinstance(n,ast.FunctionDef)]
from apc_common import atomic_json
seen=[]
class GPUError(RuntimeError):pass
cuda=types.SimpleNamespace(OutOfMemoryError=GPUError,manual_seed_all=lambda s:None,reset_peak_memory_stats=lambda i:None,max_memory_allocated=lambda i:1024)
class Experiment:
 def __init__(self,cfg,assets,callback=None,sanity=False):self.vlm=types.SimpleNamespace(calls=[])
 def run(self,row,mode,scratch):
  seen.append((row['sample_id'],mode))
  if mode=='apc_vis' and row['sample_id']=='0':raise ValueError('No detection fixture')
  return '!!!!!!!!' if mode=='apc_vis' and row['sample_id']=='1' else 'B. yes'
ns=dict(Path=Path,json=json,hashlib=hashlib,random=random,np=np,torch=types.SimpleNamespace(cuda=cuda,manual_seed=lambda s:None),
 time=__import__('time'),os=__import__('os'),signal=types.SimpleNamespace(SIGALRM=14,signal=lambda *a:None,alarm=lambda *a:None),
 shutil=shutil,re=__import__('re'),traceback=traceback,atomic_json=atomic_json,parse_option=parse_option,read_jsonl=read_jsonl,APCExperiment=Experiment)
exec(compile(ast.Module(body=functions,type_ignores=[]),'worker','exec'),ns)
with tempfile.TemporaryDirectory() as temp:
 work=Path(temp);out=work/'results';out.mkdir()
 cfg=dict(work=str(work),output=str(out),modes=['direct','apc_vis'],retry_failed=False,seed=42,sample_timeout_seconds=900)
 (work/'assets_ready.json').write_text('{}')
 rows=[dict(sample_id=str(i),question='Which?',choices={'A':'no','B':'yes'}) for i in range(4)]
 (out/'samples.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows));m=out/'manifest.json';m.write_text(json.dumps(dict(config=cfg,fingerprint='test')))
 with patch.object(sys,'argv',['worker','--manifest',str(m)]),contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):ns['main']()
 assert len(seen)==8 and json.loads((out/'full/state.json').read_text())['stage']=='complete'
 apc=read_jsonl(out/'full/predictions_apc_vis.jsonl');assert apc['0']['status']=='error' and apc['1']['prediction'] is None and len(apc)==4
 with patch.object(sys,'argv',['worker','--manifest',str(m)]),contextlib.redirect_stdout(io.StringIO()):ns['main']()
 assert len(seen)==8,'Resume repeated completed work'
print('PASS worker: paired full coverage, per-sample detection error continues, ! invalid retained, state complete, resume skips existing attempts.')
