"""Build portable Kaggle notebooks, embedding the cloned SymPL source and audited installers."""
import ast, base64, hashlib, io, json, re, zipfile
from pathlib import Path

here=Path(__file__).resolve().parent
nbdir=here.parent
root=nbdir.parent
sym=root.parent/'SymPL'
spatial=root.parent/'Spatial-SSRL'
template=json.loads((nbdir/'apc_qwen25vl_3b_sanity_kaggle.ipynb').read_text(encoding='utf8'))
def read(path):return path.read_text(encoding='utf-8-sig')
def source(i):return ''.join(template['cells'][i]['source'])
def embedded(files):
    buf=io.BytesIO()
    with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
        for name,value in sorted(files.items()):
            info=zipfile.ZipInfo(name,(1980,1,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;z.writestr(info,value)
    return buf.getvalue()

with zipfile.ZipFile(root/'replay_results/apc_step_c_replay.zip') as z:
    locks=json.loads(z.read('source_lock.json'))
    sanity=next(r for r in map(json.loads,z.read('samples.jsonl').splitlines()) if r['sample_id']=='5502')
    sanity_img=z.read('full/trace/apc_vis/5502/input.png')

# Derive only the exercised SymPL modules from the local snapshot.
symfiles={p.relative_to(sym).as_posix():p.read_bytes() for p in (sym/'sympl').rglob('*') if p.suffix in ('.py','.yaml')}
symfiles['LICENSE.md']=(sym/'LICENSE.md').read_bytes()
def edit(name,a,b):
    t=symfiles[name].decode();assert a in t,(name,a);symfiles[name]=t.replace(a,b).encode()
det='sympl/vision_modules/detection.py'
# The checked-out revision leaves its superseded prompt inside an unclosed
# triple-quoted block. Drop that inert alternative and retain the active prompt.
original=symfiles[det].decode()
start=original.index("        '''",original.index("Choose the most appropriate image based on robust visual/appearance cues",original.index("def run_detection_refinement(")))
end=original.index("        # Make conversation",start)
edit(det,original[start:end],'')
edit(det,'load_model(config_path, ckpt_path)','load_model(config_path, ckpt_path, device=str(self.device))')
edit(det,'        boxes, scores, _ = predict(', '        with torch.cuda.device(self.device):\n            boxes, scores, _ = predict(')
edit(det,'text_threshold=self.config.detection.text_threshold,','text_threshold=self.config.detection.text_threshold, device=str(self.device),')
edit(det,'f"detection_candidates_{category}.png"', 'f"detection_candidates_{__import__(\'hashlib\').sha256(category.encode()).hexdigest()[:16]}.png"')
ori='sympl/vision_modules/orientation.py'
edit(ori,'from .src.orient_anything.render import render, Model','')
edit(ori,'        axis_model = Model','        from .src.orient_anything.render import render, Model\n        axis_model = Model')
original=symfiles[ori].decode()
edit(ori,original[original.index('orientation_ckpt_path = hf_hub_download'):original.index('        self.orientation_model.load_state_dict')],'orientation_ckpt_path = config.orientation.model_path\n')
edit(ori,'"facebook/dinov2-large",','config.orientation.processor_path,')
edit('sympl/vision_modules/depth.py','depth_pred["focallength_px"].cpu().numpy().astype(np.float32)','float(depth_pred["focallength_px"].item())')
edit('sympl/sympl_pipeline.py','if ref_viewer not in objs_of_interest:', 'if not objs_of_interest or ref_viewer not in objs_of_interest:')
# Only dependencies used by SymPL. Native builds use the previously exercised commit pins.
native=read(nbdir/'apc_install_native.py').replace("repo/'apc/vision_modules/src'","repo/'sympl/vision_modules/src'")
a=native.index('for name,(url,ref)');b=native.index('# Do not install',a)
native=native[:a]+'''for name in ('GroundingDINO','ml-depth-pro','orient_anything','pytorch3d','segment-anything'):
    entry=LOCKS[name]
    path=src/name
    if not (path/'.git').exists():
        path.mkdir(exist_ok=True)
        run(['git','init','-q',str(path)])
        run(['git','remote','add','origin',entry['url']],cwd=path)
    if subprocess.run(['git','cat-file','-e',entry['sha']],cwd=path,stderr=subprocess.DEVNULL).returncode:
        run(['git','fetch','--depth','1','origin',entry['sha']],cwd=path)
    run(['git','checkout','--detach',entry['sha']],cwd=path)
    lock[name]=entry
lock_path.write_text(json.dumps(lock,indent=2))
'''+native[b:]
native='LOCKS='+repr(locks)+'\n'+native.replace("'ml-depth-pro','detectron2','pytorch3d','GroundingDINO'","'ml-depth-pro','pytorch3d','GroundingDINO'")
patches=read(nbdir/'apc_kaggle_patches.py')
patches=patches[:patches.index('    def detection(s):')]+patches[patches.index('    # External sources exist'):]
patches=patches.replace('apc/vision_modules/src','sympl/vision_modules/src')
asset=read(nbdir/'apc_prepare_assets.py').replace("'checkpoints/sam_vit_h_4b8939.pth',",'')
asset='\n'.join(l for l in asset.splitlines() if not l.startswith(" 'sam_vit_h_4b8939.pth':"))
adapter=read(nbdir/'apc_qwen_adapter.py').replace('from apc.vlms.vlm_base import VLMBase','from torch.nn import Module as VLMBase').replace('super().__init__(device)','super().__init__()').replace('raw_output=raw[:240]','raw_output=raw,prompt=text')
data=read(nbdir/'apc_prepare_data.py')
start=data.index('    images_dir = work');end=data.index('    if not items:raise')
data=data[:data.index("assets=json.loads")]+'''if cfg['sanity']:
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
'''+data[start:end]+'''
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
(out/'samples.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\\n' for r in items),encoding='utf8')
print('Prepared',len(items),'rows',flush=True)
'''
fmt=r'You FIRST think about the reasoning process as an internal monologue and then provide the final answer. The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \boxed{}.'
setup="""import os,json,sys,time,shutil,hashlib,subprocess,io,base64,zipfile
from pathlib import Path
WORK=Path('/kaggle/temp')/('sc203_'+METHOD); WORK.mkdir(parents=True,exist_ok=True)
REPO=WORK/'repo'; SCRIPTS=WORK/'scripts'; SCRIPTS.mkdir(exist_ok=True); REPO.mkdir(exist_ok=True)
OUT=Path('/kaggle/working')/(METHOD+('_sanity' if SANITY else '_full'))
if RESUME_INPUT and not (OUT/'manifest.json').exists():
    src=Path(RESUME_INPUT)
    assert (src/'manifest.json').is_file(), 'Choose the previous output folder containing manifest.json'
    shutil.copytree(src,OUT,dirs_exist_ok=True)
OUT.mkdir(parents=True,exist_ok=True)
try:
    from kaggle_secrets import UserSecretsClient
    token=UserSecretsClient().get_secret('HF_token')
    if token:os.environ['HF_TOKEN']=token.strip()
except Exception:pass
os.environ['HF_HUB_DISABLE_XET']='1'
os.environ['TOKENIZERS_PARALLELISM']='false'
CONFIG=WORK/'config.json'
"""
ssrl_setup="""from huggingface_hub import HfApi,snapshot_download
old=OUT/'manifest.json'
rev=json.loads(old.read_text())['identity']['model_sha'] if old.exists() else 'main'
if MODEL_INPUT:
    model=Path(MODEL_INPUT)
    h=hashlib.sha256()
    for p in sorted(p for p in model.iterdir() if p.suffix in ('.json','.txt','.safetensors','.jinja')):
        h.update(p.name.encode())
        with p.open('rb') as f:
            for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    sha='local-'+h.hexdigest()
else:
    sha=HfApi().model_info(CFG['model_id'],revision=rev).sha
    model=Path(snapshot_download(CFG['model_id'],revision=sha,cache_dir=str(WORK/'weights'),allow_patterns=['*.json','*.txt','*.jinja','*.safetensors']))
assert (model/'config.json').is_file() and list(model.glob('*.safetensors'))
CFG.update(model_sha=sha,weights=str(model))
"""
sym_setup="""subprocess.run([PY,str(SCRIPTS/'native.py'),str(CONFIG)],env=ENV,check=True)
subprocess.run([PY,str(SCRIPTS/'assets.py'),str(CONFIG),str(OUT/'assets_ready.json')],env=ENV,check=True)
assets=json.loads((OUT/'assets_ready.json').read_text())
CFG.update(assets=assets['root'],weights=str(Path(assets['root'])/'qwen_model'),model_sha=assets['snapshots']['qwen_model']['sha'])
subprocess.run([PY,str(SCRIPTS/'patch.py'),str(REPO),CFG['assets'],str(OUT/'patches.json')],env=ENV,check=True)
shutil.copy2(WORK/'source_lock.json',OUT/'source_lock.json')
ENV['PYTHONPATH']+=os.pathsep+str(REPO/'sympl/vision_modules/src/orient_anything')
"""
launch="""CONFIG.write_text(json.dumps(CFG))
subprocess.run([PY,str(SCRIPTS/'data.py'),str(CONFIG)],cwd=REPO,env=ENV,check=True)
ENV.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
(OUT/'environment.txt').write_text(subprocess.check_output([str(UV),'pip','freeze','--python',PY],env=ENV,text=True))
shutil.copy2(WORK/'gpu_check.txt',OUT/'gpu_check.txt')
workers=2 if METHOD=='ssrl' and not SANITY else 1
processes=[]
try:
    for rank in range(workers):
        env=ENV.copy()
        if METHOD=='ssrl':env['CUDA_VISIBLE_DEVICES']=GPU_IDS[rank]
        log=(OUT/f'worker_{rank}.log').open('a')
        child=subprocess.Popen([PY,'-u',str(SCRIPTS/'worker.py'),str(CONFIG),str(rank),str(workers)],cwd=REPO,stdout=log,stderr=subprocess.STDOUT,env=env)
        processes.append((child,log))
    while any(p.poll() is None for p,_ in processes):
        for i,(p,_) in enumerate(processes):
            if p.poll() not in (0,None):raise RuntimeError((OUT/f'worker_{i}.log').read_bytes()[-4000:].decode(errors='replace'))
        print('Saved',sum(1 for _ in OUT.glob('traces/*/result.json')),'/',json.loads((OUT/'manifest.json').read_text())['selected_rows'],flush=True)
        time.sleep(15)
    for i,(p,_) in enumerate(processes):
        if p.returncode:raise RuntimeError((OUT/f'worker_{i}.log').read_bytes()[-4000:].decode(errors='replace'))
finally:
    for p,log in processes:
        if p.poll() is None:p.terminate()
        p.wait();log.close()
"""
score="""from collections import defaultdict
rows=[json.loads(p.read_text()) for p in sorted(OUT.glob('traces/*/result.json'))]
samples=[json.loads(l) for l in (OUT/'samples.jsonl').read_text().splitlines()]
by_id={r['sample_id']:r for r in rows}
assert set(by_id)=={s['sample_id'] for s in samples},'Incomplete; resume before scoring'
def metrics(values):
    groups=defaultdict(list)
    for r in values:groups[r['circular_group']].append(r['correct'])
    return dict(rows=len(values),row_accuracy=sum(r['correct'] for r in values)/len(values),
                groups=len(groups),circular_accuracy=sum(all(g) for g in groups.values())/len(groups),
                invalid_or_error=sum(r['status']!='ok' for r in values))
result=dict(overall=metrics(rows),by_category={k:metrics([r for r in rows if r['category']==k]) for k in sorted({r['category'] for r in rows})})
(OUT/'summary.json').write_text(json.dumps(result,indent=2))
with (OUT/'predictions.jsonl').open('w',encoding='utf8') as f:
    for r in rows:f.write(json.dumps({k:v for k,v in r.items() if k!='generation'},ensure_ascii=False)+'\\n')
print(json.dumps(result,indent=2),OUT)
if SANITY:
    r=rows[0]; print('Prediction:',r['prediction'],'Expected:',r['answer'],'Status:',r['status'])
    assert r['status']=='ok', 'Technical sanity failed; inspect traces/'+r['sample_id']+'/calls.jsonl and result.json'
"""
def make(method,sanity_mode):
    files={'scripts/worker.py':read(here/'worker.py'),'scripts/data.py':data,'scripts/apc_common.py':read(nbdir/'apc_common.py'),'scripts/experiment_qwen.py':adapter,'sanity.png':sanity_img,'sample.json':json.dumps(sanity)}
    if method=='sympl':
        files.update({'repo/'+k:v for k,v in symfiles.items()});files.update({'scripts/patch.py':patches,'scripts/native.py':native,'scripts/assets.py':asset})
    else:files['repo/LICENSE']=(spatial/'LICENSE').read_bytes()
    for filename,content in files.items():
        if filename.endswith('.py'):compile(content,filename,'exec')
    raw=embedded(files);sha=hashlib.sha256(raw).hexdigest()
    cells=[]
    def cell(t,s):
        if t=='code':ast.parse(s)
        c=dict(cell_type=t,metadata={},source=s.strip().splitlines(True))
        if t=='code':c.update(execution_count=None,outputs=[])
        cells.append(c)
    cfg=dict(method=method,sanity=sanity_mode,model_id='Qwen/Qwen2.5-VL-3B-Instruct' if method=='sympl' else 'internlm/Spatial-SSRL-3B',
        model_revision='66285546d2b821cf421d4f5eb2576359d3770cd3',dataset_id='ccvl/3DSRBench',data_revision='5c90e422b68d0484d50a3ff06b3096ea7d288026',data_file='3dsrbench_v1_vlmevalkit_circular.tsv',
        repo_commit='384c75d0219dfaaf33280e01ebe5f0d2e19eeed3' if method=='sympl' else '5b27b7b314e65dadccca6e065b8a4dc039534021',
        max_tokens=512 if method=='sympl' else 4096,format_prompt=fmt if method=='ssrl' else '',runtime_sha=sha,limit=None)
    protocol='SymPL + Qwen2.5-VL-3B. Full = đủ 2.078 lượt Left/Right, Visibility chuyển từ Front/Behind và Facing theo phạm vi paper. GPU 0 cho Qwen, GPU 1 cho vision. Qwen tự chọn category.' if method=='sympl' else 'Spatial-SSRL-3B chính thức. Full = tất cả subtask của TSV Circular gốc, giữ Front/Behind. Hai worker, mỗi GPU một bản model. Có thể đổi SCOPE sang paper_tasks_visibility để đối chiếu cùng câu SymPL.'
    cell('markdown','# '+cfg['model_id']+' · '+('sanity một ảnh' if sanity_mode else 'full 3DSRBench')+'\n\n'+protocol+'\n\nKaggle: bật Internet, chọn T4 ×2. Có thể cấp secret **HF_token**. Sanity dùng đúng một ảnh 3DSRBench nhúng sẵn (ID 5502). Tải/chạy các cell theo thứ tự. Setup và cache ở /kaggle/temp; kết quả và trace nằm trong /kaggle/working. Khi hết phiên, lưu Output và attach lại làm Input, đặt RESUME_INPUT đến thư mục chứa manifest.json. Dự trù dung lượng cho toàn bộ trace; runner lưu ngay sau mỗi lượt. Điểm sanity chỉ kiểm tra kỹ thuật.')
    cell('code',f"METHOD={method!r}\nSANITY={sanity_mode}\nSCOPE="+repr('paper_tasks_visibility' if method=='sympl' else 'all_original')+"\nMODEL_INPUT=None  # SymPL: root asset bundle APC; SSRL: thư mục checkpoint HF\nDATASET_INPUT=None  # TSV Circular cục bộ, hoặc tải HF\nRESUME_INPUT=None  # output folder của cùng phiên bản notebook")
    cell('code',setup+'\npayload=base64.b64decode('+repr(base64.b64encode(raw).decode())+')\nassert hashlib.sha256(payload).hexdigest()=='+repr(sha)+"\nwith zipfile.ZipFile(io.BytesIO(payload)) as z:z.extractall(WORK)\nCFG="+repr(cfg)+"\nCFG.update(work=str(WORK),output=str(OUT),repo=str(REPO),scope=SCOPE,asset_root=str(WORK/'assets'),asset_input=MODEL_INPUT,dataset_input=DATASET_INPUT)\nif METHOD=='sympl':assert SCOPE=='paper_tasks_visibility'\nassert SCOPE in ('all_original','paper_tasks_visibility')\nCONFIG.write_text(json.dumps(CFG))")
    cell('markdown','## Môi trường GPU')
    boot=source(5).replace("'requests','tqdm'","'python-box==7.3.2','qwen-vl-utils==0.0.14','av==13.1.0','requests','tqdm'")
    cell('code',boot)
    cell('markdown','## Weights và công cụ')
    if method=='ssrl':
        setup_script="import json,hashlib,sys\nfrom pathlib import Path\nCONFIG=Path(sys.argv[1]); CFG=json.loads(CONFIG.read_text()); OUT=Path(CFG['output']); WORK=Path(CFG['work']); MODEL_INPUT=CFG['asset_input']\n"+ssrl_setup+"\nCONFIG.write_text(json.dumps(CFG))"
        asset_cell="(SCRIPTS/'ssrl_assets.py').write_text("+repr(setup_script)+",encoding='utf8')\nsubprocess.run([PY,str(SCRIPTS/'ssrl_assets.py'),str(CONFIG)],env=ENV,check=True)\nCFG=json.loads(CONFIG.read_text())"
    else:
        asset_cell=read(here/'cuda124_cell.py')+'\n'+sym_setup
    cell('code',asset_cell+"\nCONFIG.write_text(json.dumps(CFG))")
    cell('markdown','## Inference và trace\nMỗi lượt có traces/<sample_id>/calls.jsonl (start/end/error, id và parent), sample.json, result.json. Các ảnh, crop, layout và mảng lớn được lưu một lần trong artifacts/ theo hash. Trường image/array chỉ tới file tương đối từ root output. generation trong result.json chứa chat template đầy đủ, output, token và thời gian. SymPL ghi điểm DINO, candidate/selected box, depth và K, angles/vector cùng các bước hình học. Spatial-SSRL dùng một VLM. Sample lỗi/invalid được ghi và tiếp tục; CUDA/OOM/disk hoặc lỗi setup dừng.')
    cell('code',launch)
    cell('markdown','## Chấm điểm\nChấm từng lượt và CircularEval (đúng tất cả hoán vị của nhóm). Giữ sai/invalid/lỗi trong mẫu số. Dùng predictions.jsonl và summary.json để tổng hợp; giữ cả traces và artifacts để visualize.')
    cell('code',score)
    target=here/(method+'_3b_'+('sanity' if sanity_mode else 'full')+'_kaggle.ipynb')
    target.write_text(json.dumps(dict(cells=cells,metadata=template['metadata'],nbformat=4,nbformat_minor=4),ensure_ascii=False,indent=1),encoding='utf8')
for method in ['sympl','ssrl']:
    for sanity_mode in [True,False]:make(method,sanity_mode)
