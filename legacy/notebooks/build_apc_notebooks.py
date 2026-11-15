"""Build two self-contained Kaggle notebooks from the local APC source snapshot."""
import ast,base64,copy,hashlib,io,json,pprint,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];HERE=ROOT/'notebooks'
REFERENCE=Path(r'E:\SC203\SenseNova-SI\notebooks\sensenova_qwen25vl3b_full_continue_invalid_2gpu.ipynb')
ref=json.loads(REFERENCE.read_text(encoding='utf-8'))
def source(n,i):return ''.join(n['cells'][i]['source'])
def embedded(n,i):return next(x.args[0].value for x in ast.walk(ast.parse(source(n,i))) if isinstance(x,ast.Call) and isinstance(x.func,ast.Attribute) and x.func.attr=='write_text' and x.args and isinstance(x.args[0],ast.Constant) and isinstance(x.args[0].value,str))
def read(name):return (HERE/name).read_text(encoding='utf-8-sig')
def md(s):return dict(cell_type='markdown',metadata={},source=s.strip().splitlines(keepends=True))
def code(s):return dict(cell_type='code',metadata={},source=s.strip().splitlines(keepends=True),execution_count=None,outputs=[])
# Reproducible, small source bundle: code/config, license and one repo demo image; no weights/PDF/notebooks.
buf=io.BytesIO()
with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
 paths=sorted([p for p in (ROOT/'apc').rglob('*') if p.is_file() and p.suffix in ('.py','.yaml')]+[ROOT/'LICENSE',ROOT/'demo/sample_image_man.jpg'])
 for p in paths:
  info=zipfile.ZipInfo(p.relative_to(ROOT).as_posix(),(1980,1,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;z.writestr(info,p.read_bytes())
bundle=buf.getvalue();snapshot_sha=hashlib.sha256(bundle).hexdigest();b64=base64.b64encode(bundle).decode()
scripts=['apc_common.py','apc_kaggle_patches.py','apc_install_native.py','apc_prepare_assets.py','apc_qwen_adapter.py','apc_experiment.py','apc_score.py']
script_map={name:read(name) for name in scripts}
runtime_sha=hashlib.sha256(json.dumps(script_map,sort_keys=True).encode()).hexdigest()
# Reuse the already exercised uv/GPU bootstrap, matching APC's published torch/Python family.
bootstrap=source(ref,5).replace('3.11','3.10').replace('py311','py310').replace('2.5.1','2.4.1').replace('0.20.1','0.19.1').replace('cu121','cu124').replace('(3, 11)','(3, 10)')
bootstrap=bootstrap.replace('import torchvision, timm','import torchvision, timm')
# Restored Kaggle output files can lose executable bits; repair reused uv too.
bootstrap=bootstrap.replace('BOOT_ENV = os.environ.copy()', 'UV.chmod(UV.stat().st_mode | 0o111)\n\nBOOT_ENV = os.environ.copy()')
bootstrap=bootstrap.replace("subprocess.run([str(UV), 'python', 'install',", """# Restore execute permissions before uv tries to inspect an existing Python.
for binary in (WORK / 'python_runtimes').glob('cpython-3.10*/bin/python*'):
    if binary.is_file():binary.chmod(binary.stat().st_mode | 0o111)
for binary in (VENV / 'bin').glob('*'):
    if binary.is_file():binary.chmod(binary.stat().st_mode | 0o111)
subprocess.run([str(UV), 'python', 'install',""",1)
bootstrap=bootstrap.replace("subprocess.run([PY, '-c',", """# A restored venv may retain pyvenv.cfg but have a broken interpreter symlink.
if not Path(PY).exists() and (VENV / 'pyvenv.cfg').is_file():
    managed = subprocess.check_output([str(UV), 'python', 'find', '--managed-python', '3.10'], env=BOOT_ENV, text=True).strip()
    executable = Path(managed).resolve(strict=True)
    executable.chmod(executable.stat().st_mode | 0o111)
    link = Path(PY)
    link.parent.mkdir(parents=True, exist_ok=True)
    link.unlink(missing_ok=True)
    link.symlink_to(executable)
subprocess.run([PY, '-c',""",1)
# Source modules are importable in the subprocess; no kernel torch dependency.
config=dict(model_id='Qwen/Qwen2.5-VL-3B-Instruct',model_revision='main',dataset_id='ccvl/3DSRBench',
 data_revision='5c90e422b68d0484d50a3ff06b3096ea7d288026',data_file='3dsrbench_v1_vlmevalkit_circular.tsv',
 dataset_input=None,asset_input=None,asset_root='/kaggle/temp/apc_assets',experiment='apc_qwen3b_paper_tasks_visibility_v3',
 scope='paper_tasks_visibility',modes=['direct','apc_vis'],seed=42,dtype='float16',min_pixels=256*28**2,max_pixels=1280*28**2,
 intermediate_max_new_tokens=512,final_max_new_tokens=32,max_input_tokens=8192,detection_box_threshold=0.15,detection_text_threshold=0.05,
 limit=None,sample_timeout_seconds=900,load_timeout_seconds=1800,retry_failed=False,parse_policy='strict_letter_or_unique_exact_choice_v1',
 source_snapshot_sha=snapshot_sha,runtime_sha=runtime_sha)
# Official image/base64/reference/circular parsing reused from previously tested notebooks.
data_source=embedded(ref,7);tail=data_source[data_source.index('images_dir = work'):data_source.index('all_count = len(items)')]
data_script='''import base64,csv,hashlib,io,json,os,re,sys,shutil
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
'''+''.join('    '+line+'\n' if line else '\n' for line in tail.splitlines())+'''
    all_count=len(items);items=choose_samples(items,cfg)
    if not items:raise ValueError('No selected rows')
manifest=dict(config=cfg,model_sha=assets['snapshots']['qwen_model']['sha'],data_sha=data_sha,
    selected_rows=len(items),total_dataset_rows=all_count,circular_groups=len({r['circular_group'] for r in items}),
    source_lock=json.loads((work/'source_lock.json').read_text()),patches=json.loads((work/'patches.json').read_text()),assets=assets,
    protocol='Paper task families: left/right, visibility recast from front/behind, facing; 3B direct vs APC-Vis; exact-choice without LLM judge; not exact reproduction of 7B Table 1')
text=''.join(json.dumps(r,ensure_ascii=False)+'\\n' for r in items)
manifest['fingerprint']=hashlib.sha256((json.dumps(manifest,sort_keys=True)+text).encode()).hexdigest()
old_path=out/'manifest.json'
if old_path.exists() and json.loads(old_path.read_text())['fingerprint']!=manifest['fingerprint']:
    raise RuntimeError('Existing experiment has another fingerprint; choose a new experiment name, never rewrite old predictions.')
(out/'samples.jsonl').write_text(text,encoding='utf-8');atomic_json(old_path,manifest)
print('Prepared',len(items),'rows /',manifest['circular_groups'],'groups; model SHA',manifest['model_sha'],flush=True)
'''
ast.parse(data_script);script_map['apc_prepare_data.py']=data_script
config['runtime_sha']=hashlib.sha256(json.dumps(script_map,sort_keys=True).encode()).hexdigest()
native_cell=r'''
ENV.pop('HF_HUB_OFFLINE',None);ENV.pop('TRANSFORMERS_OFFLINE',None)
subprocess.run([PY,str(SCRIPTS/'apc_install_native.py'),str(CONFIG)],env=ENV,check=True)
subprocess.run([PY,str(SCRIPTS/'apc_prepare_assets.py'),str(CONFIG),str(WORK/'assets_ready.json')],env=ENV,check=True)
assets=json.loads((WORK/'assets_ready.json').read_text())
ENV['PYTHONPATH']+=os.pathsep+str(REPO/'apc/vision_modules/src/omni3d')+os.pathsep+str(REPO/'apc/vision_modules/src/orient_anything')
subprocess.run([PY,str(SCRIPTS/'apc_kaggle_patches.py'),str(REPO),assets['root'],str(WORK/'patches.json')],env=ENV,check=True)
ENV['HF_HUB_OFFLINE']='1';ENV['TRANSFORMERS_OFFLINE']='1'
# Rendering uses software GLX so no EGL/NVIDIA graphics context is required.
ENV['DISPLAY']=':99';ENV['LIBGL_ALWAYS_SOFTWARE']='1';ENV['MESA_GL_VERSION_OVERRIDE']='3.3';ENV['MPLBACKEND']='Agg'
if not Path('/tmp/.X99-lock').exists():
    display_log=(WORK/'xvfb.log').open('a')
    XVFB=subprocess.Popen(['Xvfb',':99','-screen','0','1280x1024x24','-nolisten','tcp'],stdout=display_log,stderr=subprocess.STDOUT,env=ENV)
    time.sleep(2)
probe="""
import io,json,torch,trimesh
from PIL import Image
from groundingdino import _C
import detectron2,pytorch3d,depth_pro,open3d,segment_anything
from apc.apc_pipeline import APC
scene=trimesh.Scene(trimesh.creation.box());raw=scene.save_image(resolution=(128,128),visible=False)
im=Image.open(io.BytesIO(raw));im.verify()
print('Native imports + headless Trimesh render PASS',flush=True)
"""
subprocess.run([PY,'-u','-c',probe],env=ENV,cwd=REPO,check=True)
'''
launch=r'''
OUT=Path(CFG['output']);manifest=json.loads((OUT/'manifest.json').read_text())
def export_partial():
    import zipfile
    archive=WORK/(CFG['experiment']+'_partial.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for path in OUT.rglob('*'):
            if path.is_file() and path.suffix in ('.json','.jsonl','.png') and 'trace_' not in path.as_posix():z.write(path,path.relative_to(OUT))
    from IPython.display import FileLink,display
    display(FileLink(str(archive)))

def launch(smoke=False):
    folder=OUT/('smoke' if smoke else 'full');folder.mkdir(exist_ok=True)
    state_path=folder/'state.json';state_path.unlink(missing_ok=True)
    command=[PY,'-u',str(SCRIPTS/'apc_experiment.py'),'--manifest',str(OUT/'manifest.json')]
    if smoke:command+=['--smoke']
    before=time.monotonic()
    with (folder/'worker.log').open('a',encoding='utf-8') as log:
        child=subprocess.Popen(command,env=ENV,cwd=REPO,stdout=log,stderr=subprocess.STDOUT)
        failure=None
        try:
            while child.poll() is None:
                try:s=json.loads(state_path.read_text()) if state_path.exists() else {}
                except (ValueError,OSError):s={}
                elapsed=time.monotonic()-before
                rate=s.get('new_processed',0)/max(1,s.get('elapsed_seconds',1))
                remaining=max(0,s.get('total',0)-s.get('processed',0))
                print(('SMOKE' if smoke else 'FULL'),s.get('stage','starting'),s.get('mode',''),s.get('sample_id',''),
                    str(s.get('processed',0))+'/'+str(s.get('total','?')),'errors=',s.get('errors',0),'invalid=',s.get('invalid',0),
                    'elapsed min=',round(elapsed/60,1),'ETA min=',round(remaining/rate/60,1) if rate else '?',flush=True)
                start=s.get('sample_started_unix')
                if start and time.time()-start>CFG['sample_timeout_seconds']+120:
                    failure='Sample watchdog timeout';break
                if not s.get('new_processed') and elapsed>CFG['load_timeout_seconds']+CFG['sample_timeout_seconds']:
                    failure='Pipeline load watchdog timeout';break
                time.sleep(10)
            if failure:raise RuntimeError(failure)
            if child.returncode:
                print('\n'.join((folder/'worker.log').read_text(errors='replace').splitlines()[-70:]))
                raise RuntimeError('APC worker failed; inspect log/state. Results already written are retained.')
        except BaseException:
            if child.poll() is None:child.terminate()
            export_partial();raise
        finally:
            if child.poll() is None:child.terminate()
            try:child.wait(timeout=15)
            except subprocess.TimeoutExpired:child.kill();child.wait()
    print('Finished:',folder)

launch(smoke=True)
'''
for sanity,filename in [(True,'apc_qwen25vl_3b_sanity_kaggle.ipynb'),(False,'apc_qwen25vl_3b_3dsrbench_full_kaggle.ipynb')]:
    cfg=dict(config,sanity=sanity)
    if sanity:cfg.update(sanity_image=None,sanity_question="If I stand at the person's position facing where it is facing, is the table on the left or on the right of me?",sanity_choices={'A':'left','B':'right'})
    title='sanity một ảnh' if sanity else 'ba subtask 3DSRBench của paper'
    intro=f'''# APC-Vis + Qwen2.5-VL-3B — {title}, Kaggle T4 ×2

Đây là APC với Qwen/Qwen2.5-VL-3B-Instruct gốc, KHÔNG phải SenseNova Qwen. Paper dùng 7B, đánh giá left/right, visibility đã viết lại từ front/behind, và facing; dùng exact match + LLM judge. Repo chưa có APC-Num. Notebook này chạy APC-Vis và baseline 3B trực tiếp, chỉ trên ba nhóm left/right, visibility và facing. Visibility được chuyển từ câu hỏi front/behind, lựa chọn in front of→yes và behind→no, giữ nguyên chữ cái đáp án và mọi hoán vị. Dùng parser exact-choice và CircularEval. Không gọi đây là tái lập Table 1.

Hai GPU dành cho một pipeline: Qwen trên GPU0, GroundingDINO/SAM/DepthPro/Orient-Anything trên GPU1. Không nhân đôi pipeline để chia dữ liệu như notebook SenseNova. Giữ nguyên toàn bộ model phụ. Có build native CUDA và Xvfb/Mesa software rendering; sanity phải qua cả import, rendering và APC end-to-end trước khi chạy dài.

Bật Internet, T4 ×2 và Secret HF_token. Cài đặt lần đầu có thể lâu vì build GroundingDINO/Detectron2/PyTorch3D. Không bảo đảm runtime Kaggle nào cũng có nvcc/CUDA12 phù hợp; notebook kiểm tra và dừng sớm ở dependency thay vì chạy dataset với pipeline hỏng. Có thể cần chuẩn bị asset bundle làm Kaggle Input nếu thiếu dung lượng; ASSET_ROOT mặc định ở /kaggle/temp, không được xuất vào Output.

{'Sanity dùng đúng một ảnh demo có sẵn của repo, không tải TSV. Đáp án chuẩn của demo không được coi là ground truth benchmark; technical pass chỉ kiểm tra đường chạy và parser.' if sanity else 'Mặc định scope=paper_tasks_visibility và limit=None: 2.078 dòng / 1.039 nhóm, gồm left/right 698/349, visibility 688/344 và facing 692/346. Không inference chín subtype còn lại. source_question/source_choices được giữ để audit phép chuyển visibility. Cách viết lại là bản triển khai xác định theo mô tả paper, không khẳng định giống từng chữ bộ evaluator của tác giả.'}

Nguồn: https://arxiv.org/abs/2504.17207 ; https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct ; https://huggingface.co/datasets/ccvl/3DSRBench . Source APC được nhúng từ repo local, hash snapshot/runtime được lưu. Các dependency Git được clone HTTPS và khóa commit ở source_lock.json; không chạy script SSH gốc.

Các khác biệt được ghi vào patches.json: đúng device GroundingDINO; lazy import renderer debug Orient; giữ đúng cặp position/orientation/color sau lọc; z_trans=0 khi không cần dịch camera; chống chia 0; vector hóa phép unproject; cache depth và SAM embedding chỉ trong một mẫu. Detection box threshold đặt 0.15 theo phụ lục paper; text threshold giữ 0.05 theo YAML vì paper không xác định riêng tham số này. Cấu hình pixel/token và bản 3B là thay đổi thực nghiệm.

Chưa được chạy trên GPU Kaggle tại thời điểm tạo; các kiểm tra offline không thay thế xác nhận GPU/native build.'''
    cells=[md(intro)]
    init='''from pathlib import Path
import os,sys,json,subprocess,time,hashlib,shutil,base64,io,zipfile
WORK=Path('/kaggle/working/apc_qwen3b');WORK.mkdir(parents=True,exist_ok=True)
from kaggle_secrets import UserSecretsClient
try:token=UserSecretsClient().get_secret('HF_token')
except Exception:raise RuntimeError('Add Kaggle Secret HF_token and enable access.') from None
if not token:raise RuntimeError('HF_token is empty')
os.environ['HF_TOKEN']=token.strip();del token
os.environ['TOKENIZERS_PARALLELISM']='false';os.environ['HF_HUB_DISABLE_TELEMETRY']='1';os.environ['HF_HUB_DISABLE_XET']='1'
SNAPSHOT_SHA='''+repr(snapshot_sha)+'''
REPO=WORK/('repo_'+SNAPSHOT_SHA[:12]);REPO.mkdir(exist_ok=True)
raw=base64.b64decode('''+repr(b64)+''')
assert hashlib.sha256(raw).hexdigest()==SNAPSHOT_SHA
with zipfile.ZipFile(io.BytesIO(raw)) as z:
    for name in z.namelist():
        target=(REPO/name).resolve();target.relative_to(REPO.resolve())
    z.extractall(REPO)
SCRIPTS=WORK/'scripts';SCRIPTS.mkdir(exist_ok=True)
'''
    init+='\n'.join('(SCRIPTS/'+repr(name)+').write_text('+repr(text)+',encoding="utf-8")' for name,text in script_map.items())+'\n'
    cells+=[code(init),md('## Cấu hình\n\nasset_input trỏ đến root asset bundle chuẩn bị trước (assets_manifest.json + các model/checkpoint). Để None để tải. dataset_input chỉ dùng ở full, là đường dẫn TSV trên Kaggle Input. Không đổi config khi resume; đổi experiment khi đổi mode, dtype, scope, pixel/token hoặc code. Giữ modes=[direct,apc_vis] để so sánh paired cùng backbone.'),code('CFG='+pprint.pformat(cfg,sort_dicts=False)+'''
assert CFG['modes'] and len(set(CFG['modes']))==len(CFG['modes']) and set(CFG['modes'])<={'direct','apc_vis'}
assert CFG['dtype'] in ('float16','bfloat16','float32')
assert CFG['scope']=='paper_tasks_visibility', 'This notebook evaluates only the three paper task families.'
assert CFG['limit'] is None or isinstance(CFG['limit'],int) and CFG['limit']>0
assert Path(CFG['experiment']).name==CFG['experiment']
CFG.update(work=str(WORK),repo=str(REPO),output=str(WORK/('sanity/'+CFG['runtime_sha'][:12] if CFG['sanity'] else 'results/'+CFG['experiment'])))
CONFIG=WORK/('sanity_config.json' if CFG['sanity'] else 'full_config.json')
CONFIG.write_text(json.dumps(CFG,indent=2),encoding='utf-8');print(json.dumps(CFG,indent=2))
'''),md('## Python/CUDA tách khỏi kernel\n\nDùng uv tránh ensurepip Python3.13; Python3.10, Torch2.4.1/cu124 và torchvision0.19.1; giữ đường dẫn driver NVIDIA, kiểm tra phép tính trên cả hai GPU.'),code(bootstrap+'''
CFG['uv']=str(UV);CONFIG.write_text(json.dumps(CFG,indent=2),encoding='utf-8')
ENV['PYTHONPATH']=str(SCRIPTS)+os.pathsep+str(REPO)
ENV['MPLBACKEND']='Agg'
ENV['CUDA_LAUNCH_BLOCKING']='1' if CFG['sanity'] else '0'
'''),md('## Dependency native, weights và renderer\n\nKhông dùng nguyên requirements của Orient-Anything vì nó hạ Transformers xuống 4.38. Các package native build trong venv với Torch đã cài, MAX_JOBS=2. Dung lượng model phụ đáng kể; nếu không đủ chỗ, chuẩn bị asset bundle vào Input. Cell này không chạy inference dataset.'),code(native_cell),md('## Chuẩn bị dữ liệu\n\nKhông đưa answer vào prompt. Giữ payload ảnh gốc/flip, khử trùng lặp theo SHA; CircularEval theo qid suffix. Chỉ chọn ba subtype của paper; chuyển visibility trước khi chạy cả hai mode, không đưa answer vào prompt. Giữ source_question/source_choices và question_transform để audit.'),code('DATA_ENV=ENV.copy();DATA_ENV.pop("HF_HUB_OFFLINE",None);DATA_ENV.pop("TRANSFORMERS_OFFLINE",None)\nsubprocess.run([PY,str(SCRIPTS/"apc_prepare_data.py"),str(CONFIG)],env=DATA_ENV,cwd=REPO,check=True)\nOUT=Path(CFG["output"])\nprint(json.loads((OUT/"manifest.json").read_text())["selected_rows"],"selected rows")\n'),md('## Smoke toàn bộ pipeline trên một ảnh\n\nOutput invalid hay lỗi pipeline làm sanity/smoke thất bại. Ở full: output rỗng/! giữ nguyên và tính sai; lỗi parse/detection/geometry theo mẫu được ghi và tiếp tục, không fallback sang baseline để giả làm APC. CUDA/VRAM lỗi thực sự dừng để bảo toàn worker. Timeout từng mẫu và state/log theo stage.'),code(launch)]
    if not sanity:cells+=[md('## Chạy đủ ba subtask của paper\n\nMột lượt greedy, cả hai mode trên cùng 2.078 dòng, không chạy toàn bộ 12 subtype. APC gọi VLM nhiều lần và chạy vision/renderer nên có thể rất lâu; xem ETA đo từ smoke/full. Không có thời gian dự đoán chắc chắn trước khi chạy. Các file full/predictions_direct.jsonl và predictions_apc_vis.jsonl được flush sau từng mẫu. Chạy lại cell này để resume trong cùng session; restore toàn bộ thư mục work/output ở session mới và giữ fingerprint.'),code('launch(smoke=False)')]
    else:cells+=[md('## Xem ảnh và kết quả sanity'),code('''from IPython.display import display,Image as DisplayImage
rows=[json.loads(x) for x in (OUT/'samples.jsonl').read_text().splitlines()]
display(DisplayImage(filename=rows[0]['image_path']))
for mode in CFG['modes']:
    path=OUT/'smoke'/f'predictions_{mode}.jsonl'
    last=json.loads(path.read_text().splitlines()[-1])
    print(mode,json.dumps(last,ensure_ascii=False,indent=2))
''')]
    cells+=[md('## Chấm điểm và tải báo cáo\n\nFull score giữ mẫu thiếu/lỗi trong mẫu số. Sanity chỉ xác minh kỹ thuật, không chấm accuracy của demo. Output ZIP không chứa token/weight/venv/ảnh dataset; smoke traces có thể có ảnh của chính mẫu sanity.'),code(('subprocess.run([PY,str(SCRIPTS/"apc_score.py"),str(OUT)],env=ENV,cwd=REPO,check=True)\n' if not sanity else '')+'''
import zipfile
for name in ('source_lock.json','patches.json','assets_ready.json'):
    if (WORK/name).exists():shutil.copy2(WORK/name,OUT/name)
(OUT/'environment.txt').write_text(subprocess.check_output([str(UV),'pip','freeze','--python',PY],env=ENV,text=True),encoding='utf-8')
archive=WORK/('apc_sanity_report.zip' if CFG['sanity'] else 'apc_full_results.zip')
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
    for p in OUT.rglob('*'):
        if p.is_file():z.write(p,p.relative_to(OUT))
from IPython.display import FileLink,display
display(FileLink(str(archive)))
''')]
    nb=dict(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'},'kaggle':{'isGpuEnabled':True,'isInternetEnabled':True}},nbformat=4,nbformat_minor=5)
    for i,c in enumerate(cells):
        c['id']=f'apc-{i:03d}'
        if c['cell_type']=='code':ast.parse(''.join(c['source']))
    path=HERE/filename;path.write_text(json.dumps(nb,ensure_ascii=False,indent=1),encoding='utf-8');print(path)
    if not sanity:
        alias=HERE/'apc_qwen25vl_3b_paper_tasks_kaggle.ipynb';alias.write_text(path.read_text(encoding='utf-8'),encoding='utf-8');print(alias)
# Save embedded preparation for offline fixture tests too.
(HERE/'apc_prepare_data.py').write_text(data_script,encoding='utf-8')
