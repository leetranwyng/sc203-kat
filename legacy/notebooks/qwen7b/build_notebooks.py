"""Create standalone Kaggle notebooks from the already tested trace runners."""
import ast
import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent

def source(nb, index):
    return ''.join(nb['cells'][index]['source'])

def replace(text, before, after):
    if before not in text:
        raise ValueError('Patch anchor missing: '+before[:100])
    return text.replace(before, after, 1)

def set_source(nb, index, text):
    nb['cells'][index]['source'] = text.splitlines(True)

def code(text):
    return dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=text.splitlines(True))

def markdown(text):
    return dict(cell_type='markdown',metadata={},source=text.splitlines(True))

def save(nb, name):
    for cell in nb['cells']:
        if cell['cell_type']=='code':
            cell.update(outputs=[],execution_count=None)
            text=''.join(cell['source'])
            if text.startswith('%%writefile'):
                text=text.split('\n',1)[1]
            ast.parse(text)
    path=OUT/name
    path.write_text(json.dumps(nb,ensure_ascii=False,indent=1),encoding='utf8')
    print(path.name, path.stat().st_size)

apc=json.loads((ROOT/'notebooks/replay/APC_step_C.ipynb').read_text(encoding='utf8'))
# Extract embedded worker files without running any notebook cell.
cell=source(apc,1)
tree=ast.parse(cell)
embedded={}
for node in tree.body:
    if isinstance(node,ast.Expr) and isinstance(node.value,ast.Call):
        call=node.value
        if isinstance(call.func,ast.Attribute) and call.func.attr=='write_text' and call.args and isinstance(call.args[0],ast.Constant):
            target=ast.get_source_segment(cell,call.func.value)
            if target.startswith('SCRIPTS'):
                embedded[target]=call.args[0].value

for target,script in list(embedded.items()):
    if 'apc_qwen_adapter.py' in target:
        script=script.replace("device_map={'':device}", "device_map='auto',max_memory={0:'11GiB',1:'6GiB'}")
        script=replace(script,'self.vlm_model.eval();self.dtype=dtype',
            """self.vlm_model.eval();self.dtype=dtype
        self.device=str(self.vlm_model.get_input_embeddings().weight.device)
        from pathlib import Path
        import json
        Path('/kaggle/working/apc_qwen7b/model_device_map.json').write_text(json.dumps(self.vlm_model.hf_device_map,default=str))
        if any(str(v) in ('cpu','disk') for v in self.vlm_model.hf_device_map.values()):
            raise RuntimeError('7B requires more GPU memory than the configured budget. Device map was saved.')""")
        script=replace(script,'min_pixels=config.min_pixels,max_pixels=config.max_pixels,local_files_only=True',
            "local_files_only=True,**{k:getattr(config,k) for k in ('min_pixels','max_pixels') if getattr(config,k) is not None}")
        script=replace(script,"started=time.monotonic()", """started=time.monotonic()
        import os,json
        from pathlib import Path
        folder=Path(os.environ['APC_TRACE_DIR'])/'vlm'
        folder.mkdir(exist_ok=True)
        index=len(self.calls)+1
        captured=[]
        for message_index,message in enumerate(messages):
            for content_index,content in enumerate(message.get('content',[])):
                if isinstance(content,dict) and content.get('type')=='image':
                    value=content['image']
                    image_file=folder/f'{index:03d}_{message_index}_{content_index}.png'
                    if hasattr(value,'save'):
                        value.save(image_file)
                        captured.append(image_file.name)
        # Save before generation so a failed call still has its prompt.
""")
        script=replace(script,"length=int(batch['input_ids'].shape[-1])",
            """length=int(batch['input_ids'].shape[-1])
        call_file=folder/f'{index:03d}.json'
        call_file.write_text(json.dumps(dict(prompt=text,images=captured,status='started',prompt_tokens=length),ensure_ascii=False))""")
        # The replay cell will expand raw_output and add prompt.
        script=replace(script,"self.calls.append(call)", """call.update(images=captured,status='complete')
        call_file.write_text(json.dumps(dict(call,prompt=text),ensure_ascii=False))
        self.calls.append(call)""")
    elif 'apc_experiment.py' in target:
        script=replace(script,"self.apc.prompt_parser.predefined_prompts['choose_options']+='\\nAnswer with the option letter only.'","# Use the repository option-selection prompt.")
        script=replace(script,"if self.depth_cache is None:self.depth_cache=original_depth(image)",
            """if self.depth_cache is None:
                self.depth_cache=original_depth(image)
                trace=Path(os.environ['APC_TRACE_DIR'])
                np.save(trace/'depth.npy',self.depth_cache)
                finite=self.depth_cache[np.isfinite(self.depth_cache)]
                low,high=np.percentile(finite,[2,98]) if finite.size else (0,1)
                preview=np.clip((np.nan_to_num(self.depth_cache)-low)/max(high-low,1e-6),0,1)
                Image.fromarray((preview*255).astype('uint8')).save(trace/'depth_preview.png')
                (trace/'depth_metadata.json').write_text(json.dumps(dict(shape=list(self.depth_cache.shape),preview_percentiles=[float(low),float(high)])))""")
        script=replace(script,"return result\n            setattr(self.apc,name,timed)", """trace=Path(os.environ['APC_TRACE_DIR'])
                (trace/f'{_name}.json').write_text(json.dumps(dict(result=result),default=lambda x:x.tolist() if isinstance(x,np.ndarray) else str(x),ensure_ascii=False))
                return result
            setattr(self.apc,name,timed)""")
    elif 'apc_install_native.py' in target:
        script=replace(script,
            "identity=dict(source_sha=lock[name]['sha'],torch=torch.__version__,python=sys.version.split()[0])",
            "identity=dict(source_sha=lock[name]['sha'],torch=torch.__version__,python=sys.version.split()[0],cuda_compiler=version.strip())")
        script=replace(script,
            " if marker.exists() and json.loads(marker.read_text())==identity:continue",
            """ if marker.exists() and json.loads(marker.read_text())==identity:continue
 # A failed wheel build can leave its dist-info directory in bdist.*.
 # Remove packaging staging only, retaining compiled objects in build/temp.*.
 build_root=(src/name/'build').resolve()
 build_root.relative_to((src/name).resolve())
 if build_root.is_dir():
  for staging in build_root.glob('bdist.*'):
   resolved=staging.resolve()
   if staging.is_symlink() or resolved.parent!=build_root:
    raise RuntimeError('Unexpected wheel staging path: '+str(staging))
   if resolved.is_dir():
    print('Cleaning previous wheel staging:',resolved,flush=True)
    shutil.rmtree(resolved)""")
    elif 'apc_prepare_data.py' in target:
        script=script.replace('3B direct vs APC-Vis','7B direct vs APC-Vis').replace('not exact reproduction of 7B Table 1','FP16 Kaggle runtime; exact-choice scoring')
    embedded[target]=script
# Replace the string arguments using source offsets, preserving the proven bootstrap.
replacements=[]
for node in tree.body:
    if isinstance(node,ast.Expr) and isinstance(node.value,ast.Call):
        call=node.value
        if isinstance(call.func,ast.Attribute) and call.func.attr=='write_text' and call.args and isinstance(call.args[0],ast.Constant):
            target=ast.get_source_segment(cell,call.func.value)
            if target in embedded:
                old=ast.get_source_segment(cell,call.args[0])
                replacements.append((old,repr(embedded[target])))
for old,new in replacements:
    cell=replace(cell,old,new)
cell=cell.replace('/kaggle/working/apc_qwen3b','/kaggle/working/apc_qwen7b')
set_source(apc,1,cell)
config=source(apc,3).split('\nOLD=')[0]
config=config.replace('Qwen2.5-VL-3B-Instruct','Qwen2.5-VL-7B-Instruct').replace('apc_step_c_replay','apc_qwen7b_trace')
config=config.replace("'min_pixels': 200704","'min_pixels': None").replace("'max_pixels': 1003520","'max_pixels': None")
config=config.replace("'detection_box_threshold': 0.15","'detection_box_threshold': 0.05")
config=config.replace("'max_input_tokens': 8192","'max_input_tokens': 32768")
config=config.replace("'final_max_new_tokens': 32","'final_max_new_tokens': 512")
config=config.replace("'intermediate_max_new_tokens': 512","'intermediate_max_new_tokens': 1024")
config=config.replace("'asset_root': '/kaggle/temp/apc_assets'","'asset_root': '/kaggle/temp/apc_7b_assets'")
config="RUN_FULL = True  # False runs only the one-image smoke test.\n"+config
set_source(apc,3,config)
# Reuse the CUDA 12.4 selector already used for the SymPL native extensions.
toolkit_cell=(ROOT/'notebooks/spatial_trials/cuda124_cell.py').read_text(encoding='utf8')
toolkit_cell=toolkit_cell.replace(
    'toolkit = prepare_toolkit(WORK, ENV)',
    "CUDA_WORK=Path('/kaggle/temp/apc_cuda_tools')\nCUDA_WORK.mkdir(parents=True,exist_ok=True)\ntoolkit = prepare_toolkit(CUDA_WORK, ENV)")
set_source(apc,7,toolkit_cell+'\n'+source(apc,7))
# Extend existing trace patches, dropping the previous selected replay IDs.
patchcell=source(apc,9)
node=ast.parse(patchcell).body[0]
patches=ast.literal_eval(node.value)
patches.pop('scripts/apc_prepare_data.py')
patches['apc/vision_modules/orientation.py']=[
 ('        return direction_vector',"""        record['direction_vector']=np.asarray(direction_vector).tolist()
        (folder/f'orientation_angles_{index:02d}.json').write_text(json.dumps(record,default=lambda x:np.asarray(x).tolist()))
        return direction_vector"""),
 ('        # Obtain each angle',"""        from pathlib import Path
        import json
        folder=Path(apc_args['trace_save_dir'])
        record={'category':category,'bbox':list(bbox) if bbox is not None else None,'angles':np.asarray(pred_angles).tolist()}
        index=len(list(folder.glob('orientation_input_*.png')))
        image.save(folder/f'orientation_input_{index:02d}.png')
        (folder/f'orientation_angles_{index:02d}.json').write_text(json.dumps(record,default=lambda x:np.asarray(x).tolist()))
        # Obtain each angle""")]
patchcell=replace(patchcell,ast.get_source_segment(patchcell,node.value),repr(patches))
set_source(apc,9,patchcell)
set_source(apc,11,source(apc,11).replace("path.suffix in ('.json','.jsonl','.png')","path.suffix in ('.json','.jsonl','.png','.npy','.name','.log')")+"\nlaunch(smoke=True)\n")
set_source(apc,13,"if RUN_FULL:\n    launch(smoke=False)\n")
export=source(apc,15)
start=export.index('current=')
end=export.index('import zipfile',start)
export=export[:start]+export[end:]
export=export.replace("'apc_step_c_replay.zip'","'apc_qwen7b_trace.zip'")
export=export.replace("'source_lock.json','patches.json','assets_ready.json'","'source_lock.json','patches.json','assets_ready.json','model_device_map.json'")
export=export.replace('for name in (',"""# Keep original/flip images used by every sample.
image_folder=OUT/'images'
image_folder.mkdir(exist_ok=True)
for row in (json.loads(line) for line in (OUT/'samples.jsonl').read_text().splitlines()):
    original=Path(row['image_path'])
    destination=image_folder/original.name
    if not destination.exists():shutil.copy2(original,destination)
for name in (""",1)
# One-image smoke has no full predictions to score.
export=export.replace('subprocess.run([PY,str(SCRIPTS/"apc_score.py"),str(OUT)],env=ENV,cwd=REPO,check=True)',
    'if RUN_FULL:\n    subprocess.run([PY,str(SCRIPTS/"apc_score.py"),str(OUT)],env=ENV,cwd=REPO,check=True)')
set_source(apc,15,export)
set_source(apc,0,'''# APC-Vis · Qwen2.5-VL-7B · đầy đủ trace
Kaggle T4 ×2, Internet, Secret HF_token. Chạy Run All. Smoke trên một ảnh trước, sau đó full nếu RUN_FULL=True.
Scope gồm Left/Right, Visibility theo APC và Object Facing. Chạy cả baseline 7B và APC-Vis.
Giữ checkpoint 7B FP16, chia VLM qua hai GPU và đặt công cụ thị giác trên GPU1. Độ phân giải processor mặc định, detection threshold 0.05 theo config repo.
Lưu detection candidates/box chọn/mask, depth.npy và ảnh depth, input Orient Anything/góc/vector, scene trước/sau đổi góc nhìn, ảnh render và từng prompt VLM cùng ảnh đầu vào.
FP16/SDPA và exact-choice scoring là các lựa chọn chạy trên Kaggle. Kết quả có thể khác bảng paper. Nếu thiếu VRAM, lỗi được ghi rõ để điều chỉnh cấu hình.
''')
set_source(apc,12,'## Chạy toàn bộ các tác vụ APC và ghi checkpoint sau từng câu')
# Preserve provenance of modified worker files.
digest=hashlib.sha256(''.join(embedded.values()).encode()).hexdigest()
config=source(apc,3)
config=config.replace("'runtime_sha': 'f6c626016f14018728c4efa33774c58c48a48757dbff409f92222f9058b45ce6'",f"'runtime_sha': '{digest}'")
set_source(apc,3,config)
save(apc,'APC_Qwen7B_trace_kaggle.ipynb')

pcd=json.loads((ROOT/'notebooks/replay/PCD_step_C.ipynb').read_text(encoding='utf8'))
reference=source(pcd,8).split('\n',1)[1]
runner=source(pcd,9).split('\n',1)[1].replace('Qwen2.5-VL-3B-Instruct','Qwen2.5-VL-7B-Instruct')
runner=replace(runner,'rows = [r for r in all_rows if r["qid"] in config["replay_ids"]]\n        assert len(rows)==len(config["replay_ids"])','rows = all_rows')
runner=runner.replace('if torch.cuda.device_count() != 1:', 'if torch.cuda.device_count() != 2:').replace('Each worker must see exactly one GPU via CUDA_VISIBLE_DEVICES.','The single 7B worker must see both GPUs.')
runner=replace(runner,'device_map={"": 0}',"device_map='auto',max_memory={0:'11GiB',1:'11GiB'}")
runner=replace(runner,'token_ids = {label:',"""device = model.get_input_embeddings().weight.device
    write_json(root/'model_device_map.json',model.hf_device_map)
    if any(str(value) in ('cpu','disk') for value in model.hf_device_map.values()):
        raise RuntimeError('Model did not fit both GPUs. Inspect model_device_map.json.')
    token_ids = {label:""")
runner=runner.replace('CUDA_VISIBLE_DEVICES=str(rank)','CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES","0,1")')
# Resume cached object extraction also needs the raw prompt/output retained.
runner=replace(runner,'for record in completed:\n        extraction_cache',"""for record in completed:
        reference.extraction_trace[(record['category'],record['question'])] = (
            record.get('extraction_prompt',''),record.get('extraction_raw',''))
        extraction_cache""")
# Persist records of every generate and contrastive-score call before the call can fail.
runner=replace(runner,'# Extraction is text-only;',"""original_extract=reference.extract_objects_via_llm
    original_viewpoint=reference.check_same_viewpoint
    def logged_extract(question,category,*args,**kwargs):
        path=active_trace['folder']/'extraction.json'
        write_json(path,dict(question=question,category=category,status='started'))
        result=original_extract(question,category,*args,**kwargs)
        prompt,raw=reference.extraction_trace[(category,question)]
        write_json(path,dict(prompt=prompt,raw_output=raw,objects=result,status='complete'))
        return result
    def logged_viewpoint(image,ref_obj,category,*args,**kwargs):
        path=active_trace['folder']/'viewpoint.json'
        prompt=reference.build_viewpoint_check_prompt(ref_obj,category)
        write_json(path,dict(prompt=prompt,status='started'))
        raw,same=original_viewpoint(image,ref_obj,category,*args,**kwargs)
        write_json(path,dict(prompt=prompt,raw_output=raw,is_same_viewpoint=same,status='complete'))
        return raw,same
    reference.extract_objects_via_llm=logged_extract
    reference.check_same_viewpoint=logged_viewpoint
    original_generate=reference.generate_answer
    original_contrastive=reference.contrastive_pick_option
    active_trace={}
    def logged_generate(image,prompt,*args,**kwargs):
        path=active_trace['folder']/f"generate_{active_trace['index']:03d}.json"
        active_trace['index']+=1
        write_json(path,dict(prompt=prompt,status='started'))
        raw=original_generate(image,prompt,*args,**kwargs)
        write_json(path,dict(prompt=prompt,raw_output=raw,status='complete'))
        return raw
    def logged_contrastive(image,original_prompt,camera_prompt,*args,**kwargs):
        path=active_trace['folder']/f"scores_{active_trace['index']:03d}.json"
        active_trace['index']+=1
        write_json(path,dict(original_prompt=original_prompt,camera_prompt=camera_prompt,status='started'))
        prediction,scores=original_contrastive(image,original_prompt,camera_prompt,*args,**kwargs)
        write_json(path,dict(original_prompt=original_prompt,camera_prompt=camera_prompt,
                             prediction=prediction,scores=scores,status='complete'))
        return prediction,scores
    reference.generate_answer=logged_generate
    reference.contrastive_pick_option=logged_contrastive
    # Extraction is text-only;""")
runner=replace(runner,'start = time.monotonic()\n            try:',"""start = time.monotonic()
            trace=root/'trace'/str(row['row_id'])
            trace.mkdir(parents=True,exist_ok=True)
            active_trace.update(folder=trace,index=0)
            try:""")
runner=replace(runner,'torch.cuda.synchronize()\n                record["seconds"]',"""# Counterfactual diagnostic: rotate the reference frame 180 degrees.
                reverse_question=question.replace(
                    "facing where it is facing", "facing in the direction opposite to where it is facing")
                if reverse_question == question:
                    raise ValueError('Cannot construct a 180-degree diagnostic for this question.')
                reverse_prompt=reference.build_original_prompt(reverse_question,reference.options_to_text(options))
                reverse_raw=reference.generate_answer(image,reverse_prompt,model,processor,device)
                record['reverse_direction_prompt']=reverse_prompt
                record['reverse_direction_raw']=reverse_raw
                record['reverse_direction_prediction']=parse_letter(reverse_raw,labels)[0]
                record['reverse_direction_answer']=None
                record['reverse_direction_protocol']='Reference frame rotated 180 degrees; unscored counterfactual.'
                opposite_viewpoint=reference.build_viewpoint_check_prompt(ref_obj,category)
                opposite_viewpoint=opposite_viewpoint.replace("back side","front side").replace("from behind","from the front")
                record['opposite_viewpoint_prompt']=opposite_viewpoint
                record['opposite_viewpoint_raw']=reference.generate_answer(image,opposite_viewpoint,model,processor,device)
                write_json(trace/'record.json',record)
                torch.cuda.synchronize()
                record["seconds"]""")
# Always retain a failure record; CUDA errors require a clean process and a resumed run.
runner=replace(runner,'except Exception as error:\n                write_json',"""except Exception as error:
                import traceback
                write_json(trace/'error.json',dict(error=repr(error),traceback=traceback.format_exc(),row_id=row['row_id']))
                write_json""")
# Options variable name is checked against the existing runner during build below.
pcd_setup='''from pathlib import Path
import os, subprocess, sys
WORK=Path('/kaggle/working/pcd_qwen7b')
WORK.mkdir(exist_ok=True)
TOOLS=Path('/kaggle/temp/pcd7b_tools')
TOOLS.mkdir(parents=True,exist_ok=True)
subprocess.run([sys.executable,'-m','pip','install','--quiet','--target',str(TOOLS),'uv==0.9.5'],check=True)
UV=str(TOOLS/'bin/uv')
ENV=dict(os.environ,UV_PYTHON_INSTALL_DIR='/kaggle/temp/pcd7b_python',UV_CACHE_DIR='/kaggle/temp/pcd7b_uv',
         HF_HOME='/kaggle/temp/pcd7b_hf',HF_HUB_DISABLE_XET='1',TOKENIZERS_PARALLELISM='false')
from kaggle_secrets import UserSecretsClient
try:
    ENV['HF_TOKEN']=UserSecretsClient().get_secret('HF_token')
except Exception:
    print('HF_token unavailable; downloading public checkpoint without authentication.')
VENV=Path('/kaggle/temp/pcd7b_venv')
PY=str(VENV/'bin/python')
if not Path(PY).exists():
    subprocess.run([UV,'venv','--clear','--python','3.11','--managed-python',str(VENV)],env=ENV,check=True)
subprocess.run([PY,'-c','import sys; assert sys.version_info[:2]==(3,11)'],env=ENV,check=True)
subprocess.run([UV,'pip','install','--python',PY,'torch==2.5.1','torchvision==0.20.1',
                '--index-url','https://download.pytorch.org/whl/cu124'],env=ENV,check=True)
subprocess.run([UV,'pip','install','--python',PY,'transformers==4.51.3','accelerate==1.6.0',
    'huggingface_hub==0.30.2','tokenizers==0.21.1','safetensors>=0.4.3,<1',
    'pandas>=2.2,<3','Pillow>=10,<12','tqdm>=4.66,<5','openpyxl>=3.1,<4'],env=ENV,check=True)
'''
config=source(pcd,4).split('\nOLD=')[0]
config=config.replace('MODE = "full"','MODE = "sanity"  # Change to "full" after the one-image test succeeds.')
config=config.replace('Qwen2.5-VL-3B-Instruct','Qwen2.5-VL-7B-Instruct')
config=config.replace('66285546d2b821cf421d4f5eb2576359d3770cd3','main')
config=config.replace('NUM_WORKERS = 2','NUM_WORKERS = 1').replace('NUM_WORKERS == 2','NUM_WORKERS == 1')
config=config.replace('OUTPUT_DIR = Path("/kaggle/working/pcd_step_c_replay")','OUTPUT_DIR = Path("/kaggle/working/pcd_qwen7b") / MODE')
config=config.replace('qwen25vl3b','qwen25vl7b')
prepare=source(pcd,6).replace('2048','3584').replace('36','28').replace('Sai model 3B.','Expected official Qwen2.5-VL-7B.')
prepare=prepare.replace('Workers use FP16 + SDPA; each worker loads a full 3B model on its own GPU.','One FP16 7B model spans two GPUs, with SDPA.')
prepare=replace(prepare,'from huggingface_hub import hf_hub_download, snapshot_download',
    'from huggingface_hub import hf_hub_download, snapshot_download, HfApi\nMODEL_REVISION=HfApi().model_info(MODEL_ID,revision=MODEL_REVISION).sha')
plan=source(pcd,11).replace('CONFIG["replay_ids"] = sorted(OLD)\n','')
plan=plan.replace('/kaggle/working/pcd_kaggle_runner.py','/kaggle/working/pcd_qwen7b/pcd_kaggle_runner.py').replace('/kaggle/working/pcd_reference.py','/kaggle/working/pcd_qwen7b/pcd_reference.py')
driver=prepare+'\n'+plan+'''
runner.launch_workers(CONFIG)
metrics,records=runner.export_results(CONFIG)
print(metrics.to_string(index=False))
if MODE=='sanity':
    for row in records:
        if row.get('diagnostic_camera_prediction') is None or row.get('diagnostic_contrastive_prediction') is None:
            raise RuntimeError('Sanity failed: a branch has no valid option.')
        if row.get('reverse_direction_prediction') is None:
            raise RuntimeError('Sanity failed: reversed-direction output is invalid.')
print('Done. Change MODE to full and rerun configuration/run cells to evaluate all rows.')
'''
# Keep exact driver inputs, raw predictions, trace, images and modules in the output archive.
driver+='''
import zipfile
archive=OUTPUT_DIR.parent/('pcd_qwen7b_'+MODE+'_trace.zip')
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as bundle:
    for path in OUTPUT_DIR.rglob('*'):
        if path.is_file():bundle.write(path,path.relative_to(OUTPUT_DIR))
    bundle.write(reference_path,'pcd_reference.py')
    bundle.write(runner_path,'pcd_kaggle_runner.py')
print('Download:',archive)
'''
nb=dict(nbformat=4,nbformat_minor=5,metadata=copy.deepcopy(pcd['metadata']),cells=[
markdown('''# PCD · Qwen2.5-VL-7B · hai nhánh và prompt đảo hướng
Kaggle T4 ×2, Internet, Secret HF_token (tùy chọn). Chạy Run All ở MODE="sanity", sau đó đổi MODE="full" và chạy lại hai cell cuối.
Một tiến trình chia checkpoint FP16 qua hai GPU. Alpha=4, processor mặc định, prompt và công thức chấm option từ repo PCD.
Full gồm Left/Right và Front/Behind, toàn bộ hoán vị đáp án và ảnh flip.
Mỗi câu lưu extraction, viewpoint, câu hỏi gốc, camera prompt, phản hồi cả hai nhánh, score orig/cam/contrastive, baseline, và ảnh.
Lưu cả prompt kiểm tra mặt trước đối lập với mặt sau, và prompt quay khung tham chiếu 180°. Kết quả này được ghi riêng và không chấm bằng đáp án gốc.
Checkpoint JSONL được flush từng câu. Chạy lại cell cuối để tiếp tục khi bị ngắt. Lỗi CUDA dừng tiến trình để bảo vệ các kết quả đã lưu.
'''),code(pcd_setup),
markdown('## Ghi mã nguồn tự chứa'),
code("(WORK/'pcd_reference.py').write_text("+repr(reference)+",encoding='utf8')\n(WORK/'pcd_kaggle_runner.py').write_text("+repr(runner)+",encoding='utf8')\n"),
markdown('## Chọn sanity hoặc full'),
code(config),
markdown('## Chạy trong Python 3.11 riêng và xuất trace'),
code("configuration="+repr(config)+"\n"+
     "driver="+repr(driver)+"\n"+
     # The current notebook settings, not the initial text, define a rerun.
     "values={k:globals()[k] for k in ['MODE','MODEL_ID','MODEL_REVISION','DATASET_ID','DATASET_REVISION','TSV_FILENAME','ALPHA','NUM_WORKERS','SEED','COMPUTE_BASELINE','MIN_PIXELS','MAX_PIXELS','MODEL_DIR','TSV_PATH','SANITY_QID','OUTPUT_DIR','RESUME_FROM']}\n"+
     "prefix='from pathlib import Path\\n'+''.join(k+' = '+('Path('+repr(str(v))+')' if isinstance(v,Path) else repr(v))+'\\n' for k,v in values.items())\n"+
     "job=WORK/'run_pcd.py'\njob.write_text(prefix+driver,encoding='utf8')\n"+
     "subprocess.run([PY,'-u',str(job)],env=ENV,check=True)\n")
])
# Check the Python in files embedded inside cells too.
for text in (reference,runner,driver):ast.parse(text)
save(nb,'PCD_Qwen7B_trace_kaggle.ipynb')


# Publish separate entry points so full runs never invoke the smoke experiment.
for full in (False, True):
    mode = 'full' if full else 'sanity'
    notebook = copy.deepcopy(apc)
    config = source(notebook, 3)
    config = config[config.index('CFG='):]
    config = config.replace("'experiment': 'apc_qwen7b_trace'",
                            f"'experiment': 'apc_qwen7b_{mode}_trace'")
    set_source(notebook, 3, config)
    launcher = source(notebook, 11).replace('launch(smoke=True)', '')
    set_source(notebook, 11, launcher)
    set_source(notebook, 10, '## Chuẩn bị tiến trình và ghi log')
    set_source(notebook, 12, '## Chạy toàn bộ dữ liệu' if full else '## Sanity trên một ảnh')
    set_source(notebook, 13, 'launch(smoke=False)\n' if full else 'launch(smoke=True)\n')
    export_code = source(notebook, 15)
    score = '    subprocess.run([PY,str(SCRIPTS/"apc_score.py"),str(OUT)],env=ENV,cwd=REPO,check=True)'
    export_code = replace(export_code, 'if RUN_FULL:\n'+score,
                          score.strip() if full else '# Smoke results are saved under smoke/.')
    if not full:
        export_code = export_code.replace(
            "for row in (json.loads(line) for line in (OUT/'samples.jsonl').read_text().splitlines()):",
            "for row in [json.loads((OUT/'samples.jsonl').read_text().splitlines()[0])]:")
    export_code = replace(export_code,
        "archive=WORK/('apc_sanity_report.zip' if CFG['sanity'] else 'apc_qwen7b_trace.zip')",
        f"archive=WORK/'apc_qwen7b_{mode}_trace.zip'")
    set_source(notebook, 15, export_code)
    description = (
        'Run All chạy thẳng baseline 7B và APC-Vis trên toàn bộ Left/Right, Visibility và Object Facing. '
        'Không chạy smoke inference. Môi trường và model vẫn được khởi tạo cho phiên Kaggle.'
        if full else
        'Run All khởi tạo toàn bộ pipeline rồi chạy baseline 7B và APC-Vis trên một ảnh đầu tiên của tập tác vụ. '
        'Kết quả kiểm tra và trace được lưu trong smoke/.')
    set_source(notebook, 0,
        f'# APC · Qwen2.5-VL-7B · {mode}\n'
        'Kaggle T4 ×2, Internet và Secret HF_token.\n\n'+description+'\n\n'
        'Lưu prompt/ảnh từng lần gọi VLM, detection candidates, box chọn, segmentation mask, '
        'depth.npy và ảnh depth, góc/vector hướng, scene và ảnh render. '
        'Checkpoint FP16 chia qua hai GPU, processor mặc định. '
        'FP16/SDPA và exact-choice scoring được ghi trong cấu hình.\n')
    save(notebook, f'APC_Qwen7B_{mode}_trace_kaggle.ipynb')

    notebook = copy.deepcopy(nb)
    config = source(notebook, 5)
    config = replace(config,
        'MODE = "sanity"  # Change to "full" after the one-image test succeeds.',
        f'MODE = "{mode}"')
    set_source(notebook, 5, config)
    set_source(notebook, 4, '## Cấu hình lượt chạy')
    run_code = source(notebook, 7)
    # Remove the unused initial configuration copy, retaining current cell values.
    first = ast.parse(run_code).body[0]
    run_code = run_code[len(ast.get_source_segment(run_code, first)):].lstrip()
    run_code = run_code.replace(
        'Done. Change MODE to full and rerun configuration/run cells to evaluate all rows.',
        'Done. Results and trace are ready for export.')
    set_source(notebook, 7, run_code)
    description = (
        'Run All chạy thẳng toàn bộ Left/Right và Front/Behind, gồm các hoán vị đáp án và ảnh flip. '
        'Không chạy sanity inference trước.'
        if full else
        'Run All chạy một ảnh với cả hai nhánh, baseline và các prompt đảo hướng. '
        'Kiểm tra hai nhánh trả về đáp án hợp lệ trước khi dùng notebook full.')
    set_source(notebook, 0,
        f'# PCD · Qwen2.5-VL-7B · {mode}\n'
        'Kaggle T4 ×2, Internet. Secret HF_token tùy chọn cho checkpoint công khai.\n\n'
        +description+'\n\n'
        'Một tiến trình chia checkpoint FP16 qua hai GPU. Lưu prompt gốc/camera, '
        'extraction, kiểm tra góc nhìn, phản hồi cả hai nhánh, score và ảnh. '
        'Lưu thêm prompt mặt trước/mặt sau và quay hướng nhìn 180°. '
        'Prompt quay 180° được ghi riêng, không chấm bằng đáp án gốc. '
        'Chạy lại cell cuối để tiếp tục checkpoint khi phiên chạy bị ngắt.\n')
    save(notebook, f'PCD_Qwen7B_{mode}_trace_kaggle.ipynb')
