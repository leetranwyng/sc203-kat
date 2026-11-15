"""Produce Spatial-SSRL-7B Kaggle entry points with a single sharded worker."""
import ast
import base64
import hashlib
import io
import json
import zipfile
from pathlib import Path

HERE=Path(__file__).resolve().parent

def replace(text,old,new):
    assert old in text,old[:100]
    return text.replace(old,new,1)

def set_cell(nb,index,text):
    nb['cells'][index]['source']=text.splitlines(True)

for mode in ('sanity','full'):
    nb=json.loads((HERE/f'ssrl_3b_{mode}_kaggle.ipynb').read_text(encoding='utf8'))
    settings=''.join(nb['cells'][1]['source'])
    settings=replace(settings,"SCOPE='all_original'",
                     "SCOPE='paper_tasks_original'  # Or 'paper_tasks_visibility' to match APC/SymPL.")
    set_cell(nb,1,settings)
    cell=''.join(nb['cells'][2]['source'])
    tree=ast.parse(cell)
    node=next(x for x in tree.body if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='payload' for t in x.targets))
    old_payload=ast.literal_eval(node.value.args[0])
    raw=base64.b64decode(old_payload)
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        files={name:archive.read(name) for name in archive.namelist()}
    planner=(HERE/'sympl7b_device_map.py').read_text(encoding='utf-8-sig')
    planner=replace(planner,"1: min(6.5*gib, free[1]-7*gib)","1: min(10.5*gib, free[1]-3*gib)")
    planner=planner.replace('# GPU0 needs generation/vision-encoder activations. GPU1 also hosts the\n    # external DINO, Depth Pro and Orient Anything models and their activations.',
        '# Reserve activation and KV-cache memory on both GPUs.')
    files['scripts/ssrl7b_device_map.py']=planner.encode()
    adapter=files['scripts/experiment_qwen.py'].decode()
    adapter=replace(adapter,'from qwen_vl_utils import process_vision_info',
        'from qwen_vl_utils import process_vision_info\nfrom ssrl7b_device_map import plan_device_map')
    adapter=replace(adapter,"dtype=getattr(torch,config.dtype)","dtype=getattr(torch,config.dtype)\n        placement=plan_device_map(config.pretrained_model)")
    adapter=replace(adapter,"device_map={'':device}","device_map=placement")
    adapter=replace(adapter,'self.vlm_model.eval();self.dtype=dtype',
        "self.vlm_model.eval();self.dtype=dtype\n        self.device=str(self.vlm_model.get_input_embeddings().weight.device)")
    adapter=replace(adapter,'min_pixels=config.min_pixels,max_pixels=config.max_pixels,local_files_only=True',
        "local_files_only=True,**{k:getattr(config,k) for k in ('min_pixels','max_pixels') if getattr(config,k) is not None}")
    adapter=replace(adapter,'self.calls.append(call)',"""call['hit_token_limit']=int(new.shape[-1])>=cap
        call['max_new_tokens']=cap
        self.calls.append(call)""")
    files['scripts/experiment_qwen.py']=adapter.encode()
    worker=files['scripts/worker.py'].decode()
    worker=worker.replace("min_pixels=256*28**2,max_pixels=1280*28**2","min_pixels=None,max_pixels=None")
    worker=worker.replace('max_input_tokens=12288','max_input_tokens=32768')
    worker=worker.replace('Spatial-SSRL-3B','Spatial-SSRL-7B')
    worker=replace(worker,"        return vlm","""        atomic_json(out/'effective_runtime.json',dict(
            model_id=cfg['model_id'],model_sha=cfg['model_sha'],dtype='float16',
            attention='sdpa',device_map=vlm.vlm_model.hf_device_map,
            processor=vlm.processor.image_processor.to_dict(),max_new_tokens=cfg['max_tokens']))
        return vlm""")
    files['scripts/worker.py']=worker.encode()
    data=files['scripts/data.py'].decode()
    data=replace(data,"identity={k:cfg[k]", """if not cfg['sanity']:
    from collections import Counter
    expected={'orientation_on_the_left':698,'orientation_in_front_of':688,'multi_object_facing':692}
    if Counter(r['category'] for r in items)!=expected:
        raise ValueError('Expected 2,078 rows on the same three 3DSRBench categories as VST.')
    groups=Counter((r['category'],r['circular_group']) for r in items)
    if len(groups)!=1039 or set(groups.values())!={2}:
        raise ValueError('Expected 1,039 complete A/B permutation groups.')
    for row in items:
        if 'paper_task' not in row:
            row['paper_task']={'orientation_on_the_left':'left_right',
                               'orientation_in_front_of':'front_behind',
                               'multi_object_facing':'facing'}[row['category']]

identity={k:cfg[k]""")
    files['scripts/data.py']=data.encode()
    # The same embedded single-image example is retained for sanity.
    buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as archive:
        for name,content in sorted(files.items()):
            if name.endswith('.py'):ast.parse(content.decode('utf-8-sig'))
            info=zipfile.ZipInfo(name,(1980,1,1,0,0,0))
            info.compress_type=zipfile.ZIP_DEFLATED
            archive.writestr(info,content)
    payload=buffer.getvalue()
    cell=cell.replace(old_payload,base64.b64encode(payload).decode())
    cell=cell.replace(hashlib.sha256(raw).hexdigest(),hashlib.sha256(payload).hexdigest())
    cell=cell.replace('internlm/Spatial-SSRL-3B','internlm/Spatial-SSRL-7B')
    cell=cell.replace("'66285546d2b821cf421d4f5eb2576359d3770cd3'","'main'")
    cell=cell.replace("('sc203_'+METHOD)","('sc203_'+METHOD+'_7b')")
    cell=replace(cell,"(METHOD+('_sanity' if SANITY else '_full'))",
                 "(METHOD+'_7b'+('_sanity' if SANITY else '_full_'+SCOPE))")
    set_cell(nb,2,cell)
    launch=''.join(nb['cells'][8]['source'])
    launch=replace(launch,"workers=2 if METHOD=='ssrl' and not SANITY else 1","workers=1  # One model spans both GPUs.")
    launch=replace(launch,"        if METHOD=='ssrl':env['CUDA_VISIBLE_DEVICES']=GPU_IDS[rank]",
                   "        env['CUDA_VISIBLE_DEVICES']=','.join(GPU_IDS)")
    set_cell(nb,8,launch)
    score=''.join(nb['cells'][10]['source'])
    score=replace(score,"{k:v for k,v in r.items() if k!='generation'}","r")
    score=replace(score,"(OUT/'summary.json').write_text",'''if not SANITY:
    result['scope']=SCOPE
    result['by_task']={task:metrics([r for r in rows if r['paper_task']==task])
                       for task in sorted({r['paper_task'] for r in rows})}
(OUT/'summary.json').write_text''')
    sanity_check=score[score.index('if SANITY:'):]
    score=score[:score.index('if SANITY:')]
    score+='''
diagnostics={
    'rows':len(rows),
    'invalid_or_error':sum(r['status']!='ok' for r in rows),
    'generation_limit_hits':sum(any(c.get('hit_token_limit',False) for c in r.get('generation',[])) for r in rows),
}
(OUT/'diagnostics.json').write_text(json.dumps(diagnostics,indent=2),encoding='utf8')
archive=OUT.parent/(OUT.name+'_trace.zip')
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as bundle:
    for path in OUT.rglob('*'):
        if path.is_file():bundle.write(path,path.relative_to(OUT))
from IPython.display import FileLink,display
display(FileLink(str(archive)))
'''
    score+='\n'+sanity_check
    set_cell(nb,10,score)
    run_description=('Một ảnh 3DSRBench (ID 5502) nhúng sẵn, kiểm tra prompt, generation và đọc đáp án.'
                     if mode=='sanity' else
                     'Chạy 2.078 lượt / 1.039 nhóm CircularEval: Left/Right 698, Front/Behind 688 và Object Facing 692. Giữ đủ hoán vị A/B và ảnh flip đã có trong benchmark.')
    set_cell(nb,0,f'''# Spatial-SSRL-7B · {mode}

Checkpoint chính thức **internlm/Spatial-SSRL-7B**, kiến trúc Qwen2.5-VL-7B.
Kaggle **T4 ×2**, bật Internet, Secret **HF_token** tùy chọn. Chọn **Run All**.

{run_description}

Một tiến trình chia lớp model FP16 qua hai GPU, dùng SDPA và độ phân giải processor mặc định.
Giữ format prompt <think>…</think> và đáp án trong boxed theo hướng dẫn repo, greedy decoding, tối đa 4096 token mới.
Chấm đáp án cuối sau phần reasoning. Lưu toàn bộ output để phân tích phản hồi không hợp lệ hoặc chạm giới hạn token.

Full mặc định `SCOPE="paper_tasks_original"`, giữ Front/Behind gốc như PCD và chế độ original của VST.
Chọn `SCOPE="paper_tasks_visibility"` nếu cần thay 688 lượt Front/Behind bằng Visibility yes/no như APC/SymPL và chế độ visibility của VST. Mỗi lần vẫn chạy đúng 2.078 lượt, với thư mục và điểm riêng cho từng giao thức.
MODEL_INPUT nhận thư mục checkpoint Spatial-SSRL-7B. RESUME_INPUT nhận output cùng phiên bản notebook và cùng SCOPE.
Đầu ra gồm predictions.jsonl, summary.json, diagnostics.json, effective_runtime.json, traces/, artifacts/ và ZIP.
''')
    # Limit SSRL-7B to the same three source categories in either formulation.
    s=''.join(nb['cells'][2]['source'])
    s=replace(s,"assert SCOPE in ('all_original','paper_tasks_visibility')",
                "assert SCOPE in ('paper_tasks_original','paper_tasks_visibility')")
    set_cell(nb,2,s)
    set_cell(nb,7,'''## Inference và trace
Chạy một worker trên cả hai GPU. Mỗi câu lưu prompt đầy đủ, ảnh, output thô, token, thời gian và kết quả đọc đáp án trong traces/<sample_id>.
Ảnh được lưu trong artifacts/ theo hash. Lỗi riêng từng câu được ghi và tiếp tục; lỗi CUDA/OOM dừng tiến trình để xử lý trước khi chạy tiếp.
''')
    for c in nb['cells']:
        if c['cell_type']=='code':
            c.update(execution_count=None,outputs=[])
            ast.parse(''.join(c['source']))
    target=HERE/f'ssrl_7b_{mode}_kaggle.ipynb'
    target.write_text(json.dumps(nb,ensure_ascii=False,indent=1),encoding='utf8')
    print(target.name,target.stat().st_size)
