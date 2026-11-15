"""Build SymPL 7B notebooks from the standalone 3B runtime without changing method prompts."""
import ast
import base64
import hashlib
import io
import json
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent

def change(text, old, new):
    assert old in text, old[:100]
    return text.replace(old,new,1)

def set_cell(nb,index,text):
    nb['cells'][index]['source']=text.splitlines(True)

adapter = '''"""Original SymPL Qwen interface with FP16 two-GPU loading and call logging."""
import json
import time
from pathlib import Path
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from sympl.vlms.vlm_qwenvl2_5 import ModelQwenVL2_5
from sympl7b_device_map import plan_device_map

class KaggleQwen(ModelQwenVL2_5):
    def __init__(self, config, device='cuda:0'):
        self.calls=[]
        super().__init__(config,device)

    def load_model(self,config,device='cuda:0'):
        placement=plan_device_map(config.pretrained_model)
        self.vlm_model,loading=Qwen2_5_VLForConditionalGeneration.from_pretrained(
            config.pretrained_model, torch_dtype=torch.float16, device_map=placement,
            attn_implementation='sdpa',
            local_files_only=True,low_cpu_mem_usage=True,output_loading_info=True)
        if any(loading.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')):
            raise RuntimeError('Checkpoint loading mismatch: '+str(loading))
        self.vlm_model.eval()
        self.device=str(self.vlm_model.get_input_embeddings().weight.device)
        placement=self.vlm_model.hf_device_map
        print('Qwen 7B device map:',placement,flush=True)
        if any(str(device) in ('cpu','disk') for device in placement.values()):
            raise RuntimeError('7B does not fit the configured GPU memory budget.')
        self.processor=AutoProcessor.from_pretrained(config.pretrained_model,local_files_only=True)

    def process_messages(self,messages,max_new_tokens=1024,do_sample=False,temperature=0.0):
        started=time.perf_counter()
        call=dict(prompt=self.processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True),
                  max_new_tokens=max_new_tokens,do_sample=do_sample,temperature=temperature,status='started')
        self.calls.append(call)
        try:
            result=super().process_messages(messages,max_new_tokens=max_new_tokens,
                                           do_sample=do_sample,temperature=temperature)
            call.update(raw_output=result,status='complete')
            return result
        except Exception as error:
            call.update(error=repr(error),status='error')
            raise
        finally:
            call['seconds']=time.perf_counter()-started
'''

for mode in ('sanity','full'):
    notebook=json.loads((HERE/f'sympl_3b_{mode}_kaggle.ipynb').read_text(encoding='utf8'))
    cell=''.join(notebook['cells'][2]['source'])
    tree=ast.parse(cell)
    payload_node=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='payload' for t in n.targets))
    old_payload=ast.literal_eval(payload_node.value.args[0])
    raw=base64.b64decode(old_payload)
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        files={name:archive.read(name) for name in archive.namelist()}
    files['scripts/experiment_qwen.py']=adapter.encode()
    files['scripts/sympl7b_device_map.py']=(HERE/'sympl7b_device_map.py').read_bytes()
    worker=files['scripts/worker.py'].decode()
    # Keep the repository's VLM settings (512 in YAML). Its public process_messages
    # default is 1024, used by the original pipeline's calls without an explicit cap.
    worker=change(worker,'    config.vlm.update(settings)',
                  "    config.vlm.pretrained_model=cfg['weights']")
    worker=change(worker,"    return model\n\ndef parse",'''    atomic_json(out/'effective_runtime.json',dict(
        model_id=cfg['model_id'],model_sha=cfg['model_sha'],dtype='float16',
        attention='sdpa',device_map=model.vlm_model.vlm_model.hf_device_map,
        processor=model.vlm_model.processor.image_processor.to_dict(),
        config=config.to_dict(),process_messages_default_max_new_tokens=1024))
    return model

def parse''')
    # Preserve the responses in the exported predictions, including failed calls.
    files['scripts/worker.py']=worker.encode()
    native=files['scripts/native.py'].decode()
    native=change(native,
        "identity=dict(source_sha=lock[name]['sha'],torch=torch.__version__,python=sys.version.split()[0])",
        "identity=dict(source_sha=lock[name]['sha'],torch=torch.__version__,python=sys.version.split()[0],cuda_compiler=version.strip())")
    native=change(native,
        ' if marker.exists() and json.loads(marker.read_text())==identity:continue',
        ''' if marker.exists() and json.loads(marker.read_text())==identity:continue
 build_root=(src/name/'build').resolve()
 build_root.relative_to((src/name).resolve())
 if build_root.is_dir():
  for staging in build_root.glob('bdist.*'):
   target=staging.resolve()
   if staging.is_symlink() or target.parent!=build_root:raise RuntimeError('Unexpected build staging path')
   if target.is_dir():shutil.rmtree(target)''')
    files['scripts/native.py']=native.encode()
    buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as archive:
        for name,content in sorted(files.items()):
            if name.endswith('.py'):ast.parse(content.decode())
            info=zipfile.ZipInfo(name,(1980,1,1,0,0,0))
            info.compress_type=zipfile.ZIP_DEFLATED
            archive.writestr(info,content)
    new_raw=buffer.getvalue()
    new_sha=hashlib.sha256(new_raw).hexdigest()
    cell=cell.replace(old_payload,base64.b64encode(new_raw).decode())
    cell=cell.replace(hashlib.sha256(raw).hexdigest(),new_sha)
    cell=cell.replace("'Qwen/Qwen2.5-VL-3B-Instruct'","'Qwen/Qwen2.5-VL-7B-Instruct'")
    cell=cell.replace("'66285546d2b821cf421d4f5eb2576359d3770cd3'","'main'")
    cell=cell.replace("('sc203_'+METHOD)","('sc203_'+METHOD+'_7b')")
    cell=cell.replace("(METHOD+('_sanity' if SANITY else '_full'))","(METHOD+'_7b'+('_sanity' if SANITY else '_full')+'_memory_v2')")
    set_cell(notebook,2,cell)
    score=''.join(notebook['cells'][10]['source'])
    score=change(score,"{k:v for k,v in r.items() if k!='generation'}","r")
    sanity_check=score[score.index('if SANITY:'):]
    score=score[:score.index('if SANITY:')]
    score+='''
from collections import Counter
diagnostics={
    'status_counts':dict(Counter(r['status'] for r in rows)),
    'raw_output_counts':dict(Counter(str(r.get('raw_output')) for r in rows)),
    'category_parse_failures':sum(r.get('raw_output')=='fail : reasoning category detection fail' for r in rows),
    'visibility_responses':dict(Counter(r.get('raw_output') for r in rows if r.get('paper_task')=='visibility')),
}
(OUT/'diagnostics.json').write_text(json.dumps(diagnostics,indent=2),encoding='utf8')
print(json.dumps(diagnostics,indent=2))
archive=OUT.parent/(OUT.name+'_trace.zip')
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as bundle:
    for path in OUT.rglob('*'):
        if path.is_file():bundle.write(path,path.relative_to(OUT))
from IPython.display import FileLink,display
display(FileLink(str(archive)))
'''
    score+='\n'+sanity_check
    set_cell(notebook,10,score)
    note=('Chạy một ảnh 3DSRBench (ID 5502) để kiểm tra toàn bộ pipeline và parser category.'
          if mode=='sanity' else
          'Chạy trực tiếp 2.078 lượt thuộc Left/Right, Visibility và Object Facing. Không gọi sanity trước.')
    set_cell(notebook,0,f'''# SymPL · Qwen2.5-VL-7B · {mode}

Kaggle **T4 ×2**, Internet. Có thể thêm Secret **HF_token**. Chọn **Run All**.

{note}

Dùng mã SymPL và cấu hình gốc được nhúng trong notebook. DINO box/text threshold = 0.03, num_candidates = 5, VLM refinement bật. Processor giữ độ phân giải mặc định. YAML khai báo max_new_tokens=512; hàm process_messages của repo có mặc định 1024 và các bước pipeline gọi theo mặc định đó.

Giữ prompt gốc, Qwen tự chọn category, parser category và decoder màu nguyên bản để đánh giá phiên bản 7B. Trọng số FP16 chia qua hai GPU; vision dùng GPU1. FP16/SDPA là lựa chọn tương thích T4 và có thể làm kết quả khác phần cứng paper.

Lưu traces, artifacts, prompt/phản hồi thô, box/depth/orientation/layout và effective_runtime.json. predictions.jsonl giữ cả generation. diagnostics.json thống kê category failure và phân bố phản hồi Visibility.

MODEL_INPUT nhận asset bundle có checkpoint **7B** và assets_manifest.json. RESUME_INPUT phải trỏ output cùng phiên bản 7B. Full dùng Visibility đã chuyển từ Front/Behind theo phạm vi thí nghiệm trước.
''')
    set_cell(notebook,7,'''## Inference và trace
Mỗi câu lưu calls.jsonl, result.json và sample.json trong traces/<sample_id>. Ảnh, layout và mảng depth nằm trong artifacts/ theo hash.
Phản hồi gốc của bước chọn category và bước đọc màu được giữ để phân tích lỗi. Lỗi riêng từng câu được ghi rồi tiếp tục; lỗi CUDA hoặc thiếu tài nguyên dừng để chạy tiếp từ checkpoint.
''')
    for c in notebook['cells']:
        if c['cell_type']=='code':
            c.update(outputs=[],execution_count=None)
            ast.parse(''.join(c['source']))
    output=HERE/f'sympl_7b_{mode}_kaggle.ipynb'
    output.write_text(json.dumps(notebook,ensure_ascii=False,indent=1),encoding='utf8')
    print(output.name,output.stat().st_size)
