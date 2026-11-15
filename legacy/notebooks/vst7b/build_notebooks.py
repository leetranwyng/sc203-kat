"""Emit two standalone Kaggle notebooks from one VST runner."""
import ast
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNNER = (HERE/'runner.py').read_text(encoding='utf-8-sig')

SETUP = """import json, os, re, shutil, subprocess, sys, time
from pathlib import Path

STARTED = time.time()
assert re.fullmatch(r'[A-Za-z0-9_-]+', RUN_NAME), 'RUN_NAME: letters, digits, underscore or hyphen'
assert PROTOCOL in {'original', 'visibility', 'both'}
assert MAX_IMAGE_TOKENS >= 256 and MAX_NEW_TOKENS > 0
assert SESSION_BUDGET_MINUTES > 0 and MAX_ROWS_THIS_SESSION >= 0
TEMP_ROOT = Path('/kaggle/temp/vst7b_sc203')
WORK = TEMP_ROOT / MODE
OUT = Path('/kaggle/working/vst7b_sc203') / RUN_NAME
for path in (TEMP_ROOT, WORK, OUT):
    path.mkdir(parents=True, exist_ok=True)
gpu = subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total','--format=csv,noheader'], text=True)
devices = gpu.strip().splitlines()
if len(devices) != 2 or not all('T4' in line for line in devices):
    raise RuntimeError('Select GPU T4 ×2 on Kaggle. Detected: ' + repr(devices))
print(gpu.strip())
env = dict(os.environ, PYTHONNOUSERSITE='1', UV_PYTHON_INSTALL_DIR=str(TEMP_ROOT/'python'),
           UV_CACHE_DIR=str(TEMP_ROOT/'uv_cache'), HF_HOME=str(TEMP_ROOT/'hf'),
           HF_HUB_DISABLE_XET='1', HF_HUB_DOWNLOAD_TIMEOUT='180',
           TOKENIZERS_PARALLELISM='false', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
try:
    from kaggle_secrets import UserSecretsClient
    secret = UserSecretsClient().get_secret('HF_token')
    if secret:
        env['HF_TOKEN'] = secret.strip()
except Exception:
    print('No HF_token secret. The official model and data are public.')
finally:
    if 'secret' in globals():
        del secret
tools = TEMP_ROOT/'uv_bootstrap'
uv = tools/'bin/uv'
venv = TEMP_ROOT/'venv_py311_cu124_v1'
PY = venv/'bin/python'
ready = venv/'sc203_ready'
if not uv.is_file():
    subprocess.run([sys.executable,'-m','pip','install','--no-cache-dir','--quiet',
                    '--target',str(tools),'uv==0.9.5'], check=True)
if not (ready.is_file() and PY.is_file()):
    if shutil.disk_usage(TEMP_ROOT).free < 8*2**30:
        raise OSError('Need at least 8 GiB temporary space for the isolated runtime.')
    subprocess.run([str(uv),'python','install','3.11.14'], env=env, check=True)
    if not PY.is_file():
        subprocess.run([str(uv),'venv','--clear','--managed-python','--python','3.11.14',str(venv)],
                       env=env, check=True)
    subprocess.run([str(uv),'pip','install','--no-cache','--python',str(PY),
                    '--index-url','https://download.pytorch.org/whl/cu124',
                    'torch==2.5.1+cu124','torchvision==0.20.1+cu124'], env=env, check=True)
    subprocess.run([str(uv),'pip','install','--no-cache','--python',str(PY),
                    'transformers==4.57.1','accelerate==1.11.0','huggingface_hub==0.36.0',
                    'qwen-vl-utils==0.0.14','pillow==11.3.0'], env=env, check=True)
    subprocess.run([str(PY),'-c',
                    'from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor; '
                    'from qwen_vl_utils import process_vision_info; import torch; print(torch.__version__)'],
                   env=env,check=True)
    ready.touch()
CONFIG = WORK/'config.json'
CONFIG.write_text(json.dumps(dict(mode=MODE,protocol=PROTOCOL,out=str(OUT),work=str(WORK),
    model_input=MODEL_INPUT,dataset_input=DATASET_INPUT,resume_input=RESUME_INPUT,
    thinking=THINKING,max_new_tokens=MAX_NEW_TOKENS,max_image_tokens=MAX_IMAGE_TOKENS,
    deadline=STARTED+60*SESSION_BUDGET_MINUTES,max_rows_this_session=MAX_ROWS_THIS_SESSION)),encoding='utf8')
"""

RUN = """from IPython.display import FileLink, display

# Streaming console is bounded; full per-row generations go to predictions.jsonl.
with (OUT/'worker.log').open('a', encoding='utf8', buffering=1) as log:
    process = subprocess.Popen([str(PY),'-u',str(WORK/'vst_runner.py'),str(CONFIG)],
                               env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                               encoding='utf8',errors='replace',bufsize=1)
    try:
        for line in process.stdout:
            log.write(line)
            print(line,end='',flush=True)
    except KeyboardInterrupt:
        import signal
        process.send_signal(signal.SIGINT)
        process.wait()
    rc = process.wait()

archive = OUT/'checkpoint.zip'
if archive.is_file():
    display(FileLink(os.path.relpath(archive, '/kaggle/working')))
if rc:
    print(f'Worker exited with code {rc}. Inspect worker.log and last_error.json.')
    if not archive.is_file():
        raise RuntimeError('Run failed before the first checkpoint. See worker.log.')
    print('Checkpoint preserved. This output is incomplete or failed its sanity gate.')
else:
    print('See summary.json and its complete/attempted_rows fields for coverage.')
"""

ABOUT = """Dùng checkpoint chính thức **rayruiyang/VST-7B-RL** (Qwen2.5-VL-7B đã SFT/RL). Chạy inference, không train và không chạy baseline. Paper báo 60.1 trên 3DSRBench ở Bảng 2. Notebook đo tập con với exact-choice và CircularEval, nên điểm tổng của notebook thuộc phạm vi riêng.

Kaggle: chọn **GPU T4 ×2** và bật **Internet**. Có thể thêm secret `HF_token`. Một bản FP16 được chia qua hai T4 bằng Accelerate, với SDPA. Python 3.11 và torch CUDA được cài bằng uv vào thư mục tạm. Không cần vLLM, FlashAttention 2, pipeline train, renderer hay cài extension C++.

- `MODEL_INPUT`: nếu muốn tránh tải ~16,6 GB, gắn thư mục snapshot HF có config, processor, tokenizer, index và đủ 4 shard safetensors. Notebook ghi hash snapshot cục bộ. Giữ cùng nguồn checkpoint để resume.
- `DATASET_INPUT`: nếu đã gắn dataset, đặt đường dẫn tới file `3dsrbench_v1_vlmevalkit_circular.tsv` của revision ghi trong manifest.
- `RESUME_INPUT`: khi sang phiên Kaggle mới, gắn `checkpoint.zip` từ phiên trước, rồi đặt đường dẫn ZIP hoặc thư mục chứa manifest và journal. Với cùng phiên, giữ `RUN_NAME` để tự dùng journal có sẵn. Đổi cấu hình inference thì đổi tên run.
- Kết quả trong `/kaggle/working/vst7b_sc203/<RUN_NAME>/`. Tải `checkpoint.zip` sau mỗi phiên. ZIP chỉ chứa metadata, câu hỏi, dự đoán, log và điểm. Ảnh và weights nằm ở cache tạm hoặc Kaggle Input.
"""

FULL = """`PROTOCOL='both'` mặc định chạy **2.766 lượt / 1.383 nhóm**. Left/Right 698, Object Facing 692, Front/Behind gốc 688 và Visibility chuyển yes/no 688. Hai giao thức cho họ câu Front/Behind có khóa riêng và chấm riêng. `original` hoặc `visibility` chạy 2.078 lượt nếu chỉ cần một giao thức. Tất cả đều giữ hai hoán vị lựa chọn.

Pilot chạy một cặp mỗi tác vụ trước. Nếu một tác vụ không parse được cả hai đầu ra pilot, worker dừng để xem log. Đầu ra invalid sau pilot vẫn được lưu và tính trong mẫu số. Chạy full có thể cần nhiều phiên vì VST sinh chuỗi suy luận. Giới hạn phiên mặc định 420 phút kể từ lúc setup bắt đầu. Chỉnh `MAX_ROWS_THIS_SESSION` để chia phiên ngắn hơn mà không đổi fingerprint. Nhìn tốc độ sau 25–100 lượt trước khi dự trù thời gian toàn bộ.

Nếu một phiên dừng, `summary.json` ghi số lượt đã thử, thiếu và hoàn tất; điểm trên mẫu số cố định và điểm tạm trên phần đã chạy được tách riêng. `overall_selected_protocols` gộp các lượt đã chọn, gồm hai dạng câu cùng ảnh khi dùng `both`; dùng điểm theo task khi đối chiếu với các model trước.
"""

SANITY = """Sanity dùng **một ảnh, hai hoán vị** của câu máy bay–xe đạp (`5998/5999`). Đạt kiểm tra kỹ thuật khi cả hai GPU có weight, chạy đủ hai lượt và parse được A/B cho cả hai. Đáp án sai vẫn là kết quả hợp lệ để kiểm tra kỹ thuật. Kiểm tra `runtime.json`, `summary.json` và raw output trước khi chạy notebook full.
"""


def config(mode):
    return f"""MODE = {mode!r}
RUN_NAME = 'vst7b_{mode}_v1'
PROTOCOL = 'both'  # Full: 'both', 'original' or 'visibility'. Sanity uses Left/Right.
MODEL_INPUT = None    # Path to attached official HF snapshot, optional.
DATASET_INPUT = None  # Path to pinned Circular TSV, optional.
RESUME_INPUT = None   # Path to checkpoint.zip or its extracted folder.
THINKING = True
MAX_NEW_TOKENS = 1280
MAX_IMAGE_TOKENS = 1280
SESSION_BUDGET_MINUTES = 420
MAX_ROWS_THIS_SESSION = 0  # 0: no row cap. Time budget still applies.
"""


def build(mode):
    entries = [
        ('markdown', f'# VST-7B-RL · {mode} · 3DSRBench\n\n'+ABOUT+'\n'+(SANITY if mode=='sanity' else FULL)),
        ('code', config(mode)),
        ('markdown', '## Môi trường Kaggle và checkpoint'),
        ('code', SETUP),
        ('markdown', '## Ghi mã inference vào thư mục tạm\nMã chạy trong Python riêng, dùng chung cho sanity và full.'),
        ('code', "%%writefile /kaggle/temp/vst7b_sc203/"+mode+"/vst_runner.py\n"+RUNNER),
        ('markdown', '## Chạy VST và xuất checkpoint'),
        ('code', RUN),
    ]
    cells = []
    for i,(kind,source) in enumerate(entries):
        cell = dict(id=f'vst-{mode}-{i}',cell_type=kind,metadata={},source=source.splitlines(keepends=True))
        if kind=='code':
            ast.parse(source.split('\n',1)[1] if source.startswith('%%writefile') else source)
            cell.update(execution_count=None,outputs=[])
        cells.append(cell)
    notebook = dict(cells=cells,metadata=dict(
        kernelspec=dict(display_name='Python 3', language='python', name='python3'),
        language_info=dict(name='python',version='3.11')),nbformat=4,nbformat_minor=5)
    path = HERE/f'VST_7B_RL_3DSRBench_{mode}_Kaggle.ipynb'
    path.write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+'\n',encoding='utf8')
    print(path.name)


if __name__=='__main__':
    for mode in ('sanity','full'):
        build(mode)
