# Kiểm tra GPU trước khi tải/cài package. Không cần import torch của kernel.
import csv
smi = shutil.which('nvidia-smi')
if not smi:
    smi = next((p for p in ('/usr/bin/nvidia-smi', '/usr/local/nvidia/bin/nvidia-smi') if Path(p).is_file()), None)
if not smi:
    raise RuntimeError('Phiên Kaggle chưa có nvidia-smi. Chọn Settings > Accelerator > GPU T4 x2, rồi khởi động lại session. Cài package không thể cấp GPU cho phiên CPU.')
gpu_query = subprocess.run(
    [smi, '--query-gpu=uuid,name,driver_version,memory.total', '--format=csv,noheader,nounits'],
    capture_output=True, text=True)
print(gpu_query.stdout)
if gpu_query.returncode != 0:
    print(gpu_query.stderr)
    raise RuntimeError('NVIDIA driver chưa truy cập được. Chọn accelerator 2 GPU và khởi động lại session Kaggle.')
gpu_rows = [row for row in csv.reader(gpu_query.stdout.splitlines()) if len(row) == 4]
if len(gpu_rows) < 2:
    raise RuntimeError(f'Phiên này chỉ có {len(gpu_rows)} GPU. Full benchmark cần hai GPU; khởi động lại session sau khi chọn accelerator.')
GPU_IDS = [row[0].strip() for row in gpu_rows[:2]]
print('GPU UUID được dùng:', GPU_IDS)

minimum_env_space = (0.25 if (WORK / 'venv_py310_uv/pyvenv.cfg').is_file() else 8) * 2**30
if shutil.disk_usage(WORK).free < minimum_env_space:
    raise RuntimeError('Không đủ working cho cài môi trường (8 GiB khi cài mới, 0.25 GiB khi tiếp tục). Dùng model_input hoặc session mới nếu cần.')

# Bootstrap độc lập với pip/ensurepip/Python version của kernel Kaggle.
import io, platform, tarfile, urllib.request
assert platform.system() == 'Linux' and platform.machine() in ('x86_64', 'AMD64'), 'Cell này dành cho Kaggle Linux x64.'
UV_VERSION = '0.9.5'
UV_DIR = WORK / 'tools' / ('uv-' + UV_VERSION)
UV = UV_DIR / 'uv'
if not UV.exists():
    asset = 'uv-x86_64-unknown-linux-gnu.tar.gz'
    base_url = f'https://github.com/astral-sh/uv/releases/download/{UV_VERSION}/'
    with urllib.request.urlopen(base_url + asset, timeout=120) as response:
        raw = response.read()
    with urllib.request.urlopen(base_url + asset + '.sha256', timeout=120) as response:
        expected = response.read().decode().split()[0]
    assert hashlib.sha256(raw).hexdigest() == expected, 'uv checksum mismatch'
    UV_DIR.mkdir(parents=True, exist_ok=True)
    # Chỉ lấy đúng binary; không extract đường dẫn bất kỳ từ tar.
    with tarfile.open(fileobj=io.BytesIO(raw), mode='r:gz') as archive:
        members = [m for m in archive.getmembers() if Path(m.name).name == 'uv' and m.isfile()]
        assert len(members) == 1
        with archive.extractfile(members[0]) as binary:
            UV.write_bytes(binary.read())
    UV.chmod(0o755)

UV.chmod(UV.stat().st_mode | 0o111)

BOOT_ENV = os.environ.copy()
# Chỉ bỏ đường dẫn Python của kernel; GIỮ đường dẫn thư viện NVIDIA của Kaggle.
for key in ('PYTHONPATH', 'PYTHONHOME', 'VIRTUAL_ENV'):
    BOOT_ENV.pop(key, None)
BOOT_ENV['PYTHONNOUSERSITE'] = '1'
BOOT_ENV['UV_PYTHON_INSTALL_DIR'] = str(WORK / 'python_runtimes')
BOOT_ENV['UV_CACHE_DIR'] = str(WORK / 'uv_cache')
BOOT_ENV['UV_NO_PROGRESS'] = '1'
BOOT_ENV['UV_LINK_MODE'] = 'copy'
BOOT_ENV.pop('UV_SYSTEM_PYTHON', None)
# Dùng UUID để giữ đúng GPU khi thứ tự device hoặc mask của kernel khác nhau.
BOOT_ENV['CUDA_VISIBLE_DEVICES'] = ','.join(GPU_IDS)
# Giữ LD_LIBRARY_PATH gốc và bổ sung các thư mục driver phổ biến nếu có.
driver_dirs = [p for p in ('/usr/local/nvidia/lib64', '/usr/local/nvidia/lib',
                           '/usr/lib64-nvidia', '/usr/lib/x86_64-linux-gnu')
               if Path(p).is_dir()]
inherited_dirs = [p for p in BOOT_ENV.get('LD_LIBRARY_PATH', '').split(':') if p]
BOOT_ENV['LD_LIBRARY_PATH'] = ':'.join(dict.fromkeys(inherited_dirs + driver_dirs))
VENV = WORK / 'venv_py310_uv'  # Không tái sử dụng venv Python 3.13 tạo dở.
PY = str(VENV / 'bin/python')
subprocess.run([str(UV), '--version'], env=BOOT_ENV, check=True)
# Restore execute permissions before uv tries to inspect an existing Python.
for binary in (WORK / 'python_runtimes').glob('cpython-3.10*/bin/python*'):
    if binary.is_file():binary.chmod(binary.stat().st_mode | 0o111)
for binary in (VENV / 'bin').glob('*'):
    if binary.is_file():binary.chmod(binary.stat().st_mode | 0o111)
subprocess.run([str(UV), 'python', 'install', '3.10'], env=BOOT_ENV, check=True)
if not (VENV / 'pyvenv.cfg').is_file():
    subprocess.run([str(UV), 'venv', '--python', '3.10', '--managed-python', str(VENV)], env=BOOT_ENV, check=True)
# A restored venv may retain pyvenv.cfg but have a broken interpreter symlink.
if not Path(PY).exists() and (VENV / 'pyvenv.cfg').is_file():
    managed = subprocess.check_output([str(UV), 'python', 'find', '--managed-python', '3.10'], env=BOOT_ENV, text=True).strip()
    executable = Path(managed).resolve(strict=True)
    executable.chmod(executable.stat().st_mode | 0o111)
    link = Path(PY)
    link.parent.mkdir(parents=True, exist_ok=True)
    link.unlink(missing_ok=True)
    link.symlink_to(executable)
subprocess.run([PY, '-c', 'import sys; assert sys.version_info[:2] == (3, 10); print(sys.version)'], env=BOOT_ENV, check=True)
# Dùng uv pip trực tiếp, không cần cài pip/ensurepip trong venv.
subprocess.run([str(UV), 'pip', 'install', '--python', PY, '--no-cache',
                '--index-url', 'https://download.pytorch.org/whl/cu124',
                'torch==2.4.1', 'torchvision==0.19.1'], env=BOOT_ENV, check=True)
packages = [
    'numpy==1.26.4','transformers==4.57.1','tokenizers==0.22.1',
    'accelerate==1.11.0','huggingface-hub==0.36.0','safetensors==0.6.2',
    'Pillow==11.3.0','einops==0.8.1','timm==1.0.22','sentencepiece==0.2.0',
    'python-box==7.3.2','qwen-vl-utils==0.0.14','av==13.1.0','requests','tqdm','packaging','psutil','protobuf',
]
subprocess.run([str(UV), 'pip', 'install', '--python', PY, '--no-cache', *packages], env=BOOT_ENV, check=True)
ENV = BOOT_ENV.copy()
ENV['PYTHONNOUSERSITE'] = '1'
ENV['HF_HOME'] = str(WORK / 'hf_cache')
ENV['HF_HUB_DISABLE_XET'] = '1'
ENV['OMP_NUM_THREADS'] = '2'
ENV['OPENBLAS_NUM_THREADS'] = '2'
probe = r"""
import ctypes, os, sys
import torch
print('Python:', sys.version.split()[0])
print('torch:', torch.__version__, 'CUDA wheel:', torch.version.cuda)
print('CUDA_VISIBLE_DEVICES:', os.environ.get('CUDA_VISIBLE_DEVICES'))
print('LD_LIBRARY_PATH:', os.environ.get('LD_LIBRARY_PATH'))
try:
    driver = ctypes.CDLL('libcuda.so.1')
    rc = driver.cuInit(0)
    print('NVIDIA driver cuInit result:', rc)
except OSError as exc:
    print('Cannot load NVIDIA driver:', exc)
available = torch.cuda.is_available()
count = torch.cuda.device_count()
print('CUDA available:', available, 'device count:', count)
if not available:
    try:
        torch.cuda.init()
    except Exception as exc:
        print('CUDA initialization error:', type(exc).__name__, str(exc))
    raise SystemExit(2)
if count != 2:
    print('Expected two visible GPUs.')
    raise SystemExit(3)
# Kiểm tra tính toán thật trên từng GPU, không chỉ liệt kê tên.
for i in range(count):
    p = torch.cuda.get_device_properties(i)
    x = torch.ones((8, 8), device=f'cuda:{i}')
    assert (x @ x)[0, 0].item() == 8
    torch.cuda.synchronize(i)
    print(i, p.name, round(p.total_memory / 2**30, 2), 'GiB; compute OK')
import transformers
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, AutoModel, AutoTokenizer
import torchvision, timm
print('transformers:',transformers.__version__,'torchvision:',torchvision.__version__,'timm:',timm.__version__)
"""
probe_result = subprocess.run([PY, '-c', probe], env=ENV, capture_output=True, text=True)
print(probe_result.stdout)
if probe_result.stderr:
    print(probe_result.stderr)
(WORK / 'gpu_check.txt').write_text(
    gpu_query.stdout + '\n' + probe_result.stdout + '\n' + probe_result.stderr,
    encoding='utf-8')
if probe_result.returncode != 0:
    raise RuntimeError('GPU/import check thất bại. Xem CUDA initialization error và gpu_check.txt ở trên. Nếu nvidia-smi không có hai GPU, cần đổi accelerator và khởi động lại session; không cần cài lại package.')
print('Môi trường sẵn sàng; GPU compute check thành công.')
CFG['uv']=str(UV);CONFIG.write_text(json.dumps(CFG,indent=2),encoding='utf-8')
ENV['PYTHONPATH']=str(SCRIPTS)+os.pathsep+str(REPO)
ENV['MPLBACKEND']='Agg'
ENV['CUDA_LAUNCH_BLOCKING']='1' if CFG['sanity'] else '0'
