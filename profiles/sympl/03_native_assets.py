"""Select a CUDA toolkit for the cu124 PyTorch extensions before running native.py."""
import json
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request
from pathlib import Path
import hashlib

def is_cuda124(root):
    compiler = root / 'bin/nvcc'
    if not compiler.is_file() or not os.access(compiler, os.X_OK):
        return False
    version = subprocess.run([str(compiler), '--version'], capture_output=True, text=True)
    return version.returncode == 0 and re.search(r'release 12\.4\b', version.stdout) is not None

def prepare_toolkit(work, env):
    private = work / 'cuda-12.4.1'
    roots = [Path('/usr/local/cuda-12.4'), Path('/usr/local/cuda')]
    current = shutil.which('nvcc', path=env.get('PATH'))
    if current:
        roots.append(Path(current).resolve().parent.parent)
    if (private / '.ready').exists():
        roots.append(private)
    for root in roots:
        if is_cuda124(root) and (root / 'include/cuda_runtime.h').exists() and (root / 'lib64/libcudart.so').exists():
            return root

    base = 'https://developer.download.nvidia.com/compute/cuda/redist/'
    with urllib.request.urlopen(base + 'redistrib_12.4.1.json', timeout=90) as response:
        manifest = json.load(response)
    private.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=work) as tmp:
        for package in ('cuda_nvcc', 'cuda_cudart', 'cuda_cccl'):
            item = manifest[package]['linux-x86_64']
            archive = Path(tmp) / (package + '.tar.xz')
            print('CUDA 12.4.1:', package, flush=True)
            digest = hashlib.sha256()
            with urllib.request.urlopen(base + item['relative_path'], timeout=180) as response, archive.open('wb') as dst:
                for chunk in iter(lambda: response.read(1024 * 1024), b''):
                    digest.update(chunk)
                    dst.write(chunk)
            if digest.hexdigest() != item['sha256']:
                raise RuntimeError('CUDA archive checksum mismatch: ' + package)
            unpack = Path(tmp) / package
            with tarfile.open(archive) as tar:
                tar.extractall(unpack, filter='data')
            top, = unpack.iterdir()
            shutil.copytree(top, private, symlinks=True, dirs_exist_ok=True)
    if not (private / 'lib64').exists():
        (private / 'lib64').symlink_to('lib', target_is_directory=True)
    if not is_cuda124(private):
        raise RuntimeError('Private CUDA 12.4 compiler failed to start')
    (private / '.ready').touch()
    return private

toolkit = prepare_toolkit(WORK, ENV)
ENV['CUDA_HOME'] = ENV['CUDA_PATH'] = str(toolkit)
ENV['PATH'] = os.pathsep.join([str(toolkit/'bin'), str(Path(PY).parent), ENV['PATH']])
# The Torch cu124 wheels supply the math-library headers used by C++ extensions.
wheel_headers = sorted((Path(PY).parent.parent/'lib/python3.10/site-packages/nvidia').glob('*/include'))
ENV['CPATH'] = os.pathsep.join(filter(None, [str(toolkit/'include'), *map(str,wheel_headers), ENV.get('CPATH')]))
ENV['LIBRARY_PATH'] = os.pathsep.join(filter(None,[str(toolkit/'lib64'),ENV.get('LIBRARY_PATH')]))
ENV['LD_LIBRARY_PATH'] = os.pathsep.join(filter(None,[str(toolkit/'lib64'),ENV.get('LD_LIBRARY_PATH')]))
with tempfile.TemporaryDirectory(dir=WORK) as tmp:
    cu = Path(tmp)/'probe.cu'
    cu.write_text('#include <cuda_runtime.h>\n__global__ void k() {}\nint main(){k<<<1,1>>>(); return cudaDeviceSynchronize()!=cudaSuccess;}')
    exe = Path(tmp)/'probe'
    subprocess.run([str(toolkit/'bin/nvcc'), '-arch=sm_75', str(cu), '-o', str(exe)], env=ENV, check=True)
    subprocess.run([str(exe)], env=ENV, check=True)
print('CUDA 12.4 compile + launch PASS:', toolkit, flush=True)

subprocess.run([PY,str(SCRIPTS/'native.py'),str(CONFIG)],env=ENV,check=True)
subprocess.run([PY,str(SCRIPTS/'assets.py'),str(CONFIG),str(OUT/'assets_ready.json')],env=ENV,check=True)
assets=json.loads((OUT/'assets_ready.json').read_text())
CFG.update(assets=assets['root'],weights=str(Path(assets['root'])/'qwen_model'),model_sha=assets['snapshots']['qwen_model']['sha'])
subprocess.run([PY,str(SCRIPTS/'patch.py'),str(REPO),CFG['assets'],str(OUT/'patches.json')],env=ENV,check=True)
shutil.copy2(WORK/'source_lock.json',OUT/'source_lock.json')
ENV['PYTHONPATH']+=os.pathsep+str(REPO/'sympl/vision_modules/src/orient_anything')

CONFIG.write_text(json.dumps(CFG))
