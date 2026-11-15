with setup_stage('native_assets_renderer', OUT, CFG):
    'Select a CUDA toolkit for the cu124 PyTorch extensions before running native.py.'
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
        return version.returncode == 0 and re.search('release 12\\.4\\b', version.stdout) is not None

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
    CUDA_WORK = Path('/kaggle/temp/apc_cuda_tools')
    CUDA_WORK.mkdir(parents=True, exist_ok=True)
    toolkit = prepare_toolkit(CUDA_WORK, ENV)
    ENV['CUDA_HOME'] = ENV['CUDA_PATH'] = str(toolkit)
    ENV['PATH'] = os.pathsep.join([str(toolkit / 'bin'), str(Path(PY).parent), ENV['PATH']])
    wheel_headers = sorted((Path(PY).parent.parent / 'lib/python3.10/site-packages/nvidia').glob('*/include'))
    ENV['CPATH'] = os.pathsep.join(filter(None, [str(toolkit / 'include'), *map(str, wheel_headers), ENV.get('CPATH')]))
    ENV['LIBRARY_PATH'] = os.pathsep.join(filter(None, [str(toolkit / 'lib64'), ENV.get('LIBRARY_PATH')]))
    ENV['LD_LIBRARY_PATH'] = os.pathsep.join(filter(None, [str(toolkit / 'lib64'), ENV.get('LD_LIBRARY_PATH')]))
    with tempfile.TemporaryDirectory(dir=WORK) as tmp:
        cu = Path(tmp) / 'probe.cu'
        cu.write_text('#include <cuda_runtime.h>\n__global__ void k() {}\nint main(){k<<<1,1>>>(); return cudaDeviceSynchronize()!=cudaSuccess;}')
        exe = Path(tmp) / 'probe'
        subprocess.run([str(toolkit / 'bin/nvcc'), '-arch=sm_75', str(cu), '-o', str(exe)], env=ENV, check=True)
        subprocess.run([str(exe)], env=ENV, check=True)
    print('CUDA 12.4 compile + launch PASS:', toolkit, flush=True)
    ENV.pop('HF_HUB_OFFLINE', None)
    ENV.pop('TRANSFORMERS_OFFLINE', None)
    run_logged('native_install', [PY, str(SCRIPTS / 'apc_install_native.py'), str(CONFIG)], OUT, env=ENV)
    run_logged('assets', [PY, str(SCRIPTS / 'apc_prepare_assets.py'), str(CONFIG), str(WORK / 'assets_ready.json')], OUT, env=ENV)
    assets = json.loads((WORK / 'assets_ready.json').read_text())
    ENV['PYTHONPATH'] += os.pathsep + str(REPO / 'apc/vision_modules/src/omni3d') + os.pathsep + str(REPO / 'apc/vision_modules/src/orient_anything')
    run_logged('compatibility_patches', [PY, str(SCRIPTS / 'apc_kaggle_patches.py'), str(REPO), assets['root'], str(WORK / 'patches.json')], OUT, env=ENV)
    ENV['HF_HUB_OFFLINE'] = '1'
    ENV['TRANSFORMERS_OFFLINE'] = '1'
    ENV['DISPLAY'] = ':99'
    ENV['LIBGL_ALWAYS_SOFTWARE'] = '1'
    ENV['MESA_GL_VERSION_OVERRIDE'] = '3.3'
    ENV['MPLBACKEND'] = 'Agg'
    if not Path('/tmp/.X99-lock').exists():
        display_log = (WORK / 'xvfb.log').open('a')
        XVFB = subprocess.Popen(['Xvfb', ':99', '-screen', '0', '1280x1024x24', '-nolisten', 'tcp'], stdout=display_log, stderr=subprocess.STDOUT, env=ENV)
        time.sleep(2)
    probe = "\nimport io,json,torch,trimesh\nfrom PIL import Image\nfrom groundingdino import _C\nimport detectron2,pytorch3d,depth_pro,open3d,segment_anything\nfrom apc_kaggle_memory import plan_qwen_device_map\nfrom apc.apc_pipeline import APC\nscene=trimesh.Scene(trimesh.creation.box());raw=scene.save_image(resolution=(128,128),visible=False)\nim=Image.open(io.BytesIO(raw));im.verify()\nprint('Native imports + headless Trimesh render PASS',flush=True)\n"
    run_logged('native_renderer_probe', [PY, '-u', '-c', probe], OUT, env=ENV, cwd=REPO)
