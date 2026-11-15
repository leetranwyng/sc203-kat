LOCKS={'GroundingDINO': {'url': 'https://github.com/IDEA-Research/GroundingDINO.git', 'sha': '856dde20aee659246248e20734ef9ba5214f5e44', 'requested_ref': 'main'}, 'ml-depth-pro': {'url': 'https://github.com/apple/ml-depth-pro.git', 'sha': '9e65e4dbe9568d23c546fcec53302b10445e109e', 'requested_ref': 'main'}, 'orient_anything': {'url': 'https://github.com/SpatialVision/Orient-Anything.git', 'sha': '759282c26e924988c952d6d33212c48349dc9aff', 'requested_ref': 'main'}, 'omni3d': {'url': 'https://github.com/facebookresearch/omni3d.git', 'sha': '778bd0210e5e1b584cc52daeba57cae3a3ee8c4e', 'requested_ref': 'main'}, 'detectron2': {'url': 'https://github.com/facebookresearch/detectron2.git', 'sha': '247691a1c292654867deb35cabb27f18d75a1363', 'requested_ref': 'main'}, 'pytorch3d': {'url': 'https://github.com/facebookresearch/pytorch3d.git', 'sha': '75ebeeaea0908c5527e7b1e305fbc7681382db47', 'requested_ref': 'V0.7.8'}, 'segment-anything': {'url': 'https://github.com/facebookresearch/segment-anything.git', 'sha': 'dca509fe793f601edb92606367a655c15ac00fdf', 'requested_ref': 'main'}}
"""Install native vision dependencies into the isolated Kaggle venv."""
import hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path
cfg=json.loads(Path(sys.argv[1]).read_text());repo=Path(cfg['repo']);work=Path(cfg['work']);py=sys.executable
early_nvcc=shutil.which('nvcc')
if not early_nvcc:raise RuntimeError('nvcc is missing. Native APC vision setup needs a CUDA 12 development toolkit, not only an NVIDIA driver.')
early_version=subprocess.check_output([early_nvcc,'--version'],text=True)
if 'release 12.' not in early_version:raise RuntimeError('Use a CUDA 12 toolkit with the cu124 Torch wheel. Found: '+early_version)
src=repo/'sympl/vision_modules/src';src.mkdir(parents=True,exist_ok=True)
lock_path=work/'source_lock.json';lock=json.loads(lock_path.read_text()) if lock_path.exists() else {}
repos={
 'GroundingDINO':('https://github.com/IDEA-Research/GroundingDINO.git','main'),
 'ml-depth-pro':('https://github.com/apple/ml-depth-pro.git','main'),
 'orient_anything':('https://github.com/SpatialVision/Orient-Anything.git','main'),
 'omni3d':('https://github.com/facebookresearch/omni3d.git','main'),
 'detectron2':('https://github.com/facebookresearch/detectron2.git','main'),
 'pytorch3d':('https://github.com/facebookresearch/pytorch3d.git','V0.7.8'),
 'segment-anything':('https://github.com/facebookresearch/segment-anything.git','main')}
def run(cmd,**kw):
 print('RUN',cmd[:5],flush=True);subprocess.run(cmd,check=True,**kw)
for name in ('GroundingDINO','ml-depth-pro','orient_anything','pytorch3d','segment-anything'):
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
# Do not install Orient-Anything requirements: they downgrade transformers to 4.38.
packages=['pip==25.0.1','setuptools==75.3.0','wheel','ninja','cython==0.29.37','numpy==1.26.4','scipy==1.11.4',
 'matplotlib==3.8.4','python-box==7.3.2','trimesh==4.7.4','open3d==0.19.0','pyglet==1.5.27',
 'opencv-python==4.10.0.84','timm==1.0.22','pillow-heif==0.18.0','fvcore==0.1.5.post20221221',
 'iopath==0.1.10','yacs','cloudpickle','termcolor','pycocotools','hydra-core==1.3.2','omegaconf==2.3.0',
 'shapely==2.0.6','rtree','addict','yapf==0.40.2','supervision==0.22.0','qwen-vl-utils==0.0.14','av==13.1.0']
uv=cfg['uv'];run([uv,'pip','install','--python',py,'--no-cache',*packages])
# System display and compiler tools. No hardcoded DISPLAY=:1 from run_APC.py.
apt=['apt-get'] if os.geteuid()==0 else ['sudo','-n','apt-get']
run(apt+['update','-qq']);run(apt+['install','-y','-qq','xvfb','xauth','libgl1','libglu1-mesa','libgl1-mesa-dri','build-essential','ninja-build'])
import torch
nvcc=shutil.which('nvcc')
if not nvcc:raise RuntimeError('GroundingDINO CUDA build requires nvcc. Attach/select a Kaggle runtime with a CUDA 12 toolkit; driver alone is insufficient.')
version=subprocess.check_output([nvcc,'--version'],text=True)
if 'release 12.' not in version:raise RuntimeError('CUDA toolkit must be major 12 for the cu124 wheel. Found: '+version)
env=os.environ.copy();env['CUDA_HOME']=str(Path(nvcc).resolve().parent.parent);env['FORCE_CUDA']='1';env['MAX_JOBS']='2'
env['TORCH_CUDA_ARCH_LIST']=';'.join(sorted({'.'.join(map(str,torch.cuda.get_device_capability(i))) for i in range(torch.cuda.device_count())}))
# Build with the already installed torch, without a second isolated copy of torch/CUDA.
for name in ('segment-anything','ml-depth-pro','pytorch3d','GroundingDINO'):
 marker=work/('installed_'+name+'.json')
 identity=dict(source_sha=lock[name]['sha'],torch=torch.__version__,python=sys.version.split()[0],cuda_compiler=version.strip())
 if marker.exists() and json.loads(marker.read_text())==identity:continue
 build_root=(src/name/'build').resolve()
 build_root.relative_to((src/name).resolve())
 if build_root.is_dir():
  for staging in build_root.glob('bdist.*'):
   target=staging.resolve()
   if staging.is_symlink() or target.parent!=build_root:raise RuntimeError('Unexpected build staging path')
   if target.is_dir():shutil.rmtree(target)
 run([uv,'pip','install','--python',py,'--no-cache','--no-build-isolation','--no-deps']+(['-e'] if name in ('GroundingDINO','ml-depth-pro') else [])+[str(src/name)],env=env)
 marker.write_text(json.dumps(identity),encoding='utf-8')
print('Native dependencies installed. Sources pinned in source_lock.json.',flush=True)
