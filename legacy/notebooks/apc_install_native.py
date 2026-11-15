"""Install native vision dependencies into the isolated Kaggle venv."""
import hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path
cfg=json.loads(Path(sys.argv[1]).read_text());repo=Path(cfg['repo']);work=Path(cfg['work']);py=sys.executable
early_nvcc=shutil.which('nvcc')
if not early_nvcc:raise RuntimeError('nvcc is missing. Native APC vision setup needs a CUDA 12 development toolkit, not only an NVIDIA driver.')
early_version=subprocess.check_output([early_nvcc,'--version'],text=True)
if 'release 12.' not in early_version:raise RuntimeError('Use a CUDA 12 toolkit with the cu124 Torch wheel. Found: '+early_version)
src=repo/'apc/vision_modules/src';src.mkdir(parents=True,exist_ok=True)
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
for name,(url,ref) in repos.items():
 path=src/name
 if not (path/'.git').exists():
  check=subprocess.run(['git','ls-remote','--exit-code',url,'refs/heads/'+ref,'refs/tags/'+ref],capture_output=True,text=True)
  if check.returncode:
   raise RuntimeError('Git ref missing or unreachable (case-sensitive): '+ref+' in '+url+'; '+check.stderr.strip())
  run(['git','clone','--depth','1','--branch',ref,url,str(path)])
 sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=path,text=True).strip()
 if name in lock and lock[name]['sha']!=sha:raise RuntimeError('Source revision changed: '+name)
 lock[name]=dict(url=url,sha=sha,requested_ref=ref)
 lock_path.write_text(json.dumps(lock,indent=2),encoding='utf-8')
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
for name in ('segment-anything','ml-depth-pro','detectron2','pytorch3d','GroundingDINO'):
 marker=work/('installed_'+name+'.json')
 identity=dict(source_sha=lock[name]['sha'],torch=torch.__version__,python=sys.version.split()[0])
 if marker.exists() and json.loads(marker.read_text())==identity:continue
 run([uv,'pip','install','--python',py,'--no-cache','--no-build-isolation','--no-deps']+(['-e'] if name in ('GroundingDINO','ml-depth-pro') else [])+[str(src/name)],env=env)
 marker.write_text(json.dumps(identity),encoding='utf-8')
print('Native dependencies installed. Sources pinned in source_lock.json.',flush=True)
