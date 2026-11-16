"""APC geometry and downstream replay in the unchanged APC/Qwen environment."""
import argparse
import copy
import json
import os
from pathlib import Path
import random
import shutil
import signal
import time
import traceback
from .common import read, write, rows, VARIANTS, select_oneref, xyxy_from_cxcywh, cxcywh_from_xyxy

def scene_arrays(scene):
    import numpy as np
    return {name:{key:np.asarray(value,dtype=float) for key,value in obj.items()} for name,obj in scene.items()}

def seed(sid, base):
    import torch
    import numpy as np
    import hashlib
    value=int.from_bytes(hashlib.sha256(f'{base}:{sid}'.encode()).digest()[:4],'big')
    random.seed(value);np.random.seed(value);torch.manual_seed(value);torch.cuda.manual_seed_all(value)
    return value

def evidence(folder, image, objects):
    import math
    import numpy as np
    from PIL import ImageDraw
    drawn=image.copy(); d=ImageDraw.Draw(drawn)
    colors=['#00bfe8','#ff4dd2','#ffd030','#ff7050']
    for n,obj in enumerate(objects):
        box=obj['selected_box'];color=colors[n%len(colors)]
        d.rectangle(box,outline=color,width=3)
        cx,cy=(box[0]+box[2])/2,(box[1]+box[3])/2
        vx,vy,vz=obj['orientation']; ex,ey=cx+70*vx,cy+70*vy
        # APC scene axes here: +x image-right, +y image-down, +z away from camera.
        d.line((cx,cy,ex,ey),fill=color,width=4)
        angle=math.atan2(ey-cy,ex-cx)
        d.polygon([(ex,ey),(ex-12*math.cos(angle-.5),ey-12*math.sin(angle-.5)),
                   (ex-12*math.cos(angle+.5),ey-12*math.sin(angle+.5))],fill=color)
        d.text((box[0],max(0,box[1]-16)),f'{n}: {obj["name"]} z={vz:+.2f}',fill=color)
    drawn.save(folder/'boxes_orientation.png')

def original_geometry(row, out):
    source=Path(row['trace_path']);folder=out/'geometry/original'/row['sample_id']
    if (folder/'result.json').exists():return
    folder.mkdir(parents=True,exist_ok=True)
    import numpy as np
    from PIL import Image
    scene=read(source/'scene.json')['camera']
    objects=[]
    for f in sorted(source.glob('object_*.name'),key=lambda p:int(p.stem.split('_')[1])):
        idx=int(f.stem.split('_')[1]);name=f.read_text()
        box=source/f'object_{idx}_selected_box.npy'
        if not box.exists() or name not in scene:continue
        b=np.load(box).tolist()
        objects.append(dict(instance_id=f'object_{idx}',name=name,selected_box=b,
            position=scene[name]['position'],orientation=scene[name]['orientation']))
        shutil.copy2(source/f'object_{idx}_mask.png',folder/f'object_{idx}_mask.png')
        Image.open(source/'input.png').crop(b).save(folder/f'object_{idx}_crop.png')
    write(folder/'result.json',dict(status='ok',scene=scene,objects=objects,source='frozen_original_trace'))
    evidence(folder,Image.open(source/'input.png').convert('RGB'),objects)

def geometry(experiment,cfg,row,kind,folder):
    import numpy as np
    import torch
    from PIL import Image
    image=Image.open(row['image_path']).convert('RGB');source=Path(row['trace_path'])
    depth=np.load(source/'depth.npy')
    names=read(source/'get_objects_of_interest.json')['result'][0]
    if len(names)!=len(set(names)):
        raise ValueError('Duplicate scene keys: require an explicit instance-identity redesign')
    scene={'camera':dict(position=[0,0,0],orientation=[0,0,1])};objects=[]
    stage='start'
    for idx,name in enumerate(names):
        stage='detection';experiment.callback(stage);key=f'{row["sample_id"]}/object_{idx}'
        sp=Path(cfg['output_dir'])/'perception/sam3'/key
        if kind.startswith('sam3'):
            result=read(sp/'result.json')
            if result['status']!='ok':raise RuntimeError('SAM 3: '+result.get('message','failed'))
            candidates=result['boxes'][:5]
        else:
            candidates=[xyxy_from_cxcywh(b,*image.size) for b in np.load(source/f'object_{idx}_boxes.npy')]
        write(folder/f'object_{idx}_candidates.json',dict(name=name,boxes=candidates,source=kind))
        if not candidates:raise ValueError(f'No candidates for {name}')
        stage='instance_selection'
        experiment.callback(stage)
        selection={}
        if 'oneref' in kind:
            prediction=read(Path(cfg['output_dir'])/'perception/oneref'/key/'result.json')
            if prediction['status']!='ok':raise RuntimeError('OneRef: '+prediction.get('message','failed'))
            selected,scores=select_oneref(candidates,prediction['box'],cfg['oneref_min_iou'])
            box=list(map(int,candidates[selected]));selection=dict(index=selected,ious=scores,grounding=prediction)
        else:
            relative=torch.tensor([cxcywh_from_xyxy(b,*image.size) for b in candidates])
            box=experiment.apc.detection_module.run_detection_refinement(experiment.vlm,image,name,relative,
                trace_save_dir=str(folder),visualize_trace=False)
            selected=max(range(len(candidates)),key=lambda i: -sum(abs(float(a)-float(b)) for a,b in zip(box,candidates[i])))
            selection=dict(index=selected,selected_box=box,policy='unchanged_APC_Qwen_crop_refinement')
        write(folder/f'object_{idx}_selection.json',selection)
        stage='segmentation'
        experiment.callback(stage)
        if kind.startswith('sam3'):
            mask=np.asarray(Image.open(sp/f'mask_{selected}.png'))>0
        else:
            mask=experiment.apc.detection_module.run_segmentation(image,box)[2]
        if not np.asarray(mask).any():raise ValueError('Empty object mask: '+name)
        Image.fromarray(np.asarray(mask).astype('uint8')*255).save(folder/f'object_{idx}_mask.png')
        image.crop(box).save(folder/f'object_{idx}_crop.png')
        stage='depth_to_coordinates'
        experiment.callback(stage)
        position,_=experiment.apc.depth_module.unproject_to_3D(image,depth,mask,return_box3D=False)
        stage='orientation_v1'
        experiment.callback(stage)
        orientation=experiment.apc.orientation_module.run_orientation_estimation(image,category=name,bbox=box,
                          trace_save_dir=str(folder),visualize_trace=False)
        if not np.isfinite(position).all() or not np.isfinite(orientation).all():raise ValueError('Nonfinite scene geometry')
        obj=dict(instance_id=f'object_{idx}',name=name,selected_box=list(map(float,box)),
                 position=np.asarray(position).tolist(),orientation=np.asarray(orientation).tolist())
        objects.append(obj);scene[name]=dict(position=obj['position'],orientation=obj['orientation'])
        write(folder/'partial_scene.json',dict(scene=scene,objects=objects))
    evidence(folder,image,objects)
    return dict(scene=scene,objects=objects,depth_source='frozen_DepthPro_original',
                intrinsics='unchanged APC focal_length=4*image_width',stage=stage)

def answer(experiment,cfg,row,variant,folder):
    import numpy as np
    import torch
    from PIL import Image
    kind,orientation=VARIANTS[variant];out=Path(cfg['output_dir'])
    g=read(out/'geometry'/kind/row['sample_id']/'result.json')
    if g['status']!='ok':raise RuntimeError('Geometry failed: '+g.get('message',''))
    scene=scene_arrays(g['scene']);objects=copy.deepcopy(g['objects'])
    if orientation=='v2':
        for obj in objects:
            value=read(out/'perception/oa2'/kind/row['sample_id']/obj['instance_id']/'result.json')
            if value['status']!='ok':raise RuntimeError('OA2 failed: '+value.get('message',''))
            obj['orientation']=value['direction_vector'];obj['oa2']=value
            scene[obj['name']]['orientation']=np.asarray(obj['orientation'])
    viewer=read(Path(row['trace_path'])/'scene.json')['viewer']
    if viewer not in scene:raise ValueError('Frozen viewer absent from intervened scene: '+viewer)
    evidence(folder,Image.open(row['image_path']).convert('RGB'),objects)
    write(folder/'intervention.json',dict(variant=variant,objects=objects,viewer=viewer,
        controls=['object descriptions','reference viewer','depth map','Qwen weights and prompts','renderer'],
        selected_names_source='historical trace; not gold annotations'))
    transformed=experiment.apc.do_perspective_change(scene,ref_viewer=viewer)
    write(folder/'scene.json',dict(viewer=viewer,camera=scene,transformed=transformed))
    options='\n'.join(f'{k}. {v}' for k,v in row['choices'].items())
    question=row['question']+'\nAnswer options:\n'+options
    with torch.inference_mode():
        raw,_=experiment.apc.do_perspective_prompting_visual(viewer,copy.deepcopy(transformed),question,
            conv_history=None,trace_save_dir=str(folder),visualize_trace=False,
            visualize_scene_abstraction=False,render_whole_scene=row['category']=='orientation_on_the_left',options=options)
    from apc_common import parse_option
    prediction=parse_option(raw,row['choices'])
    return dict(status='ok' if prediction is not None else 'invalid',prediction=prediction,
                raw_output=raw,correct=prediction==row['answer'],viewer=viewer)

def run(config,phase):
    cfg=read(config);context=read(Path(cfg['runtime_dir'])/'apc_context.json')
    os.chdir(context['repo'])
    import sys
    sys.path[:0]=[context['scripts'],context['repo']]
    from .experiment import make
    import torch
    out=Path(cfg['output_dir']);items=rows(out/'samples.jsonl')
    variants=cfg['variants'];kinds=list(dict.fromkeys(VARIANTS[v][0] for v in variants))
    jobs=[]
    if phase=='geometry':
        for row in items:
            original_geometry(row,out)
            for kind in kinds:
                if kind!='original':jobs.append((row,kind,out/'geometry'/kind/row['sample_id']))
    else:
        for row in items:
            for variant in variants:jobs.append((row,variant,out/'answers'/variant/row['sample_id']))
    jobs=[j for j in jobs if not (j[2]/'result.json').exists() or
          (cfg.get('retry_errors') and read(j[2]/'result.json').get('status')=='error')]
    if not jobs:return
    stage={'name':'loading'}
    def progress(name,**kw):stage['name']=name
    experiment=make(cfg,context,progress)
    def timeout(*unused):raise TimeoutError('Sample time budget exceeded')
    signal.signal(signal.SIGALRM,timeout)
    for row,name,folder in jobs:
        folder.mkdir(parents=True,exist_ok=True)
        # Each retry writes a new attempt folder; older prompts remain auditable.
        previous=list(folder.glob('attempt_*'))
        attempt=folder/f'attempt_{len(previous)+1:03d}';attempt.mkdir()
        os.environ['APC_TRACE_DIR']=str(attempt);experiment.vlm.calls=[]
        started=time.monotonic();value=dict(status='error',correct=False,prediction=None,
            sample_id=row['sample_id'],paper_task=row['paper_task'],variant=name,attempt=str(attempt.relative_to(out)),
            seed=seed(row['sample_id'],cfg['seed']))
        signal.alarm(cfg['sample_timeout_seconds'])
        try:
            with torch.inference_mode():
                if phase=='geometry':value.update(geometry(experiment,cfg,row,name,attempt),status='ok')
                else:value.update(answer(experiment,cfg,row,name,attempt))
        except Exception as exc:
            value.update(error_type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc(),failed_stage=stage['name'])
        finally:
            signal.alarm(0)
            value.update(seconds=time.monotonic()-started,vlm_calls=experiment.vlm.calls)
            write(folder/'result.json',value)
            # Small pointer index; intermediate artifacts remain in the attempt directory.
            print(phase,name,row['sample_id'],value['status'],round(value['seconds'],1),flush=True)
        if value.get('error_type')=='OutOfMemoryError' or 'device-side assert' in value.get('message',''):
            raise RuntimeError('GPU fault saved; restart process/session before retry')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('phase',choices=['geometry','answers'])
    a=p.parse_args();run(a.config,a.phase)
