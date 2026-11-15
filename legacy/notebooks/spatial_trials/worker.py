"""Self-contained worker and trace writer shared by the four notebook builds."""
import argparse
import functools
import hashlib
import io
import json
import os
import re
import shutil
import sys
import time
import traceback
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from box import Box
from apc_common import atomic_json, parse_option

cfg = json.loads(Path(sys.argv[1]).read_text())
rank, world = map(int, sys.argv[2:4])
out = Path(cfg['output'])
blobs = out / 'artifacts'
blobs.mkdir(exist_ok=True)
active = None
stack = []
counter = 0

def blob(data, ext):
    name = hashlib.sha256(data).hexdigest() + ext
    path = blobs / name
    if not path.exists():
        tmp = path.with_suffix('.' + str(os.getpid()) + '.tmp')
        tmp.write_bytes(data)
        tmp.replace(path)
    return 'artifacts/' + name

def pack(v):
    if isinstance(v, Image.Image):
        buf = io.BytesIO()
        v.save(buf, format='PNG')
        return {'image': blob(buf.getvalue(), '.png'), 'size': v.size}
    if torch.is_tensor(v):
        if v.numel() > 8192:
            return {'tensor_shape': list(v.shape), 'dtype': str(v.dtype), 'device': str(v.device)}
        return v.detach().cpu().tolist()
    if isinstance(v,np.ndarray):
        if v.size <= 4096: return v.tolist()
        buf=io.BytesIO()
        np.savez_compressed(buf, value=v)
        return {'array':blob(buf.getvalue(),'.npz'), 'shape':list(v.shape),'dtype':str(v.dtype)}
    if isinstance(v,np.generic): return v.item()
    if isinstance(v,(list,tuple)): return [pack(x) for x in v]
    if isinstance(v,dict): return {str(k):pack(x) for k,x in v.items() if k!='conv_history'}
    if v is None or isinstance(v,(str,int,bool,float)): return v
    return {'component':type(v).__name__}

def event(**payload):
    with (active/'calls.jsonl').open('a',encoding='utf8') as stream:
        stream.write(json.dumps(payload,ensure_ascii=False)+'\n')

def hook(owner,name,label=None):
    original=getattr(owner,name)
    @functools.wraps(original)
    def traced(*args,**kwargs):
        global counter
        counter+=1
        idx=counter
        label_used=label or name
        event(id=idx,parent=stack[-1] if stack else None,phase='start',component=label_used,
              args=pack(args),kwargs=pack(kwargs))
        stack.append(idx)
        start=time.perf_counter()
        try:
            value=original(*args,**kwargs)
            event(id=idx,phase='end',component=label_used,seconds=time.perf_counter()-start,result=pack(value))
            return value
        except Exception as exc:
            event(id=idx,phase='error',component=label_used,seconds=time.perf_counter()-start,error=repr(exc))
            raise
        finally: stack.pop()
    setattr(owner,name,traced)

def record_detection(boxes,scores,phrases):
    event(id=stack[-1],phase='detector_candidates',coordinates='normalized cx cy w h',
          boxes=pack(boxes),scores=pack(scores),phrases=pack(phrases))

def configure():
    from experiment_qwen import KaggleQwen
    settings=Box(dict(dtype='float16',min_pixels=256*28**2,max_pixels=1280*28**2,
        max_input_tokens=12288,max_new_tokens=cfg['max_tokens'],final_max_new_tokens=cfg['max_tokens'],
        pretrained_model=cfg['weights']))
    if cfg['method']=='ssrl':
        vlm=KaggleQwen(settings,'cuda:0')
        hook(vlm,'process_messages','Spatial-SSRL-3B')
        return vlm
    import yaml
    import sympl.sympl_pipeline as pipeline
    import sympl.vision_modules.detection as detector
    import sympl.prompt_extraction as extraction
    config=Box(yaml.safe_load(Path('sympl/configs/qwenvl2_5_7b_instruct.yaml').read_text()))
    config.vlm.update(settings)
    root=Path(cfg['assets'])
    config.detection.ckpt_path=str(root/'checkpoints/groundingdino_swint_ogc.pth')
    config.depth.ckpt_path=str(root/'checkpoints/depth_pro.pt')
    config.orientation.ckpt_path=str(root)
    config.orientation.model_path=str(root/'checkpoints/orient_dino_weight.pt')
    config.orientation.processor_path=str(root/'dinov2_large')
    pipeline.get_vlm_model_class=lambda _:KaggleQwen
    model=pipeline.SymPL(config,device_vlm='cuda:0',device_vision='cuda:1')
    # GroundingDINO returns scores before the repo's top-k selection.
    predict=detector.predict
    def logged_predict(*a,**kw):
        v=predict(*a,**kw)
        record_detection(*v)
        return v
    detector.predict=logged_predict
    for name in ('get_objects_of_interest','get_reference_viewer','select_category_from_question','estimate_scene_geometry',
                 'top_view_position','do_perspective_change','do_projection_2d','do_abstraction','do_bipartition',
                 'do_localization','symbolic_layout_prompting'):
        hook(pipeline,name)
    hook(extraction,'get_objects_of_interest','retry_entity_extraction')
    for owner,names in [(model.detection_module,['run_detection','run_detection_refinement']),
                        (model.depth_module,['run_depth_intrinsic_estimation','unproject_to_3D_from_bbox']),
                        (model.orientation_module,['run_orientation_estimation','get_3angle']),
                        (model.vlm_model,['process_messages'])]:
        for name in names:hook(owner,name)
    return model

def parse(raw,row):
    if cfg['method']=='ssrl':
        # Final answer only. Never select a letter from the reasoning prefix.
        tail=raw.rsplit('</think>',1)[-1]
        if '<think>' in tail: return None
        tail=re.sub(r'\\(?:text|mathrm|mathbf)\s*\{([^{}]*)\}',r'\1',tail)
        matches=re.findall(r'\\boxed\s*\{([^{}]*)\}',tail)
        answer=matches[-1] if matches else tail
    else:
        answer=raw.strip().lower()
        if row['paper_task']=='visibility':
            answer={'yes, visible':'yes','not':'no'}.get(answer,answer)
        elif row['paper_task']=='left_right' and answer in ('left','right'):
            answer='on the '+answer
    return parse_option(answer,row['choices'])

def run():
    global active,counter
    manifest=json.loads((out/'manifest.json').read_text())
    model=configure()
    rows=[json.loads(x) for x in (out/'samples.jsonl').read_text().splitlines()]
    for i,r in enumerate(rows):
        if i%world!=rank: continue
        active=out/'traces'/r['sample_id']
        active.mkdir(parents=True,exist_ok=True)
        record=active/'result.json'
        if record.exists():
            if json.loads(record.read_text())['fingerprint']!=manifest['fingerprint']: raise ValueError('Resume fingerprint mismatch')
            continue
        (active/'calls.jsonl').unlink(missing_ok=True)
        counter=0
        vlm=model if cfg['method']=='ssrl' else model.vlm_model
        vlm.calls.clear()
        result={**r,'fingerprint':manifest['fingerprint'],'rank':rank,'prediction':None}
        start=time.perf_counter()
        print(f"[{i+1}/{len(rows)}] {r['sample_id']} start",flush=True)
        try:
            im=Image.open(r['image_path']).convert('RGB')
            result['original_image']=blob(Path(r['image_path']).read_bytes(),'.img')
            atomic_json(active/'sample.json',result)
            with torch.inference_mode():
                if cfg['method']=='ssrl':
                    options='\n'.join(f'{k}. {v}' for k,v in r['choices'].items())
                    prompt=r['question']+'\n'+options+'\nPlease select the correct option letter. '+cfg['format_prompt']
                    raw=model.generate_response(prompt,image=im,max_new_tokens=cfg['max_tokens'])
                else:
                    raw,_=model.run_sympl(im,r['question'],trace_save_dir=str(active),visualize_trace=False)
            pred=parse(raw,r)
            status='method_failure' if raw.strip().lower().startswith('fail') else ('ok' if pred else 'invalid')
            result.update(raw_output=raw,prediction=pred,status=status)
        except Exception as exc:
            result.update(status='error',error=repr(exc),traceback=traceback.format_exc())
            if isinstance(exc,OSError) or isinstance(exc,torch.cuda.OutOfMemoryError) or 'cuda error' in str(exc).lower():
                atomic_json(active/'fatal.json',result)
                raise
        result.update(seconds=time.perf_counter()-start,correct=result['prediction']==r['answer'],generation=vlm.calls)
        atomic_json(record,result)
        print(r['sample_id'],result['status'],result['prediction'],flush=True)
if __name__=='__main__':run()
