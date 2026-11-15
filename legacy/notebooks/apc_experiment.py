"""One APC pipeline spanning two GPUs; baseline reuses exactly the same Qwen weights."""
import contextlib,copy,hashlib,json,logging,os,random,shutil,signal,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from box import Box
from apc_common import atomic_json,parse_option,read_jsonl

class APCExperiment:
    def __init__(self,cfg,assets,callback=None,sanity=False):
        self.cfg=cfg;self.sanity=sanity;self.callback=callback or (lambda *a,**k:None)
        import yaml
        self.callback('loading_qwen')
        config=Box(yaml.safe_load(Path('apc/configs/qwenvl2_5_7b_instruct.yaml').read_text()))
        root=Path(assets['root'])
        config.vlm.pretrained_model=str(root/'qwen_model');config.vlm.dtype=cfg['dtype']
        config.vlm.min_pixels=cfg['min_pixels'];config.vlm.max_pixels=cfg['max_pixels']
        config.vlm.final_max_new_tokens=cfg['final_max_new_tokens']
        config.vlm.max_new_tokens=cfg['intermediate_max_new_tokens'];config.vlm.max_input_tokens=cfg['max_input_tokens']
        config.detection.ckpt_path=str(root/'checkpoints/groundingdino_swint_ogc.pth')
        config.detection.box_threshold=cfg['detection_box_threshold'];config.detection.text_threshold=cfg['detection_text_threshold']
        config.segmentation.ckpt_path=str(root/'checkpoints/sam_vit_h_4b8939.pth');config.depth.ckpt_path=str(root/'checkpoints/depth_pro.pt')
        config.orientation.ckpt_path=str(root/'hf_cache');config.orientation.model_path=str(root/'checkpoints/orient_dino_weight.pt')
        config.orientation.processor_path=str(root/'dinov2_large')
        # Replace only the VLM class, retaining APC's stages and prompts.
        from apc import apc_pipeline
        from apc_qwen_adapter import KaggleQwen
        apc_pipeline.get_vlm_model_class=lambda kind:KaggleQwen if kind=='qwenvl2_5' else (_ for _ in ()).throw(ValueError(kind))
        self.callback('loading_vision')
        self.apc=apc_pipeline.APC(config,device_vlm='cuda:0',device_vision='cuda:1')
        self.vlm=self.apc.vlm_model;self.vlm.callback=self.callback
        self.apc.prompt_parser.predefined_prompts['choose_options']+='\nAnswer with the option letter only.'
        self.apc.logger.setLevel(logging.INFO if sanity else logging.WARNING)
        # Compute depth only once per sample even when several objects are abstracted.
        original_depth=self.apc.depth_module.run_depth_estimation
        self.depth_cache=None
        def depth_once(image):
            if self.depth_cache is None:self.depth_cache=original_depth(image)
            return self.depth_cache
        self.apc.depth_module.run_depth_estimation=depth_once
        # SAM embedding is also identical for every object of one input image.
        predictor=self.apc.detection_module.segmentation_predictor;original_set=predictor.set_image;self.sam_image_hash=None
        def set_once(image,*a,**kw):
            digest=hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()
            if digest!=self.sam_image_hash:
                result=original_set(image,*a,**kw);self.sam_image_hash=digest;return result
        predictor.set_image=set_once
        for name in ('get_objects_of_interest','do_scene_abstraction','get_reference_viewer','do_perspective_change','do_perspective_prompting_visual'):
            original=getattr(self.apc,name)
            def timed(*a,_name=name,_fn=original,**kw):
                self.callback(_name)
                if _name=='do_perspective_prompting_visual' and len(a)>1:
                    a=list(a);a[1]=copy.deepcopy(a[1]);a=tuple(a)
                result=_fn(*a,**kw)
                if _name=='get_objects_of_interest' and (not isinstance(result,tuple) or not result[0]):raise ValueError('APC could not parse object names')
                if _name=='do_scene_abstraction':
                    if len(result)<2:raise ValueError('APC detected no usable objects')
                    for obj in result.values():
                        if not np.isfinite(obj['position']).all() or not np.isfinite(obj['orientation']).all():raise FloatingPointError('Nonfinite scene geometry')
                return result
            setattr(self.apc,name,timed)
        original_render=self.apc.render_module.render_visual_prompt
        def render_prompt(*args,**kw):
            self.callback('render_visual_prompt')
            result=original_render(*args,**kw)
            if sanity:result.save(Path(kw['trace_save_dir'])/'apc_visual_prompt.png')
            return result
        self.apc.render_module.render_visual_prompt=render_prompt
        if sanity:
            def finite_hook(module,args,output):
                logits=getattr(output,'logits',None)
                if logits is not None and not torch.isfinite(logits[:,-1,:]).all().item():raise FloatingPointError('NaN/Inf in Qwen raw logits')
            self.vlm.vlm_model.register_forward_hook(finite_hook)
        self.callback('ready')

    def run(self,row,mode,scratch):
        self.depth_cache=None;self.sam_image_hash=None;self.vlm.calls=[]
        options='\n'.join(f'{k}. {v}' for k,v in row['choices'].items())
        # Choices carry object descriptions for questions such as "Which object is higher?".
        question=row['question']+'\nAnswer options:\n'+options
        with Image.open(row['image_path']) as image:
            image=image.convert('RGB')
            with torch.inference_mode():
                if mode=='direct':raw=self.vlm.generate_response(question+'\nAnswer with the option letter only.',image,max_new_tokens=self.cfg['final_max_new_tokens'])
                elif mode=='apc_vis':
                    raw,_=self.apc.run_apc(image,question,trace_save_dir=str(scratch),perspective_prompt_type='visual',
                         visualize_trace=False,visualize_scene_abstraction=False,
                         render_whole_scene=row['category']=='orientation_on_the_left',return_conv_history=False,options=options)
                else:raise ValueError(mode)
        if not isinstance(raw,str):raise TypeError('APC response must be text')
        return raw

def is_fatal(exc):
    text=str(exc).lower()
    return isinstance(exc,torch.cuda.OutOfMemoryError) or isinstance(exc,RuntimeError) and any(s in text for s in ('device-side assert','illegal memory','cuda error','cublas_status','out of memory'))

def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--manifest',required=True);parser.add_argument('--smoke',action='store_true');a=parser.parse_args()
    m=json.loads(Path(a.manifest).read_text());cfg=m['config'];out=Path(cfg['output']);folder=out/('smoke' if a.smoke else 'full');folder.mkdir(exist_ok=True)
    rows=[json.loads(x) for x in (out/'samples.jsonl').read_text().splitlines()]
    if a.smoke:rows=rows[:1]
    state_path=folder/'state.json';started=time.monotonic();state=dict(stage='starting',processed=0,total=len(rows)*len(cfg['modes']),new_processed=0,errors=0,invalid=0)
    def update(stage,**kw):
        state.update(stage=stage,updated_unix=time.time(),elapsed_seconds=time.monotonic()-started,**kw);atomic_json(state_path,state)
    previous={mode:({} if a.smoke else read_jsonl(folder/f'predictions_{mode}.jsonl',repair=True)) for mode in cfg['modes']}
    allowed={r['sample_id'] for r in rows}
    for mode,values in previous.items():
        for sid,r in values.items():assert sid in allowed and r['fingerprint']==m['fingerprint'] and r['mode']==mode,'Resume mismatch'
    pending=[]
    for row in rows:
        for mode in cfg['modes']:
            old=previous[mode].get(row['sample_id'])
            if not old or cfg['retry_failed'] and old['status']!='ok':pending.append((row,mode))
    state['processed']=state['total']-len(pending);update('initializing')
    if not pending:update('complete');return
    assets=json.loads(Path(cfg['work'],'assets_ready.json').read_text())
    try:
        experiment=APCExperiment(cfg,assets,update,sanity=a.smoke)
        handles={mode:(folder/f'predictions_{mode}.jsonl').open('a',encoding='utf-8') for mode in cfg['modes']}
        try:
            for index,(row,mode) in enumerate(pending,1):
                update('sample_start',sample_id=row['sample_id'],mode=mode,sample_started_unix=time.time())
                seed=int.from_bytes(hashlib.sha256(f"{cfg['seed']}:{row['sample_id']}".encode()).digest()[:4],'big')
                random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
                for gpu in (0,1):torch.cuda.reset_peak_memory_stats(gpu)
                scratch=Path(cfg['work'])/'scratch'/hashlib.sha256((mode+row['sample_id']).encode()).hexdigest()[:16];scratch.mkdir(parents=True,exist_ok=True)
                record=dict(sample_id=row['sample_id'],fingerprint=m['fingerprint'],mode=mode,status='error',prediction=None,raw_output='',seed=seed)
                fatal=None;t=time.monotonic()
                def timeout(signum,frame):raise TimeoutError('APC sample exceeded timeout')
                old_handler=signal.signal(signal.SIGALRM,timeout);signal.alarm(cfg['sample_timeout_seconds'])
                try:
                    raw=experiment.run(row,mode,scratch);prediction=parse_option(raw,row['choices'])
                    record.update(status='ok',raw_output=raw,prediction=prediction,parse_valid=prediction is not None,
                         output_issue='empty_output' if not raw else 'degenerate_exclamation' if re.fullmatch(r'!{4,}',raw) else 'unparsed_answer' if prediction is None else None)
                    if prediction is None:state['invalid']+=1
                    if a.smoke and prediction is None:fatal=ValueError('Smoke response was not an exact answer option')
                except Exception as exc:
                    state['errors']+=1;record.update(error_type=type(exc).__name__,message=str(exc));traceback.print_exc()
                    if a.smoke or is_fatal(exc):fatal=exc
                finally:
                    signal.alarm(0);signal.signal(signal.SIGALRM,old_handler)
                    record.update(seconds=time.monotonic()-t,vlm_calls=experiment.vlm.calls,
                       peak_vram_gib={str(i):torch.cuda.max_memory_allocated(i)/2**30 for i in (0,1)})
                    f=handles[mode];f.write(json.dumps(record,ensure_ascii=False)+'\n');f.flush()
                    if index%8==0 or fatal is not None or index==len(pending):os.fsync(f.fileno())
                    if a.smoke:
                        destination=folder/f'trace_{mode}'
                        if not destination.exists():shutil.copytree(scratch,destination)
                    resolved=scratch.resolve();resolved.relative_to((Path(cfg['work'])/'scratch').resolve());shutil.rmtree(resolved)
                state['processed']+=1;state['new_processed']+=1
                update('sample_done',last_output=record['raw_output'][:180],last_seconds=record['seconds'],sample_started_unix=None)
                if index<=3 or index%10==0:print(json.dumps(state),flush=True)
                if fatal:raise fatal
        finally:
            for f in handles.values():f.close()
        update('complete')
    except BaseException as exc:update('error',error_type=type(exc).__name__,message=str(exc));raise

if __name__=='__main__':
    import re
    main()
