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
        config.vlm.memory_report_dir=cfg['work']
        config.vlm.activation_reserve_gib=cfg['vlm_activation_reserve_gib']
        config.vlm.vision_reserve_gib=cfg['vision_reserve_gib']
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
        self.callback('loading_apc_pipeline')
        vision_initial_device='cpu' if cfg['vision_offload'] else 'cuda:1'
        self.apc=apc_pipeline.APC(config,device_vlm='cuda:0',device_vision=vision_initial_device)
        self.vlm=self.apc.vlm_model;self.vlm.callback=self.callback
        if cfg['vision_offload']:
            from apc_kaggle_memory import install_vision_scheduler
            self.vision_scheduler=install_vision_scheduler(self.apc,self.vlm,'cuda:1')
        memory={str(i):dict(free_gib=torch.cuda.mem_get_info(i)[0]/2**30, allocated_gib=torch.cuda.memory_allocated(i)/2**30) for i in (0,1)}
        atomic_json(Path(cfg['work'])/'pipeline_memory.json',memory)
        print('APC loaded, GPU memory:',json.dumps(memory),flush=True)
        # Use the repository option-selection prompt.
        self.apc.logger.setLevel(logging.INFO if sanity else logging.WARNING)
        # Compute depth only once per sample even when several objects are abstracted.
        original_depth=self.apc.depth_module.run_depth_estimation
        self.depth_cache=None
        def depth_once(image):
            if self.depth_cache is None:
                self.depth_cache=original_depth(image)
                trace=Path(os.environ['APC_TRACE_DIR'])
                np.save(trace/'depth.npy',self.depth_cache)
                finite=self.depth_cache[np.isfinite(self.depth_cache)]
                low,high=np.percentile(finite,[2,98]) if finite.size else (0,1)
                preview=np.clip((np.nan_to_num(self.depth_cache)-low)/max(high-low,1e-6),0,1)
                Image.fromarray((preview*255).astype('uint8')).save(trace/'depth_preview.png')
                (trace/'depth_metadata.json').write_text(json.dumps(dict(shape=list(self.depth_cache.shape),preview_percentiles=[float(low),float(high)])))
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
                trace=Path(os.environ['APC_TRACE_DIR'])
                (trace/f'{_name}.json').write_text(json.dumps(dict(result=result),default=lambda x:x.tolist() if isinstance(x,np.ndarray) else str(x),ensure_ascii=False))
                return result
            setattr(self.apc,name,timed)
        original_render=self.apc.render_module.render_visual_prompt
        def render_prompt(*args,**kw):
            self.callback('render_visual_prompt')
            result=original_render(*args,**kw)
            if True:result.save(Path(kw['trace_save_dir'])/'apc_visual_prompt.png')
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
            image.save(scratch/'input.png')
            with torch.inference_mode():
                if mode=='direct':raw=self.vlm.generate_response(question+'\nAnswer with the option letter only.',image,max_new_tokens=self.cfg['final_max_new_tokens'])
                elif mode=='apc_vis':
                    raw,_=self.apc.run_apc(image,question,trace_save_dir=str(scratch),perspective_prompt_type='visual',
                         visualize_trace=False,visualize_scene_abstraction=False,
                         render_whole_scene=row['category']=='orientation_on_the_left',return_conv_history=False,options=options)
                else:raise ValueError(mode)
        if not isinstance(raw,str):raise TypeError('APC response must be text')
        return raw
