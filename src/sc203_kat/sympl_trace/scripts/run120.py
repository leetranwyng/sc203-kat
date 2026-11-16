"""Sequential resumable 120-case runner, retaining every attempt."""
import functools, gc, json, random, shutil, sys, time, traceback
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from box import Box
import trace_runtime as tr
from trace_runtime import atomic_json
from apc_common import parse_option

def option(raw,row):
    value=str(raw or '').strip()
    if row['paper_task']=='visibility':value={'yes, visible':'yes','not':'no'}.get(value.lower(),value)
    if row['paper_task']=='left_right':value={'left':'on the left','right':'on the right'}.get(value.lower(),value)
    return parse_option(value,row['choices'])

def configure(cfg,out):
    import yaml
    import sympl.sympl_pipeline as pipe
    import sympl.projection as projection
    import sympl.prompt_extraction as extraction
    import sympl.bipartition as bip
    import sympl.localization as loc
    import sympl.vision_modules.detection as det
    from qwen120 import KaggleQwen
    config=Box(yaml.safe_load(Path('sympl/configs/qwenvl2_5_7b_instruct.yaml').read_text()))
    config.vlm.pretrained_model=cfg['weights'];config.vlm.quantization=cfg['quantization']
    config.vlm.max_pixels=cfg['max_pixels'];config.vlm.max_new_tokens=cfg['max_new_tokens']
    assets=Path(cfg['assets'])
    config.detection.ckpt_path=str(assets/'checkpoints/groundingdino_swint_ogc.pth')
    config.depth.ckpt_path=str(assets/'checkpoints/depth_pro.pt')
    config.orientation.ckpt_path=str(assets);config.orientation.model_path=str(assets/'checkpoints/orient_dino_weight.pt')
    config.orientation.processor_path=str(assets/'dinov2_large')
    pipe.get_vlm_model_class=lambda _:KaggleQwen
    model=pipe.SymPL(config,device_vlm='cuda:0',device_vision='cpu')
    model.device_vision='cuda:1'
    def offload(owner,method,attr):
        original=getattr(owner,method)
        @functools.wraps(original)
        def run(*a,**kw):
            weights=getattr(owner,attr);owner.device='cuda:1';weights.to('cuda:1')
            try:
                with torch.cuda.device(1):return original(*a,**kw)
            finally:
                weights.to('cpu');owner.device='cpu';gc.collect();torch.cuda.empty_cache()
        setattr(owner,method,run)
    offload(model.detection_module,'run_detection','detection_model')
    offload(model.depth_module,'run_depth_intrinsic_estimation','depth_model')
    offload(model.orientation_module,'get_3angle','orientation_model')
    hook=tr.Trace(out,out/'initialization').hook
    for name in ('get_objects_of_interest','get_reference_viewer','select_category_from_question',
                 'estimate_scene_geometry','top_view_position','do_perspective_change','do_projection_2d',
                 'do_abstraction','do_bipartition','do_localization','symbolic_layout_prompting'):hook(pipe,name)
    hook(extraction,'get_objects_of_interest','retry_entity_extraction')
    for name in ('calculate_2d_coordinates','transform_src_to_tgt','calc_Rt','transform_tgt_to_top'):hook(projection,name)
    hook(bip,'get_bipartition_parameter');hook(loc,'render_final_image')
    for owner,names in [(model.detection_module,('detection_process_image','run_detection','run_detection_refinement')),
                        (model.depth_module,('run_depth_intrinsic_estimation','unproject_to_3D_from_bbox')),
                        (model.orientation_module,('run_orientation_estimation','get_3angle','orientation_to_front_direction')),
                        (model.vlm_model,('process_messages',))]:
        for name in names:hook(owner,name)
    predict=det.predict
    def logged_predict(*a,**kw):
        boxes,scores,phrases=predict(*a,**kw)
        tr.snapshot('detector_candidates',dict(boxes=boxes,scores=scores,phrases=phrases,coordinates='normalized cxcywh'))
        return boxes,scores,phrases
    det.predict=logged_predict
    effective=dict(config=config.to_dict(),device_map=model.vlm_model.vlm_model.hf_device_map,
        processor=model.vlm_model.processor.image_processor.to_dict(),vision_offload=True,
        category_policy=cfg['category_policy'],sam_called=False,assets=json.loads((out/'assets_ready.json').read_text()))
    p=out/'effective_runtime.json'
    if p.exists():assert json.loads(p.read_text())['assets']['snapshots']==effective['assets']['snapshots'],'Asset revisions changed'
    atomic_json(p,effective)
    return model

def main(config_path):
    cfg=json.loads(Path(config_path).read_text());out=Path(cfg['output'])
    fp=json.loads((out/'manifest.json').read_text())['fingerprint']
    random.seed(cfg['seed']);np.random.seed(cfg['seed']);torch.manual_seed(cfg['seed']);torch.cuda.manual_seed_all(cfg['seed'])
    rows=[json.loads(x) for x in (out/'samples.jsonl').read_text(encoding='utf8').splitlines()]
    model=configure(cfg,out);errors=0
    for index,row in enumerate(rows):
        base=out/'traces'/str(row['sample_id']);base.mkdir(parents=True,exist_ok=True);result_path=base/'result.json'
        if result_path.exists():
            old=json.loads(result_path.read_text());assert old['fingerprint']==fp,'Resume identity mismatch'
            if old.get('trace_closed') and not (cfg.get('retry_errors') and old['status'] in ('error','method_failure','invalid')):
                from analyze120 import missing_refs
                if not missing_refs(out,out/old['attempt_path']):continue
        if shutil.disk_usage(out).free<cfg['min_free_gib']*2**30:raise RuntimeError('Disk reserve reached; checkpoint retained')
        attempt=max([int(p.name.split('_')[-1]) for p in base.glob('attempt_*')]+[0])+1
        active=base/f'attempt_{attempt:03d}';tr.ACTIVE=tr.Trace(out,active)
        sample_seed=(cfg['seed']+int(row['sample_id'])) % (2**31)
        random.seed(sample_seed);np.random.seed(sample_seed);torch.manual_seed(sample_seed);torch.cuda.manual_seed_all(sample_seed)
        result={**row,'fingerprint':fp,'attempt':attempt,'attempt_path':str(active.relative_to(out)),
                'prediction':None,'raw_output':None,'status':'started','trace_closed':False,'seed':sample_seed}
        atomic_json(active/'sample.json',row)
        start=time.perf_counter();model.vlm_model.calls.clear();fatal=False
        for gpu in (0,1):torch.cuda.reset_peak_memory_stats(gpu)
        try:
            image=Image.open(row['image_path']).convert('RGB');image.save(active/'input.png')
            tr.ACTIVE.event(phase='category_policy',component='category',category=row['paper_task'],source=cfg['category_policy'])
            with torch.inference_mode():
                raw,_=model.run_sympl(image,row['question'],
                    reasoning_category=row['paper_task'] if cfg['category_policy']=='dataset_mapping' else None,
                    trace_save_dir=str(active),visualize_trace=False,return_conv_history=False)
            pred=option(raw,row)
            result.update(raw_output=raw,prediction=pred,status='method_failure' if str(raw).lower().startswith('fail') else 'ok' if pred else 'invalid')
        except Exception as exc:
            result.update(status='error',error=repr(exc),traceback=traceback.format_exc(),failed_stage=tr.ACTIVE.last_error)
            print(f"Sample {row['sample_id']}, stage={tr.ACTIVE.last_error}: {exc!r}",flush=True)
            traceback.print_exc()
            errors+=1;fatal=isinstance(exc,(torch.cuda.OutOfMemoryError,OSError)) or 'CUDA error' in str(exc)
        finally:
            result.update(seconds=time.perf_counter()-start,correct=result['prediction']==row['answer'],
                generation=model.vlm_model.calls,trace_closed=True,
                peak_vram_gib=[torch.cuda.max_memory_allocated(g)/2**30 for g in (0,1)])
            tr.ACTIVE.event(phase='sample_end',component='runner',status=result['status'])
            atomic_json(active/'result.json',result);atomic_json(result_path,result);tr.ACTIVE=None
        from analyze120 import analyze_one
        try:analyze_one(out,result)
        except Exception as exc:atomic_json(active/'analysis_error.json',dict(error=repr(exc),traceback=traceback.format_exc()))
        print(f"[{index+1}/{len(rows)}] {row['sample_id']} {result['status']} correct={result['correct']} {result['seconds']:.1f}s",flush=True)
        if fatal or errors>=cfg['max_errors']:raise RuntimeError('Stopped after resource/runtime error; trace retained for resume')

if __name__=='__main__':main(sys.argv[1])
