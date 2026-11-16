"""Isolated auxiliary-model processes; no Qwen weights are resident here."""
import argparse
import contextlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
from .common import read, write, oa2_direction, unpad_oneref

def sam3_model(cfg):
    import torch
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    model = build_sam3_image_model(checkpoint_path=cfg['weights']['sam3'], load_from_HF=False,
                                  device='cuda', eval_mode=True, compile=False,
                                  enable_inst_interactivity=False)
    processor = Sam3Processor(model, device='cuda', confidence_threshold=cfg['sam3_threshold'])
    def infer(image, text, folder):
        import numpy as np
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
            state = processor.set_image(image)
            output = processor.set_text_prompt(state=state, prompt=text)
        boxes = output['boxes'].float().cpu().numpy()
        scores = output['scores'].float().cpu().numpy().reshape(-1)
        masks = output['masks'].cpu().numpy().reshape(-1,image.height,image.width)
        if not np.isfinite(boxes).all() or not np.isfinite(scores).all():
            raise FloatingPointError('Nonfinite SAM 3 output')
        order = np.argsort(-scores, kind='stable')
        from PIL import Image, ImageDraw
        for j,k in enumerate(order): Image.fromarray(masks[k].astype('uint8')*255).save(folder/f'mask_{j}.png')
        overlay=image.copy();draw=ImageDraw.Draw(overlay)
        for j,k in enumerate(order):
            draw.rectangle(boxes[k].tolist(),outline='orange',width=2)
            draw.text(tuple(boxes[k][:2]),f'{j}: {scores[k]:.3f}',fill='orange')
        overlay.save(folder/'candidates.png')
        return dict(boxes=boxes[order].tolist(),scores=scores[order].tolist(),
                    retained_candidates=min(5,len(order)),box_format='xyxy_absolute',
                    precision='fp32_weights_fp16_autocast',text=text)
    return infer

def oneref_architecture(tokenizer_path):
    from types import SimpleNamespace
    # Import the registered inference model only. eval.py overrides CUDA_VISIBLE_DEVICES=7.
    from models.OneRef_model import beit3_base_patch16_384_grounding
    args=SimpleNamespace(frozen_backbone=False,max_query_len=64,sentencepiece_model=tokenizer_path,
        enable_ref_mlm=False,text_mask_prob=.4,use_contrastive_loss=False,use_mask_loss=False,
        enable_ref_mim=False,mim_mid_layer=0,shared_lm_head=True)
    model=beit3_base_patch16_384_grounding(args,pretrained=False,drop_path_rate=.1,vocab_size=64010,
                                         checkpoint_activations=None)
    return model

def oneref_model(cfg):
    import torch
    import numpy as np
    from torchvision.transforms import functional as F
    model=oneref_architecture(cfg['weights']['oneref_tokenizer'])
    ckpt=torch.load(cfg['weights']['oneref'],map_location='cpu')
    state=ckpt.get('model',ckpt)
    missing, unexpected=model.load_state_dict(state,strict=False)
    # Pretraining-only heads may be in a checkpoint; never accept missing inference weights.
    allowed=('mlm_head.','mlm_sts_head.','mim_head.','mim_vts_head.')
    write(Path(cfg['output_dir'])/'oneref_loading.json',dict(missing=missing,unexpected=unexpected))
    if missing or any(not k.startswith(allowed) for k in unexpected):
        raise RuntimeError('OneRef checkpoint mismatch; inspect oneref_loading.json')
    del ckpt,state
    model.eval().to('cuda')
    def infer(image,text,folder):
        size=384; ratio=size/max(image.size); nw,nh=round(image.width*ratio),round(image.height*ratio)
        left,top=round((size-nw)/2-.1),round((size-nh)/2-.1)
        x=F.normalize(F.to_tensor(F.resize(image,[nh,nw])),[.485,.456,.406],[.229,.224,.225])
        padded=torch.zeros(3,size,size);padded[:,top:top+nh,left:left+nw]=x
        mask=torch.ones(size,size,dtype=torch.int);mask[top:top+nh,left:left+nw]=0
        with torch.inference_mode():
            output=model(image=padded[None].cuda(),img_mask=mask[None].cuda(),text=[text.lower()],training=False)
        raw=output[0][0].float().cpu().tolist()
        box=unpad_oneref(raw,image.width,image.height)
        if not np.isfinite(box).all() or box[2]<=box[0] or box[3]<=box[1]:
            raise ValueError('Invalid OneRef box after removing letterbox padding')
        from PIL import ImageDraw
        overlay=image.copy();ImageDraw.Draw(overlay).rectangle(box,outline='cyan',width=3)
        overlay.save(folder/'grounding.png')
        return dict(box=box,raw_box_cxcywh=raw,text=text,box_format='xyxy_absolute',
                    resize_ratio=ratio,padding_left_top=[left,top],precision='float32')
    return infer

def oa2_model(cfg):
    import torch
    import numpy as np
    from vision_tower import VGGT_OriAny_Ref
    from utils.app_utils import preprocess_images, val_fit_alpha
    model=VGGT_OriAny_Ref(out_dim=900,dtype=torch.float16,nopretrain=True)
    state=torch.load(cfg['weights']['oa2'],map_location='cpu',weights_only=True)
    model.load_state_dict(state,strict=True);del state
    model.eval().to('cuda')
    def infer(image,text,folder):
        # Same official pad preprocessing, no background-removal model introduced.
        with torch.inference_mode():
            batch=preprocess_images([image],mode='pad').to('cuda')[None]
            logits=model(batch).reshape(-1).float().cpu()
        if logits.numel()!=900 or not torch.isfinite(logits).all():
            raise FloatingPointError('Invalid OA V2 logits')
        np.save(folder/'logits.npy',logits.numpy())
        angles=[int(logits[:360].argmax()),int(logits[360:540].argmax())-90,int(logits[540:900].argmax())-180]
        distribution=logits[:360].sigmoid().numpy()
        alpha=float(val_fit_alpha(distribution[None].copy())[0])
        return dict(angles=angles,direction_vector=oa2_direction(*angles),symmetry_alpha=alpha,
            precision='fp32_weights_fp16_autocast',crop_policy='same_tight_box_then_official_pad518',
            convention='Rx(roll)Ry(elevation)Rz(-azimuth); semantic +X; APC=(y,-z,-x)',
            symmetry_policy='argmax single direction; all logits and alpha retained; no GT selection')
    return infer

def run(config,kind):
    cfg=read(config)
    repo=Path(cfg['repos'][kind]);os.chdir(repo);sys.path.insert(0,str(repo))
    import torch
    from PIL import Image
    jobs=read(Path(cfg['output_dir'])/f'jobs_{kind}.json')
    pending=[j for j in jobs if not (Path(j['output'])/'result.json').exists() or
             (cfg.get('retry_errors') and read(Path(j['output'])/'result.json').get('status')!='ok')]
    if not pending: return
    infer={'sam3':sam3_model,'oneref':oneref_model,'oa2':oa2_model}[kind](cfg)
    for j in pending:
        folder=Path(j['output']);folder.mkdir(parents=True,exist_ok=True)
        started=time.monotonic();torch.cuda.reset_peak_memory_stats()
        result=dict(job=j,status='error')
        try:
            image=Image.open(j['image']).convert('RGB')
            if j.get('bbox') is not None:image=image.crop(j['bbox'])
            image.save(folder/'input.png')
            result.update(infer(image,j['text'],folder),status='ok')
        except Exception as exc:
            result.update(error_type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
        result.update(seconds=time.monotonic()-started,peak_vram_gib=torch.cuda.max_memory_allocated()/2**30)
        write(folder/'result.json',result)
        print(kind,j['key'],result['status'],round(result['seconds'],2),flush=True)
        if result.get('error_type') in ('OutOfMemoryError',) or 'device-side assert' in result.get('message',''):
            raise RuntimeError('GPU failure recorded; restart worker before retrying')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('kind',choices=['sam3','oneref','oa2'])
    a=p.parse_args();run(a.config,a.kind)
