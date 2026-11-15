"""Qwen 3B interface compatible with the APC multi-call pipeline."""
import time
import torch
from transformers import Qwen2_5_VLForConditionalGeneration,AutoProcessor
from qwen_vl_utils import process_vision_info
from apc.vlms.vlm_base import VLMBase

class KaggleQwen(VLMBase):
    def __init__(self,config,device='cuda:0'):
        super().__init__(device);self.config=config;self.device=device;self.calls=[];self.callback=None
        self.load_model(config,device)
    def load_model(self,config,device='cuda:0'):
        dtype=getattr(torch,config.dtype)
        if config.dtype=='bfloat16' and not torch.cuda.is_bf16_supported(including_emulation=False):raise RuntimeError('Native BF16 unavailable on T4; select float16 and run sanity.')
        self.vlm_model,loading=Qwen2_5_VLForConditionalGeneration.from_pretrained(config.pretrained_model,torch_dtype=dtype,
          device_map={'':device},attn_implementation='sdpa',local_files_only=True,output_loading_info=True)
        if any(loading.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')):raise RuntimeError('Qwen checkpoint did not load cleanly: '+str(loading))
        self.vlm_model.eval();self.dtype=dtype
        self.processor=AutoProcessor.from_pretrained(config.pretrained_model,min_pixels=config.min_pixels,max_pixels=config.max_pixels,local_files_only=True)
    def generate_response(self,prompt,image=None,max_new_tokens=None,do_sample=False,temperature=0.0):
        content=([{'type':'image','image':image}] if image is not None else [])+[{'type':'text','text':prompt}]
        return self.process_messages([dict(role='user',content=content)],max_new_tokens,do_sample,temperature)
    def process_messages(self,messages,max_new_tokens=None,do_sample=False,temperature=0.0):
        started=time.monotonic()
        if self.callback:self.callback('vlm_call',call=len(self.calls)+1)
        text=self.processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        has_image=any(isinstance(c,dict) and c.get('type')=='image' for m in messages for c in m.get('content',[]))
        images,videos=process_vision_info(messages) if has_image else (None,None)
        batch=self.processor(text=[text],images=images,videos=videos,padding=True,return_tensors='pt')
        length=int(batch['input_ids'].shape[-1])
        if length>self.config.max_input_tokens:raise ValueError('APC intermediate prompt exceeds max_input_tokens; no silent truncation.')
        inputs={k:v.to(device=self.device,dtype=self.dtype) if k in ('pixel_values','pixel_values_videos') else v.to(self.device) for k,v in batch.items()}
        last_text='\n'.join(c.get('text','') for c in messages[-1].get('content',[]) if c.get('type')=='text')
        is_final=last_text.lstrip().startswith('Here is the question and the response you provided:')
        cap=min(max_new_tokens or self.config.max_new_tokens,self.config.final_max_new_tokens if is_final else self.config.max_new_tokens)
        tokenizer=self.processor.tokenizer;pad=tokenizer.pad_token_id
        if pad is None:pad=tokenizer.eos_token_id
        with torch.inference_mode():ids=self.vlm_model.generate(**inputs,do_sample=False,num_beams=1,max_new_tokens=cap,use_cache=True,pad_token_id=pad)
        new=ids[:,length:];raw=self.processor.batch_decode(new,skip_special_tokens=True,clean_up_tokenization_spaces=False)[0].strip()
        call=dict(prompt_tokens=length,generated_tokens=int(new.shape[-1]),seconds=time.monotonic()-started,raw_output=raw[:240])
        self.calls.append(call)
        if self.callback:self.callback('vlm_return',call=len(self.calls),vlm_seconds=call['seconds'])
        return raw
