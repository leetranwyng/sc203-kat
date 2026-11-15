"""Qwen 7B interface compatible with the APC multi-call pipeline."""
import time
import torch
from transformers import Qwen2_5_VLForConditionalGeneration,AutoProcessor
from qwen_vl_utils import process_vision_info
from apc.vlms.vlm_base import VLMBase

class KaggleQwen(VLMBase):
    def __init__(self,config,device='cuda:0'):
        super().__init__(device);self.config=config;self.device=device;self.calls=[];self.callback=None
        self.load_model(config,device)
    def load_model(self, config, device='cuda:0'):
        from pathlib import Path
        import json
        from apc_kaggle_memory import make_qwen_memory_plan, verify_loaded_partition

        dtype = getattr(torch, config.dtype)
        if config.dtype == 'bfloat16' and not torch.cuda.is_bf16_supported(including_emulation=False):
            raise RuntimeError('Native BF16 unavailable on T4; select float16.')
        report_dir = Path(config.memory_report_dir)
        report_dir.mkdir(parents=True, exist_ok=True)
        try:
            device_map, report = make_qwen_memory_plan(
                config.pretrained_model, dtype,
                activation_reserve_gib=config.activation_reserve_gib,
                vision_reserve_gib=config.vision_reserve_gib,
            )
        except Exception as exc:
            (report_dir/'memory_plan.json').write_text(json.dumps(dict(error=str(exc)), indent=2))
            raise
        (report_dir/'memory_plan.json').write_text(json.dumps(report, indent=2))
        print('Qwen memory plan:', json.dumps({k:v for k,v in report.items() if k != 'device_map'}), flush=True)
        self.vlm_model, loading = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            config.pretrained_model, dtype=dtype, device_map=device_map,
            attn_implementation='sdpa', local_files_only=True,
            low_cpu_mem_usage=True, output_loading_info=True,
        )
        if any(loading.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')):
            raise RuntimeError('Qwen checkpoint did not load cleanly: '+str(loading))
        self.vlm_model.eval()
        self.dtype = dtype
        self.device = str(self.vlm_model.get_input_embeddings().weight.device)
        actual_map = self.vlm_model.hf_device_map
        (report_dir/'model_device_map.json').write_text(json.dumps(actual_map, default=str, indent=2))
        if any(str(value) not in ('0','1','cuda:0','cuda:1') for value in actual_map.values()):
            raise RuntimeError('Qwen must remain entirely on the two GPUs. Inspect model_device_map.json.')
        if any(parameter.device.type == 'meta' for parameter in self.vlm_model.parameters()):
            raise RuntimeError('Qwen contains unloaded meta parameters; stop before inference.')
        placement = verify_loaded_partition(self.vlm_model, device_map)
        placement['memory_allocated_gib'] = [torch.cuda.memory_allocated(i)/2**30 for i in (0,1)]
        (report_dir/'qwen_placement.json').write_text(json.dumps(placement, indent=2))
        print('Qwen actual placement PASS:', json.dumps(placement), flush=True)
        self.processor = AutoProcessor.from_pretrained(
            config.pretrained_model, local_files_only=True,
            **{key:getattr(config,key) for key in ('min_pixels','max_pixels') if getattr(config,key) is not None},
        )
    def generate_response(self,prompt,image=None,max_new_tokens=None,do_sample=False,temperature=0.0):
        content=([{'type':'image','image':image}] if image is not None else [])+[{'type':'text','text':prompt}]
        return self.process_messages([dict(role='user',content=content)],max_new_tokens,do_sample,temperature)
    def process_messages(self,messages,max_new_tokens=None,do_sample=False,temperature=0.0):
        if getattr(self, 'before_generate', None):self.before_generate()
        started=time.monotonic()
        import os,json
        from pathlib import Path
        folder=Path(os.environ['APC_TRACE_DIR'])/'vlm'
        folder.mkdir(exist_ok=True)
        index=len(self.calls)+1
        captured=[]
        for message_index,message in enumerate(messages):
            for content_index,content in enumerate(message.get('content',[])):
                if isinstance(content,dict) and content.get('type')=='image':
                    value=content['image']
                    image_file=folder/f'{index:03d}_{message_index}_{content_index}.png'
                    if hasattr(value,'save'):
                        value.save(image_file)
                        captured.append(image_file.name)
        # Save before generation so a failed call still has its prompt.

        if self.callback:self.callback('vlm_call',call=len(self.calls)+1)
        text=self.processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        has_image=any(isinstance(c,dict) and c.get('type')=='image' for m in messages for c in m.get('content',[]))
        images,videos=process_vision_info(messages) if has_image else (None,None)
        batch=self.processor(text=[text],images=images,videos=videos,padding=True,return_tensors='pt')
        length=int(batch['input_ids'].shape[-1])
        call_file=folder/f'{index:03d}.json'
        call_file.write_text(json.dumps(dict(prompt=text,images=captured,status='started',prompt_tokens=length),ensure_ascii=False))
        if length>self.config.max_input_tokens:raise ValueError('APC intermediate prompt exceeds max_input_tokens; no silent truncation.')
        inputs={k:v.to(device=self.device,dtype=self.dtype) if k in ('pixel_values','pixel_values_videos') else v.to(self.device) for k,v in batch.items()}
        last_text='\n'.join(c.get('text','') for c in messages[-1].get('content',[]) if c.get('type')=='text')
        is_final=last_text.lstrip().startswith('Here is the question and the response you provided:')
        cap=min(max_new_tokens or self.config.max_new_tokens,self.config.final_max_new_tokens if is_final else self.config.max_new_tokens)
        tokenizer=self.processor.tokenizer;pad=tokenizer.pad_token_id
        if pad is None:pad=tokenizer.eos_token_id
        with torch.inference_mode():ids=self.vlm_model.generate(**inputs,do_sample=False,num_beams=1,max_new_tokens=cap,use_cache=True,pad_token_id=pad)
        new=ids[:,length:];raw=self.processor.batch_decode(new,skip_special_tokens=True,clean_up_tokenization_spaces=False)[0].strip()
        call=dict(prompt_tokens=length,generated_tokens=int(new.shape[-1]),seconds=time.monotonic()-started,raw_output=raw)
        call.update(images=captured,status='complete')
        call_file.write_text(json.dumps(dict(call,prompt=text),ensure_ascii=False))
        self.calls.append(call)
        if self.callback:self.callback('vlm_return',call=len(self.calls),vlm_seconds=call['seconds'])
        return raw
