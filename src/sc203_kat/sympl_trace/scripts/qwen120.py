"""FP16 split Qwen with exact prompts and processed inputs recorded by stage."""
import time
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
from sympl.vlms.vlm_qwenvl2_5 import ModelQwenVL2_5
from sympl7b_device_map import plan_device_map
import trace_runtime as tr

class KaggleQwen(ModelQwenVL2_5):
    def __init__(self, config, device='cuda:0'):
        self.calls=[];self.settings=config
        super().__init__(config,device)

    def load_model(self, config, device='cuda:0'):
        args=dict(torch_dtype=torch.float16, attn_implementation='sdpa',local_files_only=True,low_cpu_mem_usage=True)
        if config.get('quantization')=='4bit':
            args.update(device_map={'':0}, quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,bnb_4bit_compute_dtype=torch.float16,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True))
        else:args['device_map']=plan_device_map(config.pretrained_model)
        self.vlm_model,loading=Qwen2_5_VLForConditionalGeneration.from_pretrained(config.pretrained_model,output_loading_info=True,**args)
        if any(loading.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')):
            raise RuntimeError('Checkpoint loading mismatch: '+str(loading))
        self.vlm_model.eval()
        self.device=str(self.vlm_model.get_input_embeddings().weight.device)
        if any(str(d) in ('cpu','disk') for d in self.vlm_model.hf_device_map.values()):
            raise RuntimeError('Qwen CPU/disk offload is not configured for this experiment.')
        processor_args={}
        if config.get('max_pixels'):processor_args['max_pixels']=int(config.max_pixels)
        self.processor=AutoProcessor.from_pretrained(config.pretrained_model,local_files_only=True,**processor_args)

    def process_messages(self,messages,max_new_tokens=1024,do_sample=False,temperature=0.0):
        started=time.perf_counter();limit=min(max_new_tokens,self.settings.max_new_tokens)
        text=self.processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        images,videos=process_vision_info(messages)
        inputs=self.processor(text=[text],images=images,videos=videos,padding=True,return_tensors='pt')
        if inputs.input_ids.shape[1]>32768:raise RuntimeError('Context exceeds 32768 tokens')
        if tr.ACTIVE:
            tr.ACTIVE.event(phase='vlm_input',component='Qwen',parent=tr.ACTIVE.stack[-1],
                values=tr.ACTIVE.pack(dict(messages=messages,chat_template=text,processed_inputs=dict(inputs),
                                     max_new_tokens=limit,do_sample=do_sample,temperature=temperature)))
        call=dict(prompt=text,status='started',max_new_tokens=limit);self.calls.append(call)
        try:
            inputs=inputs.to(self.device)
            generated=self.vlm_model.generate(**inputs,max_new_tokens=limit,do_sample=do_sample,temperature=temperature)
            trimmed=generated[:,inputs.input_ids.shape[1]:]
            response=self.processor.batch_decode(trimmed,skip_special_tokens=True,clean_up_tokenization_spaces=False)[0]
            call.update(raw_output=response,status='complete',output_tokens=trimmed.shape[1],reached_token_limit=trimmed.shape[1]>=limit)
            if tr.ACTIVE:tr.ACTIVE.event(phase='vlm_output',component='Qwen',values=tr.ACTIVE.pack(call))
            return response
        except Exception as exc:
            call.update(status='error',error=repr(exc));raise
        finally:call['seconds']=time.perf_counter()-started
