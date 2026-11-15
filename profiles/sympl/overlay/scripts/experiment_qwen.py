"""Original SymPL Qwen interface with FP16 two-GPU loading and call logging."""
import json
import time
from pathlib import Path
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from sympl.vlms.vlm_qwenvl2_5 import ModelQwenVL2_5
from sympl7b_device_map import plan_device_map

class KaggleQwen(ModelQwenVL2_5):
    def __init__(self, config, device='cuda:0'):
        self.calls=[]
        super().__init__(config,device)

    def load_model(self,config,device='cuda:0'):
        placement=plan_device_map(config.pretrained_model)
        self.vlm_model,loading=Qwen2_5_VLForConditionalGeneration.from_pretrained(
            config.pretrained_model, torch_dtype=torch.float16, device_map=placement,
            attn_implementation='sdpa',
            local_files_only=True,low_cpu_mem_usage=True,output_loading_info=True)
        if any(loading.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')):
            raise RuntimeError('Checkpoint loading mismatch: '+str(loading))
        self.vlm_model.eval()
        self.device=str(self.vlm_model.get_input_embeddings().weight.device)
        placement=self.vlm_model.hf_device_map
        print('Qwen 7B device map:',placement,flush=True)
        if any(str(device) in ('cpu','disk') for device in placement.values()):
            raise RuntimeError('7B does not fit the configured GPU memory budget.')
        self.processor=AutoProcessor.from_pretrained(config.pretrained_model,local_files_only=True)

    def process_messages(self,messages,max_new_tokens=1024,do_sample=False,temperature=0.0):
        started=time.perf_counter()
        call=dict(prompt=self.processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True),
                  max_new_tokens=max_new_tokens,do_sample=do_sample,temperature=temperature,status='started')
        self.calls.append(call)
        try:
            result=super().process_messages(messages,max_new_tokens=max_new_tokens,
                                           do_sample=do_sample,temperature=temperature)
            call.update(raw_output=result,status='complete')
            return result
        except Exception as error:
            call.update(error=repr(error),status='error')
            raise
        finally:
            call['seconds']=time.perf_counter()-started
