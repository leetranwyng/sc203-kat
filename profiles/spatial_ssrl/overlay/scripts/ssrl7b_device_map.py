"""Choose a contiguous Qwen split using actual FP16 module sizes."""
def choose_split(sizes, language_prefix, visual_prefix, layer_count, budgets):
    embedding = language_prefix + '.embed_tokens'
    norm = language_prefix + '.norm'
    rotary = language_prefix + '.rotary_emb'
    layers = [language_prefix + '.layers.' + str(i) for i in range(layer_count)]
    fixed = {visual_prefix: 0, embedding: 0, norm: 1, rotary: 1, 'lm_head': 1}
    candidates = []
    for cut in range(1, layer_count):
        placement = dict(fixed)
        placement.update({name: 0 if i < cut else 1 for i, name in enumerate(layers)})
        usage = {gpu: sum(sizes.get(name, 0) for name, owner in placement.items() if owner == gpu)
                 for gpu in (0, 1)}
        if all(usage[gpu] <= budgets[gpu] for gpu in (0, 1)):
            load = max(usage[gpu] / budgets[gpu] for gpu in (0, 1))
            candidates.append((load, cut, placement, usage))
    if not candidates:
        raise RuntimeError('No contiguous FP16 split fits the reserved GPU budgets. '
                           'Free GPU memory or increase the hardware capacity.')
    _, cut, placement, usage = min(candidates, key=lambda item: item[0])
    return placement, usage, cut


def plan_device_map(checkpoint):
    import gc
    import torch
    from accelerate import init_empty_weights
    from accelerate.utils import compute_module_sizes
    from transformers import AutoConfig, Qwen2_5_VLForConditionalGeneration

    config = AutoConfig.from_pretrained(checkpoint, local_files_only=True)
    if config.tie_word_embeddings:
        raise RuntimeError('This split expects the official untied Qwen2.5-VL-7B checkpoint.')
    config._attn_implementation = 'sdpa'
    with init_empty_weights():
        empty = Qwen2_5_VLForConditionalGeneration(config)
    names = dict(empty.named_modules())
    language = 'model.language_model' if 'model.language_model' in names else 'model'
    visual = 'model.visual' if 'model.visual' in names else 'visual'
    count = len(names[language+'.layers'])
    sizes = compute_module_sizes(empty, dtype=torch.float16)
    gib = 2**30
    free = {gpu: torch.cuda.mem_get_info(gpu)[0] for gpu in (0, 1)}
    # Reserve activation and KV-cache memory on both GPUs.
    budgets = {0: min(10.5*gib, free[0]-3*gib),
               1: min(10.5*gib, free[1]-3*gib)}
    placement, usage, cut = choose_split(sizes, language, visual, count, budgets)
    for name, _ in list(empty.named_parameters()) + list(empty.named_buffers()):
        if not any(name == prefix or name.startswith(prefix+'.') for prefix in placement):
            raise RuntimeError('Unmapped Qwen tensor: '+name)
    print('Qwen FP16 planned weights GiB:', {gpu: round(value/gib, 3) for gpu,value in usage.items()},
          'layer split:', cut, '/', count, flush=True)
    del empty
    gc.collect()
    return placement
