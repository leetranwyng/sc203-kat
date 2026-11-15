"""Plan whole decoder layers for APC + FP16 Qwen on two Kaggle GPUs."""

GIB = 2**30


def validate_partition_map(device_map, tensor_names=None):
    """Reject mappings that make Accelerate recursively move a parent and its children."""
    if not device_map or "" in device_map:
        raise ValueError("A partitioned Qwen map must not contain the root key ''.")
    names = list(device_map)
    if any(device not in (0, 1) for device in device_map.values()):
        raise ValueError("The Qwen partition must use only GPU 0 and GPU 1.")
    for parent in names:
        for child in names:
            if child.startswith(parent + "."):
                raise ValueError(f"Overlapping device-map entries: {parent!r} and {child!r}.")
    if tensor_names is not None:
        for name in tensor_names:
            matches = [prefix for prefix in names if name == prefix or name.startswith(prefix + ".")]
            if len(matches) != 1:
                raise ValueError(f"Tensor {name!r} must have exactly one explicit device-map owner.")


def verify_loaded_partition(model, device_map):
    """Check real tensor devices after dispatch, including nonpersistent buffers."""
    tensors = list(model.named_parameters()) + list(model.named_buffers())
    validate_partition_map(device_map, [name for name, _ in tensors])
    byte_counts = [0, 0]
    for name, tensor in tensors:
        prefix = next(prefix for prefix in device_map if name == prefix or name.startswith(prefix + "."))
        expected = device_map[prefix]
        if tensor.device.type != "cuda" or tensor.device.index != expected:
            raise RuntimeError(f"Tensor {name} is on {tensor.device}; expected cuda:{expected}.")
        byte_counts[expected] += tensor.numel() * tensor.element_size()
    return {"verified": True, "tensor_count": len(tensors),
            "tensor_gib_by_gpu": [value / GIB for value in byte_counts]}


def plan_qwen_device_map(module_sizes, num_layers, free_bytes, *,
                         text_prefix="model.language_model", visual_prefix="model.visual",
                         activation_reserve_gib=(3.0, 2.0), vision_reserve_gib=6.0,
                         tie_word_embeddings=False):
    """Return an explicit GPU-only map, reserving GPU 1 for the APC vision stack.

    Unlike an automatic balanced map, this does not equalize Qwen weights across
    unequal GPU budgets. Decoder layers remain contiguous and never split.
    module_sizes comes from accelerate.utils.modeling.compute_module_sizes.
    """
    if num_layers < 2 or len(free_bytes) != 2 or len(activation_reserve_gib) != 2:
        raise ValueError("Expected at least two decoder layers and exactly two GPUs.")
    reserves = [float(value) * GIB for value in activation_reserve_gib]
    if min(reserves) < 0 or vision_reserve_gib < 0:
        raise ValueError("Memory reserves cannot be negative.")
    budgets = [int(free_bytes[0] - reserves[0]),
               int(free_bytes[1] - reserves[1] - vision_reserve_gib * GIB)]
    if min(budgets) <= 0:
        raise RuntimeError("Insufficient free GPU memory after reserving APC/activation space. "
                           "Stop old GPU workers or restart the Kaggle session.")
    layers = [f"{text_prefix}.layers.{index}" for index in range(num_layers)]
    embedding, norm = f"{text_prefix}.embed_tokens", f"{text_prefix}.norm"
    for name in ("", visual_prefix, embedding, norm, *layers):
        if name not in module_sizes:
            raise ValueError(f"Qwen architecture/size table is missing {name!r}.")
    if not tie_word_embeddings and "lm_head" not in module_sizes:
        raise ValueError("Untied Qwen output head is missing from the size table.")
    sizes = [int(module_sizes[name]) for name in layers]
    head_size = int(module_sizes.get("lm_head", 0))
    fixed_zero = int(module_sizes[visual_prefix] + module_sizes[embedding])
    fixed_one = int(module_sizes[norm])
    if tie_word_embeddings:
        fixed_zero += head_size
    else:
        fixed_one += head_size
    # The remaining small buffers belong to the explicitly mapped rotary module.
    remainder = int(module_sizes[""]) - fixed_zero - fixed_one - sum(sizes)
    if remainder < 0:
        raise ValueError("Overlapping/inconsistent module sizes in the Qwen memory plan.")
    fixed_zero += remainder
    candidates = []
    for cut in range(1, num_layers):
        required = [fixed_zero + sum(sizes[:cut]), fixed_one + sum(sizes[cut:])]
        if all(required[gpu] <= budgets[gpu] for gpu in (0, 1)):
            candidates.append((cut, required))
    if not candidates:
        raise RuntimeError(
            "FP16 Qwen + APC cannot fit the configured GPU reserves. "
            f"Qwen={module_sizes['']/GIB:.2f} GiB; weight budgets="
            f"{budgets[0]/GIB:.2f}/{budgets[1]/GIB:.2f} GiB. "
            "No CPU/disk offload or quantization was applied. Inspect memory_plan.json."
        )
    # Keep as much Qwen on GPU 0 as fits; GPU 1 needs space for vision inference.
    cut, required = max(candidates, key=lambda candidate: candidate[0])
    device_map = {visual_prefix: 0, embedding: 0,
                  f"{text_prefix}.rotary_emb": 0, norm: 1,
                  "lm_head": 0 if tie_word_embeddings else 1}
    device_map.update({name: 0 if index < cut else 1 for index, name in enumerate(layers)})
    validate_partition_map(device_map)
    report = {
        "strategy": "contiguous_nonoverlapping_decoder_layers_v2", "device_map": device_map,
        "layers_gpu0": cut, "layers_gpu1": num_layers - cut,
        "qwen_weights_gib": module_sizes[""] / GIB,
        "free_before_qwen_gib": [value / GIB for value in free_bytes],
        "weight_budget_gib": [value / GIB for value in budgets],
        "planned_weights_gib": [value / GIB for value in required],
        "activation_reserve_gib": list(activation_reserve_gib),
        "vision_reserve_gpu1_gib": vision_reserve_gib,
        "quantized": False, "cpu_or_disk_offload": False,
    }
    return device_map, report


def make_qwen_memory_plan(model_dir, dtype, *, activation_reserve_gib=(3.0, 2.0),
                          vision_reserve_gib=6.0):
    """Inspect the model on the meta device; allocate no real model weights."""
    import gc
    import torch
    from accelerate import init_empty_weights
    from accelerate.utils.modeling import check_device_map, compute_module_sizes
    from transformers import AutoConfig, Qwen2_5_VLForConditionalGeneration

    if torch.cuda.device_count() != 2:
        raise RuntimeError("Select Kaggle GPU T4 x2 before preparing the Qwen device map.")
    config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
    config._attn_implementation = "sdpa"
    with init_empty_weights(include_buffers=True):
        skeleton = Qwen2_5_VLForConditionalGeneration(config)
        skeleton.tie_weights()
    if hasattr(skeleton.model, "language_model"):
        text_prefix, visual_prefix = "model.language_model", "model.visual"
    else:
        text_prefix, visual_prefix = "model", "visual"
    text_model = skeleton.get_submodule(text_prefix)
    # Initialize each CUDA context before measuring usable memory.
    for gpu in (0, 1):
        with torch.cuda.device(gpu):
            torch.empty(1, device=f"cuda:{gpu}")
            torch.cuda.empty_cache()
    free_bytes = [torch.cuda.mem_get_info(gpu)[0] for gpu in (0, 1)]
    device_map, report = plan_qwen_device_map(
        compute_module_sizes(skeleton, dtype=dtype), len(text_model.layers), free_bytes,
        text_prefix=text_prefix, visual_prefix=visual_prefix,
        activation_reserve_gib=activation_reserve_gib, vision_reserve_gib=vision_reserve_gib,
        tie_word_embeddings=getattr(config.get_text_config(), "tie_word_embeddings", False),
    )
    validate_partition_map(device_map, [name for name, _ in skeleton.named_parameters()]
                           + [name for name, _ in skeleton.named_buffers()])
    check_device_map(skeleton, device_map)
    del text_model, skeleton
    gc.collect()
    return device_map, report


class VisionScheduler:
    """Keep one APC vision model on GPU 1 at a time, parking idle weights in RAM."""

    def __init__(self, models, predictor, device="cuda:1", torch_module=None):
        if torch_module is None:
            import torch as torch_module
        self.torch = torch_module
        self.models = models
        self.predictor = predictor
        self.device = device
        self.active = None

    def park_all(self):
        if self.active is None:
            return
        with self.torch.cuda.device(self.device):
            if self.active == "sam" and self.torch.is_tensor(self.predictor.features):
                self.predictor.features = self.predictor.features.to("cpu")
            self.models[self.active].to("cpu")
            self.active = None
            self.torch.cuda.empty_cache()

    def activate(self, name):
        if name == self.active:
            return
        self.park_all()
        with self.torch.cuda.device(self.device):
            self.models[name].to(self.device)
            if name == "sam" and self.torch.is_tensor(self.predictor.features):
                self.predictor.features = self.predictor.features.to(self.device)
        self.active = name


def install_vision_scheduler(apc, vlm, device="cuda:1"):
    """Install scheduling after initializing the auxiliary models on CPU.

    Model precision, inference, prompts, and thresholds are preserved. Moving the
    SAM cached embedding with its model allows VLM refinement calls between masks.
    """
    from functools import wraps
    from torchvision.transforms import Lambda

    detection, depth, orientation = apc.detection_module, apc.depth_module, apc.orientation_module
    predictor = detection.segmentation_predictor
    models = {"grounding": detection.detection_model, "sam": detection.segmentation_model,
              "depth": depth.depth_model, "orientation": orientation.orientation_model}
    scheduler = VisionScheduler(models, predictor, device)
    for module in (detection, depth, orientation):
        module.device = device
    apc.device_vision = device
    apc.render_module.device = device
    # Depth Pro closes over its initialization device in the preprocessing Lambda.
    # Retain the original GPU normalization/dtype order when weights start on CPU.
    transform = depth.depth_transform
    if not hasattr(transform, "transforms") or len(transform.transforms) < 4:
        raise RuntimeError("Unexpected Depth Pro preprocessing; inspect its upstream revision.")
    transform.transforms[1] = Lambda(lambda tensor: tensor.to(device))

    def wrap(owner, method_name, model_name):
        method = getattr(owner, method_name)

        @wraps(method)
        def scheduled(*args, **kwargs):
            scheduler.activate(model_name)
            return method(*args, **kwargs)

        setattr(owner, method_name, scheduled)

    wrap(detection, "run_detection", "grounding")
    for method in ("set_image", "predict", "predict_torch"):
        wrap(predictor, method, "sam")
    wrap(depth, "run_depth_estimation", "depth")
    wrap(orientation, "get_3angle", "orientation")
    vlm.before_generate = scheduler.park_all
    return scheduler
