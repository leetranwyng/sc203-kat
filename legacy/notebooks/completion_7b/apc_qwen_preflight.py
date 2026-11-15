"""Check the real Accelerate dispatch path before compiling APC's native dependencies."""

import json
import sys
from pathlib import Path


def probe_dispatch(device_map):
    import gc
    import torch
    from accelerate import dispatch_model
    from apc_kaggle_memory import validate_partition_map, verify_loaded_partition

    validate_partition_map(device_map)

    class Probe(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.order = list(device_map)
            for name in self.order:
                parent = self
                parts = name.split(".")
                for part in parts[:-1]:
                    if not hasattr(parent, part):
                        parent.add_module(part, torch.nn.Module())
                    parent = getattr(parent, part)
                leaf = torch.nn.Linear(4, 4)
                with torch.no_grad():
                    leaf.weight.copy_(torch.eye(4))
                    leaf.bias.zero_()
                leaf.register_buffer("probe_buffer", torch.ones(1), persistent=False)
                parent.add_module(parts[-1], leaf)

        def forward(self, value):
            for name in self.order:
                value = self.get_submodule(name)(value)
            return value

    # The same module paths/map as Qwen, but only a few KiB of actual weights.
    model = dispatch_model(Probe().eval(), device_map=device_map)
    result = verify_loaded_partition(model, device_map)
    with torch.inference_mode():
        value = torch.ones((1, 4), device="cuda:0")
        output = model(value)
        if not torch.allclose(output, value.to(output.device)):
            raise RuntimeError("Accelerate cross-GPU dispatch probe returned an incorrect value.")
    result.update(cross_gpu_forward=True, model="tiny_identity_probe", gpu_names=[
        torch.cuda.get_device_name(index) for index in (0, 1)])
    del model, value, output
    gc.collect()
    for index in (0, 1):
        with torch.cuda.device(index):
            torch.cuda.empty_cache()
    return result


def main(config_path):
    import torch
    from huggingface_hub import hf_hub_download
    from apc_kaggle_memory import make_qwen_memory_plan

    config = json.loads(Path(config_path).read_text())
    work = Path(config["work"])
    model_dir = Path(config["asset_input"] or config["asset_root"]) / "qwen_model"
    if not (model_dir / "config.json").is_file():
        if config["asset_input"]:
            raise RuntimeError("The mounted asset_input is missing qwen_model/config.json.")
        hf_hub_download(config["model_id"], "config.json", revision=config["model_revision"],
                        local_dir=str(model_dir))
    device_map, report = make_qwen_memory_plan(
        str(model_dir), getattr(torch, config["dtype"]),
        activation_reserve_gib=config["vlm_activation_reserve_gib"],
        vision_reserve_gib=config["vision_reserve_gib"],
    )
    report["dispatch_probe"] = probe_dispatch(device_map)
    (work / "dispatch_preflight.json").write_text(json.dumps(report, indent=2))
    print("PASS: Qwen map has no root/parent overlaps; real Accelerate dispatch and "
          "a tiny forward succeeded on both GPUs.", flush=True)
    print(json.dumps({key: value for key, value in report.items() if key != "device_map"}), flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
