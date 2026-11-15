import json
import torch
from huggingface_hub import hf_hub_download, snapshot_download, HfApi
MODEL_REVISION=HfApi().model_info(MODEL_ID,revision=MODEL_REVISION).sha

assert torch.cuda.is_available(), "Chọn Settings → Accelerator → GPU T4 x2."
assert torch.cuda.device_count() == 2, f"Cần đúng 2 GPU, thấy {torch.cuda.device_count()}."
for gpu in range(2):
    info = torch.cuda.get_device_properties(gpu)
    print(f"GPU {gpu}: {info.name}, {info.total_memory / 2**30:.1f} GiB")
print("One FP16 7B model spans two GPUs, with SDPA.")

input_root = Path("/kaggle/input")
if TSV_PATH is None:
    candidates = sorted(input_root.rglob(TSV_FILENAME)) if input_root.exists() else []
    if len(candidates) > 1:
        raise ValueError("Nhiều TSV được tìm thấy; đặt TSV_PATH rõ ràng.")
    if candidates:
        TSV_PATH = str(candidates[0])
    else:
        TSV_PATH = hf_hub_download(repo_id=DATASET_ID, filename=TSV_FILENAME,
                                   repo_type="dataset", revision=DATASET_REVISION)
if MODEL_DIR is None:
    candidates = []
    for path in input_root.rglob("config.json") if input_root.exists() else []:
        try:
            metadata = json.loads(path.read_text())
        except (ValueError, OSError):
            continue
        if (metadata.get("model_type") == "qwen2_5_vl"
                and metadata.get("hidden_size") == 3584
                and metadata.get("num_hidden_layers") == 28
                and list(path.parent.glob("*.safetensors"))):
            candidates.append(path.parent)
    if len(candidates) > 1:
        raise ValueError("Nhiều bản model được tìm thấy; đặt MODEL_DIR rõ ràng.")
    if candidates:
        MODEL_DIR = str(candidates[0])
    else:
        MODEL_DIR = snapshot_download(
            MODEL_ID, revision=MODEL_REVISION,
            allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja"],
            max_workers=2,
        )
model_path = Path(MODEL_DIR).resolve()
model_metadata = json.loads((model_path / "config.json").read_text())
assert (model_metadata.get("model_type"), model_metadata.get("hidden_size"),
        model_metadata.get("num_hidden_layers")) == ("qwen2_5_vl", 3584, 28), "Expected official Qwen2.5-VL-7B."
assert (model_path / "preprocessor_config.json").is_file(), "Thiếu processor."
assert (model_path / "tokenizer_config.json").is_file(), "Thiếu tokenizer."
index_path = model_path / "model.safetensors.index.json"
if index_path.exists():
    weight_names = set(json.loads(index_path.read_text())["weight_map"].values())
    assert all((model_path / name).is_file() for name in weight_names), "Thiếu weight shard."
else:
    assert (model_path / "model.safetensors").is_file(), "Thiếu model.safetensors hoặc shard index."
assert Path(TSV_PATH).is_file(), f"Không tìm thấy TSV: {TSV_PATH}"
print("Model directory:", model_path)
print("Circular TSV:", TSV_PATH)
import importlib.util

runner_path = Path(WORK_DIR)/"pcd_kaggle_runner.py"
reference_path = Path(WORK_DIR)/"pcd_reference.py"
spec = importlib.util.spec_from_file_location("pcd_kaggle_runner", runner_path)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
CONFIG = {
    "mode": MODE, "model_dir": str(model_path), "data_path": str(Path(TSV_PATH).resolve()),
    "model_id": MODEL_ID, "model_revision": MODEL_REVISION, "dataset_revision": DATASET_REVISION,
    "output_dir": str(OUTPUT_DIR), "reference_path": str(reference_path),
    "alpha": float(ALPHA), "num_workers": NUM_WORKERS, "seed": SEED,
    "compute_baseline": COMPUTE_BASELINE, "min_pixels": MIN_PIXELS, "max_pixels": MAX_PIXELS,
    "sanity_qid": SANITY_QID, "resume_from": str(RESUME_FROM) if RESUME_FROM else None,
}
selected_rows = runner.prepare_run(CONFIG)
print("Selected rows:", len(selected_rows))
for rank in range(NUM_WORKERS):
    plan = runner.read_json(OUTPUT_DIR / f"plan_gpu{rank}.json")
    print(f"GPU {rank}: {len(plan)} rows")
runner.launch_workers(CONFIG)
metrics,records=runner.export_results(CONFIG)
print(metrics.to_string(index=False))
if MODE=='sanity':
    for row in records:
        if row.get('diagnostic_camera_prediction') is None or row.get('diagnostic_contrastive_prediction') is None:
            raise RuntimeError('Sanity failed: a branch has no valid option.')
        if row.get('reverse_direction_prediction') is None:
            raise RuntimeError('Sanity failed: reversed-direction output is invalid.')
print('Done. Results and trace are ready for export.')

import zipfile
archive=OUTPUT_DIR.parent/('pcd_qwen7b_'+MODE+'_trace.zip')
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as bundle:
    for path in OUTPUT_DIR.rglob('*'):
        if path.is_file():bundle.write(path,path.relative_to(OUTPUT_DIR))
    bundle.write(reference_path,'pcd_reference.py')
    bundle.write(runner_path,'pcd_kaggle_runner.py')
print('Download:',archive)
