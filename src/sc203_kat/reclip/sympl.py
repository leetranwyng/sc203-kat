"""SymPL paired selector replay from frozen historical inputs."""
import argparse
from collections import Counter, defaultdict
import copy
import gc
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import random
import re
import shutil
import subprocess
import sys
import time
import traceback
import zipfile

from sc203_kat.ablation.common import inside, read, rows, sha, write

TASKS = {"left_right": 40, "visibility": 40, "facing": 40}
ROOT = Path(__file__).resolve().parents[3]


def write_rows(path, records):
    tmp = path.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
                           for row in records), encoding="utf-8")
    tmp.replace(path)


def bundle_candidates():
    root = Path("/kaggle/input")
    return [p.parent for p in root.rglob("bundle.json")
            if (p.parent / "samples.jsonl").is_file() and (p.parent / "original").is_dir()] + list(
        root.rglob("sc203_apc_trace120.zip"))


class SharedBundle:
    def __init__(self, path):
        paths = [Path(path)] if path else bundle_candidates()
        by_digest = {}
        for p in sorted(paths, key=lambda p: (p.is_file(), str(p))):
            with self.open(p) as read_file:
                manifest = read_file("bundle.json")
            by_digest.setdefault(hashlib.sha256(manifest).hexdigest(), p)
        if len(by_digest) != 1:
            raise ValueError("Attach the existing APC120 dataset or set trace_input to its ZIP or root.")
        self.digest, self.path = next(iter(by_digest.items()))

    @staticmethod
    def open(path):
        from contextlib import contextmanager

        @contextmanager
        def reader():
            if path.is_dir():
                yield lambda name: inside(path, name).read_bytes()
            else:
                with zipfile.ZipFile(path) as archive:
                    roots = [n[:-len("bundle.json")] for n in archive.namelist()
                             if PurePosixPath(n).name == "bundle.json" and
                             n[:-len("bundle.json")] + "samples.jsonl" in archive.namelist()]
                    if len(roots) != 1:
                        raise ValueError("The APC120 ZIP must contain exactly one bundle.")
                    def read_member(name):
                        if PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts:
                            raise ValueError("Unsafe APC120 archive reference")
                        return archive.read(roots[0] + name)
                    yield read_member
        return reader()


def saved_calls(trace_root, row):
    path = inside(trace_root, row["attempt_path"]) / "calls.jsonl"
    calls = rows(path)
    starts = {event["id"]: event for event in calls if event["phase"] == "start"}
    completed = defaultdict(list)
    for event in calls:
        if event["phase"] == "end":
            start = starts[event["id"]]
            completed[event["component"]].append((start.get("inputs", {}), event["result"]))
    return completed, sha(path)


def prepare(cfg):
    """Validate row identity and freeze the actual historical model-facing inputs."""
    trace_root = Path(cfg["sympl_trace"]).resolve()
    history = rows(trace_root / "predictions.jsonl")
    historical_cfg = read(trace_root / "run_config.json")
    historical_runtime = read(trace_root / "effective_runtime.json")
    if historical_runtime["category_policy"] != "dataset_mapping" or historical_cfg.get("quantization"):
        raise ValueError("Unexpected historical SymPL category policy or checkpoint precision.")
    if historical_cfg["seed"] != int(cfg.get("seed", historical_cfg["seed"])):
        raise ValueError("Seed must match the historical SymPL run.")
    historical = {str(row["sample_id"]): row for row in history}
    bundle = SharedBundle(cfg.get("trace_input"))
    work = Path(cfg["runtime_dir"]).resolve()
    out = Path(cfg["output_dir"]).resolve()
    out.mkdir(parents=True, exist_ok=True)
    selected = cfg.get("sample_ids")
    with bundle.open(bundle.path) as read_file:
        manifest = json.loads(read_file("bundle.json"))
        samples = [json.loads(line) for line in read_file("samples.jsonl").splitlines() if line.strip()]
        ids = {str(r["sample_id"]) for r in samples}
        if len(ids) != len(samples) or len(history) != len(historical) or ids != set(historical):
            raise ValueError("Historical SymPL output and APC bundle must match one-to-one.")
        if Counter(r["paper_task"] for r in samples) != TASKS:
            raise ValueError("Expected 40 rows in each of Left/Right, Visibility and Facing.")
        for row in samples:
            sid = str(row["sample_id"])
            old = historical[sid]
            for key in ("question_id", "question", "choices", "answer", "paper_task", "image_sha256"):
                if row[key] != old[key]:
                    raise ValueError(f"SymPL/APC input mismatch for {sid}: {key}")
            if (row["historical_prediction"] != old["apc"]["prediction"] or
                    row["historical_correct"] != old["apc"]["correct"]):
                raise ValueError(f"APC history mismatch for {sid}.")
        if selected is not None:
            requested = list(map(str, selected))
            if not requested or len(set(requested)) != len(requested) or set(requested) - ids:
                raise ValueError("sample_ids must be unique IDs in the APC120 input.")
            samples = [r for r in samples if str(r["sample_id"]) in requested]

        plan = []
        for sample in samples:
            sid = str(sample["sample_id"])
            if not sid.isdecimal():
                raise ValueError("Expected numeric dataset row ID.")
            old = historical[sid]
            calls, calls_sha = saved_calls(trace_root, old)
            # a single module-level call for each high-level stage in this run
            entities = calls["get_objects_of_interest"][-1][1][0] if calls["get_objects_of_interest"] else None
            viewer, objects = (calls["get_reference_viewer"][-1][1][:2] if calls["get_reference_viewer"]
                               else (None, None))
            canonical = objects or entities or []
            if len(set(canonical)) != len(canonical):
                raise ValueError(f"Duplicate object descriptions in {sid}; cannot match recorded proposals by name.")
            detectors = {i["category"]: value for i, value in calls["run_detection"]}
            refinements = {i["category"]: value[0] for i, value in calls["run_detection_refinement"]}
            saved_orientation = {i["category"]: dict(bbox=i["bbox"], result=value)
                                 for i, value in calls["run_orientation_estimation"]}
            depth = calls["run_depth_intrinsic_estimation"][-1][1] if calls["run_depth_intrinsic_estimation"] else None
            if depth is not None and isinstance(depth[0], dict):
                artifact = inside(trace_root, depth[0]["array"])
                if artifact.suffix != ".npz" or sha(artifact) != artifact.stem:
                    raise ValueError(f"Historical depth artifact hash mismatch for {sid}.")
            data = read_file(sample["image_path"])
            digest = hashlib.sha256(data).hexdigest()
            if digest != manifest["files"][sample["image_path"]]:
                raise ValueError(f"Shared RGB trace checksum mismatch for {sid}.")
            img_rel = f"evidence/{sid}/input.png"
            img = out / img_rel
            img.parent.mkdir(parents=True, exist_ok=True)
            if not img.is_file() or sha(img) != digest:
                img.write_bytes(data)
            plan.append(dict(sample_id=sid, task=sample["paper_task"], answer=sample["answer"],
                             question=sample["question"], choices=sample["choices"], image=img_rel,
                             source_question=old["source_question"], question_transform=old["question_transform"],
                             historical_prediction=old["prediction"], historical_status=old["status"],
                             historical_correct=old["correct"], historical_apc=old["apc"]["prediction"],
                             entities=entities, viewer=viewer, objects=objects, canonical=canonical,
                             proposals=detectors, historical_boxes=refinements, depth=depth,
                             historical_orientations=saved_orientation,
                             seed=old.get("seed", (historical_cfg["seed"] + int(sid)) % 2**31),
                             calls_sha256=calls_sha, image_sha256=digest))

    selector = cfg.get("selector", "paired")
    if selector not in ("paired", "baseline", "reclip"):
        raise ValueError("selector must be paired, baseline or reclip.")
    fingerprint_data = dict(bundle_sha256=bundle.digest, replay_source=sha(Path(__file__)),
                            scorer_source=sha(Path(__file__).with_name("scorer.py")),
                            selector=selector, rows=[(r["sample_id"], r["calls_sha256"], r["image_sha256"]) for r in plan],
                            historical_run_sha256=sha(trace_root / "run_config.json"),
                            historical_runtime_sha256=sha(trace_root / "effective_runtime.json"),
                            historical_predictions_sha256=sha(trace_root / "predictions.jsonl"),
                            clip_batch_size=cfg.get("clip_batch_size", 16))
    fingerprint = hashlib.sha256(json.dumps(fingerprint_data, sort_keys=True).encode()).hexdigest()
    lock = out / "sympl_replay_identity.json"
    if lock.is_file() and read(lock)["fingerprint"] != fingerprint:
        raise ValueError("Output belongs to another SymPL replay. Use a new output_dir.")
    write(lock, dict(fingerprint=fingerprint, **fingerprint_data))
    write(work / "sympl_replay_plan.json", plan)
    write(out / "sympl_replay_protocol.json", dict(
        method="sympl", comparison="fresh Qwen selector versus ReCLIP in one runtime",
        replayed=["entity extraction", "reference viewer", "top-5 ordered Grounding DINO proposals",
                  "Depth Pro map and camera intrinsic", "historical OA1 orientation when viewer box is unchanged"],
        rerun=["Qwen baseline selector", "ReCLIP selector", "box-dependent depth unprojection",
               "OA1 orientation when viewer box changes","view transform", "projection", "bipartition",
               "layout rendering", "symbolic Qwen prompting and original answer parser"],
        historical_cohort=dict(sympl_correct=sum(r["correct"] for r in history),
                               apc_correct=sum(r["apc"]["correct"] for r in history), n=len(history)),
        visibility_protocol=historical_cfg["scope"], historical_baseline_used_for="separate drift check only",
        selector_input="detector entity description without question, answer or choices",
        selector=selector, grounding_labels="unavailable; disagreement records are not correctness labels"))
    print(f"Frozen SymPL inputs for {len(plan)} rows. Historical references: SymPL "
          f"{sum(r['correct'] for r in history)}/{len(history)}; APC {sum(r['apc']['correct'] for r in history)}/{len(history)}.",
          flush=True)
    return work, out, historical_cfg


def pixel_boxes(normalized, size):
    """Mirror SymPL float32 tensor cxcywh conversion and int truncation."""
    import numpy as np
    values = np.asarray(normalized, dtype=np.float32).reshape(-1, 4)
    centers, half = values[:, :2], values[:, 2:] / np.float32(2)
    return (np.concatenate([centers - half, centers + half], axis=1)
            * np.array([*size, *size], dtype=np.float32)).astype(int).tolist()


def usable_expression(name):
    # Grounding DINO categories were extracted as named entities, not VQA prompts.
    # Reject malformed values rather than scoring leaked relation or answer text.
    value = name.strip()
    if not value or re.search(r"\?|(?:^|\n)\s*[A-D][.:)]\s|\b(?:answer options|which object|visible or not|left or right|in front of me|behind me)\b", value, re.I):
        raise ValueError("Historical grounding phrase contains a question or answer options.")
    return value


def precompute(cfg):
    from PIL import Image
    from sc203_kat.reclip.scorer import ReCLIPScorer

    out, work = Path(cfg["output_dir"]), Path(cfg["runtime_dir"])
    plan = read(work / "sympl_replay_plan.json")
    selector = cfg.get("selector", "paired")
    scorer = None
    scored = {}
    for row in plan:
        sample_id = row["sample_id"]
        decisions = []
        with Image.open(out / row["image"]) as raw:
            image = raw.convert("RGB")
        for index, name in enumerate(row["canonical"]):
            if name == "camera":
                continue
            boxes = pixel_boxes(row["proposals"].get(name, []), image.size)
            expression = usable_expression(name)
            state = "not_reached" if name not in row["proposals"] else "no_proposals" if not boxes else "ready"
            item = dict(sample_id=sample_id, object_index=index, role="viewer" if name == row["viewer"] else "target",
                        expression=expression, proposals_xyxy=boxes, historical_box=row["historical_boxes"].get(name),
                        input_status=state, baseline_box=None, reclip_box=None, reclip_planned_box=None, scoring=None)
            if boxes and selector != "baseline":
                if scorer is None:
                    scorer = ReCLIPScorer(device="cuda:0", cache_dir=work / "reclip_scores", weight_dir=work / "clip_weights",
                                          batch_size=cfg.get("clip_batch_size", 16))
                scoring = scorer.rank_proposals(image, expression, boxes)
                item["scoring"] = scoring
                index_chosen = scoring.get("selected_index")
                if index_chosen is not None:
                    item["reclip_planned_box"] = boxes[index_chosen]
            decisions.append(item)
        scored[sample_id] = decisions
    write(work / "sympl_reclip_scores.json", scored)
    print("ReCLIP scoring phase complete; releasing its GPU process.", flush=True)


def replay_assets(profile_cfg, output):
    """Install only Qwen and OA1 weights, using the historical checkpoints."""
    import shutil
    from huggingface_hub import hf_hub_download, snapshot_download

    historical = read(Path(profile_cfg["sympl_trace"]) / "effective_runtime.json")["assets"]
    snapshots = historical["snapshots"]
    root = Path(profile_cfg.get("asset_input") or profile_cfg["asset_root"])
    oa_patch_asset = root / "checkpoints/orient_dino_weight.pt"
    if profile_cfg.get("asset_input"):
        manifest = read(root / "assets_manifest.json")
        if (any(manifest["snapshots"].get(key) != snapshots[key] for key in ("qwen_model", "dinov2_large")) or
                manifest.get("orientation_sha") != historical["orientation_sha"]):
            raise ValueError("model_input weights differ from the historical SymPL run.")
        if not oa_patch_asset.is_file():
            raise FileNotFoundError(oa_patch_asset)
    else:
        root.mkdir(parents=True, exist_ok=True)
        for key in ("qwen_model", "dinov2_large"):
            item = snapshots[key]
            destination = root / key
            snapshot_download(item["repo"], revision=item["sha"], local_dir=str(destination),
                              allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja"])
        model = hf_hub_download("Viglong/Orient-Anything", "croplargeEX2/dino_weight.pt",
                                revision=historical["orientation_sha"], local_dir=str(root/"orientation_download"))
        oa_patch_asset.parent.mkdir(exist_ok=True)
        if not oa_patch_asset.exists():
            oa_patch_asset.symlink_to(Path(model).resolve())
    for key in ("qwen_model", "dinov2_large"):
        if not (root / key / "config.json").is_file() or not list((root / key).glob("*.safetensors")):
            raise RuntimeError(f"Missing {key} checkpoint files.")
    write(output, dict(root=str(root.resolve()), snapshots={key:snapshots[key] for key in ("qwen_model","dinov2_large")},
                       orientation_sha=historical["orientation_sha"], replay_uses_detector_and_depth_weights=False))


def launch(cfg, config_file):
    from sc203_kat.runtime.kaggle import ensure_submodule

    work, out, _ = prepare(cfg)
    if sys.platform != "linux":
        raise RuntimeError("GPU replay requires the Kaggle Linux T4 ×2 runtime.")
    ensure_submodule(ROOT, "sympl")
    profile = ROOT / "profiles/sympl"
    ns = dict(__name__="__sympl_reclip_setup__", RUNTIME_DIR=str(work), OUTPUT_DIR=str(out),
              REPOSITORY_ROOT=ROOT, UPSTREAM_KEY="sympl", PROFILE_DIR=profile, RESUME_INPUT=None,
              METHOD_OPTIONS={"sympl_trace": str(Path(cfg["sympl_trace"]).resolve())}, SANITY=bool(cfg.get("sample_ids")),
              MODEL_INPUT=cfg.get("model_input"), DATASET_INPUT=None, METHOD="sympl", SCOPE="paper_tasks_visibility")
    def stage(name):
        file = profile / name
        print("[SymPL setup]", name, flush=True)
        exec(compile(file.read_text(encoding="utf-8"), str(file), "exec"), ns)

    stage("01_prepare.py")
    stage("02_environment.py")
    py = ns["PY"]
    env = ns["ENV"]
    env["PYTHONPATH"] += os.pathsep + str(ROOT / "src")
    if cfg.get("selector", "paired") != "baseline":
        from sc203_kat.reclip.scorer import CLIP_PACKAGE

        subprocess.run([str(ns["UV"]), "pip", "install", "--python", py, "--no-cache", CLIP_PACKAGE],
                       env=env, check=True)
    # Scorer exits before the sharded Qwen constructor starts.
    subprocess.run([py, "-m", "sc203_kat.reclip.sympl", str(config_file), "--phase", "score"],
                   env=env, check=True)
    assets_script = ns["SCRIPTS"] / "assets.py"
    assets_script.write_text("import sys\nfrom sc203_kat.reclip.sympl import replay_assets\n"
                             "from sc203_kat.ablation.common import read\n"
                             "replay_assets(read(sys.argv[1]), sys.argv[2])\n", encoding="utf-8")
    stage("03_native_assets.py")
    env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    runtime_cfg = dict(cfg, sympl_replay_runtime=str(ns["CONFIG"]), runtime_dir=str(work), output_dir=str(out))
    target = work / "sympl_replay_launch.json"
    write(target, runtime_cfg)
    subprocess.run([py, "-u", "-m", "sc203_kat.reclip.sympl", str(target), "--phase", "replay"],
                   cwd=ns["REPO"], env=env, check=True)


def parsed_option(raw, row):
    from apc_common import parse_option

    value = str(raw or "").strip()
    if row["task"] == "visibility":
        value = {"yes, visible": "yes", "not": "no"}.get(value.lower(), value)
    if row["task"] == "left_right":
        value = {"left": "on the left", "right": "on the right"}.get(value.lower(), value)
    return parse_option(value, row["choices"])


def trace_value(value):
    """Keep stage evidence small, preserve NaNs explicitly for diagnostic cases."""
    import numpy as np

    if isinstance(value, np.ndarray):
        return trace_value(value.tolist())
    if isinstance(value, np.generic):
        return trace_value(value.item())
    if isinstance(value, (list, tuple)):
        return [trace_value(x) for x in value]
    if isinstance(value, dict):
        return {str(k): trace_value(v) for k, v in value.items()}
    if isinstance(value, float) and not np.isfinite(value):
        return {"nonfinite": str(value)}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return {"type": type(value).__name__}


def configure(runtime_cfg, historical_runtime):
    import torch
    from box import Box
    import sympl.sympl_pipeline as pipeline
    from sympl.vision_modules import DetectionModule, DepthModule, OrientationModule
    from experiment_qwen import KaggleQwen
    # Use the actual historical placement, so allocator differences cannot change the split.
    import experiment_qwen

    device_map = historical_runtime["device_map"]
    if set(device_map.values()) != {0, 1}:
        raise ValueError("Expected the historical two-GPU Qwen placement.")
    experiment_qwen.plan_device_map = lambda _checkpoint: device_map.copy()
    config = Box(copy.deepcopy(historical_runtime["config"]))
    config.vlm.pretrained_model = runtime_cfg["weights"]
    assets = Path(runtime_cfg["assets"])
    config.orientation.ckpt_path = str(assets)
    config.orientation.model_path = str(assets / "checkpoints/orient_dino_weight.pt")
    config.orientation.processor_path = str(assets / "dinov2_large")
    # The frozen detector/depth methods below make their large checkpoints unnecessary.
    model = pipeline.SymPL.__new__(pipeline.SymPL)
    model.config = config
    model.device_vlm = "cuda:0"
    model.device_vision = "cuda:1"
    model.prompt_parser = pipeline.PromptParser(config)
    model.vlm_model = KaggleQwen(config.vlm, "cuda:0")
    processor = model.vlm_model.processor.image_processor.to_dict()
    if processor != historical_runtime["processor"]:
        raise RuntimeError("Loaded Qwen image processor differs from the historical SymPL runtime.")
    model.detection_module = DetectionModule.__new__(DetectionModule)
    model.detection_module.num_candidates = config.detection.num_candidates
    model.detection_module.config = config
    model.depth_module = DepthModule.__new__(DepthModule)
    orientation = None

    def infer_orientation(**kwargs):
        nonlocal orientation
        if orientation is None:
            orientation = OrientationModule(config, "cpu")
        orientation.device = "cuda:1"
        weights = orientation.orientation_model
        try:
            weights.to("cuda:1")
            with torch.cuda.device(1):
                return orientation.run_orientation_estimation(**kwargs)
        finally:
            weights.to("cpu")
            orientation.device = "cpu"
            gc.collect()
            torch.cuda.empty_cache()

    return pipeline, model, infer_orientation


def run_variant(mode, row, decisions, image, depth, root, pipeline, model, infer_orientation):
    from contextlib import ExitStack
    from functools import wraps
    from unittest.mock import patch
    import numpy as np
    import torch

    seed = row["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    out = root / "evidence" / row["sample_id"] / mode
    out.mkdir(parents=True, exist_ok=True)
    layout = out / "layout.png"
    layout.unlink(missing_ok=True)
    model.vlm_model.calls.clear()
    stage_log = []
    baseline_refine = model.detection_module.run_detection_refinement
    by_name = {s["expression"]: s for s in decisions}

    def objects(*_args, conv_history=None, **_kwargs):
        stage_log.append({"stage": "get_objects_of_interest", "source": "historical", "result": row["entities"]})
        return copy.deepcopy(row["entities"]), conv_history

    def viewer(*_args, conv_history=None, **_kwargs):
        stage_log.append({"stage": "get_reference_viewer", "source": "historical", "result": row["viewer"]})
        return row["viewer"], copy.deepcopy(row["objects"]), conv_history

    def detect(_image, category):
        if category not in row["proposals"]:
            raise ValueError("Historical trace ended before detector proposals for " + category)
        boxes = row["proposals"][category]
        stage_log.append({"stage": "run_detection", "source": "historical", "category": category,
                          "candidates": len(boxes)})
        return torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)

    def refine(vlm_model, image, category, boxes_output, conv_history=None, **sympl_args):
        decision = by_name[usable_expression(category)]
        if pixel_boxes(boxes_output.tolist(), image.size) != decision["proposals_xyxy"]:
            raise ValueError("The selectors did not receive the frozen proposal order.")
        if mode == "baseline":
            box, conv_history = baseline_refine(vlm_model, image, category, boxes_output,
                                               conv_history=conv_history, **sympl_args)
            decision["baseline_box"] = list(map(int, box))
            decision["baseline_vlm_response"] = model.vlm_model.calls[-1].get("raw_output")
        else:
            box = decision["reclip_planned_box"]
            if box is None:
                raise ValueError("ReCLIP has no usable candidate for " + category)
            decision["reclip_box"] = list(box)
        stage_log.append({"stage": "run_detection_refinement", "source": mode, "category": category,
                          "selected_box": list(box)})
        decision[f"{mode}_selector_reached"] = True
        return box, conv_history

    def cached_depth(_image):
        if depth is None:
            raise ValueError("Historical trace ended before depth map generation.")
        stage_log.append({"stage": "run_depth_intrinsic_estimation", "source": "historical"})
        return depth[0].copy(), depth[1].copy()

    def orient(**kwargs):
        name = kwargs["category"]
        previous = row["historical_orientations"].get(name)
        if previous and previous["bbox"] == list(kwargs["bbox"]):
            vector, azimuth, polar = previous["result"]
            result = np.array(vector), azimuth, polar
            source = "historical_box_unchanged"
        else:
            result = infer_orientation(**kwargs)
            source = "new_oa1"
        stage_log.append({"stage": "run_orientation_estimation", "source": source,
                          "category": name, "box": list(kwargs["bbox"]), "result": trace_value(result)})
        return result

    def monitored(owner, name):
        original = getattr(owner, name)

        @wraps(original)
        def call(*args, **kwargs):
            event = {"stage": name, "status": "started"}
            stage_log.append(event)
            try:
                result = original(*args, **kwargs)
                if name == "do_localization":
                    result.save(layout)
                    event["result"] = layout.relative_to(root).as_posix()
                else:
                    event["result"] = trace_value(result)
                event["status"] = "ok"
                return result
            except Exception as exc:
                event.update(status="error", error=repr(exc))
                raise
        return call

    prediction = None
    start = time.perf_counter()
    result = {"prediction": None, "status": "error"}
    fatal = False
    try:
        with ExitStack() as stack, torch.inference_mode():
            stack.enter_context(patch.object(pipeline, "get_objects_of_interest", objects))
            stack.enter_context(patch.object(pipeline, "get_reference_viewer", viewer))
            stack.enter_context(patch.object(model.detection_module, "detection_process_image",
                                             lambda _image: (None, None)))
            stack.enter_context(patch.object(model.detection_module, "run_detection", detect))
            stack.enter_context(patch.object(model.detection_module, "run_detection_refinement", refine))
            stack.enter_context(patch.object(model.depth_module, "run_depth_intrinsic_estimation", cached_depth))
            stack.enter_context(patch.object(model.depth_module, "unproject_to_3D_from_bbox",
                                             monitored(model.depth_module, "unproject_to_3D_from_bbox")))
            stack.enter_context(patch.object(model, "orientation_module",
                                             type("FrozenOrientation", (), {"run_orientation_estimation": staticmethod(orient)})(),
                                             create=True))
            for name in ("estimate_scene_geometry", "top_view_position", "do_perspective_change",
                         "do_projection_2d", "do_abstraction", "do_bipartition", "do_localization",
                         "symbolic_layout_prompting"):
                stack.enter_context(patch.object(pipeline, name, monitored(pipeline, name)))
            raw, _ = model.run_sympl(image, row["question"], reasoning_category=row["task"],
                                      trace_save_dir=str(out), visualize_trace=False, return_conv_history=False)
        prediction = parsed_option(raw, row)
        status = "method_failure" if str(raw).lower().startswith("fail") else "ok" if prediction else "invalid"
        result.update(raw_output=raw, prediction=prediction, status=status)
    except Exception as exc:
        result.update(error=repr(exc), traceback=traceback.format_exc())
        fatal = isinstance(exc, (torch.cuda.OutOfMemoryError, OSError)) or "cuda error" in str(exc).lower()
    finally:
        result.update(seconds=time.perf_counter() - start,
                      correct=result["status"] == "ok" and prediction == row["answer"],
                      selector_reached=any(d.get(f"{mode}_selector_reached", False) for d in decisions),
                      render=layout.relative_to(root).as_posix() if layout.is_file() else None,
                      metadata={"stages": (out / "stages.json").relative_to(root).as_posix(),
                                "vlm_calls": (out / "vlm_calls.json").relative_to(root).as_posix()})
        write(out / "stages.json", stage_log)
        write(out / "vlm_calls.json", model.vlm_model.calls)
        write(out / "result.json", result)
    return result, fatal


def replay(cfg):
    import numpy as np
    from PIL import Image

    work, out = Path(cfg["runtime_dir"]), Path(cfg["output_dir"])
    trace_root = Path(cfg["sympl_trace"])
    plan = read(work / "sympl_replay_plan.json")
    scored = read(work / "sympl_reclip_scores.json")
    runtime_cfg = read(cfg["sympl_replay_runtime"])
    historical_runtime = read(trace_root / "effective_runtime.json")
    pipeline, model, infer_orientation = configure(runtime_cfg, historical_runtime)
    selector = cfg.get("selector", "paired")
    modes = ["baseline", "reclip"] if selector == "paired" else [selector]
    predictions, decisions = [], []
    for index, row in enumerate(plan, start=1):
        sid = row["sample_id"]
        saved = out / "evidence" / sid / "paired.json"
        if saved.is_file() and not (cfg.get("retry_errors") and any(
                read(saved)["prediction"].get(mode, {}).get("status") == "error" for mode in modes)):
            value = read(saved)
            predictions.append(value["prediction"])
            decisions.extend(value["decisions"])
            continue
        if shutil.disk_usage(out).free < cfg.get("disk_min_free_gib", 1.5) * 2**30:
            raise RuntimeError("Disk reserve reached. Existing paired checkpoints are retained.")
        items = scored[sid]
        with Image.open(out / row["image"]) as raw:
            image = raw.convert("RGB")
        depth = None
        if row["depth"] is not None:
            d, intrinsic = row["depth"]
            if isinstance(d, dict):
                with np.load(inside(trace_root, d["array"]), allow_pickle=False) as archive:
                    d = archive["value"].copy()
            depth = (np.asarray(d, dtype=np.float32), np.asarray(intrinsic, dtype=float))
            if depth[0].shape != (image.height, image.width) or depth[1].shape != (3, 3):
                raise ValueError("Frozen depth map/intrinsic disagrees with recorded image for " + sid)
        paired = {key: row[key] for key in ("sample_id", "task", "answer", "historical_prediction",
                  "historical_status", "historical_correct", "question", "source_question",
                  "question_transform", "choices", "image", "viewer", "seed")}
        for mode in ("baseline", "reclip"):
            paired[mode] = {"prediction": None, "status": "not_run", "correct": False,
                            "selector_reached": False, "render": None, "metadata": {}}
        fatal = False
        for mode in modes:
            paired[mode], fatal = run_variant(mode, row, items, image, depth, out,
                                               pipeline, model, infer_orientation)
            if fatal:
                break
        for item in items:
            item["box_changed"] = (item["baseline_box"] is not None and item["reclip_box"] is not None
                                   and item["baseline_box"] != item["reclip_box"])
        paired["changed_box"] = any(item["box_changed"] for item in items)
        predictions.append(paired)
        decisions.extend(items)
        write_rows(out / "replay_predictions.jsonl", predictions)
        write_rows(out / "grounding_decisions.jsonl", decisions)
        if not fatal:
            write(saved, {"prediction": paired, "decisions": items})
        print(f"[SymPL {index}/{len(plan)}] {sid}: " + ", ".join(
            f"{mode}={paired[mode]['status']}/{paired[mode]['prediction']}" for mode in modes), flush=True)
        if fatal:
            raise RuntimeError(f"SymPL stopped after resource failure on {sid}. Per-branch trace and partial export retained.")
    write_rows(out / "replay_predictions.jsonl", predictions)
    write_rows(out / "grounding_decisions.jsonl", decisions)
    if cfg.get("sample_ids") and selector == "paired":
        render_tasks = {row["task"] for row in predictions if all(
            row[mode]["render"] and row[mode]["status"] == "ok" for mode in modes)}
        reranked = sum(item.get("reclip_selector_reached", False) and len(item["proposals_xyxy"]) > 1
                       for item in decisions)
        sanity = dict(pass_=render_tasks == set(TASKS) and reranked > 0, tasks_with_paired_layouts=sorted(render_tasks),
                      multi_proposal_objects_reaching_reclip=reranked, rows=len(predictions))
        write(out / "sanity_status.json", sanity)
        print("SymPL sanity:", sanity, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--phase", choices=("all", "prepare", "score", "replay"), default="all")
    args = parser.parse_args()
    path = args.config.resolve()
    cfg = read(path)
    if args.phase == "all":
        launch(cfg, path)
    elif args.phase == "prepare":
        prepare(cfg)
    elif args.phase == "score":
        precompute(cfg)
    else:
        replay(cfg)


if __name__ == "__main__":
    main()
