"""Run APC with durable checkpoints, bounded traces and explicit technical coverage."""

from collections import Counter
import functools
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import sys
import time
import traceback

from apc_common import parse_option
from apc_kaggle_guardrails import (
    FailureBudget, ResourceLimitError, atomic_json, check_resources, fatal_exception,
    make_report, pending_rows, read_records, remove_trace, retain_trace, successful,
)

PIPELINE_STAGES = {
    "get_objects_of_interest", "do_scene_abstraction", "get_reference_viewer",
    "do_perspective_change", "do_perspective_prompting_visual", "render_visual_prompt",
    "detection", "segmentation", "depth", "unprojection", "orientation",
}


def attach_stages(experiment, callback):
    for owner, method, stage in (
        (experiment.apc.detection_module, "run_detection", "detection"),
        (experiment.apc.detection_module, "run_segmentation", "segmentation"),
        (experiment.apc.depth_module, "run_depth_estimation", "depth"),
        (experiment.apc.depth_module, "unproject_to_3D", "unprojection"),
        (experiment.apc.orientation_module, "run_orientation_estimation", "orientation"),
    ):
        original = getattr(owner, method)
        @functools.wraps(original)
        def measured(*args, _original=original, _stage=stage, **kwargs):
            callback(_stage)
            return _original(*args, **kwargs)
        setattr(owner, method, measured)


def validate_trace(trace):
    trace = Path(trace)
    required = ("input.png", "depth.npy", "scene.json", "apc_visual_prompt.png")
    missing = [name for name in required if not (trace/name).is_file()]
    orientations = list(trace.glob("orientation_angles_*.json"))
    masks = list(trace.glob("object_*_mask.png"))
    if missing or not orientations or not masks:
        raise ValueError(f"Incomplete APC evidence: missing={missing}, orientations={len(orientations)}, masks={len(masks)}")
    for path in orientations:
        vector = json.loads(path.read_text())["direction_vector"]
        if len(vector) != 3 or not all(math.isfinite(float(x)) for x in vector):
            raise FloatingPointError("Invalid orientation vector in " + path.name)
    # Real model output must include a decodable rendered image, not just model loading.
    from PIL import Image
    with Image.open(trace/"apc_visual_prompt.png") as image:
        image.verify()
    scene = json.loads((trace/"scene.json").read_text())
    warnings = []
    if scene.get("viewer") == "camera":
        warnings.append("Reference viewer fell back to camera; inspect the VLM reference-viewer response.")
    if len([name for name in scene.get("camera", {}) if name != "camera"]) < 2:
        warnings.append("Fewer than two objects survived scene abstraction; inspect detection/refinement.")
    return warnings


def run_manifest(manifest_path, retry_failed=None, experiment_factory=None, torch_module=None,
                 numpy_module=None, evidence_validator=validate_trace):
    if torch_module is None:
        import torch as torch_module
    if numpy_module is None:
        import numpy as numpy_module
    torch, np = torch_module, numpy_module
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    config = manifest["config"]
    out = Path(config["output"])
    folder = out/"run"
    folder.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in (out/"samples.jsonl").read_text(encoding="utf-8").splitlines()]
    predictions = folder/"predictions_apc_vis.jsonl"
    if config.get("resume_input") and not predictions.exists():
        source = Path(config["resume_input"])
        previous_manifest = json.loads((source/"manifest.json").read_text())
        if previous_manifest["fingerprint"] != manifest["fingerprint"]:
            raise RuntimeError("resume_input has a different protocol fingerprint")
        previous = source/"run/predictions_apc_vis.jsonl"
        if previous.is_file():
            import shutil
            shutil.copy2(previous, predictions)
    records = read_records(predictions, repair=True)
    retry_failed = config["retry_failed"] if retry_failed is None else retry_failed
    pending = pending_rows(rows, records, manifest["fingerprint"], retry_failed)
    previous_failure = out/"run_failure.json"
    if previous_failure.exists():
        history = out/"failure_history"
        history.mkdir(exist_ok=True)
        previous_failure.replace(history/f"run_{time.time_ns()}.json")
    state = dict(stage="starting", processed=len(rows)-len(pending), total=len(rows),
                 new_processed=0, errors=0, invalid=0, sample_started_unix=None)
    started = time.monotonic()
    current_trace, events = None, []
    stop_reason, failure = None, None
    kept = Counter()
    for row in rows:
        record = records.get(str(row["sample_id"]), {})
        if record.get("trace_path"):
            kept[row["paper_task"]] += 1

    def update(stage, **values):
        state.update(stage=stage, elapsed_seconds=time.monotonic()-started, updated_unix=time.time(), **values)
        if stage in PIPELINE_STAGES:
            state["pipeline_stage"] = stage
            if len(events) < 512:
                events.append(dict(stage=stage, seconds=state["elapsed_seconds"]))
        state["disk_free_gib"] = check_resources(out, config, current_trace)
        atomic_json(folder/"state.json", state)

    try:
        update("initializing")
        if pending:
            if experiment_factory is None:
                from apc_experiment import APCExperiment
                experiment_factory = APCExperiment
            assets = json.loads((Path(config["work"])/"assets_ready.json").read_text())
            experiment = experiment_factory(config, assets, update, sanity=config["run_kind"] == "sanity")
            if hasattr(experiment, "apc"):
                attach_stages(experiment, update)
            budget = FailureBudget(config)
            with predictions.open("a", encoding="utf-8") as stream:
                for row in pending:
                    if time.monotonic()-started > config["run_timeout_seconds"]:
                        raise ResourceLimitError("Run time budget reached; pending rows remain in the report.")
                    sid = str(row["sample_id"])
                    trace_root = folder/"trace"
                    current_trace = trace_root/"apc_vis"/sid
                    # The TSV uses decimal IDs. Refuse paths outside this run.
                    if not sid.isdecimal():
                        raise ValueError("Unsafe/non-decimal sample ID: " + sid)
                    remove_trace(current_trace, trace_root)
                    current_trace.mkdir(parents=True)
                    os.environ["APC_TRACE_DIR"] = str(current_trace)
                    events.clear()
                    state["pipeline_stage"] = "sample_start"
                    update("sample_start", sample_id=sid, paper_task=row["paper_task"], sample_started_unix=time.time())
                    seed = int.from_bytes(hashlib.sha256(f"{config['seed']}:{sid}".encode()).digest()[:4], "big")
                    random.seed(seed)
                    np.random.seed(seed)
                    torch.manual_seed(seed)
                    torch.cuda.manual_seed_all(seed)
                    for gpu in (0, 1):
                        torch.cuda.reset_peak_memory_stats(gpu)
                    record = dict(sample_id=sid, paper_task=row["paper_task"], fingerprint=manifest["fingerprint"],
                                  mode="apc_vis", status="error", prediction=None, parse_valid=False, raw_output="", seed=seed)
                    fatal = None
                    sample_start = time.monotonic()
                    handler = None
                    def timeout(signum, frame):
                        raise TimeoutError("APC sample exceeded sample_timeout_seconds")
                    if hasattr(signal, "SIGALRM"):
                        handler = signal.signal(signal.SIGALRM, timeout)
                        signal.alarm(config["sample_timeout_seconds"])
                    try:
                        raw = experiment.run(row, "apc_vis", current_trace)
                        prediction = parse_option(raw, row["choices"])
                        warnings = evidence_validator(current_trace)
                        record.update(status="ok" if prediction is not None else "invalid", raw_output=raw,
                                      prediction=prediction, parse_valid=prediction is not None, quality_warnings=warnings,
                                      correct=prediction == row["answer"])
                        if prediction is None:
                            record.update(error_type="InvalidAnswer", message="Response is not an exact answer option")
                            state["invalid"] += 1
                    except Exception as exc:
                        state["errors"] += 1
                        record.update(error_type=type(exc).__name__, message=str(exc),
                                      traceback=traceback.format_exc(), failed_stage=state.get("pipeline_stage"),
                                      failed_operation=state.get("stage"))
                        traceback.print_exc()
                        if fatal_exception(exc):
                            fatal = exc
                    finally:
                        if hasattr(signal, "SIGALRM"):
                            signal.alarm(0)
                            signal.signal(signal.SIGALRM, handler)
                        record.update(seconds=time.monotonic()-sample_start, stages=list(events),
                                      vlm_calls=getattr(getattr(experiment, "vlm", None), "calls", []))
                        try:
                            record["peak_vram_gib"] = {str(i):torch.cuda.max_memory_allocated(i)/2**30 for i in (0, 1)}
                        except Exception:
                            record["peak_vram_gib"] = None
                        # Release excess debug files before writing durable, small results.
                        try:
                            retain_trace(current_trace, trace_root, row, record, config, kept)
                        except Exception as trace_error:
                            record["trace_management_error"] = repr(trace_error)
                            fatal = fatal or trace_error
                        stream.write(json.dumps(record, ensure_ascii=False)+"\n")
                        stream.flush()
                        os.fsync(stream.fileno())
                        records[sid] = record
                    current_trace = None
                    state["processed"] += 1
                    state["new_processed"] += 1
                    update("sample_done", last_status=record["status"], last_seconds=record["seconds"], sample_started_unix=None)
                    atomic_json(out/"sanity_report.json", make_report(rows, records))
                    print(json.dumps(dict(sample_id=sid, task=row["paper_task"], status=record["status"],
                                          prediction=record["prediction"], seconds=round(record["seconds"], 2))), flush=True)
                    reason = budget.observe(record)
                    if fatal:
                        raise fatal
                    if reason:
                        raise RuntimeError("Stopped by failure budget: " + reason)
        update("complete", sample_started_unix=None)
    except BaseException as exc:
        failure = exc
        stop_reason = type(exc).__name__+": "+str(exc)
        try:
            atomic_json(out/"run_failure.json", dict(error_type=type(exc).__name__, message=str(exc),
                        traceback=traceback.format_exc(), pipeline_stage=state.get("pipeline_stage")))
            atomic_json(folder/"state.json", dict(state, stage="error", stop_reason=stop_reason))
        except OSError:
            pass
    report = make_report(rows, records, stop_reason)
    report["wall_seconds"] = time.monotonic()-started
    report["new_attempts"] = state["new_processed"]
    report["reused_valid_rows"] = sum(successful(r) for r in records.values()) - sum(
        successful(records.get(str(row["sample_id"]), {})) for row in pending)
    report["quality_warnings"] = {sid:r["quality_warnings"] for sid,r in records.items() if r.get("quality_warnings")}
    if all(task in report["by_task"] and report["by_task"][task]["mean_seconds"] is not None for task in ("left_right", "visibility", "facing")):
        means = {task:report["by_task"][task]["mean_seconds"] for task in report["by_task"]}
        report["rough_full_inference_hours"] = dict(
            part1=(698*means["left_right"]+688*means["visibility"])/3600,
            part2=692*means["facing"]/3600,
            note="Tiny-sample extrapolation; excludes setup/load/export and is not a time guarantee.")
    try:
        atomic_json(out/"sanity_report.json", report)
    except OSError:
        if failure is None:
            raise
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if failure:
        raise failure
    if report["technical_status"] != "PASS":
        raise RuntimeError("Sanity FAIL: inspect sanity_report.json and run/predictions_apc_vis.jsonl")
    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    retry = parser.add_mutually_exclusive_group()
    retry.add_argument("--retry-failed", dest="retry_failed", action="store_true")
    retry.add_argument("--no-retry-failed", dest="retry_failed", action="store_false")
    parser.set_defaults(retry_failed=None)
    arguments = parser.parse_args()
    run_manifest(arguments.manifest, retry_failed=arguments.retry_failed)
