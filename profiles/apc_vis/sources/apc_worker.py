"""Use the GPU-tested v4 executor with strict viewers and resumable full-run reports."""

import json
from pathlib import Path
import sys

from apc_kaggle_guardrails import ResourceLimitError, atomic_json
from apc_full_checkpoint import write_full_report
import apc_v4_worker


def full_experiment(config, assets, callback, sanity=False):
    from apc_experiment import APCExperiment
    import torch
    experiment = APCExperiment(config, assets, callback, sanity=False)
    # The same numerical check used by the successful six-row sanity, without
    # turning on verbose INFO logging for thousands of rows.
    def finite_logits(module, args, output):
        logits = getattr(output, "logits", None)
        if logits is not None and not torch.isfinite(logits[:, -1, :]).all().item():
            raise FloatingPointError("NaN/Inf in Qwen raw logits")
    experiment.vlm.vlm_model.register_forward_hook(finite_logits)
    original = experiment.run
    def run(row, mode, trace):
        try:
            return original(row, mode, trace)
        finally:
            # Full prompts/images remain in retained debug traces. Keep the
            # checkpoint compact enough to download and mount in the next session.
            calls = [{key:value for key,value in call.items() if key in
                      ("prompt_tokens", "generated_tokens", "seconds", "status")}
                     for call in experiment.vlm.calls]
            audit = Path(trace)/"reference_viewer.json"
            if audit.is_file():
                value = json.loads(audit.read_text())
                calls.append(dict(reference_viewer_strategy=value["strategy"],
                                  resolved_viewer=value["resolved_viewer"]))
            experiment.vlm.calls = calls
    experiment.run = run
    return experiment


def validate_full_trace(trace):
    warnings = apc_v4_worker.validate_trace(trace)
    audit = Path(trace)/"reference_viewer.json"
    if not audit.is_file():
        raise AssertionError("Missing reference viewer audit")
    value = json.loads(audit.read_text())
    if not value.get("resolved_viewer") or value["strategy"] == "unresolved":
        raise ValueError("Unresolved reference viewer")
    if value["strategy"] == "explicit_camera_question":
        warnings = [warning for warning in warnings if "fell back to camera" not in warning]
    return warnings


def run_full(manifest_path, retry_failed=None, runner=None):
    manifest = json.loads(Path(manifest_path).read_text())
    out = Path(manifest["config"]["output"])
    failure, paused = None, False
    try:
        (runner or apc_v4_worker.run_manifest)(manifest_path, retry_failed=retry_failed,
            experiment_factory=full_experiment, evidence_validator=validate_full_trace)
    except ResourceLimitError as exc:
        if str(exc) == "Run time budget reached; pending rows remain in the report.":
            paused = True
            source = out/"run_failure.json"
            if source.exists():
                source.replace(out/"pause_event.json")
            state_file = out/"run/state.json"
            if state_file.exists():
                state = json.loads(state_file.read_text())
                atomic_json(state_file, dict(state, stage="paused", stop_reason="session_time_budget"))
        else:
            failure = exc
    except RuntimeError as exc:
        # v4 requires all six sanity rows to pass. A full benchmark may finish
        # with explicit row errors; its score still includes those rows as wrong.
        if not str(exc).startswith("Sanity FAIL: inspect sanity_report.json"):
            failure = exc
    except BaseException as exc:
        failure = exc
    try:
        component = out/"sanity_report.json"
        if component.exists():
            destination = out/"runtime_reports/worker_report.json"
            destination.parent.mkdir(exist_ok=True)
            component.replace(destination)
        report = write_full_report(out, pause=paused, failure=failure)
    except Exception:
        if failure is not None:
            raise failure
        raise
    print("APC FULL", report["completion_status"],
          f"{report['attempted_rows']}/{report['expected_rows']}",
          "valid=", report["valid_predictions"], "errors=", report["pipeline_errors"],
          "invalid=", report["invalid_outputs"], flush=True)
    if failure is not None:
        raise failure
    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--retry-failed", action="store_true", dest="retry_failed")
    group.add_argument("--no-retry-failed", action="store_false", dest="retry_failed")
    parser.set_defaults(retry_failed=None)
    args = parser.parse_args()
    run_full(args.manifest, args.retry_failed)
