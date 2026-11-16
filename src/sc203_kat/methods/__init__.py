"""Adapters for the five migrated Kaggle pipelines.

The profile scripts are ordinary versioned source files. Historical notebooks
are never parsed, executed, or patched at runtime.
"""
from pathlib import Path
import hashlib
import json
import os
import platform
import sys
import time

METHODS = ("apc_vis", "pcd", "sympl", "vst", "spatial_ssrl")
ALIASES = {"apc": "apc_vis", "ssrl": "spatial_ssrl", "spatial-srrl": "spatial_ssrl"}


def _root(root=None):
    return Path(root).resolve() if root else Path(__file__).resolve().parents[3]


def _profile(method, root=None):
    method = ALIASES.get(method, method)
    if method not in METHODS:
        raise ValueError("Unknown method: " + method)
    return method, _root(root) / "profiles" / method


def default_config(method, root=None):
    _, profile = _profile(method, root)
    return json.loads((profile / "defaults.json").read_text(encoding="utf8"))


def describe(method, config=None, root=None):
    method, profile = _profile(method, root)
    config = config or {}
    mode = config.get("mode", "full")
    if mode not in ("sanity", "full"):
        raise ValueError("mode must be sanity or full")
    manifest = json.loads((profile / "manifest.json").read_text(encoding="utf8"))
    digest = hashlib.sha256()
    for path in sorted(profile.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
            digest.update(path.relative_to(profile).as_posix().encode())
            digest.update(path.read_bytes())
    return dict(method=method, mode=mode, profile_sha256=digest.hexdigest(),
                profile_dir=str(profile), stages=manifest["stages"],
                runtime_dir=str(config.get("runtime_dir") or f"/kaggle/temp/sc203_kat/{method}/{mode}"),
                output_dir=str(config.get("output_dir") or f"/kaggle/working/sc203_kat/{method}/{mode}"),
                origin=manifest, method_options=dict(config.get("method_options") or {}), gpu_validation="Migrated settings; run the sanity notebook on Kaggle before a new full run.")


def run(method, config=None, root=None):
    """Run a migrated profile in order; setup is isolated from kernel ML packages."""
    config = dict(config or {})
    plan = describe(method, config, root)
    if platform.system() != "Linux":
        raise RuntimeError("GPU inference profiles target Kaggle Linux. Use --dry-run on this computer.")
    from ..runtime.kaggle import ensure_submodule, source_metadata
    ensure_submodule(_root(root), plan["method"])
    source = source_metadata(_root(root), plan["method"])
    profile = Path(plan["profile_dir"])
    options = dict(config.get("method_options") or {})
    namespace = dict(__name__="__sc203_profile__", PROFILE_DIR=profile,
                     REPOSITORY_ROOT=_root(root), UPSTREAM_KEY={"apc_vis":"apc_vlm", "spatial_ssrl":"spatial_ssrl"}.get(plan["method"],plan["method"]),
                     RUNTIME_DIR=plan["runtime_dir"], OUTPUT_DIR=plan["output_dir"],
                     METHOD_OPTIONS=options, SANITY=plan["mode"] == "sanity",
                     METHOD="ssrl" if plan["method"] == "spatial_ssrl" else plan["method"],
                     SCOPE=options.get("scope", "paper_tasks_original" if plan["method"] == "spatial_ssrl" else "paper_tasks_visibility"),
                     MODEL_INPUT=config.get("model_input"), DATASET_INPUT=config.get("dataset_input"),
                     RESUME_INPUT=config.get("resume_input"))
    Path(plan["runtime_dir"]).mkdir(parents=True, exist_ok=True)
    out = Path(plan["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    identity = {k:plan[k] for k in ("method", "mode", "profile_sha256")}
    identity["model_input"] = config.get("model_input")
    identity["dataset_input"] = config.get("dataset_input")
    identity["method_options"] = {k:v for k,v in options.items() if k not in {"run_name", "session_budget_minutes", "max_rows_this_session", "retry_failed", "run_timeout_seconds", "sample_timeout_seconds", "trace_keep_per_task", "trace_max_total_mib", "trace_max_sample_mib", "archive_max_mib", "disk_min_free_gib", "max_total_errors", "max_consecutive_errors"}}
    identity_path = out / "sc203_identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text(encoding="utf8")) != identity:
        raise RuntimeError("This output directory belongs to a different profile/configuration. Choose a new output directory.")
    identity_path.write_text(json.dumps(identity, indent=2), encoding="utf8")
    (out / "source_checkout.json").write_text(json.dumps(source, indent=2), encoding="utf8")
    run_state = {"plan":plan, "source":source, "started":time.time(), "stages":[]}
    state_path = out / "sc203_run.json"
    try:
        for name in plan["stages"]:
            path = profile / name
            print(f"[sc203-kat] {plan['method']}: {name}", flush=True)
            started = time.time()
            namespace["__file__"] = str(path)
            exec(compile(path.read_text(encoding="utf8"), str(path), "exec"), namespace)
            run_state["stages"].append({"stage":name,"seconds":time.time()-started,"status":"ok"})
            state_path.write_text(json.dumps(run_state,indent=2),encoding="utf8")
        run_state["status"] = "finished"
    except BaseException as exc:
        run_state.update(status="error", failed_stage=name, error=repr(exc))
        raise
    finally:
        run_state["ended"] = time.time()
        state_path.write_text(json.dumps(run_state,indent=2),encoding="utf8")
    return run_state
