"""Portable full-run checkpoints, pinned resume assets and full-denominator reports."""

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import zipfile

from apc_kaggle_guardrails import (
    OPERATIONAL_KEYS, atomic_json, export_safely, make_report, read_records,
    select_samples as select_base, successful,
)


def select_samples(rows, config):
    rows = select_base(rows, config)
    if config["run_kind"] != "full":
        return rows
    tasks = config["paper_tasks"]
    grouped = {task:{} for task in tasks}
    for row in rows:
        grouped[row["paper_task"]].setdefault(row["circular_group"], []).append(row)
    lists = [list(grouped[task].values()) for task in tasks]
    # Exercise both part-1 task paths early; preserve every complete circular pair.
    return [row for index in range(max(map(len, lists))) for groups in lists if index < len(groups)
            for row in groups[index]]


def prepare_resume(config, work):
    source = config.get("resume_input")
    if not source:
        return
    work, source = Path(work), Path(source)
    if source.is_file():
        destination = work/"resume"/config["experiment"]
        destination.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(source) as archive:
            for name in ("manifest.json", "run/predictions_apc_vis.jsonl"):
                info = archive.getinfo(name)
                if info.file_size > 128*2**20:
                    raise ValueError("Resume entry exceeds 128 MiB: " + name)
                target = destination/name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.read(info))
        source = destination
    manifest = json.loads((source/"manifest.json").read_text(encoding="utf-8"))
    old_config = {k:v for k,v in manifest["config"].items() if k not in OPERATIONAL_KEYS}
    current = {k:v for k,v in config.items() if k not in OPERATIONAL_KEYS}
    if old_config != current:
        raise RuntimeError("Resume protocol differs: use the same v5 part/model/code configuration.")
    if not (source/"run/predictions_apc_vis.jsonl").is_file():
        raise ValueError("Resume folder is missing run/predictions_apc_vis.jsonl")
    # Restore exact revisions BEFORE native install and auxiliary-model downloads.
    atomic_json(work/"source_lock.json", manifest["source_lock"])
    assets = manifest["assets"]
    root = Path(config["asset_input"] or config["asset_root"])
    if config["asset_input"]:
        existing = json.loads((root/"assets_manifest.json").read_text())
        if {k:v for k,v in existing.items() if k != "root"} != {k:v for k,v in assets.items() if k != "root"}:
            raise RuntimeError("Mounted assets differ from the checkpoint's pinned assets")
    else:
        root.mkdir(parents=True, exist_ok=True)
        atomic_json(root/"assets_manifest.json", assets)
    config["resume_input"] = str(source.resolve())
    print("Resume loaded; native/model revisions pinned to previous manifest:", source, flush=True)


def ensure_checkout(path, url, sha):
    """Fetch an immutable commit in a generated runtime copy, preserving dirty caches."""
    path = Path(path)
    if not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", sha):
        raise ValueError("Invalid pinned git SHA")
    def git(*args, capture=False):
        command = ["git", "-C", str(path), *args]
        return subprocess.check_output(command, text=True).strip() if capture else subprocess.run(command, check=True)
    path.mkdir(parents=True, exist_ok=True)
    if not (path/".git").exists():
        git("init", "--quiet")
        git("remote", "add", "origin", url)
    else:
        current = git("rev-parse", "HEAD", capture=True)
        if current == sha:
            return
        if git("status", "--porcelain", capture=True):
            raise RuntimeError("Cached native source differs from resume and has patches; resume in a clean Kaggle session.")
    git("fetch", "--depth", "1", "origin", sha)
    git("checkout", "--detach", sha)
    if git("rev-parse", "HEAD", capture=True) != sha:
        raise RuntimeError("Pinned native checkout failed")


def compact_record(record):
    result = {k:v for k,v in record.items() if k not in ("vlm_calls", "stages")}
    for call in record.get("vlm_calls", []):
        for key in ("reference_viewer_strategy", "resolved_viewer"):
            if key in call:
                result[key] = call[key]
    return result


def write_full_report(out, *, pause=False, failure=None):
    out = Path(out)
    manifest = json.loads((out/"manifest.json").read_text())
    rows = [json.loads(line) for line in (out/"samples.jsonl").read_text().splitlines()]
    records = read_records(out/"run/predictions_apc_vis.jsonl", repair=True)
    records = {sid:compact_record(record) for sid,record in records.items()}
    report = make_report(rows, records, str(failure) if failure else None)
    report.update(completion_status="PAUSED" if pause else "ERROR" if failure else
                  "COMPLETE" if len(records) == len(rows) else "PARTIAL",
                  attempted_percent=100*len(records)/len(rows),
                  valid_percent=100*report["valid_predictions"]/len(rows),
                  all_rows_attempted=len(records)==len(rows),
                  complete=report["valid_predictions"]==len(rows) and failure is None,
                  circular_groups=len({r["circular_group"] for r in rows}),
                  accuracy_note="All selected rows/groups stay in denominator; missing, error and invalid count as wrong.",
                  protocol=manifest["protocol"], fingerprint=manifest["fingerprint"])
    if pause:
        report["technical_status"] = "PARTIAL"
        report["stop_reason"] = "session_time_budget; resume from checkpoint.zip"
    report["quality_warnings"] = {sid:r["quality_warnings"] for sid,r in records.items() if r.get("quality_warnings")}
    report["reference_viewer_repairs"] = sum(r.get("reference_viewer_strategy") == "question_subject" for r in records.values())
    report["peak_vram_gib"] = {
        str(gpu):max((r.get("peak_vram_gib", {}).get(str(gpu), 0) for r in records.values() if r.get("peak_vram_gib")), default=0)
        for gpu in (0, 1)}
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["circular_group"]].append(row)
    def hit(row):
        prediction = records.get(str(row["sample_id"]), {})
        return successful(prediction) and prediction.get("prediction") == row["answer"]
    for task, metric in report["by_task"].items():
        selected = [row for row in rows if row["paper_task"] == task]
        circular = [group for group in grouped.values() if group[0]["paper_task"] == task]
        metric.update(row_accuracy=sum(hit(row) for row in selected)/len(selected),
                      circular_groups=len(circular),
                      circular_accuracy=sum(all(hit(row) for row in group) for group in circular)/len(circular))
    for filename in ("full_report.json", "summary.json"):
        atomic_json(out/filename, report)
    # Compact merged rows support analysis without the per-call prompts in debug traces.
    with (out/"apc_vis_predictions.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            prediction = records.get(str(row["sample_id"]), dict(status="missing", prediction=None))
            merged = dict(row, **{k:v for k,v in prediction.items() if k not in ("vlm_calls", "stages")})
            merged["correct"] = bool(hit(row))
            stream.write(json.dumps(merged, ensure_ascii=False)+"\n")
    return report


def create_checkpoint(out, config):
    out = Path(out)
    target = out/"checkpoint.zip"
    temporary = target.with_suffix(".zip.tmp")
    records = read_records(out/"run/predictions_apc_vis.jsonl", repair=True)
    portable = []
    for record in records.values():
        value = compact_record(record)
        if value.get("trace_path"):
            value["previous_trace_path"] = value["trace_path"]
        value["trace_path"] = None  # Debug files are deliberately outside checkpoint.zip.
        portable.append(value)
    lines = "".join(json.dumps(record, ensure_ascii=False)+"\n" for record in portable).encode("utf-8")
    manifest = (out/"manifest.json").read_bytes()
    required = len(lines)+len(manifest)+2**20
    if required > 128*2**20 or shutil.disk_usage(out).free < config["disk_min_free_gib"]*2**30+required:
        target.unlink(missing_ok=True)
        return dict(status="SKIPPED", reason="checkpoint size/free disk budget")
    try:
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("manifest.json", manifest)
            archive.writestr("run/predictions_apc_vis.jsonl", lines)
        temporary.replace(target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        target.unlink(missing_ok=True)
        raise
    return dict(status="OK", path=str(target), rows=len(records), bytes=target.stat().st_size)


def export_full(out, config):
    out = Path(out)
    try:
        checkpoint = create_checkpoint(out, config) if (out/"manifest.json").exists() else dict(status="NOT_READY")
    except Exception as exc:
        checkpoint = dict(status="FAILED", message=str(exc))
    try:
        atomic_json(out/"checkpoint_status.json", checkpoint)
    except OSError:
        pass
    # The retained debug archive is small; primary results never contain this tree.
    (out/"traces.zip").unlink(missing_ok=True)
    status = export_safely(out, config, debug=True)
    traces = out/"sanity_traces.zip"
    if status.get("traces", {}).get("status") == "OK":
        try:
            traces.replace(out/"traces.zip")
            status["traces"]["path"] = str(out/"traces.zip")
        except OSError as exc:
            status["traces"] = dict(status="FAILED", message=str(exc))
    status["checkpoint"] = checkpoint
    try:
        atomic_json(out/"archive_status.json", status)
    except OSError:
        pass
    print("Checkpoint:", json.dumps(checkpoint), flush=True)
    return status
