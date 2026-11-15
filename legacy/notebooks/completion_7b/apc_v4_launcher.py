"""Notebook-side supervision; stop the worker before exporting its checkpoints."""

import json
from pathlib import Path
import shutil
import subprocess
import time

from apc_kaggle_guardrails import atomic_json, export_safely, make_report, read_records, terminate_worker


def collect_reports(work, out):
    destination = Path(out)/"runtime_reports"
    destination.mkdir(exist_ok=True)
    for name in ("source_lock.json", "patches.json", "trace_patches.json", "assets_ready.json",
                 "dispatch_preflight.json", "memory_plan.json", "model_device_map.json",
                 "qwen_placement.json", "pipeline_memory.json", "gpu_check.txt"):
        source = Path(work)/name
        if source.is_file():
            shutil.copy2(source, destination/name)


def show_downloads(out):
    from IPython.display import FileLink, display
    out = Path(out)
    for relative in ("sanity_report.json", "results.zip", "sanity_traces.zip", "archive_status.json", "setup_status.json",
                     "run_failure.json", "run/predictions_apc_vis.jsonl", "run/worker.log"):
        if (out/relative).is_file():
            display(FileLink(str(out/relative)))


def launch(config, python, scripts, env, retry_failed=None):
    out = Path(config["output"])
    folder = out/"run"
    folder.mkdir(exist_ok=True)
    state_path = folder/"state.json"
    state_path.unlink(missing_ok=True)
    atomic_json(out/"sanity_report.json", dict(technical_status="RUNNING", note="Wait for all selected APC samples."))
    command = [str(python), "-u", str(Path(scripts)/"apc_worker.py"), "--manifest", str(out/"manifest.json")]
    if retry_failed is True:
        command.append("--retry-failed")
    elif retry_failed is False:
        command.append("--no-retry-failed")
    started = time.monotonic()
    failure = None
    with (folder/"worker.log").open("a", encoding="utf-8") as stream:
        child = subprocess.Popen(command, env=env, cwd=config["repo"], stdout=stream, stderr=subprocess.STDOUT)
        try:
            while child.poll() is None:
                try:
                    state = json.loads(state_path.read_text()) if state_path.exists() else {}
                except (OSError, ValueError):
                    state = {}
                elapsed = time.monotonic()-started
                print("APC", state.get("pipeline_stage", state.get("stage", "starting")),
                      state.get("paper_task", ""), state.get("sample_id", ""),
                      f"{state.get('processed', 0)}/{state.get('total', '?')}",
                      "errors=", state.get("errors", 0), "invalid=", state.get("invalid", 0),
                      "elapsed min=", round(elapsed/60, 1), "free GiB=", state.get("disk_free_gib"), flush=True)
                sample_start = state.get("sample_started_unix")
                if sample_start and time.time()-sample_start > config["sample_timeout_seconds"]+120:
                    raise TimeoutError("Parent watchdog: sample stalled beyond timeout + 120s")
                if not state.get("new_processed") and not sample_start and elapsed > config["load_timeout_seconds"]+120:
                    raise TimeoutError("Parent watchdog: APC model initialization stalled")
                if elapsed > config["run_timeout_seconds"]+120:
                    raise TimeoutError("Parent watchdog: run time budget exceeded")
                time.sleep(10)
            if child.returncode:
                with (folder/"worker.log").open("rb") as log:
                    log.seek(max(0, (folder/"worker.log").stat().st_size-16000))
                    print(log.read().decode("utf-8", errors="replace"))
                raise RuntimeError(f"APC worker failed (exit {child.returncode}); inspect run_failure.json / worker.log")
        except BaseException as exc:
            failure = exc
        finally:
            terminate_worker(child)
    # No ZIP operation races a live writer. Metadata remains downloadable if ZIP fails.
    if failure:
        try:
            rows = [json.loads(line) for line in (out/"samples.jsonl").read_text().splitlines()]
            records = read_records(folder/"predictions_apc_vis.jsonl", repair=True)
            report_path = out/"sanity_report.json"
            report = json.loads(report_path.read_text())
            if report.get("technical_status") != "FAIL":
                report = make_report(rows, records, str(failure))
            report["supervisor_error"] = str(failure)
            atomic_json(report_path, report)
        except Exception as report_error:
            print("Could not update report:", repr(report_error), flush=True)
    try:
        collect_reports(config["work"], out)
    except OSError as export_error:
        print("Report collection:", repr(export_error), flush=True)
    export_safely(out, config, debug=True)
    show_downloads(out)
    if failure:
        raise failure
    report = json.loads((out/"sanity_report.json").read_text())
    if report["technical_status"] != "PASS":
        raise RuntimeError("Sanity report did not pass")
    print("APC SANITY PASS:", report["valid_predictions"], "/", report["expected_rows"], flush=True)
    return report
