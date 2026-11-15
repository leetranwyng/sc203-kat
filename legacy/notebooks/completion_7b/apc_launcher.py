"""Supervise a time-bounded full run and export an independent resume checkpoint."""

import json
from pathlib import Path
import shutil
import subprocess
import time

from apc_kaggle_guardrails import atomic_json, terminate_worker
from apc_full_checkpoint import export_full, write_full_report
from apc_v4_launcher import collect_reports as collect_base


def collect_reports(work, out):
    collect_base(work, out)
    source = Path(work)/"reference_patches.json"
    if source.is_file():
        shutil.copy2(source, Path(out)/"runtime_reports/reference_patches.json")


def show_downloads(out):
    from IPython.display import FileLink, display
    for name in ("full_report.json", "summary.json", "checkpoint.zip", "results.zip", "traces.zip",
                 "checkpoint_status.json", "archive_status.json", "setup_status.json", "run_failure.json",
                 "run/predictions_apc_vis.jsonl", "run/worker.log"):
        path = Path(out)/name
        if path.is_file():
            display(FileLink(str(path)))


def launch(config, python, scripts, env, retry_failed=None):
    out = Path(config["output"])
    folder = out/"run"
    folder.mkdir(exist_ok=True)
    state_path = folder/"state.json"
    state_path.unlink(missing_ok=True)
    atomic_json(out/"full_report.json", dict(completion_status="RUNNING"))
    command = [str(python), "-u", str(Path(scripts)/"apc_worker.py"), "--manifest", str(out/"manifest.json")]
    if retry_failed is not None:
        command.append("--retry-failed" if retry_failed else "--no-retry-failed")
    started, failure = time.monotonic(), None
    with (folder/"worker.log").open("a", encoding="utf-8") as log:
        child = subprocess.Popen(command, env=env, cwd=config["repo"], stdout=log, stderr=subprocess.STDOUT)
        try:
            while child.poll() is None:
                try:
                    state = json.loads(state_path.read_text()) if state_path.exists() else {}
                except (OSError, ValueError):
                    state = {}
                elapsed = time.monotonic()-started
                count, total = state.get("processed", 0), state.get("total", 0)
                rate = state.get("new_processed", 0)/max(1, state.get("elapsed_seconds", 1))
                eta = round(max(0, total-count)/rate/3600, 2) if rate else "?"
                print("FULL", state.get("pipeline_stage", state.get("stage", "starting")),
                      state.get("paper_task", ""), state.get("sample_id", ""), f"{count}/{total or '?'}",
                      f"{100*count/total:.1f}%" if total else "",
                      "new errors=", state.get("errors", 0), "new invalid=", state.get("invalid", 0),
                      "elapsed h=", round(elapsed/3600, 2), "ETA h=", eta,
                      "free GiB=", state.get("disk_free_gib"), flush=True)
                sample_start = state.get("sample_started_unix")
                if sample_start and time.time()-sample_start > config["sample_timeout_seconds"]+120:
                    raise TimeoutError("Full sample watchdog exceeded timeout + 120s")
                if not state.get("new_processed") and not sample_start and elapsed > config["load_timeout_seconds"]+120:
                    raise TimeoutError("Full model initialization watchdog exceeded timeout")
                # Allow the worker to finish its current bounded sample, write the
                # partial report and exit normally at the next loop boundary.
                if elapsed > config["run_timeout_seconds"]+config["sample_timeout_seconds"]+120:
                    raise TimeoutError("Full session watchdog exceeded its export reserve")
                time.sleep(20)
            if child.returncode:
                with (folder/"worker.log").open("rb") as stream:
                    stream.seek(max(0, (folder/"worker.log").stat().st_size-16000))
                    print(stream.read().decode("utf-8", errors="replace"))
                raise RuntimeError(f"Full worker failed (exit {child.returncode}); inspect checkpoint and worker.log")
        except BaseException as exc:
            failure = exc
        finally:
            terminate_worker(child)
    if failure:
        try:
            existing = json.loads((out/"full_report.json").read_text())
            if existing.get("completion_status") == "ERROR":
                report = dict(existing, supervisor_error=str(failure))
                atomic_json(out/"full_report.json", report)
                atomic_json(out/"summary.json", report)
            else:
                report = write_full_report(out, failure=failure)
        except Exception as report_error:
            print("Report update failed:", repr(report_error))
    try:
        collect_reports(config["work"], out)
    except OSError as exc:
        print("Report collection:", repr(exc))
    try:
        export_full(out, config)
    except Exception as exc:
        print("Export failed:", repr(exc), "; raw files remain in Output")
    show_downloads(out)
    if failure:
        raise failure
    report = json.loads((out/"full_report.json").read_text())
    print("FULL", report["completion_status"], report["attempted_rows"], "/", report["expected_rows"],
          "technical=", report["technical_status"], flush=True)
    return report
