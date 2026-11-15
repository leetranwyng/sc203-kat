"""Small, dependency-free controls shared by the APC notebook and worker."""

from collections import Counter, defaultdict
from contextlib import contextmanager
import errno
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback
import zipfile

GIB = 2**30
TASKS = ("left_right", "visibility", "facing")
OPERATIONAL_KEYS = {
    "work", "repo", "output", "experiment", "uv", "asset_root", "asset_input",
    "dataset_input", "retry_failed", "resume_input", "sample_timeout_seconds",
    "load_timeout_seconds", "run_timeout_seconds", "disk_min_free_gib",
    "trace_max_total_mib", "trace_max_sample_mib", "trace_keep_per_task",
    "max_consecutive_errors", "max_total_errors", "archive_max_mib",
}


class ResourceLimitError(RuntimeError):
    pass


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def read_records(path, repair=False):
    """Repair only an interrupted final record, never an interior corruption."""
    path = Path(path)
    records = {}
    if not path.exists():
        return records
    with path.open("rb+" if repair else "rb") as stream:
        while True:
            start = stream.tell()
            line = stream.readline()
            if not line:
                break
            try:
                record = json.loads(line)
            except (ValueError, UnicodeDecodeError):
                if repair and not line.endswith(b"\n") and not stream.read():
                    stream.truncate(start)
                    break
                raise RuntimeError(f"Corrupt JSONL at {path}:{start}") from None
            records[str(record["sample_id"])] = record
            if repair and not line.endswith(b"\n"):
                stream.seek(0, 2)
                stream.write(b"\n")
    return records


def protocol_fingerprint(manifest, rows):
    config = {k: v for k, v in manifest["config"].items() if k not in OPERATIONAL_KEYS}
    samples = [{k: v for k, v in row.items() if k != "image_path"} for row in rows]
    assets = {k: v for k, v in manifest.get("assets", {}).items() if k != "root"}
    identity = dict(config=config, samples=samples, data_sha=manifest["data_sha"],
                    assets=assets, source_lock=manifest.get("source_lock"),
                    patches=manifest.get("patches"))
    return hashlib.sha256(json.dumps(identity, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def select_samples(rows, config):
    """Input has already received the paper task/question transformation."""
    tasks = config.get("paper_tasks", list(TASKS))
    rows = [row for row in rows if row.get("paper_task") in tasks]
    if config.get("run_kind") == "sanity":
        by_task = {task: {} for task in tasks}
        for row in rows:
            by_task[row["paper_task"]].setdefault(row["circular_group"], []).append(row)
        selected = []
        for task in tasks:
            groups = list(by_task[task].values())[:config.get("sanity_groups_per_task", 1)]
            if not groups or any(len(group) != 2 for group in groups):
                raise ValueError(f"Sanity needs complete 2-row circular groups for {task}")
            selected.append([row for group in groups for row in group])
        # Exercise all three code paths before the second option permutation.
        return [part[index] for index in range(max(map(len, selected)))
                for part in selected if index < len(part)]
    if config.get("limit") is not None:
        groups = {row["circular_group"] for row in rows[:config["limit"]]}
        rows = [row for row in rows if row["circular_group"] in groups]
    return rows


def tree_bytes(folder):
    return sum(path.stat().st_size for path in Path(folder).rglob("*")
               if path.is_file() and not path.is_symlink())


def check_resources(folder, config, trace=None):
    free = shutil.disk_usage(folder).free
    reserve = int(config["disk_min_free_gib"] * GIB)
    if free < reserve:
        raise ResourceLimitError(f"Disk low: {free/GIB:.2f} GiB free; reserve {reserve/GIB:.2f} GiB. "
                                 "Stopped before filling the filesystem.")
    if trace is not None and tree_bytes(trace) > config["trace_max_sample_mib"] * 2**20:
        raise ResourceLimitError("This sample exceeded trace_max_sample_mib; inspect its trace.")
    return free / GIB


def remove_trace(path, trace_root):
    """Only delete a sample child of this run's explicit trace root."""
    path, trace_root = Path(path), Path(trace_root).resolve()
    resolved = path.resolve()
    resolved.relative_to(trace_root)
    if resolved == trace_root or path.is_symlink():
        raise ValueError("Refusing to delete the trace root or a symlink")
    if resolved.exists():
        shutil.rmtree(resolved)


def retain_trace(trace, root, row, record, config, kept):
    task = row["paper_task"]
    wanted = config["run_kind"] == "sanity" or record["status"] != "ok" or kept[task] < config["trace_keep_per_task"]
    fits = tree_bytes(root) <= config["trace_max_total_mib"] * 2**20
    if wanted and fits:
        kept[task] += 1
        record["trace_path"] = str(Path(trace).relative_to(Path(root).parent))
        record["trace_bytes"] = tree_bytes(trace)
    else:
        remove_trace(trace, root)
        record["trace_path"] = None
        record["trace_dropped"] = "budget" if wanted else "retention_policy"


class FailureBudget:
    def __init__(self, config):
        self.config = config
        self.consecutive = self.total = 0
        self.identical = Counter()

    def observe(self, record):
        # Local failures are still persisted/scored, but do not stop full runs.
        if self.config["run_kind"] == "full" and record.get("error_type") in (
                "ReferenceViewerError", "InvalidAnswer"):
            self.consecutive = 0
            return None
        if record["status"] == "ok":
            self.consecutive = 0
            return None
        self.consecutive += 1
        self.total += 1
        signature = (record.get("error_type", record["status"]), record.get("message", "")[:300])
        self.identical[signature] += 1
        if self.consecutive >= self.config["max_consecutive_errors"]:
            return "consecutive_errors"
        if self.identical[signature] >= self.config["max_consecutive_errors"]:
            return "repeated_error"
        if self.total >= self.config["max_total_errors"]:
            return "total_errors"
        return None


def fatal_exception(exc):
    if isinstance(exc, (ResourceLimitError, NameError, AttributeError, ImportError, TypeError, AssertionError)):
        return True
    if isinstance(exc, MemoryError) or (isinstance(exc, OSError) and exc.errno in
            (errno.ENOSPC, errno.EDQUOT, errno.EIO)):
        return True
    text = str(exc).lower()
    return isinstance(exc, RuntimeError) and any(word in text for word in (
        "out of memory", "device-side assert", "illegal memory", "cuda error", "cublas_status"))


def successful(record):
    return record.get("status") == "ok" and record.get("parse_valid") is True


def pending_rows(rows, records, fingerprint, retry_failed):
    allowed = {str(row["sample_id"]) for row in rows}
    for sid, record in records.items():
        if sid not in allowed or record.get("fingerprint") != fingerprint or record.get("mode") != "apc_vis":
            raise RuntimeError("Resume mismatch: dataset/protocol/code differs from the checkpoint")
    return [row for row in rows if str(row["sample_id"]) not in records or
            retry_failed and not successful(records[str(row["sample_id"])])]


def make_report(rows, records, stop_reason=None):
    tasks, groups = defaultdict(list), defaultdict(list)
    for row in rows:
        tasks[row["paper_task"]].append(row)
        groups[row["circular_group"]].append(row)
    valid = [record for record in records.values() if successful(record)]
    def hit(row):
        record = records.get(str(row["sample_id"]), {})
        return successful(record) and record.get("prediction") == row["answer"]
    by_task = {}
    for task, samples in tasks.items():
        attempts = [records[str(row["sample_id"])] for row in samples if str(row["sample_id"]) in records]
        good = [r for r in attempts if successful(r)]
        by_task[task] = dict(expected=len(samples), attempted=len(attempts), valid=len(good),
                            status="PASS" if len(good) == len(samples) else "FAIL" if attempts else "NOT_RUN",
                            mean_seconds=sum(r["seconds"] for r in good)/len(good) if good else None)
    passed = len(valid) == len(rows) and not stop_reason
    return dict(technical_status="PASS" if passed else "FAIL", expected_rows=len(rows),
                attempted_rows=len(records), valid_predictions=len(valid),
                pipeline_errors=sum(r["status"] == "error" for r in records.values()),
                invalid_outputs=sum(r["status"] == "invalid" for r in records.values()),
                missing_rows=len(rows)-len(records), stop_reason=stop_reason, by_task=by_task,
                row_accuracy=sum(map(hit, rows))/len(rows),
                circular_accuracy=sum(all(hit(row) for row in group) for group in groups.values())/len(groups),
                accuracy_note="Sanity accuracy is diagnostic only; PASS tests execution, not correctness of every answer.")


def results_archive(out, config, debug=False):
    """Bound uncompressed bytes before writing. Never archive the weights/venv/images."""
    out = Path(out)
    archive = out / ("sanity_traces.zip" if debug else "results.zip")
    candidates = []
    for path in out.rglob("*"):
        if not path.is_file() or path.is_symlink():
            continue
        if path.name == "archive_status.json":
            continue  # Written after this export; never pack the previous attempt's status.
        relative = path.relative_to(out)
        if "trace" in relative.parts:
            if debug:
                candidates.append(path)
        elif not debug and path.suffix in (".json", ".jsonl", ".txt", ".log"):
            candidates.append(path)
    total = sum(path.stat().st_size for path in candidates)
    maximum = int((config["trace_max_total_mib"] if debug else config["archive_max_mib"]) * 2**20)
    free = shutil.disk_usage(out).free
    reserve = config["disk_min_free_gib"] * GIB
    if not candidates or total > maximum or free < reserve + total + 2**20:
        archive.unlink(missing_ok=True)  # Do not offer a stale ZIP from a previous attempt.
        return dict(status="SKIPPED", reason="empty, byte budget or free disk", source_bytes=total)
    temporary = archive.with_suffix(".zip.tmp")
    try:
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as zipped:
            for path in candidates:
                zipped.write(path, path.relative_to(out))
        temporary.replace(archive)
    except BaseException:
        temporary.unlink(missing_ok=True)
        archive.unlink(missing_ok=True)
        raise
    return dict(status="OK", path=str(archive), bytes=archive.stat().st_size, source_bytes=total)


def export_safely(out, config, debug=False):
    """An export failure must never hide the worker/setup exception."""
    status = {}
    for kind in ([False, True] if debug else [False]):
        key = "traces" if kind else "results"
        try:
            status[key] = results_archive(out, config, debug=kind)
        except Exception as exc:
            status[key] = dict(status="FAILED", error_type=type(exc).__name__, message=str(exc))
    try:
        atomic_json(Path(out)/"archive_status.json", status)
    except OSError:
        pass
    print("Archive:", json.dumps(status), flush=True)
    return status


def terminate_worker(child):
    if child.poll() is None:
        child.terminate()
    try:
        child.wait(timeout=15)
    except subprocess.TimeoutExpired:
        child.kill()
        child.wait()


@contextmanager
def setup_stage(name, out, config):
    started = time.monotonic()
    path = Path(out)/"setup_status.json"
    atomic_json(path, dict(stage=name, status="RUNNING"))
    try:
        yield
    except BaseException as exc:
        try:
            atomic_json(path, dict(stage=name, status="FAIL", seconds=time.monotonic()-started,
                                  error_type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc()))
        except OSError:
            pass
        try:
            from apc_launcher import collect_reports
            collect_reports(config["work"], out)
        except Exception:
            pass
        export_safely(out, config)
        print(f"Setup failed in {name}; inspect {path}", flush=True)
        try:
            from apc_launcher import show_downloads
            show_downloads(out)
        except Exception:
            pass
        raise
    else:
        atomic_json(path, dict(stage=name, status="PASS", seconds=time.monotonic()-started))


def run_logged(name, command, out, *, env, cwd=None):
    folder = Path(out)/"setup_logs"
    folder.mkdir(parents=True, exist_ok=True)
    log_path = folder/(name+".log")
    started = time.monotonic()
    with log_path.open("a", encoding="utf-8") as stream:
        child = subprocess.Popen(command, env=env, cwd=cwd, stdout=stream, stderr=subprocess.STDOUT)
        try:
            next_update = 0
            while child.poll() is None:
                elapsed = time.monotonic()-started
                if elapsed >= next_update:
                    print(f"{name}: {elapsed/60:.1f} min; log={log_path}", flush=True)
                    next_update = elapsed+20
                time.sleep(1)
        except BaseException:
            terminate_worker(child)
            raise
    if child.returncode:
        with log_path.open("rb") as stream:
            stream.seek(max(0, log_path.stat().st_size-16000))
            print(stream.read().decode("utf-8", errors="replace"))
        raise RuntimeError(f"{name} failed, exit={child.returncode}; see {log_path}")
    print(f"{name}: PASS ({time.monotonic()-started:.1f}s)", flush=True)
