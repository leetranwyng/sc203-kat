"""Import the audited results_five7b records without changing original files.

``import_directory(source, destination, copy_images=False, copy_records=True)``
returns normalized records and writes predictions.jsonl, scores.json,
pairwise.json, manifest.json, and index.html under destination. By default image
links target existing local files; copy_images=True creates a portable gallery.
No model is loaded and no historic output is reparsed.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path

from sc203_kat.analysis.html import render_report
from sc203_kat.evaluation.scoring import evaluate, normalize_record, pairwise
from sc203_kat.tracing.journal import atomic_json


def import_directory(source, destination, copy_images=False, copy_records=True):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination or source in destination.parents:
        raise ValueError("Destination must be outside the source result tree")
    destination.mkdir(parents=True, exist_ok=True)
    cases_path = source / "cases.json"
    cases = json.loads(cases_path.read_text(encoding="utf8"))
    records = []
    for case in cases:
        image = (source / case["image"]).resolve()
        if copy_images:
            image_dest = destination / "media" / image.name
            image_dest.parent.mkdir(exist_ok=True)
            if not image_dest.exists():
                shutil.copy2(image, image_dest)
            image_link = image_dest.relative_to(destination).as_posix()
        else:
            image_link = image.as_uri()
        for model, result in case["models"].items():
            if not result.get("record"):
                continue
            record_path = (source / result["record"]).resolve()
            raw = json.loads(record_path.read_text(encoding="utf8"))
            if copy_records:
                safe_model = re.sub(r"[^a-zA-Z0-9_.-]+", "_", model)
                target = destination / "records" / safe_model / record_path.name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(record_path, target)
                record_link = target.relative_to(destination).as_posix()
            else:
                record_link = record_path.as_uri()
            canonical = dict(raw, sample_id=str(case["id"]), task=case["task"],
                             question=case["question"], choices=case["choices"],
                             answer=case["answer"], circular_group=case["group"],
                             image_sha256=raw.get("image_sha256") or image.stem)
            normalized = normalize_record(canonical, model=model)
            # Compact cross-model row; preserve the unmodified full record separately.
            row = {k: normalized.get(k) for k in (
                "model", "sample_id", "sample_key", "task", "question_id", "circular_group",
                "question", "choices", "answer", "prediction", "original_status", "correct",
                "outcome", "question_hash", "image_sha256", "question_transform",
                "raw_output", "seconds", "fingerprint", "reparse_rule")}
            row.update(image=image_link, record_path=record_link)
            if result.get("replay"):
                trace = source / "raw_final/apc/replay/run/trace/apc_vis" / case["id"] / "viewer.html"
                if trace.is_file():
                    row["trace_viewer"] = trace.as_uri()
            records.append(row)
    normalized_file = destination / "predictions.jsonl"
    with normalized_file.open("w", encoding="utf8") as stream:
        for row in records:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    atomic_json(destination / "scores.json", evaluate(records))
    atomic_json(destination / "pairwise.json", pairwise(records))
    atomic_json(destination / "manifest.json", dict(
        schema="sc203-kat-import-v1", source_cases_sha256=hashlib.sha256(cases_path.read_bytes()).hexdigest(),
        source=str(source), records=len(records), images_copied=copy_images,
        raw_records_copied=copy_records, trace_links="original local trace viewer paths",
        reparse_policy="Preserve audited predictions; no new parsing", error_policy="all non-correct outcomes count as wrong"))
    render_report(records, destination / "index.html")
    return records


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("destination")
    parser.add_argument("--copy-images", action="store_true")
    parser.add_argument("--link-records", action="store_true")
    args = parser.parse_args()
    rows = import_directory(args.source, args.destination, args.copy_images, not args.link_records)
    print(f"Imported {len(rows)} predictions")
