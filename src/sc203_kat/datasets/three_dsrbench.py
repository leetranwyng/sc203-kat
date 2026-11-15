"""Versioned 3DSRBench Circular TSV preparation, without ML dependencies.

``load_tsv(path, tasks=None, image_dir=None)`` returns sample dictionaries.
Images are decoded only when image_dir is supplied. Visibility is an explicit
protocol transformation, never an alias for the original Front/Behind task.
"""
from __future__ import annotations

import base64
import csv
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

CATEGORIES = {
    "orientation_on_the_left": "left_right",
    "orientation_in_front_of": "front_behind",
    "multi_object_facing": "facing",
}
TASKS = ("left_right", "front_behind", "visibility", "facing")
VISIBILITY_VERSION = "front_behind_to_visibility_yes_no_v1"


def sample_key(sample):
    return f"{sample.get('task', sample.get('paper_task'))}:{sample['sample_id']}"


def question_hash(sample):
    """Hash scientific input identity; exclude prediction, timing and file paths."""
    identity = {k: sample.get(k) for k in (
        "task", "sample_id", "question_id", "circular_group", "question",
        "choices", "answer", "image_sha256", "question_transform")}
    if not identity["image_sha256"]:
        identity["image_source_id"] = sample.get("image_source_id")
    return hashlib.sha256(json.dumps(identity, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def visibility_version(sample):
    if sample.get("task", sample.get("paper_task")) != "front_behind":
        raise ValueError("Visibility conversion requires an original Front/Behind sample")
    pattern = r"(.*?facing where it is facing, )is (.+?) in front of me or behind me\?"
    match = re.fullmatch(pattern, sample["question"], flags=re.I)
    labels = {"in front of": "yes", "behind": "no"}
    if not match or set(sample["choices"].values()) != set(labels):
        raise ValueError(f"Unsupported Visibility conversion: {sample['sample_id']}")
    result = dict(sample, task="visibility", question=match[1] + "is " + match[2] + " visible or not?",
                  choices={k: labels[v] for k, v in sample["choices"].items()},
                  source_question=sample.get("source_question", sample["question"]),
                  source_choices=sample.get("source_choices", sample["choices"]),
                  question_transform=VISIBILITY_VERSION)
    result["sample_key"] = sample_key(result)
    result["question_hash"] = question_hash(result)
    return result


def iter_tsv(path):
    csv.field_size_limit(2**31 - 1)
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        required = {"index", "qid", "category", "image", "question", "answer", "A", "B"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Not the 3DSRBench Circular TSV schema")
        yield from reader


def validate_circular(samples, expected_per_group=2):
    groups = defaultdict(list)
    keys = set()
    for sample in samples:
        key = sample_key(sample)
        if key in keys:
            raise ValueError(f"Duplicate sample: {key}")
        keys.add(key)
        groups[(sample["task"], sample["circular_group"])].append(sample)
    for key, rows in groups.items():
        if len(rows) != expected_per_group:
            raise ValueError(f"Incomplete CircularEval group {key}: {len(rows)} rows")
        reference = rows[0]
        for row in rows[1:]:
            if (row["question"] != reference["question"]
                    or row.get("image_sha256", row.get("image_source_id")) != reference.get("image_sha256", reference.get("image_source_id"))
                    or row["choices"][row["answer"]] != reference["choices"][reference["answer"]]
                    or sorted(row["choices"].values()) != sorted(reference["choices"].values())):
                raise ValueError(f"CircularEval input mismatch: {key}")
        if len(rows) == 2 and rows[0]["choices"] == rows[1]["choices"]:
            raise ValueError(f"Options were not permuted: {key}")
    return groups


def load_tsv(path, tasks=None, include_images=False, image_dir=None, validate=True):
    """Read selected tasks. Keep ``-flip`` groups distinct; strip only A/B suffix.

    include_images=True adds image_bytes (intended only for small callers).
    image_dir writes each decoded image once by SHA256 and sets image_path.
    """
    selected_tasks = set(TASKS if tasks is None else tasks)
    if not selected_tasks.issubset(TASKS):
        raise ValueError(f"Unknown tasks: {selected_tasks - set(TASKS)}")
    samples, references, direct = [], {}, set()
    for row in iter_tsv(path):
        sid, value = str(row["index"]), row["image"].strip()
        if value.isdigit():
            references[sid] = value
        else:
            direct.add(sid)
        task = CATEGORIES.get(row["category"])
        if task is None or (task not in selected_tasks and not (task == "front_behind" and "visibility" in selected_tasks)):
            continue
        choices = {k: row[k].strip() for k in "ABCD" if row.get(k, "").strip() not in ("", "nan")}
        answer = row["answer"].strip().upper()
        if answer not in choices:
            raise ValueError(f"Invalid answer/choices: {sid}")
        sample = dict(sample_id=sid, question_id=row["qid"],
                      circular_group=re.sub(r"-[0-3]$", "", row["qid"]),
                      category=row["category"], task=task, question=row["question"],
                      choices=choices, answer=answer, source_question=row["question"],
                      source_choices=dict(choices), question_transform="unchanged")
        if task in selected_tasks:
            samples.append(sample)
        if task == "front_behind" and "visibility" in selected_tasks:
            samples.append(visibility_version(sample))
    for sample in samples:
        root, seen = sample["sample_id"], set()
        while root in references:
            if root in seen:
                raise ValueError(f"Cyclic image reference: {sample['sample_id']}")
            seen.add(root)
            root = references[root]
        if root not in direct:
            raise ValueError(f"Unresolved image reference: {sample['sample_id']}")
        sample["image_source_id"] = root
    if include_images or image_dir is not None:
        needed = {s["image_source_id"] for s in samples}
        images = {}
        if image_dir is not None:
            Path(image_dir).mkdir(parents=True, exist_ok=True)
        for row in iter_tsv(path):
            sid = str(row["index"])
            if sid not in needed:
                continue
            raw = base64.b64decode(row["image"].strip().split(",")[-1], validate=True)
            digest = hashlib.sha256(raw).hexdigest()
            metadata = {"image_sha256": digest}
            if include_images:
                metadata["image_bytes"] = raw
            if image_dir is not None:
                target = Path(image_dir) / (digest + ".img")
                if not target.exists():
                    target.write_bytes(raw)
                elif hashlib.sha256(target.read_bytes()).hexdigest() != digest:
                    raise ValueError(f"Corrupt cached image: {target}")
                metadata["image_path"] = str(target)
            images[sid] = metadata
            needed.remove(sid)
            if not needed:
                break
        if needed:
            raise ValueError(f"Missing image sources: {sorted(needed)}")
        for sample in samples:
            sample.update(images[sample["image_source_id"]])
    for sample in samples:
        sample["sample_key"] = sample_key(sample)
        sample["question_hash"] = question_hash(sample)
    if validate:
        validate_circular(samples)
    return samples
