"""Normalize historic method records and score all non-correct outcomes as wrong."""
from __future__ import annotations

import itertools
import re
from collections import Counter, defaultdict

from sc203_kat.datasets.three_dsrbench import CATEGORIES, question_hash


def normalize_record(record, model=None):
    """Do not reparse text here; prediction must come from a versioned parser.

    Saved correctness booleans are not trusted. Their original values remain
    available in the input artifact rather than overriding the answer comparison.
    """
    row = dict(record)
    row["model"] = model or row.get("model")
    if not row["model"]:
        raise ValueError("A model identifier is required")
    row["sample_id"] = str(row.get("sample_id", row.get("row_id", "")))
    row["task"] = row.get("task", row.get("paper_task", CATEGORIES.get(row.get("category"))))
    if not row["sample_id"] or not row["task"]:
        raise ValueError("sample_id and task are required")
    row["question_id"] = row.get("question_id", row.get("qid", row["sample_id"]))
    row["circular_group"] = row.get("circular_group", row.get("base_qid", re.sub(r"-[0-3]$", "", row["question_id"])))
    row["choices"] = row.get("choices") or {k: row[k] for k in "ABCD" if row.get(k)}
    if row.get("answer") not in row["choices"]:
        raise ValueError(f"Missing/invalid gold answer: {row['sample_id']}")
    row.setdefault("question_transform", "unchanged")
    original_status = row.get("original_status", row.get("status", "ok"))
    row["original_status"] = original_status
    valid_status = original_status in ("ok", "success", "complete")
    row["correct"] = bool(valid_status and row.get("prediction") in row["choices"] and row["prediction"] == row["answer"])
    row["outcome"] = "Correct" if row["correct"] else "Wrong"
    row["sample_key"] = f"{row['task']}:{row['sample_id']}"
    row["question_hash"] = question_hash(row)
    return row


def _index(records):
    indexed = {}
    for raw in records:
        row = normalize_record(raw)
        key = (row["model"], row["task"], row["sample_id"])
        if key in indexed:
            raise ValueError(f"Duplicate prediction: {key}")
        indexed[key] = row
    return indexed


def evaluate(records, expected_samples=None):
    """Return a list of model/task summaries.

    expected_samples is an optional list of samples applying to all models.
    Missing predictions count as wrong when expected_samples is provided.
    Incomplete groups are reported and excluded from CircularEval unless the
    expected sample list defines the full group, in which case they are wrong.
    """
    indexed = _index(records)
    combinations = sorted({key[:2] for key in indexed})
    summaries = []
    for model, task in combinations:
        available = {sid: row for (m, t, sid), row in indexed.items() if (m, t) == (model, task)}
        samples = [dict(row) for row in expected_samples if row.get("task", row.get("paper_task")) == task] if expected_samples is not None else list(available.values())
        if len({str(s["sample_id"]) for s in samples}) != len(samples):
            raise ValueError(f"Duplicate expected sample in {task}")
        groups = defaultdict(list)
        correct, missing = 0, 0
        status_counts = Counter()
        for sample in samples:
            row = available.get(str(sample["sample_id"]))
            if row is None:
                missing += 1
                state = False
            else:
                if expected_samples is not None:
                    expected = normalize_record(dict(sample, model=model, prediction=None))
                    if expected["question_hash"] != row["question_hash"]:
                        raise ValueError(f"Prediction input differs from expected sample: {row['sample_key']}")
                state = row["correct"]
                status_counts[row["original_status"]] += 1
            correct += state
            group = sample.get("circular_group", re.sub(r"-[0-3]$", "", sample["question_id"]))
            groups[group].append(state)
        complete = [v for v in groups.values() if len(v) == 2]
        circular_correct = sum(all(v) for v in complete)
        summaries.append(dict(model=model, task=task, rows=len(samples), correct=correct,
                              wrong=len(samples) - correct, missing=missing,
                              accuracy_pct=round(100 * correct / len(samples), 4) if samples else None,
                              groups=len(complete), correct_groups=circular_correct,
                              incomplete_groups=sum(len(v) != 2 for v in groups.values()),
                              circular_accuracy_pct=round(100 * circular_correct / len(complete), 4) if complete else None,
                              original_status_counts=dict(status_counts)))
    return summaries


def pairwise(records):
    """Pairwise same-input row and complete two-permutation group outcomes."""
    indexed = _index(records)
    result = []
    for task in sorted({key[1] for key in indexed}):
        models = sorted({m for m, t, sid in indexed if t == task})
        for a, b in itertools.combinations(models, 2):
            aa = {sid: row for (m, t, sid), row in indexed.items() if (m, t) == (a, task)}
            bb = {sid: row for (m, t, sid), row in indexed.items() if (m, t) == (b, task)}
            shared = sorted(aa.keys() & bb.keys())
            groups = defaultdict(list)
            states = []
            for sid in shared:
                if aa[sid]["question_hash"] != bb[sid]["question_hash"]:
                    raise ValueError(f"Input mismatch between {a} and {b}: {task}:{sid}")
                state = (aa[sid]["correct"], bb[sid]["correct"])
                states.append(state)
                groups[aa[sid]["circular_group"]].append(state)
            for unit, pairs in [("rows", states), ("circular_groups", [(all(v[0] for v in g), all(v[1] for v in g)) for g in groups.values() if len(g) == 2])]:
                counts = Counter("both_correct" if x and y else "a_only_correct" if x else "b_only_correct" if y else "both_wrong" for x, y in pairs)
                result.append(dict(task=task, a=a, b=b, unit=unit, n=len(pairs),
                                   **{k: counts[k] for k in ("both_correct", "a_only_correct", "b_only_correct", "both_wrong")}))
    return result
