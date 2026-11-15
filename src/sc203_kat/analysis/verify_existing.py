"""Verify normalized scoring against audited five-model result summaries.

Usage: python -m sc203_kat.analysis.verify_existing /path/to/results_five7b
This reads compact audited case records and checks all row/group/pairwise counts;
it does not load models, change predictions, or modify the original results.
"""
import json
from pathlib import Path

from sc203_kat.evaluation.scoring import evaluate, pairwise


def verify(source):
    source = Path(source)
    cases = json.loads((source / "cases.json").read_text(encoding="utf8"))
    rows = []
    for case in cases:
        for model, prediction in case["models"].items():
            if not prediction.get("record"):
                continue
            state = prediction.get("original_state", prediction["state"])
            status = {"Error": "error", "Invalid": "invalid", "Failure": "method_failure"}.get(state, "ok")
            rows.append(dict(model=model, sample_id=case["id"], task=case["task"],
                             question_id=case["group"] + ":" + case["id"], circular_group=case["group"],
                             question=case["question"], choices=case["choices"], answer=case["answer"],
                             prediction=prediction.get("prediction"), status=status,
                             image_sha256=Path(case["image"]).stem))
    expected = {(r["model"], r["task"]): r for r in json.loads((source / "accuracy.json").read_text(encoding="utf8")) if r["task"] != "overall_own_scope"}
    actual = evaluate(rows)
    for score in actual:
        reference = expected[score["model"], score["task"]]
        if (score["correct"], score["wrong"], score["correct_groups"]) != (reference["correct"], reference["wrong"], reference["correct_groups"]):
            raise ValueError(f"Score regression: {score['model']} {score['task']}")
    expected_pairs = {(r["task"], frozenset([r["a"], r["b"]]), r["unit"]): r for r in json.loads((source / "pairwise.json").read_text(encoding="utf8"))}
    actual_pairs = pairwise(rows)
    for pair in actual_pairs:
        reference = expected_pairs[pair["task"], frozenset([pair["a"], pair["b"]]), pair["unit"]]
        fields = ("n", "both_correct", "both_wrong")
        if any(pair[k] != reference[k] for k in fields):
            raise ValueError(f"Pairwise regression: {pair['a']} {pair['b']} {pair['task']}")
        other_key = "a_only_correct" if pair["a"] == reference["a"] else "b_only_correct"
        if pair["a_only_correct"] != reference[other_key]:
            raise ValueError("Pairwise exclusive-correct regression")
    return dict(records=len(rows), score_summaries=len(actual), pairwise_summaries=len(actual_pairs), status="passed")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    print(json.dumps(verify(parser.parse_args().source), indent=2))
