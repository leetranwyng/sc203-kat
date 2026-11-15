import base64
import csv
import json
import tempfile
import unittest
from pathlib import Path

from sc203_kat.datasets.three_dsrbench import load_tsv, validate_circular, visibility_version
from sc203_kat.evaluation.scoring import evaluate, normalize_record, pairwise
from sc203_kat.tracing.journal import Journal, fingerprint
from sc203_kat.analysis.html import render_report


def record(sid="0", model="APC-Vis", prediction="A", status="ok", **kwargs):
    value = dict(model=model, sample_id=sid, task="left_right", question_id="ABC" + ("-1" if sid == "1" else ""),
                 circular_group="ABC", question="Which side?", choices={"A": "left", "B": "right"},
                 answer="A", prediction=prediction, status=status, image_sha256="a"*64)
    value.update(kwargs)
    return value


class DatasetTests(unittest.TestCase):
    def test_visibility_is_independent_protocol(self):
        source = dict(record(), task="front_behind", choices={"A": "in front of", "B": "behind"},
                      question="If I stand at the dog's position facing where it is facing, is the tree in front of me or behind me?")
        converted = visibility_version(source)
        self.assertEqual(source["task"], "front_behind")
        self.assertEqual(converted["task"], "visibility")
        self.assertEqual(converted["choices"], {"A": "yes", "B": "no"})
        self.assertEqual(converted["answer"], "A")
        self.assertIn("visible or not", converted["question"])
        self.assertEqual(converted["source_question"], source["question"])

    def test_tsv_reference_and_flip_groups(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"input.tsv"
            fields = ["index", "qid", "category", "image", "question", "answer", "A", "B"]
            with path.open("w", newline="", encoding="utf8") as stream:
                writer = csv.DictWriter(stream, fields, delimiter="\t")
                writer.writeheader()
                for sid, qid, image, a, b, answer in [
                    ("0", "ABC", base64.b64encode(b"image").decode(), "left", "right", "A"),
                    ("1", "ABC-1", "0", "right", "left", "B"),
                    ("2", "ABC-flip", base64.b64encode(b"flipped").decode(), "left", "right", "B"),
                    ("3", "ABC-flip-1", "2", "right", "left", "A")]:
                    writer.writerow(dict(index=sid, qid=qid, category="orientation_on_the_left", image=image,
                                         question="Which side?", answer=answer, A=a, B=b))
            samples = load_tsv(path, tasks=["left_right"], image_dir=Path(tmp)/"images")
            self.assertEqual(len(samples), 4)
            self.assertEqual(samples[0]["image_sha256"], samples[1]["image_sha256"])
            self.assertNotEqual(samples[0]["image_sha256"], samples[2]["image_sha256"])
            self.assertEqual(samples[3]["circular_group"], "ABC-flip")
            self.assertEqual(len(list((Path(tmp)/"images").iterdir())), 2)

    def test_incomplete_group_rejected(self):
        with self.assertRaises(ValueError):
            validate_circular([record()])


class ScoreTests(unittest.TestCase):
    def test_errors_are_wrong_even_if_prediction_matches(self):
        for status in ["error", "invalid", "method_failure"]:
            row = normalize_record(record(status=status))
            self.assertFalse(row["correct"])
            self.assertEqual(row["outcome"], "Wrong")
            self.assertEqual(row["original_status"], status)

    def test_circular_requires_both_correct(self):
        rows = [record(), record("1", prediction="A", choices={"A": "right", "B": "left"}, answer="B")]
        score = evaluate(rows)[0]
        self.assertEqual(score["accuracy_pct"], 50)
        self.assertEqual(score["circular_accuracy_pct"], 0)
        self.assertEqual(score["wrong"], 1)

    def test_pairwise_counts_include_technical_wrong(self):
        rows = [record(), record(model="PCD", status="error")]
        result = pairwise(rows)[0]
        self.assertEqual(result["a_only_correct"], 1)
        self.assertEqual(result["n"], 1)

    def test_pairwise_rejects_input_mismatch(self):
        with self.assertRaises(ValueError):
            pairwise([record(), record(model="PCD", image_sha256="b"*64)])

    def test_duplicate_record_rejected(self):
        with self.assertRaises(ValueError):
            evaluate([record(), record()])

    def test_missing_prediction_counts_wrong(self):
        other = record("1", choices={"A": "right", "B": "left"}, answer="B")
        score = evaluate([record()], expected_samples=[record(), other])[0]
        self.assertEqual((score["rows"], score["correct"], score["wrong"], score["missing"]), (2, 1, 1, 1))
        self.assertEqual(score["circular_accuracy_pct"], 0)


class JournalTests(unittest.TestCase):
    def test_identity_changes_with_scientific_settings(self):
        self.assertNotEqual(fingerprint({"prompt": "a"}), fingerprint({"prompt": "b"}))
        self.assertEqual(fingerprint({"a": 1, "b": 2}), fingerprint({"b": 2, "a": 1}))

    def test_resume_and_torn_tail(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"predictions.jsonl"
            journal = Journal(path, "scientific-v1")
            journal.append(record())
            with path.open("ab") as stream:
                stream.write(b'{"sample_id":')
            reader = Journal(path, "scientific-v1")
            self.assertEqual(len(reader.read(repair_tail=True)), 1)
            self.assertTrue(path.read_bytes().endswith(b"\n"))
            self.assertEqual(reader.pending([record(), record("1")], model="APC-Vis")[0]["sample_id"], "1")
            with self.assertRaises(ValueError):
                reader.append(record())

    def test_wrong_fingerprint_and_interior_corruption_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"p.jsonl"
            Journal(path, "v1").append(record())
            with self.assertRaises(ValueError):
                Journal(path, "v2").read()
            path.write_bytes(b'broken\n{"a":1}')
            with self.assertRaises(ValueError):
                Journal(path, "v1").read(repair_tail=True)

    def test_html_escapes_script_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = render_report([record(raw_output="</script><script>alert(1)</script>")], Path(tmp)/"index.html")
            text = path.read_text(encoding="utf8")
            self.assertNotIn("</script><script>alert(1)", text)
            self.assertIn("\\u003c/script>", text)


if __name__ == "__main__":
    unittest.main()
