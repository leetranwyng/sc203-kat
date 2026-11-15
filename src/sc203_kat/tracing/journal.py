"""Durable append-only prediction journal with explicit scientific identity."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


def fingerprint(scientific_config):
    """Hash the caller-supplied scientific config, never secrets or runtime paths.

    Include source/model/data revisions, prompts, preprocessing and generation.
    Hardware UUIDs, session budgets and trace budgets belong in runtime metadata.
    """
    return hashlib.sha256(json.dumps(scientific_config, sort_keys=True,
                                     ensure_ascii=False, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def atomic_json(path, value):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf8") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(target)


class Journal:
    """One journal per worker. Never share a journal between concurrent writers.

    read(repair_tail=True) repairs only an invalid final unterminated line.
    Any interior corruption, fingerprint mismatch or duplicate key raises.
    """
    def __init__(self, path, fingerprint):
        self.path = Path(path)
        self.fingerprint = fingerprint
        self._keys = None
        self._needs_newline = False

    @staticmethod
    def key(record):
        return (record.get("model"), record.get("task", record.get("paper_task")), str(record["sample_id"]))

    def read(self, repair_tail=False):
        if not self.path.exists():
            self._keys = set()
            return []
        data = self.path.read_bytes()
        lines = data.splitlines(keepends=True)
        records, keys, offset = [], set(), 0
        for index, line in enumerate(lines):
            try:
                record = json.loads(line)
            except (ValueError, UnicodeError):
                is_torn_tail = index == len(lines) - 1 and not line.endswith((b"\n", b"\r"))
                if repair_tail and is_torn_tail:
                    with self.path.open("r+b") as stream:
                        stream.truncate(offset)
                    break
                raise ValueError(f"Invalid journal line {index + 1}")
            if record.get("fingerprint") != self.fingerprint:
                raise ValueError(f"Journal fingerprint mismatch at line {index + 1}")
            key = self.key(record)
            if key in keys:
                raise ValueError(f"Duplicate journal record: {key}")
            keys.add(key)
            records.append(record)
            offset += len(line)
        self._keys = keys
        self._needs_newline = bool(offset and not data[:offset].endswith(b"\n"))
        return records

    def append(self, record):
        if self._keys is None:
            self.read(repair_tail=True)
        if record.get("fingerprint", self.fingerprint) != self.fingerprint:
            raise ValueError("Prediction fingerprint differs from journal")
        if self.key(record) in self._keys:
            raise ValueError(f"Duplicate journal record: {self.key(record)}")
        value = dict(record, fingerprint=self.fingerprint)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("ab") as stream:
            if self._needs_newline:
                stream.write(b"\n")
            stream.write((json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n").encode())
            stream.flush()
            os.fsync(stream.fileno())
        self._needs_newline = False
        self._keys.add(self.key(record))

    def pending(self, samples, model=None):
        done = {self.key(row) for row in self.read(repair_tail=True)}
        return [sample for sample in samples if self.key(dict(sample, model=model or sample.get("model"))) not in done]


class EventWriter:
    """Record each model/tool call start before invoking it; write end or error."""
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, **event):
        with self.path.open("a", encoding="utf8") as stream:
            stream.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
