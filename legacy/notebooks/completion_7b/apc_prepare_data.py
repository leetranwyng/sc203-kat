"""Select APC sanity/full rows before decoding their images from the benchmark TSV."""

import base64
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import sys

from apc_common import choose_samples
from apc_kaggle_guardrails import atomic_json, check_resources, protocol_fingerprint


def metadata_from_tsv(tsv):
    rows, references = [], {}
    with Path(tsv).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        required = {"index", "qid", "question", "image", "answer", "category"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Unexpected TSV schema: " + str(reader.fieldnames))
        for source in reader:
            sid, qid = str(source["index"]), str(source["qid"])
            if sid in references:
                raise ValueError("Duplicate sample ID: " + sid)
            choices = {key: source[key] for key in "ABCD" if
                       str(source.get(key, "")).strip().lower() not in ("", "none", "nan", "null")}
            answer = source["answer"].strip().upper()
            if len(choices) < 2 or answer not in choices:
                raise ValueError("Invalid choices/answer: " + sid)
            value = source["image"].strip()
            references[sid] = value if re.fullmatch(r"\d+", value) else None
            rows.append(dict(sample_id=sid, question_id=qid, question=source["question"],
                             choices=choices, answer=answer, category=source["category"],
                             circular_group=re.sub(r"-[0-3]$", "", qid)))
    return rows, references


def resolve_reference(sid, references):
    seen = set()
    while True:
        if sid in seen or sid not in references:
            raise ValueError("Cyclic/missing image reference: " + sid)
        seen.add(sid)
        if references[sid] is None:
            return sid
        sid = references[sid]


def materialize_images(tsv, rows, references, folder, config):
    from PIL import Image
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    needed = {resolve_reference(row["sample_id"], references) for row in rows}
    payloads = {}
    with Path(tsv).open(encoding="utf-8-sig", newline="") as stream:
        for source in csv.DictReader(stream, delimiter="\t"):
            sid = str(source["index"])
            if sid not in needed:
                continue
            check_resources(folder, config)
            value = source["image"].strip()
            value = value.split(",", 1)[1] if value.startswith("data:image/") else value
            raw = base64.b64decode(value, validate=True)
            digest = hashlib.sha256(raw).hexdigest()
            target = folder/(digest+".img")
            if not target.exists() or target.stat().st_size != len(raw):
                with Image.open(io.BytesIO(raw)) as image:
                    image.verify()
                temporary = target.with_suffix(".tmp")
                temporary.write_bytes(raw)
                temporary.replace(target)
            payloads[sid] = (str(target), digest)
            if len(payloads) == len(needed):
                break
    if set(payloads) != needed:
        raise ValueError("Missing selected image payloads")
    for row in rows:
        row["image_path"], row["image_sha256"] = payloads[resolve_reference(row["sample_id"], references)]


def main(config_path):
    from huggingface_hub import HfApi, hf_hub_download
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    work, out = Path(config["work"]), Path(config["output"])
    out.mkdir(parents=True, exist_ok=True)
    csv.field_size_limit(2**31-1)
    if config.get("dataset_input"):
        tsv = Path(config["dataset_input"])
        digest = hashlib.sha256()
        with tsv.open("rb") as stream:
            for block in iter(lambda: stream.read(16*2**20), b""):
                digest.update(block)
        data_sha = "local:"+digest.hexdigest()
    else:
        info = HfApi().dataset_info(config["dataset_id"], revision=config["data_revision"], files_metadata=True)
        data_sha = info.sha
        entry = next(item for item in info.siblings if item.rfilename == config["data_file"])
        download = work/"data/tsv"
        download.mkdir(parents=True, exist_ok=True)
        target = download/config["data_file"]
        required = 0 if target.exists() and target.stat().st_size == entry.size else entry.size or 0
        from shutil import disk_usage
        if disk_usage(work).free < required + config["disk_min_free_gib"]*2**30:
            raise RuntimeError("Insufficient disk for TSV + reserve; set dataset_input to a mounted TSV.")
        tsv = Path(hf_hub_download(config["dataset_id"], config["data_file"], repo_type="dataset",
                                 revision=data_sha, local_dir=str(download)))
    all_rows, references = metadata_from_tsv(tsv)
    rows = choose_samples(all_rows, config)
    if not rows:
        raise ValueError("No selected rows")
    materialize_images(tsv, rows, references, work/"data/selected_images", config)
    assets = json.loads((work/"assets_ready.json").read_text(encoding="utf-8"))
    manifest = dict(config=config, data_sha=data_sha, assets=assets,
                    selected_rows=len(rows), total_dataset_rows=len(all_rows),
                    circular_groups=len({row["circular_group"] for row in rows}),
                    source_lock=json.loads((work/"source_lock.json").read_text(encoding="utf-8")),
                    patches=json.loads((work/"patches.json").read_text(encoding="utf-8")),
                    tasks=dict(Counter(row["paper_task"] for row in rows)),
                    protocol="APC-Vis FP16; front/behind recast to visibility; strict exact-choice; no LLM judge")
    manifest["fingerprint"] = protocol_fingerprint(manifest, rows)
    previous = out/"manifest.json"
    if previous.exists() and json.loads(previous.read_text())["fingerprint"] != manifest["fingerprint"]:
        raise RuntimeError("Existing experiment has another protocol; choose a new experiment name.")
    (out/"samples.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False)+"\n" for row in rows), encoding="utf-8")
    atomic_json(previous, manifest)
    print("Selected", len(rows), "rows:", manifest["tasks"], flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
