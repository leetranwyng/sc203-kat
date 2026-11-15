"""Self-contained Kaggle orchestration; ML dependencies load only inside GPU workers."""

import base64
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time


CATEGORIES = ("orientation_on_the_left", "orientation_in_front_of")
LABELS = ("A", "B", "C", "D")
EXPORT_COLUMNS = [
    "index", "qid", "question", "A", "B", "C", "D", "prediction",
    "category", "col9", "col10", "answer", "hit",
]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def circular_id(qid):
    # Match the repo scorer: remove the option-rotation suffix, retain '-flip'.
    return re.sub(r"-\d$", "", str(qid))


def get_options(row):
    return {label: str(row.get(label, "")).strip() for label in LABELS
            if str(row.get(label, "")).strip()}


def read_benchmark(tsv_path):
    """Read the official circular TSV, including index-based image references."""
    csv.field_size_limit(2**30)
    images, rows, seen = {}, [], set()
    with Path(tsv_path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        required = {"index", "qid", "question", "answer", "category", "image", "A", "B"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"TSV is missing columns: {required - set(reader.fieldnames or [])}")
        for raw in reader:
            index = raw["index"].strip()
            if not index or index in seen:
                raise ValueError(f"Missing/duplicate TSV index: {index!r}")
            seen.add(index)
            images[index] = raw["image"].strip()
            if raw["category"] not in CATEGORIES:
                continue
            row = {key: raw.get(key, "").strip() for key in
                   ("qid", "question", "answer", "category", *LABELS)}
            row["row_id"] = index
            row["base_qid"] = circular_id(row["qid"])
            options = get_options(row)
            if len(options) < 2 or row["answer"] not in options:
                raise ValueError(f"Invalid options/answer for {row['qid']}")
            rows.append(row)
    if not rows:
        raise ValueError("No supported allocentric rows. Use the official circular TSV.")
    if len({row["qid"] for row in rows}) != len(rows):
        raise ValueError("Duplicate qid in the selected benchmark.")
    return rows, images


def resolve_image(index, images):
    seen = set()
    while True:
        if index in seen:
            raise ValueError(f"Cyclic image reference: {index}")
        seen.add(index)
        value = images[index]
        if value in images:
            index = value
            continue
        if not value:
            raise ValueError(f"Empty image for index {index}")
        if value.startswith("data:image/"):
            value = value.split(",", 1)[1]
        # Reject a short unresolved numeric index rather than base64-decoding it.
        if value.isdigit() and len(value) < 32:
            raise ValueError(f"Missing image reference: {value}")
        try:
            return base64.b64decode(value, validate=True)
        except (ValueError, base64.binascii.Error) as error:
            raise ValueError(f"Invalid embedded image for index {index}") from error


def validate_circular_groups(rows):
    groups = {}
    for row in rows:
        groups.setdefault(row["base_qid"], []).append(row)
    for qid, group in groups.items():
        first_options = get_options(group[0])
        expected = len(first_options)
        signatures = {tuple(get_options(row).values()) for row in group}
        answers = {get_options(row)[row["answer"]] for row in group}
        if (len(group) != expected or len(signatures) != expected or len(answers) != 1
                or len({row["category"] for row in group}) != 1
                or any(sorted(get_options(row).values()) != sorted(first_options.values())
                       for row in group)):
            raise ValueError(f"Incomplete/inconsistent circular group {qid}: {len(group)} rows, "
                             f"expected {expected}. Use the circular TSV without row limits.")
    return groups


def model_signature(model_dir):
    model_dir = Path(model_dir)
    weights = sorted(model_dir.glob("*.safetensors"))
    if not weights:
        raise ValueError(f"No safetensors weights in {model_dir}")
    files = sorted({*weights, *model_dir.glob("*.json"), *model_dir.glob("*.txt"),
                    *model_dir.glob("*.model"), *model_dir.glob("*.jinja")})
    # Content hashes also protect resume when a mounted local model changes.
    return {path.name: sha256_file(path) for path in files}


def runtime_versions():
    versions = {}
    for package in ("torch", "torchvision", "transformers", "accelerate", "pandas", "Pillow",
                    "tokenizers", "huggingface_hub", "safetensors"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def prepare_run(config):
    root = Path(config["output_dir"])
    root.mkdir(parents=True, exist_ok=True)
    image_dir = root / "images"
    image_dir.mkdir(exist_ok=True)
    all_rows, images = read_benchmark(config["data_path"])
    validate_circular_groups(all_rows)
    print("Allocentric rows (all rotations and flips):", len(all_rows))
    for category in CATEGORIES:
        subset = [row for row in all_rows if row["category"] == category]
        print(f"  {category}: {len(subset)} rows, "
              f"{len({row['base_qid'] for row in subset})} circular groups")
    if config["mode"] == "sanity":
        selected_qid = config.get("sanity_qid")
        rows = [row for row in all_rows if row["qid"] == selected_qid] if selected_qid else all_rows[:1]
        if len(rows) != 1:
            raise ValueError(f"Sanity qid must select exactly one row: {selected_qid!r}")
    else:
        rows = all_rows
    for row in rows:
        payload = resolve_image(row["row_id"], images)
        image_hash = hashlib.sha256(payload).hexdigest()
        image_path = image_dir / f"{image_hash}.img"
        if not image_path.exists():
            image_path.write_bytes(payload)
        row["image_path"] = str(image_path)
        row["image_sha256"] = image_hash
    del images
    groups = sorted({row["base_qid"] for row in rows})
    owners = {qid: index % config["num_workers"] for index, qid in enumerate(groups)}
    manifest = {
        "schema": 1, "mode": config["mode"], "categories": list(CATEGORIES),
        "alpha": config["alpha"], "max_pixels": config["max_pixels"],
        "min_pixels": config["min_pixels"], "compute_baseline": config["compute_baseline"],
        "num_workers": config["num_workers"], "seed": config["seed"],
        "dtype": "float16", "attention": "sdpa", "transformers": "4.51.3",
        "runtime_versions": runtime_versions(),
        "dataset_sha256": sha256_file(config["data_path"]),
        "model_files": model_signature(config["model_dir"]),
        "reference_sha256": sha256_file(config["reference_path"]),
        "runner_sha256": sha256_file(__file__),
        "rows": [{key: row[key] for key in ("row_id", "qid", "answer", "image_sha256")}
                 for row in rows],
    }
    # Paths and download locations do not change the scientific run identity.
    manifest["run_fingerprint"] = fingerprint(manifest)
    old_manifest = root / "run_manifest.json"
    if config.get("resume_from") and not old_manifest.exists():
        previous = Path(config["resume_from"])
        previous_manifest = read_json(previous / "run_manifest.json")
        if previous_manifest["run_fingerprint"] != manifest["run_fingerprint"]:
            raise ValueError("Resume manifest differs. Match the model, data, code, and settings.")
        for rank in range(config["num_workers"]):
            source = previous / f"predictions_gpu{rank}.jsonl"
            destination = root / source.name
            if source.exists():
                if destination.exists():
                    if sha256_file(source) == sha256_file(destination):
                        continue
                    raise ValueError(f"Refusing to overwrite existing checkpoint: {destination}")
                shutil.copy2(source, destination)
    if old_manifest.exists():
        if read_json(old_manifest)["run_fingerprint"] != manifest["run_fingerprint"]:
            raise ValueError("Existing run uses different settings. Choose a new OUTPUT_DIR.")
    elif any(root.glob("predictions_gpu*.jsonl")) and not config.get("resume_from"):
        raise ValueError("Predictions exist without a manifest; choose a new OUTPUT_DIR.")
    write_json(old_manifest, manifest)
    write_json(root / "config.json", config)
    for rank in range(config["num_workers"]):
        shard = rows if config["mode"] == "sanity" else [
            row for row in rows if owners[row["base_qid"]] == rank]
        write_json(root / f"plan_gpu{rank}.json", shard)
    print("Run fingerprint:", manifest["run_fingerprint"])
    return rows


def read_checkpoint(path, repair_tail=False):
    """Keep completed rows; only an interrupted, unterminated tail may be repaired."""
    path = Path(path)
    if not path.exists():
        return []
    records, end = [], 0
    with path.open("rb") as stream:
        while line := stream.readline():
            if not line.endswith(b"\n"):
                if not repair_tail:
                    raise ValueError(f"Interrupted final checkpoint line: {path}")
                with path.open("r+b") as writable:
                    writable.truncate(end)
                break
            try:
                records.append(json.loads(line))
            except (json.JSONDecodeError, UnicodeDecodeError) as error:
                raise ValueError(f"Corrupt checkpoint at byte {end}: {path}") from error
            end = stream.tell()
    if len({row["row_id"] for row in records}) != len(records):
        raise ValueError(f"Duplicate row_id in checkpoint: {path}")
    return records


def validate_records(rows, records, run_fingerprint):
    expected = {row["row_id"]: row for row in rows}
    for record in records:
        row = expected.get(record["row_id"])
        if row is None or record.get("run_fingerprint") != run_fingerprint:
            raise ValueError("Checkpoint row belongs to a different run/shard.")
        if any(record.get(key) != row[key] for key in ("qid", "answer", "category")):
            raise ValueError("Checkpoint question/answer/category differs from the plan.")
        for prefix in ("", "baseline_"):
            prediction = record.get(prefix + "prediction")
            if prediction is None and prefix:
                continue
            if (prediction not in get_options(row)
                    or record.get(prefix + "hit") != int(prediction == row["answer"])):
                raise ValueError(f"Invalid stored prediction/hit for {row['qid']}")


def parse_letter(text, labels):
    # Deliberately match process_category() in the reference implementation.
    letter = text.strip()[0].upper() if text.strip() else "A"
    fallback = letter not in labels
    return (labels[0] if fallback else letter), fallback or not text.strip()


def run_worker(config_path, rank):
    import importlib.util
    import torch
    import transformers
    from PIL import Image
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    config = read_json(config_path)
    root = Path(config["output_dir"])
    manifest = read_json(root / "run_manifest.json")
    if (sha256_file(__file__) != manifest["runner_sha256"]
            or sha256_file(config["reference_path"]) != manifest["reference_sha256"]
            or runtime_versions() != manifest["runtime_versions"]):
        raise RuntimeError("Code/runtime changed after preparation; prepare again in a new OUTPUT_DIR.")
    rows = read_json(root / f"plan_gpu{rank}.json")
    checkpoint = root / f"predictions_gpu{rank}.jsonl"
    completed = read_checkpoint(checkpoint, repair_tail=True)
    validate_records(rows, completed, manifest["run_fingerprint"])
    done = {record["row_id"] for record in completed}
    pending = [row for row in rows if row["row_id"] not in done]
    print(f"GPU worker {rank}: {len(done)}/{len(rows)} cached; {len(pending)} pending", flush=True)
    if not pending:
        return
    if transformers.__version__ != "4.51.3":
        raise RuntimeError("Install the pinned transformers version and restart the session.")
    if torch.cuda.device_count() != 2:
        raise RuntimeError("The single 7B worker must see both GPUs.")
    torch.cuda.set_device(0)
    torch.set_num_threads(2)
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda:0")
    print(f"Physical GPU {rank}: {torch.cuda.get_device_name(0)}; torch={torch.__version__}", flush=True)
    spec = importlib.util.spec_from_file_location("pcd_reference", config["reference_path"])
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    processor_kwargs = {key: config[key] for key in ("min_pixels", "max_pixels")
                        if config[key] is not None}
    processor = AutoProcessor.from_pretrained(config["model_dir"], local_files_only=True,
                                             **processor_kwargs)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        config["model_dir"], torch_dtype=torch.float16, device_map='auto',max_memory={0:'11GiB',1:'11GiB'},
        attn_implementation="sdpa", local_files_only=True, low_cpu_mem_usage=True,
    ).eval()
    device = model.get_input_embeddings().weight.device
    write_json(root/'model_device_map.json',model.hf_device_map)
    if any(str(value) in ('cpu','disk') for value in model.hf_device_map.values()):
        raise RuntimeError('Model did not fit both GPUs. Inspect model_device_map.json.')
    token_ids = {label: reference.get_candidate_token_ids_for_label(label, processor.tokenizer)
                 for label in LABELS}
    original_extract=reference.extract_objects_via_llm
    original_viewpoint=reference.check_same_viewpoint
    def logged_extract(question,category,*args,**kwargs):
        path=active_trace['folder']/'extraction.json'
        write_json(path,dict(question=question,category=category,status='started'))
        result=original_extract(question,category,*args,**kwargs)
        prompt,raw=reference.extraction_trace[(category,question)]
        write_json(path,dict(prompt=prompt,raw_output=raw,objects=result,status='complete'))
        return result
    def logged_viewpoint(image,ref_obj,category,*args,**kwargs):
        path=active_trace['folder']/'viewpoint.json'
        prompt=reference.build_viewpoint_check_prompt(ref_obj,category)
        write_json(path,dict(prompt=prompt,status='started'))
        raw,same=original_viewpoint(image,ref_obj,category,*args,**kwargs)
        write_json(path,dict(prompt=prompt,raw_output=raw,is_same_viewpoint=same,status='complete'))
        return raw,same
    reference.extract_objects_via_llm=logged_extract
    reference.check_same_viewpoint=logged_viewpoint
    original_generate=reference.generate_answer
    original_contrastive=reference.contrastive_pick_option
    active_trace={}
    def logged_generate(image,prompt,*args,**kwargs):
        path=active_trace['folder']/f"generate_{active_trace['index']:03d}.json"
        active_trace['index']+=1
        write_json(path,dict(prompt=prompt,status='started'))
        raw=original_generate(image,prompt,*args,**kwargs)
        write_json(path,dict(prompt=prompt,raw_output=raw,status='complete'))
        return raw
    def logged_contrastive(image,original_prompt,camera_prompt,*args,**kwargs):
        path=active_trace['folder']/f"scores_{active_trace['index']:03d}.json"
        active_trace['index']+=1
        write_json(path,dict(original_prompt=original_prompt,camera_prompt=camera_prompt,status='started'))
        prediction,scores=original_contrastive(image,original_prompt,camera_prompt,*args,**kwargs)
        write_json(path,dict(original_prompt=original_prompt,camera_prompt=camera_prompt,
                             prediction=prediction,scores=scores,status='complete'))
        return prediction,scores
    reference.generate_answer=logged_generate
    reference.contrastive_pick_option=logged_contrastive
    # Extraction is text-only; viewpoint cache keys include the actual (possibly flipped) image.
    extraction_cache, viewpoint_cache = {}, {}
    for record in completed:
        reference.extraction_trace[(record['category'],record['question'])] = (
            record.get('extraction_prompt',''),record.get('extraction_raw',''))
        extraction_cache[(record["category"], record["question"])] = (
            record["ref_obj"], record["target_obj"])
        viewpoint_cache[(record["category"], record["image_sha256"], record["ref_obj"])] = (
            record["viewpoint_raw"], record["is_same_viewpoint"])
    with checkpoint.open("a", encoding="utf-8") as sink, torch.inference_mode():
        for ordinal, row in enumerate(pending, 1):
            start = time.monotonic()
            trace=root/'trace'/str(row['row_id'])
            trace.mkdir(parents=True,exist_ok=True)
            active_trace.update(folder=trace,index=0)
            try:
                torch.cuda.reset_peak_memory_stats()
                with Image.open(row["image_path"]) as original_image:
                    image = original_image.convert("RGB")
                category, question = row["category"], row["question"]
                options = get_options(row)
                labels = list(options)
                extraction_key = (category, question)
                if extraction_key not in extraction_cache:
                    extraction_cache[extraction_key] = reference.extract_objects_via_llm(
                        question, category, model, processor, device)
                ref_obj, target_obj = extraction_cache[extraction_key]
                rewritten = reference.build_rewrite_q(ref_obj, target_obj, category)
                options_text = reference.options_to_text(options)
                original_prompt = reference.build_original_prompt(question, options_text)
                camera_prompt = reference.build_camera_prompt(rewritten, options_text)
                viewpoint_key = (category, row["image_sha256"], ref_obj)
                if viewpoint_key not in viewpoint_cache:
                    viewpoint_cache[viewpoint_key] = reference.check_same_viewpoint(
                        image, ref_obj, category, model, processor, device)
                viewpoint_raw, is_same = viewpoint_cache[viewpoint_key]
                scores, camera_raw, fallback = None, None, False
                if is_same:
                    camera_raw = reference.generate_answer(image, camera_prompt, model, processor, device)
                    prediction, fallback = parse_letter(camera_raw, labels)
                else:
                    prediction, scores = reference.contrastive_pick_option(
                        image, original_prompt, camera_prompt, labels, config["alpha"],
                        model, processor, device, token_ids)
                if scores is None:
                    _, scores = reference.contrastive_pick_option(
                        image, original_prompt, camera_prompt, labels, config["alpha"],
                        model, processor, device, token_ids)
                if scores is not None and not all(
                        math.isfinite(value) for candidate in scores.values() for value in candidate.values()):
                    raise FloatingPointError("Non-finite option scores; inspect model/dtype before resuming.")
                record = {**row, "run_fingerprint": manifest["run_fingerprint"],
                          "gpu_rank": rank, "ref_obj": ref_obj, "target_obj": target_obj,
                          "extraction_fallback": (ref_obj, target_obj) == ("object", "object"),
                          "rewrite_question": rewritten, "viewpoint_raw": viewpoint_raw,
                          "is_same_viewpoint": is_same, "branch": "camera_direct" if is_same else "pcd",
                          "camera_raw": camera_raw, "answer_parse_fallback": fallback,
                          "prediction": prediction, "hit": int(prediction == row["answer"]),
                          "scores": scores}
                if config["compute_baseline"]:
                    baseline_raw = reference.generate_answer(image, original_prompt, model, processor, device)
                    baseline, baseline_fallback = parse_letter(baseline_raw, labels)
                    record.update(baseline_raw=baseline_raw, baseline_prediction=baseline,
                                  baseline_hit=int(baseline == row["answer"]),
                                  baseline_parse_fallback=baseline_fallback)
                if True:
                    record.update(original_prompt=original_prompt, camera_prompt=camera_prompt,
                                  viewpoint_prompt=reference.build_viewpoint_check_prompt(ref_obj, category),
                                  option_token_ids=token_ids, image_size=list(image.size))
                if camera_raw is None:
                    camera_raw = reference.generate_answer(image,camera_prompt,model,processor,device)
                record["diagnostic_camera_prediction"]=parse_letter(camera_raw,labels)[0]
                record["diagnostic_camera_raw"]=camera_raw
                record["diagnostic_contrastive_prediction"]=max(labels,key=lambda label:scores[label]["contrastive"])
                record["extraction_prompt"],record["extraction_raw"]=reference.extraction_trace[extraction_key]
                # Counterfactual diagnostic: rotate the reference frame 180 degrees.
                reverse_question=question.replace(
                    "facing where it is facing", "facing in the direction opposite to where it is facing")
                if reverse_question == question:
                    raise ValueError('Cannot construct a 180-degree diagnostic for this question.')
                reverse_prompt=reference.build_original_prompt(reverse_question,reference.options_to_text(options))
                reverse_raw=reference.generate_answer(image,reverse_prompt,model,processor,device)
                record['reverse_direction_prompt']=reverse_prompt
                record['reverse_direction_raw']=reverse_raw
                record['reverse_direction_prediction']=parse_letter(reverse_raw,labels)[0]
                record['reverse_direction_answer']=None
                record['reverse_direction_protocol']='Reference frame rotated 180 degrees; unscored counterfactual.'
                opposite_viewpoint=reference.build_viewpoint_check_prompt(ref_obj,category)
                opposite_viewpoint=opposite_viewpoint.replace("back side","front side").replace("from behind","from the front")
                record['opposite_viewpoint_prompt']=opposite_viewpoint
                record['opposite_viewpoint_raw']=reference.generate_answer(image,opposite_viewpoint,model,processor,device)
                write_json(trace/'record.json',record)
                torch.cuda.synchronize()
                record["seconds"] = time.monotonic() - start
                record["peak_vram_gib"] = torch.cuda.max_memory_allocated() / 2**30
                sink.write(json.dumps(record, ensure_ascii=False) + "\n")
                sink.flush()
                os.fsync(sink.fileno())
                print(f"[{len(done) + ordinal}/{len(rows)}] {row['qid']} "
                      f"pred={prediction} answer={row['answer']} branch={record['branch']} "
                      f"time={record['seconds']:.2f}s vram={record['peak_vram_gib']:.2f}GiB", flush=True)
            except Exception as error:
                import traceback
                write_json(trace/'error.json',dict(error=repr(error),traceback=traceback.format_exc(),row_id=row['row_id']))
                write_json(root / f"error_gpu{rank}.json", {
                    "row_id": row["row_id"], "qid": row["qid"], "error": repr(error),
                    "hint": "Fix the error, then rerun. Completed JSONL rows are preserved. "
                            "If changing model/settings, use a new OUTPUT_DIR.",
                })
                raise


def launch_workers(config):
    root = Path(config["output_dir"])
    processes, streams = [], []
    try:
        for rank in range(config["num_workers"]):
            log_path = root / f"worker_gpu{rank}.log"
            stream = log_path.open("w", encoding="utf-8")
            streams.append(stream)
            environment = dict(os.environ, CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES","0,1"),
                               TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="2",
                               PYTHONUNBUFFERED="1")
            process = subprocess.Popen(
                [sys.executable, "-u", str(Path(__file__).resolve()), "--worker",
                 str(root / "config.json"), str(rank)], env=environment,
                stdout=stream, stderr=subprocess.STDOUT,
            )
            processes.append(process)
        while True:
            statuses = [process.poll() for process in processes]
            for rank, status in enumerate(statuses):
                if status not in (None, 0):
                    tail = "\n".join((root / f"worker_gpu{rank}.log").read_text(
                        encoding="utf-8", errors="replace").splitlines()[-30:])
                    raise RuntimeError(f"GPU {rank} exited with status {status}:\n{tail}")
            for rank in range(config["num_workers"]):
                log = (root / f"worker_gpu{rank}.log").read_text(encoding="utf-8", errors="replace")
                print(f"GPU {rank}: " + (log.splitlines()[-1] if log.splitlines() else "loading..."), flush=True)
            if all(status == 0 for status in statuses):
                break
            time.sleep(15)
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        for stream in streams:
            stream.close()


def compute_metrics(rows, records, include_baseline=True):
    groups = validate_circular_groups(rows)
    if len({record["row_id"] for record in records}) != len(records):
        raise ValueError("Duplicate predictions across shards.")
    by_id = {record["row_id"]: record for record in records}
    expected = {row["row_id"] for row in rows}
    if set(by_id) != expected:
        raise ValueError(f"Incomplete run: {len(expected - set(by_id))} missing, "
                         f"{len(set(by_id) - expected)} unexpected rows. Rerun workers to resume.")
    report = []
    for method, hit_key in [("pcd", "hit")] + ([("baseline", "baseline_hit")] if include_baseline else []):
        for category in ("overall", *CATEGORIES):
            selected = [row for row in rows if category == "overall" or row["category"] == category]
            selected_groups = [group for group in groups.values()
                               if category == "overall" or group[0]["category"] == category]
            correct = sum(all(by_id[row["row_id"]][hit_key] == 1 for row in group)
                          for group in selected_groups)
            report.append({"method": method, "category": category,
                           "rows": len(selected), "circular_groups": len(selected_groups),
                           "row_accuracy_pct": 100 * sum(by_id[row["row_id"]][hit_key]
                                                          for row in selected) / len(selected) if selected else None,
                           "circular_accuracy_pct": 100 * correct / len(selected_groups) if selected_groups else None})
    return report


def export_results(config):
    import pandas as pd

    root = Path(config["output_dir"])
    run_fingerprint = read_json(root / "run_manifest.json")["run_fingerprint"]
    all_records, rows = [], []
    for rank in range(config["num_workers"]):
        plan = read_json(root / f"plan_gpu{rank}.json")
        records = read_checkpoint(root / f"predictions_gpu{rank}.jsonl")
        validate_records(plan, records, run_fingerprint)
        if len(records) != len(plan):
            raise ValueError(f"GPU {rank} has {len(records)}/{len(plan)} rows; rerun workers.")
        all_records.extend(records)
        if config["mode"] != "sanity" or rank == 0:
            rows.extend(plan)
    if config["mode"] == "sanity":
        write_json(root / "sanity_results.json", all_records)
        return pd.DataFrame([{key: record.get(key) for key in (
            "gpu_rank", "qid", "branch", "prediction", "answer", "hit", "baseline_prediction",
            "viewpoint_raw", "extraction_fallback", "seconds", "peak_vram_gib")}
                             for record in all_records]), all_records
    metrics = compute_metrics(rows, all_records, config["compute_baseline"])
    metric_frame = pd.DataFrame(metrics)
    metric_frame.to_csv(root / "metrics.csv", index=False)
    write_json(root / "metrics.json", metrics)
    all_records.sort(key=lambda row: int(row["row_id"]))
    write_json(root / "predictions.json", all_records)
    pd.DataFrame(all_records).to_csv(root / "predictions.csv", index=False)
    alpha = str(config["alpha"]).replace(".", "p")
    model_name = f"PCD_Qwen2.5-VL-7B-Instruct_alpha{alpha}"
    export_dir = root / "outputs" / model_name
    export_dir.mkdir(parents=True, exist_ok=True)
    excel_rows = []
    for index, row in enumerate(all_records):
        excel_rows.append([index, row["qid"], row["question"], *[row[key] for key in LABELS],
                           row["prediction"], row["category"], row["rewrite_question"],
                           row["prediction"], row["answer"], row["hit"]])
    pd.DataFrame(excel_rows, columns=EXPORT_COLUMNS).to_excel(
        export_dir / f"{model_name}_3DSRBenchv1_openai_result.xlsx", index=False)
    write_json(export_dir / f"{model_name}_results.json", all_records)
    write_json(root / "diagnostics.json", {
        "rows": len(all_records),
        "extraction_fallback_rows": sum(row["extraction_fallback"] for row in all_records),
        "answer_parse_fallback_rows": sum(row["answer_parse_fallback"] for row in all_records),
        "camera_direct_rows": sum(row["is_same_viewpoint"] for row in all_records),
        "pcd_rows": sum(not row["is_same_viewpoint"] for row in all_records),
    })
    return metric_frame, all_records


if __name__ == "__main__":
    if len(sys.argv) != 4 or sys.argv[1] != "--worker":
        raise SystemExit("Usage: kaggle_runner.py --worker CONFIG_JSON RANK")
    run_worker(sys.argv[2], int(sys.argv[3]))