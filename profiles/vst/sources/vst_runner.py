"""VST-7B-RL inference, checkpointing and scoring for the SC203 3DSRBench subset."""
import base64
import csv
import hashlib
import io
import json
import os
import re
import shutil
import sys
import time
import traceback
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

MODEL = 'rayruiyang/VST-7B-RL'
MODEL_REV = 'b18ec54b43e1d0b8a171cf41d9f221fbedd803ae'
DATASET = 'ccvl/3DSRBench'
DATA_REV = '5c90e422b68d0484d50a3ff06b3096ea7d288026'
DATA_FILE = '3dsrbench_v1_vlmevalkit_circular.tsv'
CATEGORIES = {'orientation_on_the_left': 'left_right',
              'orientation_in_front_of': 'front_behind',
              'multi_object_facing': 'facing'}
COUNTS = {'left_right': 698, 'front_behind': 688, 'visibility': 688, 'facing': 692}
SYSTEM = ('You are a helpful assistant. You should first think about the reasoning process in the mind '
          'and then provide the user with the answer. The reasoning process is enclosed within '
          '<think> </think> tags, i.e. <think> reasoning process here </think> answer here.')
RULE = 'vst_final_option_v1'
MIB = 2**20


def json_bytes(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True).encode('utf8')


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def atomic(path, data):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_bytes(data)
    temporary.replace(path)


def write_json(path, value):
    atomic(path, json_bytes(value) + b'\n')


def check_disk(path, required, purpose):
    free = shutil.disk_usage(path).free
    if free < required:
        raise OSError(f'{purpose}: {free/2**30:.2f} GiB free, need {required/2**30:.2f} GiB on {path}.')


def norm(text):
    text = unicodedata.normalize('NFKC', text).casefold().strip().rstrip('.!?')
    return ' '.join(re.sub(r"""['’"“”]""", '', text).split())


def parse_answer(raw, choices, capped=False):
    if capped:
        return None, 'token_limit', ''
    tail = raw.rsplit('</think>', 1)[-1].strip()
    if re.search(r'<think\b', tail, flags=re.I):
        return None, 'unclosed_think', tail
    if not tail:
        return None, 'empty_final', tail
    tail = re.sub(r'^\s*(?:(?:the )?final answer|(?:the )?answer)\s*(?::|\bis\b)\s*', '', tail, flags=re.I)
    tagged = re.fullmatch(r'<answer>\s*(.*?)\s*</answer>\s*[.!]?', tail, flags=re.S | re.I)
    if tagged:
        tail = tagged[1].strip()
    boxed = re.fullmatch(r'\\boxed\{\s*(?:\\text\{([^{}]*)\}|([^{}]*))\s*\}\s*[.!]?', tail)
    if boxed:
        tail = (boxed[1] if boxed[1] is not None else boxed[2]).strip()
    tail = tail.strip().strip('*').strip()
    letter = re.fullmatch(r'(?:option\s+)?(?:\(([A-D])\)|([A-D]))[.!]?', tail, flags=re.I)
    if letter:
        pred = (letter[1] or letter[2]).upper()
        return (pred, 'letter', tail) if pred in choices else (None, 'letter_not_in_choices', tail)
    text = norm(tail)
    matches = []
    aliases = {'left': 'on the left', 'right': 'on the right',
               'in front': 'in front of', 'not visible': 'no', 'visible': 'yes'}
    for key, value in choices.items():
        option = norm(value)
        if text in (option, norm(f'{key}. {value}'), norm(f'{key}) {value}')) or aliases.get(text)==option:
            matches.append(key)
    return (matches[0], 'unique_option_text', tail) if len(matches)==1 else (None, 'ambiguous_or_unmatched', tail)


def visibility_version(row):
    match = re.fullmatch(r'(.*?facing where it is facing, )is (.+?) in front of me or behind me\?',
                         row['question'], re.I)
    labels = {'in front of': 'yes', 'behind': 'no'}
    if not match or set(row['choices'].values()) != set(labels):
        raise ValueError(f'Unknown visibility conversion for sample {row["sample_id"]}')
    return {**row, 'task': 'visibility', 'sample_key': 'visibility:'+row['sample_id'],
            'question': match[1]+'is '+match[2]+' visible or not?',
            'choices': {key: labels[value] for key, value in row['choices'].items()},
            'question_transform': 'front_behind_to_visibility_yes_no_v1'}


def tsv_rows(path):
    csv.field_size_limit(2**31-1)
    with path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream, delimiter='\t')
        required = {'index','qid','category','image','question','answer','A','B'}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError('Invalid Circular TSV columns')
        yield from reader


def prepare_samples(tsv, image_dir, mode, protocol):
    from PIL import Image
    # Selected question rows and all numeric references; no base64 kept in RAM.
    selected, references, direct = [], {}, set()
    for row in tsv_rows(tsv):
        sid, qid, value = str(row['index']), row['qid'], row['image'].strip()
        if value.isdigit():
            references[sid] = value
        else:
            direct.add(sid)
        category = row['category']
        if category not in CATEGORIES or (mode=='sanity' and sid not in ('5998','5999')):
            continue
        choices = {key: row[key] for key in 'ABCD' if row.get(key,'').strip() not in ('','nan')}
        task, answer = CATEGORIES[category], row['answer'].strip().upper()
        if set(choices) != {'A','B'} or answer not in choices:
            raise ValueError(f'Invalid choices/answer in {sid}')
        item = dict(sample_id=sid, sample_key=task+':'+sid, question_id=qid,
                    circular_group=re.sub(r'-[0-3]$', '', qid), category=category, task=task,
                    question=row['question'], choices=choices, answer=answer,
                    source_question=row['question'], source_choices=choices, question_transform='unchanged')
        if task!='front_behind' or protocol in ('original','both'):
            selected.append(item)
        if task=='front_behind' and protocol in ('visibility','both'):
            selected.append(visibility_version(item))
    sources = {}
    for item in selected:
        sid = item['sample_id']
        root, seen = sid, set()
        while root in references:
            if root in seen:
                raise ValueError(f'Cyclic image reference: {sid}')
            seen.add(root)
            root = references[root]
        if root not in direct:
            raise ValueError(f'Unresolved image reference: {sid}')
        sources[sid] = root
    needed = set(sources.values())
    image_dir.mkdir(parents=True, exist_ok=True)
    lookup = {}
    for row in tsv_rows(tsv):
        if row['index'] not in needed:
            continue
        raw = base64.b64decode(row['image'].strip().split(',')[-1], validate=True)
        digest = hashlib.sha256(raw).hexdigest()
        target = image_dir/(digest+'.img')
        if not target.is_file() or target.stat().st_size != len(raw):
            with Image.open(io.BytesIO(raw)) as photo:
                photo.verify()
            atomic(target, raw)
        lookup[row['index']] = digest
        needed.remove(row['index'])
        if not needed:
            break
    for item in selected:
        item['image_sha256'] = lookup[sources[item['sample_id']]]
    expected = {'left_right': 2} if mode=='sanity' else {
        task: count for task,count in COUNTS.items()
        if task not in ('front_behind','visibility') or protocol=='both'
        or (task=='front_behind' and protocol=='original') or (task=='visibility' and protocol=='visibility')}
    if Counter(r['task'] for r in selected) != expected:
        raise ValueError(f'Unexpected task counts; expected {expected}')
    if len({r['sample_key'] for r in selected}) != len(selected):
        raise ValueError('Duplicate sample_key')
    groups = defaultdict(list)
    for item in selected:
        groups[item['task'],item['circular_group']].append(item)
    for pair in groups.values():
        if len(pair)!=2:
            raise ValueError(f'Incomplete CircularEval group {pair[0]["circular_group"]}')
        a,b = pair
        if (a['image_sha256']!=b['image_sha256'] or a['question']!=b['question']
                or list(a['choices'].values())!=list(reversed(list(b['choices'].values())))
                or a['choices'][a['answer']]!=b['choices'][b['answer']]):
            raise ValueError(f'CircularEval mismatch in {a["sample_key"]}')
    if mode=='sanity' and len({r['image_sha256'] for r in selected})!=1:
        raise ValueError('Sanity pair must share one image')
    # Complete one real pair per task first; this doubles as the full-run pilot.
    by_task = defaultdict(list)
    for (task,_),pair in groups.items():
        by_task[task].append(pair)
    return [r for task in by_task.values() for r in task[0]] + [
        r for task in by_task.values() for pair in task[1:] for r in pair]


def download_source(cfg, work):
    from huggingface_hub import hf_hub_download, try_to_load_from_cache
    if cfg['dataset_input']:
        return Path(cfg['dataset_input'])
    cached = try_to_load_from_cache(DATASET, DATA_FILE, revision=DATA_REV, repo_type='dataset')
    if not isinstance(cached, str):
        check_disk(work, 1700*MIB, 'TSV download and selected images')
    return Path(hf_hub_download(DATASET, DATA_FILE, repo_type='dataset', revision=DATA_REV))


def model_path(cfg, work):
    from huggingface_hub import HfApi, snapshot_download, try_to_load_from_cache
    if cfg['model_input']:
        root = Path(cfg['model_input'])
    else:
        allowed = ('.json','.jinja','.txt','.safetensors')
        info = HfApi().model_info(MODEL, revision=MODEL_REV, files_metadata=True)
        if info.sha != MODEL_REV:
            raise RuntimeError('Unexpected model revision')
        names = [f.rfilename for f in info.siblings if '/' not in f.rfilename
                 and f.rfilename.endswith(allowed)]
        size = sum(f.size or 0 for f in info.siblings if f.rfilename in names and
                   not isinstance(try_to_load_from_cache(MODEL, f.rfilename, revision=MODEL_REV), str))
        check_disk(work, size+1024*MIB, 'Model snapshot; set MODEL_INPUT if disk is tight')
        root = Path(snapshot_download(MODEL, revision=MODEL_REV, allow_patterns=names, max_workers=2))
    config = json.loads((root/'config.json').read_text(encoding='utf8'))
    if config['model_type']!='qwen2_5_vl':
        raise ValueError('Expected the official VST-7B-RL Qwen2.5-VL checkpoint')
    index = json.loads((root/'model.safetensors.index.json').read_text(encoding='utf8'))
    shards = sorted(set(index['weight_map'].values()))
    if len(shards)!=4 or any(not (root/name).is_file() for name in shards):
        raise ValueError('All four official VST-7B-RL safetensor shards are required')
    if cfg['model_input']:
        files = [*shards, 'config.json', 'model.safetensors.index.json', 'tokenizer.json',
                 'preprocessor_config.json', 'tokenizer_config.json', 'chat_template.json',
                 'chat_template.jinja', 'generation_config.json']
        identity = {'source':'local', 'files_sha256': {n:sha256(root/n) for n in files if (root/n).is_file()}}
    else:
        identity = {'source':'huggingface', 'revision':MODEL_REV}
    return root, identity


def read_journal(path, repair=False):
    rows = {}
    if not path.is_file():
        return rows
    with path.open('rb+' if repair else 'rb') as stream:
        while True:
            start = stream.tell()
            line = stream.readline()
            if not line:
                break
            try:
                row = json.loads(line)
            except (ValueError, UnicodeDecodeError):
                if repair and not line.endswith(b'\n') and not stream.read(1):
                    stream.truncate(start)
                    break
                raise ValueError(f'Corrupt JSONL at offset {start}') from None
            key = row['sample_key']
            if key in rows:
                raise ValueError(f'Duplicate journal key: {key}')
            rows[key] = row
            if repair and not line.endswith(b'\n'):
                stream.seek(0, os.SEEK_END)
                stream.write(b'\n')
                break
    return rows


def restore(out, source, manifest):
    path, saved = out/'manifest.json', out/'predictions.jsonl'
    old = json.loads(path.read_text(encoding='utf8')) if path.is_file() else None
    if old is None and saved.is_file():
        raise ValueError('Predictions exist without manifest')
    if old and old['fingerprint']!=manifest['fingerprint']:
        raise ValueError('Output fingerprint mismatch; use another RUN_NAME for changed settings')
    if not saved.is_file() and source:
        src = Path(source)
        if src.is_dir():
            old = json.loads((src/'manifest.json').read_bytes())
            payload = (src/'predictions.jsonl').read_bytes()
        else:
            with ZipFile(src) as archive:
                old = json.loads(archive.read('manifest.json'))
                payload = archive.read('predictions.jsonl')
        if old['fingerprint']!=manifest['fingerprint']:
            raise ValueError('Resume fingerprint mismatch; use the identical model, data, and generation settings')
        write_json(path, manifest)
        atomic(saved, payload)
    write_json(path, manifest)
    if not saved.exists():
        saved.touch()
    return read_journal(saved, repair=True)


def score(samples, predictions):
    def metric(rows):
        good = lambda item: predictions.get(item['sample_key'],{}).get('prediction')==item['answer']
        groups = defaultdict(list)
        for row in rows:
            groups[row['task'],row['circular_group']].append(row)
        complete = [pair for pair in groups.values() if all(r['sample_key'] in predictions for r in pair)]
        correct = sum(map(good, rows))
        circular = sum(all(map(good,pair)) for pair in complete)
        observed = [predictions[r['sample_key']] for r in rows if r['sample_key'] in predictions]
        invalid = Counter(p['parse_reason'] for p in observed if p['status']=='invalid')
        return dict(expected_rows=len(rows), attempted_rows=len(observed),
                    missing_rows=len(rows)-len(observed), complete=len(observed)==len(rows),
                    correct_rows=correct, row_accuracy_pct=100*correct/len(rows),
                    accuracy_on_attempted_pct=100*correct/len(observed) if observed else None,
                    expected_groups=len(groups), completed_groups=len(complete), correct_groups=circular,
                    circular_accuracy_pct=100*circular/len(groups),
                    circular_on_completed_pct=100*circular/len(complete) if complete else None,
                    invalid_outputs=dict(invalid))
    return {**{task:metric([r for r in samples if r['task']==task])
               for task in dict.fromkeys(r['task'] for r in samples)},
            'overall_selected_protocols':metric(samples)}


def save_checkpoint(out, samples, records):
    metrics = score(samples, records)
    write_json(out/'summary.json', metrics)
    check_disk(out, 256*MIB, 'Result archive headroom')
    tmp = out/'checkpoint.zip.tmp'
    with ZipFile(tmp, 'w', compression=ZIP_DEFLATED, compresslevel=3) as archive:
        for name in ('manifest.json','samples.jsonl','predictions.jsonl','summary.json',
                     'runtime.json','worker.log','last_error.json'):
            if (out/name).is_file():
                archive.write(out/name, name)
    tmp.replace(out/'checkpoint.zip')
    return metrics


def run(cfg, out):
    import torch
    from PIL import Image
    from transformers import AutoProcessor, GenerationConfig, Qwen2_5_VLForConditionalGeneration
    from qwen_vl_utils import process_vision_info
    work = Path(cfg['work'])
    image_dir = work/'images'
    if torch.cuda.device_count()!=2 or any('T4' not in torch.cuda.get_device_name(i) for i in (0,1)):
        raise RuntimeError('Choose GPU T4 ×2 in Kaggle accelerator settings')
    for i in (0,1):
        torch.ones(16, device=f'cuda:{i}', dtype=torch.float16).sum().item()
    check_disk(out, 512*MIB, 'Output headroom before inference')
    tsv = download_source(cfg, work)
    check_disk(work, 512*MIB, 'Temporary image cache')
    data_digest = sha256(tsv)
    print('Preparing paired samples...', flush=True)
    samples = prepare_samples(tsv, image_dir, cfg['mode'], cfg['protocol'])
    print(f'Prepared {len(samples)} rows', flush=True)
    weights, model_identity = model_path(cfg, work)
    settings = dict(model=MODEL, model_identity=model_identity, data=DATASET, data_revision=DATA_REV,
                    data_sha256=data_digest, protocol=cfg['protocol'], mode=cfg['mode'],
                    max_new_tokens=cfg['max_new_tokens'], min_pixels=256*28*28,
                    max_pixels=cfg['max_image_tokens']*28*28, thinking=cfg['thinking'],
                    dtype='float16', attention='sdpa', decode='greedy',
                    parser=RULE, prompt_version='vst_mcq_v1', torch=torch.__version__)
    from importlib.metadata import version
    settings['libraries'] = {name:version(name) for name in ('transformers','accelerate','qwen-vl-utils','pillow')}
    manifest = {**settings, 'sample_count':len(samples),
                'fingerprint':hashlib.sha256(json_bytes([settings,samples])).hexdigest()}
    records = restore(out, cfg['resume_input'], manifest)
    planned = {r['sample_key']:r for r in samples}
    if any(k not in planned or r['fingerprint']!=manifest['fingerprint'] for k,r in records.items()):
        raise ValueError('Journal contains rows from a different run')
    atomic(out/'samples.jsonl', b''.join(json_bytes(r)+b'\n' for r in samples))
    pilot = samples[:2*len({r['task'] for r in samples})]

    def check_pilot():
        if not all(r['sample_key'] in records for r in pilot):
            return
        required = 2 if cfg['mode']=='sanity' else 1
        for task in {r['task'] for r in pilot}:
            valid = sum(records[r['sample_key']]['status']=='ok' for r in pilot if r['task']==task)
            if valid<required:
                raise RuntimeError(f'Pilot parser gate failed in {task}; review raw outputs before full run')

    if len(records)==len(samples):
        save_checkpoint(out,samples,records)
        check_pilot()
        return

    model = processor = None
    try:
        check_pilot()
        print('Loading one VST-7B-RL model across both T4s...', flush=True)
        processor = AutoProcessor.from_pretrained(
            weights, local_files_only=True, min_pixels=settings['min_pixels'],
            max_pixels=settings['max_pixels'], trust_remote_code=False)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            weights, local_files_only=True, dtype=torch.float16, attn_implementation='sdpa',
            device_map='balanced', max_memory={0:'12GiB',1:'12GiB','cpu':0},
            low_cpu_mem_usage=True, use_safetensors=True)
        model.eval()
        placements = set(str(device).replace('cuda:', '') for device in model.hf_device_map.values())
        if placements != {'0','1'}:
            raise RuntimeError(f'Expected both GPUs without offload; got {placements}')
        input_device = model.get_input_embeddings().weight.device
        generation = GenerationConfig(
            do_sample=False, use_cache=True, max_new_tokens=settings['max_new_tokens'],
            eos_token_id=model.generation_config.eos_token_id,
            pad_token_id=processor.tokenizer.pad_token_id,
            bos_token_id=model.generation_config.bos_token_id)
        stop_ids = generation.eos_token_id
        stop_ids = set(stop_ids if isinstance(stop_ids,(list,tuple)) else [stop_ids])
        runtime = dict(device_map=model.hf_device_map, input_device=str(input_device),
                       devices=[torch.cuda.get_device_name(i) for i in (0,1)],
                       cuda=torch.version.cuda, peak_allocated_mib={})
        write_json(out/'runtime.json', runtime)
        (out/'last_error.json').unlink(missing_ok=True)
        progress = 0
        with (out/'predictions.jsonl').open('ab') as journal, torch.inference_mode():
            for row in samples:
                key = row['sample_key']
                if key in records:
                    continue
                if time.time()>=cfg['deadline'] or (cfg['max_rows_this_session'] and progress>=cfg['max_rows_this_session']):
                    print('Session limit reached; publishing a resumable checkpoint.', flush=True)
                    break
                if progress%25==0:
                    check_disk(out, 512*MIB, 'Journal headroom')
                started = time.monotonic()
                with Image.open(image_dir/(row['image_sha256']+'.img')) as photo:
                    image = photo.convert('RGB')
                options = '\n'.join(f'{k}. {value}' for k,value in row['choices'].items())
                instruction = ('After </think>, write only the option letter.' if cfg['thinking']
                               else 'Answer with only the option letter.')
                prompt = row['question']+'\n\nOptions:\n'+options+'\n\n'+instruction
                messages = [{'role':'user','content':[
                    {'type':'image','image':image,'min_pixels':settings['min_pixels'],
                     'max_pixels':settings['max_pixels']}, {'type':'text','text':prompt}]}]
                if cfg['thinking']:
                    messages.insert(0, {'role':'system','content':[{'type':'text','text':SYSTEM}]})
                chat = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                images, videos = process_vision_info(messages)
                inputs = processor(text=[chat], images=images, videos=videos, return_tensors='pt').to(input_device)
                input_tokens = inputs.input_ids.shape[-1]
                output = model.generate(**inputs, generation_config=generation)
                generated = output[0,input_tokens:]
                tokens = generated.tolist()
                capped = len(tokens)>=settings['max_new_tokens'] and tokens[-1] not in stop_ids
                answer_tokens = tokens[:-1] if tokens and tokens[-1] in stop_ids else tokens
                # Keep reasoning delimiters even if a tokenizer marks them special.
                raw = processor.tokenizer.decode(answer_tokens, skip_special_tokens=False,
                                                 clean_up_tokenization_spaces=False)
                prediction, reason, final = parse_answer(raw, row['choices'], capped)
                record = dict(**row, fingerprint=manifest['fingerprint'], prediction=prediction,
                              status='ok' if prediction else 'invalid', parse_reason=reason,
                              correct=prediction==row['answer'], raw_output=raw, final_text=final,
                              prompt=prompt, prompt_tokens=input_tokens, new_tokens=len(tokens),
                              capped=capped, seconds=round(time.monotonic()-started,3))
                journal.write(json_bytes(record)+b'\n')
                journal.flush()
                os.fsync(journal.fileno())
                records[key] = record
                progress += 1
                del output, generated, inputs, images, videos, messages, image
                if progress<=len(pilot) or progress%25==0:
                    print(f'{len(records)}/{len(samples)} {key} {reason} {len(tokens)} tokens '
                          f'{record["seconds"]:.1f}s', flush=True)
                if key in {r['sample_key'] for r in pilot}:
                    check_pilot()
                if progress%100==0:
                    save_checkpoint(out, samples, records)
        runtime['peak_allocated_mib'] = {str(i):round(torch.cuda.max_memory_allocated(i)/MIB,1) for i in (0,1)}
        write_json(out/'runtime.json', runtime)
    except (Exception, KeyboardInterrupt) as error:
        write_json(out/'last_error.json', dict(error=repr(error), traceback=traceback.format_exc(),
                                              completed_rows=len(records)))
        print(f'Stopped after {len(records)} saved rows: {error}', flush=True)
        return False
    finally:
        metrics = save_checkpoint(out,samples,records)
        for task, result in metrics.items():
            print(f'{task}: {result["attempted_rows"]}/{result["expected_rows"]}, '
                  f'accuracy={result["row_accuracy_pct"]:.2f}%, '
                  f'CircularEval={result["circular_accuracy_pct"]:.2f}%', flush=True)
    return True


def main():
    import fcntl
    cfg = json.loads(Path(sys.argv[1]).read_text(encoding='utf8'))
    out = Path(cfg['out'])
    out.mkdir(parents=True, exist_ok=True)
    with (out/'.worker.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        return 2 if run(cfg,out) is False else 0


if __name__=='__main__':
    raise SystemExit(main())
