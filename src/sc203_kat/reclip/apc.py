"""Controlled APC-Vis instance selection replay with frozen detector proposals."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import signal
import subprocess
import tempfile
import zipfile
import sys
import time
import traceback

from sc203_kat.ablation.common import inside, read, rows, sha, write
from sc203_kat.ablation.run import find_bundle


SANITY_IDS = ('5832', '4798', '8268')
TASKS = {'left_right', 'visibility', 'facing'}
BRANCHES = {'both': ('baseline', 'reclip'), 'baseline': ('baseline',), 'reclip': ('reclip',)}


def atomic_rows(path, values):
    path = Path(path)
    temp = path.with_suffix('.tmp')
    with temp.open('w', encoding='utf-8') as stream:
        for value in values:
            stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')
    temp.replace(path)


def trace_bundle(cfg):
    """Accept an unpacked Kaggle dataset or its original trace ZIP."""
    hint = cfg.get('trace_input')
    root = Path(hint) if hint else Path('/kaggle/input')
    if not root.is_file():
        try:
            return find_bundle(hint)
        except ValueError:
            manifests = list(root.rglob('bundle.json'))
            if manifests:
                raise
            matches = list(root.rglob('sc203_apc_trace120.zip'))
            if len(matches) != 1:
                raise
            root = matches[0]
    if not zipfile.is_zipfile(root):
        raise ValueError('trace_input must name the mounted trace folder or original trace ZIP')
    cache = Path(cfg['runtime_dir']) / 'trace_bundle'
    cache.mkdir(parents=True, exist_ok=True)
    archive_id = sha(root)
    unpacked = cache / archive_id
    if not unpacked.exists():
        with zipfile.ZipFile(root) as archive, tempfile.TemporaryDirectory(dir=cache) as temp:
            candidates = [entry for entry in archive.infolist() if not entry.is_dir()
                          and PurePosixPath(entry.filename).name == 'bundle.json']
            if len(candidates) != 1:
                raise ValueError('Expected one bundle.json inside the trace ZIP')
            manifest = candidates[0]
            prefix = PurePosixPath(manifest.filename).parent
            meta = json.loads(archive.read(manifest))
            members = ['bundle.json', *meta['files']]
            for name in members:
                # Destination guard also rejects traversal hidden in the bundle manifest.
                target = inside(temp, name)
                info = archive.getinfo(str(prefix / name))
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(info) as source, target.open('wb') as destination:
                    shutil.copyfileobj(source, destination)
            Path(temp).replace(unpacked)
    return unpacked


def prepare(cfg, root):
    bundle = trace_bundle(cfg)
    metadata = read(bundle / 'bundle.json')
    for name, digest in metadata['files'].items():
        if sha(inside(bundle, name)) != digest:
            raise ValueError('Trace input checksum mismatch: ' + name)

    all_rows = rows(bundle / 'samples.jsonl')
    ids = {str(row['sample_id']) for row in all_rows}
    counts = {task: sum(row['paper_task'] == task for row in all_rows) for task in TASKS}
    if len(ids) != 120 or len(all_rows) != 120 or any(n != 40 for n in counts.values()):
        raise ValueError('Attach the existing 120-case APC trace dataset (40 cases per task).')
    if sum(row['historical_correct'] is True for row in all_rows) != 36:
        raise ValueError('Historical APC results do not match the 36/120 reference cohort.')
    if any(not sid.isascii() or not sid.isdigit() for sid in ids):
        raise ValueError('APC trace should contain numeric sample IDs.')
    default_ids = ids if cfg['mode'] == 'full' else set(SANITY_IDS)
    selected = set(map(str, cfg.get('sample_ids') or default_ids))
    if cfg['mode'] == 'full' and selected != ids:
        raise ValueError('Full mode must use the entire fixed 120-case cohort.')
    if not selected <= ids:
        raise ValueError('Required sanity sample IDs are absent from the trace bundle.')
    cohort = [row for row in all_rows if str(row['sample_id']) in selected]
    if cfg['mode'] == 'sanity' and (len(cohort) != 3 or {r['paper_task'] for r in cohort} != TASKS):
        raise ValueError('Sanity mode requires exactly one sample from each of the three tasks.')
    out = Path(cfg['output_dir'])
    out.mkdir(parents=True, exist_ok=True)
    identity = dict(
        dataset_sha=sha(bundle / 'bundle.json'),
        source_sha={name: sha(root / 'src/sc203_kat/reclip' / name) for name in ('apc.py', 'scorer.py')},
        repository_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
        sample_ids=[str(row['sample_id']) for row in cohort],
        seed=cfg['seed'], selector=cfg['selector'], model_input=cfg.get('model_input'),
    )
    lock = out / 'identity.json'
    if lock.is_file() and read(lock) != identity:
        raise ValueError('This output directory belongs to a different APC replay. Choose a new output_dir.')
    write(lock, identity)
    write(out / 'cohort.json', [dict(sample_id=str(r['sample_id']), task=r['paper_task']) for r in cohort])
    protocol = dict(
        method='apc_vis', trial='paired detector proposal reranking',
        frozen=['RGB image', 'object descriptions', 'reference viewer', 'ordered Grounding DINO proposals',
                'Depth Pro map', 'Qwen2.5-VL-7B FP16 checkpoint', 'OA1', 'renderer and prompt templates'],
        recomputed=['Qwen crop selection for the paired control', 'SAM mask', '3D position', 'OA1 orientation',
                    'viewer-frame transform', 'renderer', 'VLM final response'],
        proposals='Original APC normalized cxcywh proposals, in exact retained order; converted to integer pixel xyxy.',
        expression='Frozen object phrase, identical for both selectors. No question, choices or gold answer passed to ReCLIP.',
        selector_baseline='Native APC crop-grid prompt and native index parser in this runtime.',
        selector_reclip='Isolated crop-and-blur ReCLIP scoring; original proposal selected by stable index.',
        history='Saved APC predictions are a reference only. Compare ReCLIP against the same-environment baseline replay.',
        question_protocol='Use each frozen APC question and options as recorded; existing visibility rewrites are preserved.',
        seed_policy='Same per-ID seed before each replay and before baseline selector; deterministic model generation.',
        historical_reference_correct=36, historical_reference_total=120,
    )
    write(out / 'protocol.json', protocol)
    items = []
    for row in cohort:
        sid = str(row['sample_id'])
        trace = inside(bundle, row['trace_path'])
        relative_image = Path('inputs') / sid / 'input.png'
        image = out / relative_image
        # The row hash identifies original dataset bytes. APC stores a PNG re-encoding.
        image_digest = metadata['files'][(PurePosixPath(row['trace_path']) / 'input.png').as_posix()]
        if not image.is_file() or sha(image) != image_digest:
            image.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(trace / 'input.png', image)
        if sha(image) != image_digest:
            raise ValueError(f'Image digest differs for sample {sid}')
        items.append(dict(row, sample_id=sid, image=relative_image.as_posix(), trace=str(trace)))
    return items


def candidate_coordinates(boxes, image_size):
    """Mirror native APC float32 cxcywh conversion, preserving order and identity."""
    import numpy as np
    boxes = np.asarray(boxes, dtype=np.float32)
    if boxes.size == 0:
        return []
    if boxes.ndim != 2 or boxes.shape[1] != 4 or not np.isfinite(boxes).all():
        raise ValueError('Expected finite N×4 normalized APC proposals')
    center, extent = boxes[:, :2], boxes[:, 2:]
    corners = np.concatenate((center - extent / 2, center + extent / 2), axis=1)
    return (corners * np.array(image_size * 2, dtype=np.float32)).astype(int).tolist()


def sample_inputs(out, row):
    import numpy as np
    from PIL import Image

    source = Path(row['trace'])
    with Image.open(out / row['image']) as opened:
        image = opened.convert('RGB')
    viewer = read(source / 'scene.json')['viewer']
    names = read(source / 'get_objects_of_interest.json')['result'][0]
    if len(set(names)) != len(names) or (viewer != 'camera' and viewer not in names):
        raise ValueError(f'Cannot align frozen objects and viewer for {row["sample_id"]}')
    specs = []
    for index, name in enumerate(names):
        stem = f'object_{index}'
        if (source / f'{stem}.name').read_text(encoding='utf-8') != name:
            raise ValueError(f'APC proposal label does not match object {stem}: {name}')
        proposal_path = source / f'{stem}_boxes.npy'
        normalized = np.load(proposal_path, allow_pickle=False) if proposal_path.is_file() else np.empty((0, 4))
        historical_path = source / f'{stem}_selected_box.npy'
        historical = np.load(historical_path, allow_pickle=False).tolist() if historical_path.is_file() else None
        specs.append(dict(sample_id=row['sample_id'], task=row['paper_task'], object_index=index,
                          role='viewer' if name == viewer else 'target', expression=name,
                          proposals_xyxy=candidate_coordinates(normalized, image.size),
                          proposals_cxcywh=normalized.tolist(), historical_box=historical))
    return image, viewer, specs


def blank_selection(status='not_run'):
    return dict(status=status, index=None, box=None)


def score_phase(cfg):
    from .scorer import ReCLIPScorer

    out = Path(cfg['output_dir'])
    work = Path(cfg['runtime_dir'])
    scorer = ReCLIPScorer(device='cuda:0', cache_dir=work / 'reclip_cache', weight_dir=work / 'reclip_weights')
    try:
        for row in cfg['items']:
            image, _, specs = sample_inputs(out, row)
            for spec in specs:
                target = out / 'scores' / row['sample_id'] / f'{spec["object_index"]}.json'
                try:
                    scoring = scorer.rank_proposals(image, spec['expression'], spec['proposals_xyxy'])
                except Exception as exc:
                    scoring = dict(selected_index=None, selected_box=None, top2_margin=None, proposals=[],
                                   **exception_data(exc))
                write(target, dict(expression=spec['expression'], boxes=spec['proposals_xyxy'], scoring=scoring))
                if fatal_gpu_fault(scoring):
                    raise RuntimeError(f'ReCLIP GPU failed; see {target}')
            print(f'ReCLIP scored APC {row["sample_id"]}: {len(specs)} objects', flush=True)
        write(out / 'reclip_provenance.json', scorer.provenance)
    finally:
        scorer.release()


def select_baseline(experiment, image, spec, folder):
    import torch

    proposals = spec['proposals_xyxy']
    if not proposals:
        return blank_selection('no_proposals')
    boxes = torch.tensor(spec['proposals_cxcywh'], dtype=torch.float32)
    chosen = experiment.apc.detection_module.run_detection_refinement(
        experiment.vlm, image, spec['expression'], boxes_output=boxes,
        trace_save_dir=str(folder), visualize_trace=False,
    )
    raw = experiment.vlm.calls[-1]['raw_output']
    # The native selector returns only a box. Recover its exact index for trace provenance.
    matches = [index for index in range(min(experiment.apc.detection_module.num_candidates, len(proposals)))
               if str(index) in raw]
    selected_index = matches[0] if matches else 0
    chosen = [int(value) for value in chosen]
    if chosen != proposals[selected_index]:
        raise ValueError('Native box disagrees with the frozen candidate at the selected index')
    return dict(status='ok', index=selected_index, box=chosen, raw_output=raw, fallback_used=not matches)


def reclip_selection(out, spec):
    path = out / 'scores' / spec['sample_id'] / f'{spec["object_index"]}.json'
    saved = read(path)
    if saved['expression'] != spec['expression'] or saved['boxes'] != spec['proposals_xyxy']:
        raise ValueError('Cached scorer input differs from the frozen proposals')
    score = saved['scoring']
    selected = blank_selection(score['status'])
    if score['status'] == 'ok':
        index = score['selected_index']
        box = spec['proposals_xyxy'][index]
        if box != score['selected_box']:
            raise ValueError('Scorer index does not map back to the original pixel proposal')
        selected.update(index=index, box=box)
    return selected, score


def exception_data(exc):
    return dict(status='error', error_type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())


def fatal_gpu_fault(value):
    message = value.get('message', '').lower()
    return value.get('error_type') == 'OutOfMemoryError' or 'device-side assert' in message or 'cuda out of memory' in message


def selection_decisions(cfg, experiment, row, image, specs, folder):
    from sc203_kat.ablation.apc_worker import seed

    branches = BRANCHES[cfg['selector']]
    seed(row['sample_id'], cfg['seed'])
    os.environ['APC_TRACE_DIR'] = str(folder)
    experiment.vlm.calls = []
    decisions = []
    for spec in specs:
        baseline, alternative = blank_selection(), blank_selection()
        scoring = None
        if 'baseline' in branches:
            try:
                baseline = select_baseline(experiment, image, spec, folder)
            except Exception as exc:
                baseline.update(exception_data(exc))
                if fatal_gpu_fault(baseline):
                    raise
        if 'reclip' in branches:
            try:
                alternative, scoring = reclip_selection(Path(cfg['output_dir']), spec)
            except Exception as exc:
                alternative.update(exception_data(exc))
        decisions.append(dict(spec, baseline_box=baseline['box'], reclip_box=alternative['box'],
                              baseline_index=baseline['index'], reclip_index=alternative['index'],
                              baseline_status=baseline['status'], reclip_status=alternative['status'],
                              baseline=baseline, reclip=alternative, scoring=scoring,
                              proposal_source='frozen_apc_grounding_dino'))
    return decisions


def replay(experiment, row, image, viewer, decisions, branch, folder):
    import numpy as np
    import torch
    from apc_common import parse_option

    apc = experiment.apc
    detector = apc.detection_module
    by_name = {item['expression']: item for item in decisions}
    candidates = {name: torch.tensor(item['proposals_cxcywh'], dtype=torch.float32) for name, item in by_name.items()}
    if any(item[branch]['status'] not in ('ok', 'no_proposals') for item in decisions):
        raise ValueError(f'{branch} could not select a usable proposal for every detected object')

    def frozen_entities(image, prompt, conv_history=None):
        return list(by_name), conv_history

    def frozen_viewer(prompt, names, scene, conv_history=None):
        if viewer not in scene:
            raise ValueError(f'Frozen reference viewer was not reconstructed: {viewer}')
        return viewer, conv_history

    def chosen_box(vlm, image, category, boxes_output, **kwargs):
        if not torch.equal(boxes_output, candidates[category]):
            raise ValueError(f'Proposal sequence changed before selection for {category}')
        return by_name[category][branch]['box']

    patches = [(apc, 'get_objects_of_interest', frozen_entities),
               (apc, 'get_reference_viewer', frozen_viewer),
               (detector, 'run_detection', lambda image, category: candidates[category]),
               (detector, 'run_detection_refinement', chosen_box)]
    original_methods = [(owner, name, getattr(owner, name)) for owner, name, _ in patches]
    depth = np.load(Path(row['trace']) / 'depth.npy', allow_pickle=False)
    if depth.shape != (image.height, image.width):
        raise ValueError(f'Frozen depth shape {depth.shape} differs from RGB image size {image.size}')
    experiment.depth_cache = depth
    options = '\n'.join(f'{key}. {value}' for key, value in row['choices'].items())
    question = row['question'] + '\nAnswer options:\n' + options
    try:
        for owner, name, method in patches:
            setattr(owner, name, method)
        with torch.inference_mode():
            raw, _ = apc.run_apc(image, question, trace_save_dir=str(folder), perspective_prompt_type='visual',
                                 visualize_trace=False, visualize_scene_abstraction=False,
                                 render_whole_scene=row['category'] == 'orientation_on_the_left',
                                 return_conv_history=False, options=options)
        prediction = parse_option(raw, row['choices'])
        return dict(status='ok' if prediction is not None else 'invalid', prediction=prediction,
                    correct=prediction == row['answer'], raw_output=raw)
    finally:
        for owner, name, method in original_methods:
            setattr(owner, name, method)
        experiment.depth_cache = None


def publish_records(out, items):
    predictions, decisions = [], []
    for row in items:
        folder = out / 'checkpoints' / row['sample_id']
        if (folder / 'decision.json').is_file():
            decisions.extend(read(folder / 'decision.json'))
        if (folder / 'prediction.json').is_file():
            predictions.append(read(folder / 'prediction.json'))
    atomic_rows(out / 'grounding_decisions.jsonl', decisions)
    atomic_rows(out / 'replay_predictions.jsonl', predictions)


def replay_phase(cfg):
    import torch
    from sc203_kat.ablation.apc_worker import seed
    from sc203_kat.ablation.experiment import make

    out = Path(cfg['output_dir'])
    context = read(Path(cfg['runtime_dir']) / 'apc_context.json')
    os.chdir(context['repo'])
    sys.path[:0] = [context['scripts'], context['repo']]
    working = [row for row in cfg['items'] if not (out / 'checkpoints' / row['sample_id'] / 'prediction.json').is_file()]
    branches = BRANCHES[cfg['selector']]
    if not working:
        publish_records(out, cfg['items'])
        return
    stage = {'name': 'initializing'}

    def callback(name, **kwargs):
        stage['name'] = name

    def timeout(*_):
        raise TimeoutError('APC replay sample exceeded its time budget')

    signal.signal(signal.SIGALRM, timeout)
    experiment = make(cfg, context, callback)
    for row in working:
        if shutil.disk_usage(out).free < cfg['disk_min_free_gib'] * 2**30:
            raise RuntimeError('Low disk space; finished APC replay cases are checkpointed')
        sid = row['sample_id']
        checkpoint = out / 'checkpoints' / sid
        folder = out / 'traces' / sid
        folder.mkdir(parents=True, exist_ok=True)
        attempt = folder / f'attempt_{len(list(folder.glob("attempt_*"))) + 1:03d}'
        attempt.mkdir()
        image, viewer, specs = sample_inputs(out, row)
        selector_folder = attempt / 'selector'
        selector_folder.mkdir()
        signal.alarm(cfg['sample_timeout_seconds'])
        try:
            with torch.inference_mode():
                decisions = selection_decisions(cfg, experiment, row, image, specs, selector_folder)
        finally:
            signal.alarm(0)
        write(checkpoint / 'decision.json', decisions)
        prediction = {key: row[key] for key in ('sample_id', 'question', 'choices', 'answer', 'image',
                                                'historical_prediction', 'historical_correct', 'historical_status')}
        prediction.update(task=row['paper_task'], viewer=viewer, question_transform=row.get('question_transform'))
        for branch in branches:
            branch_folder = attempt / branch
            branch_folder.mkdir()
            experiment.vlm.calls = []
            experiment.sam_image_hash = None
            if hasattr(experiment, 'vision_scheduler'):
                experiment.vision_scheduler.park_all()
            os.environ['APC_TRACE_DIR'] = str(branch_folder)
            sample_seed = seed(sid, cfg['seed'])
            stage['name'] = 'selector'
            started = time.monotonic()
            signal.alarm(cfg['sample_timeout_seconds'])
            try:
                result = replay(experiment, row, image, viewer, decisions, branch, branch_folder)
            except Exception as exc:
                result = dict(prediction=None, correct=False, failed_stage=stage['name'], **exception_data(exc))
            finally:
                signal.alarm(0)
            render = branch_folder / 'apc_visual_prompt.png'
            result.update(render=render.relative_to(out).as_posix() if render.is_file() else None,
                          trace=branch_folder.relative_to(out).as_posix(), seed=sample_seed,
                          selector_reached=any(d[f'{branch}_status'] not in ('not_run', 'no_proposals') for d in decisions),
                          seconds=time.monotonic() - started, vlm_calls=experiment.vlm.calls)
            prediction[branch] = result
            if fatal_gpu_fault(result):
                write(checkpoint / 'gpu_fault.json', prediction)
                raise RuntimeError('APC GPU fault saved. Restart this Kaggle session before retrying.')
        write(checkpoint / 'prediction.json', prediction)
        publish_records(out, cfg['items'])
        print('APC replay', sid, {branch: prediction[branch]['status'] for branch in branches}, flush=True)


def run(cfg, root):
    from sc203_kat.ablation.setup import setup_apc
    from .scorer import CLIP_PACKAGE

    if sys.platform != 'linux':
        raise RuntimeError('Use Kaggle Linux with two GPUs for APC-Vis replay.')
    cfg.setdefault('selector', 'both')
    cfg.setdefault('mode', 'full')
    cfg.setdefault('seed', 42)
    cfg.setdefault('sample_timeout_seconds', 900)
    cfg.setdefault('disk_min_free_gib', 3)
    if cfg['mode'] not in ('full', 'sanity') or cfg['selector'] not in BRANCHES:
        raise ValueError('Set mode to full/sanity and selector to both/baseline/reclip')
    for key in ('runtime_dir', 'output_dir'):
        cfg[key] = str(Path(cfg[key]).resolve())
    cfg.update(pins=read(root / 'configs/ablations/upstreams.json'), needed_models=[], need_sam1=True, need_oa1=True)
    cfg['items'] = prepare(cfg, root)
    out, work = Path(cfg['output_dir']), Path(cfg['runtime_dir'])
    work.mkdir(parents=True, exist_ok=True)
    config = work / 'reclip_apc_config.json'
    write(config, cfg)
    state = out / 'state.json'
    write(state, dict(status='running', stage='setup'))
    try:
        context = setup_apc(cfg, root)
        env = dict(os.environ, **context['env'])
        env['UV_PYTHON_INSTALL_DIR'] = str(work / 'python_runtimes')
        if 'reclip' in BRANCHES[cfg['selector']]:
            write(state, dict(status='running', stage='clip_prepass'))
            subprocess.run([context['uv'], 'pip', 'install', '--python', context['py'], '--no-cache', 'ftfy==6.3.1'],
                           check=True, env=env)
            subprocess.run([context['uv'], 'pip', 'install', '--python', context['py'], '--no-cache', '--no-deps', CLIP_PACKAGE],
                           check=True, env=env)
            subprocess.run([context['py'], '-u', '-m', 'sc203_kat.reclip.apc', '--config', str(config), '--phase', 'score'],
                           check=True, env=env)
        write(state, dict(status='running', stage='apc_replay'))
        subprocess.run([context['py'], '-u', '-m', 'sc203_kat.reclip.apc', '--config', str(config), '--phase', 'replay'],
                       check=True, env=env)
        if cfg['mode'] == 'sanity':
            records = rows(out / 'replay_predictions.jsonl')
            decisions = rows(out / 'grounding_decisions.jsonl')
            if {r['task'] for r in records} != TASKS or any(
                r[branch]['status'] != 'ok' or not r[branch]['render']
                for r in records for branch in BRANCHES[cfg['selector']]
            ):
                raise RuntimeError('Sanity did not produce valid rendered answers on all three tasks; inspect traces.')
            if 'reclip' in BRANCHES[cfg['selector']] and not any(
                d['scoring'] and sum(p['rank'] is not None for p in d['scoring']['proposals']) >= 2 for d in decisions
            ):
                raise RuntimeError('Sanity did not exercise ReCLIP comparison on two or more valid proposals.')
    except BaseException as exc:
        previous_stage = read(state)['stage']
        write(state, dict(stage=previous_stage, **exception_data(exc)))
        raise
    publish_records(out, cfg['items'])
    write(state, dict(status='finished', stage='apc_replay'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--phase', choices=['run', 'score', 'replay'], default='run')
    args = parser.parse_args()
    cfg = read(args.config)
    if args.phase == 'score':
        score_phase(cfg)
    elif args.phase == 'replay':
        replay_phase(cfg)
    else:
        run(cfg, Path(__file__).resolve().parents[3])


if __name__ == '__main__':
    main()
