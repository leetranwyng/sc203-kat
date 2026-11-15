"""Read the saved diagnostics for the paired APC/PCD gallery."""
import base64
import io
import json
import re
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image


def load_traces(root, output):
    replay = root / 'replay_results'
    media = output / 'trace'
    media.mkdir(exist_ok=True)
    html = (replay / 'replay_report.html').read_text(encoding='utf8')
    sections = re.split(r'data-case-id="([^"]+)"', html)
    overlays = {}
    for case, section in zip(sections[1::2], sections[2::2]):
        arrow = re.search(r'src="data:image/jpeg;base64,([^"]+)"><figcaption>Hướng APC', section)
        if arrow and case.isdigit():
            overlays[case] = base64.b64decode(arrow[1])
    apc = {}
    with ZipFile(replay / 'apc_step_c_replay.zip') as archive:
        paths = set(archive.namelist())
        for line in archive.read('apc_vis_predictions.jsonl').splitlines():
            row = json.loads(line)
            case = str(row['sample_id'])
            folder = f'full/trace/apc_vis/{case}/'
            result = dict(pred=row['prediction'], status=row['status'], boxes=[], vectors={})
            apc[case] = result
            def array(name):
                return np.load(io.BytesIO(archive.read(folder + name)))
            if folder+'input.png' in paths:
                result['size'] = Image.open(io.BytesIO(archive.read(folder+'input.png'))).size
            if folder+'detector.jsonl' in paths:
                result['boxes'] = [json.loads(x) for x in archive.read(folder+'detector.jsonl').splitlines()]
            for name in sorted(n for n in paths if n.startswith(folder+'object_') and n.endswith('.name')):
                index = Path(name).stem
                box = index+'_selected_box.npy'
                if folder+box in paths:
                    obj = archive.read(name).decode().strip()
                    result.setdefault('selected', {})[obj] = array(box).tolist()
            if folder+'scene.json' in paths:
                scene = json.loads(archive.read(folder+'scene.json'))
                result.update(viewer=scene['viewer'],vectors={k:v['orientation'] for k,v in scene['camera'].items() if k!='camera'})
            if result['vectors'] and case in overlays:
                result['arrow'] = f'trace/{case}_orient.jpg'
                (output / result['arrow']).write_bytes(overlays[case])
            if folder+'apc_visual_prompt.png' in paths:
                result['render'] = f'trace/{case}_render.png'
                (output / result['render']).write_bytes(archive.read(folder+'apc_visual_prompt.png'))
    with ZipFile(replay/'pcd_step_c_replay_results.zip') as archive:
        records = json.loads(archive.read('pcd_step_c_replay/predictions.json'))
    pcd = {str(r['row_id']):r for r in records}
    return apc, pcd
