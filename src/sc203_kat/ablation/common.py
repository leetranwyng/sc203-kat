import hashlib
import json
import math
import os
from pathlib import Path

VARIANTS = {
    'sam3': ('sam3', 'v1'),
    'oneref': ('oneref', 'v1'),
    'oa2': ('original', 'v2'),
    'sam3_oa2': ('sam3', 'v2'),
    'oneref_oa2': ('oneref', 'v2'),
    'sam3_oneref': ('sam3_oneref', 'v1'),
    'sam3_oneref_oa2': ('sam3_oneref', 'v2'),
}

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False,
                              default=lambda x: x.tolist()), encoding='utf-8')
    tmp.replace(path)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(2**20), b''): h.update(chunk)
    return h.hexdigest()

def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]

def append(path, value):
    with Path(path).open('a', encoding='utf-8') as f:
        f.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')
        f.flush(); os.fsync(f.fileno())

def inside(root, relative):
    root = Path(root).resolve(); path = (root / relative).resolve()
    if not path.is_relative_to(root): raise ValueError('Path escapes dataset: ' + str(relative))
    return path

def iou(a, b):
    inter = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    area = lambda x: max(0, x[2]-x[0])*max(0, x[3]-x[1])
    return inter / max(area(a)+area(b)-inter, 1e-12)

def select_oneref(candidates, predicted, min_iou=0.0):
    if not candidates: raise ValueError('No detection candidates')
    scores = [iou(box, predicted) for box in candidates]
    index = max(range(len(scores)), key=scores.__getitem__)
    if scores[index] <= min_iou:
        raise ValueError('OneRef box overlaps no eligible candidate; no default-index fallback')
    return index, scores

def xyxy_from_cxcywh(box, width, height):
    x, y, w, h = map(float, box)
    return [int((x-w/2)*width), int((y-h/2)*height),
            int((x+w/2)*width), int((y+h/2)*height)]

def cxcywh_from_xyxy(box, width, height):
    x1, y1, x2, y2 = box
    return [(x1+x2)/2/width, (y1+y2)/2/height, (x2-x1)/width, (y2-y1)/height]

def oa2_direction(azimuth, elevation, roll):
    """OA2 Rx(roll) Ry(elevation) Rz(-azimuth), semantic +X -> APC (y,-z,-x).

    At roll=0 this equals APC's original orientation_to_direction exactly.
    This is a coordinate adapter, not ground-truth orientation validation.
    """
    a, e, r = map(math.radians, (azimuth, elevation, roll))
    return [-math.cos(r)*math.sin(a) + math.sin(r)*math.sin(e)*math.cos(a),
            math.sin(r)*math.sin(a) + math.cos(r)*math.sin(e)*math.cos(a),
            -math.cos(e)*math.cos(a)]

def unpad_oneref(box, width, height, size=384):
    ratio = size/max(width, height)
    nw, nh = round(width*ratio), round(height*ratio)
    left, top = round((size-nw)/2-.1), round((size-nh)/2-.1)
    x1,y1,x2,y2 = xyxy_from_cxcywh_float(box, size, size)
    return [max(0., min(width, (x1-left)/ratio)), max(0., min(height, (y1-top)/ratio)),
            max(0., min(width, (x2-left)/ratio)), max(0., min(height, (y2-top)/ratio))]

def xyxy_from_cxcywh_float(box, width, height):
    x,y,w,h = map(float,box)
    return [(x-w/2)*width,(y-h/2)*height,(x+w/2)*width,(y+h/2)*height]
