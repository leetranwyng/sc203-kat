"""Lossless, content-addressed trace storage; no model code in this module."""
import functools
import hashlib
import inspect
import io
import json
import os
import time
import traceback
from pathlib import Path
import numpy as np
from PIL import Image

ACTIVE = None

def artifact_ref(value, kind):
    """Only tagged storage references are artifacts, not arbitrary image fields."""
    if not isinstance(value, dict): return False
    path = value.get(kind)
    suffix = '.png' if kind == 'image' else '.npz'
    if not isinstance(path, str) or not path.startswith('artifacts/') or not path.endswith(suffix):
        return False
    digest = path[len('artifacts/'):-len(suffix)]
    return len(digest) == 64 and all(c in '0123456789abcdef' for c in digest)

def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')
    temp.replace(path)

class Trace:
    def __init__(self, root, folder):
        self.root, self.folder = Path(root), Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        (self.root/'artifacts').mkdir(exist_ok=True)
        self.stack, self.counter = [], 0
        self.last_error = None
        self.started = time.time()
        self.preview_data = {}

    def blob(self, data, suffix):
        name = hashlib.sha256(data).hexdigest() + suffix
        p = self.root/'artifacts'/name
        if not p.exists():
            tmp = p.with_suffix(suffix+'.tmp')
            tmp.write_bytes(data)
            tmp.replace(p)
        return 'artifacts/'+name

    def pack(self, v):
        if isinstance(v, Image.Image):
            b = io.BytesIO(); v.save(b, format='PNG')
            return {'image': self.blob(b.getvalue(), '.png'), 'size': list(v.size), 'mode': v.mode}
        # Save complete input tensors, including the detector image tensor.
        if type(v).__module__.startswith('torch') and hasattr(v, 'detach'):
            return self.pack(v.detach().cpu().numpy())
        if isinstance(v, np.ndarray):
            if v.size <= 256:
                return self.pack(v.tolist())
            b = io.BytesIO(); np.savez_compressed(b, value=v)
            return {'array': self.blob(b.getvalue(), '.npz'), 'shape': list(v.shape), 'dtype': str(v.dtype)}
        if isinstance(v, np.generic): return self.pack(v.item())
        if isinstance(v, float) and not np.isfinite(v): return {'nonfinite': str(v)}
        if isinstance(v, (str, int, float, bool)) or v is None: return v
        if isinstance(v, Path): return str(v)
        if isinstance(v, dict): return {str(k): self.pack(x) for k,x in v.items() if k != 'conv_history'}
        if isinstance(v, (list, tuple)): return [self.pack(x) for x in v]
        if hasattr(v, 'to_dict'): return self.pack(v.to_dict())
        return {'component_type': type(v).__name__}

    def event(self, **kw):
        record = dict(timestamp=time.time(), elapsed=time.time()-self.started, **kw)
        with (self.folder/'calls.jsonl').open('a', encoding='utf8') as f:
            f.write(json.dumps(record, ensure_ascii=False, allow_nan=False)+'\n')
            f.flush()

    def snapshot(self, label, values):
        self.event(phase='snapshot', component=label, parent=self.stack[-1] if self.stack else None,
                   values=self.pack(values))

    def hook(self, owner, name, label=None):
        original = getattr(owner, name)
        signature = inspect.signature(original)
        @functools.wraps(original)
        def wrapped(*a, **kw):
            if ACTIVE is None: return original(*a, **kw)
            t = ACTIVE
            t.counter += 1; cid = t.counter
            component = label or name
            bound = signature.bind_partial(*a, **kw)
            inputs = {k:v for k,v in bound.arguments.items() if k not in ('self', 'conv_history')}
            t.event(phase='start', id=cid, parent=t.stack[-1] if t.stack else None,
                    component=component, inputs=t.pack(inputs))
            t.stack.append(cid); start = time.perf_counter()
            try:
                result = original(*a, **kw)
                packed = t.pack(result)
                t.event(phase='end', id=cid, component=component, result=packed, seconds=time.perf_counter()-start)
                return result
            except Exception as exc:
                if t.last_error is None: t.last_error = component
                t.event(phase='error', id=cid, component=component, error=repr(exc), traceback=traceback.format_exc())
                raise
            finally: t.stack.pop()
        setattr(owner, name, wrapped)

def snapshot(label, values):
    if ACTIVE is not None: ACTIVE.snapshot(label, values)

def unpack(v, root):
    if isinstance(v, dict):
        if artifact_ref(v, 'array'): return np.load(Path(root)/v['array'], allow_pickle=False)['value']
        if artifact_ref(v, 'image'): return Image.open(Path(root)/v['image']).copy()
        if 'nonfinite' in v: return float(v['nonfinite'])
        return {k:unpack(x,root) for k,x in v.items()}
    if isinstance(v,list): return [unpack(x,root) for x in v]
    return v
