"""Portable trace paths for Kaggle, without changing object descriptions."""
import hashlib
import html
import json
import posixpath
from pathlib import Path, PurePosixPath
import re
from urllib.parse import unquote, urlsplit

POLICY = 'ascii_letters_digits_dot_underscore_hyphen_v1'
SAFE = re.compile(r'[A-Za-z0-9_.-]+\Z')

def safe_component(value):
    """Idempotent filename; a digest prevents renamed categories from colliding."""
    value=str(value)
    reserved=value.split('.')[0].upper() in {'CON','PRN','AUX','NUL',
        *('COM'+str(i) for i in range(1,10)),*('LPT'+str(i) for i in range(1,10))}
    if SAFE.fullmatch(value) and value not in ('.','..') and len(value)<=180 and not reserved:return value
    suffix=PurePosixPath(value).suffix
    if not re.fullmatch(r'\.[A-Za-z0-9]{1,12}',suffix):suffix=''
    stem=value[:-len(suffix)] if suffix else value
    stem=re.sub(r'[^A-Za-z0-9_.-]+','_',stem).strip('._-') or 'file'
    return stem[:140]+'_'+hashlib.sha256(value.encode('utf8')).hexdigest()[:12]+suffix

def path_map(names):
    mapping={};used=set()
    for name in sorted(names):
        path=PurePosixPath(name)
        if path.is_absolute() or '..' in path.parts or '\\' in name:raise ValueError('Unsafe source path: '+name)
        target='/'.join(safe_component(part) for part in path.parts)
        if target.casefold() in used:raise ValueError('Portable trace path collision: '+name)
        used.add(target.casefold());mapping[name]=target
    return mapping

def rewrite_url(value,origin,mapping):
    if not value or value.startswith(('data:','http:','https:','mailto:','javascript:','#')):return value
    decoded=unquote(value);old_parent=posixpath.dirname(origin)
    target=posixpath.normpath(posixpath.join(old_parent,decoded));tail=''
    if target not in mapping:
        parts=urlsplit(decoded);target=posixpath.normpath(posixpath.join(old_parent,parts.path))
        tail=('?'+parts.query if parts.query else '')+('#'+parts.fragment if parts.fragment else '')
    if target not in mapping:return value
    return posixpath.relpath(mapping[target],posixpath.dirname(mapping[origin]) or '.')+tail

def rewrite_payload(data,origin,mapping):
    ext=PurePosixPath(origin).suffix.lower()
    if ext not in ('.html','.json','.jsonl'):return data
    text=data.decode('utf8')
    if ext=='.html':
        pattern=r'(?P<head>\b(?:src|href|poster)\s*=\s*)(?P<q>[\x22\x27])(?P<url>.*?)(?P=q)'
        def replace(match):
            old=html.unescape(match['url']);new=rewrite_url(old,origin,mapping)
            if new==old:return match.group(0)
            return match['head']+match['q']+html.escape(new,quote=True)+match['q']
        return re.sub(pattern,replace,text,flags=re.IGNORECASE).encode('utf8')
    def walk(value):
        if isinstance(value,str):return rewrite_url(value,origin,mapping)
        if isinstance(value,list):return [walk(v) for v in value]
        if isinstance(value,dict):return {k:walk(v) for k,v in value.items()}
        return value
    if ext=='.json':
        original=json.loads(text);new=walk(original)
        return data if new==original else (json.dumps(new,ensure_ascii=False,indent=2)+'\n').encode('utf8')
    original=[json.loads(line) for line in text.splitlines() if line.strip()];new=walk(original)
    return data if new==original else ''.join(json.dumps(v,ensure_ascii=False)+'\n' for v in new).encode('utf8')

def patch_apc_filenames(repo):
    """Change path interpolation only; model categories and prompts stay verbatim."""
    changes=[]
    for relative,prefix in [('apc/vision_modules/detection.py','detection_candidates'),
                             ('apc/vision_modules/orientation.py','orientation')]:
        path=Path(repo)/relative;source=path.read_text(encoding='utf8')
        old=f'f"{prefix}_{{category}}.png"';new=f'f"{prefix}_{{safe_component(category)}}.png"'
        if new not in source:
            if source.count(old)!=1:raise ValueError('Expected one artifact filename in '+relative)
            source=source.replace(old,new)
            source='from sc203_kat.ablation.portable import safe_component\n'+source
            compile(source,str(path),'exec');path.write_text(source,encoding='utf8')
        changes.append(dict(file=relative,sha256=hashlib.sha256(source.encode()).hexdigest()))
    return changes
