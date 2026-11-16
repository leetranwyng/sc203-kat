"""Merge the two completed notebook outputs into one portable comparison page."""
import argparse
import copy
from pathlib import Path
import shutil
from .common import read,write,rows
from .report import build

def merge(sources,destination):
    sources=[Path(p).resolve() for p in sources];dest=Path(destination).resolve()
    if dest.exists() and any(dest.iterdir()):raise ValueError('Choose an empty merged-output folder')
    if any(dest==s or dest.is_relative_to(s) for s in sources):raise ValueError('Output must be outside source runs')
    identities=[read(s/'identity.json') for s in sources]
    reference=copy.deepcopy(identities[0]);reference.pop('variants')
    reference.pop('weight_inputs',None)
    variants=[]
    for s,identity in zip(sources,identities):
        check=copy.deepcopy(identity);check.pop('variants');check.pop('weight_inputs',None)
        if check!=reference:raise ValueError('Runs have different code/cohort/settings; cannot merge')
        if set(variants)&set(identity['variants']):raise ValueError('Overlapping variants; select one authoritative run per variant')
        variants+=identity['variants']
    dest.mkdir(parents=True,exist_ok=True)
    for idx,s in enumerate(sources):
        for name in ('answers','geometry','perception','logs'):
            if (s/name).exists():
                if name=='logs':shutil.copytree(s/name,dest/'logs'/str(idx),dirs_exist_ok=True)
                else:shutil.copytree(s/name,dest/name,dirs_exist_ok=True)
        if idx==0:
            shutil.copytree(s/'original',dest/'original')
            shutil.copy2(s/'samples.jsonl',dest/'samples.jsonl')
        write(dest/'source_runs'/f'{idx}.json',dict(path=str(s),identity=identities[idx],
            weights=read(s/'weights_lock.json') if (s/'weights_lock.json').exists() else {}))
    identity=copy.deepcopy(identities[0]);identity['variants']=variants;write(dest/'identity.json',identity)
    build(dest)
    return dest/'index.html'

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--inputs',nargs='+',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();print(merge(a.inputs,a.output))
