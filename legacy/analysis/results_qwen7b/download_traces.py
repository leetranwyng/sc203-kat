"""Download the exact Kaggle trace zips, resuming verified byte ranges."""
import concurrent.futures
import os
import re
import struct
import sys
import time
import zlib
from pathlib import Path
from urllib.request import Request, urlopen
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parent / 'raw'
URL = 'https://www.kaggle.com/api/v1/kernels/output/download/'
NAMES = {
    'apc': ('khaitrannhu/apc-vis-7b-qwen/apc_qwen7b/apc_qwen7b_kaggle_dispatch_v3.zip', 4851941376),
    'sympl': ('ltrnnguyn/sc203-sympl/sympl_7b_full_memory_v2_trace.zip', 4030163244),
}
STEP = 64 * 1024 * 1024


def salvage_apc(partial):
    outer = ROOT / 'apc_output.zip'
    if partial.exists() or not outer.is_file() or outer.stat().st_size < 100:
        return
    with outer.open('rb') as source, partial.open('wb') as target:
        header = struct.unpack('<IHHHHHIIIHH', source.read(30))
        name = source.read(header[-2]).decode()
        source.read(header[-1])
        if not name.endswith('apc_qwen7b_kaggle_dispatch_v3.zip') or header[3] != 8:
            raise ValueError('Unexpected first ZIP entry')
        stream = zlib.decompressobj(-15)
        while (data := source.read(1024 * 1024)) and not stream.eof:
            target.write(stream.decompress(data))
    print('Recovered APC bytes', partial.stat().st_size, flush=True)


def download(model, destination):
    location, size = NAMES[model]
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix('.zip.part')
    if model == 'apc':
        salvage_apc(partial)
    current = partial.stat().st_size if partial.exists() else 0
    # Truncate only on success, while each worker writes its own disjoint range.
    state = partial.with_suffix('.ranges.txt')
    previous = state.read_text().splitlines() if state.exists() else []
    if previous:
        current = int(previous[0])
    completed = set(int(x) for x in previous[1:])
    if not previous:
        state.write_text(str(current)+'\n')
    if not partial.exists():
        partial.touch()
    with partial.open('r+b') as stream:
        stream.truncate(size)

    def fetch(offset):
        last = min(offset+STEP,size)-1
        for attempt in range(4):
            try:
                req = Request(URL+location,headers={'Range':f'bytes={offset}-{last}'})
                with urlopen(req,timeout=45) as response, partial.open('r+b',buffering=0) as stream:
                    if response.status != 206 or response.headers.get('Content-Range') != f'bytes {offset}-{last}/{size}':
                        raise ValueError('Unexpected byte range')
                    stream.seek(offset)
                    remaining = last-offset+1
                    while remaining:
                        data = response.read(min(1024*1024,remaining))
                        if not data:
                            raise IOError('Truncated byte range')
                        stream.write(data)
                        remaining -= len(data)
                return offset
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(attempt+1)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool, state.open('a',buffering=1) as log:
        futures = [pool.submit(fetch,offset) for offset in range(current,size,STEP) if offset not in completed]
        for i, f in enumerate(concurrent.futures.as_completed(futures),1):
            log.write(str(f.result())+'\n')
            if i%8==0 or i==len(futures):
                print(model,i,'/',len(futures),'ranges',flush=True)
    with ZipFile(partial) as archive:
        print(model,len(archive.infolist()),'members',flush=True)
    partial.replace(destination)
    state.unlink()


if __name__ == '__main__':
    download(sys.argv[1],Path(sys.argv[2]))
