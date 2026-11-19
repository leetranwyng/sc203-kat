"""Fetch the frozen SymPL public replay inputs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path, PurePosixPath
from threading import local
from urllib.parse import urlparse

API = 'https://api.kaggle.com/v1/kernels.KernelsApiService/ListKernelSessionOutput'
PREFIX = 'sc203-kat/sympl_trace_full_devicefix/'
FINGERPRINT = '6a696e376a04502bd17d11e4f634276e922bc1f13ede6515b286992480c4dd95'
THREAD = local()


def session():
    if not hasattr(THREAD, 'client'):
        import requests
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
        client = requests.Session()
        client.trust_env = False
        client.headers['User-Agent'] = 'kaggle-api/v1.7.0'
        client.mount('https://', HTTPAdapter(max_retries=Retry(
            total=4, backoff_factor=1, allowed_methods={'GET', 'POST'},
            status_forcelist=[429, 500, 502, 503, 504],
        )))
        THREAD.client = client
    return THREAD.client


def list_files():
    body = {'userName': 'ltrnnguyn', 'kernelSlug': 'sc203-sympl-trace'}
    files, seen = {}, set()
    while True:
        response = session().post(API, json=body, timeout=(15, 90))
        response.raise_for_status()
        data = response.json()
        for item in data.get('files', []):
            name = item['fileName']
            if name.startswith(PREFIX):
                files[name[len(PREFIX):]] = item['url']
        token = data.get('nextPageToken')
        if not token:
            return files
        if token in seen:
            raise ValueError('Repeated Kaggle output page token')
        seen.add(token)
        body['pageToken'] = token


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def fetch(root, sample_ids=None):
    root = Path(root).resolve()
    files = list_files()

    def download(name):
        path = PurePosixPath(name)
        target = (root / name).resolve()
        if path.is_absolute() or not target.is_relative_to(root):
            raise ValueError(f'Unsafe SymPL trace path: {name}')
        digest = path.stem if name.startswith('artifacts/') else None
        if target.is_file() and (not digest or hashlib.sha256(target.read_bytes()).hexdigest() == digest):
            return target
        parsed = urlparse(files[name])
        if parsed.scheme != 'https' or parsed.hostname != 'www.kaggleusercontent.com':
            raise ValueError(f'Unexpected Kaggle file host for {name}')
        try:
            response = session().get(files[name], timeout=(15, 90))
            response.raise_for_status()
        except Exception as error:
            raise RuntimeError(f'Failed to download {name}: {type(error).__name__}') from None
        if digest and hashlib.sha256(response.content).hexdigest() != digest:
            raise ValueError(f'Bad content hash: {name}')
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + '.part')
        tmp.write_bytes(response.content)
        tmp.replace(target)
        return target

    manifest = json.loads(download('manifest.json').read_text(encoding='utf-8'))
    if manifest['fingerprint'] != FINGERPRINT:
        raise ValueError('Unexpected public SymPL trace version. Supply an audited local sympl_trace directory.')
    for name in ('run_config.json', 'effective_runtime.json', 'summary.json'):
        download(name)
    predictions = rows(download('predictions.jsonl'))
    all_ids = {str(r['sample_id']) for r in predictions}
    if len(all_ids) != len(predictions) or len(all_ids) != 120:
        raise ValueError('Expected the historical 120 unique SymPL rows')
    wanted = set(map(str, sample_ids)) if sample_ids is not None else all_ids
    if not wanted <= all_ids:
        raise ValueError(f'Historical IDs absent: {wanted - all_ids}')

    def depth_refs(row):
        events = rows(download(row['attempt_path'] + '/calls.jsonl'))
        artifacts = set()
        for event in events:
            if event['component'] == 'run_depth_intrinsic_estimation' and event['phase'] == 'end':
                depth = event['result'][0]
                if isinstance(depth, dict) and 'array' in depth:
                    artifacts.add(depth['array'])
        return artifacts

    with ThreadPoolExecutor(max_workers=12) as pool:
        selected = [r for r in predictions if str(r['sample_id']) in wanted]
        artifacts = set().union(*pool.map(depth_refs, selected))
        for index, _ in enumerate(pool.map(download, sorted(artifacts)), 1):
            if index % 20 == 0:
                print(f'SymPL history: {index}/{len(artifacts)} depth arrays', flush=True)
    print(f'SymPL history: {len(selected)} call traces and {len(artifacts)} depth arrays', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output_dir')
    parser.add_argument('--sample-ids', nargs='+')
    args = parser.parse_args()
    fetch(args.output_dir, args.sample_ids)


if __name__ == '__main__':
    main()
