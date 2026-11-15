from pathlib import Path
import hashlib, json, os, subprocess

# Chay cell nay trong cung session, sau cell tai weight bi loi.
root = Path(CFG['asset_input'] or CFG['asset_root'])
checkpoints = root / 'checkpoints'
checkpoints.mkdir(parents=True, exist_ok=True)
url = 'https://ml-site.cdn-apple.com/models/depth-pro/depth_pro.pt'
target = checkpoints / 'depth_pro.pt'
part = target.with_suffix('.part')
if not target.exists():
    subprocess.run([
        'curl', '--fail', '--location', '--continue-at', '-',
        '--retry', '8', '--retry-all-errors', '--retry-delay', '5',
        '--connect-timeout', '60', '--speed-limit', '1024', '--speed-time', '180',
        '--output', str(part), url,
    ], check=True)
    if part.stat().st_size < 1_000_000:
        raise RuntimeError('File Depth Pro qua nho; dung lai de tranh nap weight hong.')
    part.replace(target)

# Cac file hoan thanh truoc loi da duoc downloader cu kiem tra Content-Length.
manifest_path = root / 'assets_manifest.json'
manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
files = manifest.setdefault('files', {})
for name in ('sam_vit_h_4b8939.pth', 'groundingdino_swint_ogc.pth', 'depth_pro.pt'):
    path = checkpoints / name
    if path.is_file():
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(16 * 1024 * 1024), b''):
                digest.update(block)
        files[name] = {'bytes': path.stat().st_size, 'sha256': digest.hexdigest()}
manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')

ENV.pop('HF_HUB_OFFLINE', None)
ENV.pop('TRANSFORMERS_OFFLINE', None)
subprocess.run([PY, str(SCRIPTS / 'apc_prepare_assets.py'), str(CONFIG),
                str(WORK / 'assets_ready.json')], env=ENV, check=True)
assets = json.loads((WORK / 'assets_ready.json').read_text())
print('PASS: assets_ready.json da duoc tao. Chay lai cell cai native/assets de tiep tuc phan setup con lai.')
