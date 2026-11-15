"""Build a code-only bundle; contains no weights, results, Git directories or secrets."""
import argparse
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAPPING = {"apc_vis": "apc_vlm", "pcd": "pcd", "sympl": "sympl", "vst": "vst", "spatial_ssrl": "spatial_ssrl"}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=MAPPING)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    paths = []
    for folder in ["src", "configs", "profiles", "environments", "manifests", "third_party/" + MAPPING[args.method]]:
        base = ROOT / folder
        if base.exists():
            paths.extend(p for p in base.rglob("*") if p.is_file() and not any(part in {".git", "__pycache__", ".venv"} for part in p.relative_to(ROOT).parts) and p.suffix not in {".pyc", ".safetensors", ".pth", ".pt"})
    paths.extend(ROOT / name for name in ["pyproject.toml", "README.md", "THIRD_PARTY_NOTICES.md"] if (ROOT / name).is_file())
    paths = sorted(set(paths))
    manifest = {"method": args.method, "files_sha256": {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    try:
        manifest["framework_commit"] = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
        manifest["dirty_worktree"] = bool(subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain"], text=True).strip())
    except subprocess.CalledProcessError:
        manifest["framework_commit"] = None
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            archive.write(path, path.relative_to(ROOT).as_posix())
        archive.writestr("bundle_manifest.json", json.dumps(manifest, indent=2))
    print(f"{out}: {len(paths)} files, {out.stat().st_size / 2**20:.1f} MiB")

if __name__ == "__main__":
    main()
