"""Fetch a pinned checkout or safely expand a portable code bundle."""
import hashlib
import json
import os
import subprocess
import zipfile
from pathlib import Path

def extract_bundle(bundle, target):
    target = Path(target).resolve()
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(bundle) as archive:
        names = archive.namelist()
        for info in archive.infolist():
            name = info.filename.replace("\\", "/")
            destination = (target / name).resolve()
            if not destination.is_relative_to(target) or Path(name).is_absolute() or ":" in name:
                raise ValueError("Unsafe path in code bundle: " + name)
            if (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("Symbolic links are not supported in code bundles")
        archive.extractall(target)
    manifest_path = target / "bundle_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for name, digest in manifest["files_sha256"].items():
            path = target / name
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise ValueError("Bundle checksum mismatch: " + name)
    return target

def checkout(repo_url, ref, target):
    if not ref:
        raise ValueError("A tested code commit or release ref is required")
    target = Path(target)
    if not (target / ".git").is_dir():
        subprocess.run(["git", "clone", "--no-checkout", repo_url, str(target)], check=True)
    subprocess.run(["git", "-C", str(target), "checkout", "--detach", ref], check=True)
    return target
