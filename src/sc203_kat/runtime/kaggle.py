"""Git-only source bootstrap for public Kaggle experiments.

A checkout is always detached at the freshly fetched remote ref. The selected
paper submodule is checked out at the parent repository's gitlink, never at its
moving default branch. No GPU packages or authentication tokens are needed here.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path
from urllib.parse import unquote, urlsplit
from urllib.request import url2pathname

_SUBMODULES = {
    "apc_vis": "apc_vlm", "pcd": "pcd", "sympl": "sympl",
    "vst": "vst", "spatial_ssrl": "spatial_ssrl",
}


def _environment(env=None):
    result = dict(os.environ)
    if env:
        result.update({str(key): str(value) for key, value in env.items()})
    result["GIT_TERMINAL_PROMPT"] = "0"
    result["GCM_INTERACTIVE"] = "Never"
    return result


def _git(root, *args, env=None):
    command = ["git"]
    if root is not None:
        command += ["-C", str(root)]
    command += list(args)
    result = subprocess.run(command, check=False, capture_output=True, text=True,
                            encoding="utf-8", errors="replace", env=_environment(env))
    if result.returncode:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(f"Git {args[0] if args else ''} failed: {detail}")
    return result.stdout.strip()


def _origin_key(value):
    value = str(value).strip().rstrip("/")
    parsed = urlsplit(value)
    if parsed.scheme == "file":
        return ("local", str(Path(url2pathname(unquote(parsed.path))).resolve()).casefold() if os.name == "nt" else str(Path(url2pathname(unquote(parsed.path))).resolve()))
    if parsed.scheme in ("https", "http", "ssh"):
        if parsed.username or parsed.password:
            raise ValueError("Use an origin URL without embedded credentials")
        path = parsed.path.rstrip("/")
        if path.endswith(".git"):
            path = path[:-4]
        host = (parsed.hostname or "").casefold()
        if host == "github.com":
            path = path.casefold()
        return (parsed.scheme.casefold(), host, parsed.port, path)
    # Local paths are useful for offline tests and do not grant file-protocol
    # permission to submodule subprocesses; tests opt into that explicitly.
    if "://" in value or "@" in value:
        raise ValueError("Unsupported origin URL")
    path = str(Path(value).resolve())
    return ("local", path.casefold() if os.name == "nt" else path)


def _repository_root(root, env=None):
    root = Path(root).resolve()
    if not (root / ".git").exists():
        raise ValueError(f"Not a Git checkout: {root}")
    actual = Path(_git(root, "rev-parse", "--show-toplevel", env=env)).resolve()
    if actual != root:
        raise ValueError("The target must be the root of its Git checkout")
    return root


def _clean(root, env=None):
    status = _git(root, "status", "--porcelain", "--untracked-files=normal", env=env)
    if status:
        raise RuntimeError(f"Refusing to overwrite a dirty checkout: {root}. Commit changes or choose another code directory.")


def _method(method):
    if method not in _SUBMODULES:
        raise ValueError("Unknown method: " + str(method))
    return _SUBMODULES[method]


def _gitlink(root, method, env=None):
    key = _method(method)
    path = "third_party/" + key
    record = _git(root, "ls-tree", "HEAD", "--", path, env=env)
    fields = record.split(None, 3)
    if len(fields) != 4 or fields[0] != "160000" or fields[1] != "commit" or fields[3] != path:
        raise ValueError("Missing pinned Git submodule: " + path)
    configured = _git(root, "config", "--file", ".gitmodules", "--get", f"submodule.{key}.path", env=env)
    if configured != path:
        raise ValueError("Submodule path does not match .gitmodules: " + path)
    url = _git(root, "config", "--file", ".gitmodules", "--get", f"submodule.{key}.url", env=env)
    return key, path, fields[2], url


def ensure_submodule(root, method, env=None):
    """Initialize only this method's paper source at its recorded parent pin.

    Existing changes inside the submodule are refused. A clean submodule at an
    older pin can be advanced by the normal Git submodule update operation.
    """
    root = _repository_root(root, env)
    key, relative, pinned, url = _gitlink(root, method, env)
    path = root / relative
    if (path / ".git").exists():
        _clean(path, env)
    _git(root, "submodule", "sync", "--", relative, env=env)
    _git(root, "submodule", "update", "--init", "--depth", "1", "--", relative, env=env)
    actual = _git(path, "rev-parse", "HEAD", env=env)
    if actual != pinned:
        raise RuntimeError("Submodule commit does not match the parent pin: " + relative)
    return {"method": method, "name": key, "path": relative, "repository": url, "commit": actual}


def checkout(repo_url, ref, target, method=None, env=None):
    """Clone/fetch a ref, detach at its remote commit, and optionally prepare a method.

    ``target`` may be absent or empty. Existing source is never deleted/reset.
    Reusing a checkout requires the same origin and a clean worktree. The ref is
    fetched even when the cache exists, so a branch such as ``main`` cannot use a
    stale local branch. Tags and full Git commit IDs are supported as well.
    """
    if not isinstance(ref, str) or not ref.strip() or ref.startswith("-") or any(char.isspace() for char in ref):
        raise ValueError("A non-empty Git tag, branch, or full commit is required")
    expected_origin = _origin_key(repo_url)
    target = Path(target).resolve()
    if (target / ".git").exists():
        _repository_root(target, env)
        origin = _git(target, "remote", "get-url", "origin", env=env)
        if _origin_key(origin) != expected_origin:
            raise ValueError("Cached checkout origin differs from REPO_URL; choose another code directory")
        _clean(target, env)
    else:
        if target.exists() and (not target.is_dir() or any(target.iterdir())):
            raise ValueError("Code target already contains files; choose an empty directory")
        target.parent.mkdir(parents=True, exist_ok=True)
        _git(None, "clone", "--no-checkout", "--no-tags", "--depth", "1", str(repo_url), str(target), env=env)
    _git(target, "fetch", "--depth", "1", "--no-tags", "origin", ref, env=env)
    commit = _git(target, "rev-parse", "--verify", "FETCH_HEAD^{commit}", env=env)
    _git(target, "checkout", "--detach", commit, env=env)
    if _git(target, "rev-parse", "HEAD", env=env) != commit:
        raise RuntimeError("Checkout commit differs from fetched ref")
    if method is not None:
        ensure_submodule(target, method, env)
    return target


def source_metadata(root, method=None, env=None):
    """Return traceable source metadata without modifying the source checkout."""
    root = _repository_root(root, env)
    value = {"repository": _git(root, "remote", "get-url", "origin", env=env),
             "commit": _git(root, "rev-parse", "HEAD", env=env)}
    if method is not None:
        key, relative, pinned, url = _gitlink(root, method, env)
        path = root / relative
        actual = _git(path, "rev-parse", "HEAD", env=env) if (path / ".git").exists() else None
        value["submodule"] = {"method": method, "name": key, "path": relative,
                              "repository": url, "pinned_commit": pinned, "commit": actual}
    return value
