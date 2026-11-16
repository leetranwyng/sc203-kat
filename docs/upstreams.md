# Upstream repositories and forks

This local repository contains five independent, shallow source checkouts registered as Git submodules. They were copied through Git from the existing local clones, so they do not depend on the old repositories' object stores. No weights, caches or untracked result archives were copied.

`manifests/upstreams.json` records the checked-out commit, official URL and any local tracked edits. The submodule URLs point to verified GitHub forks under `leetranwyng`. Each fork retains its official paper repository as its GitHub parent, and each local submodule has an `upstream` remote pointing to that parent. The main repository is https://github.com/leetranwyng/sc203-kat (public).

## Working with the forks

1. Work on a new branch in the relevant submodule when changing paper-source code.
2. Commit and push that branch to its `origin` (`leetranwyng` fork).
3. Commit the updated submodule pointer in the main repo. The referenced commit must be available remotely before sharing it.
4. Fetch author updates from `upstream` explicitly; do not move baseline pins automatically.
5. Runtime profiles contain migration overlays, so changing a submodule alone may not change the effective inference source. Update the profile and record its hash as well.

## Runtime snapshots versus Git commits

The tested APC configuration records `source_snapshot_sha=9059ed359baf89b49675ed3e0407f843b3ff28de78e5cae1c870adf79f44a339`. This is a 64-character SHA256 content hash of an embedded source snapshot, not a Git commit. The clean source submodule at `d7ca0517b96afbce22ddb7332a8b6361071e0bcf` must not be described as identical to that patched runtime snapshot. The migrated method profiles retain the runtime source and modifications used by the working notebooks.

The SymPL local worktree differs from its clean HEAD in `sympl/prompt_extraction.py`, `sympl/prompts.py`, and `sympl/utils.py`. The migration preserves those changes as a patch in `manifests/`; it does not silently alter the clean upstream pin.

## Kaggle source checkout

Notebooks clone the public parent repo at a selected ref and initialize only the needed submodule at its gitlink commit. Both commits are saved with the run. Submodule origins are the `leetranwyng` forks and author repositories remain available as upstream remotes in the development checkout.

For a local working copy, use `git clone --recurse-submodules https://github.com/leetranwyng/sc203-kat.git`. Kaggle initializes submodules selectively to avoid downloading unrelated paper repositories. Native dependencies and model weights retain their own pinned manifests.
