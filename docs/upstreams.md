# Upstream repositories and forks

This local repository contains five independent, shallow source checkouts registered as Git submodules. They were copied through Git from the existing local clones, so they do not depend on the old repositories' object stores. No weights, caches or untracked result archives were copied.

`manifests/upstreams.json` records the checked-out commit, official URL and any local tracked edits. The current submodule URLs point to official upstream repositories. A local Git checkout is not a GitHub fork: no remote GitHub repositories were created by this migration.

## Creating the team forks later

1. Create one GitHub fork per upstream under the team's account or organization.
2. Keep that fork's `upstream` remote pointing at the original project.
3. Change the corresponding `.gitmodules` URL to the team fork URL, run `git submodule sync`, and update `remote_fork_url` in the manifest.
4. Commit paper-source modifications in the appropriate fork, then commit its new submodule pointer in the parent repository.
5. Keep scientific runs pinned to exact commits. Do not use an automatically moving branch for Kaggle runs.

## Runtime snapshots versus Git commits

The tested APC configuration records `source_snapshot_sha=9059ed359baf89b49675ed3e0407f843b3ff28de78e5cae1c870adf79f44a339`. This is a 64-character SHA256 content hash of an embedded source snapshot, not a Git commit. The clean source submodule at `d7ca0517b96afbce22ddb7332a8b6361071e0bcf` must not be described as identical to that patched runtime snapshot. The migrated method profiles retain the runtime source and modifications used by the working notebooks.

The SymPL local worktree differs from its clean HEAD in `sympl/prompt_extraction.py`, `sympl/prompts.py`, and `sympl/utils.py`. The migration preserves those changes as a patch in `manifests/`; it does not silently alter the clean upstream pin.

## Downloading or packaging

A normal clone requires `git submodule update --init` for the required method. GitHub's parent source ZIP does not automatically supply submodule source contents. The Kaggle packager must include the required method source snapshot, or fetch the pinned submodule explicitly. Large native dependencies are resolved by their own locked asset/source manifests.
