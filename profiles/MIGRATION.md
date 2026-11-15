# Migration boundary

The five profiles contain ordinary Python files extracted once from the historical
Kaggle notebooks listed in each manifest. Notebook parsing and source extraction
are not part of runtime. Source archives were expanded to overlay/; embedded
benchmark seeds and historical predictions were removed. The main repository
uses pinned upstream submodules; overlay/repo supplies the notebook-era adapted
source required by the successful baseline. These overlays can gradually become
commits in the corresponding fork while preserving their hashes.

Runtime executes the listed stages with a shared namespace. This is a migration
compatibility layer, not a claim that every method has already been rewritten
as one universal inference API. Model-specific dependency installation, GPU
placement, parsing and resume checks are deliberately retained.

Changes during migration:
- Fresh APC benchmark includes all 2,078 LR/Visibility/Facing rows. Resume must be
  explicitly supplied; no frozen result blobs and no automatic diagnostic replay.
- PCD does not rerun its direct-Qwen baseline by default.
- Spatial-SSRL uses the intended single sharded 7B worker and three-task scope.
- Floating Qwen and SSRL revisions were replaced by actual successful run SHAs.
- SymPL uses the same pinned SAM download mirror as the APC asset fix.
- VST retains max_new_tokens=1280 from the completed benchmark manifest.

CPU validation includes compilation and profile planning. New thin notebooks have
not been run on Kaggle GPUs yet; start with the sanity notebooks.

The outer sc203_identity.json also pins the entire effective profile source hash.
Do not remove it when resuming a sc203-kat output; inherited worker fingerprints
retain their historical schema and are not sufficient to detect edits to this new
profile layer. External legacy checkpoint imports must be audited as migrations.
