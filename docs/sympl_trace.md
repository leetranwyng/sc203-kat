# SymPL trace120

Use `notebooks/sympl/SymPL_7B_APC120_sanity_kaggle.ipynb`, then
`notebooks/sympl/SymPL_7B_APC120_stage_traces_kaggle.ipynb` with the same configuration.
Mount the existing APC trace120 dataset. `trace_input=None` discovers either the
extracted bundle or its ZIP. Six sanity cases cover two cases per task; full uses
all 120 pinned cases. Only the input images and historical APC results are shared;
SymPL recomputes its own perception and geometry.

## Source ownership and isolation

The old standalone notebooks embedded a second runtime. They reused the dataset
layout but did **not** call sc203-kat. Their SymPL source matches the existing
`profiles/sympl/overlay/repo` byte for byte after newline normalization.

The new notebooks are Git launchers. `python -m sc203_kat.sympl_trace --config ...`
reuses these existing components without changing them:

- `runtime.kaggle`: pinned Git bootstrap and SymPL submodule metadata.
- `profiles/sympl/01_prepare.py`: isolated upstream checkout and profile overlay.
- `profiles/sympl/02_environment.py`: GPU checks and Python/Torch environment.
- `profiles/sympl/03_native_assets.py`: CUDA/native setup, checkpoint download and patches.
- Existing profile scripts for assets, native installs, model placement and option parsing.

The opt-in `src/sc203_kat/sympl_trace` module owns only paired input validation,
trace collection, sequential execution, diagnostics, the gallery and health gate.
Observation hooks are applied to the **runtime copy**, not the source profile or
third-party submodule. Existing `sc203-kat run` dispatch, profiles, model notebooks
and dependency pins remain unchanged. Trace experiments use separate output and
runtime paths. No imports are added to other model flows.

## Confirmed failure from trace_error.ipynb

The dataset preflight passed (120 rows, 5642 hashes), CUDA computed successfully
on two T4s, and model weights loaded. Samples 5486, 5832, 5050, 4854 and 8270 then
returned runtime errors. The old runner did not print their saved traceback;
the notebook alone does not identify those inference errors.

The subsequent gallery crashed in `html.escape(v['image'])`: named image inputs
serialize as `{'image': {'image': 'artifacts/<sha>.png', ...}}`, while the gallery
incorrectly treated any dict with an image key as an artifact. Artifact detection
now checks the path type and content-addressed format. Deserialization uses the
same discriminator. Regression tests cover nested images, arrays and nulls.

Workers now print each original traceback. Analysis and sanity exit codes are
recorded separately from the worker's exit code; partial artifacts are archived
even on failure. Sanity CLI exits nonzero on FAIL, and full does not start after
a failed smoke gate. Accuracy is not a technical health gate.

## Recover an old run without inference

Run `notebooks/sympl/SymPL_7B_APC120_recover_kaggle.ipynb` in a session with the old output and
runtime available. Set `OLD_OUTPUT` to the old directory. It prints saved
per-sample errors and rebuilds the gallery, without model loading or GPU setup.
If the runtime has disappeared, original per-sample errors and the archived
evidence are still available, but renderer replay needs the original source.

## Outputs

`calls.jsonl` records named inputs, returns, exceptions and call parents.
PNG/NPZ artifacts retain images, tensors, depth filtering, detector candidates,
orientation, transforms, projection, partition and VLM inputs/outputs.
`gallery.html` compares APC/SymPL evidence with task/outcome filters and review JSON.
`failure_cases.csv`, `trace_validation.json`, `sanity_report.json` and
`execution_status.json` separate model answers, trace integrity and infrastructure.
Stage diagnoses are conservative: final-answer labels cannot establish which
upstream geometry estimate was wrong without further annotation.

## Validation limits

CPU tests reproduce the observed gallery failure and verify the fix, lossless
artifact round trips, exact renderer replay, unchanged depth numerics, failure
triage, input corruption checks and sanity gates. GPU inference must be checked
on Kaggle. The five earlier inference exceptions still require their saved
`result.json`/`calls.jsonl`; no claim is made that the gallery fix resolves them.
