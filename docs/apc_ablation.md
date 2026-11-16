# APC-Vis perception ablations on the diagnostic trace cohort

Use `notebooks/apc_ablation/sanity.ipynb` first, then run the two full notebooks:
`01_sam3_oneref.ipynb` (SAM 3 and OneRef), and `02_orientanything_v2.ipynb` (OA2), on Kaggle
T4 ×2 with Internet. All three clone public sc203-kat at `v0.3.0-ablation.4`.
The implementation has CPU/static tests; GPU feasibility and numerical behavior
of the new adapters must be checked with the sanity notebook. No claim of a
successful Kaggle GPU run is made by the source release.

## Inputs

1. Upload the prepared `sc203_apc_trace120.zip` as a Kaggle Dataset (data and
   historical evidence, not source code). Kaggle should extract it. The folder
   must contain `bundle.json`, `samples.jsonl`, and `original/<sample_id>/`.
   The bundle contains 120 selected traces, 40 each for Left/Right, Visibility,
   and Facing. It deliberately includes more baseline errors than correct cases.
2. Enable the Kaggle Secret `HF_token`, with approved `facebook/sam3` access.
3. Optional `model_input`: the existing complete APC assets bundle. Without it,
   the ablation asset preparation downloads only the required pinned APC weights.
   Grounding DINO and Depth Pro weights are not downloaded or loaded.
4. Optional `weight_inputs`: paths to SAM 3 `.pt`, OA2 `.pt`, OneRef **REC base
   mixup** `.pth`, and its `beit3.spm`. Other OneRef architectures are not
   interchangeable. Default downloads are pinned in `configs/ablations/upstreams.json`.

Sanity defaults to sample 5832 (one image/question). This checks integration,
not accuracy. Set `sample_ids` to several IDs to inspect broader behavior before
running all 120. `sample_ids=None` runs the entire supplied cohort.

## Experimental design

| Variant | Proposals | Instance selection | Mask | Orientation |
|---|---|---|---|---|
| baseline | Historical Grounding DINO | Historical APC choice | Historical SAM | Historical OA V1 |
| sam3 | SAM 3, top 5 by confidence | Unchanged APC crop-grid Qwen prompt | Matching SAM 3 mask | OA V1 on new crop |
| oneref | Historical Grounding DINO | OneRef REC box → largest IoU candidate | Original SAM, mask index 2 | OA V1 on new crop |
| oa2 | Historical Grounding DINO | Historical APC choice | Historical SAM | OA V2 on same selected crop |
| sam3_oa2 | SAM 3 | APC crop-grid choice | SAM 3 | OA V2 |
| oneref_oa2 | Historical Grounding DINO | OneRef IoU selection | SAM | OA V2 |
| sam3_oneref | SAM 3 | OneRef IoU selection | SAM 3 | OA V1 |
| sam3_oneref_oa2 | SAM 3 | OneRef IoU selection | SAM 3 | OA V2 |

The sanity runs all three modified branches on one image. The two full notebooks
split these three branches over all 120 traces. **Baseline is historical data only;
no baseline inference or replay is executed.** Enable combinations only after inspecting the
single-component runs. This is **controlled replay**, not a full end-to-end rerun
of APC's object extraction and perspective selection: all variants freeze those
two upstream outputs and the full-image Depth Pro map. Original baseline positions
and orientations are reused only as controlled inputs to relevant interventions.
Only the modified branches execute their renderer and downstream Qwen calls.
When a box/mask changes, APC's unprojection recomputes the position from the same
depth map and OA V1 recomputes orientation. Renderer, Qwen checkpoint, FP16 device
partition, prompts and generation settings are inherited from the existing APC profile.

Compare interventions against the **saved historical APC prediction**. Because the
baseline is not replayed, any environment-induced drift in downstream VLM behavior
cannot be independently separated from the intervention effect in this run.
Corrected/broken answers are evidence of a component intervention's downstream
effect; they are not automatic ground-truth labels of detection/orientation errors.

OneRef receives the full image and the original extracted object description,
not the question's requested relation or answer. Its predicted box chooses among
fixed proposals by IoU. Zero overlap produces an explicit error, not a silent
first-box fallback. This cannot recover a correct instance absent from the candidate
set, and a category-only description may remain ambiguous. SAM 3 alone changes
**both** proposals and masks, so its gain is not a detector-only attribution.
SAM 3 uses confidence 0.5 and a maximum of five candidates for APC refinement;
all model-returned boxes/scores/masks are retained for auditing. No claim that
this threshold is optimal is made.

OA2 uses its official padded-image preprocessing and checkpoint, no extra background
removal. Store all 900 logits, argmax azimuth/elevation/roll, fitted symmetry alpha,
and the resulting direction. The adapter uses the official rotation composition
`Rx(roll) Ry(elevation) Rz(-azimuth)` on semantic +X, then maps `(x,y,z)` to APC
`(y,-z,-x)`. At zero roll, tests verify equality with APC's original azimuth/elevation
formula. OA2 retains roll, whereas the old APC adapter ignores it. Projected arrows
are **not** proof of correct 3D orientation; inspect sanity overlays and recorded
coordinate conventions. Multi-modal/symmetric directions are saved but this initial
ablation uses one argmax direction, not an oracle choice using the answer.

## Execution and failure handling

Phases: SAM 3 / OneRef caches → APC geometry → OA2 crops → APC renderer and Qwen.
Each model family runs in a separate process; new perception models are never
resident alongside Qwen. APC keeps the proven two-GPU FP16 placement and vision
offload scheduler. Auxiliary models use one T4, then release the entire process.
SAM 3 and OA2 use FP16 autocast; OneRef uses FP32. No automatic quantization or
BF16 fallback on T4. This does not guarantee every upstream kernel will work on
T4; the sanity is the GPU acceptance check.

Python/torch environments are separate: APC 3.10/cu124, SAM 3 3.12/cu126,
OA2 3.11/cu124, OneRef 3.10/cu118. `uv` avoids `ensurepip`; interrupted environments
are preserved and recreated instead of trusting an orphaned `pyvenv.cfg`.
The OneRef wrapper avoids importing upstream `eval.py`, which hardcodes GPU 7.
It reproduces the 384-pixel letterbox preprocessing and reverses its padding.

Weights/environments are temporary; only evidence is saved to working. Expect
substantial temporary disk use for four torch environments and all APC models.
The default OneRef release ZIP contains three 2.69 GB checkpoints; HTTP Range
extracts only `mixup` without storing the full 6.49 GB archive. Supplying that `.pth` via Kaggle Input saves download/extraction space.
There is no automatic deletion of evidence to hide disk errors. Low disk stops
the run with durable per-sample results; ZIP export of results is optional.

Completed units are skipped. `retry_errors=True` creates new attempt folders for
failed APC units; raw earlier attempts remain. For a new session, mount previous
output and set `resume_input` to its root. Identity checks enforce the same source,
weights, settings and cohort. Changing variants/settings requires a new output
directory. Source is always GitHub-cloned; ZIP files are used only for data/weights/results.

## Outputs

- `index.html`: all images/questions, variants, boxes and projected arrows, masks,
  rendered views, Qwen calls, coordinates, component artifacts; filters for task,
  correction, regression and disagreement.
- `accuracy.csv`, `paired_outcomes.csv`, `summary.json`, `predictions.jsonl`.
- `original/`: unchanged source evidence, including original candidate boxes,
  masks, depth, orientation outputs and VLM calls.
- `perception/{sam3,oneref,oa2}/`: exact input/crop, model outputs, masks/logits,
  elapsed time, peak VRAM, exception traceback if a component fails.
- `geometry/`, `answers/`: per-sample results and numbered attempt directories;
  prompts/images before each Qwen call, selected candidates, scenes and renders.
- `source_checkout.json`, `weights_lock.json`, `identity.json`, `protocol.json`,
  `state.json`, `logs/`: reproducibility and failure provenance.

All attempted invalid/error cases count as Wrong. Unattempted cases are displayed
as incomplete, not quietly excluded from a full-cohort accuracy. No CircularEval
score is reported unless a later experiment supplies all permutations of each
group. This selected cohort is for hypotheses and paired debugging, not claiming
an improved full 3DSRBench score. Recall, IoU and angular accuracy need independent
instance/mask/orientation ground truth; this bundle does not supply it.

## Primary implementation references

- [SAM 3 official code](https://github.com/facebookresearch/sam3)
- [Orient Anything V2 official code](https://github.com/SpatialVision/Orient-Anything-V2)
- [OneRef official code](https://github.com/linhuixiao/OneRef)

Exact source and checkpoint commits are recorded in `configs/ablations/upstreams.json`.


## Combine the two outputs

After downloading/extracting both result folders, run from the repository:

```powershell
python -m sc203_kat.ablation.merge --inputs PATH_TO_NOTEBOOK_01_OUTPUT PATH_TO_NOTEBOOK_02_OUTPUT --output PATH_TO_NEW_COMBINED_FOLDER
```

Use an empty output folder. The command checks matching cohorts, source and settings,
copies evidence to make a portable report, and generates `index.html`, accuracy and
paired-outcome tables. It runs no model inference. Include the entire report folder
when sharing the HTML so its images remain available.


## Kaggle filename compatibility

The trace ZIP uses only ASCII letters, digits, dots, underscores and hyphens in
its paths. Object categories remain verbatim in questions, prompts, JSON values
and `.name` files. `filename_map.json` records renamed artifacts; image references
in HTML/JSON and the SHA256 entries in `bundle.json` are updated together. The
APC runtime also sanitizes newly written candidate/orientation image filenames.
No inference needs to be repeated to repair the input archive.

`package_apc_trace_input.py` now streams directly to ZIP instead of creating a
second expanded copy. To fix an existing data archive:

```powershell
python scripts/package_apc_trace_input.py --repair-zip OLD.zip --output FIXED.zip
```

Upload the corrected ZIP as the trace Dataset and import the three notebooks
from release `v0.3.0-ablation.4`. Existing schema-1 inputs remain readable.


## OneRef `pkg_resources` import failure

The v0.3.0-ablation.2 Kaggle run installed setuptools 84.2.1, while
`torchmetrics==0.11.4` still imports `pkg_resources`. Setuptools 82 removed that
module. OneRef now explicitly pins `setuptools==75.8.0`, and its actual model
entry point is imported after installation, before model inference begins.

If the original Kaggle session is still active, the standalone recovery helper
`scripts/recover_apc_oneref.py` installs only this packaging dependency, retries
OneRef perception and failed OneRef geometry/answers, then rebuilds the full
report with the saved SAM 3/OA2 results. It uses the old source/identity and saves
the environment repair under `runtime_audit/`; native setup and model downloads
are skipped. The original run configuration is preserved. A closed Kaggle session
needs the new sanity notebook because its temporary environments are gone.

[Setuptools removal notes](https://setuptools.pypa.io/en/latest/history.html#v82-0-0).


## OneRef spaCy/Click compatibility

Version v0.3.0-ablation.3 fixed `pkg_resources`, then the actual OneRef import
exposed spaCy 3.7.5 importing a missing `click` module. Pin `click==8.1.8`,
`typer==0.9.0`, `weasel==0.3.4` alongside `setuptools==75.8.0`. Remove
`typer-slim` before reinstalling Typer because both distributions populate the
`typer` namespace. These versions satisfy the published spaCy and Weasel
requirements and retain the Click-based APIs used by that spaCy release.

Auxiliary environments and the real OneRef import now run before APC native
compilation. The new `OneRef CPU runtime` GitHub workflow additionally exercises
the production constructor and a CPU forward with random weights. It does not
measure pretrained performance or prove T4 execution.

The recovery helper supports a setup failure before `repos` and `weights` are
saved: it reuses `apc_context.json` to skip the completed native setup, resumes
the old orchestrator and installs only the remaining auxiliary dependencies.
No baseline inference is run.
