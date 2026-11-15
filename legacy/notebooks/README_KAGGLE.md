# APC + Qwen2.5-VL-3B on Kaggle

## Outputs

- `apc_qwen25vl_3b_sanity_kaggle.ipynb`: one included repo demo image, no benchmark TSV download. Runs both direct Qwen and complete APC-Vis, including vision models and renderer.
- `apc_qwen25vl_3b_3dsrbench_full_kaggle.ipynb`: only the three paper task families: left/right, visibility recast from front/behind, and facing (2,078 rows / 1,039 circular groups); paired `direct` and `apc_vis` modes by default. One pipeline uses GPU0 for Qwen and GPU1 for vision.

Both notebooks are self-contained: local APC source, configuration, the demo image and helper code are embedded. They clone third-party vision dependencies by HTTPS, resolve and record commit IDs. They do not need this local workspace or the SenseNova checkout at runtime.

## What was verified

Local `paper.pdf`: Table 1 is the principal results table; Appendix C specifies Qwen2.5-VL-7B-Instruct. The paper restricts 3DSRBench to left/right, recast visibility, and facing, with exact matching followed by LLM judging. APC-Num raises NotImplementedError in the released local code. The YAML uses detection thresholds 0.05, while Appendix C describes 0.15. Notebook defaults use box threshold 0.15 as described in the paper and retain text threshold 0.05 from YAML; both are recorded.

These notebooks compare direct 3B with APC-Vis on the paper task families. Front/behind is deterministically rewritten to a visibility question with yes/no choices while preserving option positions and correct letters. Original question/choices are retained for audit. There is no LLM judge, and exact author wording/evaluator equivalence is not claimed. Full data results must not be called reproduction of Table 1, or compared directly to that table without explaining the different backbone/tasks/protocol.

## Run order

1. Kaggle: Internet enabled, T4 x2, add and authorize secret `HF_token`.
2. Run sanity from the top. A fresh isolated Python 3.10 environment uses Torch 2.4.1/cu124, torchvision 0.19.1 and Transformers 4.57.1.
3. Native setup checks for a CUDA 12 developer toolkit (`nvcc`), then builds vision extensions. Driver-only environments or CUDA-major mismatches are rejected with a clear message. This has not been GPU-tested locally.
4. Native import + Trimesh rendering preflight must pass, then the full one-image APC smoke must pass.
5. Run the full notebook with unchanged model/dtype/pixels. It first performs another pipeline smoke, then all rows. `limit=None` selects all 2,078 rows in the three task families, not the other nine subtypes. `modes=['apc_vis']` is available if a new experiment only needs APC; the default includes a fair same-backbone direct baseline.
6. Download the ZIP and inspect `summary.json`, per-mode merged predictions, invalid outputs and pipeline errors.

## Disk / assets

The VLM plus SAM ViT-H, GroundingDINO, DepthPro, Orient-Anything, DINOv2 and BERT are large. There is no claim they fit every Kaggle disk allocation along with native build artifacts. `asset_root=/kaggle/temp/apc_assets` is a scratch location, not saved Output, and may share a filesystem with Working. Free-space checks stop partial setup rather than silently changing models. If space is insufficient, use a prepared asset bundle as Kaggle Input (`asset_input`):

```
assets_manifest.json
qwen_model/               # complete Qwen config/processor/tokenizer/chat-template/safetensors
bert_base_uncased/        # config/tokenizer/safetensors
dinov2_large/             # config/processor/safetensors
checkpoints/
  sam_vit_h_4b8939.pth
  groundingdino_swint_ogc.pth
  depth_pro.pt
  orient_dino_weight.pt
```

The helper `apc_prepare_assets.py` can also prepare the bundle on a Linux machine with enough storage and huggingface_hub installed. It reads a JSON config containing model_id, model_revision, asset_root and asset_input=null; the second argument is the destination JSON manifest. Upload the complete materialized files, not broken symlinks to another machine's cache. The setup still needs native source/compiler dependencies even with an asset Input.

## Output handling / resume

Invalid final text (empty, repeated !, or unparseable) is retained and counted incorrect. Recoverable sample-level detection/parsing/geometry errors are written and the next sample is attempted; there is no hidden direct-Qwen fallback pretending to be APC. CUDA device failures and OOM stop the process. Smoke refuses invalid output. Pipeline errors remain in the score denominator, and coverage/attempted rows/successful rows are separate.

Each mode has a JSONL in `full/`; writes are flushed per sample and fsynced every eight attempts and on termination. Resume checks fingerprint and sample IDs. Full pipeline state tracks stage, sample, mode, errors and invalids; the parent has a watchdog. Keep config, source locks, assets/revisions and paths identical for a resumed experiment. Default `retry_failed=False` keeps already-attempted error records; do not silently change configuration in an existing experiment. Save the whole output and metadata if the Kaggle session is going away.

## Compatibility patches

All edits affect the extracted runtime snapshot only; the original local `apc/` is unchanged. `patches.json` records before/after hashes.

- GroundingDINO model/predict use the selected vision GPU.
- Unused Orient debug renderer is lazily imported; DINOv2 embedding accepts compatible extra kwargs.
- Renderer initializes camera shift, avoids zero-XY division and keeps positions/orientations/colors aligned after filtering.
- Depth unprojection loop is vectorized and checked against the original formula.
- Depth output and SAM image embedding are cached only within one sample, avoiding repeated computation for multiple objects.
- Qwen uses explicit dtype/SDPA/cache/pad token and trims prompt tokens, preserving integer grid/ID tensors.
- Final option choosing uses the same 32-token cap as direct inference; intermediate APC calls retain a separate 512-token cap.

## Offline validation

`python notebooks/validate_apc_notebooks.py` checks patch idempotence, GPU routing strings, renderer edge cases with a fake display, geometric equivalence, base64/reference decoding, deduplication, fingerprint resume, per-mode/circular scoring, sample errors/invalid continuation and notebook syntax. It does not claim native extension build, rendering on Kaggle, VRAM fit, accuracy or throughput are validated. End-to-end Kaggle confirmation is the sanity notebook's purpose.

## Paper-task subset validation

`validate_paper_tasks.py` checks all 2,078 selected rows against the locally available dataset metadata, including 688 visibility conversions. Counts: left/right 698 rows / 349 groups; visibility 688 / 344; facing 692 / 346. The full result includes `paired_comparison.json` with direct-vs-APC row/circular deltas and per-task circular deltas. The clearly named full subset notebook is `apc_qwen25vl_3b_paper_tasks_kaggle.ipynb`; the previous full filename is updated to the same subset behavior.
