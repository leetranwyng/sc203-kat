# ReCLIP instance selection for APC-Vis and SymPL

Use the corresponding sanity notebook before the full notebook. In Kaggle, enable
Internet and GPU T4 ×2, attach the existing Dataset
**ltrnnguyn/sc203-apc-trace-for-ablation**, then run all cells. The four notebooks
fetch the same Git tag. Each sanity notebook uses 5832 (Left/Right), 4798
(Visibility), and 8268 (Facing). Each full notebook uses the 120 shared IDs.
If auto-detection is ambiguous, set trace_input to the attached bundle directory
or ZIP. SymPL automatically retrieves the public frozen trace metadata and depth
arrays. Set sympl_trace only if using an audited local copy instead.

## Protocol

Both methods freeze the entities, reference viewer, question, image, proposal
order, depth and camera intrinsics. These are taken from each method's own
recorded upstream trace. The two selectors in a method share those inputs.
Replays use Qwen2.5-VL-7B and OA1. APC reruns box-dependent SAM masks, positions,
orientation and rendering. SymPL recomputes positions and reuses the saved OA1
direction for an unchanged viewer box, or runs OA1 on the new viewer box.
Layout generation and final VLM calls are rerun. ReCLIP scoring runs in a separate
process before Qwen is loaded. It receives the saved object descriptions,
excluding questions and answer choices.

The public SymPL trace matches the attached APC bundle on all 120 IDs, source
image hashes, runtime questions/options, labels, and saved APC predictions.
The historical baselines scored 36/120 (APC-Vis) and 61/120 (SymPL).
These references are reported separately from the new same-environment baseline.
The 40 archived Visibility rows already use the transformed yes/no question.
The experiment preserves those inputs verbatim.

The scoring implementation uses isolated proposal scoring from AllenAI ReCLIP
commit 3ed4f47d37b61f8a68a839c3b86dbd2fbf252af5, with one OpenAI
ViT-B/32 instead of the paper's backbone ensemble. OpenAI CLIP is pinned at
d05afc436d78f1c48dc0dbf8e5980a9d471f35f6. The checkpoint SHA-256 is
40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af.
Each proposal is scored with a crop and the full image blurred with radius
100, with the sharp proposal pasted back in place. Both views are resized to
224 × 224 with bicubic interpolation and use OpenAI CLIP normalization. Text is
"a photo of " plus the lowercased expression. The crop and blur scaled cosine
logits are summed. There is no spatial relation resolver or ground-truth box
fallback. Source and runtime details are also saved with scoring decisions.

## Outputs

Read gallery.html in the notebook output directory. Keep the output directory
and its image subfolders together when downloading. paired_predictions.csv and
summary.json include accuracy, answer and box changes, branch errors, net gain,
and history drift. Errors and invalid answers count as wrong for attempted pairs.
Rows missing a completed arm are reported separately. grounding_decisions.jsonl
contains all scored proposals and their original IDs. grounding_disagreements.jsonl
lists objects for manual review. For labeled evaluation, set grounding_labels
to a JSONL of records with sample_id, object_index, confirmed=true and box_xyxy
in original image pixels. Recall@K means recall over that object's recorded
proposal list at IoU ≥ 0.5. Instance selection accuracy uses the same confirmed
objects and IoU criterion. Without these annotations the grounding metrics are
unset.

## Local checks and Kaggle gate

Real pretrained CPU inference on APC trace 4830 (expression "kid") selected
proposal 0 from multiple proposals. The pinned authors' executor selected the
same box, with a maximum absolute combined logit difference of 1.91e-6 on this
trace. This verifies implementation parity on that input. It is not a
grounding label or an end-to-end accuracy result. The local checks also covered
empty/invalid/singleton proposals, index preservation, and cache identity.
Three SymPL call traces and depth arrays were fetched through the Kaggle API.
Full input preflight verified both histories against the shared cohort.
GPU sanity and full paired inference require a Kaggle session.
