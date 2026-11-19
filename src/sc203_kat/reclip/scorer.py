"""Isolated crop + blur proposal scoring from the pinned ReCLIP executor."""
from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter
from PIL import __version__ as PILLOW_VERSION

RECLIP_COMMIT = "3ed4f47d37b61f8a68a839c3b86dbd2fbf252af5"
CLIP_COMMIT = "d05afc436d78f1c48dc0dbf8e5980a9d471f35f6"
CLIP_PACKAGE = f"clip @ git+https://github.com/openai/CLIP.git@{CLIP_COMMIT}"
BACKBONE = "ViT-B/32"
WEIGHT_SHA256 = "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af"
WEIGHT_URL = f"https://openaipublic.azureedge.net/clip/models/{WEIGHT_SHA256}/ViT-B-32.pt"


def _version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def normalized_cxcywh_to_xyxy(box, width: int, height: int) -> list[float]:
    """Convert Grounding DINO coordinates while preserving proposal order."""
    cx, cy, w, h = map(float, box)
    return [(cx - w / 2) * width, (cy - h / 2) * height,
            (cx + w / 2) * width, (cy + h / 2) * height]


class ReCLIPScorer:
    """Lazily load a single OpenAI CLIP backbone and cache ranked detector boxes."""

    def __init__(self, device="cuda:0", cache_dir=None, weight_dir=None, batch_size=32):
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        self.device = str(device)
        self.batch_size = batch_size
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.weight_dir = Path(weight_dir) if weight_dir else None
        self.blur_radius = 100
        self._model = None
        self._preprocess = None
        self._tokenize = None
        self.provenance = {
            "scorer_version": 1,
            "upstream": "https://github.com/allenai/reclip",
            "upstream_commit": RECLIP_COMMIT,
            "upstream_mode": "isolated_proposal_scoring",
            "openai_clip_commit": CLIP_COMMIT,
            "backbone": BACKBONE,
            "weights_url": WEIGHT_URL,
            "weights_sha256": WEIGHT_SHA256,
            "image_views": ["crop", "sharp_box_over_blurred_full_image"],
            "blur_radius_pixels": self.blur_radius,
            "preprocess": "224x224 square bicubic resize, center crop, OpenAI CLIP normalization",
            "text_template": "a photo of {lowercased_expression}",
            "score": "crop scaled cosine logit + blur scaled cosine logit",
            "ties": "earliest proposal index",
            "bbox_policy": "pixel xyxy; originals returned, clamp and round only for scoring",
            "relation_resolver": False,
            "device": self.device,
            "precision": "openai_clip_fp16" if self.device.startswith("cuda") else "float32",
            "batch_size": self.batch_size,
            "torch": _version("torch"),
            "torchvision": _version("torchvision"),
            "pillow": PILLOW_VERSION,
        }

    def _load(self) -> None:
        if self._model is not None:
            return
        import clip
        from torchvision import transforms

        implementation = getattr(clip, "clip", None)
        if getattr(implementation, "_MODELS", {}).get(BACKBONE) != WEIGHT_URL:
            raise ImportError(f"Expected pinned OpenAI CLIP: {CLIP_PACKAGE}")
        options = {"device": self.device, "jit": False}
        if self.weight_dir is not None:
            self.weight_dir.mkdir(parents=True, exist_ok=True)
            options["download_root"] = str(self.weight_dir)
        # clip.load validates the checkpoint against the SHA-256 in WEIGHT_URL.
        model, preprocess = clip.load(BACKBONE, **options)
        preprocess.transforms[0] = transforms.Resize(
            (224, 224), interpolation=transforms.InterpolationMode.BICUBIC,
        )
        self._model = model.eval()
        self._preprocess = preprocess
        self._tokenize = clip.tokenize

    def release(self) -> None:
        """Free the scorer before allocating the answer VLM."""
        if self._model is None:
            return
        self._model = self._preprocess = self._tokenize = None
        if self.device.startswith("cuda"):
            import torch
            torch.cuda.empty_cache()

    def _score(self, image: Image.Image, expression: str, boxes: list[list[int]]):
        import torch

        self._load()
        background = image.filter(ImageFilter.GaussianBlur(self.blur_radius))
        crop_logits, blur_logits = [], []
        with torch.inference_mode():
            tokens = self._tokenize(["a photo of " + expression.lower()]).to(self.device)
            text = self._model.encode_text(tokens)
            text = text / text.norm(dim=-1, keepdim=True)

            def logits(views):
                inputs = torch.stack([self._preprocess(view) for view in views]).to(self.device)
                features = self._model.encode_image(inputs)
                features = features / features.norm(dim=-1, keepdim=True)
                return ((self._model.logit_scale.exp() * features) @ text.T).flatten().float().cpu().tolist()

            for start in range(0, len(boxes), self.batch_size):
                batch = boxes[start:start + self.batch_size]
                crop_logits.extend(logits([image.crop(box) for box in batch]))
                blurred_views = []
                for box in batch:
                    mask = Image.new("L", image.size, 0)
                    ImageDraw.Draw(mask).rectangle(box, fill=255)
                    view = background.copy()
                    view.paste(image, mask=mask)
                    blurred_views.append(view)
                blur_logits.extend(logits(blurred_views))
        return crop_logits, blur_logits

    @staticmethod
    def _proposals(boxes, width: int, height: int) -> list[dict]:
        proposals = []
        for index, supplied in enumerate(boxes):
            values = list(map(float, supplied))
            if len(values) != 4:
                raise ValueError(f"Proposal {index} must have four xyxy coordinates")
            original = [value if math.isfinite(value) else None for value in values]
            scored = None
            if None not in original:
                clipped = [round(min(bound, max(0, value)))
                           for value, bound in zip(values, (width, height, width, height))]
                if clipped[0] < clipped[2] and clipped[1] < clipped[3]:
                    scored = clipped
            proposals.append({
                "index": index, "box": original, "scored_box": scored,
                "status": "ok" if scored is not None else "invalid_box",
                "crop_score": None, "blur_score": None, "combined_score": None, "rank": None,
            })
        return proposals

    def _cache_key(self, image: Image.Image, expression: str, proposals: list[dict]) -> str:
        payload = {
            "image_size": image.size,
            "expression": expression,
            "boxes_xyxy": [proposal["box"] for proposal in proposals],
            "scoring": self.provenance,
        }
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode("utf-8"))
        digest.update(image.tobytes())
        return digest.hexdigest()

    @staticmethod
    def _write_cache(path: Path, result: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         delete=False) as handle:
            temp = Path(handle.name)
            try:
                json.dump(result, handle, allow_nan=False)
            except BaseException:
                handle.close()
                temp.unlink(missing_ok=True)
                raise
        try:
            os.replace(temp, path)
        finally:
            temp.unlink(missing_ok=True)

    def rank_proposals(self, image_rgb, expression: str, boxes_xyxy) -> dict:
        """Return original and clamped boxes in input order, with crop/blur logits.

        A singleton gets real logits and a null margin. No box is synthesized
        when the detector did not supply a usable proposal.
        """
        if not isinstance(expression, str) or not expression.strip():
            raise ValueError("expression must be a nonempty object description")
        if not isinstance(image_rgb, Image.Image):
            image_rgb = Image.fromarray(image_rgb)
        image = image_rgb.convert("RGB")
        proposals = self._proposals(boxes_xyxy, *image.size)
        valid = [proposal for proposal in proposals if proposal["status"] == "ok"]
        key = self._cache_key(image, expression, proposals)
        path = self.cache_dir / f"{key}.json" if self.cache_dir is not None else None
        if path is not None and path.is_file():
            try:
                cached = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                cached = None
            if isinstance(cached, dict) and cached.get("cache_key") == key:
                return dict(cached, cache_hit=True)

        result = {
            "status": "no_proposals" if not proposals else "no_valid_proposals",
            "selected_index": None, "selected_box": None, "selected_scored_box": None,
            "top2_margin": None, "proposals": proposals, "cache_key": key, "cache_hit": False,
            "provenance": self.provenance,
        }
        if valid:
            crop, blur = self._score(image, expression, [proposal["scored_box"] for proposal in valid])
            if len(crop) != len(valid) or len(blur) != len(valid):
                raise RuntimeError("CLIP returned the wrong number of proposal scores")
            for proposal, crop_score, blur_score in zip(valid, crop, blur):
                combined_score = crop_score + blur_score
                if not math.isfinite(combined_score):
                    raise RuntimeError("CLIP returned a nonfinite proposal score")
                proposal.update(crop_score=crop_score, blur_score=blur_score,
                                combined_score=combined_score)
            leaders = sorted(valid, key=lambda proposal: (-proposal["combined_score"], proposal["index"]))
            for rank, proposal in enumerate(leaders, start=1):
                proposal["rank"] = rank
            winner = leaders[0]
            margin = winner["combined_score"] - leaders[1]["combined_score"] if len(leaders) > 1 else None
            result.update(status="ok", selected_index=winner["index"], selected_box=winner["box"],
                          selected_scored_box=winner["scored_box"], top2_margin=margin)
        if path is not None:
            self._write_cache(path, result)
        return result
