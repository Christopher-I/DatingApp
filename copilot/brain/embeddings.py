"""Photo/text embedders.

Two implementations behind one interface:

  HashEmbedder  deterministic, dependency-free. Not meaningful embeddings, but it
                lets the whole pipeline (calibration, scoring, red-flag checks,
                tests) run without torch/open_clip. Same bytes -> same vector.

  ClipEmbedder  the real thing: a frozen open_clip model. One forward pass per
                photo -> one 512-768d vector. Whole photo, never a cropped face,
                per the spec. Lazily imported so the core has no heavy deps.

Both embed the *whole* photo. Photos are passed as raw bytes, embedded, and the
bytes are dropped by the caller — nothing is written to disk.
"""

from __future__ import annotations

import hashlib
from typing import Protocol

from .vecmath import normalize


class Embedder(Protocol):
    dim: int

    def embed_image(self, data: bytes) -> list[float]: ...
    def embed_text(self, text: str) -> list[float]: ...


class HashEmbedder:
    """Deterministic stand-in embedder for dev and tests."""

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    def _hash_vector(self, seed: bytes) -> list[float]:
        out: list[float] = []
        counter = 0
        while len(out) < self.dim:
            digest = hashlib.sha256(seed + counter.to_bytes(4, "big")).digest()
            for i in range(0, len(digest), 4):
                if len(out) >= self.dim:
                    break
                val = int.from_bytes(digest[i : i + 4], "big")
                out.append((val / 2**32) * 2.0 - 1.0)  # map to [-1, 1)
            counter += 1
        return normalize(out)

    def embed_image(self, data: bytes) -> list[float]:
        return self._hash_vector(b"img:" + data)

    def embed_text(self, text: str) -> list[float]:
        return self._hash_vector(b"txt:" + text.encode("utf-8"))


class ClipEmbedder:
    """Frozen open_clip embedder. Imports torch/open_clip/PIL lazily."""

    def __init__(self, model_name: str = "ViT-B-32",
                 pretrained: str = "laion2b_s34b_b79k") -> None:
        try:
            import open_clip  # type: ignore
            import torch  # type: ignore
        except ImportError as exc:  # pragma: no cover - depends on optional deps
            raise RuntimeError(
                "ClipEmbedder needs `open_clip-torch` and `torch` installed. "
                "Use COPILOT_EMBEDDER=hash for the dependency-free path."
            ) from exc

        self._torch = torch
        self._model, _, self._preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self._model.eval()
        self._tokenizer = open_clip.get_tokenizer(model_name)
        with torch.no_grad():
            probe = self._model.encode_text(self._tokenizer(["probe"]))
        self.dim = int(probe.shape[-1])

    def embed_image(self, data: bytes) -> list[float]:  # pragma: no cover - needs deps
        import io
        from PIL import Image  # type: ignore

        image = Image.open(io.BytesIO(data)).convert("RGB")
        tensor = self._preprocess(image).unsqueeze(0)
        with self._torch.no_grad():
            feats = self._model.encode_image(tensor)
        return normalize(feats[0].tolist())

    def embed_text(self, text: str) -> list[float]:  # pragma: no cover - needs deps
        tokens = self._tokenizer([text])
        with self._torch.no_grad():
            feats = self._model.encode_text(tokens)
        return normalize(feats[0].tolist())


class SiglipEmbedder:
    """Google SigLIP via HuggingFace transformers. Often separates visual
    similarity better than plain CLIP. Understands text and image, so it also
    supports the zero-shot red-flag image-vibe prompts."""

    def __init__(self, model_name: str = "google/siglip-base-patch16-224") -> None:
        try:
            import torch  # type: ignore
            from transformers import AutoModel, AutoProcessor  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional deps
            raise RuntimeError(
                "SiglipEmbedder needs `transformers` and `torch`. "
                "Use COPILOT_EMBEDDER=hash for the dependency-free path."
            ) from exc
        self._torch = torch
        self._model = AutoModel.from_pretrained(model_name).eval()
        self._processor = AutoProcessor.from_pretrained(model_name)
        self.dim = int(self._model.config.text_config.hidden_size)

    def embed_image(self, data: bytes) -> list[float]:  # pragma: no cover - needs deps
        import io
        from PIL import Image  # type: ignore

        image = Image.open(io.BytesIO(data)).convert("RGB")
        inputs = self._processor(images=image, return_tensors="pt")
        with self._torch.no_grad():
            feats = self._model.get_image_features(**inputs)
        return normalize(feats[0].tolist())

    def embed_text(self, text: str) -> list[float]:  # pragma: no cover - needs deps
        inputs = self._processor(
            text=[text], return_tensors="pt", padding="max_length"
        )
        with self._torch.no_grad():
            feats = self._model.get_text_features(**inputs)
        return normalize(feats[0].tolist())


class Dinov2Embedder:
    """Meta DINOv2 via HuggingFace transformers. Pure self-supervised *visual*
    features — usually the strongest at "how visually similar is this to the ones
    he liked." Image-only: it has no text encoder, so it cannot do the zero-shot
    red-flag image prompts (use CLIP or SigLIP for that layer)."""

    def __init__(self, model_name: str = "facebook/dinov2-base") -> None:
        try:
            import torch  # type: ignore
            from transformers import AutoImageProcessor, AutoModel  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional deps
            raise RuntimeError(
                "Dinov2Embedder needs `transformers` and `torch`. "
                "Use COPILOT_EMBEDDER=hash for the dependency-free path."
            ) from exc
        self._torch = torch
        self._model = AutoModel.from_pretrained(model_name).eval()
        self._processor = AutoImageProcessor.from_pretrained(model_name)
        self.dim = int(self._model.config.hidden_size)

    def embed_image(self, data: bytes) -> list[float]:  # pragma: no cover - needs deps
        import io
        from PIL import Image  # type: ignore

        image = Image.open(io.BytesIO(data)).convert("RGB")
        inputs = self._processor(images=image, return_tensors="pt")
        with self._torch.no_grad():
            outputs = self._model(**inputs)
        return normalize(outputs.pooler_output[0].tolist())

    def embed_text(self, text: str) -> list[float]:  # pragma: no cover - needs deps
        raise NotImplementedError(
            "DINOv2 is image-only. Use CLIP or SigLIP for text/image-vibe prompts."
        )


#: registry so callers (and the comparison harness) can build embedders by name.
_EMBEDDERS = {
    "hash": lambda cfg: HashEmbedder(),
    "clip": lambda cfg: ClipEmbedder(cfg.clip_model, cfg.clip_pretrained),
    "siglip": lambda cfg: SiglipEmbedder(),
    "dinov2": lambda cfg: Dinov2Embedder(),
}


def get_embedder(config) -> Embedder:
    """Pick an embedder from config, falling back to HashEmbedder if the requested
    one needs optional deps that aren't installed."""
    factory = _EMBEDDERS.get(config.embedder)
    if factory is None:
        return HashEmbedder()
    try:
        return factory(config)
    except RuntimeError:
        # Deps missing — fall back rather than crash.
        return HashEmbedder()
