"""Red-flag content filter.

A hard-pass override, independent of the match score. It only detects
self-disclosed content in the profile (bio text, stated fields, image vibe) — not
inferred traits. Every hit carries a human-readable reason so the owner can audit
and tune the thresholds.

  bio keywords    substring match against an owner-editable phrase list
  bio similarity  embedding similarity of bio sentences to example phrases
  image vibe      zero-shot CLIP text-image similarity to prompts like
                  "dark/gory/creepy imagery"
  stated fields   e.g. children, where the app exposes them
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .vecmath import cosine


@dataclass
class RedFlagResult:
    hit: bool
    reasons: list[str] = field(default_factory=list)


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    # Strip trailing sentence punctuation so "in therapy." matches "in therapy".
    cleaned = [p.strip().rstrip(".!?").strip() for p in parts]
    return [p for p in cleaned if p]


class RedFlagFilter:
    def __init__(self, embedder, config) -> None:
        self.embedder = embedder
        self.config = config  # RedFlagConfig
        self._example_vectors = [
            embedder.embed_text(p) for p in config.example_phrases
        ]
        self._image_prompt_vectors = [
            embedder.embed_text(p) for p in config.image_prompts
        ]

    def check(self, profile) -> RedFlagResult:
        reasons: list[str] = []

        bio = profile.bio_text or ""
        lowered = bio.lower()
        for phrase in self.config.keyword_phrases:
            if phrase.lower() in lowered:
                reasons.append(f"bio keyword: {phrase!r}")

        if self._example_vectors:
            for sentence in _sentences(bio):
                vec = self.embedder.embed_text(sentence)
                best = max(
                    (cosine(vec, ev) for ev in self._example_vectors),
                    default=0.0,
                )
                if best >= self.config.bio_similarity_threshold:
                    reasons.append(
                        f"bio matches a flagged theme (sim={best:.2f}): {sentence!r}"
                    )
                    break

        if self._image_prompt_vectors and profile.photos:
            for photo in profile.photos:
                img_vec = self.embedder.embed_image(photo)
                for prompt, pv in zip(self.config.image_prompts, self._image_prompt_vectors):
                    sim = cosine(img_vec, pv)
                    if sim >= self.config.image_similarity_threshold:
                        reasons.append(f"image vibe {prompt!r} (sim={sim:.2f})")
                        break
                if reasons and reasons[-1].startswith("image vibe"):
                    break

        # Stated-field disqualifiers (e.g. children) are handled by the engine's
        # structured pre-filter, which runs before this filter.
        return RedFlagResult(hit=bool(reasons), reasons=reasons)
