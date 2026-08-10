"""The evaluation orchestrator.

Order (per the spec):
  1. Structured hard-pass pre-filters (cheap; gate before the photo model runs).
     e.g. age outside 26-33, disqualifying stated fields.
  2. Red-flag content filter (hard-pass override, independent of match score).
  3. Photo matching engine -> score -> action.

Also owns continuous learning: every real swipe (owner's or approved-in-queue)
appends {embedding, label} and can trigger a cheap retrain.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .matcher import Action, Label, Matcher, ProfileScore
from .redflags import RedFlagFilter


@dataclass
class Decision:
    action: Action
    reason: str
    score: ProfileScore | None = None
    red_flags: list[str] = field(default_factory=list)


class Engine:
    def __init__(self, embedder, matcher: Matcher, redflag_filter: RedFlagFilter,
                 config) -> None:
        self.embedder = embedder
        self.matcher = matcher
        self.redflags = redflag_filter
        self.config = config  # full Config

    # --- structured pre-filter ------------------------------------------
    def _structured_pass_reason(self, profile) -> str | None:
        sc = self.config.structured
        if profile.age is not None and not (sc.min_age <= profile.age <= sc.max_age):
            return f"age {profile.age} outside [{sc.min_age}, {sc.max_age}]"
        for name in sc.disqualifying_fields:
            if profile.fields.get(name):
                return f"disqualifying field: {name}"
        return None

    # --- evaluation ------------------------------------------------------
    def evaluate(self, profile) -> Decision:
        structured = self._structured_pass_reason(profile)
        if structured is not None:
            return Decision(action=Action.PASS, reason=structured)

        red = self.redflags.check(profile)
        if red.hit:
            return Decision(
                action=Action.PASS,
                reason="red flag: " + "; ".join(red.reasons),
                red_flags=red.reasons,
            )

        score = self.matcher.score_profile(profile)
        reason = {
            Action.LIKE: "score above high threshold",
            Action.PASS: "score below low threshold",
            Action.REVIEW: "score in review band"
            if self.matcher.classifier.is_ready
            else "classifier not yet calibrated",
        }[score.action]
        return Decision(action=score.action, reason=reason, score=score)


class LearningStore:
    """In-memory accumulator of labelled embeddings feeding the classifier.

    Backed by the persistent `swipe_labels` table in production; this keeps the
    working set for cheap in-process retrains.
    """

    def __init__(self, matcher: Matcher) -> None:
        self.matcher = matcher
        self._vectors: list[list[float]] = []
        self._labels: list[Label] = []

    def __len__(self) -> int:
        return len(self._labels)

    def counts(self) -> dict[Label, int]:
        return {
            Label.LIKE: sum(1 for l in self._labels if l == Label.LIKE),
            Label.PASS: sum(1 for l in self._labels if l == Label.PASS),
        }

    def add_profile(self, profile, label: Label) -> None:
        """Add every photo of a profile under one label (the owner's swipe)."""
        for photo in profile.photos:
            self._vectors.append(list(self.matcher.embedder.embed_image(photo)))
            self._labels.append(label)

    def add_embedding(self, vector, label: Label) -> None:
        self._vectors.append(list(vector))
        self._labels.append(label)

    def retrain(self) -> bool:
        """Refit the classifier. Returns True if enough data to be trusted."""
        if not self._labels:
            return False
        self.matcher.classifier.fit(self._vectors, self._labels)
        counts = self.counts()
        min_per_class = self.matcher.config.min_labels_per_class
        return counts[Label.LIKE] >= min_per_class and counts[Label.PASS] >= min_per_class
