"""The matching engine: few-shot classifier over frozen embeddings.

Given a profile's photo embeddings, produce a 0-1 score for how well it matches
the owner's learned type, then map that score to an action.

Classifiers (simple and robust, per the spec):
  CentroidClassifier    cosine sim to the "liked" centroid minus the "passed"
                        centroid, squashed to 0-1. No training, works with few
                        examples. Always available.
  LogisticClassifier    scikit-learn logistic regression over embeddings. Sharper
                        with more data; falls back to the centroid if sklearn is
                        absent or a class is missing.

Profile aggregation: score each photo, take a weighted mean (extra weight on the
primary photo), and also expose min/max so a single strong disqualifier photo can
be caught by a caller.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol, Sequence

from .vecmath import cosine, logistic, mean, sub


class Label(str, Enum):
    LIKE = "like"
    PASS = "pass"


class Action(str, Enum):
    LIKE = "like"
    REVIEW = "review"
    PASS = "pass"


class Classifier(Protocol):
    is_ready: bool

    def fit(self, vectors: Sequence[Sequence[float]], labels: Sequence[Label]) -> None: ...
    def predict_proba(self, vector: Sequence[float]) -> float: ...


class CentroidClassifier:
    """Cosine-to-centroid baseline. score = sigmoid(scale * (sim_like - sim_pass))."""

    def __init__(self, scale: float = 6.0) -> None:
        self.scale = scale
        self._like_centroid: list[float] | None = None
        self._pass_centroid: list[float] | None = None

    @property
    def is_ready(self) -> bool:
        return self._like_centroid is not None and self._pass_centroid is not None

    def fit(self, vectors: Sequence[Sequence[float]], labels: Sequence[Label]) -> None:
        likes = [v for v, l in zip(vectors, labels) if l == Label.LIKE]
        passes = [v for v, l in zip(vectors, labels) if l == Label.PASS]
        self._like_centroid = mean(likes) if likes else None
        self._pass_centroid = mean(passes) if passes else None

    def predict_proba(self, vector: Sequence[float]) -> float:
        if not self.is_ready:
            return 0.5
        sim_like = cosine(vector, self._like_centroid)
        sim_pass = cosine(vector, self._pass_centroid)
        return logistic(self.scale * (sim_like - sim_pass))


class LogisticClassifier:
    """scikit-learn logistic regression, falling back to CentroidClassifier."""

    def __init__(self) -> None:
        self._model = None
        self._fallback = CentroidClassifier()
        self._using_fallback = True

    @property
    def is_ready(self) -> bool:
        return self._model is not None or self._fallback.is_ready

    def fit(self, vectors: Sequence[Sequence[float]], labels: Sequence[Label]) -> None:
        self._fallback.fit(vectors, labels)
        distinct = {l for l in labels}
        if len(distinct) < 2:
            self._model = None
            self._using_fallback = True
            return
        try:
            from sklearn.linear_model import LogisticRegression  # type: ignore
        except ImportError:
            self._model = None
            self._using_fallback = True
            return
        X = [list(v) for v in vectors]
        y = [1 if l == Label.LIKE else 0 for l in labels]
        model = LogisticRegression(max_iter=1000)
        model.fit(X, y)
        self._model = model
        self._using_fallback = False

    def predict_proba(self, vector: Sequence[float]) -> float:
        if self._model is None:
            return self._fallback.predict_proba(vector)
        proba = self._model.predict_proba([list(vector)])[0]
        # class order is [0, 1]; index 1 is "like".
        return float(proba[1])


@dataclass
class ProfileScore:
    aggregate: float          # weighted-mean score used for the decision
    per_photo: list[float]    # score of each photo, in order
    minimum: float
    maximum: float
    action: Action


class Matcher:
    def __init__(self, embedder, classifier: Classifier, config) -> None:
        self.embedder = embedder
        self.classifier = classifier
        self.config = config  # MatchConfig

    def score_embeddings(self, embeddings: Sequence[Sequence[float]],
                         primary_index: int = 0) -> ProfileScore:
        if not embeddings:
            raise ValueError("score_embeddings() needs at least one embedding")
        per_photo = [self.classifier.predict_proba(e) for e in embeddings]
        weights = [
            self.config.primary_photo_weight if i == primary_index else 1.0
            for i in range(len(per_photo))
        ]
        aggregate = sum(w * p for w, p in zip(weights, per_photo)) / sum(weights)
        return ProfileScore(
            aggregate=aggregate,
            per_photo=per_photo,
            minimum=min(per_photo),
            maximum=max(per_photo),
            action=self._decide(aggregate),
        )

    def score_profile(self, profile) -> ProfileScore:
        embeddings = [self.embedder.embed_image(photo) for photo in profile.photos]
        return self.score_embeddings(embeddings, profile.primary_index)

    def _decide(self, aggregate: float) -> Action:
        if not self.classifier.is_ready:
            return Action.REVIEW
        if aggregate >= self.config.high_threshold:
            return Action.LIKE
        if aggregate < self.config.low_threshold:
            return Action.PASS
        return Action.REVIEW
