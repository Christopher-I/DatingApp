"""Calibration evaluation + embedder comparison.

Answers "which embedder best separates his likes from his passes, and how well?"
Splits by *profile* (never by photo, so photos of the same person can't leak
across the train/test boundary), trains the classifier on the train profiles,
then scores held-out profiles and reports AUC + accuracy.

AUC (area under the ROC curve) is the headline metric: it's threshold-free and
tells you how often the model ranks a random liked profile above a random passed
one. 0.5 is a coin flip; 1.0 is perfect. Accuracy is reported at a 0.5 cut for a
rough second read.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from .brain.matcher import Label, LogisticClassifier, Matcher


def train_test_split(indices, test_frac: float = 0.2, seed: int = 0):
    """Return (train_indices, test_indices), shuffled deterministically."""
    idx = list(indices)
    random.Random(seed).shuffle(idx)
    k = max(1, int(round(len(idx) * test_frac)))
    return idx[k:], idx[:k]


def roc_auc(scores, labels) -> float:
    """AUC via the Mann-Whitney statistic. `labels` are 1 (like) / 0 (pass).

    Returns nan if either class is empty (undefined). O(n*m); fine for calibration
    sets of a few hundred profiles.
    """
    pos = [s for s, l in zip(scores, labels) if l == 1]
    neg = [s for s, l in zip(scores, labels) if l == 0]
    if not pos or not neg:
        return float("nan")
    wins = 0.0
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5
    return wins / (len(pos) * len(neg))


def accuracy(scores, labels, threshold: float = 0.5) -> float:
    if not labels:
        return float("nan")
    correct = sum(
        1 for s, l in zip(scores, labels) if (s >= threshold) == (l == 1)
    )
    return correct / len(labels)


def _as_label(value) -> Label:
    if isinstance(value, Label):
        return value
    if value in (1, "like", "LIKE"):
        return Label.LIKE
    return Label.PASS


def evaluate_vectors(vectors, labels, classifier_factory=LogisticClassifier,
                     seed: int = 0, test_frac: float = 0.2):
    """Held-out AUC + accuracy directly on stored embedding vectors.

    Used after calibration, when the store holds {vector, label} rows (photo-level,
    not profile-level). `labels` may be `Label` values, 0/1, or 'like'/'pass'.
    Returns (auc, accuracy).
    """
    idx = list(range(len(vectors)))
    train_idx, test_idx = train_test_split(idx, test_frac, seed)
    clf = classifier_factory()
    clf.fit([vectors[i] for i in train_idx], [_as_label(labels[i]) for i in train_idx])
    scores = [clf.predict_proba(vectors[i]) for i in test_idx]
    labels01 = [1 if _as_label(labels[i]) == Label.LIKE else 0 for i in test_idx]
    return roc_auc(scores, labels01), accuracy(scores, labels01)


@dataclass
class EmbedderReport:
    name: str
    auc: float
    accuracy: float
    n_train_profiles: int
    n_test_profiles: int
    n_train_photos: int


def evaluate_embedder(name, embedder, labeled_profiles, match_config,
                      seed: int = 0, test_frac: float = 0.2) -> EmbedderReport:
    """`labeled_profiles` is a list of (Profile, Label). One classifier is trained
    per embedder on the train profiles' photos, then each test profile is scored
    with the profile aggregation used in production."""
    train_idx, test_idx = train_test_split(range(len(labeled_profiles)), test_frac, seed)

    matcher = Matcher(embedder, LogisticClassifier(), match_config)
    train_vecs: list[list[float]] = []
    train_labels: list[Label] = []
    for i in train_idx:
        profile, label = labeled_profiles[i]
        for photo in profile.photos:
            train_vecs.append(list(embedder.embed_image(photo)))
            train_labels.append(label)
    matcher.classifier.fit(train_vecs, train_labels)

    scores: list[float] = []
    labels01: list[int] = []
    for i in test_idx:
        profile, label = labeled_profiles[i]
        scores.append(matcher.score_profile(profile).aggregate)
        labels01.append(1 if label == Label.LIKE else 0)

    return EmbedderReport(
        name=name,
        auc=roc_auc(scores, labels01),
        accuracy=accuracy(scores, labels01),
        n_train_profiles=len(train_idx),
        n_test_profiles=len(test_idx),
        n_train_photos=len(train_vecs),
    )


def compare_embedders(embedders: dict, labeled_profiles, match_config,
                      seed: int = 0, test_frac: float = 0.2) -> list[EmbedderReport]:
    """Run every embedder in `embedders` (name -> Embedder) on the same split and
    return reports sorted best-AUC first."""
    reports = [
        evaluate_embedder(name, emb, labeled_profiles, match_config, seed, test_frac)
        for name, emb in embedders.items()
    ]
    reports.sort(
        key=lambda r: (r.auc if r.auc == r.auc else -1.0),  # nan sorts last
        reverse=True,
    )
    return reports


def format_report(reports) -> str:
    lines = [
        f"{'embedder':<10} {'AUC':>6} {'acc@.5':>7} "
        f"{'train_prof':>10} {'test_prof':>9} {'train_photos':>12}",
        "-" * 60,
    ]
    for r in reports:
        auc = f"{r.auc:.3f}" if r.auc == r.auc else "  nan"
        lines.append(
            f"{r.name:<10} {auc:>6} {r.accuracy:>7.3f} "
            f"{r.n_train_profiles:>10} {r.n_test_profiles:>9} {r.n_train_photos:>12}"
        )
    return "\n".join(lines)
