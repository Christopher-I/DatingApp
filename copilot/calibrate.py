"""Calibration flow demo.  python -m copilot.calibrate

Shows the real capture path end to end on the mock driver + an in-memory store:
  1. import likes  -> positives
  2. labeling pass -> positives + negatives, persisted as embeddings
  3. fit the classifier from the store and report readiness
  4. held-out AUC/accuracy on the stored vectors

With the hash embedder the AUC is meaningless (random vectors) — this proves the
capture/persist/fit/measure plumbing. On a real session (Tinder driver + CLIP in
the 3.12 env) the same calls learn your pattern from your own swipes, implicitly,
with no attribute tagging of anyone.
"""

from __future__ import annotations

from .brain.embeddings import get_embedder
from .brain.matcher import Label, LogisticClassifier, Matcher
from .calibration import fit_from_store, import_likes, run_labeling_session
from .config import Config, MatchConfig
from .data.store import SQLiteStore
from .drivers.mock import MockDriver
from .eval import evaluate_vectors

_LIKE_WORDS = ("coffee", "hiking", "music", "photograph")


def _decide(profile) -> Label:
    text = profile.bio_text.lower()
    return Label.LIKE if any(w in text for w in _LIKE_WORDS) else Label.PASS


def run() -> None:
    config = Config()  # hash embedder by default
    embedder = get_embedder(config)
    store = SQLiteStore(":memory:")

    imported = import_likes(MockDriver(count=80, seed=1), embedder, store)
    print(f"1. imported likes:   {imported.profiles} profiles, "
          f"{imported.labels} positive labels")

    stats = run_labeling_session(
        MockDriver(count=120, seed=2), embedder, store, _decide, max_profiles=120
    )
    print(f"2. labeling session: {stats.profiles} profiles "
          f"({stats.likes} like / {stats.passes} pass), {stats.labels} labels")

    matcher = Matcher(embedder, LogisticClassifier(), MatchConfig(min_labels_per_class=10))
    fit = fit_from_store(store, matcher, min_per_class=10)
    print(f"3. fit from store:   like={fit.likes} pass={fit.passes} ready={fit.ready}")

    data = store.get_swipe_labels()
    vectors = [v for v, _ in data]
    labels = [l for _, l in data]
    auc, acc = evaluate_vectors(vectors, labels)
    print(f"4. held-out check:   AUC={auc:.3f}  acc@.5={acc:.3f}  "
          f"(meaningless with the hash embedder — use CLIP on real photos)")

    store.close()


if __name__ == "__main__":
    run()
