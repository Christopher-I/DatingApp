"""Calibration capture: turn real swipes into stored, labelled embeddings.

Two sources of labelled data feed the matching engine:

  import_likes()          walk the owner's existing likes (the my-likes deck) and
                          store each as a positive. Free positives, no swiping.

  run_labeling_session()  the owner swipes a fresh deck in labelling mode; each
                          left/right becomes a stored pass/like AND advances the
                          real deck. This is where the negatives come from, since
                          the apps don't expose who you passed.

Both embed each photo and persist only the vector (never the photo). Then:

  fit_from_store()        read the stored {vector, label} rows and fit the
                          classifier — this is the continuous-learning step; every
                          new swipe just appends a row and a cheap refit sharpens
                          the model.

The `decide` callback in a labelling session is the owner's actual swipe in
production (driven by his left/right in the browser); tests pass a function.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable

from .brain.matcher import Label
from .drivers.base import Direction, Profile

_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp")


def import_image_files(folder: str, embedder, store, label: Label = Label.PASS,
                       app: str = "tinder") -> int:
    """Embed every image in `folder` and store it under `label`. Lets the owner
    seed examples from a folder of saved photos (e.g. hand-picked "no" examples)
    without swiping. Returns the number of images imported. Images are read,
    embedded, and dropped — never copied anywhere."""
    label_value = label.value if isinstance(label, Label) else label
    count = 0
    for name in sorted(os.listdir(folder)):
        if not name.lower().endswith(_IMAGE_EXTS):
            continue
        try:
            with open(os.path.join(folder, name), "rb") as fh:
                data = fh.read()
            vector = embedder.embed_image(data)
        except Exception:
            continue  # skip unreadable/undecodable files
        store.add_swipe_label(app, vector, label_value, primary=True)
        count += 1
    return count


@dataclass
class ImportResult:
    profiles: int
    labels: int


def import_likes(driver, embedder, store, app: str = "tinder",
                 limit: int | None = None) -> ImportResult:
    """Capture every profile the driver serves as a positive example."""
    profiles = 0
    labels = 0
    while limit is None or profiles < limit:
        profile = driver.get_next_profile()
        if profile is None:
            break
        profiles += 1
        for i, photo in enumerate(profile.photos):
            store.add_swipe_label(
                app, embedder.embed_image(photo), Label.LIKE.value,
                primary=(i == profile.primary_index),
            )
            labels += 1
    return ImportResult(profiles=profiles, labels=labels)


@dataclass
class LabelingStats:
    profiles: int
    likes: int
    passes: int
    labels: int


def run_labeling_session(driver, embedder, store,
                         decide: Callable[[Profile], Label],
                         app: str = "tinder",
                         max_profiles: int = 300) -> LabelingStats:
    """Owner labels a fresh deck. Each decision is stored and applied to the deck."""
    profiles = likes = passes = labels = 0
    while profiles < max_profiles:
        profile = driver.get_next_profile()
        if profile is None:
            break
        label = decide(profile)
        profiles += 1
        for i, photo in enumerate(profile.photos):
            store.add_swipe_label(
                app, embedder.embed_image(photo), label.value,
                primary=(i == profile.primary_index),
            )
            labels += 1
        if label == Label.LIKE:
            likes += 1
            driver.swipe(Direction.LIKE)
        else:
            passes += 1
            driver.swipe(Direction.PASS)
    return LabelingStats(profiles=profiles, likes=likes, passes=passes, labels=labels)


@dataclass
class FitResult:
    ready: bool
    likes: int
    passes: int


def fit_from_store(store, matcher, app: str | None = None,
                   min_per_class: int | None = None) -> FitResult:
    """Load stored {vector, label} rows and fit the classifier. No re-embedding —
    the vectors are already persisted, so this is a cheap refit."""
    data = store.get_swipe_labels(app)
    vectors = [v for v, _ in data]
    labels = [Label.LIKE if l == "like" else Label.PASS for _, l in data]
    matcher.classifier.fit(vectors, labels)
    likes = sum(1 for l in labels if l == Label.LIKE)
    passes = len(labels) - likes
    threshold = (min_per_class if min_per_class is not None
                 else matcher.config.min_labels_per_class)
    ready = matcher.classifier.is_ready and likes >= threshold and passes >= threshold
    return FitResult(ready=ready, likes=likes, passes=passes)
