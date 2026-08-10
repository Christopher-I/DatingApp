"""Run the embedder comparison.  python -m copilot.compare

With no real data, this runs on synthetic MockDriver profiles so you can see the
harness work end to end. Those numbers are meaningless — the hash embedder makes
random vectors, so AUC will sit near 0.5. The point is to prove the pipeline runs
and to be ready for the real thing.

To get real numbers you need two things this can't fabricate:
  1. Real labeled photos: your likes (positives) and passes (negatives).
  2. A working ML environment (Python 3.12 + torch + open_clip/transformers) so
     clip / siglip / dinov2 actually load. This file's default 3.14 process only
     has the hash embedder.

Once both exist, call compare_embedders() with your labeled profiles and the same
embedder names, and read the table.
"""

from __future__ import annotations

from .brain.embeddings import HashEmbedder, get_embedder
from .brain.matcher import Label
from .config import Config, MatchConfig
from .drivers.mock import MockDriver
from .eval import compare_embedders, format_report

_LIKE_WORDS = ("coffee", "hiking", "music", "photograph")


def _synthetic_labeled(count: int = 200, seed: int = 1):
    """Fabricate (Profile, Label) pairs from the mock driver for a dry run."""
    driver = MockDriver(count=count, seed=seed)
    labeled = []
    while (profile := driver.get_next_profile()) is not None:
        text = profile.bio_text.lower()
        label = Label.LIKE if any(w in text for w in _LIKE_WORDS) else Label.PASS
        labeled.append((profile, label))
    return labeled


def _build_embedders(config: Config, names) -> dict:
    """Instantiate each requested embedder, silently skipping ones whose deps
    aren't installed (falls back to hash for the always-available baseline)."""
    out: dict = {}
    for name in names:
        emb = get_embedder(Config(embedder=name,
                                  clip_model=config.clip_model,
                                  clip_pretrained=config.clip_pretrained))
        # get_embedder falls back to HashEmbedder when deps are missing; only add
        # a real model once, and always keep the hash baseline.
        if name != "hash" and isinstance(emb, HashEmbedder):
            continue  # deps missing; skip so the table isn't misleading
        out[name] = emb
    if "hash" not in out:
        out["hash"] = HashEmbedder()
    return out


def run(names=("hash", "clip", "siglip", "dinov2"), labeled=None) -> None:
    config = Config()
    if labeled is None:
        labeled = _synthetic_labeled()
        print("NOTE: running on synthetic data with whatever embedders are "
              "installed.\nThese numbers are meaningless until you supply real "
              "labeled photos.\n")
    embedders = _build_embedders(config, names)
    reports = compare_embedders(embedders, labeled, MatchConfig())
    print(format_report(reports))
    if len(embedders) == 1:
        print("\nOnly the hash baseline is available — install torch + "
              "open_clip/transformers in a Python 3.12 env to compare real models.")


if __name__ == "__main__":
    run()
