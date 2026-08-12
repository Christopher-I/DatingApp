"""Live assist: the model reads each recs card and shows its prediction.

    COPILOT_EMBEDDER=clip python -m copilot.run_assist --db copilot.db

Trains on your recs-labeled data only (same capture pipeline as the live deck, so
no my-likes source confound), then as you browse the swipe deck it prints its read
of the current card and, when you swipe, whether it agreed with you. You still
swipe manually; it advises and keeps learning (it refits periodically as your
swipes accumulate).

This is the ~0.77-AUC model at work: a triage aid, not an oracle. p_like is the
model's probability you'd like the profile; the band (LIKE / REVIEW / PASS) uses
the configured thresholds, which you can tune in config.py.
"""

from __future__ import annotations

import argparse
import os
import time

from .brain.embeddings import HashEmbedder, get_embedder
from .brain.matcher import Action, Label, LogisticClassifier, Matcher
from .config import from_env
from .data.store import SQLiteStore
from .eval import evaluate_vectors


def _band(matcher: Matcher, p_like: float) -> Action:
    if p_like >= matcher.config.high_threshold:
        return Action.LIKE
    if p_like < matcher.config.low_threshold:
        return Action.PASS
    return Action.REVIEW


def run(cdp_url: str, db_path: str, limit: int, refit_every: int) -> None:
    config = from_env()
    embedder = get_embedder(config)
    if isinstance(embedder, HashEmbedder):
        print("WARNING: hash embedder — set COPILOT_EMBEDDER=clip.\n")

    store = SQLiteStore(db_path)
    matcher = Matcher(embedder, LogisticClassifier(), config.match)

    def refit() -> tuple[int, int]:
        recs = store.get_swipe_labels("tinder", source="recs")
        likes = sum(1 for _, l in recs if l == "like")
        passes = len(recs) - likes
        if likes and passes:
            matcher.classifier.fit(
                [v for v, _ in recs],
                [Label.LIKE if l == "like" else Label.PASS for _, l in recs],
            )
        return likes, passes

    rl, rp = refit()
    if matcher.classifier.is_ready:
        recs = store.get_swipe_labels("tinder", source="recs")
        auc, _ = evaluate_vectors([v for v, _ in recs], [l for _, l in recs])
        print(f"model trained on {rl + rp} recs labels ({rl} like / {rp} pass); "
              f"held-out AUC≈{auc:.3f}")
    else:
        print("not enough recs labels yet — will just record until both classes exist.")

    from .drivers.tinder_cdp import TinderDriver
    driver = TinderDriver(cdp_url=cdp_url, source="recs")
    driver.start()
    driver.install_swipe_hook()
    print("Browse the deck and swipe as normal. I'll read each card and score it.\n")

    last_url = None
    pending = None        # {"vec", "score"} for the card on screen
    processed = 0
    n = agree = scored = 0
    try:
        while n < limit:
            log = driver.read_swipe_log()
            while processed < len(log):
                direction = log[processed].get("dir")
                processed += 1
                if pending is None:
                    continue
                label = Label.LIKE if direction in ("like", "superlike") else Label.PASS
                store.add_swipe_label("tinder", pending["vec"], label.value,
                                      primary=True, source="recs")
                n += 1
                if pending["score"] is not None:
                    model_like = pending["score"] >= 0.5
                    hit = (model_like == (label == Label.LIKE))
                    agree += int(hit)
                    scored += 1
                    print(f"[{n}] you={direction:<9} p_like={pending['score']:.2f}  "
                          f"{'AGREE' if hit else 'disagree'}  "
                          f"(running agree {agree}/{scored} = {agree/scored:.0%})")
                else:
                    print(f"[{n}] you={direction:<9} (model not ready)")
                pending = None
                if n % refit_every == 0:
                    refit()

            card = driver.current_card()
            url = card.get("url") if card else None
            if url and url != last_url:
                photo = driver.fetch_photo(url)
                if photo:
                    vec = embedder.embed_image(photo)
                    score = (matcher.classifier.predict_proba(vec)
                             if matcher.classifier.is_ready else None)
                    pending = {"vec": vec, "score": score}
                    if score is not None:
                        print(f"  → reads this card: {_band(matcher, score).value.upper():<6} "
                              f"(p_like={score:.2f})")
                else:
                    pending = None
                last_url = url
            time.sleep(0.25)
    except KeyboardInterrupt:
        print("\nstopped.")
    except Exception as exc:
        if "closed" in str(exc).lower() or "TargetClosed" in type(exc).__name__:
            print("\nThe Tinder tab/Chrome was closed — stopping. Everything "
                  "captured so far is saved. Reopen the deck and rerun to continue.")
        else:
            raise
    finally:
        driver.stop()

    if scored:
        print(f"\nagreed with you on {agree}/{scored} = {agree / scored:.0%} of swipes "
              "this session.")
    store.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Live model assist on the recs deck.")
    parser.add_argument("--cdp-url",
                        default=os.environ.get("COPILOT_CDP_URL") or "http://127.0.0.1:9222")
    parser.add_argument("--db", default=os.environ.get("COPILOT_DB_PATH") or "copilot.db")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--refit-every", type=int, default=20,
                        help="Refit the model every N swipes (continuous learning).")
    args = parser.parse_args()
    run(args.cdp_url, args.db, args.limit, args.refit_every)


if __name__ == "__main__":
    main()
