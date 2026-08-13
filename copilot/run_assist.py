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
from .brain.matcher import Label, LogisticClassifier, Matcher
from .config import from_env
from .data.store import SQLiteStore
from .eval import calibrate_skip_threshold, evaluate_vectors


def run(cdp_url: str, db_path: str, limit: int, refit_every: int) -> None:
    config = from_env()
    embedder = get_embedder(config)
    if isinstance(embedder, HashEmbedder):
        print("WARNING: hash embedder — set COPILOT_EMBEDDER=clip.\n")

    store = SQLiteStore(db_path)
    matcher = Matcher(embedder, LogisticClassifier(), config.match)
    skip_thr = 0.0  # p_like below this = confident-enough "no" to auto-skip

    def refit() -> tuple[int, int]:
        nonlocal skip_thr
        recs = store.get_swipe_labels("tinder", source="recs")
        likes = sum(1 for _, l in recs if l == "like")
        passes = len(recs) - likes
        if likes and passes:
            vecs = [v for v, _ in recs]
            labs = [l for _, l in recs]
            matcher.classifier.fit(
                vecs, [Label.LIKE if l == "like" else Label.PASS for l in labs]
            )
            skip_thr = calibrate_skip_threshold(vecs, labs, target_miss=0.06)
        return likes, passes

    rl, rp = refit()
    if matcher.classifier.is_ready:
        recs = store.get_swipe_labels("tinder", source="recs")
        auc, _ = evaluate_vectors([v for v, _ in recs], [l for _, l in recs])
        print(f"model trained on {rl + rp} recs labels ({rl} like / {rp} pass); "
              f"held-out AUC≈{auc:.3f}")
        if skip_thr > 0:
            print(f"auto-skip cutoff: p_like < {skip_thr:.2f} (tuned to lose <6% of "
                  "your likes). Below it = SKIP, otherwise you decide. No auto-like — "
                  "the model can't reliably spot your yeses.")
        else:
            print("no safe auto-skip region yet — everything goes to review.")
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
    would_skip = skip_cost = 0   # model would auto-skip; and of those, ones you liked
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
                score = pending["score"]
                if score is not None:
                    model_like = score >= 0.5
                    hit = (model_like == (label == Label.LIKE))
                    agree += int(hit)
                    scored += 1
                    if skip_thr > 0 and score < skip_thr:
                        would_skip += 1
                        if label == Label.LIKE:
                            skip_cost += 1
                    print(f"[{n}] you={direction:<9} p_like={score:.2f}  "
                          f"{'AGREE' if hit else 'disagree'}  "
                          f"(agree {agree}/{scored} = {agree/scored:.0%})")
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
                        call = "SKIP" if (skip_thr > 0 and score < skip_thr) else "keep"
                        print(f"  → {call:<4} (p_like={score:.2f})")
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
        print(f"\nagreed with you on {agree}/{scored} = {agree / scored:.0%} of swipes.")
        if would_skip:
            print(f"would have auto-skipped {would_skip}/{scored} = "
                  f"{would_skip/scored:.0%} of the deck for you, "
                  f"wrongly skipping {skip_cost} you liked.")
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
