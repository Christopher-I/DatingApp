"""Record your own manual swipes as labels (the negatives the model needs).

    # with the debug Chrome up and on tinder.com/app/recs:
    COPILOT_EMBEDDER=clip python -m copilot.run_labeling --limit 60 --db copilot.db

You swipe normally on the desktop deck with the X / heart / star buttons (or the
Left / Right / Up arrow keys). This session watches: it captures the card you're
looking at, and when it sees you swipe, it records that profile as a pass (X) or
like (heart/star). It never swipes for you — you stay in control, at your own
human pace, which also keeps it clear of any automation/anti-ban concern.

Photos are fetched, embedded with CLIP, and dropped — never written to disk. Set
COPILOT_EMBEDDER=clip or it falls back to the meaningless hash embedder.

Stops after --limit swipes, or press Ctrl-C to stop early. Then it fits the model
from everything captured so far (your imported likes + these swipes) and prints a
held-out AUC.
"""

from __future__ import annotations

import argparse
import os
import time

from .brain.embeddings import HashEmbedder, get_embedder
from .brain.matcher import Label, LogisticClassifier, Matcher
from .calibration import fit_from_store
from .config import from_env
from .data.store import SQLiteStore
from .drivers.base import Profile
from .eval import evaluate_vectors


def run(cdp_url: str, db_path: str, limit: int) -> None:
    config = from_env()
    embedder = get_embedder(config)
    if isinstance(embedder, HashEmbedder):
        print("WARNING: hash embedder — set COPILOT_EMBEDDER=clip for real "
              "embeddings. Numbers below are meaningless with hash.\n")

    store = SQLiteStore(db_path)
    from .drivers.tinder_cdp import TinderDriver

    driver = TinderDriver(cdp_url=cdp_url, source="recs")
    driver.start()
    driver.install_swipe_hook()
    print("Watching your swipes on the recs deck. Swipe as normal with the X / "
          "heart / star buttons (or arrow keys).")
    print(f"Recording up to {limit} swipes — Ctrl-C to stop early.\n")

    last_url = None
    pending: Profile | None = None   # the card currently on screen
    processed = 0                    # swipe-log entries consumed
    likes = passes = 0
    try:
        while (likes + passes) < limit:
            # 1) Attribute any new swipes to the card that was on screen. Do this
            #    BEFORE refreshing `pending`, so a swipe is always credited to the
            #    card the owner was actually looking at.
            log = driver.read_swipe_log()
            while processed < len(log):
                direction = log[processed].get("dir")
                processed += 1
                if not pending or not pending.photos:
                    continue  # nothing captured to attribute — skip
                label = Label.LIKE if direction in ("like", "superlike") else Label.PASS
                for photo in pending.photos:
                    store.add_swipe_label(
                        "tinder", embedder.embed_image(photo), label.value,
                        primary=True, source="recs",
                    )
                if label == Label.LIKE:
                    likes += 1
                else:
                    passes += 1
                print(f"[{likes + passes}] {direction:<9} recorded  "
                      f"(likes={likes} passes={passes})")
                pending = None

            # 2) Capture the card now on screen (only re-fetch when it changes).
            card = driver.current_card()
            url = card.get("url") if card else None
            if url and url != last_url:
                photo = driver.fetch_photo(url)
                pending = Profile(external_ref="rec", photos=[photo]) if photo else None
                last_url = url

            time.sleep(0.25)
    except KeyboardInterrupt:
        print("\nstopped early.")
    finally:
        driver.stop()

    matcher = Matcher(embedder, LogisticClassifier(), config.match)
    fit = fit_from_store(store, matcher)
    print(f"\nstore now holds like={fit.likes} pass={fit.passes}; "
          f"classifier ready={fit.ready}")

    # The honest number: recs-labeled only, so both classes came from the same
    # capture pipeline (no my-likes-vs-recs source confound). This is the one to
    # trust, and it firms up as you label more.
    recs = store.get_swipe_labels("tinder", source="recs")
    recs_like = sum(1 for _, l in recs if l == "like")
    recs_pass = len(recs) - recs_like
    if recs_like and recs_pass:
        auc, acc = evaluate_vectors([v for v, _ in recs], [l for _, l in recs])
        print(f"HONEST (recs-only, {len(recs)} labels: {recs_like} like / "
              f"{recs_pass} pass): AUC={auc:.3f}  acc@.5={acc:.3f}")
        print("  (0.5 = coin flip, 1.0 = perfect; needs a few hundred labels to be stable)")
    else:
        print("keep swiping — need both likes and passes in the recs set for an honest score.")
    store.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Record your manual swipes as labels.")
    parser.add_argument("--cdp-url",
                        default=os.environ.get("COPILOT_CDP_URL") or "http://127.0.0.1:9222")
    parser.add_argument("--db", default=os.environ.get("COPILOT_DB_PATH") or "copilot.db")
    parser.add_argument("--limit", type=int, default=60,
                        help="Stop after this many swipes.")
    args = parser.parse_args()
    run(args.cdp_url, args.db, args.limit)


if __name__ == "__main__":
    main()
