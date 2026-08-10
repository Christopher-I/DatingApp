"""Real calibration run against a live Tinder session.

    python -m copilot.run_calibration --chrome-profile "/path/to/Chrome/Profile" \
        --limit 300 --db copilot.db

What it does:
  1. Opens your logged-in Chrome, scrapes the my-likes grid, and stores each
     liked profile's photos as CLIP embeddings (positives). Photos are fetched,
     embedded, and dropped — never written to disk.
  2. Fits the classifier from the store and prints a held-out AUC so you can see
     how separable the data is so far.

This needs the ML environment (torch + open_clip) and Selenium +
undetected-chromedriver, and a real logged-in session. Set COPILOT_EMBEDDER=clip
(or it defaults to the dependency-free hash embedder, whose numbers are
meaningless). Automating Tinder violates its ToS; this is your account and your
risk.

Negatives (passes) come from a labeling session, which needs live swipe capture
and is not wired here yet — this run gets the positives in and gives you a first
readiness read.
"""

from __future__ import annotations

import argparse
import os

from .brain.embeddings import HashEmbedder, get_embedder
from .brain.matcher import LogisticClassifier, Matcher
from .calibration import fit_from_store, import_likes
from .config import Config, MatchConfig, from_env
from .data.store import SQLiteStore
from .eval import evaluate_vectors


def run(chrome_profile: str | None, limit: int | None, db_path: str,
        headless: bool = False) -> None:
    config: Config = from_env()
    embedder = get_embedder(config)
    if isinstance(embedder, HashEmbedder):
        print("WARNING: using the hash embedder — set COPILOT_EMBEDDER=clip for "
              "real embeddings. Numbers below are meaningless with hash.\n")

    store = SQLiteStore(db_path)

    # Import defensively so this module still imports without selenium installed.
    from .drivers.tinder_web import TinderWebDriver

    driver = TinderWebDriver(chrome_profile_dir=chrome_profile,
                             headless=headless, source="likes")
    print("opening Chrome and scraping my-likes... (verify selectors if this "
          "finds nothing)")
    driver.start()
    try:
        result = import_likes(driver, embedder, store, app="tinder", limit=limit)
    finally:
        driver.stop()
    print(f"imported {result.profiles} liked profiles, "
          f"{result.labels} positive labels")

    matcher = Matcher(embedder, LogisticClassifier(), config.match)
    fit = fit_from_store(store, matcher)
    print(f"store now holds like={fit.likes} pass={fit.passes}; "
          f"classifier ready={fit.ready}")
    if fit.passes == 0:
        print("no negatives yet — run a labeling session to capture passes "
              "before trusting any score.")
        store.close()
        return

    data = store.get_swipe_labels("tinder")
    vectors = [v for v, _ in data]
    labels = [l for _, l in data]
    auc, acc = evaluate_vectors(vectors, labels)
    print(f"held-out: AUC={auc:.3f}  acc@.5={acc:.3f}")
    store.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate from your Tinder likes.")
    parser.add_argument("--chrome-profile", default=os.environ.get("COPILOT_CHROME_PROFILE"),
                        help="Path to your logged-in Chrome profile directory.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max liked profiles to import.")
    parser.add_argument("--db", default=os.environ.get("COPILOT_DB_PATH") or "copilot.db",
                        help="SQLite database path.")
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    run(args.chrome_profile, args.limit, args.db, args.headless)


if __name__ == "__main__":
    main()
