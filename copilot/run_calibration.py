"""Real calibration run against your open Chrome (attach, no new browser).

    # 1. Quit Chrome fully, then relaunch it with the debug port:
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --remote-debugging-port=9222
    # 2. Then:
    COPILOT_EMBEDDER=clip python -m copilot.run_calibration --limit 300 --db copilot.db

What it does:
  1. Attaches to your already-logged-in Chrome, scrapes the my-likes grid, and
     stores each liked profile's photos as CLIP embeddings (positives). Photos are
     fetched, embedded, and dropped — never written to disk.
  2. Fits the classifier from the store and prints a held-out AUC so you can see
     how separable the data is so far.

Needs the ML environment (torch + open_clip) and Playwright (pip install
playwright — no `playwright install` needed, it attaches to your Chrome). Set
COPILOT_EMBEDDER=clip or it falls back to the hash embedder, whose numbers are
meaningless. Automating Tinder violates its ToS; your account, your risk.

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


def run(cdp_url: str, limit: int | None, db_path: str) -> None:
    config: Config = from_env()
    embedder = get_embedder(config)
    if isinstance(embedder, HashEmbedder):
        print("WARNING: using the hash embedder — set COPILOT_EMBEDDER=clip for "
              "real embeddings. Numbers below are meaningless with hash.\n")

    store = SQLiteStore(db_path)

    # Import defensively so this module still imports without playwright installed.
    from .drivers.tinder_cdp import TinderDriver

    driver = TinderDriver(cdp_url=cdp_url, source="likes")
    print(f"attaching to Chrome at {cdp_url} and scraping my-likes... "
          "(verify selectors if this finds nothing)")
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
    parser.add_argument("--cdp-url", default=os.environ.get("COPILOT_CDP_URL") or "http://127.0.0.1:9222",
                        help="DevTools URL of your running Chrome (launch it with "
                             "--remote-debugging-port=9222).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max liked profiles to import.")
    parser.add_argument("--db", default=os.environ.get("COPILOT_DB_PATH") or "copilot.db",
                        help="SQLite database path.")
    args = parser.parse_args()
    run(args.cdp_url, args.limit, args.db)


if __name__ == "__main__":
    main()
