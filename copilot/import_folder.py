"""Import a folder of images as labelled examples.

    # negatives (hand-picked "no" examples):
    COPILOT_EMBEDDER=clip python -m copilot.import_folder --folder "/path/to/example of no" --label pass --db copilot.db

    # positives:
    COPILOT_EMBEDDER=clip python -m copilot.import_folder --folder "/path/to/likes" --label like --db copilot.db

Embeds every image in the folder with CLIP and stores it under the given label.
A quick way to seed examples without swiping. Use COPILOT_EMBEDDER=clip for real
embeddings (or it falls back to the meaningless hash embedder). A small hand-picked
folder is a narrower signal than your real swipes — a useful supplement, not a
replacement.
"""

from __future__ import annotations

import argparse
import os

from .brain.embeddings import HashEmbedder, get_embedder
from .brain.matcher import Label, LogisticClassifier, Matcher
from .calibration import fit_from_store, import_image_files
from .config import from_env
from .data.store import SQLiteStore
from .eval import evaluate_vectors


def run(folder: str, label: str, db_path: str) -> None:
    config = from_env()
    embedder = get_embedder(config)
    if isinstance(embedder, HashEmbedder):
        print("WARNING: hash embedder — set COPILOT_EMBEDDER=clip for real "
              "embeddings. Numbers below are meaningless with hash.\n")

    store = SQLiteStore(db_path)
    label_enum = Label.LIKE if label == "like" else Label.PASS
    count = import_image_files(folder, embedder, store, label_enum)
    print(f"imported {count} images from {folder!r} as {label_enum.value}")

    matcher = Matcher(embedder, LogisticClassifier(), config.match)
    fit = fit_from_store(store, matcher)
    print(f"store now holds like={fit.likes} pass={fit.passes}; "
          f"classifier ready={fit.ready}")
    if fit.likes and fit.passes:
        data = store.get_swipe_labels("tinder")
        vectors = [v for v, _ in data]
        labels = [l for _, l in data]
        auc, acc = evaluate_vectors(vectors, labels)
        print(f"held-out: AUC={auc:.3f}  acc@.5={acc:.3f}  "
              "(0.5 = coin flip, 1.0 = perfect)")
    else:
        print("still missing one class — add the other label to get a score.")
    store.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Import a folder of images as labels.")
    parser.add_argument("--folder", required=True, help="Folder of images to import.")
    parser.add_argument("--label", choices=("pass", "like"), default="pass")
    parser.add_argument("--db", default=os.environ.get("COPILOT_DB_PATH") or "copilot.db")
    args = parser.parse_args()
    run(args.folder, args.label, args.db)


if __name__ == "__main__":
    main()
