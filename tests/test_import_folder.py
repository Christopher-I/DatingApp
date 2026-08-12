import os

from copilot.brain.embeddings import HashEmbedder
from copilot.brain.matcher import Label
from copilot.calibration import import_image_files
from copilot.data.store import SQLiteStore


def _write(path: str, data: bytes) -> None:
    with open(path, "wb") as fh:
        fh.write(data)


def test_import_image_files_stores_labels(tmp_path):
    # A few fake image files plus a non-image that must be ignored.
    _write(str(tmp_path / "a.png"), b"fake-png-a")
    _write(str(tmp_path / "b.jpg"), b"fake-jpg-b")
    _write(str(tmp_path / "c.webp"), b"fake-webp-c")
    _write(str(tmp_path / "notes.txt"), b"ignore me")

    store = SQLiteStore(":memory:")
    emb = HashEmbedder(dim=16)
    count = import_image_files(str(tmp_path), emb, store, Label.PASS, app="tinder")

    assert count == 3  # the .txt is skipped
    rows = store.get_swipe_labels("tinder")
    assert len(rows) == 3
    assert all(label == "pass" for _, label in rows)
    store.close()


def test_import_image_files_like_label(tmp_path):
    _write(str(tmp_path / "x.png"), b"x")
    store = SQLiteStore(":memory:")
    count = import_image_files(str(tmp_path), HashEmbedder(dim=8), store, Label.LIKE)
    assert count == 1
    assert store.get_swipe_labels()[0][1] == "like"
    store.close()
