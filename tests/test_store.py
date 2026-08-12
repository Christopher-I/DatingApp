from copilot.data.store import SQLiteStore


def test_swipe_labels_roundtrip():
    store = SQLiteStore(":memory:")
    store.add_swipe_label("tinder", [0.1, 0.2, 0.3], "like", primary=True)
    store.add_swipe_label("tinder", [0.4, 0.5, 0.6], "pass")
    labels = store.get_swipe_labels("tinder")
    assert len(labels) == 2
    vec, label = labels[0]
    assert vec == [0.1, 0.2, 0.3]
    assert label == "like"
    store.close()


def test_swipe_labels_filter_by_source():
    store = SQLiteStore(":memory:")
    store.add_swipe_label("tinder", [0.1], "like", source="my_likes")
    store.add_swipe_label("tinder", [0.2], "like", source="recs")
    store.add_swipe_label("tinder", [0.3], "pass", source="recs")
    assert len(store.get_swipe_labels("tinder")) == 3
    recs = store.get_swipe_labels("tinder", source="recs")
    assert len(recs) == 2
    assert {label for _, label in recs} == {"like", "pass"}
    assert len(store.get_swipe_labels("tinder", source="my_likes")) == 1
    store.close()


def test_settings_roundtrip():
    store = SQLiteStore(":memory:")
    store.set_setting("high_threshold", 0.72)
    store.set_setting("red_phrases", ["no hookups", "just moved here"])
    assert store.get_setting("high_threshold") == 0.72
    assert store.get_setting("red_phrases") == ["no hookups", "just moved here"]
    assert store.get_setting("missing", default=None) is None
    store.close()


def test_stats_accumulate_and_read():
    store = SQLiteStore(":memory:")
    store.bump_stats("2026-08-10", swipes=10, likes=4, matches=1)
    store.bump_stats("2026-08-10", swipes=5, likes=2, matches=0)
    rows = store.get_stats(days=7)
    assert rows[-1] == {"day": "2026-08-10", "swipes": 15, "likes": 6, "matches": 1}
    store.close()


def test_style_corpus_order():
    store = SQLiteStore(":memory:")
    store.add_style_message("first")
    store.add_style_message("second")
    assert store.get_style_corpus() == ["first", "second"]
    store.close()
