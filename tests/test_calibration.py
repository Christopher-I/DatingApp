from copilot.brain.embeddings import HashEmbedder
from copilot.brain.matcher import LogisticClassifier, Matcher, Label
from copilot.calibration import fit_from_store, import_likes, run_labeling_session
from copilot.config import MatchConfig
from copilot.data.store import SQLiteStore
from copilot.drivers.mock import MockDriver


def test_import_likes_stores_positives():
    emb = HashEmbedder(dim=16)
    store = SQLiteStore(":memory:")
    driver = MockDriver(count=10, seed=1, photos_per_profile=3)
    result = import_likes(driver, emb, store, app="tinder")
    assert result.profiles == 10
    assert result.labels == 30  # 10 profiles x 3 photos
    rows = store.get_swipe_labels("tinder")
    assert len(rows) == 30
    assert all(label == "like" for _, label in rows)
    store.close()


def test_labeling_session_records_both_classes_and_swipes():
    emb = HashEmbedder(dim=16)
    store = SQLiteStore(":memory:")
    driver = MockDriver(count=10, seed=2, photos_per_profile=2)

    def decide(profile):
        # like the even-numbered mock refs, pass the odd — deterministic
        return Label.LIKE if int(profile.external_ref[-1]) % 2 == 0 else Label.PASS

    stats = run_labeling_session(driver, emb, store, decide, max_profiles=10)
    assert stats.profiles == 10
    assert stats.likes == 5
    assert stats.passes == 5
    assert stats.labels == 20  # 10 profiles x 2 photos
    # the deck actually advanced: one swipe recorded per profile
    assert len(driver.swipes) == 10
    rows = store.get_swipe_labels()
    assert {label for _, label in rows} == {"like", "pass"}
    store.close()


def test_fit_from_store_ready_with_enough_per_class():
    emb = HashEmbedder(dim=16)
    store = SQLiteStore(":memory:")
    # 4 like profiles, 4 pass profiles, 2 photos each -> 8 per class.
    import_likes(MockDriver(count=4, seed=3, photos_per_profile=2), emb, store)
    run_labeling_session(
        MockDriver(count=4, seed=4, photos_per_profile=2), emb, store,
        decide=lambda p: Label.PASS, max_profiles=4,
    )
    matcher = Matcher(emb, LogisticClassifier(), MatchConfig())
    result = fit_from_store(store, matcher, min_per_class=5)
    assert result.likes == 8
    assert result.passes == 8
    assert result.ready is True
    store.close()


def test_fit_from_store_not_ready_below_threshold():
    emb = HashEmbedder(dim=16)
    store = SQLiteStore(":memory:")
    import_likes(MockDriver(count=1, seed=5, photos_per_profile=1), emb, store)
    matcher = Matcher(emb, LogisticClassifier(), MatchConfig())
    result = fit_from_store(store, matcher, min_per_class=5)
    assert result.ready is False  # only likes, and too few
    store.close()
