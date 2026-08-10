from copilot.brain.embeddings import HashEmbedder
from copilot.brain.matcher import (
    Action,
    CentroidClassifier,
    Label,
    LogisticClassifier,
    Matcher,
)
from copilot.config import MatchConfig


def _labelled(embedder, texts, label):
    return [(embedder.embed_text(t), label) for t in texts]


def test_centroid_separates_two_clusters():
    emb = HashEmbedder(dim=64)
    likes = [emb.embed_text(f"like-{i}") for i in range(8)]
    passes = [emb.embed_text(f"pass-{i}") for i in range(8)]
    clf = CentroidClassifier()
    clf.fit(likes + passes, [Label.LIKE] * 8 + [Label.PASS] * 8)

    # A known "like" vector should score above a known "pass" vector.
    like_score = clf.predict_proba(emb.embed_text("like-0"))
    pass_score = clf.predict_proba(emb.embed_text("pass-0"))
    assert like_score > pass_score


def test_classifier_not_ready_with_single_class():
    clf = CentroidClassifier()
    clf.fit([[1.0, 0.0]], [Label.LIKE])
    assert clf.is_ready is False  # needs both a like and a pass centroid


def test_matcher_review_when_not_ready():
    emb = HashEmbedder(dim=32)
    matcher = Matcher(emb, CentroidClassifier(), MatchConfig())
    score = matcher.score_embeddings([emb.embed_text("x")])
    assert score.action == Action.REVIEW


def test_matcher_thresholds_map_to_actions():
    emb = HashEmbedder(dim=32)

    class FixedClassifier:
        is_ready = True

        def __init__(self, value):
            self.value = value

        def fit(self, v, l):
            pass

        def predict_proba(self, vector):
            return self.value

    cfg = MatchConfig(high_threshold=0.7, low_threshold=0.4)
    high = Matcher(emb, FixedClassifier(0.9), cfg)
    mid = Matcher(emb, FixedClassifier(0.5), cfg)
    low = Matcher(emb, FixedClassifier(0.1), cfg)
    v = emb.embed_text("x")
    assert high.score_embeddings([v]).action == Action.LIKE
    assert mid.score_embeddings([v]).action == Action.REVIEW
    assert low.score_embeddings([v]).action == Action.PASS


def test_primary_photo_weighting():
    emb = HashEmbedder(dim=16)

    class SeqClassifier:
        is_ready = True

        def fit(self, v, l):
            pass

        def predict_proba(self, vector):
            # score encoded in the first component sign for the test
            return 0.9 if vector[0] > 0 else 0.1

    cfg = MatchConfig(primary_photo_weight=3.0)
    matcher = Matcher(emb, SeqClassifier(), cfg)
    strong = [1.0] + [0.0] * 15   # -> 0.9
    weak = [-1.0] + [0.0] * 15    # -> 0.1

    primary_strong = matcher.score_embeddings([strong, weak, weak], primary_index=0)
    primary_weak = matcher.score_embeddings([weak, strong, strong], primary_index=0)
    # Same photos, but weighting the strong one as primary yields a higher aggregate.
    assert primary_strong.aggregate > primary_weak.aggregate
    assert primary_strong.minimum == 0.1
    assert primary_strong.maximum == 0.9


def test_logistic_falls_back_without_sklearn_or_second_class():
    emb = HashEmbedder(dim=16)
    clf = LogisticClassifier()
    clf.fit([emb.embed_text("a")], [Label.LIKE])  # single class -> fallback
    # Should not raise; returns a probability.
    p = clf.predict_proba(emb.embed_text("a"))
    assert 0.0 <= p <= 1.0
