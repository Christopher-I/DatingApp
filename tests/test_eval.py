from copilot.brain.matcher import Label
from copilot.config import MatchConfig
from copilot.drivers.base import Profile
from copilot.eval import (
    accuracy,
    compare_embedders,
    evaluate_embedder,
    evaluate_vectors,
    roc_auc,
    train_test_split,
)


def test_roc_auc_perfect_and_random():
    # All positives score above all negatives -> AUC 1.0
    assert roc_auc([0.9, 0.8, 0.7], [1, 1, 0]) == 1.0
    # Reversed -> AUC 0.0
    assert roc_auc([0.1, 0.2, 0.9], [1, 1, 0]) == 0.0
    # Ties count as 0.5
    assert roc_auc([0.5, 0.5], [1, 0]) == 0.5


def test_roc_auc_undefined_single_class():
    import math

    assert math.isnan(roc_auc([0.9, 0.8], [1, 1]))


def test_accuracy_at_threshold():
    assert accuracy([0.9, 0.1], [1, 0], threshold=0.5) == 1.0
    assert accuracy([0.4, 0.6], [1, 0], threshold=0.5) == 0.0


def test_train_test_split_sizes_and_disjoint():
    train, test = train_test_split(range(10), test_frac=0.2, seed=0)
    assert len(test) == 2
    assert len(train) == 8
    assert set(train).isdisjoint(test)
    assert set(train) | set(test) == set(range(10))


class _SeparableEmbedder:
    """Maps a photo whose bytes start with b'L' to a positive-leaning vector and
    everything else to a negative-leaning one, so a real classifier can separate
    them. Proves the harness reports high AUC when the data is separable."""

    dim = 4

    def embed_image(self, data: bytes) -> list[float]:
        return [1.0, 0.0, 0.0, 0.0] if data[:1] == b"L" else [0.0, 1.0, 0.0, 0.0]

    def embed_text(self, text: str) -> list[float]:
        return [0.0, 0.0, 0.0, 0.0]


def _labeled_profiles(n_each: int = 20):
    profiles = []
    for i in range(n_each):
        profiles.append((Profile(external_ref=f"L{i}", photos=[b"L%d" % i]), Label.LIKE))
        profiles.append((Profile(external_ref=f"P{i}", photos=[b"P%d" % i]), Label.PASS))
    return profiles


def test_evaluate_embedder_separates_when_data_is_separable():
    labeled = _labeled_profiles()
    report = evaluate_embedder(
        "separable", _SeparableEmbedder(), labeled, MatchConfig(), seed=0
    )
    assert report.auc > 0.95
    assert report.n_test_profiles > 0
    assert report.n_train_photos > 0


def test_evaluate_vectors_separable():
    # 20 clearly-separable vectors per class -> high held-out AUC.
    likes = [[1.0, 0.0] for _ in range(20)]
    passes = [[0.0, 1.0] for _ in range(20)]
    vectors = likes + passes
    labels = ["like"] * 20 + ["pass"] * 20
    auc, acc = evaluate_vectors(vectors, labels, seed=0)
    assert auc > 0.95
    assert acc > 0.9


def test_calibrate_skip_threshold_separable():
    from copilot.eval import calibrate_skip_threshold

    # Separable: likes score high, passes low -> a safe skip cutoff exists between.
    likes = [[1.0, 0.0] for _ in range(20)]
    passes = [[0.0, 1.0] for _ in range(20)]
    thr = calibrate_skip_threshold(likes + passes, ["like"] * 20 + ["pass"] * 20)
    assert thr > 0.0  # there is a region below which auto-skip loses no likes


def test_calibrate_skip_threshold_guards():
    from copilot.eval import calibrate_skip_threshold

    # No likes to protect, or no data at all -> no auto-skip region (0.0).
    assert calibrate_skip_threshold([[1.0, 0.0]] * 5, ["pass"] * 5) == 0.0
    assert calibrate_skip_threshold([], []) == 0.0


def test_compare_sorts_best_auc_first():
    from copilot.brain.embeddings import HashEmbedder

    labeled = _labeled_profiles()
    reports = compare_embedders(
        {"separable": _SeparableEmbedder(), "hash": HashEmbedder(dim=8)},
        labeled,
        MatchConfig(),
        seed=0,
    )
    # The separable embedder should rank first (higher or equal AUC).
    assert reports[0].name == "separable"
