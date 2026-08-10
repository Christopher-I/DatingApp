"""Pure-logic tests for the CDP (attach-to-Chrome) driver. No Playwright, no
browser — the DOM parts only run against a live session."""

import pytest

from copilot.drivers.base import Direction
from copilot.drivers.tinder_cdp import TinderDriver


def test_instantiates_without_playwright():
    d = TinderDriver(source="likes")
    assert d.name == "tinder"
    assert d.source == "likes"
    assert d._cdp_url == "http://127.0.0.1:9222"


def test_ref_from_photos_is_stable():
    ref1 = TinderDriver._ref_from([b"same"], "fallback")
    ref2 = TinderDriver._ref_from([b"same"], "other")
    assert ref1 == ref2 and ref1.startswith("tinder-")
    assert TinderDriver._ref_from([], "fallback") == "fallback"


def test_requires_start():
    d = TinderDriver(source="recs")
    with pytest.raises(RuntimeError, match="start"):
        d.get_next_profile()


def test_swipe_invalid_on_likes_source():
    d = TinderDriver(source="likes")
    with pytest.raises(RuntimeError, match="recs deck"):
        d.swipe(Direction.LIKE)
