"""Tests for the Tinder driver's pure logic (no browser, no selenium).

The DOM scraping itself can only be exercised against a live logged-in session,
so these cover the parts that don't touch Selenium: URL parsing, age parsing,
ref derivation, and the no-driver / wrong-source guards.
"""

import pytest

from copilot.drivers.base import Direction
from copilot.drivers.tinder_web import TinderWebDriver, _BG_URL_RE, _parse_age


def test_parse_age():
    assert _parse_age("29") == 29
    assert _parse_age("Age 31") == 31
    assert _parse_age("") is None
    assert _parse_age("no digits") is None


def test_bg_url_regex():
    assert _BG_URL_RE.search('url("https://img/x.jpg")').group(1) == "https://img/x.jpg"
    assert _BG_URL_RE.search("url('https://a/b.webp')").group(1) == "https://a/b.webp"
    assert _BG_URL_RE.search("url(https://c/d.png)").group(1) == "https://c/d.png"


def test_ref_from_photos_is_stable():
    ref1 = TinderWebDriver._ref_from([b"same-bytes"], "fallback")
    ref2 = TinderWebDriver._ref_from([b"same-bytes"], "other")
    assert ref1 == ref2                      # derived from photo bytes, deterministic
    assert ref1.startswith("tinder-")
    assert TinderWebDriver._ref_from([], "fallback") == "fallback"


def test_requires_start_before_use():
    d = TinderWebDriver(source="recs")
    with pytest.raises(RuntimeError, match="start"):
        d.get_next_profile()


def test_swipe_invalid_on_likes_source():
    d = TinderWebDriver(source="likes")
    with pytest.raises(RuntimeError, match="recs deck"):
        d.swipe(Direction.LIKE)
