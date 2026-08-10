"""Offline test of the my-likes collection JavaScript against a synthetic DOM.

Runs the *actual* `_COLLECT_LIKE_PHOTOS_JS` from the driver in a real headless
Chromium against a hand-built page that mimics Tinder's my-likes card structure
(verified against a live DOM dump). No Tinder session, no network — this proves
the selector + URL-parsing + dedupe logic is correct.

Skips automatically if Playwright's Chromium isn't installed
(`python -m playwright install chromium`).
"""

import pytest

playwright_sync = pytest.importorskip("playwright.sync_api")
from copilot.drivers.tinder_cdp import _COLLECT_LIKE_PHOTOS_JS  # noqa: E402


def _card(name: str, photo_url: str | None) -> str:
    # Mirrors the real structure: a [data-testid="likesYouCard"] with a
    # div[role="img"] photo (background-image url) plus a gradient overlay div
    # (linear-gradient, must be excluded) that also uses background-image.
    photo_style = f"background-image: url('{photo_url}')" if photo_url else ""
    return f"""
    <button aria-label="Open {name}'s profile" type="button">
      <div data-testid="likesYouCard">
        <span><div role="img" style="{photo_style}"></div></span>
        <div style="background-image: linear-gradient(rgba(0,0,0,0), #000)"></div>
      </div>
    </button>
    """


_PHOTO_URLS = [
    "https://images-ssl.gotinder.com/u/aaa/photo1.jpg?Policy=abc",
    "https://images-ssl.gotinder.com/u/bbb/photo2.jpg?Policy=def",
    "https://images-ssl.gotinder.com/u/ccc/photo3.jpg?Policy=ghi",
]

_HTML = (
    "<main id='main-content'><div class='Ovy(s)' style='height:400px;overflow:auto'>"
    + _card("Alpha", _PHOTO_URLS[0])
    + _card("Beta", _PHOTO_URLS[1])
    + _card("Gamma", _PHOTO_URLS[2])
    + _card("Delta", None)  # a card whose photo hasn't loaded yet -> no URL
    + "</div></main>"
)


def test_collect_like_photos_js():
    with playwright_sync.sync_playwright() as p:
        try:
            browser = p.chromium.launch(headless=True)
        except Exception as exc:  # chromium not installed
            pytest.skip(f"chromium unavailable: {exc}")
        try:
            page = browser.new_page()
            page.set_content(_HTML)
            urls = page.evaluate(_COLLECT_LIKE_PHOTOS_JS)
        finally:
            browser.close()

    # Exactly the three loaded photo URLs, in order, gradients excluded, and the
    # not-yet-loaded card contributes nothing.
    assert urls == _PHOTO_URLS
