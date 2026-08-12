"""Offline test of the my-likes collection JavaScript against a synthetic DOM.

Runs the *actual* `_COLLECT_LIKE_PHOTOS_JS` from the driver in a real headless
Chromium against a page that mimics Tinder's my-likes structure AND its
lazy-loading: each card's photo (div[role="img"] + background-image) only appears
once the card is scrolled into view, via an IntersectionObserver. This proves the
scroll-into-view + poll + dedupe logic actually loads every card, not just the few
visible at first. No Tinder session, no network.

Skips automatically if Playwright's Chromium isn't installed
(`python -m playwright install chromium`).
"""

import pytest

playwright_sync = pytest.importorskip("playwright.sync_api")
from copilot.drivers.tinder_cdp import _COLLECT_LIKE_PHOTOS_JS  # noqa: E402

_PHOTO_URLS = [
    "https://images-ssl.gotinder.com/u/aaa/photo1.jpg?Policy=abc",
    "https://images-ssl.gotinder.com/u/bbb/photo2.jpg?Policy=def",
    "https://images-ssl.gotinder.com/u/ccc/photo3.jpg?Policy=ghi",
    "https://images-ssl.gotinder.com/u/ddd/photo4.jpg?Policy=jkl",
    "https://images-ssl.gotinder.com/u/eee/photo5.jpg?Policy=mno",
]


def _card(name: str, photo_url: str) -> str:
    # Photo starts as a bare placeholder (no role="img", no background-image),
    # exactly like an unloaded Tinder card. The observer below "loads" it when it
    # scrolls into view. The gradient overlay uses background-image too and must
    # be excluded (it's a linear-gradient, not a url()).
    return f"""
    <button aria-label="Open {name}'s profile" type="button">
      <div data-testid="likesYouCard" style="min-height:250px">
        <span><div class="photo" data-url="{photo_url}"></div></span>
        <div style="background-image: linear-gradient(rgba(0,0,0,0), #000)"></div>
      </div>
    </button>
    """


_HTML = (
    "<main id='main-content'>"
    "<div class='Ovy(s)' id='scroll' style='height:300px;overflow:auto'>"
    + "".join(_card(f"P{i}", url) for i, url in enumerate(_PHOTO_URLS))
    + "</div></main>"
    "<script>"
    "const io = new IntersectionObserver((entries) => {"
    "  entries.forEach((e) => {"
    "    if (e.isIntersecting) {"
    "      const ph = e.target;"
    "      ph.setAttribute('role', 'img');"
    "      ph.style.backgroundImage = \"url('\" + ph.dataset.url + \"')\";"
    "      io.unobserve(ph);"
    "    }"
    "  });"
    "}, { root: document.getElementById('scroll'), threshold: 0.05 });"
    "document.querySelectorAll('.photo').forEach((p) => io.observe(p));"
    "</script>"
)


def test_collect_like_photos_js_with_lazy_loading():
    with playwright_sync.sync_playwright() as p:
        try:
            browser = p.chromium.launch(headless=True)
        except Exception as exc:  # chromium not installed
            pytest.skip(f"chromium unavailable: {exc}")
        try:
            page = browser.new_page(viewport={"width": 500, "height": 300})
            page.set_content(_HTML)
            urls = page.evaluate(_COLLECT_LIKE_PHOTOS_JS)
        finally:
            browser.close()

    # All five photos are collected — including the ones that only 300px-tall
    # viewport could not show until scrolled into view — gradients excluded.
    assert sorted(urls) == sorted(_PHOTO_URLS)
