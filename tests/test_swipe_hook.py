"""Offline test of the swipe-observer hook against synthetic gamepad buttons.

Runs the real `_SWIPE_HOOK_JS` in a headless browser, then clicks buttons built to
match Tinder's recs gamepad (identified by their `gamepad-sparks-*` classes, with
a hidden a11y label) and clicking the inner icon span (as a real click does), and
checks the observed directions. Also checks the arrow-key backstop. No Tinder.

Skips if Playwright's Chromium isn't installed.
"""

import pytest

playwright_sync = pytest.importorskip("playwright.sync_api")
from copilot.drivers.tinder_cdp import _SWIPE_HOOK_JS  # noqa: E402


def _button(dir_class: str, label: str) -> str:
    # Mirrors the real structure: the click lands on an inner icon span, and the
    # direction class lives on the enclosing <button>.
    return (
        f'<div class="gamepad-button-wrapper">'
        f'<button type="button" class="button gamepad-button {dir_class}">'
        f'<span class="Pos(r) Z(1) Expand">'
        f'<span class="gamepad-icon-wrapper" data-btn="{label}"><svg></svg></span>'
        f'<span class="Hidden">{label}</span>'
        f"</span></button></div>"
    )


_HTML = (
    "<main>"
    + _button("Bgc($c-ds-background-gamepad-sparks-nope-default)", "Nope")
    + _button("Bgc($c-ds-background-gamepad-sparks-super-like-default)", "Super Like")
    + _button("Bgc($c-ds-background-gamepad-sparks-like-default)", "Like")
    + "</main>"
)


def test_swipe_hook_detects_button_clicks_and_keys():
    with playwright_sync.sync_playwright() as p:
        try:
            browser = p.chromium.launch(headless=True)
        except Exception as exc:
            pytest.skip(f"chromium unavailable: {exc}")
        try:
            page = browser.new_page()
            page.set_content(_HTML)
            page.evaluate(_SWIPE_HOOK_JS)

            # Click the inner icon of each button (as a real click does).
            page.click('span[data-btn="Nope"] svg')
            page.click('span[data-btn="Like"] svg')
            page.click('span[data-btn="Super Like"] svg')
            # Arrow-key backstop.
            page.keyboard.press("ArrowLeft")   # pass
            page.keyboard.press("ArrowRight")  # like

            log = page.evaluate("() => window.__swipeLog")
        finally:
            browser.close()

    dirs = [e["dir"] for e in log]
    assert dirs == ["pass", "like", "superlike", "pass", "like"]
