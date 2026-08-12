"""Tinder driver that ATTACHES to your already-open Chrome (no new browser).

Uses Playwright's CDP (Chrome DevTools Protocol) connection to drive the Chrome
you already have running and logged in. No separate automation profile, no second
login, nothing new to install a browser for. It runs only while you run it, on
your open machine, so it naturally satisfies "only when the PC is open."

How to use it:
  1. Fully quit Chrome (so it can reopen with the debug port on your normal
     profile — a running Chrome won't expose the port).
  2. Relaunch Chrome with the debug port. On macOS:
       "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
           --remote-debugging-port=9222
     Your Tinder login is already there — it's your normal profile.
  3. Run the app; this driver connects to http://localhost:9222.

Same `Driver` interface as the rest of the app, so import_likes / labeling /
run_calibration all work unchanged. Selectors and helpers are shared with the
Selenium driver (`tinder_web._SEL`) so there's one place to fix when the site
changes. Playwright is an optional dep, imported lazily. Automating Tinder
violates its ToS; it's your account and your risk.
"""

from __future__ import annotations

import hashlib

from .base import Conversation, Direction, Driver, Profile
from .tinder_web import _SEL, _parse_age

# The active recs card and its primary photo + age. The deck keeps a couple of
# cards in the DOM; the current one is aria-hidden="false".
_RECS_CARD_JS = r"""() => {
  const card = document.querySelector('.recsCardboard__cards div[data-keyboard-gamepad][aria-hidden="false"]')
            || document.querySelector('.recsCardboard__cards div[data-keyboard-gamepad]');
  if (!card) return { url: null, age: '' };
  const img = card.querySelector('div[role="img"]');
  const bg = img ? (getComputedStyle(img).backgroundImage || '') : '';
  const m = bg.match(/url\(["']?(.*?)["']?\)/);
  const url = (m && m[1].startsWith('http')) ? m[1] : null;
  const ageEl = card.querySelector('span[itemprop="age"]');
  return { url, age: ageEl ? (ageEl.textContent || '') : '' };
}"""

# Observe the owner's own swipes (he swipes manually with the X / heart buttons or
# arrow keys). A single delegated listener records each swipe's direction into a
# log the session polls. Capture phase so it fires even if React stops bubbling.
# Button identity comes from the gamepad style classes; order matters —
# "super-like" contains "like", so check it before "like-default".
_SWIPE_HOOK_JS = r"""() => {
  if (window.__swipeHook) return true;
  window.__swipeHook = true;
  window.__swipeLog = [];
  const dirFromButton = (el) => {
    const b = el.closest && el.closest('button');
    if (!b) return null;
    const c = b.className || '';
    if (c.includes('gamepad-sparks-nope')) return 'pass';
    if (c.includes('gamepad-sparks-super-like')) return 'superlike';
    if (c.includes('gamepad-sparks-like-default')) return 'like';
    // backstop: the hidden a11y label inside the button ("Nope"/"Like"/"Super Like")
    const hidden = b.querySelector('.Hidden');
    const t = hidden ? (hidden.textContent || '').trim().toLowerCase() : '';
    if (t === 'nope') return 'pass';
    if (t === 'like') return 'like';
    if (t === 'super like') return 'superlike';
    return null;
  };
  document.addEventListener('click', (e) => {
    const d = dirFromButton(e.target);
    if (d) window.__swipeLog.push({ dir: d, t: Date.now() });
  }, true);
  document.addEventListener('keydown', (e) => {
    let d = null;
    if (e.key === 'ArrowLeft') d = 'pass';
    else if (e.key === 'ArrowRight') d = 'like';
    else if (e.key === 'ArrowUp') d = 'superlike';
    if (d) window.__swipeLog.push({ dir: d, t: Date.now() });
  }, true);
  return true;
}"""

# Collect each liked person's primary-photo URL. The cards are all in the DOM,
# but their photos lazy-load only when scrolled into view (an unloaded card has no
# div[role="img"] yet). So we bring each card into view, poll briefly for its
# image, and collect the URL; between rounds we scroll to the bottom in case the
# grid pages in more cards, stopping when the card count stops growing.
_COLLECT_LIKE_PHOTOS_JS = r"""async () => {
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const urls = [];
  const seen = new Set();
  const parseBg = (el) => {
    if (!el) return null;
    const m = (getComputedStyle(el).backgroundImage || '').match(/url\(["']?(.*?)["']?\)/);
    return (m && m[1].startsWith('http')) ? m[1] : null;
  };
  const collect = () => {
    document.querySelectorAll('[data-testid="likesYouCard"]').forEach((card) => {
      const u = parseBg(card.querySelector('div[role="img"]'));
      if (u && !seen.has(u)) { seen.add(u); urls.push(u); }
    });
  };
  const scroller = document.querySelector('[class~="Ovy(s)"]')
    || document.scrollingElement || document.documentElement;
  let lastCount = -1, stable = 0;
  for (let round = 0; round < 80 && stable < 3; round++) {
    const cards = Array.from(document.querySelectorAll('[data-testid="likesYouCard"]'));
    for (const card of cards) {
      card.scrollIntoView({ block: 'center' });
      // poll up to ~1s for this card's lazy image to appear
      for (let t = 0; t < 10 && !parseBg(card.querySelector('div[role="img"]')); t++) {
        await sleep(100);
      }
    }
    collect();
    try { scroller.scrollTo(0, scroller.scrollHeight); } catch (e) {}
    await sleep(500);
    const count = document.querySelectorAll('[data-testid="likesYouCard"]').length;
    if (count === lastCount) stable++; else { stable = 0; lastCount = count; }
  }
  collect();
  return urls;
}"""


class TinderDriver(Driver):
    name = "tinder"

    def __init__(self, cdp_url: str = "http://127.0.0.1:9222",
                 source: str = "recs", fetch_timeout_ms: int = 15000) -> None:
        # Note: use 127.0.0.1, not localhost — Chrome's debug port binds to IPv4,
        # and localhost can resolve to IPv6 (::1) first, giving ECONNREFUSED.
        self._cdp_url = cdp_url
        self.source = source            # "recs" or "likes"
        self._fetch_timeout_ms = fetch_timeout_ms
        self._pw = None
        self._browser = None
        self._page = None
        self._likes_queue: list[Profile] | None = None

    # --- lifecycle -------------------------------------------------------
    def start(self) -> None:
        try:
            from playwright.sync_api import sync_playwright  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dep
            raise RuntimeError(
                "TinderDriver needs Playwright: pip install playwright. "
                "(No `playwright install` needed — it attaches to your Chrome.)"
            ) from exc

        self._pw = sync_playwright().start()
        try:
            self._browser = self._pw.chromium.connect_over_cdp(self._cdp_url)
        except Exception as exc:  # pragma: no cover - needs a live Chrome
            self._pw.stop()
            self._pw = None
            raise RuntimeError(
                f"could not attach to Chrome at {self._cdp_url}. Launch Chrome with "
                "--remote-debugging-port=9222 first (quit Chrome fully, then relaunch)."
            ) from exc

        context = (self._browser.contexts[0] if self._browser.contexts
                   else self._browser.new_context())
        pages = list(context.pages)
        # Prefer the actual Tinder tab rather than whatever happens to be first.
        tinder = [p for p in pages if "tinder.com" in (p.url or "")]
        self._page = tinder[0] if tinder else (pages[0] if pages else context.new_page())
        self._page.set_default_timeout(self._fetch_timeout_ms)
        landing = ("https://tinder.com/app/my-likes" if self.source == "likes"
                   else "https://tinder.com/app/recs")
        target = "my-likes" if self.source == "likes" else "/app/recs"
        # Only navigate if not already there — reloading a live deck can drop it.
        if target not in (self._page.url or ""):
            self._page.goto(landing)

    def stop(self) -> None:
        # Disconnect Playwright but leave the owner's Chrome open and untouched.
        # Defensive: the tab/browser may already be gone.
        try:
            if self._browser is not None:
                self._browser.close()
        except Exception:
            pass
        self._browser = None
        try:
            if self._pw is not None:
                self._pw.stop()
        except Exception:
            pass
        self._pw = None
        self._page = None

    def _require_page(self):
        if self._page is None:
            raise RuntimeError("call start() before using the driver")
        return self._page

    # --- DOM helpers -----------------------------------------------------
    def _css(self, selector: str, root=None):
        root = root or self._require_page()
        return root.query_selector_all(selector)

    def _fetch_image_bytes(self, url: str) -> bytes | None:
        # Use Playwright's request API (shares the browser's cookies, and is not
        # subject to page CORS), so cross-origin CDN images fetch cleanly.
        page = self._require_page()
        try:
            resp = page.request.get(url)
        except Exception:
            return None
        if not resp.ok:
            return None
        return resp.body() or None

    # --- recs deck (labeling) -------------------------------------------
    def install_swipe_hook(self) -> None:
        """Inject the listener that records the owner's manual swipes."""
        self._require_page().evaluate(_SWIPE_HOOK_JS)

    def read_swipe_log(self) -> list[dict]:
        """Return all swipe events observed so far: [{dir, t}, ...]."""
        return self._require_page().evaluate("() => window.__swipeLog || []")

    def current_card(self) -> dict:
        """{'url': primary-photo URL | None, 'age': str} for the active recs card."""
        return self._require_page().evaluate(_RECS_CARD_JS)

    def fetch_photo(self, url: str) -> bytes | None:
        return self._fetch_image_bytes(url)

    @staticmethod
    def _ref_from(photos: list[bytes], fallback: str) -> str:
        if photos:
            return "tinder-" + hashlib.sha256(photos[0]).hexdigest()[:16]
        return fallback

    # --- likes scraping --------------------------------------------------
    def _scrape_liked_profiles(self, limit: int | None) -> list[Profile]:
        """Scroll the my-likes grid to load every card, collect each card's
        primary-photo URL, and fetch the bytes. One primary photo per liked person
        — a strong positive signal, and the spec weights the primary photo highest
        anyway. (Grabbing all of a person's photos would mean opening each profile;
        that can be layered on later.)"""
        page = self._require_page()
        page.wait_for_timeout(1500)  # let the grid render before scrolling
        urls = page.evaluate(_COLLECT_LIKE_PHOTOS_JS)
        if limit is not None:
            urls = urls[:limit]
        profiles: list[Profile] = []
        for index, url in enumerate(urls):
            data = self._fetch_image_bytes(url)
            if data:
                profiles.append(Profile(
                    external_ref=self._ref_from([data], f"tinder-like-{index}"),
                    photos=[data], age=None, bio_text="",
                ))
        return profiles

    # --- Driver interface ------------------------------------------------
    def get_next_profile(self) -> Profile | None:
        if self.source == "likes":
            if self._likes_queue is None:
                self._likes_queue = self._scrape_liked_profiles(limit=None)
            return self._likes_queue.pop(0) if self._likes_queue else None

        card = self.current_card()
        url = card.get("url") if card else None
        if not url:
            return None
        photo = self._fetch_image_bytes(url)
        if not photo:
            return None
        return Profile(
            external_ref=self._ref_from([photo], "tinder-rec"),
            photos=[photo], age=_parse_age(card.get("age") or ""), bio_text="",
        )

    def swipe(self, direction: Direction) -> Direction:
        # Used only by autopilot; the labeling session observes the owner's own
        # manual swipes instead. Arrow keys are Tinder's own shortcuts and are far
        # more stable than the icon-only buttons.
        if self.source != "recs":
            raise RuntimeError("swipe() is only valid on the recs deck (source='recs')")
        key = {
            Direction.LIKE: "ArrowRight",
            Direction.PASS: "ArrowLeft",
            Direction.SUPERLIKE: "ArrowUp",
        }[direction]
        self._require_page().keyboard.press(key)
        return direction

    def open_conversations(self) -> list[Conversation]:
        page = self._require_page()
        page.goto("https://tinder.com/app/messages")
        conversations: list[Conversation] = []
        hrefs = [t.get_attribute("href") for t in self._css(_SEL["thread"])]
        for href in hrefs:
            if not href:
                continue
            match_id = href.rstrip("/").rsplit("/", 1)[-1]
            page.goto(f"https://tinder.com{href}" if href.startswith("/") else href)
            messages = [
                {"from": "them", "text": b.inner_text().strip()}
                for b in self._css(_SEL["message_bubble"]) if b.inner_text().strip()
            ]
            conversations.append(Conversation(match_id=match_id, messages=messages))
        return conversations

    def send_message(self, match_id: str, text: str) -> None:
        page = self._require_page()
        page.goto(f"https://tinder.com/app/messages/{match_id}")
        inputs = self._css(_SEL["message_input"])
        if not inputs:
            raise RuntimeError("message input not found")
        inputs[0].fill(text)
        send = self._css(_SEL["send_button"])
        if send:
            send[0].click()
