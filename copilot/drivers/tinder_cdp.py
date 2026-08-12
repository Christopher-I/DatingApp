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
from .tinder_web import _SEL, _BG_URL_RE, _parse_age

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
        self._page = context.pages[0] if context.pages else context.new_page()
        self._page.set_default_timeout(self._fetch_timeout_ms)
        landing = ("https://tinder.com/app/my-likes" if self.source == "likes"
                   else "https://tinder.com/app/recs")
        self._page.goto(landing)

    def stop(self) -> None:
        # Disconnect Playwright but leave the owner's Chrome open and untouched.
        if self._browser is not None:
            self._browser.close()
            self._browser = None
        if self._pw is not None:
            self._pw.stop()
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

    def _text_or_empty(self, selector: str, root=None) -> str:
        found = self._css(selector, root)
        return found[0].inner_text().strip() if found else ""

    def _bg_image_url(self, element) -> str | None:
        style = element.evaluate("e => getComputedStyle(e).backgroundImage")
        if not style or style == "none":
            return None
        match = _BG_URL_RE.search(style)
        return match.group(1) if match else None

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

    def _photo_bytes_from(self, root) -> list[bytes]:
        urls: list[str] = []
        seen: set[str] = set()
        for el in self._css(_SEL["detail_photo"], root):
            url = self._bg_image_url(el)
            if url and url.startswith("http") and url not in seen:
                seen.add(url)
                urls.append(url)
        photos: list[bytes] = []
        for url in urls:
            data = self._fetch_image_bytes(url)
            if data:
                photos.append(data)
        return photos

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

        cards = self._css(_SEL["card"])
        if not cards:
            return None
        card = cards[0]
        for _ in range(8):
            nxt = self._css(_SEL["next_photo"], card)
            if not nxt:
                break
            try:
                nxt[0].click()
            except Exception:
                break
        photos = self._photo_bytes_from(card)
        if not photos:
            return None
        age = _parse_age(self._text_or_empty(_SEL["card_age"], card))
        bio = self._text_or_empty(_SEL["card_bio"], card)
        return Profile(
            external_ref=self._ref_from(photos, "tinder-card"),
            photos=photos, age=age, bio_text=bio,
        )

    def swipe(self, direction: Direction) -> Direction:
        if self.source != "recs":
            raise RuntimeError("swipe() is only valid on the recs deck (source='recs')")
        selector = {
            Direction.LIKE: _SEL["like_button"],
            Direction.PASS: _SEL["pass_button"],
            Direction.SUPERLIKE: _SEL["superlike_button"],
        }[direction]
        buttons = self._css(selector)
        if not buttons:
            raise RuntimeError(f"swipe button not found: {selector}")
        buttons[0].click()
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
