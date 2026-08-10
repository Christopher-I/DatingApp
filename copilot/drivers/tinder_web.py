"""Tinder web driver — Selenium + undetected-chromedriver.

Drives the owner's real, logged-in Chrome session (preferred over a
reverse-engineered private API: real-browser traffic survives bot detection far
better). Two sources behind one interface:

  source="likes"  scrape the my-likes grid -> Profile per liked person, with
                  photo bytes. Feeds calibration.import_likes() (positives).
  source="recs"   read the top card of the swipe deck for a labeling session,
                  and click like/pass/superlike.

Photos are fetched as raw bytes through an in-page `fetch()` (so the browser's
own cookies/origin authorize the CDN request), handed to the embedder, and
dropped. Nothing is written to disk.

IMPORTANT, and unchanged from the spec: automating Tinder violates its ToS; this
runs on the owner's own account and is his own risk. Selectors below WILL drift
and must be verified against the live DOM before a run — they are centralized in
_SEL for exactly that reason. Selenium and undetected-chromedriver are optional
deps, imported lazily; nothing here runs in CI. Anti-ban behaviour (caps, delays,
active-hours) lives in the scheduler, not here.
"""

from __future__ import annotations

import base64
import hashlib
import re

from .base import Conversation, Direction, Driver, Profile

# Centralized so they're easy to re-verify when the site changes. Placeholders —
# confirm against the live DOM before relying on them.
_SEL = {
    # swipe deck (recs)
    "card": 'div[data-testid="cardStack"] div[aria-label]',
    "card_photo": 'div[aria-label] div[style*="background-image"]',
    "card_age": 'span[itemprop="age"]',
    "card_bio": 'div.BreakWord',
    "next_photo": 'button[aria-label="Next Photo"]',
    "like_button": 'button[aria-label="Like"]',
    "pass_button": 'button[aria-label="Nope"]',
    "superlike_button": 'button[aria-label="Super Like"]',
    # my-likes grid (verified 2026-08 against the live DOM)
    "like_tile": '[data-testid="likesYouCard"]',
    "like_photo": 'div[role="img"]',        # bg-image photo inside each card
    "tile_photo": 'div[style*="background-image"]',
    # opened profile detail (after clicking a tile / expanding a card)
    "detail_photo": 'div[style*="background-image"]',
    "detail_age": 'span[itemprop="age"]',
    "detail_bio": 'div.BreakWord',
    "detail_close": 'button[aria-label="Close"]',
    # messages
    "messages_tab": 'a[href="/app/messages"]',
    "thread": 'a[href^="/app/messages/"]',
    "message_bubble": 'div[class*="msg"]',
    "message_input": 'textarea',
    "send_button": 'button[type="submit"]',
}

# Fetch an image URL from inside the page and return base64 (browser cookies apply).
_FETCH_IMAGE_JS = """
const url = arguments[0];
const done = arguments[arguments.length - 1];
fetch(url).then(r => r.blob()).then(b => {
  const reader = new FileReader();
  reader.onloadend = () => done(String(reader.result).split(',')[1] || null);
  reader.onerror = () => done(null);
  reader.readAsDataURL(b);
}).catch(() => done(null));
"""

_BG_URL_RE = re.compile(r'url\((?:"|\')?(.*?)(?:"|\')?\)')


class TinderWebDriver(Driver):
    name = "tinder"

    def __init__(self, chrome_profile_dir: str | None = None,
                 headless: bool = False, source: str = "recs",
                 fetch_timeout: float = 15.0) -> None:
        self._chrome_profile_dir = chrome_profile_dir
        self._headless = headless
        self.source = source            # "recs" or "likes"
        self._fetch_timeout = fetch_timeout
        self._driver = None
        self._likes_queue: list[Profile] | None = None  # lazily scraped in likes mode

    # --- lifecycle -------------------------------------------------------
    def start(self) -> None:
        try:
            import undetected_chromedriver as uc  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dep
            raise RuntimeError(
                "TinderWebDriver needs `selenium` and `undetected-chromedriver`. "
                "Use MockDriver for offline work."
            ) from exc

        options = uc.ChromeOptions()
        if self._chrome_profile_dir:
            # Reuse the owner's real, already-logged-in Chrome profile.
            options.add_argument(f"--user-data-dir={self._chrome_profile_dir}")
        if self._headless:
            options.add_argument("--headless=new")
        self._driver = uc.Chrome(options=options)
        self._driver.set_script_timeout(self._fetch_timeout + 5)
        landing = ("https://tinder.com/app/my-likes" if self.source == "likes"
                   else "https://tinder.com/app/recs")
        self._driver.get(landing)

    def stop(self) -> None:
        if self._driver is not None:
            self._driver.quit()
            self._driver = None

    def _require_driver(self):
        if self._driver is None:
            raise RuntimeError("call start() before using the driver")
        return self._driver

    # --- low-level DOM helpers ------------------------------------------
    def _css(self, selector: str, root=None):
        from selenium.webdriver.common.by import By  # type: ignore

        root = root or self._require_driver()
        return root.find_elements(By.CSS_SELECTOR, selector)

    def _text_or_empty(self, selector: str, root=None) -> str:
        found = self._css(selector, root)
        return found[0].text.strip() if found else ""

    def _bg_image_url(self, element) -> str | None:
        style = element.value_of_css_property("background-image")
        if not style or style == "none":
            return None
        match = _BG_URL_RE.search(style)
        return match.group(1) if match else None

    def _fetch_image_bytes(self, url: str) -> bytes | None:
        driver = self._require_driver()
        payload = driver.execute_async_script(_FETCH_IMAGE_JS, url)
        if not payload:
            return None
        try:
            return base64.b64decode(payload)
        except (ValueError, TypeError):
            return None

    def _photo_bytes_from(self, root) -> list[bytes]:
        """Collect deduped photo bytes from a card/detail element."""
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
        """Walk the my-likes grid: open each tile, gather photos/age/bio, close."""
        driver = self._require_driver()
        profiles: list[Profile] = []
        tiles = self._css(_SEL["like_tile"])
        for index, _ in enumerate(tiles):
            if limit is not None and len(profiles) >= limit:
                break
            # Re-query each iteration: opening/closing a detail invalidates handles.
            current = self._css(_SEL["like_tile"])
            if index >= len(current):
                break
            tile = current[index]
            try:
                tile.click()
            except Exception:
                continue  # tile went stale; skip
            photos = self._photo_bytes_from(driver)
            if not photos:
                # Fall back to the tile's own primary image.
                url = self._bg_image_url(tile) if self._css(_SEL["tile_photo"], tile) else None
                if url:
                    data = self._fetch_image_bytes(url)
                    if data:
                        photos = [data]
            age = _parse_age(self._text_or_empty(_SEL["detail_age"]))
            bio = self._text_or_empty(_SEL["detail_bio"])
            if photos:
                profiles.append(Profile(
                    external_ref=self._ref_from(photos, f"tinder-like-{index}"),
                    photos=photos, age=age, bio_text=bio,
                ))
            # Close the detail to return to the grid.
            close = self._css(_SEL["detail_close"])
            if close:
                try:
                    close[0].click()
                except Exception:
                    driver.get("https://tinder.com/app/my-likes")
        return profiles

    # --- Driver interface ------------------------------------------------
    def get_next_profile(self) -> Profile | None:
        if self.source == "likes":
            if self._likes_queue is None:
                self._likes_queue = self._scrape_liked_profiles(limit=None)
            return self._likes_queue.pop(0) if self._likes_queue else None

        # recs deck: read the top card
        driver = self._require_driver()
        cards = self._css(_SEL["card"])
        if not cards:
            return None
        card = cards[0]
        # Click through the card's photos to load them all, then collect.
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
        driver = self._require_driver()
        driver.get("https://tinder.com/app/messages")
        conversations: list[Conversation] = []
        threads = self._css(_SEL["thread"])
        hrefs = [t.get_attribute("href") for t in threads]
        for href in hrefs:
            if not href:
                continue
            match_id = href.rstrip("/").rsplit("/", 1)[-1]
            driver.get(href)
            messages = [
                {"from": "them", "text": b.text.strip()}
                for b in self._css(_SEL["message_bubble"]) if b.text.strip()
            ]
            conversations.append(Conversation(match_id=match_id, messages=messages))
        return conversations

    def send_message(self, match_id: str, text: str) -> None:
        driver = self._require_driver()
        driver.get(f"https://tinder.com/app/messages/{match_id}")
        inputs = self._css(_SEL["message_input"])
        if not inputs:
            raise RuntimeError("message input not found")
        inputs[0].send_keys(text)
        send = self._css(_SEL["send_button"])
        if send:
            send[0].click()


def _parse_age(text: str) -> int | None:
    match = re.search(r"\d{2}", text or "")
    return int(match.group()) if match else None
