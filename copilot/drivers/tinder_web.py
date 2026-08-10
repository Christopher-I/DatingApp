"""Tinder web driver — Selenium + undetected-chromedriver skeleton.

This drives the owner's real, logged-in Chrome session (preferred over a
reverse-engineered private API: real-browser traffic survives bot detection far
better). It is a documented skeleton: the class shape, selectors, and flow are
here, but the live DOM selectors WILL drift and must be verified against the real
site, and it is only exercised against the owner's own account and his own risk.

Nothing here runs in CI — Selenium and undetected-chromedriver are optional deps,
imported lazily. Use MockDriver for offline development.

Anti-ban behaviour (caps, delays, active-hours, sessions) lives in the scheduler,
not here — this driver only knows how to read the DOM and click.
"""

from __future__ import annotations

from .base import Conversation, Direction, Driver, Profile

# Selectors are centralized so they're easy to re-verify when the site changes.
# These are placeholders and must be confirmed against the live DOM.
_SEL = {
    "card": 'div[data-testid="cardStack"] div[aria-label]',
    "photo": 'div[aria-label] div[style*="background-image"]',
    "age": 'span[itemprop="age"]',
    "bio": 'div.BreakWord',
    "like_button": 'button[aria-label="Like"]',
    "pass_button": 'button[aria-label="Nope"]',
    "superlike_button": 'button[aria-label="Super Like"]',
    "messages_tab": 'a[href="/app/messages"]',
}


class TinderWebDriver(Driver):
    name = "tinder"

    def __init__(self, chrome_profile_dir: str | None = None,
                 headless: bool = False) -> None:
        self._chrome_profile_dir = chrome_profile_dir
        self._headless = headless
        self._driver = None  # the Selenium WebDriver, created in start()

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
        self._driver.get("https://tinder.com/app/recs")

    def stop(self) -> None:
        if self._driver is not None:
            self._driver.quit()
            self._driver = None

    def _require_driver(self):
        if self._driver is None:
            raise RuntimeError("call start() before using the driver")
        return self._driver

    def get_next_profile(self) -> Profile | None:  # pragma: no cover - needs browser
        driver = self._require_driver()
        # Real implementation:
        #   1. locate the top card element (_SEL["card"])
        #   2. read its data-testid/aria attributes for an opaque ref
        #   3. click through each photo, screenshot the photo region, and read the
        #      raw image bytes from the DOM/screen (never save to disk)
        #   4. read age/bio/fields where exposed
        # Returning None here so the skeleton is safe to instantiate.
        raise NotImplementedError(
            "TinderWebDriver.get_next_profile: implement DOM scraping against the "
            "live site. See _SEL for the selectors to verify."
        )

    def swipe(self, direction: Direction) -> Direction:  # pragma: no cover - needs browser
        driver = self._require_driver()
        button = {
            Direction.LIKE: _SEL["like_button"],
            Direction.PASS: _SEL["pass_button"],
            Direction.SUPERLIKE: _SEL["superlike_button"],
        }[direction]
        # driver.find_element(By.CSS_SELECTOR, button).click()
        raise NotImplementedError(
            f"TinderWebDriver.swipe: click {button} on the live card."
        )

    def open_conversations(self) -> list[Conversation]:  # pragma: no cover - needs browser
        raise NotImplementedError(
            "TinderWebDriver.open_conversations: navigate to messages and read threads."
        )

    def send_message(self, match_id: str, text: str) -> None:  # pragma: no cover - needs browser
        raise NotImplementedError(
            "TinderWebDriver.send_message: open the thread and type the approved text."
        )
