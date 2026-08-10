"""The driver interface every app implements.

    get_next_profile() -> Profile | None        # transient; photos are bytes
    swipe(direction)   -> Direction
    open_conversations() -> list[Conversation]
    send_message(match_id, text)

Photos live only inside a Profile object in memory. The brain embeds them and the
caller drops the Profile — nothing is persisted.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from enum import Enum


class Direction(str, Enum):
    LIKE = "like"
    PASS = "pass"
    SUPERLIKE = "superlike"


@dataclass
class Profile:
    """A single profile as scraped from the app, held in memory only."""

    external_ref: str                     # opaque, app-specific id (never a name)
    photos: list[bytes] = field(default_factory=list)  # raw image bytes, transient
    age: int | None = None
    bio_text: str = ""
    fields: dict = field(default_factory=dict)  # structured fields the app exposes
    handles: dict = field(default_factory=dict)  # e.g. instagram, if surfaced
    primary_index: int = 0                # which photo is the primary/first one


@dataclass
class Conversation:
    match_id: str
    messages: list[dict] = field(default_factory=list)  # {"from": "me"|"them", "text": str}
    last_activity: str | None = None      # ISO timestamp if known


class Driver(abc.ABC):
    """Base class for a per-app automation driver."""

    #: short app name, e.g. "tinder"
    name: str = "base"

    @abc.abstractmethod
    def get_next_profile(self) -> Profile | None:
        """Return the next profile to evaluate, or None if the deck is empty."""

    @abc.abstractmethod
    def swipe(self, direction: Direction) -> Direction:
        """Apply a swipe to the current profile. Returns the direction applied."""

    @abc.abstractmethod
    def open_conversations(self) -> list[Conversation]:
        """Return current matches and their message threads."""

    @abc.abstractmethod
    def send_message(self, match_id: str, text: str) -> None:
        """Send `text` to the match. Owner-approved only in this design."""

    # Optional lifecycle hooks; drivers may override.
    def start(self) -> None:
        """Open the session (browser/emulator). No-op by default."""

    def stop(self) -> None:
        """Tear down the session. No-op by default."""
