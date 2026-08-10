"""A deterministic in-memory driver for offline development, demos, and tests.

It fabricates profiles from a seeded generator (deterministic "photo" bytes, ages,
bios, fields) so the whole pipeline can run end to end with no live account. It
also records swipes and lets you inspect them.
"""

from __future__ import annotations

import hashlib
import random

from .base import Conversation, Direction, Driver, Profile

_BIO_SNIPPETS = [
    "Love hiking and trying new coffee spots.",
    "Marathon runner, dog person, big into live music.",
    "Just here for the plot. Ask me about my sourdough.",
    "Anxious attachment but working on it in therapy every week.",
    "Traveling the world one city at a time.",
    "Foodie, gym rat, and amateur photographer.",
]


class MockDriver(Driver):
    name = "mock"

    def __init__(self, count: int = 20, seed: int = 0,
                 photos_per_profile: int = 3) -> None:
        self._rng = random.Random(seed)
        self._count = count
        self._photos_per_profile = photos_per_profile
        self._served = 0
        self._current: Profile | None = None
        self.swipes: list[tuple[str, Direction]] = []  # (external_ref, direction)
        self._conversations: list[Conversation] = []

    def _fake_photo(self, ref: str, index: int) -> bytes:
        # Deterministic pseudo-image bytes. Real drivers return decoded image bytes;
        # HashEmbedder only needs stable bytes, and ClipEmbedder would need real
        # images (use a live driver for that).
        return hashlib.sha256(f"{ref}:{index}".encode()).digest()

    def get_next_profile(self) -> Profile | None:
        if self._served >= self._count:
            self._current = None
            return None
        ref = f"mock-{self._served:04d}"
        photos = [self._fake_photo(ref, i) for i in range(self._photos_per_profile)]
        profile = Profile(
            external_ref=ref,
            photos=photos,
            age=self._rng.randint(21, 38),
            bio_text=self._rng.choice(_BIO_SNIPPETS),
            fields={"has_children": self._rng.random() < 0.15},
        )
        self._current = profile
        self._served += 1
        return profile

    def swipe(self, direction: Direction) -> Direction:
        if self._current is None:
            raise RuntimeError("swipe() called with no current profile")
        self.swipes.append((self._current.external_ref, direction))
        # A liked profile occasionally becomes a match with an opening line.
        if direction in (Direction.LIKE, Direction.SUPERLIKE) and self._rng.random() < 0.3:
            self._conversations.append(
                Conversation(
                    match_id=f"match-{self._current.external_ref}",
                    messages=[{"from": "them", "text": "hey! how's your week going?"}],
                )
            )
        return direction

    def open_conversations(self) -> list[Conversation]:
        return list(self._conversations)

    def send_message(self, match_id: str, text: str) -> None:
        for convo in self._conversations:
            if convo.match_id == match_id:
                convo.messages.append({"from": "me", "text": text})
                return
        raise KeyError(f"unknown match_id: {match_id}")
