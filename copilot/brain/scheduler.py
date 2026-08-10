"""Swipe scheduling + anti-ban logic and the shadowban monitor.

Pure, deterministic logic given an injected RNG and clock, so it is fully
testable. It decides *whether* and *how* to act; the driver only clicks.

Rules (from the spec):
  - daily cap (~50 to start), well under the ~100/day danger zone
  - sessions of 30-50 swipes over ~15-25 minutes, not one burst
  - randomized 2-5s delays, never a fixed interval
  - supervised active-hours only; never run overnight unattended
  - keep a human-like like/pass ratio
  - watch match rate; a sudden drop to ~zero is the shadowban signal -> pause
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime, time


@dataclass
class SessionPlan:
    swipe_count: int
    span_minutes: int


class SwipeScheduler:
    def __init__(self, config, rng: random.Random | None = None) -> None:
        self.config = config  # SchedulerConfig
        self._rng = rng or random.Random()
        self._swipes_today = 0
        self._likes_today = 0
        self._current_day: str | None = None

    # --- day bookkeeping -------------------------------------------------
    def _roll_day(self, now: datetime) -> None:
        key = now.date().isoformat()
        if key != self._current_day:
            self._current_day = key
            self._swipes_today = 0
            self._likes_today = 0

    # --- windows ---------------------------------------------------------
    def is_active_window(self, now: datetime) -> bool:
        start = time(self.config.active_start_hour, 0)
        end = time(self.config.active_end_hour, 0)
        return start <= now.time() < end

    # --- gating ----------------------------------------------------------
    def swipes_remaining_today(self, now: datetime) -> int:
        self._roll_day(now)
        return max(0, self.config.daily_cap - self._swipes_today)

    def like_ratio(self) -> float:
        if self._swipes_today == 0:
            return 0.0
        return self._likes_today / self._swipes_today

    def can_swipe_now(self, now: datetime) -> bool:
        self._roll_day(now)
        return (
            self.is_active_window(now)
            and self.swipes_remaining_today(now) > 0
        )

    def can_like(self, now: datetime) -> bool:
        """Whether a LIKE is allowed without breaching the like-ratio guard.

        Uses a projected ratio: allow the like if the ratio *after* it would stay
        under the cap. The first swipe of the day is always allowed so a session
        can get going.
        """
        self._roll_day(now)
        if self._swipes_today == 0:
            return True
        projected = (self._likes_today + 1) / (self._swipes_today + 1)
        return projected <= self.config.max_like_ratio

    # --- sampling --------------------------------------------------------
    def next_delay_seconds(self) -> float:
        return self._rng.uniform(
            self.config.delay_min_seconds, self.config.delay_max_seconds
        )

    def plan_session(self) -> SessionPlan:
        return SessionPlan(
            swipe_count=self._rng.randint(
                self.config.session_min_swipes, self.config.session_max_swipes
            ),
            span_minutes=self._rng.randint(
                self.config.session_span_min_minutes,
                self.config.session_span_max_minutes,
            ),
        )

    # --- recording -------------------------------------------------------
    def record_swipe(self, now: datetime, liked: bool) -> None:
        self._roll_day(now)
        self._swipes_today += 1
        if liked:
            self._likes_today += 1


class ShadowbanMonitor:
    """Tracks recent daily match rate and flags a sudden collapse.

    A match rate that drops to a small fraction of the trailing baseline is the
    signal to pause and back off.
    """

    def __init__(self, config) -> None:
        self.config = config  # SchedulerConfig
        # list of (day_iso, swipes, matches)
        self._history: list[tuple[str, int, int]] = []

    def record_day(self, day_iso: str, swipes: int, matches: int) -> None:
        self._history = [h for h in self._history if h[0] != day_iso]
        self._history.append((day_iso, swipes, matches))
        self._history.sort()

    @staticmethod
    def _rate(swipes: int, matches: int) -> float:
        return (matches / swipes) if swipes else 0.0

    def baseline_rate(self) -> float:
        window = self._history[:-1][-self.config.shadowban_window_days :]
        window = [h for h in window if h[1] > 0]
        if not window:
            return 0.0
        return sum(self._rate(s, m) for _, s, m in window) / len(window)

    def latest_rate(self) -> float:
        if not self._history:
            return 0.0
        _, swipes, matches = self._history[-1]
        return self._rate(swipes, matches)

    def is_shadowbanned(self) -> bool:
        """True if the latest day's match rate collapsed vs the baseline."""
        baseline = self.baseline_rate()
        if baseline <= 0.0:
            return False  # not enough history to judge
        latest_day = self._history[-1]
        if latest_day[1] == 0:
            return False  # no swipes that day, nothing to conclude
        return self.latest_rate() < baseline * self.config.shadowban_drop_ratio
