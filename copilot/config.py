"""Runtime configuration.

Defaults are tuned to the spec (age 26-33, ~50 swipes/day, 2-5s delays). Values
can be overridden by environment variables or, at runtime, by the `settings`
table so the owner can tune thresholds from the dashboard without a redeploy.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace


@dataclass(frozen=True)
class MatchConfig:
    # Score >= high -> auto-like; score < low -> auto-pass; between -> review queue.
    high_threshold: float = 0.70
    low_threshold: float = 0.40
    # Extra weight on the primary (first) photo when aggregating a profile.
    primary_photo_weight: float = 2.0
    # Minimum labelled examples of each class before the classifier is trusted.
    # Below this, everything goes to the review queue.
    min_labels_per_class: int = 10


@dataclass(frozen=True)
class StructuredConfig:
    # Cheap hard-pass pre-filters, applied before the photo model runs.
    min_age: int = 26
    max_age: int = 33
    # Fields that, if present and truthy, are an automatic pass.
    disqualifying_fields: tuple[str, ...] = ("has_children",)


@dataclass(frozen=True)
class SchedulerConfig:
    daily_cap: int = 50
    session_min_swipes: int = 30
    session_max_swipes: int = 50
    session_span_min_minutes: int = 15
    session_span_max_minutes: int = 25
    delay_min_seconds: float = 2.0
    delay_max_seconds: float = 5.0
    # Human-like like/pass ratio guard. Don't let the running like-rate exceed this.
    max_like_ratio: float = 0.6
    # Supervised active-hours window, local 24h clock. Never run overnight.
    active_start_hour: int = 9
    active_end_hour: int = 23
    # Shadowban monitor: if match rate over the recent window drops below this
    # fraction of the baseline, raise an alert and pause.
    shadowban_drop_ratio: float = 0.25
    shadowban_window_days: int = 3


@dataclass(frozen=True)
class RedFlagConfig:
    # Owner-editable. Bio substrings (case-insensitive) that hard-pass.
    keyword_phrases: tuple[str, ...] = ()
    # Example phrases for embedding-similarity matching against bio sentences.
    example_phrases: tuple[str, ...] = ()
    bio_similarity_threshold: float = 0.75
    # Zero-shot image prompts; similarity above threshold hard-passes.
    image_prompts: tuple[str, ...] = (
        "dark, gory, or creepy imagery",
    )
    image_similarity_threshold: float = 0.28


@dataclass(frozen=True)
class Config:
    embedder: str = "hash"          # "hash" (dev) or "clip"
    clip_model: str = "ViT-B-32"
    clip_pretrained: str = "laion2b_s34b_b79k"
    drafter: str = "stub"           # "stub" or "anthropic"
    draft_model: str = "claude-opus-5"
    db_path: str = "copilot_dev.db"
    pg_dsn: str | None = None

    match: MatchConfig = field(default_factory=MatchConfig)
    structured: StructuredConfig = field(default_factory=StructuredConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    redflags: RedFlagConfig = field(default_factory=RedFlagConfig)


def _int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return int(raw) if raw else default


def from_env() -> Config:
    """Build a Config from environment variables, falling back to defaults."""
    sched = SchedulerConfig(
        daily_cap=_int("COPILOT_DAILY_CAP", 50),
        session_min_swipes=_int("COPILOT_SESSION_MIN", 30),
        session_max_swipes=_int("COPILOT_SESSION_MAX", 50),
    )
    return Config(
        embedder=os.environ.get("COPILOT_EMBEDDER", "hash"),
        clip_model=os.environ.get("COPILOT_CLIP_MODEL", "ViT-B-32"),
        clip_pretrained=os.environ.get("COPILOT_CLIP_PRETRAINED", "laion2b_s34b_b79k"),
        drafter=os.environ.get("COPILOT_DRAFTER", "stub"),
        draft_model=os.environ.get("COPILOT_DRAFT_MODEL", "claude-opus-5"),
        db_path=os.environ.get("COPILOT_DB_PATH") or "copilot_dev.db",
        pg_dsn=os.environ.get("COPILOT_PG_DSN"),
        scheduler=sched,
    )


def with_overrides(config: Config, **kwargs) -> Config:
    """Return a copy of `config` with top-level fields replaced."""
    return replace(config, **kwargs)
