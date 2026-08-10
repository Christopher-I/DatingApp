"""Assemble the brain from a Config in one place."""

from __future__ import annotations

from .brain.drafter import MessageDrafter, get_llm
from .brain.embeddings import get_embedder
from .brain.engine import Engine, LearningStore
from .brain.matcher import LogisticClassifier, Matcher
from .brain.redflags import RedFlagFilter
from .brain.scheduler import ShadowbanMonitor, SwipeScheduler
from .config import Config, from_env


class Copilot:
    """Bundle of the assembled components for a given config."""

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or from_env()
        self.embedder = get_embedder(self.config)
        self.matcher = Matcher(self.embedder, LogisticClassifier(), self.config.match)
        self.redflags = RedFlagFilter(self.embedder, self.config.redflags)
        self.engine = Engine(self.embedder, self.matcher, self.redflags, self.config)
        self.learning = LearningStore(self.matcher)
        self.scheduler = SwipeScheduler(self.config.scheduler)
        self.shadowban = ShadowbanMonitor(self.config.scheduler)
        self.drafter = MessageDrafter(get_llm(self.config))


def build(config: Config | None = None) -> Copilot:
    return Copilot(config)
