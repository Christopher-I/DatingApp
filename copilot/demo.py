"""End-to-end offline demo. Run with:  python -m copilot.demo

Uses the MockDriver + HashEmbedder so it needs no live account and no ML deps.
It:
  1. runs a calibration pass (owner labels profiles, model learns)
  2. runs an autopilot pass through the engine (structured filter -> red flags ->
     score -> action), respecting the scheduler's caps and active-hours
  3. drafts a message for one match

Note: HashEmbedder produces arbitrary (not semantic) vectors, so the *scores*
here are meaningless — the demo proves the plumbing runs end to end, not that the
model is accurate. Install open_clip and set COPILOT_EMBEDDER=clip for real
scoring against real photos via a live driver.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime

from .brain.matcher import Action, Label
from .config import from_env
from .drivers.base import Direction
from .drivers.mock import MockDriver
from .factory import build

_LIKE_WORDS = ("coffee", "hiking", "music", "photograph")


def _fake_owner_label(profile) -> Label:
    """Stand-in for the owner's real taste during calibration. Deterministic."""
    text = profile.bio_text.lower()
    return Label.LIKE if any(w in text for w in _LIKE_WORDS) else Label.PASS


def run() -> None:
    config = from_env()
    # Lower the calibration bar so this short demo can reach a "ready" model.
    config = replace(config, match=replace(config.match, min_labels_per_class=5))
    copilot = build(config)

    # --- 1. calibration -------------------------------------------------
    print("== calibration ==")
    cal_driver = MockDriver(count=60, seed=1)
    while (profile := cal_driver.get_next_profile()) is not None:
        label = _fake_owner_label(profile)
        copilot.learning.add_profile(profile, label)
        cal_driver.swipe(Direction.LIKE if label == Label.LIKE else Direction.PASS)
    ready = copilot.learning.retrain()
    counts = copilot.learning.counts()
    print(f"labelled {len(copilot.learning)} photos "
          f"(like={counts[Label.LIKE]}, pass={counts[Label.PASS]}); "
          f"classifier ready={ready}")

    # --- 2. autopilot pass ----------------------------------------------
    print("\n== autopilot ==")
    now = datetime(2026, 8, 10, 14, 0, 0)  # inside the active window
    driver = MockDriver(count=15, seed=2)
    tallies = {Action.LIKE: 0, Action.REVIEW: 0, Action.PASS: 0}
    while (profile := driver.get_next_profile()) is not None:
        if not copilot.scheduler.can_swipe_now(now):
            print("scheduler: daily cap reached or outside active hours; stopping")
            break
        decision = copilot.engine.evaluate(profile)
        tallies[decision.action] += 1
        agg = f"{decision.score.aggregate:.2f}" if decision.score else "  - "
        print(f"{profile.external_ref}  age={profile.age:>2}  "
              f"{decision.action.value:<6} score={agg}  {decision.reason}")
        if decision.action == Action.LIKE and copilot.scheduler.can_like(now):
            copilot.scheduler.record_swipe(now, liked=True)
            driver.swipe(Direction.LIKE)
        elif decision.action == Action.PASS:
            copilot.scheduler.record_swipe(now, liked=False)
            driver.swipe(Direction.PASS)
        # REVIEW items are held for the owner's queue, not swiped.
    print(f"\nactions: {{ {', '.join(f'{k.value}={v}' for k, v in tallies.items())} }}")
    print(f"like ratio this session: {copilot.scheduler.like_ratio():.2f}")

    # --- 3. draft a message ---------------------------------------------
    print("\n== drafter ==")
    copilot.drafter.style_corpus = [
        "haha yeah that place is underrated",
        "what's the best coffee you've had recently?",
    ]
    convos = driver.open_conversations()
    if convos:
        reply = copilot.drafter.draft_reply(convos[0])
        print(f"reply to {convos[0].match_id}: {reply}")
    else:
        print("no matches this run; sample cold opener:")
        sample = MockDriver(count=1, seed=9).get_next_profile()
        print(copilot.drafter.draft_opener(sample))


if __name__ == "__main__":
    run()
