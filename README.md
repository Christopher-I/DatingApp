# Dating Copilot

A personal tool to cut time spent on Tinder (first), then Hinge and Bumble. It
learns the owner's visual type from his own swipe history, scores new profiles
from their photos, automates swiping within safe limits, and drafts messages in
his voice for one-tap approval.

Built Tinder-first, with a driver abstraction so Hinge and Bumble slot in later
behind the same core.

> **Owner's account, owner's risk.** Automating these apps violates their ToS,
> and the anti-ban layer is detection-evasion against their bot systems. The
> pipeline also processes photos and messages of third parties — by design those
> photos are embedded in memory and discarded, and **only embedding vectors are
> ever stored**, never raw images.

## Architecture

```
Dashboard (local web UI)      review queue, inbox, toggles, stats   [planned]
─────────────────────────────────────────────────────────────────
Brain (app-agnostic)          matcher · red-flag filter · drafter · scheduler
─────────────────────────────────────────────────────────────────
Driver interface              Tinder (Chrome web) · Hinge/Bumble (later)
─────────────────────────────────────────────────────────────────
Data                          embeddings + stats (SQLite dev / Postgres+pgvector)
```

The matching engine follows the spec's few-shot approach: embed the **whole
photo** with a frozen CLIP model, then a lightweight classifier (centroid
baseline, logistic-regression when scikit-learn is present) over those
embeddings. No CNN trained from scratch.

## Quickstart (offline, no ML deps)

The core runs on the Python standard library alone. `open_clip`, `torch`,
`scikit-learn`, `selenium`, and `anthropic` are all optional and lazily imported.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install pytest          # only needed to run the tests
python -m pytest -q         # 23 tests, all offline
python -m copilot.demo      # end-to-end run on the MockDriver + HashEmbedder
```

The demo calibrates a model, runs an autopilot pass through the engine
(structured filter → red-flag override → score → action) under the scheduler's
caps and active-hours, and drafts a message. With the `hash` embedder the scores
are not semantic — it proves the plumbing, not accuracy. For real scoring, set
`COPILOT_EMBEDDER=clip` (needs `open_clip-torch` + `torch`) and drive real photos
through a live driver.

## Configuration

Copy `.env.example` to `.env`. Everything has a safe default; nothing is required
for the demo or tests. Key switches:

- `COPILOT_EMBEDDER` — `hash` (dev) or `clip` (real embeddings)
- `COPILOT_DRAFTER` — `stub` (offline) or `anthropic` (real drafts, defaults to
  `claude-opus-5`)
- `COPILOT_DB_PATH` / `COPILOT_PG_DSN` — SQLite dev file or Postgres DSN
- daily cap / session size / age range / thresholds — see `copilot/config.py`

## Module map

| Path | What it is | Status |
|------|-----------|--------|
| `copilot/config.py` | Tunable thresholds, caps, windows, red-flag lists | done |
| `copilot/drivers/base.py` | The `Driver` interface + `Profile`/`Conversation` | done |
| `copilot/drivers/mock.py` | Deterministic offline driver for dev/tests | done |
| `copilot/drivers/tinder_web.py` | Selenium + undetected-chromedriver | **skeleton** — DOM selectors need live verification |
| `copilot/brain/embeddings.py` | `HashEmbedder` (dev) + `ClipEmbedder` (real) | done |
| `copilot/brain/matcher.py` | Classifier + profile aggregation + thresholds | done |
| `copilot/brain/redflags.py` | Bio keyword/similarity + image-vibe hard-pass | done |
| `copilot/brain/scheduler.py` | Caps, delays, active-hours, ratio, shadowban | done |
| `copilot/brain/engine.py` | Structured → red-flag → score orchestration + learning | done |
| `copilot/brain/drafter.py` | Voice-matched opener/reply drafts | done (stub + Anthropic) |
| `copilot/data/store.py` | SQLite store (vectors as JSON) | done |
| `copilot/data/schema.sql` | Supabase / Postgres + pgvector schema | done |
| `copilot/factory.py` | Assemble the brain from a `Config` | done |
| Dashboard (FastAPI/React) | review queue, inbox, stats | not started |

## Build order (from the spec)

1. Driver interface + Tinder web automation — interface + mock **done**, Tinder **skeleton**
2. Calibration mode + matching engine + red-flag filter — **done**
3. Scheduler (human-mimicry, caps, shadowban monitor) — **done**
4. Style corpus + message drafter + approval queue — drafter **done**, queue UI pending
5. Dashboard — **not started**
6. Phase two: Hinge & Bumble drivers (emulator + Appium)

## Estimate

- Lean (functional single-user): ~70–85h
- Polished (spec-complete, hardened automation, dashboard): ~95–125h
- Phase two per additional app (emulator + tuning): ~25–40h each

## Next steps

- Wire the SQLite/pg store into `LearningStore` so calibration persists and
  continuous learning survives restarts.
- Implement the Tinder DOM scraping/clicking against a live logged-in session.
- Build the approval-queue + review-queue dashboard (FastAPI + light React).
