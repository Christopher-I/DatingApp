"""Dating Copilot — a personal, app-agnostic swipe/score/draft engine.

Layers:
  drivers/  one implementation per app (Tinder web, later Hinge/Bumble)
  brain/    matching engine, red-flag filter, scheduler, message drafter
  data/     embeddings + stats store (SQLite for dev, Postgres/pgvector for prod)

Photos are read, scored in memory, and discarded. Only embeddings are stored.
"""

__version__ = "0.1.0"
