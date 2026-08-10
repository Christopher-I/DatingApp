"""Persistence: embeddings + stats. SQLite for dev, Postgres/pgvector for prod.

No raw photos are ever stored — only embedding vectors and metadata.
"""

from .store import SQLiteStore

__all__ = ["SQLiteStore"]
