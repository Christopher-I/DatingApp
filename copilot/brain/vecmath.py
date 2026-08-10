"""Tiny pure-Python vector helpers.

Embeddings are plain ``list[float]`` so the core runs with no numpy/torch. If a
real CLIP embedder is installed it returns numpy arrays; these helpers accept any
sequence of floats, so both paths work.
"""

from __future__ import annotations

import math
from typing import Sequence

Vector = Sequence[float]


def dot(a: Vector, b: Vector) -> float:
    return sum(x * y for x, y in zip(a, b))


def norm(a: Vector) -> float:
    return math.sqrt(dot(a, a))


def normalize(a: Vector) -> list[float]:
    n = norm(a)
    if n == 0.0:
        return [0.0] * len(a)
    return [x / n for x in a]


def cosine(a: Vector, b: Vector) -> float:
    """Cosine similarity in [-1, 1]. Returns 0 for a zero vector."""
    na, nb = norm(a), norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot(a, b) / (na * nb)


def mean(vectors: Sequence[Vector]) -> list[float]:
    if not vectors:
        raise ValueError("mean() of no vectors")
    dim = len(vectors[0])
    acc = [0.0] * dim
    for v in vectors:
        for i in range(dim):
            acc[i] += v[i]
    return [x / len(vectors) for x in acc]


def sub(a: Vector, b: Vector) -> list[float]:
    return [x - y for x, y in zip(a, b)]


def logistic(x: float) -> float:
    # Numerically stable sigmoid.
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)
