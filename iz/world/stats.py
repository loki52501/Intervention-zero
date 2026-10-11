"""Small statistics helpers: Wilson interval and takeover-free mission rate."""
from __future__ import annotations

from math import sqrt

from iz.types import Outcome


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def tfmr(mission_outcomes: list[list[Outcome]]) -> float:
    if not mission_outcomes:
        return 0.0
    free = sum(1 for outs in mission_outcomes if all(o.takeover_free for o in outs))
    return free / len(mission_outcomes)
