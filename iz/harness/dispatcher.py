"""Greedy mission->robot assignment."""
from __future__ import annotations

import math

from pydantic import BaseModel, Field

from iz.types import DispatcherWeights

BATTERY_FLOOR = 15.0


class RobotState(BaseModel):
    id: str
    battery_pct: float
    xy: tuple[float, float]
    capabilities: list[str] = Field(default_factory=list)
    load: int = 0


def assign(missions: list[dict], robots: list[RobotState], weights: DispatcherWeights,
           hour: int | None = None, peak_hours: list[int] | None = None) -> dict[str, str]:
    loads = {r.id: r.load for r in robots}
    elig = sorted((r for r in robots if r.battery_pct >= BATTERY_FLOOR), key=lambda r: r.id)
    peak = hour is not None and hour in (peak_hours or [])
    out: dict[str, str] = {}
    for m in missions:
        mx, my = m.get("xy", (0.0, 0.0))
        best, best_c = None, None
        for r in elig:
            c = (weights.distance * math.hypot(r.xy[0] - mx, r.xy[1] - my)
                 + weights.battery * (100 - r.battery_pct) / 10
                 + weights.capability * (0 if (not r.capabilities or m.get("kind") in r.capabilities) else 10)
                 + weights.congestion * float(m.get("zone_congestion", 0.0))
                 + weights.peak_avoid * (5 if peak else 0)
                 + 2 * loads[r.id])
            if best_c is None or c < best_c:
                best, best_c = r, c
        if best is not None:
            out[m["id"]] = best.id
            loads[best.id] += 1
    return out
