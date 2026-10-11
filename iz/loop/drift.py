"""Drift detection on takeover-free mission rate (TFMR): one-sided two-proportion z-test."""
from __future__ import annotations

from math import erfc, sqrt

from iz.world.stats import wilson

MIN_N = 10


def two_proportion_z(s1: int, n1: int, s2: int, n2: int) -> tuple[float, float]:
    """s1/n1 = window, s2/n2 = baseline. One-sided H1: window rate < baseline rate."""
    if n1 <= 0 or n2 <= 0:
        return 0.0, 1.0
    p1, p2 = s1 / n1, s2 / n2
    pool = (s1 + s2) / (n1 + n2)
    se = sqrt(pool * (1 - pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return 0.0, 1.0
    z = (p1 - p2) / se
    return z, 0.5 * erfc(-z / sqrt(2))  # P(Z <= z)


def _series(history: list, site: str | None) -> list[int]:
    out: list[int] = []
    for h in history:
        if isinstance(h, dict):
            if site is not None and h.get("site") not in (None, site):
                continue
            v = h.get("takeover_free", h.get("success"))
        else:
            v = h
        out.append(1 if v else 0)
    return out


def _test(base: list[int], win: list[int], alpha: float) -> dict:
    sb, sw = sum(base), sum(win)
    z, p = two_proportion_z(sw, len(win), sb, len(base))
    return {"alarm": bool(len(win) >= MIN_N and len(base) >= MIN_N and p < alpha),
            "window_tfmr": sw / len(win) if win else 0.0,
            "baseline_tfmr": sb / len(base) if base else 0.0,
            "p_value": p, "z": z, "n_window": len(win), "n_baseline": len(base)}


def _raindrop_rate() -> float | None:
    try:
        from iz.obs.rd_query import signal_timeseries
        r = signal_timeseries("takeover")
        if r.get("degraded") or not r.get("series"):
            return None
        return float(sum(x["count"] for x in r["series"][-6:]))
    except Exception:  # noqa: BLE001
        return None


def check(history: list, site: str | None = None, baseline_window: int = 60, window: int = 30,
          alpha: float = 0.01) -> dict | None:
    """history: list of bool/int (takeover-free) or dicts {site, takeover_free}, oldest first."""
    s = _series(history, site)
    if len(s) < MIN_N * 2:
        return None
    win = s[-window:]
    base = s[-(window + baseline_window):-window] or s[:-len(win)]
    if len(base) < MIN_N:
        return None
    res = _test(base, win, alpha)
    res["site"] = site
    rate = _raindrop_rate()
    if rate is not None:
        res["raindrop_rate"] = rate
    return res


def detection_latency(history: list, change_index: int, baseline_window: int = 60, window: int = 30,
                      alpha: float = 0.01, site: str | None = None) -> int | None:
    """Missions after change_index until the alarm fires (None if never)."""
    s = _series(history, site)
    base = s[max(0, change_index - baseline_window):change_index]
    if len(base) < MIN_N:
        return None
    for t in range(change_index + MIN_N, len(s) + 1):
        win = s[max(change_index, t - window):t]
        if _test(base, win, alpha)["alarm"]:
            return t - change_index
    return None


def recovery_missions(history: list, change_index: int, baseline_window: int = 60, window: int = 30,
                      site: str | None = None) -> int | None:
    """Missions after change_index until window TFMR is back inside the baseline 95% Wilson CI
    (after having left it)."""
    s = _series(history, site)
    base = s[max(0, change_index - baseline_window):change_index]
    if len(base) < MIN_N:
        return None
    lo, _hi = wilson(sum(base), len(base))
    left = False
    for t in range(change_index + MIN_N, len(s) + 1):
        win = s[max(change_index, t - window):t]
        r = sum(win) / len(win)
        if r < lo:
            left = True
        elif left:
            return t - change_index
    return None


def rolling_series(run_id: str, site: str | None = None, window: int = 20,
                   rows: list | None = None) -> list[dict]:
    """Rolling TFMR via $setWindowFields when Atlas is reachable (and no rows given), else Python."""
    if rows is None:
        try:
            from iz.store.db import mongo_available
            if mongo_available():
                from iz.store.metrics import rolling_tfmr
                return rolling_tfmr(run_id, site, window)
        except Exception:  # noqa: BLE001
            pass
    s = _series(rows or [], site)
    out = []
    for i in range(len(s)):
        w = s[max(0, i - window + 1):i + 1]
        out.append({"tfmr": sum(w) / len(w)})
    return out
