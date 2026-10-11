import os

os.environ["RAINDROP_ENABLED"] = "0"

import importlib.util
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
if not (ROOT / "iz" / "fleet" / "coordinator.py").exists():
    pytest.skip("iz/fleet/coordinator.py missing", allow_module_level=True)

from iz.shop.controller import DEFAULT_PARAMS
from iz.shop.episode import _spill_list, run_episode
from iz.shop.scene import build_shop

# Found by _find_seed(): first seed in 0..60 with a spill in robot 0's aisle, -4<x<3.
SEED = 0


def _find_seed():
    for s in range(61):
        _, m = build_shop(s)
        ay = m["aisle_ys"][m["target_aisle"]]
        for sx, sy, _r in _spill_list(m):
            if abs(sy - ay) < 0.5 and -4.0 < sx < 3.0:
                return s
    return None


def _count(events, typ):
    return [e for e in events if e.get("type") == typ]


def test_seed_is_valid():
    assert _find_seed() == SEED


def test_two_robots_hazard_shared_then_reroute():
    r = run_episode(DEFAULT_PARAMS, SEED, max_time_s=40, robots=2)
    ev = r["coordination_events"]
    hz = _count(ev, "hazard_shared")
    assert hz and hz[0]["robot_id"] == "robot"
    t0 = hz[0]["t"]
    rr = [e for e in _count(ev, "reroute") if e["robot_id"] == "robot_1" and e["t"] >= t0]
    assert rr and rr[0]["dist_to_hazard_m"] > 1.0
    assert len(_count(ev, "task_claimed")) == 1
    assert r["failure"] != "too_close_to_robot"
    assert r["fleet"]["safety_violations"] == 0


def test_robots1_unchanged():
    a = run_episode(DEFAULT_PARAMS, 0)
    b = run_episode(DEFAULT_PARAMS, 0, robots=1)
    assert a == b
    assert "robot_results" not in a and "fleet" not in a


def test_three_robots_compile_and_claims():
    r = run_episode(DEFAULT_PARAMS, SEED, max_time_s=10, robots=3)
    assert len(r["robot_results"]) == 3
    claims = _count(r["coordination_events"], "task_claimed")
    assert len(claims) == 2
    assert len({c.get("aisle") for c in claims}) == 2


def test_no_emit_by_default(monkeypatch):
    from iz.obs import rd

    calls = []

    class _H:
        def finish(self, *a, **k):
            pass

    def fake(*a, **k):
        calls.append(a)
        return _H()

    monkeypatch.setattr(rd, "_begin", fake)
    run_episode(DEFAULT_PARAMS, SEED, max_time_s=5, robots=2)
    assert len(calls) == 0
    run_episode(DEFAULT_PARAMS, SEED, max_time_s=5, robots=2, emit_rd=True)
    assert len(calls) == 1
    assert calls[0][1] == "fleet_episode"
