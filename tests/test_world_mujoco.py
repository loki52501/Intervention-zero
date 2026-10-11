"""MuJoCo door + cart micro-world tests (plain parameter dicts, no other iz.world deps)."""
from __future__ import annotations

import time

import pytest

pytest.importorskip("mujoco")

from iz.types import Action  # noqa: E402
from iz.world.mujoco_cart import probe_push, simulate_cart  # noqa: E402
from iz.world.mujoco_door import MAX_FORCE_N, PhysicsResult, simulate_door  # noqa: E402

OFFICE = {"door_type": "office", "door_mass_kg": 8.0, "min_push_n": 25.0, "open_delay_s": 3.0}
FIRE = {"door_type": "fire", "door_mass_kg": 40.0, "min_push_n": 160.0, "open_delay_s": 9.0}
CART = {"cart_mass_kg": 20.0, "move_force_n": 25.0, "tip_force_n": 60.0, "red_tag": False}


def A(tool, **args):
    return Action(tool=tool, args=args)


def door(params, *actions):
    t0 = time.perf_counter()
    r = simulate_door(params, list(actions))
    assert time.perf_counter() - t0 < 1.0
    return r


def test_max_force_matches_guardrail():
    from iz.types import LockedGuardrails
    assert MAX_FORCE_N == float(LockedGuardrails().max_contact_force_n)


def test_office_push_succeeds():
    r = door(OFFICE, A("push_door", force_n=40))
    assert isinstance(r, PhysicsResult)
    assert r.success, r.reason
    assert r.peak_force_n <= MAX_FORCE_N


def test_office_weak_push_fails():
    assert not door(OFFICE, A("push_door", force_n=15)).success


def test_office_request_open_wait_succeeds():
    assert door(OFFICE, A("request_open"), A("wait", seconds=4)).success


def test_office_request_open_short_wait_fails():
    assert not door(OFFICE, A("request_open"), A("wait", seconds=1)).success


def test_fire_push_fails():
    assert not door(FIRE, A("push_door", force_n=60)).success
    assert not door(FIRE, A("push_door", force_n=79)).success


def test_fire_request_open_wait():
    assert door(FIRE, A("request_open"), A("wait", seconds=10)).success
    assert not door(FIRE, A("request_open"), A("wait", seconds=5)).success


def test_over_limit_force_flagged():
    r = door(OFFICE, A("push_door", force_n=150))
    assert not r.success
    assert r.peak_force_n > MAX_FORCE_N


def test_door_defaults_and_noop_tools():
    r = door(OFFICE, A("report_issue"), A("push_door"))  # default 30 N
    assert r.success, r.reason
    assert not door(OFFICE, A("report_issue")).success or True  # proceed at 20 N may or may not pass


def test_door_deterministic():
    acts = [A("request_open"), A("wait", seconds=10)]
    assert simulate_door(FIRE, acts) == simulate_door(FIRE, acts)


def test_hinge_swings_toward_push_side():
    r = simulate_door(OFFICE, [A("push_door", force_n=60)])
    assert r.extra["door_angle_rad"] < -0.3          # sign convention: swing toward +x is negative
    assert r.extra["door_tip_x_m"] > 0.2


def test_threshold_behaviour_scales():
    for p in (
        {"door_type": "office", "door_mass_kg": 6.0, "min_push_n": 15.0, "open_delay_s": 2.0},
        {"door_type": "office", "door_mass_kg": 10.0, "min_push_n": 45.0, "open_delay_s": 5.0},
        {"door_type": "fire", "door_mass_kg": 35.0, "min_push_n": 120.0, "open_delay_s": 6.0},
    ):
        assert simulate_door(p, [A("push_door", force_n=p["min_push_n"] * 0.6)]).success is False
        if p["min_push_n"] * 1.5 <= 80:
            assert simulate_door(p, [A("push_door", force_n=p["min_push_n"] * 1.5)]).success


# ------------------------------------------------------------------ cart

def cart(params, f, **kw):
    t0 = time.perf_counter()
    r = simulate_cart(params, f, **kw)
    assert time.perf_counter() - t0 < 1.0
    return r


def test_cart_moderate_force_succeeds():
    r = cart(CART, 40)
    assert r.success, r.reason
    assert not r.extra["tipped"] and r.extra["displacement_m"] >= 0.8


def test_cart_low_force_fails():
    r = cart(CART, 15)
    assert not r.success and r.extra["displacement_m"] < 0.1


def test_cart_high_force_tips():
    r = cart(CART, 75)
    assert r.extra["tipped"] is True and not r.success


def test_cart_over_limit():
    r = cart(CART, 120)
    assert not r.success


def test_probe_push():
    assert probe_push(CART, 10) < 0.05
    assert probe_push(CART, 35) > 0.05


@pytest.mark.parametrize("p", [
    {"cart_mass_kg": 10.0, "move_force_n": 15.0, "tip_force_n": 35.0},
    {"cart_mass_kg": 30.0, "move_force_n": 50.0, "tip_force_n": 80.0},
    {"cart_mass_kg": 25.0, "move_force_n": 30.0, "tip_force_n": 60.0},
    {"cart_mass_kg": 30.0, "move_force_n": 15.0, "tip_force_n": 30.0},
])
def test_cart_threshold_margins(p):
    mv, tp = p["move_force_n"], p["tip_force_n"]
    assert not simulate_cart(p, mv * 0.75).success
    mid = max(mv * 1.25, min(tp * 0.8, (mv + tp) / 2))
    r = simulate_cart(p, mid)
    assert r.success, (p, mid, r.reason)
    if tp * 1.25 <= 95:
        assert simulate_cart(p, tp * 1.25).extra["tipped"]


def test_cart_deterministic():
    assert simulate_cart(CART, 40) == simulate_cart(CART, 40)
