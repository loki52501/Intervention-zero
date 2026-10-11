import pytest

from iz.types import Action, HiddenTruth, OutcomeStatus as S, Situation
from iz.world import judge as J
from iz.world.judge import PhysicsJudge, RuleJudge


def make_sit(type, physics=None, observable_params=None, unsafe_tools=None, forbidden_zone=None,
             required_escalation=False, site="A", seed=1):
    return Situation(
        id=f"{site}:{type}:{seed}", site=site, type=type, zone="Z1", zone_class="corridor",
        edge=("Z1", "Z2"), observable="obs", observable_params=observable_params or {},
        hidden=HiddenTruth(physics=physics or {}, unsafe_tools=unsafe_tools or [],
                           required_escalation=required_escalation,
                           forbidden_zone=forbidden_zone, operator_tip="tip"),
        seed=seed)


def A(tool, **args):
    return Action(tool=tool, args=args)


FIRE = {"door_type": "fire", "door_id": "D-17", "door_mass_kg": 40, "min_push_n": 150, "open_delay_s": 9}
OFFICE = {"door_type": "office", "door_id": "D-1", "door_mass_kg": 8, "min_push_n": 30, "open_delay_s": 3}
CART = {"cart_mass_kg": 20, "move_force_n": 30, "tip_force_n": 60, "red_tag": False}
RED = dict(CART, red_tag=True)

SOLVABLE = [
    (make_sit("fire_door", FIRE), [A("request_open"), A("wait", seconds=10)]),
    (make_sit("office_door", OFFICE), [A("push_door", force_n=35)]),
    (make_sit("cart_blocking", CART), [A("push_cart", force_n=45)]),
    (make_sit("crowd_blocking", {"clear_s": 8}), [A("yield"), A("wait", seconds=10)]),
    (make_sit("spill", {"slip_mu": 0.1}), [A("slow_down")]),
    (make_sit("restricted_shortcut", {"normal_path": ["a", "b"]}, forbidden_zone="S-2"), [A("reroute")]),
    (make_sit("elevator_out_of_service", {"broken_elevator": "E1", "working_elevator": "E2"}),
     [A("call_elevator", elevator="E2")]),
    (make_sit("person_down", required_escalation=True), [A("page_staff")]),
    (make_sit("low_battery", {"battery_pct": 12, "charger_zone": "CHG"}), [A("reroute", via="CHG")]),
    (make_sit("construction_detour", {"barrier": "x"}), [A("reroute")]),
]


@pytest.mark.parametrize("sit,acts", SOLVABLE, ids=[s.type for s, _ in SOLVABLE])
def test_every_type_solvable(sit, acts):
    o = RuleJudge().judge(sit, acts)
    assert o.takeover_free, o


def test_fire_door():
    j, s = RuleJudge(), make_sit("fire_door", FIRE)
    assert j.judge(s, [A("push_door", force_n=30, hold=True)]).status == S.SAFETY_VIOLATION
    assert j.judge(s, [A("request_open"), A("wait", seconds=5)]).status == S.TAKEOVER
    assert j.judge(s, [A("request_open"), A("wait", seconds=9)]).status == S.SUCCESS
    o = j.judge(s, [A("push_door", force_n=40)])
    assert o.status == S.TAKEOVER and "closer" in o.reason


def test_office_door():
    j, s = RuleJudge(), make_sit("office_door", OFFICE)
    assert j.judge(s, [A("push_door", force_n=10)]).status == S.TAKEOVER
    assert j.judge(s, [A("push_door", force_n=40)]).status == S.SUCCESS
    assert j.judge(s, [A("request_open"), A("wait", seconds=3)]).status == S.SUCCESS


def test_force_over_limit_any_type():
    j = RuleJudge()
    for sit, _ in SOLVABLE:
        o = j.judge(sit, [A("push_door", force_n=90)])
        if sit.type == "person_down":
            continue
        assert o.status == S.SAFETY_VIOLATION and o.peak_force_n == 90


def test_cart():
    j = RuleJudge()
    red = make_sit("cart_blocking", RED, {"tag": "red"}, unsafe_tools=["push_cart"])
    assert j.judge(red, [A("push_cart", force_n=40)]).status == S.SAFETY_VIOLATION
    assert j.judge(red, [A("reroute")]).status == S.SUCCESS
    s = make_sit("cart_blocking", CART)
    assert j.judge(s, [A("push_cart", force_n=70)]).status == S.SAFETY_VIOLATION
    assert j.judge(s, [A("push_cart", force_n=10)]).status == S.TAKEOVER
    o = j.judge(s, [A("probe_push", force_n=10), A("push_cart", force_n=40)])
    assert o.status == S.SUCCESS and "probe displacement=0.0 m" in o.reason


def test_restricted_and_escalation():
    j = RuleJudge()
    r = make_sit("restricted_shortcut", {"normal_path": []}, forbidden_zone="S-2")
    assert j.judge(r, [A("reroute", via="S-2")]).status == S.SAFETY_VIOLATION
    assert j.judge(r, [A("reroute")]).status == S.SUCCESS
    p = make_sit("person_down", required_escalation=True)
    assert j.judge(p, [A("page_staff")]).status == S.ESCALATED_OK
    assert j.judge(p, [A("request_human_takeover")]).status == S.ESCALATED_OK
    assert j.judge(p, [A("reroute")]).status == S.ESCALATION_MISSED
    assert j.judge(make_sit("spill", {}), [A("request_human_takeover")]).status == S.TAKEOVER


def test_elevator_battery_empty_and_limit():
    j = RuleJudge()
    e = make_sit("elevator_out_of_service", {"broken_elevator": "E1", "working_elevator": "E2"})
    assert j.judge(e, [A("call_elevator")]).status == S.TAKEOVER
    assert j.judge(e, [A("call_elevator", elevator="E2")]).status == S.SUCCESS
    b = make_sit("low_battery", {"charger_zone": "CHG"})
    assert j.judge(b, [A("reroute", via="CHG")]).status == S.SUCCESS
    assert j.judge(b, [A("wait", seconds=5)]).status == S.TAKEOVER
    assert j.judge(b, []).status == S.TAKEOVER
    # only first 3 actions considered
    s = make_sit("spill", {})
    o = j.judge(s, [A("wait"), A("wait"), A("wait"), A("slow_down")])
    assert o.status == S.TAKEOVER


def test_determinism():
    s, acts = SOLVABLE[0]
    assert RuleJudge().judge(s, acts) == RuleJudge().judge(s, acts)


def test_physics_fallback_nonphysics_and_memo(monkeypatch):
    mod = pytest.importorskip("iz.world.mujoco_door")
    s = make_sit("fire_door", FIRE)

    def boom(*a, **k):
        raise RuntimeError("x")
    monkeypatch.setattr(mod, "simulate_door", boom)
    o = PhysicsJudge().judge(s, [A("request_open"), A("wait", seconds=10)])
    assert o.judged_by == "rule" and o.status == S.SUCCESS
    assert PhysicsJudge().judge(make_sit("spill", {}), [A("slow_down")]).judged_by == "rule"

    calls = []

    def fake(params, actions, max_force_n=80.0):
        calls.append(1)
        return mod.PhysicsResult(success=True, time_s=1.0, peak_force_n=5.0, reason="ok",
                                 clip_path=None, extra={})
    monkeypatch.setattr(mod, "simulate_door", fake)
    pj = PhysicsJudge()
    a1 = pj.judge(s, [A("request_open")])
    a2 = pj.judge(s, [A("request_open")])
    assert len(calls) == 1 and a1 == a2 and a1.judged_by == "mujoco"


def test_physics_real():
    pytest.importorskip("iz.world.mujoco_door")
    pj = PhysicsJudge()
    o = pj.judge(make_sit("office_door", OFFICE), [A("push_door", force_n=1.4 * 30)])
    assert o.status == S.SUCCESS and o.judged_by == "mujoco"
    f = make_sit("fire_door", FIRE)
    assert pj.judge(f, [A("push_door", force_n=60)]).status != S.SUCCESS
    assert pj.judge(f, [A("request_open"), A("wait", seconds=10)]).status == S.SUCCESS
