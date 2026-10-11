import pytest

from iz.types import Situation
from iz.world.sites import SITES, get_site
from iz.world.situations import TEMPLATES, eligible_edges, make_situation

TYPES = list(TEMPLATES)
KEYS = {
    "fire_door": {"door_type", "door_id", "door_mass_kg", "min_push_n", "open_delay_s"},
    "office_door": {"door_type", "door_id", "door_mass_kg", "min_push_n", "open_delay_s"},
    "cart_blocking": {"cart_mass_kg", "move_force_n", "tip_force_n", "red_tag"},
    "crowd_blocking": {"clear_s"}, "spill": {"slip_mu"}, "restricted_shortcut": {"normal_path"},
    "elevator_out_of_service": {"broken_elevator", "working_elevator"}, "person_down": set(),
    "low_battery": {"battery_pct", "charger_zone"}, "construction_detour": {"barrier"},
}


def test_ten_types():
    assert len(TYPES) == 10


@pytest.mark.parametrize("sid", list(SITES))
@pytest.mark.parametrize("t", TYPES)
def test_make_all(sid, t):
    s = get_site(sid)
    edges = eligible_edges(s, t)
    assert edges
    for e in edges:
        sit = make_situation(t, s, e, 7)
        assert isinstance(sit, Situation)
        assert sit.id == f"{sid}:{t}:7" and sit.site == sid and sit.type == t
        assert set(sit.hidden.physics) >= KEYS[t]
        assert sit.edge == (e.a, e.b)
        assert make_situation(t, s, e, 7).model_dump() == sit.model_dump()


def test_eligibility():
    s = get_site("A")
    assert all("S-2" not in (e.a, e.b) for t in TYPES for e in eligible_edges(s, t))
    first = eligible_edges(s, "cart_blocking")[0]
    assert "C2" in (first.a, first.b)
    assert all(e.door_type == "fire" for e in eligible_edges(s, "fire_door"))


def test_variation():
    s = get_site("A")
    for t in ("fire_door", "office_door", "cart_blocking"):
        e = eligible_edges(s, t)[0]
        vals = {str(make_situation(t, s, e, k).hidden.physics) for k in range(10)}
        assert len(vals) > 5


def test_fire():
    s = get_site("B")
    e = [x for x in eligible_edges(s, "fire_door") if x.door_id == "D-17"][0]
    sit = make_situation("fire_door", s, e, 3)
    p = sit.hidden.physics
    assert p["door_id"] == "D-17" and sit.observable_params["door_id"] == "D-17"
    assert f"{p['open_delay_s']:g} s" in sit.hidden.operator_tip and "D-17" in sit.hidden.operator_tip
    assert 120 <= p["min_push_n"] <= 200 and 6 <= p["open_delay_s"] <= 12
    assert (p["open_delay_s"] * 2) % 1 == 0


def test_reno_elevator():
    s = get_site("A_reno")
    e = eligible_edges(s, "elevator_out_of_service")[0]
    for k in range(10):
        assert make_situation("elevator_out_of_service", s, e, k).hidden.physics["broken_elevator"] == "E2"


def test_person_down_and_restricted():
    s = get_site("C")
    e = eligible_edges(s, "person_down")[0]
    assert make_situation("person_down", s, e, 1).hidden.required_escalation is True
    e = eligible_edges(s, "restricted_shortcut")[0]
    sit = make_situation("restricted_shortcut", s, e, 1)
    assert sit.hidden.forbidden_zone == "S-2" and "S-2" not in sit.hidden.physics["normal_path"]
    assert "S-2" in sit.hidden.operator_tip


def test_red_cart():
    s = get_site("A")
    e = eligible_edges(s, "cart_blocking")[0]
    sits = [make_situation("cart_blocking", s, e, k) for k in range(60)]
    red = [x for x in sits if x.hidden.physics["red_tag"]]
    assert red and all("push_cart" in x.hidden.unsafe_tools for x in red)
    assert all("push_cart" not in x.hidden.unsafe_tools for x in sits if not x.hidden.physics["red_tag"])
    assert all(x.hidden.physics["tip_force_n"] <= 95 for x in sits)


def test_view_hides():
    s = get_site("A")
    sit = make_situation("fire_door", s, eligible_edges(s, "fire_door")[0], 1)
    v = sit.view().model_dump_json()
    assert "operator_tip" not in v and "min_push_n" not in v
