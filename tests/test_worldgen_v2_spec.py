import math

import pytest

from iz.worldgen import generate as G
from iz.worldgen import spec as S
from iz.worldgen.generate import (WorldRefused, demo_specs, keyword_spec_v2, repair, repair_or_refuse,
                                  sanitize)
from iz.worldgen.spec import (DR, Hazard, Lighting, Obstacle, Push, RobotTask, Shoppers, Terrain,
                              WorldSpec, validate_spec)
from tests.test_worldgen_spec import good_raw


def test_v1_dict_still_validates():
    spec = WorldSpec.model_validate(good_raw())
    again = WorldSpec.model_validate(spec.model_dump())
    assert again == spec
    assert validate_spec(spec) == []
    assert spec.dr is None and spec.template == "retail" and spec.robots == 1
    assert spec.spec_version == 1 and spec.people is None


def test_v2_full_spec_roundtrip():
    spec = WorldSpec.model_validate(good_raw(spills=[]))
    spec.template = "warehouse"
    spec.venue = "an icy dock"
    spec.terrain = Terrain(type="steps", step_h_m=0.1, step_n=4, region=[10.0, 2.0, 11.5, 4.0])
    spec.obstacles = [Obstacle(kind="box", x=7, y=4, mass_kg=12.0),
                      Obstacle(kind="door", x=8, y=4, w=0.1, l=1.6, h=2.0, open=True)]
    spec.hazards = [Hazard(kind="cable", x=7, y=5), Hazard(kind="low_light", x=9, y=5, radius=1.0)]
    spec.people = Shoppers(count=3, speed_mps=0.9, density="medium")
    spec.lighting = Lighting(ambient=0.3)
    spec.dr = DR(friction=[0.5, 0.2], push=Push(speed_mps=[1.0, 0.5]))
    spec.robots = 2
    spec.tasks = [RobotTask(robot=0, steps=["go_to", "open_door"], target=S.XY(x=5, y=4), door=1)]
    spec.spec_version = 2
    spec = WorldSpec.model_validate(spec.model_dump())  # runs validators (sort, alias)
    again = WorldSpec.model_validate_json(spec.model_dump_json())
    assert again == spec
    assert again.dr.friction == [0.2, 0.5]  # sorted
    assert again.shoppers.count == 3  # people is the v2 alias


def _seal_doors(open_):
    spec = WorldSpec.model_validate(good_raw(spills=[]))
    h = spec.size_m[1]
    spec.obstacles = [Obstacle(kind="door", x=6.0, y=yc, w=0.1, l=3.0, h=2.0, open=open_)
                      for yc in (1.5, 4.5, h - 1.5)]
    return spec


def test_closed_door_blocks_route():
    spec = _seal_doors(False)
    probs = validate_spec(spec)
    assert any("no free route" in p for p in probs)
    assert any("route blocked by obstacles/closed doors" in p for p in probs)
    spec2 = WorldSpec.model_validate(good_raw(spills=[]))
    h = spec2.size_m[1]
    spec2.obstacles = [Obstacle(kind="barrier", x=6.0, y=h / 2, w=0.1, l=4.0, h=2.0)]
    spec2.obstacles = [Obstacle(kind="barrier", x=6.0, y=yc, w=0.1, l=3.0, h=2.0) for yc in (1.5, 4.5, h - 1.5)]
    assert any("no free route" in p for p in validate_spec(spec2))
    assert validate_spec(_seal_doors(True)) == []
    # repair opens a door across the only route
    fixed = repair(_seal_doors(False))
    assert validate_spec(fixed) == []
    assert any("closed across the only route" in n for n in fixed.notes)


def test_repair_relocates_obstacle_on_start():
    spec = WorldSpec.model_validate(good_raw(spills=[]))
    spec.obstacles = [Obstacle(kind="box", x=0.8, y=0.8)]
    assert any("sits on the start" in p for p in validate_spec(spec))
    r = repair(spec)
    assert validate_spec(r) == []
    o = r.obstacles[0]
    assert math.hypot(o.x - S.ROBOT_START[0], o.y - S.ROBOT_START[1]) > 0.6
    assert any("obstacle 0" in n for n in r.notes)


def test_slope_over_15_reported_and_clamped():
    spec = WorldSpec.model_validate(good_raw(spills=[]))
    spec.terrain = Terrain(type="slope", slope_deg=25)
    assert any("impassable" in p for p in validate_spec(spec))
    r = repair(spec)
    assert r.terrain.slope_deg == 15
    assert validate_spec(r) == []


def test_refuse_after_3_rounds(monkeypatch):
    spec = WorldSpec.model_validate(good_raw(spills=[]))
    spec.obstacles = [Obstacle(kind="barrier", x=10.0, y=0.9, w=0.1, l=2.0, h=2.0),
                      Obstacle(kind="barrier", x=11.0, y=1.8, w=2.2, l=0.1, h=2.0)]
    assert validate_spec(spec)
    monkeypatch.setattr(G, "_relocate_obstacle", lambda sp, i: False)
    with pytest.raises(WorldRefused) as ei:
        repair_or_refuse(spec)
    assert ei.value.reasons
    assert str(ei.value).startswith("world refused: ")


def test_keyword_mapping():
    d = keyword_spec_v2("icy loading dock with a ramp")
    assert d["dr"]["friction"] == [0.2, 0.5]
    assert d["terrain"]["type"] == "ramp" and 6 <= d["terrain"]["slope_deg"] <= 10
    c = keyword_spec_v2("a construction site")
    assert c["terrain"]["type"] == "steps"
    assert sum(o["kind"] == "box" for o in c["obstacles"]) >= 1
    assert sum(h["kind"] == "cable" for h in c["hazards"]) >= 1
    assert sum(h["kind"] == "low_light" for h in c["hazards"]) >= 1
    assert c["lighting"]["ambient"] <= 0.35


def test_demo_specs_validate():
    specs = demo_specs()
    assert len(specs) == 3
    for slug, sp in specs.items():
        assert validate_spec(sp) == [], slug
        assert sp.name == slug


def test_hazard_spill_counts_as_spill():
    spec = WorldSpec.model_validate(good_raw(spills=[]))
    spec.hazards = [Hazard(kind="spill", x=0.8, y=0.8, radius=0.3)]
    assert any("sits on the start" in p for p in validate_spec(spec))
    assert validate_spec(repair(spec)) == []


def test_robot_starts():
    spec = WorldSpec.model_validate(good_raw(spills=[]))
    spec.robots = 3
    pts = S.robot_starts(spec)
    assert len(pts) == 3 and pts[0] == S.ROBOT_START
    assert len(set(pts)) == 3
    free = S._grid(spec, True)
    for i, a in enumerate(pts):
        c = S._cell(a)
        assert free[c[0]][c[1]]
        for b in pts[i + 1:]:
            assert math.hypot(a[0] - b[0], a[1] - b[1]) >= 0.9 - 1e-9


def test_sanitize_v2_dict_and_v1_unchanged():
    raw = keyword_spec_v2("crowded hospital corridor with a fire door, gravel and pushy staff")
    sp, _ = sanitize(raw)
    assert sp.spec_version == 2 and sp.template == "hospital_corridor" and sp.dr is not None
    assert sp.people is not None and sp.shoppers.count == sp.people.count >= 5
    sp1, _ = sanitize(good_raw())
    assert sp1.spec_version == 1 and sp1.dr is None
