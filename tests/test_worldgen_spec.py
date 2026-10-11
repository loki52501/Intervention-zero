from iz.llm.client import FakeLLM
from iz.worldgen import spec as S
from iz.worldgen.generate import prompt_to_spec, repair, variations
from iz.worldgen.spec import WorldSpec, validate_spec


def good_raw(**over):
    d = {
        "name": "corner_store", "venue": "store", "size_m": [12, 9], "aisles": 3, "shelf_units": 8,
        "products": [{"name": "cereal", "color": "red", "shelf_index": 5, "height_m": 0.9},
                     {"name": "soup", "color": "blue", "shelf_index": 2, "height_m": 0.5}],
        "target_product": "cereal", "checkout": "auto",
        "spills": [{"x": 6.0, "y": 4.0, "radius": 0.3}],
        "shoppers": {"count": 2, "speed_mps": 0.8, "density": "medium"},
        "cart": {"mass_kg": 20, "loaded": False},
        "task": list(S.STEPS), "success": {"max_time_s": 120}, "notes": [],
    }
    d.update(over)
    return d


def test_schema_roundtrip():
    spec = WorldSpec.model_validate(good_raw())
    again = WorldSpec.model_validate_json(spec.model_dump_json())
    assert again == spec
    assert "properties" in WorldSpec.model_json_schema()
    assert validate_spec(spec) == []


def test_prompt_to_spec_repairs_out_of_bounds_spill():
    llm = FakeLLM(responses=[good_raw(spills=[{"x": 55.0, "y": -9.0, "radius": 0.4}])])
    spec = prompt_to_spec("a store with a spill", llm=llm, seed=1)
    assert validate_spec(spec) == []
    s = spec.spills[0]
    w, h = spec.size_m
    assert 0 <= s.x <= w and 0 <= s.y <= h
    assert any("spill 0" in n for n in spec.notes)
    assert len(llm.calls) == 2  # re-asked once (FakeLLM repeats the bad answer), then repaired


def test_repair_clamps_and_fixes_target_and_aisles():
    llm = FakeLLM(responses=[good_raw(size_m=[40, 2], aisles=5, target_product="ghost", cart={"mass_kg": 999})])
    spec = prompt_to_spec("x", llm=llm)
    assert validate_spec(spec) == []
    assert spec.size_m[0] == 20 and spec.cart.mass_kg == 40
    assert spec.target_product in [p.name for p in spec.products]


def test_blocking_spill_moved_off_route():
    spec = WorldSpec.model_validate(good_raw(spills=[]))
    aw = S.aisle_width(spec)
    # ring of spills sealing off the checkout at (11, 0.9)
    import math
    spec.spills = [S.Spill(x=11 + 0.9 * math.cos(a * math.pi / 4), y=0.9 + 0.9 * math.sin(a * math.pi / 4), radius=0.6)
                   for a in range(8)]
    assert aw >= 1.4
    assert any("block" in p for p in validate_spec(spec))
    fixed = repair(spec)
    assert validate_spec(fixed) == []
    assert any('blocked the route' in n for n in fixed.notes)


def test_variations_deterministic_and_differ():
    base = prompt_to_spec("p", llm=FakeLLM(responses=[good_raw()]))
    a = variations(base, 4, seed=7)
    b = variations(base, 4, seed=7)
    c = variations(base, 4, seed=8)
    assert [x.model_dump() for x in a] == [x.model_dump() for x in b]
    assert [x.model_dump() for x in a] != [x.model_dump() for x in c]
    assert len({x.path_seed for x in a}) == 4
    assert len({(x.cart.mass_kg, x.spills[0].x) for x in a}) > 1
    assert all(validate_spec(x) == [] for x in a)
    assert S.shopper_paths(a[0]) != S.shopper_paths(a[1])
