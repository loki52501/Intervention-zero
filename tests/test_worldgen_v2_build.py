import mujoco
import numpy as np
import pytest

from iz.worldgen import spec as S
from iz.worldgen import terrain as T
from iz.worldgen.build import build_world, check_world, compile_world
from iz.worldgen.spec import (DR, Hazard, Lighting, Obstacle, Product, Shoppers, Terrain,
                              WorldSpec)
from tests.test_worldgen_build import _small_store

_P = [Product(name="cereal", color="red", shelf_index=1, height_m=0.9),
      Product(name="milk", color="blue", shelf_index=3, height_m=0.6)]


def _base(name, **kw):
    return WorldSpec(name=name, venue="store", size_m=[12.0, 8.0], aisles=3, shelf_units=6,
                     products=list(_P), target_product="cereal", **kw)


def rough_floor():
    return _base("rough_floor", terrain=Terrain(type="rough", roughness_m=0.03), dr=DR())


def slope_patch():
    return _base("slope_patch", terrain=Terrain(type="slope", slope_deg=10))


def stairs_site():
    return _base("stairs_site",
                 terrain=Terrain(type="steps", step_h_m=0.1, step_n=4, region=[7.5, 4.0, 11.8, 7.4]),
                 obstacles=[Obstacle(kind="box", x=1.2, y=7.2, w=0.6, l=0.6, h=0.6),
                            Obstacle(kind="barrier", x=10.8, y=4.2, w=0.3, l=1.2, h=0.9)],
                 hazards=[Hazard(kind="cable", x=3.0, y=0.5, radius=0.6),
                          Hazard(kind="low_light", x=6.0, y=0.9, radius=0.6)],
                 lighting=Lighting(ambient=0.3))


def door_corridor():
    return _base("door_corridor",
                 obstacles=[Obstacle(kind="door", x=11.2, y=2.9, w=0.08, l=1.0, h=2.0, open=False),
                            Obstacle(kind="door", x=11.2, y=5.5, w=0.08, l=1.0, h=2.0, open=True)])


def free_boxes():
    return _base("free_boxes",
                 obstacles=[Obstacle(kind="box", x=1.0, y=7.2, w=0.5, l=0.5, h=0.5, mass_kg=12),
                            Obstacle(kind="box", x=11.0, y=7.0, w=0.5, l=0.5, h=0.5, mass_kg=12)],
                 dr=DR())


def fleet_icy():
    return _base("fleet_icy", robots=3, dr=DR(friction=[0.2, 0.5]), people=Shoppers(count=4))


SPECS = {f.__name__: f for f in (rough_floor, slope_patch, stairs_site, door_corridor, free_boxes, fleet_icy)}


@pytest.mark.parametrize("name", list(SPECS))
def test_v2_build_check(name):
    spec = SPECS[name]()
    assert S.validate_spec(spec) == []
    xml, meta = build_world(spec, seed=1)
    assert check_world(xml, meta) == []
    mujoco.MjModel.from_xml_string(xml)
    compile_world(xml, meta)
    assert meta["spec_version"] == spec.spec_version
    assert len(meta["robot_starts"]) == spec.robots


def _model(spec, seed=1):
    xml, meta = build_world(spec, seed=seed)
    return compile_world(xml, meta), xml, meta


def test_rough_has_hfield():
    m, xml, meta = _model(rough_floor())
    assert m.nhfield == 1
    g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "terrain")
    assert g >= 0 and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_HFIELD
    assert meta["terrain"]["has_hfield"] and meta["terrain"]["collides_with"] == "cart+products"
    assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor") >= 0


def test_slope_and_steps_terrain():
    m, _, meta = _model(slope_patch())
    assert m.nhfield == 1 and meta["terrain"]["region"]
    m, _, meta = _model(stairs_site())
    assert m.nhfield == 0
    assert all(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"stair_{k}") >= 0 for k in range(4))
    assert meta["hazards"] and {h["kind"] for h in meta["hazards"]} == {"cable", "low_light"}
    assert "ambient=\"0.30 0.30 0.30\"" in build_world(stairs_site(), 1)[0]


def test_doors():
    m, _, meta = _model(door_corridor())
    assert len(meta["doors"]) == 2
    for d in meta["doors"]:
        j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, d["joint"])
        assert j >= 0
        assert tuple(np.round(m.jnt_range[j], 6)) == (-1.7, 0.0)
    opened = [d for d in meta["doors"] if d["open"]]
    assert len(opened) == 1
    j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, opened[0]["joint"])
    assert abs(m.qpos0[m.jnt_qposadr[j]] + 1.6) < 1e-9


def test_free_boxes():
    m, _, meta = _model(free_boxes(), seed=1)
    for i in (0, 1):
        assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"obst_{i}_free") >= 0
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"obst_{i}")
        assert abs(m.body_mass[b] - 12 * meta["dr"]["mass_scale"]) < 1e-2
        assert meta["obstacles"][i]["free"]


def test_fleet_icy():
    m, _, meta = _model(fleet_icy())
    assert sum(1 for b in range(m.nbody)
               if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or "").startswith("shopper_")) == 4
    assert len(meta["robot_starts"]) == 3 and meta["people"] == 4
    assert 0.2 <= meta["dr"]["friction"] <= 0.5


def test_dr_deterministic_per_seed():
    a, ma = build_world(fleet_icy(), seed=1)
    b, mb = build_world(fleet_icy(), seed=1)
    assert a == b and ma["dr"] == mb["dr"]
    c, mc = build_world(fleet_icy(), seed=2)
    assert mc["dr"]["friction"] != ma["dr"]["friction"] or mc["dr"]["pushes"] != ma["dr"]["pushes"]
    for xml, meta in ((a, ma), (c, mc)):
        m = mujoco.MjModel.from_xml_string(xml)
        g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        assert abs(m.geom_friction[g][0] - meta["dr"]["friction"]) < 1e-3


def test_v1_bytes_unchanged():
    xml, meta = build_world(_small_store(), seed=1)
    for bad in ("hfield", "obst_", "door_", "stair_", "cable_"):
        assert bad not in xml
    assert 'friction="1 0.005 0.0001"' in xml
    assert 'ambient="0.45 0.45 0.45"' in xml
    assert meta["dr"]["enabled"] is False
    assert meta["terrain"]["has_hfield"] is False
    assert meta["obstacles"] == [] and meta["doors"] == []


def test_apply_dr_in_loop():
    m, _, meta = _model(fleet_icy())
    d = mujoco.MjData(m)
    T.apply_dr_model(m, meta)
    mujoco.mj_forward(m, d)
    for _ in range(600):
        T.apply_dr(m, d, meta, d.time)
        mujoco.mj_step(m, d)
    assert np.all(np.isfinite(d.qpos))
