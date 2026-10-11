import json
import math

import mujoco
import numpy as np

from iz.worldgen import terrain as T

D = {"friction": [0.4, 1.0], "mass_scale": [0.9, 1.1],
     "push": {"interval_s": [5, 10], "speed_mps": [0.1, 1.0]},
     "joint_noise_rad": 0.03, "action_latency_steps": [0, 3]}


def test_sample_dr_deterministic_per_seed():
    a = T.sample_dr(D, "w", 1)
    assert a == T.sample_dr(D, "w", 1)
    assert a != T.sample_dr(D, "w", 2)
    assert 0.4 <= a["friction"] <= 1.0
    assert 0.9 <= a["mass_scale"] <= 1.1
    assert 0 <= a["latency_steps"] <= 3
    ps = a["pushes"]
    assert ps
    for p, q in zip(ps, ps[1:]):
        assert 5 - 1e-2 <= q["t"] - p["t"] <= 10 + 1e-2
    for p in ps:
        assert 0.1 - 1e-3 <= math.hypot(p["vx"], p["vy"]) <= 1.0 + 1e-3
    json.dumps(a)
    assert T.sample_dr(None, "w", 1)["enabled"] is False


def _compile(asset, geom):
    xml = (f"<mujoco><asset>{asset}</asset><worldbody><geom type='plane' size='6 5 .1'/>"
           f"{geom}</worldbody></mujoco>")
    return mujoco.MjModel.from_xml_string(xml)


def test_rough_hfield_compiles():
    t = {"type": "rough", "roughness_m": 0.04}
    hf = T.hfield_data(t, (0, 0, 10, 8), "w", 0)
    assert hf["nrow"] * hf["ncol"] == len(hf["elev"])
    asset, geom = T.hfield_xml(hf)
    m = _compile(asset, geom)
    assert m.nhfield == 1
    T.set_hfield(m, hf)
    d = np.array(m.hfield_data)
    assert d.min() < 0.05 and d.max() > 0.95
    assert hf == T.hfield_data(t, (0, 0, 10, 8), "w", 0)
    assert hf["elev"] != T.hfield_data(t, (0, 0, 10, 8), "w", 1)["elev"]
    assert T.hfield_data({"type": "flat"}, (0, 0, 1, 1), "w", 0) is None
    assert T.hfield_data(t, None, "w", 0) is None


def test_slope_and_height_at():
    t = {"type": "slope", "slope_deg": 10}
    r = (8, 2, 12, 6)
    assert abs(T.height_at(t, r, 12, 4) - math.tan(math.radians(10)) * 4) < 1e-6
    assert abs(T.height_at(t, r, 8, 4)) < 1e-6
    assert T.height_at(t, r, 5, 4) == 0.0
    assert T.height_at(t, r, 10, 9) == 0.0
    hf = T.hfield_data(t, r, "w", 0)
    assert abs(hf["zmax"] - 0.6) < 1e-9  # hfield zmax clipped to 0.6 m (height_at is not)
    asset, geom = T.hfield_xml(hf)
    assert _compile(asset, geom).nhfield == 1


def test_stairs_xml():
    t = {"type": "steps", "step_n": 4, "step_h_m": 0.1}
    xml, rects = T.stairs_xml(t, (0, 0, 4, 3))
    assert len(rects) == 4
    m = _compile("", xml)
    hs = []
    for k in range(4):
        g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"stair_{k}")
        assert g >= 0
        assert m.geom_contype[g] == 64 and m.geom_conaffinity[g] == 10
        hs.append(m.geom_size[g][2])
    assert hs == sorted(hs) and abs(hs[3] - 0.2) < 1e-9
    assert T.stairs_xml({"type": "flat"}, (0, 0, 1, 1)) == ("", [])
    assert abs(T.height_at(t, (0, 0, 4, 3), 3.5, 1) - 0.4) < 1e-9


ROBOT = """<mujoco><option gravity='0 0 0'/><worldbody>
<body name='robot'><joint name='base_x' type='slide' axis='1 0 0' damping='0'/>
<joint name='base_y' type='slide' axis='0 1 0' damping='0'/>
<geom size='.1' mass='5'/>
<body name='torso' pos='0 0 1'><geom size='.1' mass='30'/></body></body>
<body name='cart' pos='3 0 0'><freejoint/><geom size='.1' mass='7'/></body>
</worldbody></mujoco>"""


def test_apply_dr_push_moves_planar_body():
    m = mujoco.MjModel.from_xml_string(ROBOT)
    d = mujoco.MjData(m)
    meta = {"dr": {"enabled": True, "mass_scale": 1.0,
                   "pushes": [{"t": 0.0, "vx": 1.0, "vy": 0.0, "dur": 0.1}]}}
    while d.time < 0.2:
        T.apply_dr(m, d, meta, d.time)
        mujoco.mj_step(m, d)
    v = d.qvel[m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "base_x")]]
    assert abs(v - 1.0) < 0.25
    T.apply_dr(m, d, meta, d.time)
    assert not d.xfrc_applied.any()


def test_apply_dr_model_scales_robot_only():
    m = mujoco.MjModel.from_xml_string(ROBOT)
    rb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "robot")
    tb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso")
    cb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "cart")
    before = (m.body_mass[rb], m.body_mass[tb], m.body_mass[cb])
    T.apply_dr_model(m, {"dr": {"enabled": True, "mass_scale": 1.1}})
    assert abs(m.body_mass[rb] - before[0] * 1.1) < 1e-9
    assert abs(m.body_mass[tb] - before[1] * 1.1) < 1e-9
    assert m.body_mass[cb] == before[2]


def test_delay_action():
    buf = []
    out = [T.delay_action(buf, i, 2) for i in range(5)]
    assert out[2:] == [0, 1, 2]
    assert T.delay_action([], 9, 0) == 9
    rng = __import__("random").Random(1)
    n = T.joint_noise(rng, 5, 0.03)
    assert len(n) == 5 and all(abs(x) <= 0.03 for x in n)
