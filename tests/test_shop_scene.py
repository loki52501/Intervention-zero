import math
import os

import mujoco
import numpy as np
import pytest

from iz.shop.scene import build_shop

OBJ = mujoco.mjtObj
BODIES = ["robot", "torso", "head", "cart", "r_hand_b", "l_hand_b", "prod_0", "checkout" if False else "cart"]
JOINTS = ["base_x", "base_y", "base_yaw", "r_shoulder", "r_elbow", "l_shoulder", "l_elbow", "cart_free"]
ACTS = ["act_vx", "act_vy", "act_wz", "act_r_shoulder", "act_r_elbow", "act_l_shoulder", "act_l_elbow"]
SENSORS = ["rf_front", "rf_left", "rf_right", "rf_cart", "touch_r", "f_wrist", "imu_acc", "imu_gyro",
           "cart_up", "cart_pos", "jp_r_shoulder", "jp_r_elbow", "jp_l_shoulder", "jp_l_elbow"]
SITES = ["r_hand", "l_hand", "cart_handle"]
CAMS = ["head_cam"]
EQS = ["grip_cart", "grip_prod"]


def _model(seed, **kw):
    xml, meta = build_shop(seed, **kw)
    return mujoco.MjModel.from_xml_string(xml), meta


def _id(m, t, n):
    return mujoco.mj_name2id(m, t, n)


@pytest.mark.parametrize("seed", range(6))
def test_compiles_and_names(seed):
    m, meta = _model(seed)
    for t, names in [(OBJ.mjOBJ_BODY, BODIES), (OBJ.mjOBJ_JOINT, JOINTS), (OBJ.mjOBJ_ACTUATOR, ACTS),
                     (OBJ.mjOBJ_SENSOR, SENSORS), (OBJ.mjOBJ_SITE, SITES), (OBJ.mjOBJ_CAMERA, CAMS),
                     (OBJ.mjOBJ_EQUALITY, EQS)]:
        for n in names:
            assert _id(m, t, n) >= 0, n
    for i in range(4):
        assert _id(m, OBJ.mjOBJ_GEOM, f"shelf_{i}") >= 0
    assert _id(m, OBJ.mjOBJ_GEOM, "checkout") >= 0
    for i in range(len(meta["spills"])):
        assert _id(m, OBJ.mjOBJ_GEOM, f"spill_{i}") >= 0
    for i in range(meta["n_shoppers"]):
        assert _id(m, OBJ.mjOBJ_BODY, f"shopper_{i}") >= 0 and _id(m, OBJ.mjOBJ_GEOM, f"shopper_{i}_g") >= 0
    assert meta["target_product"] == "prod_0" and 12 <= meta["cart_mass_kg"] <= 35
    assert meta["route"][0] == 0 and len(meta["waypoints"]) > max(meta["route"])
    assert m.opt.timestep == 0.002
    # equality constraints start inactive
    d = mujoco.MjData(m)
    assert d.eq_active.sum() == 0
    # robot qpos is world xy
    mujoco.mj_forward(m, d)
    rid = _id(m, OBJ.mjOBJ_BODY, "robot")
    assert np.allclose(d.xpos[rid][:2], meta["robot_xy"], atol=1e-3)


def test_deterministic_and_dense():
    a, _ = build_shop(3)
    b, _ = build_shop(3)
    assert a == b
    assert build_shop(201)[1]["n_shoppers"] >= 4


def test_stable_500_steps():
    for seed in (0, 1, 4):
        m, meta = _model(seed)
        d = mujoco.MjData(m)
        for _ in range(500):
            mujoco.mj_step(m, d)
        assert not np.isnan(d.qpos).any()
        up = m.sensor_adr[_id(m, OBJ.mjOBJ_SENSOR, "cart_up")]
        assert d.sensordata[up + 2] > 0.999
        for i in range(5):
            assert d.xpos[_id(m, OBJ.mjOBJ_BODY, f"prod_{i}")][2] > 0.4   # products stay on ledges


def _reach(m, d, base_xy, yaw, target, tol=0.04):
    qa = lambda n: m.jnt_qposadr[_id(m, OBJ.mjOBJ_JOINT, n)]
    d.qpos[qa("base_x")], d.qpos[qa("base_y")], d.qpos[qa("base_yaw")] = base_xy[0], base_xy[1], yaw
    sid = _id(m, OBJ.mjOBJ_SITE, "r_hand")
    best = 9
    for s in np.linspace(-0.5, 3.0, 36):
        for e in np.linspace(-2.4, 2.4, 49):
            d.qpos[qa("r_shoulder")], d.qpos[qa("r_elbow")] = s, e
            mujoco.mj_kinematics(m, d)
            best = min(best, np.linalg.norm(d.site_xpos[sid] - target))
    return best < tol, best


@pytest.mark.parametrize("seed", range(6))
def test_arm_can_reach_product_and_basket(seed):
    m, meta = _model(seed)
    d = mujoco.MjData(m)
    tgt = np.array([*meta["target_xy"], meta["target_z"]])
    ok, best = _reach(m, d, meta["pick_xy"], meta["pick_yaw"], tgt)
    assert ok, f"cannot reach product, best {best:.3f}"
    # place: right side faces the cart ahead; hover 0.25 m above the basket rim over its rear part
    cart = np.array([meta["pick_xy"][0] + meta["grip_offset"], meta["pick_xy"][1], 1.05])
    ok, best = _reach(m, d, meta["pick_xy"], meta["place_yaw"], cart, tol=0.06)
    assert ok, f"cannot reach basket, best {best:.3f}"


def tipping_run(v, yaw_rate=1.5, mass=30.0, turn_s=1.6):
    """Robot welded to the cart drives straight in the west corridor, then turns at yaw_rate.
    Returns max cart tilt (deg)."""
    xml, meta = build_shop(0, cart_mass_kg=mass, n_shoppers=0)
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    qa = lambda n: m.jnt_qposadr[_id(m, OBJ.mjOBJ_JOINT, n)]
    d.qpos[qa("base_x")], d.qpos[qa("base_y")], d.qpos[qa("base_yaw")] = -4.3, -3.4, math.pi / 2
    ca = qa("cart_free")
    d.qpos[ca:ca + 3] = [-4.3, -2.4, 0.0]
    d.qpos[ca + 3:ca + 7] = [math.cos(math.pi / 4), 0, 0, math.sin(math.pi / 4)]
    mujoco.mj_forward(m, d)
    d.eq_active[_id(m, OBJ.mjOBJ_EQUALITY, "grip_cart")] = 1
    vx, vy, wz = [_id(m, OBJ.mjOBJ_ACTUATOR, n) for n in ("act_vx", "act_vy", "act_wz")]
    up = m.sensor_adr[_id(m, OBJ.mjOBJ_SENSOR, "cart_up")]
    mx, t = 0.0, 0.0
    while t < 1.2 + turn_s:
        sp = min(v, v * t / 0.8) if t < 1.2 else v
        psi = d.qpos[qa("base_yaw")]
        d.ctrl[vx], d.ctrl[vy] = sp * math.cos(psi), sp * math.sin(psi)
        d.ctrl[wz] = 0.0 if t < 1.2 else yaw_rate
        mujoco.mj_step(m, d)
        t += m.opt.timestep
        mx = max(mx, math.degrees(math.acos(float(np.clip(d.sensordata[up + 2], -1, 1)))))
    return mx


def test_cart_tips_when_fast_not_when_slow():
    fast = tipping_run(1.4)
    slow = tipping_run(0.6)
    print(f"30 kg cart, yaw 1.5 rad/s: 1.4 m/s -> {fast:.1f} deg, 0.6 m/s -> {slow:.1f} deg")
    assert fast > 35.0
    assert slow < 35.0


def test_render_png():
    from PIL import Image
    m, meta = _model(0)
    d = mujoco.MjData(m)
    for _ in range(200):
        mujoco.mj_step(m, d)
    r = mujoco.Renderer(m, 480, 640)
    r.update_scene(d, camera="cam_34")
    img = r.render()
    os.makedirs("out", exist_ok=True)
    Image.fromarray(img).save("out/shop_seed0.png")
    assert img.std() > 10
    r.update_scene(d, camera="head_cam")
    assert r.render().shape == (480, 640, 3)
