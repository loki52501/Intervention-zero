"""Tests for the Unitree G1 kinematic-puppet body (iz/robot/g1.py)."""
import mujoco
import numpy as np
import pytest

from iz.robot import g1_available

pytestmark = pytest.mark.skipif(not g1_available(), reason="Menagerie G1 not cloned")

LEGS = [f"{s}_{j}_joint" for s in ("left", "right") for j in ("hip_pitch", "hip_roll", "knee", "ankle_pitch")]


def _model(ids=(0,)):
    from iz.robot.g1 import build_standalone_xml
    m = mujoco.MjModel.from_xml_string(build_standalone_xml(ids))
    return m, mujoco.MjData(m)


def _id(m, kind, name):
    i = mujoco.mj_name2id(m, kind, name)
    assert i >= 0, f"{name} missing"
    return i


def test_contract_names():
    m, _ = _model()
    O = mujoco.mjtObj
    for n in ["robot", "torso", "head", "r_hand_b", "l_hand_b"]:
        _id(m, O.mjOBJ_BODY, n)
    for n in ["r_hand", "l_hand", "r_wrist", "l_wrist", "imu"]:
        _id(m, O.mjOBJ_SITE, n)
    _id(m, O.mjOBJ_CAMERA, "head_cam")
    _id(m, O.mjOBJ_CAMERA, "cam_follow")
    for n in ["base_x", "base_y", "base_yaw", "r_shoulder", "r_elbow", "l_shoulder", "l_elbow"] + LEGS:
        _id(m, O.mjOBJ_JOINT, n)
    for n in ["act_vx", "act_vy", "act_wz", "act_r_shoulder", "act_r_elbow", "act_l_shoulder", "act_l_elbow"]:
        _id(m, O.mjOBJ_ACTUATOR, n)
    for n in LEGS:
        _id(m, O.mjOBJ_ACTUATOR, "act_" + n)
    for n in ["rf_front", "rf_left", "rf_right", "touch_r", "f_wrist", "imu_acc", "imu_gyro",
              "jp_r_shoulder", "jp_r_elbow", "jp_l_shoulder", "jp_l_elbow"]:
        _id(m, O.mjOBJ_SENSOR, n)
    for n in ["base_g", "waist_g", "torso_g", "head_g", "r_hand_g", "l_hand_g"]:
        _id(m, O.mjOBJ_GEOM, n)


def test_legs_never_collide():
    m, _ = _model()
    leg_bodies = set()
    for i in range(m.nbody):
        n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) or ""
        if any(k in n for k in ("hip", "knee", "ankle")):
            leg_bodies.add(i)
    assert leg_bodies
    n_geoms = 0
    for g in range(m.ngeom):
        if m.geom_bodyid[g] in leg_bodies:
            n_geoms += 1
            assert m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0
    assert n_geoms > 0


def test_forward_drive_keeps_height():
    m, d = _model()
    mujoco.mj_forward(m, d)
    z0 = d.body("pelvis").xpos[2]
    d.ctrl[_id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_vx")] = 1.0
    for _ in range(int(2.0 / m.opt.timestep)):
        mujoco.mj_step(m, d)
    assert np.isfinite(d.qpos).all()
    assert abs(d.body("pelvis").xpos[2] - z0) < 0.01
    assert d.qpos[m.joint("base_x").qposadr[0]] > 1.0


def test_two_robots():
    m, _ = _model((0, 1))
    O = mujoco.mjtObj
    _id(m, O.mjOBJ_BODY, "robot_1")
    _id(m, O.mjOBJ_ACTUATOR, "act_vx_1")
    _id(m, O.mjOBJ_SITE, "r_hand_1")
    _id(m, O.mjOBJ_JOINT, "r_shoulder_1")
    _id(m, O.mjOBJ_SENSOR, "rf_front_1")


def _hand_x(m, d, shoulder, elbow):
    d.qpos[:] = m.qpos0
    d.qpos[m.joint("r_shoulder").qposadr[0]] = shoulder
    d.qpos[m.joint("r_elbow").qposadr[0]] = elbow
    mujoco.mj_kinematics(m, d)
    return d.site("r_hand").xpos[0] - d.body("robot").xpos[0]


def test_sign_conventions():
    m, d = _model()
    base = _hand_x(m, d, 0.0, 0.0)
    assert _hand_x(m, d, 0.6, 0.0) > base + 0.05
    assert _hand_x(m, d, 0.0, 0.6) > base + 0.05


def test_meta_arm():
    from iz.robot.g1 import build_robot_xml
    _, meta, _ = build_robot_xml(0)
    assert meta["robot_body"] == "g1"
    assert meta["arm"]["l1"] > 0.1 and meta["arm"]["l2"] > 0.1
    assert len(meta["arm"]["shoulder_xyz"]) == 3
    assert len(meta["leg_joints"]) == 8 and len(meta["leg_acts"]) == 8


def test_feet_near_floor():
    from iz.robot.g1 import foot_min_z
    assert -0.01 <= foot_min_z() <= 0.03
