"""Tests for sagittal 2-link arm IK."""
import math

import numpy as np
import pytest

from iz.robot import arm


def test_round_trip():
    rng = np.random.default_rng(0)
    l1, l2 = 0.2, 0.3
    for _ in range(50):
        r = rng.uniform(0.12, 0.49)
        ang = rng.uniform(-math.pi, math.pi)
        fwd, z = r * math.cos(ang), r * math.sin(ang)
        qs, qe, ok = arm.two_link_ik(fwd, z, l1, l2, 0.3, -0.2)
        assert ok
        f2, z2 = arm.forward(qs, qe, l1, l2, 0.3, -0.2)
        assert abs(f2 - fwd) < 1e-6 and abs(z2 - z) < 1e-6


def test_unreachable():
    for p in [(2.0, 1.0), (0.0, 0.0), (0.01, -0.02), (-3, 3)]:
        qs, qe, ok = arm.two_link_ik(*p, 0.2, 0.3, 0.1, 0.1)
        assert not ok and math.isfinite(qs) and math.isfinite(qe)


def test_pick_targets():
    meta = dict(shoulder_xyz=(0.0, -0.2, 1.2), l1=0.2, l2=0.3, a1_0=0.0, a2_0=0.0)
    qs, qe, ok = arm.pick_targets(meta, 0.83, None, 0.35)
    f, z = arm.forward(qs, qe, 0.2, 0.3)
    assert ok and abs(f - 0.35) < 1e-6 and abs(z - (0.9 - 1.2)) < 1e-6
    qs, qe, ok = arm.pick_targets(meta, 0.83, 0.95, 0.35)
    f, z = arm.forward(qs, qe, 0.2, 0.3)
    assert abs(z - (0.95 - 1.2)) < 1e-6


def _g1():
    robot = pytest.importorskip("iz.robot")
    if not robot.g1_available():
        pytest.skip("G1 assets not available")
    g1 = pytest.importorskip("iz.robot.g1")
    return g1


def test_g1_hand_reaches_ledge():
    g1 = _g1()
    mujoco = pytest.importorskip("mujoco")
    xml = g1.build_standalone_xml((0,))
    meta = g1.build_robot_xml(0)[1]
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    a = meta["arm"]
    sh = a["shoulder_xyz"]
    l1, l2 = a["l1"], a["l2"]
    target_z = 0.9
    dz = target_z - sh[2]
    reach = 0.8 * (l1 + l2)
    fwd = math.sqrt(max(reach ** 2 - dz ** 2, 0.0))
    standoff = sh[0] + fwd
    qs, qe, ok = arm.pick_targets(a, target_z - 0.07, target_z, standoff)
    assert ok
    names = ("r_shoulder", "r_elbow")
    jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in names]
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "r_hand")
    assert sid >= 0 and min(jids) >= 0
    target = np.array([standoff, sh[1], target_z])
    data.qpos[model.jnt_qposadr[jids[0]]] = qs
    data.qpos[model.jnt_qposadr[jids[1]]] = qe
    mujoco.mj_kinematics(model, data)
    err = data.site_xpos[sid] - target
    print("analytic IK err xyz", err)
    if math.hypot(err[0], err[2]) >= 0.05:
        arm.refine_ik(model, data, jids, sid, target)
        err = data.site_xpos[sid] - target
        print("refined IK err xyz", err)
    assert math.hypot(err[0], err[2]) < 0.05
