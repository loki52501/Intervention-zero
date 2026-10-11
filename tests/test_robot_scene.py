"""Robot body (capsule vs Unitree G1) behind ROBOT_BODY in the shop scene + worldgen."""
import mujoco
import numpy as np
import pytest

from iz.shop.controller import DEFAULT_PARAMS, PARAM_BOUNDS, LOCKED, clamp_params
from iz.shop.scene import build_shop
from tests.test_shop_scene import ACTS, BODIES, CAMS, EQS, JOINTS, SENSORS, SITES

OBJ = mujoco.mjtObj


def _g1_ok():
    try:
        from iz.robot import g1_available
        return bool(g1_available())
    except Exception:
        return False


MODES = ["capsule", pytest.param("g1", marks=pytest.mark.skipif(not _g1_ok(), reason="g1 unavailable"))]
G1_ONLY = pytest.mark.skipif(not _g1_ok(), reason="g1 unavailable")


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("seed", range(3))
def test_scene_compiles_and_runs(monkeypatch, mode, seed):
    monkeypatch.setenv("ROBOT_BODY", mode)
    xml, meta = build_shop(seed)
    assert meta["robot_body"] == mode
    m = mujoco.MjModel.from_xml_string(xml)
    for t, names in [(OBJ.mjOBJ_BODY, BODIES), (OBJ.mjOBJ_JOINT, JOINTS), (OBJ.mjOBJ_ACTUATOR, ACTS),
                     (OBJ.mjOBJ_SENSOR, SENSORS), (OBJ.mjOBJ_SITE, SITES), (OBJ.mjOBJ_CAMERA, CAMS),
                     (OBJ.mjOBJ_EQUALITY, EQS)]:
        for n in names:
            assert mujoco.mj_name2id(m, t, n) >= 0, n
    d = mujoco.MjData(m)
    for _ in range(500):
        mujoco.mj_step(m, d)
    assert np.all(np.isfinite(d.qpos)) and np.all(np.isfinite(d.sensordata))


@G1_ONLY
def test_g1_meta(monkeypatch):
    monkeypatch.setenv("ROBOT_BODY", "g1")
    _, meta = build_shop(0)
    assert meta["robot_body"] == "g1" and "arm" in meta
    assert abs(meta["place_yaw"]) < 1e-9
    assert abs(abs(meta["pick_yaw"]) - np.pi / 2) < 1e-9
    from iz.shop.scene import robot_header_parts
    comp, default_xml, asset_xml = robot_header_parts()
    assert "meshdir" in comp and default_xml and asset_xml


def test_capsule_header_parts_empty(monkeypatch):
    monkeypatch.setenv("ROBOT_BODY", "capsule")
    from iz.shop.scene import robot_header_parts
    assert robot_header_parts() == ("", "", "")


@G1_ONLY
def test_g1_episode_drives_legs(monkeypatch):
    monkeypatch.setenv("ROBOT_BODY", "g1")
    from iz.shop import controller as C
    from iz.shop.episode import run_episode
    seen = {}
    orig = C.ShopController.step

    def spy(self, t):
        orig(self, t)
        if self._gait_acts:
            seen.setdefault("ids", list(self._gait_acts.values()))
            seen.setdefault("vals", []).append(np.array([self.data.ctrl[i] for i in seen["ids"]]))
    monkeypatch.setattr(C.ShopController, "step", spy)
    res = run_episode(dict(DEFAULT_PARAMS), 0, max_time_s=15)
    assert res is not None
    assert seen.get("ids"), "no leg actuators found"
    vals = np.array(seen["vals"])
    assert float(np.ptp(vals, axis=0).max()) > 0.05


@G1_ONLY
def test_g1_world_builds_and_checks(monkeypatch):
    monkeypatch.setenv("ROBOT_BODY", "g1")
    from iz.worldgen.build import build_world, check_world
    from tests.test_worldgen_build import _small_store
    xml, meta = build_world(_small_store(), 0)
    mujoco.MjModel.from_xml_string(xml)
    assert check_world(xml, meta) == []


def test_gait_params_clamp():
    assert clamp_params({"gait_cadence_hz": 9})[0]["gait_cadence_hz"] == 2.4
    assert clamp_params({"stride_scale": 0.0})[0]["stride_scale"] == 0.4
    assert PARAM_BOUNDS["gait_cadence_hz"] == (1.0, 2.4) and PARAM_BOUNDS["stride_scale"] == (0.4, 1.2)
    assert "gait_cadence_hz" not in LOCKED and "stride_scale" not in LOCKED
    assert DEFAULT_PARAMS["gait_cadence_hz"] == 1.6 and DEFAULT_PARAMS["stride_scale"] == 0.8
