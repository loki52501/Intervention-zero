"""Tests for the parametric gait engine (not a learned walking policy)."""
import math

import numpy as np

from iz.robot.gait import (CADENCE_BOUNDS, DEFAULTS, STOP_POSE, STRIDE_SCALE_BOUNDS,
                           GaitEngine)

RANGES = {"hip_pitch": (-2.5, 2.8), "knee": (-0.08, 2.8), "ankle_pitch": (-0.87, 0.52)}


def run(eng, v, seconds, w=0.0, dt=0.01):
    return [eng.step(dt, v, w) for _ in range(int(round(seconds / dt)))]


def test_stop_pose_when_idle():
    eng = GaitEngine()
    out = run(eng, 0.0, 1.0)[-1]
    for k, val in STOP_POSE.items():
        assert abs(out[k] - val) < 1e-6
    assert out["l_shoulder"] == 0.0 and out["r_shoulder"] == 0.0
    assert eng.stopped and eng.phase == 0.0


def test_walk_oscillates_antiphase_flat_foot_and_ranges():
    eng = GaitEngine({"gait_cadence_hz": 1.6})
    outs = run(eng, 1.0, 2.0)
    lh = np.array([o["left_hip_pitch_joint"] for o in outs])
    rh = np.array([o["right_hip_pitch_joint"] for o in outs])
    assert lh.max() - lh.min() > 0.2
    assert np.corrcoef(lh, rh)[0, 1] < -0.8
    assert not eng.stopped
    for o in outs:
        for side in ("left", "right"):
            h, k, a = (o[f"{side}_hip_pitch_joint"], o[f"{side}_knee_joint"],
                       o[f"{side}_ankle_pitch_joint"])
            assert abs(a + (h + k)) < 1e-9
        for key, val in o.items():
            assert math.isfinite(val)
            for name, (lo, hi) in RANGES.items():
                if name in key:
                    assert lo <= val <= hi, (key, val)
    ls = np.array([o["l_shoulder"] for o in outs])
    assert abs(np.corrcoef(ls, lh)[0, 1]) > 0.8  # arm swings in opposition to same-side leg


def test_stride_grows_and_caps():
    strides = []
    for v in (0.2, 0.5, 1.0, 5.0):
        e = GaitEngine()
        run(e, v, 0.5)
        strides.append(e.stride)
    assert strides[0] < strides[1] < strides[2]
    assert abs(strides[3] - 0.6) < 1e-9


def test_params_clamped():
    e = GaitEngine({"gait_cadence_hz": 99, "stride_scale": -3})
    assert e.cadence == CADENCE_BOUNDS[1] and e.stride_scale == STRIDE_SCALE_BOUNDS[0]
    e = GaitEngine({"gait_cadence_hz": 0.0, "stride_scale": 50})
    assert e.cadence == CADENCE_BOUNDS[0] and e.stride_scale == STRIDE_SCALE_BOUNDS[1]
    assert GaitEngine().cadence == DEFAULTS["gait_cadence_hz"]


def test_suffix_and_turn_in_place():
    e = GaitEngine(suffix="_1")
    o = e.step(0.01, 0.0, 1.0)
    assert all(k.endswith("_1") for k in o)
    assert "l_shoulder_1" in o and "left_knee_joint_1" in o and len(o) == 10
    assert not e.stopped
