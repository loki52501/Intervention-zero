"""Parametric gait engine for the Unitree G1.

HONEST CLAIM: this is a parametric gait engine (sinusoidal joint targets), NOT a learned
walking policy. The pelvis is held by the simulator; learning in Intervention Zero happens
at behaviour, environment and fleet level, not in locomotion.

G1 sign conventions (joint axes 0 1 0): hip_pitch NEGATIVE swings the leg forward, knee
POSITIVE flexes, ankle_pitch POSITIVE is toes down. Flat foot: ankle = -(hip + knee).
Pure math: no numpy or MuJoCo needed.
"""
from __future__ import annotations

import math

STOP_POSE = {
    "left_hip_pitch_joint": -0.31, "left_knee_joint": 0.67, "left_ankle_pitch_joint": -0.36,
    "left_hip_roll_joint": 0.0,
    "right_hip_pitch_joint": -0.31, "right_knee_joint": 0.67, "right_ankle_pitch_joint": -0.36,
    "right_hip_roll_joint": 0.0,
}
LEG_LEN = 0.6
CADENCE_BOUNDS = (1.0, 2.4)
STRIDE_SCALE_BOUNDS = (0.4, 1.2)
DEFAULTS = {"gait_cadence_hz": 1.6, "stride_scale": 0.8}
MAX_STRIDE = 0.6
STOP_BLEND_RATE = 4.0  # 1/s


def _clamp(x, lo, hi):
    return max(lo, min(hi, x))


class GaitEngine:
    def __init__(self, params: dict | None = None, suffix: str = ""):
        p = dict(DEFAULTS)
        p.update(params or {})
        self.cadence = _clamp(float(p["gait_cadence_hz"]), *CADENCE_BOUNDS)
        self.stride_scale = _clamp(float(p["stride_scale"]), *STRIDE_SCALE_BOUNDS)
        self.suffix = suffix
        self.phase = 0.0
        self.stride = 0.0
        self.stopped = True
        self._targets = dict(STOP_POSE)
        self._arms = {"l_shoulder": 0.0, "r_shoulder": 0.0}

    @staticmethod
    def _leg(side: str, phase: float, A: float, s: float) -> dict:
        hip = STOP_POSE[f"{side}_hip_pitch_joint"] - s * A * math.sin(phase)
        # knee lift in swing; amplitude 0.7*min(A, 0.25) (spec 1.2*A would push the flat-foot ankle past
        # its -0.87 limit at v=1.0)
        knee = STOP_POSE[f"{side}_knee_joint"] + 0.7 * min(A, 0.25) * max(0.0, math.sin(phase + math.pi / 2))
        return {
            f"{side}_hip_pitch_joint": hip,
            f"{side}_knee_joint": knee,
            f"{side}_ankle_pitch_joint": -(hip + knee),
            f"{side}_hip_roll_joint": 0.04 * math.sin(phase),
        }

    def step(self, dt: float, v_cmd: float, w_cmd: float) -> dict:
        speed = abs(v_cmd) + 0.25 * abs(w_cmd)
        if speed < 0.05:
            self.stopped = True
            self.stride = 0.0
            k = min(1.0, STOP_BLEND_RATE * dt)
            for key, tgt in STOP_POSE.items():
                self._targets[key] += (tgt - self._targets[key]) * k
            self._arms = {"l_shoulder": 0.0, "r_shoulder": 0.0}
        else:
            self.stopped = False
            self.phase += 2 * math.pi * self.cadence * dt
            self.stride = _clamp(speed / self.cadence * self.stride_scale, 0.0, MAX_STRIDE)
            A = math.atan2(self.stride / 2, LEG_LEN)
            s = -1.0 if v_cmd < 0 else 1.0
            self._targets = {**self._leg("left", self.phase, A, s),
                             **self._leg("right", self.phase + math.pi, A, s)}
            sw = 0.5 * A * math.sin(self.phase)
            self._arms = {"l_shoulder": sw, "r_shoulder": -sw}
        out = {**self._targets, **self._arms}
        return {k + self.suffix: float(v) for k, v in out.items()}
