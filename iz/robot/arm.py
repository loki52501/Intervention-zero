"""Sagittal-plane 2-link arm IK for the G1 shoulder-pitch / elbow chain.

Pure math (plus an optional MuJoCo-driven numeric refinement that receives the model and
data as arguments; MuJoCo is never imported here). This is analytic geometry, not learned.

Convention: x forward, z up, target relative to the shoulder. Link angle th1 is measured
from -z toward +x. +q_shoulder rotates the upper link toward +x. +q_elbow rotates the
forearm further toward +x/up (elbow-down solution). th = q + a0 (a0 = joint offset).
"""
from __future__ import annotations

import math


def forward(q_sh: float, q_el: float, l1: float, l2: float,
            a1_0: float = 0.0, a2_0: float = 0.0) -> tuple[float, float]:
    th1 = q_sh + a1_0
    th12 = th1 + q_el + a2_0
    fwd = l1 * math.sin(th1) + l2 * math.sin(th12)
    z = -l1 * math.cos(th1) - l2 * math.cos(th12)
    return fwd, z


def two_link_ik(fwd: float, z: float, l1: float, l2: float,
                a1_0: float = 0.0, a2_0: float = 0.0) -> tuple[float, float, bool]:
    d = math.hypot(fwd, z)
    reachable = True
    d_max = (l1 + l2) * 0.999
    d_min = abs(l1 - l2) * 1.001
    if d > d_max or d < d_min:
        reachable = False
        if d < 1e-9:
            fwd, z, d = 0.0, -1.0, 1.0  # arbitrary direction: straight down
        tgt = d_max if d > d_max else d_min
        fwd, z, d = fwd * tgt / d, z * tgt / d, tgt
    c2 = (d * d - l1 * l1 - l2 * l2) / (2 * l1 * l2)
    c2 = max(-1.0, min(1.0, c2))
    th2 = math.acos(c2)  # >= 0: forearm forward of upper link
    phi = math.atan2(fwd, -z)
    th1 = phi - math.atan2(l2 * math.sin(th2), l1 + l2 * math.cos(th2))
    return th1 - a1_0, th2 - a2_0, reachable


def pick_targets(meta_arm: dict, ledge_z: float, reach_height_m: float | None,
                 standoff: float) -> tuple[float, float, bool]:
    """Joint targets to put the hand at (standoff, reach height) in the robot frame."""
    sx, _sy, sz = meta_arm["shoulder_xyz"]
    z_world = reach_height_m if reach_height_m is not None else ledge_z + 0.07
    return two_link_ik(standoff - sx, z_world - sz, meta_arm["l1"], meta_arm["l2"],
                       meta_arm.get("a1_0", 0.0), meta_arm.get("a2_0", 0.0))


def refine_ik(model, data, joint_ids, site_id, target_xyz, span: float = 0.5):
    """Numeric 2-variable refinement (no scipy): 15x15 grid around the current joint
    values, then a 7x7 refine. Minimises the sagittal (x, z) error plus a small y term.
    Leaves the best joint values in data.qpos and returns (q_sh, q_el)."""
    import numpy as np
    import mujoco

    adr = [int(model.jnt_qposadr[j]) for j in joint_ids]
    tgt = np.asarray(target_xyz, dtype=float)

    def cost(a, b):
        data.qpos[adr[0]], data.qpos[adr[1]] = a, b
        mujoco.mj_kinematics(model, data)
        e = data.site_xpos[site_id] - tgt
        return e[0] ** 2 + e[2] ** 2 + 0.05 * e[1] ** 2

    c0 = (float(data.qpos[adr[0]]), float(data.qpos[adr[1]]))
    best = (cost(*c0), c0)
    for n, half in ((15, span), (7, span / 7.0)):
        centre = best[1]
        for a in np.linspace(centre[0] - half, centre[0] + half, n):
            for b in np.linspace(centre[1] - half, centre[1] + half, n):
                c = cost(a, b)
                if c < best[0]:
                    best = (c, (float(a), float(b)))
    cost(*best[1])
    return best[1]
