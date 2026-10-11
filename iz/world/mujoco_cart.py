"""MuJoCo cart micro-world: a tall cart (free body) pushed from behind by a planar pusher.

Calibration: friction mu = move_force_n / (m g), so the cart slides once the push exceeds move_force_n.
Tipping about the leading bottom edge needs F*(h - hc) > m*g*(d - mu*hc) (d = half depth, hc = CoM height,
h = push height). We pick the footprint depth d and the push height h from the params so that threshold equals
tip_force_n (an empirical correction factor TIP_CORR absorbs dynamic effects).
Deterministic: fixed timestep and initial state.
"""
from __future__ import annotations

import math

import mujoco
import numpy as np

from iz.world.mujoco_door import MAX_FORCE_N, DT, PhysicsResult, Recorder

G = 9.81
HC = 0.6                # CoM height (uniform 1.2 m tall box)
CART_H = 1.2
HALF_W = 0.25
PUSH_V = 3.0
TILT_DEG = 25.0
TIP_CORR = 1.15
PUSHER_START_GAP = 0.003


def _geometry(params: dict) -> dict:
    m = float(params.get("cart_mass_kg", 20.0))
    move = float(params.get("move_force_n", 25.0))
    tip = float(params.get("tip_force_n", move + 25.0))
    mu = move / (m * G)
    d = min(max(tip * (1.0 - HC) / (m * G) + mu * HC, 0.06), 0.35)
    h = HC + TIP_CORR * m * G * (d - mu * HC) / tip
    h = min(max(h, 0.25), CART_H - 0.05)
    return dict(m=m, mu=mu, d=d, h=h, move=move, tip=tip)


def build_cart_xml(params: dict) -> str:
    g = _geometry(params)
    d, h = g["d"], g["h"]
    px = -d - PUSHER_START_GAP - 0.08
    return f"""<mujoco model="cart">
  <compiler angle="radian"/>
  <option timestep="{DT / 2}" gravity="0 0 -{G}" cone="elliptic" impratio="20"/>
  <visual><global offwidth="640" offheight="480"/></visual>
  <worldbody>
    <light pos="-2 -3 5" dir="0.3 0.5 -1" diffuse="0.9 0.9 0.9"/>
    <geom name="floor" type="plane" size="8 8 0.1" rgba="0.78 0.78 0.74 1" friction="0.01 0.005 0.0001"/>
    <body name="cart" pos="0 0 0.001">
      <freejoint name="cart_free"/>
      <geom name="cart_box" type="box" pos="0 0 {HC}" size="{d} {HALF_W} {CART_H / 2}" mass="{g['m']}"
            friction="{g['mu']} 0.005 0.0001" rgba="0.85 0.7 0.2 1"/>
    </body>
    <body name="pusher" pos="{px} 0 {h}">
      <joint name="px" type="slide" axis="1 0 0"/>
      <geom name="pusher_box" type="box" size="0.08 0.2 0.05" mass="5" friction="0 0 0" priority="1"
            rgba="0.15 0.45 0.85 1"/>
    </body>
  </worldbody>
  <actuator>
    <velocity name="drive_x" joint="px" kv="4000" ctrllimited="true" ctrlrange="-5 5" forcelimited="true" forcerange="-200 200"/>
  </actuator>
</mujoco>"""


def simulate_cart(params: dict, force_n: float, duration_s: float = 4.0, max_force_n: float = MAX_FORCE_N,
                  record_path: str | None = None) -> PhysicsResult:
    g = _geometry(params)
    f = min(max(float(force_n), 0.0), 200.0)
    model = mujoco.MjModel.from_xml_string(build_cart_xml(params))
    data = mujoco.MjData(model)
    model.actuator_forcerange[0] = [-f, f]
    data.ctrl[0] = PUSH_V
    cart_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cart")
    mujoco.mj_forward(model, data)
    x0 = float(data.xpos[cart_id][0])
    rec = Recorder(model, lookat=(0.5, 0.0, 0.5), distance=5.5, azimuth=65, elevation=-18) if record_path else None

    peak, max_tilt, tipped, disp = 0.0, 0.0, False, 0.0
    n_steps = int(duration_s / model.opt.timestep)
    for _ in range(n_steps):
        mujoco.mj_step(model, data)
        peak = max(peak, abs(float(data.actuator_force[0])))
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, float(data.xmat[cart_id][8])))))
        max_tilt = max(max_tilt, tilt)
        disp = float(data.xpos[cart_id][0]) - x0
        if rec is not None:
            rec.capture(data)
        if tilt > TILT_DEG:
            tipped = True
            if rec is None:
                break
        if rec is not None and (disp >= 1.6 or (tipped and tilt > 80.0)):
            break  # clip is long enough
    t_end = float(data.time)
    extra = {"displacement_m": disp, "max_tilt_deg": max_tilt, "tipped": tipped}
    if tipped:
        success, reason = False, f"cart tipped at {max_tilt:.0f} deg"
    elif peak > max_force_n:
        success, reason = False, f"force {peak:.0f} N over limit"
    elif disp >= 0.8:
        success, reason = True, f"moved {disp:.1f} m upright"
    else:
        success, reason = False, f"cart did not move ({disp:.2f} m)" if disp < 0.1 else f"cart moved only {disp:.2f} m"
    clip = None
    if rec is not None:
        rec.capture(data, force=True)
        clip = rec.save(record_path)
        if clip is None:
            extra["render_error"] = rec.error or "no frames"
    return PhysicsResult(success=bool(success), time_s=round(t_end, 3), peak_force_n=round(peak, 3),
                         reason=reason, clip_path=clip, extra=extra)


def probe_push(params: dict, force_n: float = 10.0) -> float:
    """Displacement (m) of the cart after a 1.0 s push at force_n."""
    return float(simulate_cart(params, force_n, duration_s=1.0).extra["displacement_m"])
