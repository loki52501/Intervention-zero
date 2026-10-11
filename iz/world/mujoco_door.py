"""MuJoCo door micro-world.

Geometry: wall along y at x=0 with a doorway; robot comes from -x and pushes +x. The door leaf hangs on a
vertical hinge at the -y side of the doorway. Swinging the leaf toward +x is a NEGATIVE hinge angle, so the
hinge range is [-1.7, 0] rad (a 0..1.6 range blocks the door - spike finding).

Peak force convention: max |actuator_force| of the robot x-drive over steps where the robot geom touches the
door leaf (free-motion force is ignored; it is not a contact force).
Deterministic: fixed timestep, fixed initial state, no randomness.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import mujoco

from iz.types import Action, LockedGuardrails

MAX_FORCE_N: float = float(LockedGuardrails().max_contact_force_n)

DT = 0.004
ROBOT_START_X = -0.35   # robot front face starts ~0.1 m from the closed leaf (no run-up momentum)
CROSS_X = 0.6
DRIVE_V = 0.6
STALL_S = 1.2           # robot gives up after pushing at its force limit (no free motion) this long after touching
MAX_WAIT_S = 30.0
# hinge stiffness k = (min_push_n + K_OFFSET) / K_SLOPE (N*m/rad): empirical fit so that a push succeeds
# iff force >= ~min_push_n (measured threshold within ~7% of min_push_n for office 15-45 and fire 120-200)
K_SLOPE, K_OFFSET = 1.64, 8.5


@dataclass(frozen=True)
class PhysicsResult:
    success: bool
    time_s: float
    peak_force_n: float
    reason: str
    clip_path: str | None = None
    extra: dict = field(default_factory=dict)


class Recorder:
    """Captures frames every 1/fps of sim time and writes an mp4. Never raises out of capture/save."""

    def __init__(self, model, lookat, distance, azimuth, elevation, fps=30):
        self.model, self.fps, self.frames, self.next_t, self.error = model, fps, [], 0.0, None
        self.cam = mujoco.MjvCamera()
        self.cam.lookat[:] = lookat
        self.cam.distance, self.cam.azimuth, self.cam.elevation = distance, azimuth, elevation
        try:
            self.renderer = mujoco.Renderer(model, 480, 640)
        except Exception as e:  # noqa: BLE001
            self.renderer, self.error = None, repr(e)

    def capture(self, data, force=False):
        if self.renderer is None or (not force and data.time + 1e-9 < self.next_t):
            return
        try:
            self.renderer.update_scene(data, camera=self.cam)
            self.frames.append(self.renderer.render().copy())
            self.next_t += 1.0 / self.fps
        except Exception as e:  # noqa: BLE001
            self.renderer, self.error = None, repr(e)

    def save(self, path: str) -> str | None:
        try:
            if self.renderer is None or not self.frames:
                return None
            import os

            import imageio
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            with imageio.get_writer(path, fps=self.fps, macro_block_size=1) as w:
                for f in self.frames + [self.frames[-1]] * self.fps:
                    w.append_data(f)
            return path
        except Exception as e:  # noqa: BLE001
            self.error = repr(e)
            return None
        finally:
            try:
                if self.renderer is not None:
                    self.renderer.close()
            except Exception:  # noqa: BLE001
                pass


def _geometry(params: dict) -> dict:
    w = float(params.get("door_width_m", 0.9))
    mass = float(params.get("door_mass_kg", 8.0))
    push = float(params.get("min_push_n", 30.0))
    k = (push + K_OFFSET) / K_SLOPE
    inertia = mass * w * w / 3.0
    damping = 0.5 * math.sqrt(k * inertia)
    return dict(w=w, mass=mass, k=k, damping=damping, robot_y=min(-w / 2 + 0.65, w / 2 - 0.24),
                opener=1.8 * k + 50.0, fire=params.get("door_type", "office") == "fire")


def build_door_xml(params: dict) -> str:
    g = _geometry(params)
    half = g["w"] / 2
    leaf_rgba = "0.75 0.12 0.10 1" if g["fire"] else "0.55 0.38 0.22 1"
    wall_len = 3.0
    wall_c = half + wall_len / 2
    return f"""<mujoco model="door">
  <compiler angle="radian"/>
  <option timestep="{DT}" gravity="0 0 -9.81"/>
  <visual><global offwidth="640" offheight="480"/></visual>
  <worldbody>
    <light pos="-2 -3 5" dir="0.3 0.5 -1" diffuse="0.9 0.9 0.9"/>
    <geom name="floor" type="plane" size="6 6 0.1" rgba="0.78 0.78 0.74 1" contype="0" conaffinity="0"/>
    <geom name="wall_hi" type="box" pos="0 {wall_c} 1" size="0.05 {wall_len / 2} 1" rgba="0.85 0.85 0.88 1" contype="1" conaffinity="1"/>
    <geom name="wall_lo" type="box" pos="0 {-wall_c} 1" size="0.05 {wall_len / 2} 1" rgba="0.85 0.85 0.88 1" contype="1" conaffinity="1"/>
    <body name="door" pos="0 {-half} 0">
      <joint name="hinge" type="hinge" axis="0 0 1" range="-1.7 0" limited="true"
             stiffness="{g['k']}" springref="0" damping="{g['damping']}" armature="0.01"/>
      <geom name="leaf" type="box" pos="0 {half} 1" size="0.03 {half} 1" mass="{g['mass']}"
            rgba="{leaf_rgba}" contype="2" conaffinity="2"/>
    </body>
    <body name="robot" pos="{ROBOT_START_X} {g['robot_y']} 0">
      <joint name="rx" type="slide" axis="1 0 0"/>
      <joint name="ry" type="slide" axis="0 1 0" damping="200"/>
      <joint name="ryaw" type="hinge" axis="0 0 1" damping="50"/>
      <geom name="rgeom" type="cylinder" size="0.22 0.4" pos="0 0 0.4" mass="25"
            rgba="0.15 0.45 0.85 1" contype="3" conaffinity="3"/>
      <geom name="rnose" type="box" pos="0.2 0 0.55" size="0.04 0.1 0.04" rgba="1 1 1 1" contype="0" conaffinity="0" mass="0.01"/>
    </body>
  </worldbody>
  <actuator>
    <velocity name="drive_x" joint="rx" kv="4000" ctrllimited="true" ctrlrange="-2 2" forcelimited="true" forcerange="-200 200"/>
    <position name="hold_y" joint="ry" kp="20000"/>
    <position name="hold_yaw" joint="ryaw" kp="5000"/>
    <motor name="opener" joint="hinge" gear="1" ctrllimited="true" ctrlrange="-5000 5000"/>
  </actuator>
</mujoco>"""


def _clamp_force(v, default):
    try:
        f = float(default if v is None else v)
    except (TypeError, ValueError):
        f = float(default)
    return min(max(f, 0.0), 200.0)


def _clamp_wait(v):
    try:
        s = float(5.0 if v is None else v)
    except (TypeError, ValueError):
        s = 5.0
    return min(max(s, 0.0), MAX_WAIT_S)


def simulate_door(params: dict, actions: list[Action], max_force_n: float = MAX_FORCE_N,
                  record_path: str | None = None) -> PhysicsResult:
    g = _geometry(params)
    delay = float(params.get("open_delay_s", 5.0))
    model = mujoco.MjModel.from_xml_string(build_door_xml(params))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    obj = mujoco.mjtObj
    aid = {n: mujoco.mj_name2id(model, obj.mjOBJ_ACTUATOR, n) for n in ("drive_x", "hold_y", "hold_yaw", "opener")}
    gid = {n: mujoco.mj_name2id(model, obj.mjOBJ_GEOM, n) for n in ("rgeom", "leaf")}
    jx = mujoco.mj_name2id(model, obj.mjOBJ_JOINT, "rx")
    qx, vx = model.jnt_qposadr[jx], model.jnt_dofadr[jx]
    qh = model.jnt_qposadr[mujoco.mj_name2id(model, obj.mjOBJ_JOINT, "hinge")]
    data.ctrl[aid["hold_y"]] = 0.0   # slide qpos is relative to the body's initial pose
    data.ctrl[aid["hold_yaw"]] = 0.0

    rec = None
    if record_path:
        rec = Recorder(model, lookat=(0.2, 0.0, 0.8), distance=5.0, azimuth=60, elevation=-25)

    st = {"t_req": None, "peak": 0.0, "crossed": None, "min_angle": 0.0}

    def run(duration, drive, fmax):
        """Advance sim. drive: target x-velocity (0 = hold still)."""
        model.actuator_forcerange[aid["drive_x"]] = [-fmax, fmax]
        data.ctrl[aid["drive_x"]] = drive
        end_t = data.time + duration
        sat, contacted = 0.0, False
        while data.time < end_t - 1e-9:
            if st["t_req"] is not None and data.time >= st["t_req"] + delay:
                data.ctrl[aid["opener"]] = -g["opener"]
            mujoco.mj_step(model, data)
            touching = False
            for i in range(data.ncon):
                c = data.contact[i]
                if (c.geom1 == gid["rgeom"] and c.geom2 == gid["leaf"]) or \
                   (c.geom2 == gid["rgeom"] and c.geom1 == gid["leaf"]):
                    touching = True
                    break
            if touching:
                st["peak"] = max(st["peak"], abs(float(data.actuator_force[aid["drive_x"]])))
            st["min_angle"] = min(st["min_angle"], float(data.qpos[qh]))
            if rec is not None:
                rec.capture(data)
            xr = float(data.qpos[qx])
            if st["crossed"] is None and ROBOT_START_X + xr > CROSS_X:
                st["crossed"] = float(data.time)
                if drive > 0:
                    return
            if drive > 0:
                contacted = contacted or touching
                if contacted and abs(float(data.actuator_force[aid["drive_x"]])) >= 0.95 * fmax:
                    sat += DT
                    if sat >= STALL_S:
                        return  # pushing flat-out against a door that does not yield: give up
                else:
                    sat = 0.0

    for a in actions:
        if st["crossed"] is not None:
            break
        if a.tool == "request_open":
            if st["t_req"] is None:
                st["t_req"] = float(data.time)
        elif a.tool == "wait":
            run(_clamp_wait(a.args.get("seconds")), 0.0, 0.0)
        elif a.tool == "push_door":
            run(6.0, DRIVE_V, _clamp_force(a.args.get("force_n"), 30.0))
        # any other tool: no-op
    if st["crossed"] is None:
        run(6.0, DRIVE_V, 20.0)  # final gentle 'proceed'

    peak = st["peak"]
    crossed = st["crossed"] is not None
    x = ROBOT_START_X + float(data.qpos[qx])
    angle = st["min_angle"]
    if crossed and peak > max_force_n:
        success, reason = False, f"force {peak:.0f} N over limit"
    elif crossed:
        success, reason = True, f"crossed at {st['crossed']:.1f} s"
    else:
        success, reason = False, f"door closer held, robot stalled at x={x:.2f}"
        if peak > max_force_n:
            reason += f" (force {peak:.0f} N over limit)"
    t_end = float(st["crossed"]) if crossed else float(data.time)
    extra = {"door_angle_rad": angle, "door_tip_x_m": g["w"] * math.sin(-angle), "robot_x_m": x,
             "crossed": crossed}
    clip = None
    if rec is not None:
        rec.capture(data, force=True)
        clip = rec.save(record_path)
        if clip is None:
            extra["render_error"] = rec.error or "no frames"
    return PhysicsResult(success=bool(success), time_s=round(t_end, 3), peak_force_n=round(peak, 3),
                         reason=reason, clip_path=clip, extra=extra)
