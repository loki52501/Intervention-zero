"""Render a 3 s walk clip of the G1 puppet driven by the PARAMETRIC gait engine.

HONEST CLAIM: locomotion here is a parametric gait engine (iz/robot/gait.py), not a learned
walking policy. Usage: .venv\\Scripts\\python.exe scripts\\g1_walk.py
"""
from __future__ import annotations

import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def main() -> int:
    import imageio
    import mujoco
    import numpy as np

    from iz.robot import g1
    from iz.robot.gait import GaitEngine

    model = mujoco.MjModel.from_xml_string(g1.build_standalone_xml((0,)))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, 480, 640)
    cam = "cam_follow"

    def act(name):
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)

    a_vx = act("act_vx")
    leg_ids = {j: act("act_" + j) for j in g1.LEG_JOINTS}
    arm_ids = {k: act("act_" + k) for k in ("l_shoulder", "r_shoulder")}
    assert a_vx >= 0 and min(leg_ids.values()) >= 0 and min(arm_ids.values()) >= 0

    eng = GaitEngine()
    dur, tick, fps, ramp, v_target = 3.0, 0.01, 30, 0.3, 1.0
    mujoco.mj_forward(model, data)
    x0 = float(data.qpos[model.joint("base_x").qposadr[0]])
    pelvis = model.body("pelvis").id
    zs, nan, frames, preview = [], False, [], None
    next_frame, n_ticks = 0.0, int(round(dur / tick))
    sub = max(1, int(round(tick / model.opt.timestep)))
    for i in range(n_ticks):
        t = i * tick
        v = v_target * min(1.0, t / ramp)
        for k, val in eng.step(tick, v, 0.0).items():
            aid = arm_ids[k] if k in arm_ids else leg_ids[k]
            data.ctrl[aid] = val
        data.ctrl[a_vx] = v
        for _ in range(sub):
            mujoco.mj_step(model, data)
        if not np.all(np.isfinite(data.qpos)):
            nan = True
            break
        zs.append(float(data.xpos[pelvis][2]))
        if data.time + 1e-9 >= next_frame:
            renderer.update_scene(data, camera=cam)
            frames.append(renderer.render().copy())
            if preview is None and data.time >= 1.5:  # mid-stride
                preview = frames[-1]
            next_frame += 1.0 / fps

    os.makedirs(os.path.join(ROOT, "out", "clips"), exist_ok=True)
    mp4 = os.path.join(ROOT, "out", "clips", "g1_walk.mp4")
    with imageio.get_writer(mp4, fps=fps, codec="libx264", macro_block_size=1, quality=8) as w:
        for f in frames:
            w.append_data(f)
    png = os.path.join(ROOT, "out", "g1_preview.png")
    imageio.imwrite(png, preview if preview is not None else frames[-1])
    renderer.close()
    x1 = float(data.qpos[model.joint("base_x").qposadr[0]])
    print(f"pelvis z min={min(zs):.4f} max={max(zs):.4f} drift={max(zs) - min(zs):.4f} m")
    print(f"distance travelled (base_x)={x1 - x0:.3f} m in {data.time:.2f} s; NaN={nan}")
    print(f"frames={len(frames)} mp4={mp4} png={png}")
    return 1 if nan else 0


if __name__ == "__main__":
    sys.exit(main())
