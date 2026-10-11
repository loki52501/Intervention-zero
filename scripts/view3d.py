"""Live 3D MuJoCo viewer (native window).

  .venv\\Scripts\\python.exe scripts\\view3d.py building --site B --robots 4
  .venv\\Scripts\\python.exe scripts\\view3d.py door --door fire --policy before   # push_door: closer wins
  .venv\\Scripts\\python.exe scripts\\view3d.py door --door fire --policy after    # request_open + wait: crosses

Building mode is visual (mocap robots following shortest paths); door mode is real physics
from iz/world/mujoco_door.py. Mouse: drag to orbit, right-drag to pan, scroll to zoom.
"""
from __future__ import annotations

import argparse
import math
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco  # noqa: E402
import mujoco.viewer  # noqa: E402

from iz.world import mujoco_door  # noqa: E402
from iz.world.scene3d import build_site_xml  # noqa: E402
from iz.world.sites import get_site, shortest_path  # noqa: E402

COLORS = ["0.1 0.45 0.9 1", "0.1 0.7 0.35 1", "0.9 0.55 0.1 1", "0.6 0.25 0.85 1", "0.9 0.2 0.5 1", "0.2 0.7 0.8 1"]


def _robots_xml(n: int) -> str:
    return "".join(
        f'<body name="bot{i}" mocap="true" pos="0 0 0.35">'
        f'<geom type="cylinder" size="0.3 0.3" rgba="{COLORS[i % len(COLORS)]}"/>'
        f'<geom type="box" pos="0.25 0 0.2" size="0.08 0.15 0.05" rgba="1 1 1 1"/></body>'
        for i in range(n))


def building(site_id: str, n_robots: int, speed: float) -> None:
    site = get_site(site_id)
    xml = build_site_xml(site)
    xml = xml.replace("</worldbody>", _robots_xml(n_robots) + "</worldbody>")
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    zones = {z.id: z for z in site.zones}
    walkable = [z.id for z in site.zones if z.kind != "human_only"]
    rng = random.Random(0)
    first = 1  # mocap index 0 is the scene's own robot; ours start at 1
    plans = []
    for i in range(n_robots):
        start = walkable[i % len(walkable)]
        plans.append({"path": [start], "seg": 0, "t": 0.0})

    def new_route(p):
        a = p["path"][-1]
        for _ in range(10):
            b = rng.choice(walkable)
            if b != a:
                path = shortest_path(site, a, b)
                if path and len(path) > 1:
                    p.update(path=path, seg=0, t=0.0)
                    return
        p.update(path=[a, a], seg=0, t=0.0)

    for p in plans:
        new_route(p)
    data.mocap_pos[0] = (0, 0, -5)  # hide the scene's default robot

    with mujoco.viewer.launch_passive(model, data) as v:
        xs = [z.xy[0] for z in site.zones]
        ys = [z.xy[1] for z in site.zones]
        v.cam.lookat[:] = ((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, 0)
        v.cam.distance = max(max(xs) - min(xs), max(ys) - min(ys), 6) * 1.3 + 4
        v.cam.elevation, v.cam.azimuth = -55, 90
        last = time.time()
        while v.is_running():
            now = time.time()
            dt, last = now - last, now
            for i, p in enumerate(plans):
                a, b = zones[p["path"][p["seg"]]].xy, zones[p["path"][p["seg"] + 1]].xy
                seg_len = math.hypot(b[0] - a[0], b[1] - a[1]) or 1.0
                p["t"] += dt * speed / seg_len
                if p["t"] >= 1.0:
                    p["seg"] += 1
                    p["t"] = 0.0
                    if p["seg"] >= len(p["path"]) - 1:
                        new_route(p)
                    continue
                x, y = a[0] + (b[0] - a[0]) * p["t"], a[1] + (b[1] - a[1]) * p["t"]
                ang = math.atan2(b[1] - a[1], b[0] - a[0])
                data.mocap_pos[first + i] = (x, y, 0.35)
                data.mocap_quat[first + i] = (math.cos(ang / 2), 0, 0, math.sin(ang / 2))
            mujoco.mj_forward(model, data)
            v.sync()
            time.sleep(1 / 60)


def door(kind: str, policy: str) -> None:
    params = {"door_type": kind, "door_mass_kg": 40.0 if kind == "fire" else 8.0,
              "min_push_n": 400.0 if kind == "fire" else 20.0, "open_delay_s": 4.0}
    g = mujoco_door._geometry(params)
    model = mujoco.MjModel.from_xml_string(mujoco_door.build_door_xml(params))
    data = mujoco.MjData(model)
    aid = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in ("drive_x", "opener")}
    # before: push at 60 N from t=0; after: request_open at t=0, wait 4.5 s, then drive gently (20 N)
    t_req = 0.0 if policy == "after" else None
    print(f"{kind} door, policy={policy}: " +
          ("push_door 60 N" if policy == "before" else "request_open, wait 4.5 s, proceed at 20 N"))
    with mujoco.viewer.launch_passive(model, data) as v:
        v.cam.lookat[:] = (0.2, 0.0, 0.8)
        v.cam.distance, v.cam.azimuth, v.cam.elevation = 5.0, 60, -25
        while v.is_running():
            t0 = time.time()
            if policy == "before":
                model.actuator_forcerange[aid["drive_x"]] = [-60, 60]
                data.ctrl[aid["drive_x"]] = 0.6
            else:
                if data.time >= t_req + params["open_delay_s"]:
                    data.ctrl[aid["opener"]] = -g["opener"]
                drive = 0.6 if data.time >= 4.5 else 0.0
                model.actuator_forcerange[aid["drive_x"]] = [-20, 20]
                data.ctrl[aid["drive_x"]] = drive
            for _ in range(int((1 / 60) / model.opt.timestep)):
                mujoco.mj_step(model, data)
            v.sync()
            if data.time > 14:  # loop the scene
                mujoco.mj_resetData(model, data)
            time.sleep(max(0.0, 1 / 60 - (time.time() - t0)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    b = sub.add_parser("building")
    b.add_argument("--site", default="B")
    b.add_argument("--robots", type=int, default=4)
    b.add_argument("--speed", type=float, default=2.0, help="m/s")
    d = sub.add_parser("door")
    d.add_argument("--door", choices=["fire", "office"], default="fire")
    d.add_argument("--policy", choices=["before", "after"], default="before")
    a = ap.parse_args()
    if a.mode == "building":
        building(a.site, a.robots, a.speed)
    else:
        door(a.door, a.policy)
