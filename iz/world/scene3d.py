"""Whole-site 3D MuJoCo scene built from a types.Site (visual only, no physics) + MP4 clip rendering.

Independent of other iz.world modules: uses only types.Site/Zone/Edge fields.
"""
from __future__ import annotations

import math
import os

from iz.types import Site

ZONE_RGBA = {
    "public": "0.80 0.85 0.95 1", "corridor": "0.82 0.82 0.80 1", "ward": "0.70 0.90 0.80 1",
    "office": "0.95 0.90 0.70 1", "kitchen": "0.95 0.80 0.65 1", "elevator": "0.65 0.70 0.80 1",
    "service": "0.78 0.75 0.85 1", "human_only": "0.90 0.40 0.40 1",
}
DOOR_RGBA = {"fire": "0.85 0.10 0.10 1", "office": "0.55 0.35 0.20 1", "shortcut": "0.98 0.55 0.10 1",
             "badge": "0.20 0.30 0.80 1"}
TILE_HALF = 1.6


def _f(v: float) -> str:
    return f"{v:.3f}"


def _zone_map(site: Site) -> dict:
    return {z.id: z for z in site.zones}


def build_site_xml(site: Site, robot_at: str | None = None) -> str:
    zones = _zone_map(site)
    out: list[str] = []
    xs = [z.xy[0] for z in site.zones] or [0.0]
    ys = [z.xy[1] for z in site.zones] or [0.0]
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    for z in site.zones:
        out.append(f'<geom name="tile_{z.id}" type="box" pos="{_f(z.xy[0])} {_f(z.xy[1])} 0.01" '
                   f'size="{TILE_HALF} {TILE_HALF} 0.01" rgba="{ZONE_RGBA.get(z.kind, ZONE_RGBA["public"])}"/>')
    elevators = [z for z in site.zones if z.kind == "elevator"]
    for n, z in enumerate(elevators):
        for k, dx in ((1, -0.7), (2, 0.7)):
            name = f"E{k}" if n == 0 else f"{z.id}_E{k}"
            out.append(f'<geom name="{name}" type="box" pos="{_f(z.xy[0] + dx)} {_f(z.xy[1])} 0.6" '
                       f'size="0.45 0.45 0.6" rgba="0.35 0.38 0.45 1"/>')
    for i, e in enumerate(site.edges):
        if e.a not in zones or e.b not in zones:
            continue
        ax, ay = zones[e.a].xy
        bx, by = zones[e.b].xy
        mx, my = (ax + bx) / 2, (ay + by) / 2
        dx, dy = bx - ax, by - ay
        ln = math.hypot(dx, dy) or 1.0
        ang = math.atan2(dy, dx)
        nx, ny = -dy / ln, dx / ln
        out.append(f'<geom name="strip_{i}" type="box" pos="{_f(mx)} {_f(my)} 0.005" euler="0 0 {_f(ang)}" '
                   f'size="{_f(ln / 2)} 0.5 0.005" rgba="0.88 0.88 0.86 1"/>')
        for side in (-1, 1):
            ox, oy = nx * 0.55 * side, ny * 0.55 * side
            out.append(f'<geom name="rail_{i}_{0 if side < 0 else 1}" type="capsule" '
                       f'fromto="{_f(ax + ox)} {_f(ay + oy)} 0.25 {_f(bx + ox)} {_f(by + oy)} 0.25" size="0.04" '
                       f'rgba="0.6 0.6 0.65 1"/>')
        if e.door_type:
            tall = 1.0
            out.append(f'<geom name="door_{i}" type="box" pos="{_f(mx)} {_f(my)} {_f(tall / 2)}" '
                       f'euler="0 0 {_f(ang)}" size="0.06 0.5 {_f(tall / 2)}" '
                       f'rgba="{DOOR_RGBA.get(e.door_type, DOOR_RGBA["badge"])}"/>')
    rz = zones.get(robot_at) if robot_at else None
    if rz is None and site.zones:
        rz = site.zones[0]
    rx, ry = (rz.xy if rz else (0.0, 0.0))
    out.append(f'<body name="robot" mocap="true" pos="{_f(rx)} {_f(ry)} 0.35">'
               f'<geom name="robot_geom" type="cylinder" size="0.3 0.3" rgba="0.1 0.45 0.9 1"/>'
               f'<geom name="robot_nose" type="box" pos="0.25 0 0.2" size="0.08 0.15 0.05" rgba="1 1 1 1"/></body>')
    body = "\n    ".join(out)
    span = max(max(xs) - min(xs), max(ys) - min(ys), 6.0)
    return f"""<mujoco model="site_{site.id}">
  <compiler angle="radian"/>
  <visual><global offwidth="640" offheight="480"/></visual>
  <default><geom contype="0" conaffinity="0"/></default>
  <worldbody>
    <light pos="{_f(cx)} {_f(cy)} {_f(span)}" dir="0 0 -1" diffuse="0.9 0.9 0.9"/>
    <geom name="ground" type="plane" pos="{_f(cx)} {_f(cy)} -0.01" size="{_f(span)} {_f(span)} 0.1" rgba="0.25 0.27 0.30 1"/>
    {body}
  </worldbody>
</mujoco>"""


def render_site_clip(site: Site, path: list[str], out_path: str, seconds_per_edge: float = 1.0,
                     fps: int = 30) -> str | None:
    """Move the robot along zone centres in `path` and write an mp4. Returns out_path, or None on any failure."""
    try:
        import imageio
        import mujoco

        zones = _zone_map(site)
        pts = [zones[z].xy for z in path if z in zones]
        if len(pts) < 1:
            return None
        model = mujoco.MjModel.from_xml_string(build_site_xml(site, robot_at=path[0]))
        data = mujoco.MjData(model)
        xs = [z.xy[0] for z in site.zones]
        ys = [z.xy[1] for z in site.zones]
        span = max(max(xs) - min(xs), max(ys) - min(ys), 6.0)
        cam = mujoco.MjvCamera()
        cam.lookat[:] = ((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, 0.0)
        cam.distance, cam.azimuth, cam.elevation = span * 1.25 + 4.0, 90.0, -62.0
        n = max(int(seconds_per_edge * fps), 1)
        frames = []
        renderer = mujoco.Renderer(model, 480, 640)
        try:
            segs = list(zip(pts[:-1], pts[1:])) or [(pts[0], pts[0])]
            for (x0, y0), (x1, y1) in segs:
                ang = math.atan2(y1 - y0, x1 - x0)
                for k in range(n):
                    t = k / n
                    data.mocap_pos[0] = (x0 + (x1 - x0) * t, y0 + (y1 - y0) * t, 0.35)
                    data.mocap_quat[0] = (math.cos(ang / 2), 0, 0, math.sin(ang / 2))
                    mujoco.mj_forward(model, data)
                    renderer.update_scene(data, camera=cam)
                    frames.append(renderer.render().copy())
        finally:
            renderer.close()
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        with imageio.get_writer(out_path, fps=fps, macro_block_size=1) as w:
            for f in frames + [frames[-1]] * fps:
                w.append_data(f)
        return out_path
    except Exception:  # noqa: BLE001 - rendering must never crash the caller
        return None
