"""Terrain (hfield / box stairs) and domain-randomisation (DR) helpers for generated worlds.

Pure module: takes plain dicts (Terrain.model_dump(), DomainRand dump), imports only math, random,
zlib, numpy and mujoco. It does not import iz.worldgen.spec or build.

Collision bits: 1 static, 2 cart, 4 robot body (conaffinity 17), 8 products, 16 shoppers, 32 hands.
NEW BIT 64 = terrain. Terrain geoms use contype=64 conaffinity=10, so they touch the cart (2) and
products (8) but never the robot (the robot is on planar base joints base_x/base_y/base_yaw at z=0 and
any terrain contact would jam it).

Honest scope: terrain affects cart/products physics and the gait engine via height_at(); the planar
robot base does not climb; locomotion is a parametric gait engine, not a learned policy. The simulator
is authored by us.

hfield: inline `elevation="..."` in <hfield> COMPILES on MuJoCo 3.x (verified by tests), so build.py
does not need set_hfield(); set_hfield() is kept as a harmless post-compile writer.
"""
from __future__ import annotations

import math
import random
import zlib

import numpy as np

TERRAIN_CONTYPE = 64
TERRAIN_CONAFFINITY = 10
HF_RES = 0.1
_ZMAX_CAP = 0.6
_HF_BASE = 0.05


def terrain_rng(name: str, seed: int) -> np.random.Generator:
    return np.random.default_rng(zlib.crc32(f"terrain:{name}:{seed}".encode()))


def _blur(a: np.ndarray, k: int = 5) -> np.ndarray:
    ker = np.ones(k) / k
    pad = k // 2
    out = np.pad(a, ((0, 0), (pad, pad)), mode="edge")
    out = np.apply_along_axis(lambda r: np.convolve(r, ker, mode="valid"), 1, out)
    out = np.pad(out, ((pad, pad), (0, 0)), mode="edge")
    return np.apply_along_axis(lambda c: np.convolve(c, ker, mode="valid"), 0, out)


def hfield_data(t: dict, region, name: str, seed: int):
    """Heightfield dict for rough/slope/ramp terrain, else None."""
    typ = t.get("type")
    if typ not in ("rough", "slope", "ramp") or region is None:
        return None
    x0, y0, x1, y1 = (float(v) for v in region)
    lx, ly = x1 - x0, y1 - y0
    if lx <= 0 or ly <= 0:
        return None
    ncol = max(int(round(lx / HF_RES)) + 1, 3)
    nrow = max(int(round(ly / HF_RES)) + 1, 3)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    if typ == "rough":
        r = float(t.get("roughness_m") or 0.0)
        if r <= 0:
            return None
        zmax = min(max(r, 0.005), _ZMAX_CAP)
        g = terrain_rng(name, seed).random((nrow, ncol))
        g = _blur(_blur(g))
        rng_ = g.max() - g.min()
        g = (g - g.min()) / rng_ if rng_ > 0 else np.zeros_like(g)
        pos_z = -zmax / 2
    else:
        rise = math.tan(math.radians(float(t.get("slope_deg") or 0.0))) * lx
        zmax = min(rise, _ZMAX_CAP)
        if zmax <= 0:
            return None
        u = np.linspace(0.0, 1.0, ncol)
        if typ == "ramp":
            u = np.minimum(u / 0.6, 1.0)
        g = np.tile(u, (nrow, 1))
        pos_z = 0.0
    return {"nrow": nrow, "ncol": ncol, "size": [lx / 2, ly / 2, zmax, _HF_BASE],
            "elev": [float(v) for v in g.ravel()], "pos": [cx, cy, pos_z], "zmax": float(zmax)}


def hfield_xml(hf: dict, friction: float = 1.0) -> tuple[str, str]:
    """(asset_xml, geom_xml) with inline elevation (compiles on MuJoCo 3)."""
    rx, ry, zm, base = hf["size"]
    elev = " ".join("%.4f" % v for v in hf["elev"])
    asset = (f'<hfield name="terrain_hf" nrow="{hf["nrow"]}" ncol="{hf["ncol"]}" '
             f'size="{rx:.4f} {ry:.4f} {zm:.4f} {base:.4f}" elevation="{elev}"/>')
    px, py, pz = hf["pos"]
    geom = (f'<geom name="terrain" type="hfield" hfield="terrain_hf" pos="{px:.4f} {py:.4f} {pz:.4f}" '
            f'contype="{TERRAIN_CONTYPE}" conaffinity="{TERRAIN_CONAFFINITY}" '
            f'rgba="0.55 0.5 0.42 1" friction="{friction:g} 0.005 0.0001"/>')
    return asset, geom


def set_hfield(model, hf) -> None:
    """Write normalized elev into model.hfield_data (harmless if inline elevation was used)."""
    import mujoco
    hid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_HFIELD, "terrain_hf")
    if hid < 0:
        return
    n = int(model.hfield_nrow[hid]) * int(model.hfield_ncol[hid])
    adr = int(model.hfield_adr[hid])
    e = np.asarray(hf["elev"], dtype=float)
    if e.size == n:
        model.hfield_data[adr:adr + n] = e


def _steps_ok(t: dict, region) -> bool:
    return (region is not None and t.get("type") == "steps"
            and int(t.get("step_n") or 0) >= 1 and float(t.get("step_h_m") or 0) > 0)


def stairs_xml(t: dict, region, friction: float = 1.0):
    """step_n box treads climbing along +x. Returns (xml, footprint rects)."""
    if not _steps_ok(t, region):
        return "", []
    x0, y0, x1, y1 = (float(v) for v in region)
    n, h = int(t["step_n"]), float(t["step_h_m"])
    depth, cy, hy = (x1 - x0) / n, (y0 + y1) / 2, (y1 - y0) / 2
    parts, rects = [], []
    for k in range(n):
        xa = x0 + k * depth
        hh = (k + 1) * h
        grey = 0.62 if k % 2 == 0 else 0.7
        parts.append(
            f'<geom name="stair_{k}" type="box" pos="{xa + depth / 2:.4f} {cy:.4f} {hh / 2:.4f}" '
            f'size="{depth / 2:.4f} {hy:.4f} {hh / 2:.4f}" contype="{TERRAIN_CONTYPE}" '
            f'conaffinity="{TERRAIN_CONAFFINITY}" rgba="{grey} {grey} {grey} 1" '
            f'friction="{friction:g} 0.005 0.0001"/>')
        parts.append(
            f'<geom name="stair_{k}_edge" type="box" pos="{xa + 0.02:.4f} {cy:.4f} {hh + 0.001:.4f}" '
            f'size="0.02 {hy:.4f} 0.001" contype="0" conaffinity="0" rgba="0.95 0.8 0.1 1"/>')
        rects.append([xa, y0, xa + depth, y1])
    return "".join(parts), rects


def height_at(t: dict, region, x: float, y: float) -> float:
    """Analytic ground height (0 outside region; rough has mean 0)."""
    if region is None:
        return 0.0
    x0, y0, x1, y1 = (float(v) for v in region)
    if not (x0 <= x <= x1 and y0 <= y <= y1):
        return 0.0
    typ, lx = t.get("type"), x1 - x0
    if typ in ("slope", "ramp"):
        # analytic, NOT clipped (only the rendered hfield is clipped to _ZMAX_CAP)
        zmax = math.tan(math.radians(float(t.get("slope_deg") or 0.0))) * lx
        u = (x - x0) / lx
        if typ == "ramp":
            u = min(u / 0.6, 1.0)
        return float(zmax * u)
    if _steps_ok(t, region):
        n = int(t["step_n"])
        k = min(int((x - x0) / (lx / n)), n - 1)
        return float((k + 1) * float(t["step_h_m"]))
    return 0.0


def sample_dr(dr, name: str, seed: int, horizon_s: float = 120.0) -> dict:
    """Per-(name, seed) domain-randomisation draw; JSON-serialisable."""
    if dr is None:
        return {"enabled": False, "friction": 1.0, "mass_scale": 1.0, "pushes": [],
                "joint_noise_rad": 0.0, "latency_steps": 0}
    rng = random.Random(f"dr:{name}:{seed}")
    f = rng.uniform(*dr["friction"])
    m = rng.uniform(*dr["mass_scale"])
    lo, hi = dr["action_latency_steps"]
    lat = rng.randint(int(lo), int(hi))
    pushes = []
    push = dr.get("push")
    if push:
        iv, sp = push["interval_s"], push["speed_mps"]
        t = rng.uniform(*iv)
        while t < horizon_s:
            speed = rng.uniform(*sp)
            ang = rng.uniform(0, 2 * math.pi)
            pushes.append({"t": round(t, 3), "vx": speed * math.cos(ang),
                           "vy": speed * math.sin(ang), "dur": 0.1})
            t += rng.uniform(*iv)
    return {"enabled": True, "friction": round(f, 4), "mass_scale": round(m, 4), "pushes": pushes,
            "joint_noise_rad": dr.get("joint_noise_rad", 0.0), "latency_steps": int(lat), "seed": seed}


def _subtree_bodies(model, root: int) -> list[int]:
    out = []
    for b in range(model.nbody):
        c = b
        while c > 0 and c != root:
            c = int(model.body_parentid[c])
        if c == root:
            out.append(b)
    return out


def _body_id(model, name: str) -> int:
    import mujoco
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)


def apply_dr_model(model, meta: dict, skip_bodies=("world",)) -> None:
    """Scale mass/inertia of the 'robot' subtree by meta['dr']['mass_scale'].

    NOT idempotent: call exactly ONCE after compile.
    """
    import mujoco
    dr = (meta or {}).get("dr") or {}
    if not dr.get("enabled"):
        return
    rid = _body_id(model, "robot")
    if rid < 0:
        return
    s = float(dr.get("mass_scale", 1.0))
    for b in _subtree_bodies(model, rid):
        if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) in skip_bodies:
            continue
        model.body_mass[b] *= s
        model.body_inertia[b] *= s


_MASS_CACHE: dict = {}


def apply_dr(model, data, meta: dict, t: float, body: str = "torso") -> None:
    """Per-step push application; call BEFORE mj_step. Never emits events."""
    dr = (meta or {}).get("dr") or {}
    if not dr.get("enabled"):
        return
    bid = _body_id(model, body)
    if bid < 0:
        bid = _body_id(model, "robot")
        if bid < 0:
            return
    data.xfrc_applied[bid, :3] = 0.0
    key = (id(model), bid, int(model.nbody), float(model.body_mass.sum()))
    if key not in _MASS_CACHE:
        rid = _body_id(model, "robot")
        _MASS_CACHE[key] = float(sum(model.body_mass[b] for b in _subtree_bodies(model, rid if rid >= 0 else bid)))
    M = _MASS_CACHE[key]
    for p in dr.get("pushes", []):
        if p["t"] <= t < p["t"] + p["dur"]:
            data.xfrc_applied[bid, 0] = M * p["vx"] / p["dur"]
            data.xfrc_applied[bid, 1] = M * p["vy"] / p["dur"]
            break


def delay_action(buf: list, action, latency_steps: int):
    """FIFO action delay. Latency 0 returns the action itself."""
    if latency_steps <= 0:
        return action
    buf.append(action)
    if len(buf) > latency_steps:
        return buf.pop(0)
    return buf[0]


def joint_noise(rng: random.Random, n: int, sigma: float) -> list[float]:
    return [rng.uniform(-sigma, sigma) for _ in range(n)]
