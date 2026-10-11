"""Multi-robot scene builder: clone the 'robot' body of a shop-contract scene n-1 times.

Copies get every name suffixed with _k, no cameras, and their own base_x_k/base_y_k/base_yaw_k
slide/hinge joints plus velocity actuators act_vx_k/act_vy_k/act_wz_k.  Arm joints in copies are
NOT actuated (passive, damped): helper robots only travel; they do not manipulate.  Robots never
collide with each other or the cart (collision bits 4 vs 17 / hands 32 vs 8), so inter-robot
safety is a coordinator concern (see coordinator.py).
"""
from __future__ import annotations

import copy
import xml.etree.ElementTree as ET


def _obstacle_rects(root):
    """(xmin, xmax, ymin, ymax) of tall static box geoms directly under worldbody (shelves)."""
    out = []
    wb = root.find("worldbody")
    if wb is None:
        return out
    for g in wb.findall("geom"):
        nm = g.get("name", "")
        if g.get("type") != "box" or nm.startswith("wall") or nm == "floor":
            continue
        try:
            p = [float(v) for v in g.get("pos", "0 0 0").split()]
            s = [float(v) for v in g.get("size").split()]
        except (AttributeError, ValueError):
            continue
        if s[2] < 0.3:
            continue
        out.append((p[0] - s[0], p[0] + s[0], p[1] - s[1], p[1] + s[1]))
    return out


def _near_rect(x, y, rects, m=0.3):
    return any(r[0] - m <= x <= r[1] + m and r[2] - m <= y <= r[3] + m for r in rects)


def _starts(meta, n, rects):
    rx, ry = meta["robot_xy"][0], meta["robot_xy"][1]
    sh = meta.get("store_half")
    out = [[float(rx), float(ry)]]
    for k in range(1, n):
        y = ry + (1.6 if k == 1 else -1.6)
        if sh:
            y = min(max(y, -sh[1] + 0.5), sh[1] - 0.5)
        x = rx
        if _near_rect(x, y, rects):
            x = rx - 0.9 * k
        out.append([float(x), float(y)])
    return out


def add_fleet_robots(xml: str, meta: dict, n: int):
    if n <= 1:
        return xml, meta
    root = ET.fromstring(xml.lstrip("﻿"))
    wb = root.find("worldbody")
    robot = next((b for b in wb.findall("body") if b.get("name") == "robot"), None)
    if robot is None:
        raise ValueError("no worldbody body named 'robot'")
    act = root.find("actuator")
    if act is None:
        act = ET.SubElement(root, "actuator")
    starts = _starts(meta, n, _obstacle_rects(root))
    for k in range(1, n):
        c = copy.deepcopy(robot)
        for el in c.iter():
            if el.get("name"):
                el.set("name", f"{el.get('name')}_{k}")
        for parent in list(c.iter()):
            for cam in parent.findall("camera"):
                parent.remove(cam)
        sx, sy = starts[k]
        c.set("pos", f"{sx:.4f} {sy:.4f} 0")
        for j in c.iter("joint"):
            if j.get("name") == f"base_x_{k}":
                j.set("ref", f"{sx:.4f}")
            elif j.get("name") == f"base_y_{k}":
                j.set("ref", f"{sy:.4f}")
        tint = [(0.9, 0.55, 0.3), (0.45, 0.8, 0.5), (0.8, 0.5, 0.8)][(k - 1) % 3]
        for g in c.iter("geom"):          # slight tint so robots are told apart on film
            if g.get("name") in (f"torso_g_{k}", f"base_g_{k}"):
                g.set("rgba", f"{tint[0]} {tint[1]} {tint[2]} 1")
        wb.append(c)
        for nm, jn, kv, cr, fr in (("act_vx", "base_x", 900, "-2 2", "-450 450"),
                                   ("act_vy", "base_y", 900, "-2 2", "-450 450"),
                                   ("act_wz", "base_yaw", 1500, "-3 3", "-500 500")):
            ET.SubElement(act, "velocity", {"name": f"{nm}_{k}", "joint": f"{jn}_{k}", "kv": str(kv),
                                            "ctrlrange": cr, "forcerange": fr})
    m2 = copy.deepcopy(meta)
    m2["fleet_robots"] = ["robot"] + [f"robot_{k}" for k in range(1, n)]
    m2["fleet_starts"] = starts
    return ET.tostring(root, encoding="unicode"), m2
