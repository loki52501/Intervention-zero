"""Fleet coordinator: hazard sharing through the blackboard, reroute, right-of-way, task claiming
and inter-robot clearance accounting.

Honest scope: helper motion is scripted pure pursuit on waypoint routes (no learned walking
policy).  The coordination rules are explicit and logged as events so the fleet-level behaviour
is auditable.  The blackboard is duck-typed: publish / query_near / claim_task.
"""
from __future__ import annotations

import math

import numpy as np

from iz.shop.controller import LOCKED

ROBOT_RADIUS = 0.30          # centre-to-hull (robot base box 0.26 x 0.22)
HAZARD_READ_RADIUS_M = 6.0
STALL_S = 3.0
STALL_SPEED = 0.05
CROSSING_DEG = 30.0
HAZARD_TTL_S = 60.0
HELPER_SPEED = 0.8
DEDUPE_M = 0.3


def _seg_dist(p, a, b) -> float:
    p, a, b = (np.asarray(v, float)[:2] for v in (p, a, b))
    ab = b - a
    L2 = float(ab @ ab)
    s = 0.0 if L2 < 1e-12 else float(np.clip((p - a) @ ab / L2, 0.0, 1.0))
    return float(np.linalg.norm(p - (a + s * ab)))


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


class Coordinator:
    def __init__(self, meta, blackboard, world, robot_ids, params=None):
        self.meta = meta
        self.bb = blackboard
        self.world = world
        self.robot_ids = list(robot_ids)
        self.yield_distance_m = float((params or {}).get("yield_distance_m", 0.6))
        self.events: list[dict] = []
        self.safety_violations: list[dict] = []
        self._routes: dict[str, list] = {}
        self._aisle: dict[str, int | None] = {}
        self._assigned = False
        self._published: list[tuple[float, float]] = []
        self._handled: dict[str, set] = {r: set() for r in self.robot_ids}
        self._known: list[tuple[float, float, float]] = []     # hazards seen on the blackboard
        self._poses: dict[str, tuple] = {}
        self._slow_since: dict[str, float | None] = {}
        self._stall_pub: set[str] = set()
        self._finished: set[str] = set()
        self._stopped: set[str] = set()
        self._yielding_prev: set[str] = set()
        self._viol_pairs: set[tuple[str, str]] = set()

    # ------------------------------------------------------------------ helpers
    def _ev(self, typ, t, rid, **kw):
        e = {"type": typ, "t": float(t), "robot_id": rid}
        e.update(kw)
        self.events.append(e)
        return e

    def _graph(self):
        m = self.meta
        ays, wps = m.get("aisle_ys"), m.get("waypoints")
        if not ays or not wps or len(wps) < 1 + 2 * len(ays):
            return None
        n = len(ays)
        try:
            ent = [list(map(float, wps[1 + a][:2])) for a in range(n)]
            ext = [list(map(float, wps[1 + n + a][:2])) for a in range(n)]
        except (IndexError, TypeError):
            return None
        return ent, ext

    def _checkout(self):
        c = self.meta.get("checkout_robot_xy") or self.meta.get("checkout_xy") or [0.0, 0.0]
        return [float(c[0]), float(c[1])]

    def _aisle_route(self, a):
        g = self._graph()
        ent, ext = g
        return [ent[a], ext[a], self._checkout()]

    def _n_aisles(self):
        return len(self.meta.get("aisle_ys") or []) if self._graph() else 0

    # ------------------------------------------------------------------ tasks
    def assign_tasks(self, t=0.0):
        self._assigned = True
        out: dict[str, int | None] = {}
        r0 = self.robot_ids[0]
        a0 = int(self.meta.get("target_aisle", 0))
        out[r0] = a0
        self._aisle[r0] = a0
        n = self._n_aisles()
        order = ([a0] + [a for a in range(n) if a != a0]) if n else [0]
        for rid in self.robot_ids[1:]:
            got = None
            for a in order:
                if self.bb.claim_task(self.world, f"aisle_{a}_sweep", rid):
                    got = a
                    break
            out[rid] = got
            self._aisle[rid] = got
            if got is not None:
                self._ev("task_claimed", t, rid, task_id=f"aisle_{got}_sweep", aisle=got)
            if n and got is not None:
                self._routes[rid] = self._aisle_route(got)
            else:
                self._routes[rid] = [self._checkout()]
        return out

    def route(self, robot_id):
        rem = [list(p) for p in self._routes.get(robot_id, [])]
        pose = self._poses.get(robot_id)
        return ([[pose[0], pose[1]]] + rem) if pose else rem

    # ------------------------------------------------------------------ tick
    def tick(self, t, poses, vels, detections):
        if not self._assigned:
            self.assign_tasks(t)
        self._poses.update({k: tuple(v) for k, v in poses.items()})
        self._publish(t, poses, vels, detections)
        self._read(t, poses)
        self._right_of_way(t, poses, vels)
        self._clearance(t, poses)

    def _publish(self, t, poses, vels, detections):
        for rid, dets in (detections or {}).items():
            for (x, y, r) in dets:
                if any(math.hypot(x - px, y - py) < DEDUPE_M for px, py in self._published):
                    continue
                near = self.bb.query_near(self.world, (x, y), DEDUPE_M, kinds=["hazard"], now_sim=t)
                self._published.append((float(x), float(y)))
                if near:
                    continue
                d = self.bb.publish(self.world, "hazard", float(x), float(y), float(r), rid,
                                    ttl=HAZARD_TTL_S, sim_t=t)
                self._ev("hazard_shared", t, rid, hazard_id=(d or {}).get("_id"), x=float(x), y=float(y),
                         radius=float(r))
        for rid, p in poses.items():
            v = (vels or {}).get(rid)
            if v is None:
                continue
            speed = math.hypot(v[0], v[1])
            if speed < STALL_SPEED and rid not in self._stopped and rid not in self._finished:
                s0 = self._slow_since.setdefault(rid, t)
                if s0 is None:
                    self._slow_since[rid] = s0 = t
                if t - s0 > STALL_S and rid not in self._stall_pub:
                    self._stall_pub.add(rid)
                    d = self.bb.publish(self.world, "blocked_lane", float(p[0]), float(p[1]), 0.6, rid,
                                        ttl=HAZARD_TTL_S, sim_t=t)
                    self._published.append((float(p[0]), float(p[1])))
                    self._ev("hazard_shared", t, rid, hazard_id=(d or {}).get("_id"), kind="blocked_lane",
                             x=float(p[0]), y=float(p[1]), radius=0.6)
            else:
                self._slow_since[rid] = None
                if speed >= STALL_SPEED:
                    self._stall_pub.discard(rid)

    def _read(self, t, poses):
        for rid in self.robot_ids[1:]:
            if rid not in poses or rid in self._finished:
                continue
            pose = poses[rid]
            hz = self.bb.query_near(self.world, (pose[0], pose[1]), HAZARD_READ_RADIUS_M,
                                    kinds=["hazard", "blocked_lane"], now_sim=t)
            for h in hz:
                key = (round(float(h["x"]), 1), round(float(h["y"]), 1))
                if all(math.hypot(h["x"] - k[0], h["y"] - k[1]) > 1e-6 for k in self._known):
                    self._known.append((float(h["x"]), float(h["y"]), float(h.get("radius", 0.5))))
            for h in hz:
                hid = h.get("_id", (h["x"], h["y"]))
                if hid in self._handled[rid]:
                    continue
                self._handled[rid].add(hid)
                hx, hy, hr = float(h["x"]), float(h["y"]), float(h.get("radius", 0.5))
                pts = self.route(rid)
                if point_dist(pts, (hx, hy)) >= hr + 0.3:
                    continue
                self._reroute(t, rid, pose, (hx, hy, hr), h)

    def _reroute(self, t, rid, pose, hz, h):
        hx, hy, hr = hz
        margin = hr + 0.3
        px, py = pose[0], pose[1]
        g = self._graph()
        done = None
        if g is not None:
            ent, ext = g
            n = len(ent)
            cur = min(range(n), key=lambda a: abs(self.meta["aisle_ys"][a] - py))
            inside = px > ent[cur][0] + 0.5 and abs(self.meta["aisle_ys"][cur] - py) < 0.6

            def clear(a):
                return all(_seg_dist((k[0], k[1]), ent[a], ext[a]) >= k[2] + 0.3 for k in self._known) \
                    and _seg_dist((hx, hy), ent[a], ext[a]) >= margin

            cands = sorted([a for a in range(n) if clear(a)],
                           key=lambda a: abs(self.meta["aisle_ys"][a] - py))
            if cands:
                a = cands[0]
                new = ([ent[cur]] if inside else []) + self._aisle_route(a)
                if point_dist([[px, py]] + new, (hx, hy)) >= margin:
                    self._routes[rid] = new
                    self._aisle[rid] = a
                    done = f"aisle_{a}"
        if done is None:
            rem = self._routes.get(rid, [])
            pts = [[px, py]] + rem
            idx = next((i for i in range(len(pts) - 1) if _seg_dist((hx, hy), pts[i], pts[i + 1]) < margin), None)
            if idx is not None:
                a, b = np.array(pts[idx], float), np.array(pts[idx + 1], float)
                d = b - a
                d = d / (np.linalg.norm(d) + 1e-9)
                perp = np.array([-d[1], d[0]])
                rel = np.array([px, py]) - np.array([hx, hy])
                if float(d[0] * rel[1] - d[1] * rel[0]) < 0:  # 2D cross (np.cross 2D raises on NumPy 2)
                    perp = -perp
                off = hr + 0.6
                c = np.array([hx, hy])
                detour = [list(c - d * off + perp * off), list(c + d * off + perp * off)]
                self._routes[rid] = rem[:idx] + detour + rem[idx:]
                done = "detour"
        if done is not None:
            self._ev("reroute", t, rid, hazard_id=h.get("_id"), via=done,
                     dist_to_hazard_m=float(math.hypot(px - hx, py - hy)))

    def _right_of_way(self, t, poses, vels):
        stopped = set()
        ids = [r for r in self.robot_ids if r in poses and r in (vels or {})]
        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                pa, pb = np.array(poses[a][:2], float), np.array(poses[b][:2], float)
                va, vb = np.array(vels[a][:2], float), np.array(vels[b][:2], float)
                d = float(np.linalg.norm(pb - pa))
                if d >= self.yield_distance_m + 2 * ROBOT_RADIUS:
                    continue
                na, nb = np.linalg.norm(va), np.linalg.norm(vb)
                if na <= 0.05 or nb <= 0.05:
                    continue
                ang = math.degrees(math.acos(float(np.clip(va @ vb / (na * nb), -1, 1))))
                if ang <= CROSSING_DEG or float((pb - pa) @ (vb - va)) >= 0:
                    continue
                loser = b if self.robot_ids.index(b) > self.robot_ids.index(a) else a
                stopped.add(loser)
                if loser not in self._yielding_prev:
                    other = a if loser == b else b
                    self._ev("yield", t, loser, to=other, dist_m=d)
        self._stopped = stopped
        self._yielding_prev = set(stopped)

    def _clearance(self, t, poses):
        ids = [r for r in self.robot_ids if r in poses]
        lim = float(LOCKED["min_person_clearance_m"])
        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                d = math.hypot(poses[a][0] - poses[b][0], poses[a][1] - poses[b][1])
                clr = d - 2 * ROBOT_RADIUS
                if clr < lim and (a, b) not in self._viol_pairs:
                    self._viol_pairs.add((a, b))
                    self.safety_violations.append({"type": "too_close_to_robot", "t": float(t),
                                                   "robots": [a, b], "clearance_m": float(clr)})

    # ------------------------------------------------------------------ control
    def should_stop(self, robot_id):
        return robot_id in self._stopped

    def mark_finished(self, robot_id):
        self._finished.add(robot_id)

    def helper_command(self, robot_id, pose):
        rt = self._routes.setdefault(robot_id, [])
        while rt and math.hypot(rt[0][0] - pose[0], rt[0][1] - pose[1]) < 0.35:
            rt.pop(0)
        if not rt:
            self._finished.add(robot_id)
            return 0.0, 0.0, 0.0
        if self.should_stop(robot_id):
            return 0.0, 0.0, 0.0
        dx, dy = rt[0][0] - pose[0], rt[0][1] - pose[1]
        L = math.hypot(dx, dy) + 1e-9
        err = _wrap(math.atan2(dy, dx) - pose[2])
        wz = float(np.clip(2.0 * err, -1.5, 1.5))
        return HELPER_SPEED * dx / L, HELPER_SPEED * dy / L, wz

    def counts(self):
        c = {k: 0 for k in ("hazard_shared", "reroute", "yield", "task_claimed")}
        for e in self.events:
            if e["type"] in c:
                c[e["type"]] += 1
        c["safety_violations"] = len(self.safety_violations)
        return c


def point_dist(pts, p) -> float:
    from iz.shop.episode import point_polyline_dist
    return point_polyline_dist(p, pts)
