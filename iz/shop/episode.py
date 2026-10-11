"""Shop episode runner: physics-truth failure detection, operator tips, hooks.

Failures and operator tips are derived ONLY from simulator truth (contacts,
body poses, forces), never from the controller's own beliefs.
"""
from __future__ import annotations

import math
import time
from collections import deque

import numpy as np

import mujoco

TILT_FAIL_DEG = 35.0
SHELF_FORCE_FAIL_N = 40.0
SLIDE_FAIL_M = 0.4
PERSON_CLEARANCE_M = 0.35  # LOCKED, mirrors controller.LOCKED
SHOPPER_SPEED = 0.8
SPILL_RADIUS_DEFAULT = 0.6
SHOPPER_PAUSE_M = 1.1
FLEET_HELPER_DELAY_S = 4.0  # helpers hold still this long so robot 0 can publish first


# ---------------------------------------------------------------- pure helpers
def tilt_deg_from_zaxis(zaxis) -> float:
    c = float(np.clip(zaxis[2], -1.0, 1.0))
    return math.degrees(math.acos(c))


def point_polyline_dist(p, pts) -> float:
    p = np.asarray(p, float)[:2]
    pts = [np.asarray(q, float)[:2] for q in pts]
    if not pts:
        return 0.0
    if len(pts) == 1:
        return float(np.linalg.norm(p - pts[0]))
    best = 1e9
    for a, b in zip(pts[:-1], pts[1:]):
        ab = b - a
        L2 = float(ab @ ab)
        s = 0.0 if L2 < 1e-12 else float(np.clip((p - a) @ ab / L2, 0.0, 1.0))
        best = min(best, float(np.linalg.norm(p - (a + s * ab))))
    return best


def shopper_position(path, t: float, speed: float = SHOPPER_SPEED):
    """Ping-pong along polyline at constant speed; deterministic in t."""
    pts = [np.asarray(q, float)[:2] for q in path]
    if len(pts) < 2:
        return pts[0] if pts else np.zeros(2)
    seg = [float(np.linalg.norm(b - a)) for a, b in zip(pts[:-1], pts[1:])]
    total = sum(seg)
    if total < 1e-9:
        return pts[0]
    d = (t * speed) % (2 * total)
    if d > total:
        d = 2 * total - d
    for i, L in enumerate(seg):
        if d <= L or i == len(seg) - 1:
            a, b = pts[i], pts[i + 1]
            return a + (b - a) * (min(d, L) / max(L, 1e-9))
        d -= L
    return pts[-1]


def classify_failure(failure):
    """Map a failure name to (status, failure)."""
    if failure is None:
        return "success", None
    if failure in ("person_contact", "too_close_to_person", "too_close_to_robot"):
        return "safety_violation", failure
    return "takeover", failure


def make_tip(failure, info: dict) -> str:
    v = info.get("speed_mps", 0.0)
    if failure == "cart_tipped":
        return (f"Cart tipped at {v:.1f} m/s turning near waypoint {info.get('near_wp', '?')} "
                f"(tilt {info.get('tilt_deg', 0):.0f} deg, yaw rate {abs(info.get('yaw_rate', 0)):.1f} rad/s). "
                f"Slow to <= 0.6 m/s before turns over 30 deg.")
    if failure == "spill_slide":
        return (f"Cart slid {info.get('slide_m', 0):.2f} m off path on the spill at {v:.1f} m/s; "
                f"cap speed near spills to 0.4 m/s.")
    if failure == "too_close_to_person":
        return (f"Passed {info.get('dist_m', 0):.2f} m from a shopper at {v:.1f} m/s; "
                f"keep >= 0.8 m and wait.")
    if failure == "person_contact":
        return (f"Made contact with a shopper at {v:.1f} m/s; stop and wait until the shopper "
                f"is >= 0.8 m away.")
    if failure == "collision_shelf":
        return (f"Hit a shelf with {info.get('force_n', 0):.0f} N at {v:.1f} m/s; "
                f"brake earlier (start slowing >= 0.9 m from obstacles) and reduce speed in aisles.")
    if failure == "dropped_product":
        return (f"Product dropped during transfer (wrist force {info.get('wrist_force_n', 0):.0f} N); "
                f"lower the reach height and move slower while holding the product.")
    if failure == "timeout":
        return (f"Timed out after {info.get('t', 0):.0f} s without checkout "
                f"(progress {info.get('progress', 0):.0%}); "
                f"avoid long waits and unnecessarily low speeds.")
    return ""


# ---------------------------------------------------------------- runner
def _name_id(model, objtype, name):
    return mujoco.mj_name2id(model, objtype, name)


def _sensor(model, data, name):
    sid = _name_id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
    if sid < 0:
        return None
    a, d = int(model.sensor_adr[sid]), int(model.sensor_dim[sid])
    return data.sensordata[a:a + d]


def _spill_list(meta):
    out = []
    for s in meta.get("spills", []) or []:
        if isinstance(s, dict):
            c = s.get("xy") or s.get("pos") or [s.get("x", 0), s.get("y", 0)]
            r = s.get("radius") or max(s.get("sx", 0), s.get("sy", 0)) or SPILL_RADIUS_DEFAULT
            out.append((float(c[0]), float(c[1]), float(r)))
        else:
            s = list(s)
            if len(s) >= 4:  # x, y, half-x, half-y
                out.append((float(s[0]), float(s[1]), float(max(s[2], s[3]))))
            elif len(s) == 3:
                out.append((float(s[0]), float(s[1]), float(s[2])))
            else:
                out.append((float(s[0]), float(s[1]), SPILL_RADIUS_DEFAULT))
    return out


def _spec_get(spec, key, default=None):
    if spec is None:
        return default
    if isinstance(spec, dict):
        return spec.get(key, default)
    return getattr(spec, key, default)


def run_episode(params: dict, seed: int, max_time_s: float = 70, viewer=None, recorder=None,
                spec=None, robots: int = 1, blackboard=None, world_id=None,
                emit_rd: bool = False) -> dict:
    from iz.shop.scene import build_shop
    from iz.shop.controller import ShopController

    if spec is not None:
        from iz.worldgen.build import build_world
        xml, meta = build_world(spec, seed)
    else:
        xml, meta = build_shop(int(seed))
    robots = int(robots)
    if world_id is None:
        world_id = _spec_get(spec, "name") or f"shop:{seed}"
    fleet = robots > 1
    coord = None
    if fleet:
        from iz.fleet.scene import add_fleet_robots
        from iz.fleet.coordinator import Coordinator, ROBOT_RADIUS
        from iz.store.blackboard import MemoryBlackboard
        xml, meta = add_fleet_robots(xml, meta, robots)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    ctrl = ShopController(model, data, meta, dict(params))
    dt = float(model.opt.timestep)

    B, G = mujoco.mjtObj.mjOBJ_BODY, mujoco.mjtObj.mjOBJ_GEOM
    robot_id = _name_id(model, B, "robot")
    cart_id = _name_id(model, B, "cart")
    prod_id = _name_id(model, B, meta["target_product"])
    root = model.body_rootid
    robot_geoms = [g for g in range(model.ngeom) if root[model.geom_bodyid[g]] == robot_id]
    cart_geoms = [g for g in range(model.ngeom) if root[model.geom_bodyid[g]] == cart_id]
    mine = set(robot_geoms) | set(cart_geoms)
    near_geoms = [g for g in robot_geoms + cart_geoms if model.geom_contype[g] != 0]
    shelf_geoms = {g for g in range(model.ngeom)
                   if (mujoco.mj_id2name(model, G, g) or "").startswith("shelf_")}
    shopper_geoms = [g for g in range(model.ngeom)
                     if (mujoco.mj_id2name(model, G, g) or "").startswith("shopper_")]
    shopper_gset = set(shopper_geoms)

    # shoppers (mocap)
    shoppers = []
    paths = meta.get("shopper_paths", []) or []
    for i, path in enumerate(paths):
        bid = _name_id(model, B, f"shopper_{i}")
        if bid < 0 or model.body_mocapid[bid] < 0 or len(path) == 0:
            continue
        mi = int(model.body_mocapid[bid])
        shoppers.append([mi, path, float(data.mocap_pos[mi][2]), 0.0])

    wp = meta.get("waypoints", [])
    route_pts = [wp[i] for i in meta.get("route", [])] if wp else []
    spills = _spill_list(meta)
    checkout = np.asarray(meta["checkout_xy"], float)[:2]
    prod_z0 = float(data.xpos[prod_id][2])

    vx_a, vy_a, wz_a = (int(model.joint(n).dofadr[0]) for n in ("base_x", "base_y", "base_yaw"))
    helpers = []  # fleet only
    fleet_bodies = [robot_id]
    h_geoms = []
    min_rob_clear = {}
    if fleet:
        bb = blackboard if blackboard is not None else MemoryBlackboard()
        coord = Coordinator(meta, bb, world_id, meta["fleet_robots"], dict(params))
        coord.assign_tasks(0.0)
        for rid in meta["fleet_robots"]:
            if rid == "robot":
                continue
            k_ = rid.rsplit("_", 1)[-1]
            bid = _name_id(model, B, rid)
            fleet_bodies.append(bid)
            jn = [model.joint(f"base_{a}_{k_}") for a in ("x", "y", "yaw")]
            an = [_name_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"act_{a}_{k_}") for a in ("vx", "vy", "wz")]
            gs = [g for g in range(model.ngeom)
                  if root[model.geom_bodyid[g]] == root[bid] and model.geom_contype[g] != 0]
            h_geoms.extend(gs)
            helpers.append(dict(rid=rid, bid=bid, q=[int(j.qposadr[0]) for j in jn],
                                v=[int(j.dofadr[0]) for j in jn], act=an, geoms=gs))
            min_rob_clear[rid] = 9.0
        min_rob_clear["robot"] = 9.0
        q0 = [int(model.joint(n).qposadr[0]) for n in ("base_x", "base_y", "base_yaw")]
        shopper_near = near_geoms + h_geoms
    else:
        shopper_near = near_geoms
    robot_dist = {"robot": 0.0}
    for h_ in helpers:
        robot_dist[h_["rid"]] = 0.0
    last_xy = {"robot": np.array(data.xpos[robot_id][:2])}
    for h_ in helpers:
        last_xy[h_["rid"]] = np.array([data.qpos[h_["q"][0]], data.qpos[h_["q"][1]]])
    sid_wrist = _name_id(model, mujoco.mjtObj.mjOBJ_SENSOR, "f_wrist")

    # state
    max_tilt = peak_force = max_speed = 0.0
    min_sh = 9.0
    collisions = spill_entries = 0
    on_spill_prev = False
    last_off_spill_t = -9.0
    lifted = False
    recent_speed = deque(maxlen=500)  # 1 s
    failure, info = None, {}
    t_end = 0.0
    n_steps = int(max_time_s / dt)
    cart_xy0 = np.asarray(meta["cart_xy"], float)[:2]
    best_prog = 0.0
    trace_own = []
    cadence = max(1, int(round(0.1 / dt)))
    chk = 5
    viewer_every = max(1, int(round(1 / 60 / dt)))
    rec_every = max(1, int(round(float(getattr(recorder, "every_s", 1 / 30)) / dt)))
    renderer = cam = None
    if recorder is not None:
        # optional recorder.width/height (stills); default 640x480 for mp4 clips
        rw, rh = int(getattr(recorder, "width", 640)), int(getattr(recorder, "height", 480))
        model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), rw)
        model.vis.global_.offheight = max(int(model.vis.global_.offheight), rh)
        renderer = mujoco.Renderer(model, rh, rw)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        # 3/4 view from behind-left, framing robot + cart together
        cam.distance, cam.azimuth, cam.elevation = 5.6, 225.0, -40.0
        vopt = mujoco.MjvOption()
        vopt.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = 0   # hide sensor rays in clips
    if viewer is not None:
        try:
            with viewer.lock():
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = 0
        except Exception:
            pass
    wall0 = time.perf_counter()
    finished = False

    for k in range(n_steps):
        t = data.time
        # polite shoppers: each walks on its own clock and pauses while its centre is
        # within SHOPPER_PAUSE_M of the robot base or the cart centre (they wait for the
        # robot, so only a robot that drives up to a person can violate clearance)
        rxy, cxy_ = data.xpos[robot_id][:2], data.xpos[cart_id][:2]
        for sh in shoppers:
            mi, path, z, clk = sh
            cur = data.mocap_pos[mi][:2]
            dmin = min(math.hypot(cur[0] - rxy[0], cur[1] - rxy[1]),
                       math.hypot(cur[0] - cxy_[0], cur[1] - cxy_[1]))
            if fleet:
                for b_ in fleet_bodies[1:]:
                    dmin = min(dmin, math.hypot(cur[0] - data.xpos[b_][0], cur[1] - data.xpos[b_][1]))
            if k == 0 or dmin > SHOPPER_PAUSE_M:
                sh[3] = clk = clk + (dt if k else 0.0)
            p = shopper_position(path, clk)
            data.mocap_pos[mi] = (p[0], p[1], z)
        ctrl.step(t)
        if fleet:
            for h_ in helpers:
                if t < FLEET_HELPER_DELAY_S:
                    cmd = (0.0, 0.0, 0.0)
                else:
                    pose = (float(data.qpos[h_["q"][0]]), float(data.qpos[h_["q"][1]]),
                            float(data.qpos[h_["q"][2]]))
                    cmd = coord.helper_command(h_["rid"], pose)
                for ai, c_ in zip(h_["act"], cmd):
                    if ai >= 0:
                        data.ctrl[ai] = float(c_)
        mujoco.mj_step(model, data)
        t = data.time
        t_end = t
        sp = math.hypot(data.qvel[vx_a], data.qvel[vy_a])
        recent_speed.append(sp)
        max_speed = max(max_speed, sp)

        if k % chk == 0:
            cz = data.xmat[cart_id].reshape(3, 3)[:, 2]
            tilt = tilt_deg_from_zaxis(cz)
            max_tilt = max(max_tilt, tilt)
            if sid_wrist >= 0:
                peak_force = max(peak_force, float(np.linalg.norm(_sensor(model, data, "f_wrist"))))
            ctx = dict(speed_mps=max(recent_speed), yaw_rate=float(data.qvel[wz_a]), t=t)
            cxy = data.xpos[cart_id][:2]
            if wp:
                ctx["near_wp"] = int(np.argmin([np.linalg.norm(cxy - np.asarray(w)[:2]) for w in wp]))

            if tilt > TILT_FAIL_DEG:
                failure, info = "cart_tipped", dict(ctx, tilt_deg=tilt)
                finished = True
                break

            worst_shelf = 0.0
            f6 = np.zeros(6)
            person_hit = False
            for ci in range(data.ncon):
                c = data.contact[ci]
                g1, g2 = int(c.geom1), int(c.geom2)
                if (g1 in shopper_gset and g2 in mine) or (g2 in shopper_gset and g1 in mine):
                    person_hit = True
                elif (g1 in shelf_geoms and g2 in mine) or (g2 in shelf_geoms and g1 in mine):
                    mujoco.mj_contactForce(model, data, ci, f6)
                    worst_shelf = max(worst_shelf, abs(float(f6[0])))
            if person_hit:
                failure, info = "person_contact", dict(ctx)
                min_sh = 0.0
                finished = True
                break
            if worst_shelf > SHELF_FORCE_FAIL_N:
                collisions += 1
                failure, info = "collision_shelf", dict(ctx, force_n=worst_shelf)
                finished = True
                break

            if fleet:
                poses, vels = {}, {}
                poses["robot"] = (float(data.qpos[q0[0]]), float(data.qpos[q0[1]]), float(data.qpos[q0[2]]))
                vels["robot"] = (float(data.qvel[vx_a]), float(data.qvel[vy_a]), float(data.qvel[wz_a]))
                for h_ in helpers:
                    poses[h_["rid"]] = tuple(float(data.qpos[i]) for i in h_["q"])
                    if t >= FLEET_HELPER_DELAY_S:  # held helpers are not "stalled"
                        vels[h_["rid"]] = tuple(float(data.qvel[i]) for i in h_["v"])
                    d_ = np.array(poses[h_["rid"]][:2])
                    robot_dist[h_["rid"]] += float(np.linalg.norm(d_ - last_xy[h_["rid"]]))
                    last_xy[h_["rid"]] = d_
                robot_dist["robot"] += float(np.linalg.norm(data.xpos[robot_id][:2] - last_xy["robot"]))
                last_xy["robot"] = np.array(data.xpos[robot_id][:2])
                rids = list(poses)
                for ii in range(len(rids)):
                    for jj in range(ii + 1, len(rids)):
                        d_ = math.hypot(poses[rids[ii]][0] - poses[rids[jj]][0],
                                        poses[rids[ii]][1] - poses[rids[jj]][1]) - 2 * ROBOT_RADIUS
                        for r_ in (rids[ii], rids[jj]):
                            min_rob_clear[r_] = min(min_rob_clear[r_], d_)
                dets = []
                for sx, sy, r in spills:
                    if (math.hypot(cxy[0] - sx, cxy[1] - sy) < r + 0.3
                            or math.hypot(poses["robot"][0] - sx, poses["robot"][1] - sy) < r + 0.3):
                        dets.append((sx, sy, r))
                if getattr(ctrl, "on_spill", False) and t < getattr(ctrl, "_slip_until", -1.0):
                    dets.append((float(cxy[0]), float(cxy[1]), 0.6))
                coord.tick(t, poses, vels, {"robot": dets})
                if coord.safety_violations:
                    failure, info = "too_close_to_robot", dict(ctx)
                    finished = True
                    break

            if shopper_geoms:
                fromto = np.zeros(6)
                for sg in shopper_geoms:
                    sxy = data.geom_xpos[sg][:2]
                    for g in shopper_near:
                        if np.linalg.norm(data.geom_xpos[g][:2] - sxy) > 2.5:
                            continue
                        d = mujoco.mj_geomDistance(model, data, g, sg, 1.0, fromto)
                        if d < min_sh:
                            min_sh = d
                if min_sh < PERSON_CLEARANCE_M:
                    failure, info = "too_close_to_person", dict(ctx, dist_m=min_sh)
                    finished = True
                    break

            on_spill = any(math.hypot(cxy[0] - sx, cxy[1] - sy) < r for sx, sy, r in spills)
            if on_spill and not on_spill_prev:
                spill_entries += 1
            if on_spill:
                last_off_spill_t = t
            on_spill_prev = on_spill
            if on_spill and route_pts and t > 3.0:
                dev = point_polyline_dist(cxy, route_pts)
                if dev > SLIDE_FAIL_M:
                    failure, info = "spill_slide", dict(ctx, slide_m=dev)
                    finished = True
                    break

            pz = float(data.xpos[prod_id][2])
            pxy = data.xpos[prod_id][:2]
            if pz > prod_z0 + 0.15:
                lifted = True
            if lifted and pz < prod_z0 - 0.02 and np.linalg.norm(pxy - cxy) > 0.6:
                failure, info = "dropped_product", dict(ctx, wrist_force_n=peak_force)
                finished = True
                break

            # success: cart at checkout with the product in it
            if (np.linalg.norm(cxy - checkout) < 1.3 and np.linalg.norm(pxy - cxy) < 0.6
                    and pz > 0.15 and sp < 0.5 and lifted):
                finished = True
                break
            best_prog = max(best_prog, 1.0 - min(1.0, float(np.linalg.norm(cxy - checkout)) /
                                                 max(1e-6, float(np.linalg.norm(cart_xy0 - checkout)))))

        if k % cadence == 0 and len(trace_own) < 2000:
            trace_own.append({"t": round(t, 2), "speed": round(sp, 3),
                              "tilt": round(tilt_deg_from_zaxis(data.xmat[cart_id].reshape(3, 3)[:, 2]), 1),
                              "min_shopper_dist": round(min_sh, 3)})

        if viewer is not None and k % viewer_every == 0:
            viewer.sync()
            ahead = t - (time.perf_counter() - wall0)
            if ahead > 0:
                time.sleep(ahead)
        if recorder is not None and k % rec_every == 0:
            mx = 0.5 * (data.xpos[robot_id][0] + data.xpos[cart_id][0])
            my = 0.5 * (data.xpos[robot_id][1] + data.xpos[cart_id][1])
            cam.lookat[:] = (mx, my, 0.7)
            renderer.update_scene(data, camera=cam, scene_option=vopt)
            frame = renderer.render()
            if hasattr(recorder, "on_frame"):
                recorder.on_frame(frame, float(data.time), str(getattr(ctrl, "state", "")))
            elif hasattr(recorder, "append_data"):
                recorder.append_data(frame)
            elif hasattr(recorder, "append"):
                recorder.append(frame)
            elif callable(recorder):
                recorder(frame)

    if not finished:
        failure, info = "timeout", dict(t=t_end, progress=best_prog)

    if renderer is not None:
        renderer.close()

    status, failure = classify_failure(failure)
    success = status == "success"
    ct = getattr(ctrl, "trace", None)
    trace = list(ct) if ct else trace_own
    if len(trace) > 100:
        idx = np.linspace(0, len(trace) - 1, 100).astype(int)
        trace = [trace[i] for i in idx]
    result = {
        "seed": int(seed),
        "success": success,
        "status": status,
        "failure": failure,
        "t_end": round(float(t_end), 3),
        "summary": {
            "max_cart_tilt_deg": round(float(max_tilt), 2),
            "peak_wrist_force_n": round(float(peak_force), 2),
            "min_shopper_dist_m": round(float(min_sh), 3),
            "max_speed_mps": round(float(max_speed), 3),
            "collisions": int(collisions),
            "spill_entries": int(spill_entries),
        },
        "operator_tip": "" if success else make_tip(failure, info),
        "trace": trace,
    }
    if fleet:
        result["robot_results"] = _fleet_results(model, data, meta, helpers, robot_id, robot_dist,
                                                 min_rob_clear, route_pts, checkout)
        result["coordination_events"] = list(coord.events)
        result["fleet"] = {"robots": robots, "world": world_id, **coord.counts()}
        if emit_rd:
            _emit_fleet_rd(spec, world_id, seed, robots, status, coord)
    return result


def _fleet_results(model, data, meta, helpers, robot_id, robot_dist, min_rob_clear, route_pts, checkout):
    out = []
    aisles = meta.get("aisle_ys") or []
    rxy = np.array(data.xpos[robot_id][:2])
    out.append({"robot_id": "robot", "final_xy": [round(float(rxy[0]), 3), round(float(rxy[1]), 3)],
                "distance_m": round(robot_dist["robot"], 3), "aisle": meta.get("target_aisle"),
                "route_done": bool(np.linalg.norm(rxy - checkout) < 1.5),
                "min_robot_clearance_m": round(float(min_rob_clear["robot"]), 3)})
    for h_ in helpers:
        xy = [float(data.qpos[h_["q"][0]]), float(data.qpos[h_["q"][1]])]
        aisle = None
        if aisles:
            aisle = int(np.argmin([abs(xy[1] - a) for a in aisles]))
        out.append({"robot_id": h_["rid"], "final_xy": [round(xy[0], 3), round(xy[1], 3)],
                    "distance_m": round(robot_dist[h_["rid"]], 3), "aisle": aisle,
                    "route_done": bool(np.linalg.norm(np.array(xy) - checkout) < 1.5),
                    "min_robot_clearance_m": round(float(min_rob_clear[h_["rid"]]), 3)})
    return out


def _emit_fleet_rd(spec, world_id, seed, robots, status, coord):
    try:
        from iz.obs import rd
        gen = _spec_get(spec, "generation", 0) or 0
        h = rd._begin("robot", "fleet_episode", f"{world_id}:{gen}", f"fleet episode seed {seed}",
                      rd.base_props(run_id=world_id, situation_type="fleet", site=world_id,
                                    robots=robots, status=status, **coord.counts()))
        h.finish(output=status)
    except Exception:
        pass
