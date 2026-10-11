"""SHOP robot controller: a sensor-driven state machine whose PARAMETERS are the genome.

Honest scope: the evolvable genome is driving policy (speeds, distances, force limit,
reach height). Motor control is hand-written; the robot learns *how to drive* from
takeover evidence.

Sensing
-------
Everything comes from ``data.sensordata`` (via ``model.sensor_adr``) except these
documented stand-ins:
  * person detector : world positions of ``shopper_*`` mocap bodies (``data.mocap_pos``)
  * floor sensor    : known spill positions from ``meta["spills"]`` plus a cart-slip
                      detector (lateral cart velocity from the ``cart_pos`` sensor)
  * product finder  : world position of the target product body (head-camera stand-in)
  * encoders / FK   : joint positions come from jointpos sensors when present (qpos
                      fallback); hand position is forward kinematics of those angles.
"""
from __future__ import annotations

import math
from types import MappingProxyType

import mujoco
import numpy as np

DEFAULT_PARAMS = {
    "cruise_speed": 1.4,
    "turn_speed": 1.3,
    "turn_slowdown_deg": 999,
    "brake_distance_m": 0.35,
    "yield_distance_m": 0.6,
    "spill_speed": 1.4,
    "push_force_limit_n": 220,
    "reach_height_m": 1.0,
    "gait_cadence_hz": 1.6,    # G1 gait engine (iz/robot/gait.py); no effect on capsule models
    "stride_scale": 0.8,
}

PARAM_BOUNDS = {
    "cruise_speed": (0.2, 1.6),
    "turn_speed": (0.15, 1.6),
    "turn_slowdown_deg": (10, 999),
    "brake_distance_m": (0.2, 1.5),
    "yield_distance_m": (0.4, 2.0),
    "spill_speed": (0.1, 1.6),
    "push_force_limit_n": (60, 300),
    "reach_height_m": (0.5, 1.5),
    "gait_cadence_hz": (1.0, 2.4),
    "stride_scale": (0.4, 1.2),
}

# Never evolvable. Enforced in ShopController (params cannot touch these).
LOCKED = MappingProxyType({
    "min_person_clearance_m": 0.35,
    "max_wrist_force_n": 300.0,
})

RF_MAX = 10.0
YAW_RATE_MAX = 1.5
TRACE_DT = 0.1
ARM_RATE = 1.6          # rad/s ctrl slew for arm targets
ACCEL = 2.0             # m/s^2 speed ramp-up (braking is immediate)
GRASP_DIST = 0.9        # base-to-cart distance at which the cart is gripped
WP_RADIUS = 0.45
ROTATE_DEG = 50.0       # heading error above which the base pivots in place

GO_TO_CART, GRASP_CART, PUSH, AT_SHELF, REGRASP, DONE = (
    "GO_TO_CART", "GRASP_CART", "PUSH", "AT_SHELF", "RE-GRASP", "DONE")


def clamp_params(params):
    """Merge over defaults, clamp to bounds; return (clean, ignored_keys)."""
    clean = dict(DEFAULT_PARAMS)
    ignored = []
    for k, v in (params or {}).items():
        if k in LOCKED or k not in DEFAULT_PARAMS:
            ignored.append(k)
            continue
        lo, hi = PARAM_BOUNDS[k]
        try:
            clean[k] = min(max(float(v), lo), hi)
        except (TypeError, ValueError):
            ignored.append(k)
    return clean, ignored


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class ShopController:
    def __init__(self, model, data, meta, params=None):
        self.model, self.data, self.meta = model, data, meta
        self.params, self.ignored_params = clamp_params(params)
        self.locked = LOCKED
        m = model
        self._nm = lambda t, n: mujoco.mj_name2id(m, t, n)
        O = mujoco.mjtObj

        # --- sensors (name -> (adr, dim)) ---
        self._sens = {}
        for i in range(m.nsensor):
            n = mujoco.mj_id2name(m, O.mjOBJ_SENSOR, i)
            if n:
                self._sens[n] = (int(m.sensor_adr[i]), int(m.sensor_dim[i]))
        # jointpos sensors by joint id
        self._jp = {}
        for i in range(m.nsensor):
            if m.sensor_type[i] == mujoco.mjtSensor.mjSENS_JOINTPOS:
                self._jp[int(m.sensor_objid[i])] = int(m.sensor_adr[i])

        # --- actuators / joints ---
        self._act = {n: self._nm(O.mjOBJ_ACTUATOR, n)
                     for n in ("act_vx", "act_vy", "act_wz", "act_r_shoulder", "act_r_elbow",
                               "act_l_shoulder", "act_l_elbow")}
        for n in ("act_vx", "act_vy", "act_wz"):
            if self._act[n] < 0:
                raise ValueError(f"missing actuator {n}")
        self._jnt = {n: self._nm(O.mjOBJ_JOINT, n)
                     for n in ("base_x", "base_y", "base_yaw", "r_shoulder", "r_elbow")}
        # push_force_limit_n -> force range of the base drive (physical force limit)
        fl = float(self.params["push_force_limit_n"])
        for n in ("act_vx", "act_vy"):
            a = self._act[n]
            m.actuator_forcelimited[a] = 1
            m.actuator_forcerange[a] = (-fl, fl)

        # --- equalities ---
        self._eq_cart = self._nm(O.mjOBJ_EQUALITY, "grip_cart")
        self._eq_prod = self._nm(O.mjOBJ_EQUALITY, "grip_prod")
        self._hand_site = self._nm(O.mjOBJ_SITE, "r_hand")
        self._cart_body = self._nm(O.mjOBJ_BODY, "cart")
        self._prod_body = self._nm(O.mjOBJ_BODY, meta["target_product"])

        # --- person detector ---
        self._shoppers = []
        for i in range(m.nbody):
            n = mujoco.mj_id2name(m, O.mjOBJ_BODY, i)
            if n and n.startswith("shopper_") and m.body_mocapid[i] >= 0:
                self._shoppers.append(int(m.body_mocapid[i]))

        # --- floor sensor (known spills) ---
        self._spills = []
        for s in meta.get("spills", []) or []:
            if isinstance(s, dict):
                self._spills.append((float(s["x"]), float(s["y"]), float(s.get("r", 0.6))))
            else:
                self._spills.append((float(s[0]), float(s[1]), float(s[2]) if len(s) > 2 else 0.6))

        # --- route ---
        wps = [np.array(w[:2], float) for w in meta["waypoints"]]
        route = [wps[i] for i in meta["route"]]
        pp = np.array(data.xpos[self._prod_body][:2])
        self._shelf_k = int(np.argmin([np.linalg.norm(w - pp) for w in route])) if route else 0
        self._route_a = route[: self._shelf_k + 1]
        self._route_b = route[self._shelf_k + 1:]
        cx = np.array(meta["checkout_xy"][:2], float)
        if not self._route_b or np.linalg.norm(self._route_b[-1] - cx) > 0.6:
            self._route_b.append(cx)
        self._checkout = cx
        self._grasp_dist = float(meta.get("grasp_dist_m", GRASP_DIST))
        self._cart_off = np.zeros(2)

        # --- state ---
        self.state = GO_TO_CART
        self._phase = "REACH"           # sub-phase inside AT_SHELF
        self._leg = 0
        self._wp = 0
        self._t_state = 0.0
        self._last_t = None
        self._speed = 0.0
        self.speed_cmd = 0.0
        self.yielding = False
        self.braking = False
        self.on_spill = False
        self.done = False
        self.trace = []
        self._next_trace = 0.0
        self._arm_ctrl = np.zeros(2)
        self._arm_goal = np.zeros(2)
        self._last_solve = -1.0
        self._scratch = mujoco.MjData(m)
        self._prev_cart = None
        self._slip_until = -1.0
        self.min_shopper_dist = float("inf")
        self._brake_t = 0.0
        self._pushed_once = False
        self._face_phase = None
        self._face = 0.0
        self._hand_rel = np.array([0.5, 0.0])

        # --- G1 legs (only when leg actuators exist; capsule models have none) ---
        self._gait_acts = {}
        for i in range(m.nu):
            an = mujoco.mj_id2name(m, O.mjOBJ_ACTUATOR, i)
            if an and an.startswith("act_") and an.endswith("_joint") and \
                    any(k in an for k in ("hip_pitch", "hip_roll", "knee", "ankle_pitch")):
                self._gait_acts[an[4:]] = i
        self._gait = None
        if self._gait_acts:
            from iz.robot.gait import GaitEngine
            self._gait = GaitEngine({k: self.params[k] for k in ("gait_cadence_hz", "stride_scale")})

    # ------------------------------------------------------------ sensors
    def read(self, name):
        """Raw sensor vector from data.sensordata, or None if the sensor is absent."""
        if name not in self._sens:
            return None
        a, d = self._sens[name]
        return np.array(self.data.sensordata[a:a + d])

    def rf(self, name):
        v = self.read(name)
        if v is None or not np.isfinite(v[0]) or v[0] < 0:
            return RF_MAX
        return float(min(v[0], RF_MAX))

    def wrist_force(self):
        v = self.read("f_wrist")
        return float(np.linalg.norm(v)) if v is not None else 0.0

    def touch(self):
        v = self.read("touch_r")
        return float(v[0]) if v is not None else 0.0

    def cart_tilt_deg(self):
        v = self.read("cart_up")
        if v is None:
            return 0.0
        return math.degrees(math.acos(max(-1.0, min(1.0, float(v[2])))))

    def cart_pos(self):
        v = self.read("cart_pos")
        return np.array(v if v is not None else self.data.xpos[self._cart_body])

    def _joint(self, name):
        j = self._jnt[name]
        if j in self._jp:
            return float(self.data.sensordata[self._jp[j]])
        return float(self.data.qpos[self.model.jnt_qposadr[j]])

    def base_pose(self):
        return (np.array([self._joint("base_x"), self._joint("base_y")]), self._joint("base_yaw"))

    def shopper_positions(self):
        return [np.array(self.data.mocap_pos[i][:2]) for i in self._shoppers]

    def min_person_dist(self):
        """Closest shopper centre to the robot base, cart centre or the point between."""
        if not self._shoppers:
            return float("inf")
        p, _ = self.base_pose()
        c = self.cart_pos()[:2]
        pts = [p, c, 0.5 * (p + c)]
        return float(min(np.linalg.norm(s - q) for s in self.shopper_positions() for q in pts))

    def _near_spill(self, margin=0.6):
        c = self.cart_pos()[:2]
        return any(math.hypot(c[0] - x, c[1] - y) < r + margin for x, y, r in self._spills)

    # ------------------------------------------------------------ helpers
    def _set_state(self, s, t):
        self.state, self._t_state = s, t

    def _grip(self, eq, active):
        """Switch a weld on at the CURRENT relative pose (so nothing jumps) or off."""
        m, d = self.model, self.data
        if eq < 0:
            return
        if active:
            b1, b2 = int(m.eq_obj1id[eq]), int(m.eq_obj2id[eq])
            p1, R1 = d.xpos[b1], d.xmat[b1].reshape(3, 3)
            rel = R1.T @ (d.xpos[b2] - p1)
            q1inv = np.zeros(4)
            mujoco.mju_negQuat(q1inv, d.xquat[b1])
            qrel = np.zeros(4)
            mujoco.mju_mulQuat(qrel, q1inv, d.xquat[b2])
            m.eq_data[eq, 0:3] = 0.0
            m.eq_data[eq, 3:6] = rel
            m.eq_data[eq, 6:10] = qrel
            if eq == self._eq_cart and m.eq_data.shape[1] > 10:
                # stable pushing: full rotational coupling so the cart stays rigidly
                # aligned behind the hands instead of pivoting/jackknifing on the grip
                m.eq_data[eq, 10] = 1.0
        d.eq_active[eq] = 1 if active else 0

    def _drive(self, target, speed, yaw_rate_max=YAW_RATE_MAX):
        """Unicycle: heading control + forward speed -> world vx, vy, wz."""
        p, yaw = self.base_pose()
        vec = target - p
        desired = math.atan2(vec[1], vec[0])
        err = _wrap(desired - yaw)
        wz = max(-yaw_rate_max, min(yaw_rate_max, 2.0 * err))
        if abs(err) > math.radians(ROTATE_DEG):
            speed = 0.0          # pivot in place: a pushed cart cannot follow a sharp arc
        return speed, err, wz

    def _apply_base(self, speed, yaw, wz):
        d = self.data
        d.ctrl[self._act["act_vx"]] = speed * math.cos(yaw)
        d.ctrl[self._act["act_vy"]] = speed * math.sin(yaw)
        d.ctrl[self._act["act_wz"]] = wz

    def _stop_base(self):
        for n in ("act_vx", "act_vy", "act_wz"):
            self.data.ctrl[self._act[n]] = 0.0
        self._speed = 0.0
        self.speed_cmd = 0.0

    def _front_hazard(self, pushing):
        """Forward clearance. rf_front sees the cart while pushing, so ignore it then."""
        d_cart = self.rf("rf_cart") if pushing else RF_MAX
        d_front = self.rf("rf_front")
        if pushing:
            p, _ = self.base_pose()
            drc = float(np.linalg.norm(self.cart_pos()[:2] - p))
            if d_front < drc + 0.5:
                d_front = RF_MAX
        return min(d_cart, d_front)

    # ------------------------------------------------------------ arm
    def _solve_arm(self, target_z, fwd_pref=0.55):
        """Grid IK on a scratch copy: choose (shoulder, elbow) so the hand is at height
        target_z with a comfortable forward extension fwd_pref in the robot frame.
        Returns (z_error, q1, q2, hand_xy_in_robot_frame)."""
        m = self.model
        j1, j2 = self._jnt["r_shoulder"], self._jnt["r_elbow"]
        a1, a2 = int(m.jnt_qposadr[j1]), int(m.jnt_qposadr[j2])
        rng = [tuple(m.jnt_range[j]) if m.jnt_limited[j] else (-1.5, 1.5) for j in (j1, j2)]
        s = self._scratch
        s.qpos[:] = self.data.qpos
        base, yaw = self.base_pose()
        c, sn = math.cos(yaw), math.sin(yaw)
        base3 = np.array([base[0], base[1], 0.0])

        def evalq(q1, q2):
            s.qpos[a1], s.qpos[a2] = q1, q2
            mujoco.mj_kinematics(m, s)
            h = s.site_xpos[self._hand_site]
            dx, dy = h[0] - base[0], h[1] - base[1]
            rel = np.array([c * dx + sn * dy, -sn * dx + c * dy])
            return 3.0 * abs(h[2] - target_z) + abs(rel[0] - fwd_pref), rel, abs(h[2] - target_z)

        best = (1e9, 0.0, 0.0, None, 0.0)
        for q1 in np.linspace(rng[0][0], rng[0][1], 21):
            for q2 in np.linspace(rng[1][0], rng[1][1], 21):
                cst, rel, ez = evalq(q1, q2)
                if cst < best[0]:
                    best = (cst, q1, q2, rel, ez)
        _, b1, b2, _, _ = best
        h1 = (rng[0][1] - rng[0][0]) / 20
        h2 = (rng[1][1] - rng[1][0]) / 20
        for q1 in np.linspace(max(rng[0][0], b1 - h1), min(rng[0][1], b1 + h1), 7):
            for q2 in np.linspace(max(rng[1][0], b2 - h2), min(rng[1][1], b2 + h2), 7):
                cst, rel, ez = evalq(q1, q2)
                if cst < best[0]:
                    best = (cst, q1, q2, rel, ez)
        return best[4], best[1], best[2], best[3]

    def hand_pos(self):
        return np.array(self.data.site_xpos[self._hand_site])

    def _drive_arm(self, goal, dt):
        """Slew arm position targets; freeze if wrist force exceeds the locked limit."""
        if self.wrist_force() > LOCKED["max_wrist_force_n"]:
            goal = self._arm_ctrl.copy()
        step = ARM_RATE * dt
        self._arm_ctrl = self._arm_ctrl + np.clip(np.asarray(goal) - self._arm_ctrl, -step, step)
        for n, v in (("act_r_shoulder", self._arm_ctrl[0]), ("act_r_elbow", self._arm_ctrl[1])):
            if self._act[n] >= 0:
                self.data.ctrl[self._act[n]] = v
        for n in ("act_l_shoulder", "act_l_elbow"):
            if self._act[n] >= 0:
                self.data.ctrl[self._act[n]] = 0.0

    # ------------------------------------------------------------ main step
    def step(self, t):
        d = self.data
        dt = 0.002 if self._last_t is None else max(1e-4, min(0.05, t - self._last_t))
        self._last_t = t
        P = self.params
        base, yaw = self.base_pose()
        self.min_shopper_dist = self.min_person_dist()
        eff_yield = max(P["yield_distance_m"], LOCKED["min_person_clearance_m"])
        self.yielding = self.min_shopper_dist < eff_yield and self.state != DONE

        # cart-slip floor sensor
        cp = self.cart_pos()
        if self._prev_cart is not None and self.state == PUSH:
            vel = (cp[:2] - self._prev_cart) / dt
            h = np.array([math.cos(yaw), math.sin(yaw)])
            lat = abs(vel[0] * -h[1] + vel[1] * h[0])
            if lat > 0.3:
                self._slip_until = t + 1.0
        self._prev_cart = cp[:2].copy()
        self.on_spill = self._near_spill() or t < self._slip_until

        if self.state == GO_TO_CART or self.state == REGRASP:
            self._go_to_cart(t, dt, base, yaw)
        elif self.state == GRASP_CART:
            self._stop_base()
            self._drive_arm(np.zeros(2), dt)
            if t - self._t_state > 0.3:
                self._grip(self._eq_cart, True)
                self._set_state(PUSH, t)
                self._wp = 0
                self._pushed_once = False
        elif self.state == PUSH:
            self._push(t, dt, base, yaw)
        elif self.state == AT_SHELF:
            self._at_shelf(t, dt, base, yaw)
        elif self.state == DONE:
            self._stop_base()
            self._drive_arm(np.zeros(2), dt)
            self.done = True

        if self._gait_acts:
            self._apply_gait(dt)

        if t >= self._next_trace:
            self._next_trace = t + TRACE_DT
            self.trace.append({
                "t": round(t, 2), "state": self.state,
                "speed_cmd": round(self.speed_cmd, 3),
                "rf_front": round(self.rf("rf_front"), 3),
                "rf_cart": round(self.rf("rf_cart"), 3),
                "cart_tilt_deg": round(self.cart_tilt_deg(), 2),
                "wrist_force": round(self.wrist_force(), 2),
                "min_shopper_dist": round(self.min_shopper_dist, 3) if np.isfinite(self.min_shopper_dist) else None,
            })

    def _apply_gait(self, dt):
        """Parametric gait engine (NOT a learned policy) drives the G1 leg actuators.
        Arm swing only while walking to the cart, never while pushing or reaching."""
        d = self.data
        wz = float(d.ctrl[self._act["act_wz"]])
        targets = self._gait.step(dt, float(self.speed_cmd), wz)
        for j, a in self._gait_acts.items():
            if j in targets:
                d.ctrl[a] = targets[j]
        if self.state == GO_TO_CART:
            for n in ("l_shoulder", "r_shoulder"):
                a = self._act.get("act_" + n, -1)
                if a >= 0 and n in targets:
                    d.ctrl[a] = targets[n]

    # ---------------------------------------------------------- states
    def _go_to_cart(self, t, dt, base, yaw):
        self._drive_arm(np.zeros(2), dt)
        cp = self.cart_pos()[:2]
        dist = float(np.linalg.norm(cp - base))
        if dist < self._grasp_dist:
            self._stop_base()
            self._set_state(GRASP_CART, t)
            return
        if self.yielding:
            self._stop_base()
            return
        speed, err, wz = self._drive(cp, 0.6)
        self._speed = speed
        self.speed_cmd = speed
        self._apply_base(speed, yaw, wz)

    def _push(self, t, dt, base, yaw):
        P = self.params
        route = self._route_a if self._leg == 0 else self._route_b
        # arms stowed
        self._drive_arm(np.zeros(2), dt)
        if not self._pushed_once:
            # drop leading waypoints that lie behind the cart (e.g. the start node)
            self._pushed_once = True
            fwd = self.cart_pos()[:2] - base
            # only the start node (grip pose) is skipped; aisle entry nodes are kept even
            # when they lie off to the side, otherwise the cart is cut across a shelf end
            if self._leg == 0 and len(route) > 1 and float(np.dot(route[0] - base, fwd)) < 0.3:
                self._wp = 1
        while self._wp < len(route) - 1 and np.linalg.norm(route[self._wp] - base) < WP_RADIUS:
            self._wp += 1
        target = route[self._wp]
        last = self._wp == len(route) - 1
        arrive_r = WP_RADIUS if not last else 0.35
        if last and np.linalg.norm(target - base) < arrive_r + (0.6 if self._leg == 1 else 0.0):
            if self._leg == 0:
                self._stop_base()
                self._grip(self._eq_cart, False)
                self._set_state(AT_SHELF, t)
                self._phase = "REACH"
                self._last_solve = -1.0
            else:
                self._stop_base()
                self._set_state(DONE, t)
            return

        speed, err, wz = self._drive(target, P["cruise_speed"])
        if abs(math.degrees(err)) > P["turn_slowdown_deg"]:
            speed = min(speed, P["turn_speed"])
        if self.on_spill:
            speed = min(speed, P["spill_speed"])
        # obstacle braking (rangefinders)
        self.braking = False
        hz = self._front_hazard(pushing=True)
        bd = P["brake_distance_m"]
        if hz < bd:
            speed, self.braking = 0.0, True
        elif hz < 2 * bd:
            speed *= (hz - bd) / bd
            self.braking = True
        # a full stop that lasts too long becomes a slow creep (shoppers wait for the robot,
        # so waiting forever is a deadlock); the creep still respects the locked clearance
        if self.yielding or (self.braking and speed == 0.0):
            self._brake_t += dt
        else:
            self._brake_t = 0.0
        if self.yielding:
            speed = 0.0
            wz = 0.0
            # only shoppers AHEAD of the cart block; one beside/behind may be passed slowly
            h = np.array([math.cos(yaw), math.sin(yaw)])
            cxy = self.cart_pos()[:2]
            ahead = any(float((s - cxy) @ h) > -0.2 for s in self.shopper_positions()
                        if np.linalg.norm(s - cxy) < P["yield_distance_m"] + 0.6)
            if self._brake_t > 1.5 and not ahead and \
                    self.min_shopper_dist > LOCKED["min_person_clearance_m"] + 0.6:
                speed = 0.2
        # ramp up only
        if speed > self._speed:
            speed = min(speed, self._speed + ACCEL * dt)
        self._speed = speed
        self.speed_cmd = speed
        self._apply_base(speed, yaw, wz)
        # cross-track correction: hold robot AND cart on the segment centreline (aisle
        # centre) using the holonomic base, so the rigid cart never drifts into a shelf
        prev = route[self._wp - 1] if self._wp > 0 else None
        if prev is not None and speed > 0.0:
            seg = target - prev
            L = float(np.linalg.norm(seg))
            if L > 1e-6:
                u = seg / L
                nrm = np.array([-u[1], u[0]])
                cxy = self.cart_pos()[:2]
                e = 0.5 * float((base - prev) @ nrm) + 0.5 * float((cxy - prev) @ nrm)
                vl = max(-0.35, min(0.35, -1.5 * e))
                d = self.data
                d.ctrl[self._act["act_vx"]] += vl * nrm[0]
                d.ctrl[self._act["act_vy"]] += vl * nrm[1]

    def _at_shelf(self, t, dt, base, yaw):
        d = self.data
        prod = np.array(d.xpos[self._prod_body])
        self.speed_cmd = 0.0
        if self._phase == "REACH":
            tgt = np.array([prod[0], prod[1], self.params["reach_height_m"]])
            self._servo_hand(t, dt, tgt, base, yaw)
            hand = self.hand_pos()
            if not hasattr(self, "_t_reach") or self._t_reach is None:
                self._t_reach = t
            dh = float(np.linalg.norm(hand - prod))
            if (self.touch() > 0.5 and dh < 0.25) or dh < 0.12 or \
                    (t - self._t_reach > 4.0 and dh < 0.4):   # within reach: close the grip
                self._grip(self._eq_prod, True)
                self._phase = "PLACE"
                self._last_solve = -1.0
                self._t_phase = t
        elif self._phase == "PLACE":
            cp = self.cart_pos()
            tgt = np.array([cp[0], cp[1], cp[2] + 0.9])
            self._servo_hand(t, dt, tgt, base, yaw, 0.45)
            near = np.linalg.norm(self.hand_pos() - tgt)
            if near < 0.15 or (t - self._t_phase > 6.0 and near < 0.35):
                self._grip(self._eq_prod, False)
                self._phase = "RELEASE"
                self._t_phase = t
        else:  # RELEASE: let product fall, stow arm, then re-grasp the cart
            self._stop_base()
            self._drive_arm(np.zeros(2), dt)
            if t - self._t_phase > 0.8:
                self._leg = 1
                self._wp = 0
                self._set_state(REGRASP, t)

    def _servo_hand(self, t, dt, target, base, yaw, fwd_pref=0.55):
        """Arm IK (height + comfortable extension) and base servo so the hand lands on target."""
        if self._face_phase != self._phase:
            # heading latched per phase so the arm plane does not swing while closing in
            self._face_phase = self._phase
            self._face = math.atan2(target[1] - base[1], target[0] - base[0])
        if t - self._last_solve >= 0.25:
            ez, q1, q2, rel = self._solve_arm(target[2], fwd_pref)
            self._arm_goal = np.array([q1, q2])
            self._hand_rel = rel
            self._last_solve = t
        self._drive_arm(self._arm_goal, dt)
        cf, sf = math.cos(self._face), math.sin(self._face)
        rel = self._hand_rel
        off = np.array([cf * rel[0] - sf * rel[1], sf * rel[0] + cf * rel[1]])
        goal = target[:2] - off
        err = goal - base
        if self._phase == "REACH" and len(self._route_a) >= 2:
            # at the shelf the base only slides along the aisle centreline (never into the
            # shelf); the arm does the reaching across to the product
            ax = self._route_a[-1] - self._route_a[-2]
            nax = float(np.linalg.norm(ax))
            if nax > 1e-6:
                ax = ax / nax
                err = ax * float(err @ ax)
        n = float(np.linalg.norm(err))
        v = err / n * min(0.6, 1.5 * n) if (n > 0.03 and not self.yielding) else np.zeros(2)
        d = self.data
        d.ctrl[self._act["act_vx"]] = v[0]
        d.ctrl[self._act["act_vy"]] = v[1]
        d.ctrl[self._act["act_wz"]] = max(-1.0, min(1.0, 2.0 * _wrap(self._face - yaw)))
