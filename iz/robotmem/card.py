"""Robot card: capability manifest introspected from the shop MJCF."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

import mujoco

SKILLS = ["get_cart", "push_cart", "yield", "slow_for_spill",
          "reach_and_pick", "place_in_cart", "go_to_checkout"]

_S = mujoco.mjtSensor
_SENSOR_MEANING = {
    _S.mjSENS_RANGEFINDER: "distance to nearest obstacle along site axis (m)",
    _S.mjSENS_TOUCH: "contact force on site (N)",
    _S.mjSENS_FORCE: "3-axis force at site (N)",
    _S.mjSENS_ACCELEROMETER: "linear acceleration (m/s^2)",
    _S.mjSENS_GYRO: "angular velocity (rad/s)",
    _S.mjSENS_FRAMEZAXIS: "body z-axis (tilt check)",
    _S.mjSENS_FRAMEPOS: "body world position (m)",
    _S.mjSENS_JOINTPOS: "joint angle (encoder)",
}
_TYPE_NAME = {
    _S.mjSENS_RANGEFINDER: "rangefinder", _S.mjSENS_TOUCH: "touch",
    _S.mjSENS_FORCE: "force", _S.mjSENS_ACCELEROMETER: "accelerometer",
    _S.mjSENS_GYRO: "gyro", _S.mjSENS_FRAMEZAXIS: "framezaxis",
    _S.mjSENS_FRAMEPOS: "framepos", _S.mjSENS_JOINTPOS: "jointpos",
}
_OBJ = mujoco.mjtObj


def _nm(m, obj, i):
    return mujoco.mj_id2name(m, obj, int(i)) or f"#{int(i)}"


def _default_xml():
    from iz.shop.scene import build_shop
    return build_shop(0)[0]


def _limit(lo, hi):
    return [round(float(lo), 3), round(float(hi), 3)]


def robot_card(xml: str | None = None) -> dict:
    from iz.shop.controller import LOCKED, PARAM_BOUNDS
    if xml is None:
        xml = _default_xml()
    m = mujoco.MjModel.from_xml_string(xml)

    sensors = []
    for i in range(m.nsensor):
        t = _S(int(m.sensor_type[i]))
        ot = int(m.sensor_objtype[i])
        site = ""
        if ot in (_OBJ.mjOBJ_SITE, _OBJ.mjOBJ_JOINT, _OBJ.mjOBJ_BODY):
            site = _nm(m, ot, m.sensor_objid[i])
        extra = ""
        if t == _S.mjSENS_RANGEFINDER and m.sensor_cutoff[i] > 0:
            extra = f", cutoff {float(m.sensor_cutoff[i]):g}"
        sensors.append({"name": _nm(m, _OBJ.mjOBJ_SENSOR, i),
                        "type": _TYPE_NAME.get(t, t.name.replace("mjSENS_", "").lower()),
                        "site": site,
                        "meaning": _SENSOR_MEANING.get(t, "sensor reading") + extra})

    actuators = []
    for i in range(m.nu):
        jname = _nm(m, _OBJ.mjOBJ_JOINT, m.actuator_trnid[i][0])
        # velocity actuator: biasprm = [0, 0, -kv]; position: [0, -kp, 0]
        is_vel = m.actuator_biasprm[i][1] == 0 and m.actuator_biasprm[i][2] != 0
        ctrl = _limit(*m.actuator_ctrlrange[i]) if m.actuator_ctrllimited[i] else None
        actuators.append({
            "name": _nm(m, _OBJ.mjOBJ_ACTUATOR, i), "joint": jname,
            "kind": "velocity" if is_vel else "position",
            "limit": ctrl, "ctrl_limit": ctrl,
            "force_limit": _limit(*m.actuator_forcerange[i]) if m.actuator_forcelimited[i] else None})

    joints = []
    for j in range(m.njnt):
        jt = mujoco.mjtJoint(int(m.jnt_type[j])).name.replace("mjJNT_", "").lower()
        if jt == "free":
            continue
        d = {"name": _nm(m, _OBJ.mjOBJ_JOINT, j), "type": jt}
        if m.jnt_limited[j]:
            d["range"] = _limit(*m.jnt_range[j])
        joints.append(d)

    dims = {}
    for b in ("robot", "torso", "head"):
        bid = mujoco.mj_name2id(m, _OBJ.mjOBJ_BODY, b)
        if bid >= 0:
            dims[b] = {"mass_kg": round(float(m.body_subtreemass[bid]), 2)}
    geoms = {}
    for g in ("base_g", "torso_g", "head_g"):
        gid = mujoco.mj_name2id(m, _OBJ.mjOBJ_GEOM, g)
        if gid >= 0:
            geoms[g] = [round(float(x), 3) for x in m.geom_size[gid]]
    dims["geom_half_sizes_m"] = geoms

    card = {
        "name": "IZ-1 retail humanoid",
        "body": "torso+head+2x2-DoF arms on velocity base",
        "dims": dims,
        "sensors": sensors,
        "actuators": actuators,
        "joints": joints,
        "skills": list(SKILLS),
        "tunable_params": {k: list(v) for k, v in PARAM_BOUNDS.items()},
        "locked": dict(LOCKED),
    }
    card["hash"] = hashlib.sha256(
        json.dumps(card, sort_keys=True, default=str).encode()).hexdigest()[:16]
    return card


def _ntok(s: str) -> int:
    return (len(s) + 3) // 4


def card_text(card: dict, budget_tokens: int = 350) -> str:
    """Compact prompt text; guaranteed <= budget_tokens (len/4 estimate)."""
    acts = card["actuators"]
    base = [a for a in acts if a["kind"] == "velocity"]
    arm = [a for a in acts if a["kind"] != "velocity"]
    by_type: dict[str, list[str]] = {}
    meaning = {}
    for s in card["sensors"]:
        by_type.setdefault(s["type"], []).append(s["name"])
        meaning[s["type"]] = s["meaning"]
    head = f"ROBOT {card['name']}: {card['body']}."
    drive = "Drive: " + ", ".join(f"{a['name']} vel {a['limit']}" for a in base)
    arms = ("Arms: " + ", ".join(
        f"{a['joint']}<={a['force_limit'][1] if a['force_limit'] else '?'}N" for a in arm)) if arm else ""
    sens_full = "Sensors: " + "; ".join(
        f"{t}({','.join(n)}) {meaning[t]}" for t, n in by_type.items())
    sens_short = "Sensors: " + ", ".join(f"{t}x{len(n)}" for n, (t, n) in
                                          zip(range(len(by_type)), by_type.items()))
    skills = "Skills: " + ", ".join(card["skills"])
    tun = "Tunable: " + ", ".join(
        f"{k} {v[0]:g}-{v[1]:g}" for k, v in card["tunable_params"].items())
    lock = "LOCKED (never change): " + ", ".join(f"{k}={v}" for k, v in card["locked"].items())
    for sens in (sens_full, sens_short):
        txt = "\n".join(x for x in (head, drive, arms, sens, skills, tun, lock) if x)
        if _ntok(txt) <= budget_tokens:
            return txt
    txt = "\n".join(x for x in (head, sens_short, skills, tun, lock) if x)
    if _ntok(txt) <= budget_tokens:
        return txt
    # last resort: keep the tail (tunable + LOCKED) intact, truncate the rest
    tail = "\n".join((tun, lock))
    room = max(0, budget_tokens * 4 - len(tail) - 1)
    return (head[:room] + "\n" + tail)[: budget_tokens * 4]


def store_card(card: dict, db=None) -> dict:
    """Upsert into Atlas 'robot_cards' keyed by name+hash."""
    if db is None:
        from iz.store.db import get_db
        db = get_db()
    db["robot_cards"].update_one(
        {"name": card["name"], "hash": card["hash"]},
        {"$set": dict(card), "$setOnInsert": {"created": datetime.now(timezone.utc)}},
        upsert=True)
    return {"name": card["name"], "hash": card["hash"]}
