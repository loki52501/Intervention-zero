"""Single prompt -> validated WorldSpec, plus seeded variations for train / held-out sets."""
from __future__ import annotations

import json
import math
import random
import re
from typing import Any

from pydantic import ValidationError

from iz.worldgen import spec as S
from iz.worldgen.spec import WorldSpec, validate_spec

SYSTEM = f"""You design small robot-training worlds (a wheeled robot with a shopping cart) from one user prompt.
Return ONE JSON object matching the schema. Vocabulary and hard limits:
- venue: one of {list(S.VENUES)}.
- size_m = [width, height] metres; width 6..20, height 5..15. Origin is the bottom-left corner.
- aisles 1..5. Layout: the bottom {S.CORRIDOR_Y} m is a free corridor (robot starts at (0.8,0.8), checkout 'auto'
  is at (width-1, 0.9)); above it, shelf rows (depth {S.SHELF_DEPTH} m) alternate with aisles. Aisle width is
  (height - {S.CORRIDOR_Y} - {S.TOP_MARGIN} - (aisles+1)*{S.SHELF_DEPTH}) / aisles and MUST be >= {S.MIN_AISLE_W} m,
  so tall rooms are needed for many aisles (e.g. 3 aisles needs height >= 7.9).
- shelf_units: number of {S.UNIT_LEN} m shelf units (1..40); it must fit: floor((width-2)/{S.UNIT_LEN})*(aisles+1).
  Shelf unit j is in row j % (aisles+1); products reference shelf_index in [0, shelf_units).
- products: list of {{name, color, shelf_index, height_m 0.2..1.8}}; target_product must equal one product name.
- checkout: "auto" or {{x, y}} inside the room and not on a shelf.
- spills: list of {{x, y, radius 0.1..0.6}} slippery hazards (metres, inside the room, must not seal every route).
- shoppers: {{count 0..8, speed_mps 0.3..1.4, density low|medium|rush_hour}}.
- cart: {{mass_kg 8..40, loaded bool}}.
- task: ordered subset of {list(S.STEPS)}; normally all five in that order.
- success: {{max_time_s 30..600}}.
- notes: short list of strings (assumptions you made).
Map vague words to numbers (crowded -> more shoppers, rush_hour; heavy cart -> high mass_kg; wet floor -> spills).
V2 vocabulary (all optional):
- template: one of {list(S.TEMPLATES)}; venue may be free text.
- terrain: {{type flat|rough|slope|steps|ramp, roughness_m 0..0.06, slope_deg 0..15, step_h_m 0..0.12, step_n 0..6}}.
- obstacles: [{{kind box|barrier|door, x, y, w, l, h, yaw, mass_kg?, open?}}] (never on the start, cart, product or checkout).
- hazards: [{{kind spill|low_light|cable, x, y, radius}}].
- people: {{count 0..8, speed_mps, density}}.  lighting: {{ambient 0.2..1.0}}.
- dr: {{friction [lo,hi], mass_scale [lo,hi], push {{interval_s, speed_mps}}, joint_noise_rad, action_latency_steps [lo,hi]}}.
- robots: 1..3.  tasks: [{{robot, steps from {list(S.V2_STEPS)}}}].
WORD -> NUMBER:
- "icy" or "loading dock" -> dr.friction [0.2,0.5]
- "wet" -> spill hazards
- "ramp" -> terrain ramp, slope_deg 6..10
- "slope/hill" -> slope 8..12
- "rough/gravel/uneven" -> rough, roughness_m 0.02..0.05
- "stairs/steps" -> steps, step_h_m 0.08..0.12, step_n 3..5
- "construction site" -> steps + box obstacles + cable hazards + a low_light hazard + lighting.ambient 0.3
- "dark/night" -> ambient 0.25
- "fire door/closed door" -> a door obstacle open=false
- "crowded" -> people count 5..8
- "pushy/windy" -> dr.push speed_mps [0.5,1.0]
- "fleet/team/two robots" -> robots 2..3
Output only the JSON object."""


def _f(v: Any, lo: float, hi: float, default: float) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return default
    if math.isnan(x):
        return default
    return min(hi, max(lo, x))


def _i(v: Any, lo: int, hi: int, default: int) -> int:
    return int(round(_f(v, lo, hi, default)))


def _num(v: Any, default: float = 0.0) -> float:
    try:
        x = float(v)
        return default if math.isnan(x) else x
    except (TypeError, ValueError):
        return default


def sanitize(raw: dict[str, Any]) -> tuple[WorldSpec, list[str]]:
    """Clamp an LLM dict into pydantic-valid ranges (positions are left for validate_spec)."""
    notes: list[str] = []
    raw = dict(raw) if isinstance(raw, dict) else {}
    d: dict[str, Any] = {}
    d["name"] = str(raw.get("name") or "world")[:60]
    v = raw.get("venue")
    if v in S.VENUES:
        d["venue"] = v
    elif isinstance(v, str) and v.strip():
        d["venue"] = v.strip()[:60]
    else:
        d["venue"] = "custom"
    tpl = raw.get("template")
    d["template"] = tpl if tpl in S.TEMPLATES else (d["venue"] if d["venue"] in S.VENUES else "retail")
    size = raw.get("size_m")
    if isinstance(size, (list, tuple)) and len(size) == 2:
        d["size_m"] = [_f(size[0], 6, 20, 12), _f(size[1], 5, 15, 8)]
        if d["size_m"] != [_num(size[0]), _num(size[1])]:
            notes.append(f"clamped size_m {size} -> {d['size_m']}")
    else:
        d["size_m"] = [12.0, 8.0]
        notes.append("size_m missing/invalid -> [12, 8]")
    d["aisles"] = _i(raw.get("aisles"), 1, 5, 3)
    d["shelf_units"] = _i(raw.get("shelf_units"), 1, 40, 6)
    prods = []
    for p in raw.get("products") or []:
        if isinstance(p, dict) and p.get("name"):
            prods.append({"name": str(p["name"]), "color": str(p.get("color") or "red"),
                          "shelf_index": max(0, _i(p.get("shelf_index"), 0, 10**6, 0)),
                          "height_m": _f(p.get("height_m"), 0.2, 1.8, 0.8)})
    d["products"] = prods
    d["target_product"] = str(raw.get("target_product") or "")
    co = raw.get("checkout")
    if isinstance(co, dict) and "x" in co and "y" in co:
        d["checkout"] = {"x": _num(co["x"]), "y": _num(co["y"])}
    else:
        d["checkout"] = "auto"
    d["spills"] = [{"x": _num(s.get("x")), "y": _num(s.get("y")), "radius": _f(s.get("radius"), 0.1, 0.6, 0.3)}
                   for s in raw.get("spills") or [] if isinstance(s, dict)]
    sh = raw.get("shoppers") if isinstance(raw.get("shoppers"), dict) else {}
    dens = sh.get("density")
    d["shoppers"] = {"count": _i(sh.get("count"), 0, 8, 0), "speed_mps": _f(sh.get("speed_mps"), 0.3, 1.4, 0.8),
                     "density": dens if dens in ("low", "medium", "rush_hour") else "low"}
    ca = raw.get("cart") if isinstance(raw.get("cart"), dict) else {}
    d["cart"] = {"mass_kg": _f(ca.get("mass_kg"), 8, 40, 15), "loaded": bool(ca.get("loaded", False))}
    task = [t for t in (raw.get("task") or []) if t in S.STEPS]
    d["task"] = task or list(S.STEPS)
    su = raw.get("success") if isinstance(raw.get("success"), dict) else {}
    d["success"] = {"max_time_s": _f(su.get("max_time_s"), 30, 600, 120)}
    d["path_seed"] = _i(raw.get("path_seed"), 0, 10**9, 0)
    d["notes"] = [str(n) for n in (raw.get("notes") or []) if isinstance(n, (str, int, float))]
    _sanitize_v2(raw, d, notes)
    try:
        return WorldSpec.model_validate(d), notes
    except ValidationError as e:  # pragma: no cover - defensive
        return WorldSpec(notes=[f"LLM output unusable ({len(e.errors())} errors); default world used"]), notes


V2_KEYS = ("template", "terrain", "obstacles", "hazards", "people", "lighting", "dr", "robots", "tasks")


def _pair(v: Any, lo: float, hi: float, default: list, cast=float) -> list:
    if not (isinstance(v, (list, tuple)) and len(v) == 2):
        return list(default)
    a, b = sorted((_f(v[0], lo, hi, default[0]), _f(v[1], lo, hi, default[1])))
    return [cast(round(a)), cast(round(b))] if cast is int else [a, b]


def _xy(o: Any) -> tuple[float, float] | None:
    if not isinstance(o, dict):
        return None
    try:
        x, y = float(o["x"]), float(o["y"])
    except (KeyError, TypeError, ValueError):
        return None
    return None if (math.isnan(x) or math.isnan(y)) else (x, y)


def _sanitize_v2(raw: dict[str, Any], d: dict[str, Any], notes: list[str]) -> None:
    present = [k for k in V2_KEYS if k in raw]
    if not present:
        return
    d["spec_version"] = 2
    if isinstance(raw.get("terrain"), dict):
        t = raw["terrain"]
        typ = t.get("type")
        if typ not in ("flat", "rough", "slope", "steps", "ramp"):
            if typ is not None:
                notes.append(f"terrain type {typ!r} -> flat")
            typ = "flat"
        reg = t.get("region")
        region = None
        if isinstance(reg, (list, tuple)) and len(reg) == 4:
            try:
                r = [float(q) for q in reg]
                region = [min(r[0], r[2]), min(r[1], r[3]), max(r[0], r[2]), max(r[1], r[3])]
            except (TypeError, ValueError):
                notes.append("terrain region malformed -> auto")
        d["terrain"] = {"type": typ, "roughness_m": _f(t.get("roughness_m"), 0, 0.06, 0.0),
                        "slope_deg": _f(t.get("slope_deg"), 0, 30, 0.0),
                        "step_h_m": _f(t.get("step_h_m"), 0, 0.12, 0.0), "step_n": _i(t.get("step_n"), 0, 6, 0),
                        "region": region}
    obs = []
    for i, o in enumerate(raw.get("obstacles") or []):
        xy = _xy(o)
        if xy is None:
            notes.append(f"obstacle {i} malformed -> dropped")
            continue
        kind = o.get("kind") if o.get("kind") in ("box", "barrier", "door") else "box"
        mass = o.get("mass_kg")
        obs.append({"kind": kind, "x": xy[0], "y": xy[1], "w": _f(o.get("w"), 0.05, 4, 0.6),
                    "l": _f(o.get("l"), 0.05, 4, 0.6), "h": _f(o.get("h"), 0.05, 2.5, 0.8),
                    "yaw": _num(o.get("yaw")),
                    "mass_kg": None if mass is None else _f(mass, 0.1, 200, 10.0),
                    "open": bool(o.get("open", False))})
    if "obstacles" in raw:
        d["obstacles"] = obs
    hzs = []
    for i, hz in enumerate(raw.get("hazards") or []):
        xy = _xy(hz)
        if xy is None or hz.get("kind") not in ("spill", "low_light", "cable"):
            notes.append(f"hazard {i} malformed -> dropped")
            continue
        hzs.append({"kind": hz["kind"], "x": xy[0], "y": xy[1], "radius": _f(hz.get("radius"), 0.1, 1.5, 0.3)})
    if "hazards" in raw:
        d["hazards"] = hzs
    if isinstance(raw.get("people"), dict):
        pe = raw["people"]
        d["people"] = {"count": _i(pe.get("count"), 0, 8, 0), "speed_mps": _f(pe.get("speed_mps"), 0.3, 1.4, 0.8),
                       "density": pe.get("density") if pe.get("density") in ("low", "medium", "rush_hour") else "low"}
    if isinstance(raw.get("lighting"), dict):
        d["lighting"] = {"ambient": _f(raw["lighting"].get("ambient"), 0.2, 1.0, 0.45)}
    if "robots" in raw:
        d["robots"] = _i(raw.get("robots"), 1, 3, 1)
    tasks = []
    for k, t in enumerate(raw.get("tasks") or []):
        if not isinstance(t, dict):
            notes.append(f"task {k} malformed -> dropped")
            continue
        steps = [x for x in (t.get("steps") or []) if x in S.V2_STEPS]
        tgt = _xy(t.get("target"))
        door = t.get("door")
        tasks.append({"robot": _i(t.get("robot"), 0, 2, 0), "steps": steps or list(S.STEPS),
                      "target": None if tgt is None else {"x": tgt[0], "y": tgt[1]},
                      "door": None if door is None or isinstance(door, bool) else _i(door, 0, 10**6, 0)})
    if "tasks" in raw:
        d["tasks"] = tasks
    dr_raw = raw.get("dr")
    if "dr" in raw or any(k in raw for k in ("terrain", "obstacles", "hazards", "robots", "tasks")):
        dr_raw = dr_raw if isinstance(dr_raw, dict) else {}
        pu = dr_raw.get("push") if isinstance(dr_raw.get("push"), dict) else {}
        d["dr"] = {"friction": _pair(dr_raw.get("friction"), 0.05, 2.0, [0.4, 1.0]),
                   "mass_scale": _pair(dr_raw.get("mass_scale"), 0.5, 1.5, [0.9, 1.1]),
                   "push": {"interval_s": _pair(pu.get("interval_s"), 0.0, 600.0, [5.0, 10.0]),
                            "speed_mps": _pair(pu.get("speed_mps"), 0.0, 10.0, [0.1, 1.0])},
                   "joint_noise_rad": _f(dr_raw.get("joint_noise_rad"), 0, 0.2, 0.03),
                   "action_latency_steps": _pair(dr_raw.get("action_latency_steps"), 0, 10, [0, 3], int)}


# ---------------------------------------------------------------- repair

def _relocate_obstacle(spec: WorldSpec, idx: int) -> bool:
    """Move obstacle idx to the nearest free cell clear of the base route and the key points."""
    o = spec.obstacles[idx]
    tmp = spec.model_copy(update={"obstacles": [x for k, x in enumerate(spec.obstacles) if k != idx]})
    free = S._grid(tmp, with_spills=True)
    wps = S.route_waypoints(tmp)
    pts = S.base_route_points(tmp) + wps + [S.CART_STAGE, S.checkout_xy(tmp)]
    pts = pts[::2] + wps + [S.CART_STAGE]
    need = 0.6 + math.hypot(o.w, o.l) / 2
    best, bd = None, 1e9
    for i in range(0, len(free), 2):
        for j in range(0, len(free[i]), 2):
            if not free[i][j]:
                continue
            x, y = i * S.GRID, j * S.GRID
            dd = math.hypot(x - o.x, y - o.y)
            if dd >= bd or min(math.hypot(x - a, y - b) for a, b in pts) < need:
                continue
            best, bd = (x, y), dd
    if best is None:
        return False
    o.x, o.y = round(best[0], 2), round(best[1], 2)
    return True


def _seg_dist(p, a, b) -> float:
    ax, ay, bx, by = a[0], a[1], b[0], b[1]
    dx, dy = bx - ax, by - ay
    t = 0.0 if dx == dy == 0 else max(0.0, min(1.0, ((p[0] - ax) * dx + (p[1] - ay) * dy) / (dx * dx + dy * dy)))
    return math.hypot(p[0] - (ax + t * dx), p[1] - (ay + t * dy))


def _drop_obstacle(sp: WorldSpec, i: int) -> None:
    sp.obstacles.pop(i)
    for t in sp.tasks:
        if t.door is not None:
            t.door = None if t.door == i else (t.door - 1 if t.door > i else t.door)


def _repair_obstacles(sp: WorldSpec, log: list[str]) -> None:
    w, h = sp.size_m
    for i, o in enumerate(sp.obstacles):
        nx, ny = min(w - 0.3, max(0.3, o.x)), min(h - 0.3, max(0.3, o.y))
        if (nx, ny) != (o.x, o.y):
            log.append(f"repair: obstacle {i} ({o.kind}) out of bounds ({o.x:.2f},{o.y:.2f}) -> clamped ({nx:.2f},{ny:.2f})")
            o.x, o.y = nx, ny
    for i, hz in enumerate(sp.hazards):
        nx, ny = min(w - 0.3, max(0.3, hz.x)), min(h - 0.3, max(0.3, hz.y))
        if (nx, ny) != (hz.x, hz.y):
            log.append(f"repair: hazard {i} ({hz.kind}) out of bounds ({hz.x:.2f},{hz.y:.2f}) -> clamped ({nx:.2f},{ny:.2f})")
            hz.x, hz.y = nx, ny
    if sp.terrain.slope_deg > 15:
        log.append(f"repair: slope {sp.terrain.slope_deg:.1f} -> 15 deg")
        sp.terrain.slope_deg = 15.0
    pat = re.compile(r"obstacle (\d+) \((\w+)\) sits on the (.+)$")
    for _ in range(len(sp.obstacles) + 1):
        hit = next((m for m in map(pat.match, validate_spec(sp)) if m), None)
        if hit is None:
            break
        i, kind, where = int(hit.group(1)), hit.group(2), hit.group(3)
        if _relocate_obstacle(sp, i):
            o = sp.obstacles[i]
            log.append(f"repair: obstacle {i} ({kind}) on the {where} -> moved to ({o.x:.2f},{o.y:.2f})")
        else:
            _drop_obstacle(sp, i)
            log.append(f"repair: obstacle {i} ({kind}) on the {where} dropped (no free position)")
    tried: set[int] = set()
    for _ in range(len(sp.obstacles) + 1):
        if not any("no free route" in p for p in validate_spec(sp)):
            break
        wps = S.route_waypoints(sp)
        doors = [i for i, o in enumerate(sp.obstacles) if o.kind == "door" and not o.open]

        def near(i, line):
            o = sp.obstacles[i]
            return min(_seg_dist((o.x, o.y), a, b) for a, b in zip(line, line[1:]))

        if doors:
            i = min(doors, key=lambda k: near(k, wps))
            sp.obstacles[i].open = True
            log.append(f"repair: door {i} closed across the only route -> opened")
            continue
        line = [wps[0], wps[2]]
        cands = sorted((i for i, o in enumerate(sp.obstacles) if o.kind in ("box", "barrier") and i not in tried),
                       key=lambda k: near(k, line))
        if not cands:
            break
        i = cands[0]
        tried.add(i)
        old = (sp.obstacles[i].x, sp.obstacles[i].y)
        if _relocate_obstacle(sp, i):
            o = sp.obstacles[i]
            log.append(f"repair: obstacle {i} ({o.kind}) blocked the route at ({old[0]:.2f},{old[1]:.2f}) -> moved to ({o.x:.2f},{o.y:.2f})")


def _relocate_spill(spec: WorldSpec, idx: int, off_route: bool) -> bool:
    return _relocate_obj(spec, spec.spills[idx], off_route)


def _relocate_obj(spec: WorldSpec, s: Any, off_route: bool) -> bool:
    free = S._grid(spec, with_spills=False)
    route = (S.base_route_points(spec) + S.route_waypoints(spec)) if off_route else []
    w, h = spec.size_m
    best, bd = None, 1e9
    for i, col in enumerate(free):
        for j, ok in enumerate(col):
            if not ok:
                continue
            x, y = i * S.GRID, j * S.GRID
            if route and min(math.hypot(x - a, y - b) for a, b in route) < s.radius + 0.5:
                continue
            dd = math.hypot(x - s.x, y - s.y)
            if dd < bd:
                best, bd = (x, y), dd
    if best is None:
        return False
    s.x, s.y = round(best[0], 2), round(best[1], 2)
    return True


def repair(spec: WorldSpec) -> WorldSpec:
    """Deterministically fix what validate_spec complains about; record each fix in notes."""
    sp = spec.model_copy(deep=True)
    log: list[str] = []
    w, h = sp.size_m
    # aisles vs height
    while sp.aisles > 1 and S.aisle_width(sp) < S.MIN_AISLE_W:
        sp.aisles -= 1
        log.append(f"repair: aisles reduced to {sp.aisles} (aisle width >= {S.MIN_AISLE_W} m)")
    if S.aisle_width(sp) < S.MIN_AISLE_W:
        need = S.CORRIDOR_Y + S.TOP_MARGIN + 2 * S.SHELF_DEPTH + S.MIN_AISLE_W
        sp.size_m[1] = min(15.0, round(need + 0.1, 2))
        log.append(f"repair: height raised to {sp.size_m[1]}")
    w, h = sp.size_m
    cap = S.max_units(sp)
    if sp.shelf_units > cap:
        log.append(f"repair: shelf_units {sp.shelf_units} -> {cap}")
        sp.shelf_units = max(1, cap)
    # products
    if not sp.products:
        sp.products.append(S.Product(name="item", shelf_index=0))
        log.append("repair: added default product 'item'")
    seen: set[str] = set()
    for p in sp.products:
        base, n = p.name, 2
        while p.name in seen:
            p.name = f"{base}_{n}"
            n += 1
        seen.add(p.name)
        if p.shelf_index >= sp.shelf_units:
            log.append(f"repair: product '{p.name}' shelf_index {p.shelf_index} -> {p.shelf_index % sp.shelf_units}")
            p.shelf_index %= sp.shelf_units
    if S.target(sp) is None:
        log.append(f"repair: target_product '{sp.target_product}' -> '{sp.products[0].name}'")
        sp.target_product = sp.products[0].name
    # checkout
    if isinstance(sp.checkout, S.XY):
        c = sp.checkout
        nx, ny = min(w - 0.3, max(0.3, c.x)), min(h - 0.3, max(0.3, c.y))
        on_shelf = any(a - 0.3 < nx < b + 0.3 and cc - 0.3 < ny < d + 0.3
                       for a, cc, b, d in S.shelf_rects(sp))
        if on_shelf:
            sp.checkout = "auto"
            log.append("repair: checkout was on/near a shelf -> auto")
        elif (nx, ny) != (c.x, c.y):
            log.append(f"repair: checkout clamped to ({nx:.2f},{ny:.2f})")
            sp.checkout = S.XY(x=nx, y=ny)
    if not sp.task:
        sp.task = list(S.STEPS)
    _repair_obstacles(sp, log)
    # spills: bounds, then shelves, then route blocking
    for i, s in enumerate(sp.spills):
        nx, ny = min(w - 0.3, max(0.3, s.x)), min(h - 0.3, max(0.3, s.y))
        if (nx, ny) != (s.x, s.y):
            log.append(f"repair: spill {i} out of bounds ({s.x:.2f},{s.y:.2f}) -> clamped ({nx:.2f},{ny:.2f})")
            s.x, s.y = nx, ny
        if any(a - s.radius < s.x < b + s.radius and c - s.radius < s.y < d + s.radius
               for a, c, b, d in S.shelf_rects(sp)):
            if _relocate_spill(sp, i, off_route=False):
                log.append(f"repair: spill {i} overlapped a shelf -> moved to ({s.x:.2f},{s.y:.2f})")
    for _ in range(len(sp.spills) + len(sp.hazards) + 1):
        probs = [p for p in validate_spec(sp) if "spill" in p or "hazard" in p]
        objs = list(sp.spills) + [hz for hz in sp.hazards if hz.kind == "spill"]
        if not probs or not objs:
            break
        pts = S.base_route_points(sp) + S.route_waypoints(sp)
        clear = [min(math.hypot(s.x - a, s.y - b) for a, b in pts) - min(s.radius, 0.6) for s in objs]
        i = clear.index(min(clear))
        s = objs[i]
        old = (s.x, s.y)
        label = f"spill {i}" if i < len(sp.spills) else f"hazard (spill) {i - len(sp.spills)}"
        if _relocate_obj(sp, s, off_route=True):
            log.append(f"repair: {label} blocked the route at ({old[0]:.2f},{old[1]:.2f}) -> moved to ({s.x:.2f},{s.y:.2f})")
        else:
            if i < len(sp.spills):
                sp.spills.pop(i)
            else:
                sp.hazards.remove(s)
            log.append(f"repair: {label} dropped (no off-route position)")
    keep = []
    for k, t in enumerate(sp.tasks):
        if t.robot >= sp.robots:
            log.append(f"repair: task {k} dropped (robot {t.robot} >= robots {sp.robots})")
            continue
        steps = []
        for st in t.steps:
            if st == "go_to" and t.target is None:
                log.append(f"repair: task {k} go_to step dropped (no target)")
            elif st == "open_door" and not (t.door is not None and 0 <= t.door < len(sp.obstacles)
                                            and sp.obstacles[t.door].kind == "door"):
                log.append(f"repair: task {k} open_door step dropped (no door)")
            else:
                steps.append(st)
        t.steps = steps
        keep.append(t)
    sp.tasks = keep
    sp.notes = list(sp.notes) + log
    return sp


class WorldRefused(ValueError):
    def __init__(self, reasons: list[str]):
        self.reasons = list(reasons)
        super().__init__("world refused: " + "; ".join(self.reasons))


def repair_or_refuse(spec: WorldSpec, max_rounds: int = 3) -> WorldSpec:
    sp = spec
    for _ in range(max_rounds):
        if not validate_spec(sp):
            return sp
        sp = repair(sp)
    probs = validate_spec(sp)
    if probs:
        raise WorldRefused(probs)
    return sp


# ---------------------------------------------------------------- generation

def prompt_to_spec(prompt: str, llm=None, seed: int = 0, refuse: bool = False) -> WorldSpec:
    from iz.llm.client import fixer_model, make_llm
    if llm is None:
        llm = make_llm(cached=True)  # CachedLLM-wrapped
    model = fixer_model()
    schema = WorldSpec.model_json_schema()
    user = f"World request: {prompt}\nSeed: {seed}"
    resp = llm.complete_json(model, SYSTEM, user, schema)
    spec, snotes = sanitize(resp.data)
    spec.notes += snotes
    problems = validate_spec(spec)
    if problems:
        user2 = (user + "\n\nYour previous answer:\n" + json.dumps(resp.data, sort_keys=True)[:4000]
                 + "\n\nIt failed validation:\n- " + "\n- ".join(problems) + "\nReturn a corrected full JSON object.")
        resp2 = llm.complete_json(model, SYSTEM, user2, schema)
        spec2, snotes2 = sanitize(resp2.data)
        spec2.notes += snotes2
        if len(validate_spec(spec2)) <= len(problems):
            spec = spec2
        spec.notes.append(f"re-asked LLM once after {len(problems)} validation problem(s)")
    if refuse:
        return repair_or_refuse(spec)
    spec = repair(spec)
    left = validate_spec(spec)
    if left:
        spec.notes.append("unresolved after repair: " + "; ".join(left))
    return spec


def variations(spec: WorldSpec, n: int, seed: int = 0) -> list[WorldSpec]:
    """n deterministic jittered copies (hazards, shopper paths, cart mass, product placement)."""
    out = []
    for i in range(n):
        rng = random.Random(f"var:{seed}:{i}:{spec.name}")
        v = spec.model_copy(deep=True)
        v.name = f"{spec.name}_v{seed}_{i}"
        v.notes = list(spec.notes) + [f"variation {i} (seed {seed})"]
        v.path_seed = rng.randrange(1, 10**9)
        v.cart.mass_kg = round(_f(spec.cart.mass_kg * rng.uniform(0.75, 1.25), 8, 40, spec.cart.mass_kg), 2)
        v.shoppers.speed_mps = round(_f(spec.shoppers.speed_mps + rng.uniform(-0.2, 0.2), 0.3, 1.4, 0.8), 2)
        for s in v.spills:
            s.x = round(s.x + rng.uniform(-0.6, 0.6), 2)
            s.y = round(s.y + rng.uniform(-0.6, 0.6), 2)
            s.radius = round(_f(s.radius + rng.uniform(-0.05, 0.05), 0.1, 0.6, s.radius), 2)
        if v.shelf_units > 1:
            for p in v.products:
                if p.name != v.target_product or rng.random() < 0.5:
                    p.shelf_index = rng.randrange(v.shelf_units)
        if v.obstacles or v.hazards:  # v2 draws come after every v1 draw (v1 outputs unchanged)
            for o in v.obstacles:
                o.x = round(o.x + rng.uniform(-0.4, 0.4), 2)
                o.y = round(o.y + rng.uniform(-0.4, 0.4), 2)
            for hz in v.hazards:
                hz.x = round(hz.x + rng.uniform(-0.5, 0.5), 2)
                hz.y = round(hz.y + rng.uniform(-0.5, 0.5), 2)
        if v.people is not None:
            v.people = v.shoppers.model_copy()
        out.append(repair(v))
    return out


# ---------------------------------------------------------------- offline v2 keyword mapper

DEMO_PROMPTS: list[str] = [
    "an icy loading dock with pallets and a ramp",
    "a construction site with stairs, cables and poor lighting",
    "a hospital corridor with a closed fire door and crowded visitors",
]


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "_", text.lower())[:40]


def keyword_spec_v2(prompt: str) -> dict[str, Any]:
    """Deterministic WORD->NUMBER mapping onto a fixed 14x9 layout (range midpoints). Feed to sanitize()."""
    t = prompt.lower()

    def has(pat: str) -> bool:
        return re.search(pat, t) is not None

    venue = "store"
    if has(r"hospital|clinic|ward"):
        venue = "hospital_corridor"
    elif has(r"warehouse|loading dock|dock|pallet"):
        venue = "warehouse"
    elif has(r"pharmacy"):
        venue = "pharmacy"
    elif has(r"office"):
        venue = "office"
    d: dict[str, Any] = {
        "name": _slug(prompt), "venue": venue, "size_m": [14, 9], "aisles": 2, "shelf_units": 6,
        "products": [{"name": "box", "shelf_index": 1}], "target_product": "box",
        "terrain": {"type": "flat"}, "obstacles": [], "hazards": [], "lighting": {},
        "notes": [f"keyword_spec_v2: {prompt}"],
    }
    dr: dict[str, Any] = {}
    if has(r"\bicy\b|\bice\b|loading dock"):
        dr["friction"] = [0.2, 0.5]
    if has(r"pushy|windy|\bwind\b"):
        dr["push"] = {"speed_mps": [0.5, 1.0]}
    terrain = d["terrain"]
    if has(r"rough|gravel|uneven"):
        terrain.update(type="rough", roughness_m=0.035)
    if has(r"slope|\bhill"):
        terrain.update(type="slope", slope_deg=10.0)
    if has(r"\bramps?\b"):
        terrain.update(type="ramp", slope_deg=8.0)
    construction = has(r"construction")
    if construction or has(r"stairs?|\bsteps?\b"):
        terrain.update(type="steps", step_h_m=0.10, step_n=4, region=[10.5, 2.2, 13.6, 4.6])
    if construction:
        d["obstacles"] += [{"kind": "box", "x": 7.0, "y": 3.2}, {"kind": "box", "x": 9.0, "y": 6.0},
                           {"kind": "box", "x": 6.5, "y": 7.6}]
        d["hazards"] += [{"kind": "cable", "x": 7.8, "y": 4.6, "radius": 0.3},
                         {"kind": "cable", "x": 9.8, "y": 5.0, "radius": 0.3},
                         {"kind": "low_light", "x": 9.5, "y": 7.5, "radius": 1.0}]
        d["lighting"]["ambient"] = 0.3
    if has(r"pallets?|crates?"):
        d["obstacles"] += [{"kind": "box", "x": 7.0, "y": 3.0, "w": 1.2, "l": 0.8, "h": 0.15, "mass_kg": 20},
                           {"kind": "box", "x": 9.5, "y": 5.5, "w": 1.2, "l": 0.8, "h": 0.15, "mass_kg": 20}]
    if has(r"\bwet\b|puddle"):
        d["hazards"] += [{"kind": "spill", "x": 7.5, "y": 5.0, "radius": 0.4}]
    if has(r"poor light|dim|dark|night"):
        d["lighting"]["ambient"] = 0.25 if has(r"dark|night") else 0.3
    if has(r"fire door|closed door"):
        d["obstacles"].append({"kind": "door", "x": 7.5, "y": 4.0, "w": 0.1, "l": 2.0, "h": 2.0, "open": False})
    if has(r"crowd"):
        d["people"] = {"count": 6, "speed_mps": 0.8, "density": "rush_hour"}
    if has(r"three robots"):
        d["robots"] = 3
    elif has(r"fleet|team|two robots"):
        d["robots"] = 2
    if dr:
        d["dr"] = dr
    return d


def demo_specs() -> dict[str, WorldSpec]:
    out: dict[str, WorldSpec] = {}
    for p in DEMO_PROMPTS:
        slug = _slug(p)
        sp, _ = sanitize(keyword_spec_v2(p))
        sp.name = slug
        out[slug] = repair_or_refuse(sp)
    return out
