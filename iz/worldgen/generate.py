"""Single prompt -> validated WorldSpec, plus seeded variations for train / held-out sets."""
from __future__ import annotations

import json
import math
import random
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
    d["venue"] = v if v in S.VENUES else "custom"
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
    try:
        return WorldSpec.model_validate(d), notes
    except ValidationError as e:  # pragma: no cover - defensive
        return WorldSpec(notes=[f"LLM output unusable ({len(e.errors())} errors); default world used"]), notes


# ---------------------------------------------------------------- repair

def _relocate_spill(spec: WorldSpec, idx: int, off_route: bool) -> bool:
    s = spec.spills[idx]
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
    for _ in range(len(sp.spills) + 1):
        probs = [p for p in validate_spec(sp) if "spill" in p or "hazard" in p]
        if not probs or not sp.spills:
            break
        pts = S.base_route_points(sp) + S.route_waypoints(sp)
        clear = [min(math.hypot(s.x - a, s.y - b) for a, b in pts) - s.radius for s in sp.spills]
        i = clear.index(min(clear))
        s = sp.spills[i]
        old = (s.x, s.y)
        if _relocate_spill(sp, i, off_route=True):
            log.append(f"repair: spill {i} blocked the route at ({old[0]:.2f},{old[1]:.2f}) -> moved to ({s.x:.2f},{s.y:.2f})")
        else:
            sp.spills.pop(i)
            log.append(f"repair: spill {i} dropped (no off-route position)")
    sp.notes = list(sp.notes) + log
    return sp


# ---------------------------------------------------------------- generation

def prompt_to_spec(prompt: str, llm=None, seed: int = 0) -> WorldSpec:
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
        out.append(repair(v))
    return out
