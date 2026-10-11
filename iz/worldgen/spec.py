"""WorldSpec: the structured description of a generated environment, plus validation.

Layout convention (shared by validation, repair and any builder):
  * y in [0, CORRIDOR_Y) is the front corridor (robot start, checkout).
  * `aisles` aisles are separated by `aisles + 1` shelf rows of depth SHELF_DEPTH.
  * shelf unit j sits in row (j % (aisles+1)), slot (j // (aisles+1)), length UNIT_LEN.
  * a product on shelf unit j is reached from the aisle facing that unit.
"""
from __future__ import annotations

import math
import random
from collections import deque
from typing import Literal

from pydantic import BaseModel, Field

VENUES = ("store", "pharmacy", "warehouse", "hospital_corridor", "office", "custom")
STEPS = ("get_cart", "go_to_product", "pick_product", "place_in_cart", "go_to_checkout")
MIN_AISLE_W = 1.4
SHELF_DEPTH = 0.4
UNIT_LEN = 1.5
CORRIDOR_Y = 1.6
TOP_MARGIN = 0.3
X_MARGIN = 1.0
ROBOT_START = (0.8, 0.8)
ROBOT_CLEARANCE = 0.3
GRID = 0.1

Venue = Literal["store", "pharmacy", "warehouse", "hospital_corridor", "office", "custom"]
Step = Literal["get_cart", "go_to_product", "pick_product", "place_in_cart", "go_to_checkout"]


class Product(BaseModel):
    name: str
    color: str = "red"
    shelf_index: int = Field(0, ge=0)
    height_m: float = Field(0.8, ge=0.2, le=1.8)


class XY(BaseModel):
    x: float
    y: float


class Spill(BaseModel):
    x: float
    y: float
    radius: float = Field(0.3, ge=0.1, le=0.6)


class Shoppers(BaseModel):
    count: int = Field(0, ge=0, le=8)
    speed_mps: float = Field(0.8, ge=0.3, le=1.4)
    density: Literal["low", "medium", "rush_hour"] = "low"


class Cart(BaseModel):
    mass_kg: float = Field(15.0, ge=8.0, le=40.0)
    loaded: bool = False


class Success(BaseModel):
    max_time_s: float = Field(120.0, ge=30.0, le=600.0)


class WorldSpec(BaseModel):
    name: str = "world"
    venue: Venue = "store"
    size_m: list[float] = Field(default_factory=lambda: [12.0, 8.0], min_length=2, max_length=2)
    aisles: int = Field(3, ge=1, le=5)
    shelf_units: int = Field(6, ge=1, le=40)
    products: list[Product] = Field(default_factory=list)
    target_product: str = ""
    checkout: XY | Literal["auto"] = "auto"
    spills: list[Spill] = Field(default_factory=list)
    shoppers: Shoppers = Field(default_factory=Shoppers)
    cart: Cart = Field(default_factory=Cart)
    task: list[Step] = Field(default_factory=lambda: list(STEPS))
    success: Success = Field(default_factory=Success)
    path_seed: int = 0
    notes: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------- layout

def aisle_width(spec: WorldSpec) -> float:
    h = spec.size_m[1]
    return (h - CORRIDOR_Y - TOP_MARGIN - (spec.aisles + 1) * SHELF_DEPTH) / spec.aisles


def _row_y0(spec: WorldSpec, row: int) -> float:
    return CORRIDOR_Y + row * (SHELF_DEPTH + aisle_width(spec))


def aisle_center_y(spec: WorldSpec, aisle: int) -> float:
    return _row_y0(spec, aisle) + SHELF_DEPTH + aisle_width(spec) / 2


def shelf_rects(spec: WorldSpec) -> list[tuple[float, float, float, float]]:
    """(x0, y0, x1, y1) per shelf unit."""
    rows = spec.aisles + 1
    out = []
    for j in range(spec.shelf_units):
        row, slot = j % rows, j // rows
        x0 = X_MARGIN + slot * UNIT_LEN
        y0 = _row_y0(spec, row)
        out.append((x0, y0, x0 + UNIT_LEN, y0 + SHELF_DEPTH))
    return out


def max_units(spec: WorldSpec) -> int:
    slots = max(0, int((spec.size_m[0] - 2 * X_MARGIN) // UNIT_LEN))
    return slots * (spec.aisles + 1)


def checkout_xy(spec: WorldSpec) -> tuple[float, float]:
    if isinstance(spec.checkout, XY):
        return (spec.checkout.x, spec.checkout.y)
    return (spec.size_m[0] - 1.0, 0.9)


def target(spec: WorldSpec) -> Product | None:
    for p in spec.products:
        if p.name == spec.target_product:
            return p
    return None


def product_access_xy(spec: WorldSpec, p: Product) -> tuple[float, float]:
    rows = spec.aisles + 1
    j = p.shelf_index
    row, slot = j % rows, j // rows
    aisle = min(row, spec.aisles - 1)
    return (X_MARGIN + slot * UNIT_LEN + UNIT_LEN / 2, aisle_center_y(spec, aisle))


def shopper_paths(spec: WorldSpec) -> list[list[list[float]]]:
    """Deterministic patrol paths (from path_seed) running along aisles/corridor."""
    rng = random.Random(f"paths:{spec.name}:{spec.path_seed}")
    w = spec.size_m[0]
    ys = [aisle_center_y(spec, a) for a in range(spec.aisles)] + [CORRIDOR_Y / 2 + 0.3]
    paths = []
    for _ in range(spec.shoppers.count):
        y = rng.choice(ys)
        x0 = rng.uniform(1.0, w * 0.4)
        x1 = rng.uniform(w * 0.6, w - 1.0)
        paths.append([[round(x0, 2), round(y, 2)], [round(x1, 2), round(y, 2)]])
    return paths


# ---------------------------------------------------------------- routing

def _grid(spec: WorldSpec, with_spills: bool) -> list[list[bool]]:
    w, h = spec.size_m
    nx, ny = int(w / GRID) + 1, int(h / GRID) + 1
    free = [[True] * ny for _ in range(nx)]
    rects = [(a - ROBOT_CLEARANCE, b - ROBOT_CLEARANCE, c + ROBOT_CLEARANCE, d + ROBOT_CLEARANCE)
             for a, b, c, d in shelf_rects(spec)]
    for i in range(nx):
        x = i * GRID
        for j in range(ny):
            y = j * GRID
            if x < ROBOT_CLEARANCE or y < ROBOT_CLEARANCE or x > w - ROBOT_CLEARANCE or y > h - ROBOT_CLEARANCE:
                free[i][j] = False
                continue
            for a, b, c, d in rects:
                if a < x < c and b < y < d:
                    free[i][j] = False
                    break
            if free[i][j] and with_spills:
                for s in spec.spills:
                    if math.hypot(x - s.x, y - s.y) < s.radius + 0.15:
                        free[i][j] = False
                        break
    return free


def _cell(p: tuple[float, float]) -> tuple[int, int]:
    return (round(p[0] / GRID), round(p[1] / GRID))


def _path(free, a, b):
    nx, ny = len(free), len(free[0])
    for c in (a, b):
        if not (0 <= c[0] < nx and 0 <= c[1] < ny) or not free[c[0]][c[1]]:
            return None
    prev = {a: None}
    q = deque([a])
    while q:
        c = q.popleft()
        if c == b:
            out = []
            while c is not None:
                out.append(c)
                c = prev[c]
            return out[::-1]
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            n = (c[0] + dx, c[1] + dy)
            if 0 <= n[0] < nx and 0 <= n[1] < ny and free[n[0]][n[1]] and n not in prev:
                prev[n] = c
                q.append(n)
    return None


def _reachable(free, a, b) -> bool:
    return _path(free, a, b) is not None


def route_waypoints(spec: WorldSpec) -> list[tuple[float, float]]:
    t = target(spec)
    return [ROBOT_START, product_access_xy(spec, t) if t else ROBOT_START, checkout_xy(spec)]


def base_route_points(spec: WorldSpec) -> list[tuple[float, float]]:
    """Cells (in metres) along the spill-free shortest route; [] if none."""
    free = _grid(spec, with_spills=False)
    s, a, c = (_cell(p) for p in route_waypoints(spec))
    p1, p2 = _path(free, s, a), _path(free, a, c)
    if p1 is None or p2 is None:
        return []
    return [(i * GRID, j * GRID) for i, j in p1 + p2]


# ---------------------------------------------------------------- validation

def validate_spec(spec: WorldSpec) -> list[str]:
    probs: list[str] = []
    w, h = spec.size_m
    if not (6 <= w <= 20 and 5 <= h <= 15):
        probs.append(f"size_m {spec.size_m} out of range (width 6..20, height 5..15)")
    aw = aisle_width(spec)
    if aw < MIN_AISLE_W:
        probs.append(f"aisle width {aw:.2f} m < {MIN_AISLE_W} m needed for robot+cart "
                     f"(reduce aisles or increase height)")
    if spec.shelf_units > max_units(spec):
        probs.append(f"shelf_units {spec.shelf_units} exceeds capacity {max_units(spec)} for this size")
    names = [p.name for p in spec.products]
    if len(set(names)) != len(names):
        probs.append("duplicate product names")
    if not spec.products:
        probs.append("no products defined")
    if target(spec) is None:
        probs.append(f"target_product '{spec.target_product}' not among products {names}")
    for p in spec.products:
        if p.shelf_index >= spec.shelf_units:
            probs.append(f"product '{p.name}' shelf_index {p.shelf_index} >= shelf_units {spec.shelf_units}")
    c = checkout_xy(spec)
    if not (0.3 <= c[0] <= w - 0.3 and 0.3 <= c[1] <= h - 0.3):
        probs.append(f"checkout {c} outside the {w}x{h} room")
    for i, s in enumerate(spec.spills):
        if not (0 <= s.x <= w and 0 <= s.y <= h):
            probs.append(f"spill {i} at ({s.x:.2f},{s.y:.2f}) is out of bounds")
    if not spec.task:
        probs.append("task is empty")
    if probs:
        return probs  # geometry checks below need a sane layout
    free = _grid(spec, with_spills=False)
    start, acc, co = (_cell(p) for p in route_waypoints(spec))
    if not (_reachable(free, start, acc) and _reachable(free, acc, co)):
        probs.append("no free route start -> target product -> checkout")
        return probs
    freeh = _grid(spec, with_spills=True)
    if not (_reachable(freeh, start, acc) and _reachable(freeh, acc, co)):
        probs.append("hazards (spills) block every route to the target or checkout")
    wps = route_waypoints(spec)
    for i, s in enumerate(spec.spills):
        for nm, p in (("start", wps[0]), ("target access point", wps[1]), ("checkout", wps[2])):
            if math.hypot(p[0] - s.x, p[1] - s.y) < s.radius + 0.2:
                probs.append(f"spill {i} sits on the {nm}")
    return probs
