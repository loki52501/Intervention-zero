"""WorldSpec -> MuJoCo scene.  build_world(spec, seed) -> (xml, meta) emitting the SHOP contract.

The robot, cart, welds, actuators and sensors are taken VERBATIM from iz.shop.scene.build_shop
(so the robot is identical and iz/shop/controller|episode|learn work unchanged); everything
else (floor, walls, shelves, products, checkout, spills, shoppers, routes) is generated here
from the spec using the layout convention of iz.worldgen.spec:

  x in [0, w], y in [0, h]; y < CORRIDOR_Y is the front corridor (robot start, checkout);
  aisles run along x; shelf unit j is row j % (aisles+1), slot j // (aisles+1).

Target product is always body ``prod_0``.  Spec products become free bodies prod_0..prod_n.

Spec v2 additions (all emit nothing when at their defaults, so v1 specs give byte-identical XML):
  * collision bits: terrain (hfield, stairs) is contype 64 / conaffinity 10 and so touches only the cart and
    products (the planar robot base ignores it); static obstacles, barriers and door leaves are contype 1 /
    conaffinity 14 like shelves; free obstacles are contype 1 / conaffinity 15.
  * doors follow iz/world/mujoco_door.py: hinge at the -y end of the leaf, range [-1.7, 0], spring to closed.
    An open door is authored in its open pose (body quat yaw-1.6, joint ref/springref -1.6).
  * per-seed domain randomisation comes from terrain.sample_dr and is stored in meta["dr"].
  * an hfield's elevation is NOT inline in the XML: it is in meta["terrain"]["hf"] and applied with
    terrain.set_hfield after compiling (use compile_world(xml, meta)).
HONEST SCOPE: the planar robot base does not climb terrain; locomotion is a parametric gait engine, not a
learned policy; the simulator is authored by us.
"""
from __future__ import annotations

import heapq
import math
import random
import re

import mujoco
import numpy as np

from iz.worldgen import spec as S

try:  # W2 module; features depending on it are skipped honestly if it is missing
    from iz.worldgen import terrain as T
except Exception:  # noqa: BLE001
    T = None

LEDGE_ZS = [0.60, 0.90, 1.20]
LEDGE_HY = 0.10
PROD_HALF = (0.035, 0.035, 0.07)
SHELF_H = 0.75           # half height (1.5 m tall units)
WALL_H = 0.25
GRID = 0.1
GRIP_OFFSET = 1.0
CART_AHEAD = 1.4         # cart centre ahead of robot start (robot has to drive up to it)
COUNTER_HALF = (0.8, 0.2, 0.45)

COLORS = {
    "red": (0.9, 0.2, 0.2), "blue": (0.2, 0.55, 0.9), "green": (0.25, 0.7, 0.35),
    "yellow": (0.95, 0.75, 0.1), "orange": (0.95, 0.5, 0.15), "purple": (0.6, 0.35, 0.8),
    "pink": (0.95, 0.45, 0.7), "white": (0.95, 0.95, 0.95), "black": (0.15, 0.15, 0.17),
    "brown": (0.55, 0.38, 0.22), "teal": (0.2, 0.7, 0.7), "cyan": (0.3, 0.8, 0.8),
}
PALETTE = [(0.9, 0.2, 0.2), (0.2, 0.55, 0.9), (0.95, 0.75, 0.1), (0.25, 0.7, 0.35),
           (0.85, 0.4, 0.75), (0.95, 0.5, 0.15), (0.3, 0.8, 0.8)]
TARGET_RGB = (1.0, 0.15, 0.55)

# venue -> floor a, floor b, wall, shelf body, shelf top accent, sky ambient
THEMES = {
    "store": ((0.90, 0.91, 0.93), (0.82, 0.84, 0.87), (0.55, 0.58, 0.64), (0.93, 0.92, 0.88), (0.15, 0.40, 0.70)),
    "pharmacy": ((0.94, 0.97, 0.95), (0.86, 0.92, 0.89), (0.62, 0.78, 0.72), (0.96, 0.97, 0.96), (0.10, 0.60, 0.45)),
    "warehouse": ((0.72, 0.72, 0.70), (0.64, 0.64, 0.62), (0.50, 0.52, 0.55), (0.80, 0.82, 0.86), (0.95, 0.65, 0.10)),
    "hospital_corridor": ((0.92, 0.95, 0.97), (0.86, 0.90, 0.94), (0.70, 0.82, 0.90), (0.95, 0.96, 0.98), (0.20, 0.55, 0.85)),
    "office": ((0.88, 0.86, 0.82), (0.80, 0.78, 0.74), (0.62, 0.60, 0.58), (0.90, 0.88, 0.84), (0.45, 0.35, 0.60)),
    "custom": ((0.90, 0.91, 0.93), (0.82, 0.84, 0.87), (0.55, 0.58, 0.64), (0.93, 0.92, 0.88), (0.15, 0.40, 0.70)),
}


def _f(*v):
    return " ".join(f"{x:.4f}" for x in v)


def _as_spec(spec):
    if isinstance(spec, S.WorldSpec):
        return spec
    if isinstance(spec, dict):
        return S.WorldSpec.model_validate(spec)
    return S.WorldSpec.model_validate(spec.model_dump())


# ------------------------------------------------------------------ layout helpers
def _face(spec, j):
    """(+1 | -1): which y-side of shelf unit j faces the aisle used to reach it."""
    row = j % (spec.aisles + 1)
    return 1 if row < spec.aisles else -1


def _ledge_z(height_m):
    return min(LEDGE_ZS, key=lambda z: abs(z - height_m))


def _all_spills(spec):
    return list(S.all_spills(spec)) if hasattr(S, "all_spills") else list(spec.spills)


def _blocking(spec):
    return [tuple(map(float, r)) for r in S.blocking_rects(spec)] if hasattr(S, "blocking_rects") else []


def _nl(txt, indent):
    """Make an XML fragment end with exactly one newline ('' stays '')."""
    txt = (txt or "").strip("\n")
    return (indent + txt.lstrip() + "\n") if txt else ""


def _yaw_quat(yaw):
    return f"{math.cos(yaw / 2):.6f} 0 0 {math.sin(yaw / 2):.6f}"


def compile_world(xml: str, meta: dict):
    """Compile the XML and apply the hfield elevation kept in meta (elevation is not inline)."""
    model = mujoco.MjModel.from_xml_string(xml)
    hf = (meta.get("terrain") or {}).get("hf")
    if hf and T is not None:
        T.set_hfield(model, hf)
    return model


def _obstacles(spec, counter):
    """Rects (x0,y0,x1,y1) the robot must keep away from: shelves incl. ledges, and counter."""
    rows = spec.aisles + 1
    out = []
    for j, (x0, y0, x1, y1) in enumerate(S.shelf_rects(spec)):
        row = j % rows
        lo = y0 - (LEDGE_HY * 2 if row > 0 else 0.0)
        hi = y1 + (LEDGE_HY * 2 if row < spec.aisles else 0.0)
        out.append((x0, lo, x1, hi))
    cx, cy, hx, hy = counter
    out.append((cx - hx, cy - hy, cx + hx, cy + hy))
    out.extend(_blocking(spec))
    return out


def _counter(spec, c):
    """Checkout counter placed behind the cart zone against the nearest wall."""
    w, h = spec.size_m
    hx, hy, _ = COUNTER_HALF
    dists = {"s": c[1], "n": h - c[1], "w": c[0], "e": w - c[0]}
    side = min(dists, key=dists.get)
    off = 0.65 + hy
    if side == "s":
        return (c[0], max(hy, c[1] - off), hx, hy)
    if side == "n":
        return (c[0], min(h - hy, c[1] + off), hx, hy)
    if side == "w":
        return (max(hy, c[0] - off), c[1], hy, hx)
    return (min(w - hy, c[0] + off), c[1], hy, hx)


class _Grid:
    def __init__(self, spec, counter, start_x):
        w, h = spec.size_m
        self.nx, self.ny = int(w / GRID) + 1, int(h / GRID) + 1
        X, Y = np.meshgrid(np.arange(self.nx) * GRID, np.arange(self.ny) * GRID, indexing="ij")
        d = np.minimum(np.minimum(X, w - X), np.minimum(Y, h - Y))
        for x0, y0, x1, y1 in _obstacles(spec, counter):
            dx = np.maximum(np.maximum(x0 - X, X - x1), 0.0)
            dy = np.maximum(np.maximum(y0 - Y, Y - y1), 0.0)
            d = np.minimum(d, np.hypot(dx, dy))
        self.d = d
        self.behind = X < start_x - 0.35        # no U-turns with a cart in front

    def cell(self, p):
        return (int(round(p[0] / GRID)), int(round(p[1] / GRID)))

    def ok(self, i, j, hard=0.3):
        return 0 <= i < self.nx and 0 <= j < self.ny and self.d[i, j] >= hard and not self.behind[i, j]

    def los(self, a, b, clear=0.42):
        n = int(max(abs(b[0] - a[0]), abs(b[1] - a[1])) / 0.05) + 1
        for k in range(n + 1):
            t = k / n
            i, j = self.cell((a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])))
            if not self.ok(i, j, clear):
                return False
        return True

    def path(self, a, b):
        """Dijkstra, 8-connected, penalising proximity to obstacles.  Returns metre points."""
        sa, sb = self.cell(a), self.cell(b)
        if not self.ok(*sa) or not self.ok(*sb):
            return None
        dist = {sa: 0.0}
        prev = {}
        pq = [(0.0, sa)]
        while pq:
            cd, c = heapq.heappop(pq)
            if c == sb:
                break
            if cd > dist.get(c, 1e18):
                continue
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    n = (c[0] + dx, c[1] + dy)
                    if not self.ok(*n):
                        continue
                    if dx and dy and not (self.ok(c[0] + dx, c[1]) and self.ok(c[0], c[1] + dy)):
                        continue
                    pen = 1.0 + 25.0 * max(0.0, 0.85 - float(self.d[n])) ** 1.5
                    nd = cd + math.hypot(dx, dy) * pen
                    if nd < dist.get(n, 1e18):
                        dist[n] = nd
                        prev[n] = c
                        heapq.heappush(pq, (nd, n))
        if sb not in dist:
            return None
        cells = [sb]
        while cells[-1] != sa:
            cells.append(prev[cells[-1]])
        return [(i * GRID, j * GRID) for i, j in cells[::-1]]

    def smooth(self, pts):
        out = [pts[0]]
        i = 0
        while i < len(pts) - 1:
            j = len(pts) - 1
            while j > i + 1 and not self.los(pts[i], pts[j]):
                j -= 1
            out.append(pts[j])
            i = j
        return out


def _plan(spec, start, access, goal, counter):
    g = _Grid(spec, counter, start[0])
    legs = []
    for a, b in ((start, access), (access, goal)):
        p = g.path(a, b)
        legs.append(g.smooth(p) if p else [a, b])
    pts = legs[0] + legs[1][1:]
    k = len(legs[0]) - 1                                  # index of the access point
    keep = [p for i, p in enumerate(pts)
            if i == 0 or i == k or i == len(pts) - 1 or math.hypot(p[0] - access[0], p[1] - access[1]) > 0.8]
    return [list(map(float, p)) for p in keep], g


# ------------------------------------------------------------------ robot / cart from the shop scene
def _shop_parts(cart_mass, robot_xy, cart_xy):
    import re
    from iz.shop.scene import build_shop
    xml, shop_meta = build_shop(0, cart_mass_kg=cart_mass, n_shoppers=0)
    i_r = xml.index('    <body name="robot"')
    i_c = xml.index('    <body name="cart"')
    i_w = xml.index("  </worldbody>")
    i_e = xml.index("  <equality>")
    robot, cart, tail = xml[i_r:i_c], xml[i_c:i_w], xml[i_e:]
    robot = re.sub(r'<body name="robot" pos="[^"]*"', f'<body name="robot" pos="{_f(robot_xy[0], robot_xy[1], 0)}"', robot)
    robot = re.sub(r'(name="base_x"[^>]*?)ref="[^"]*"', rf'\1ref="{robot_xy[0]:.4f}"', robot)
    robot = re.sub(r'(name="base_y"[^>]*?)ref="[^"]*"', rf'\1ref="{robot_xy[1]:.4f}"', robot)
    cart = re.sub(r'<body name="cart" pos="[^"]*"', f'<body name="cart" pos="{_f(cart_xy[0], cart_xy[1], 0)}"', cart)
    return robot, cart, tail, shop_meta


# ------------------------------------------------------------------ build
def build_world(spec, seed: int = 0):
    spec = _as_spec(spec)
    rng = random.Random(f"world:{spec.name}:{seed}:{spec.path_seed}")
    w, h = spec.size_m
    rows = spec.aisles + 1
    aw = S.aisle_width(spec)
    rects = S.shelf_rects(spec)
    th = THEMES.get(spec.venue, THEMES["store"])
    fa, fb, wall_c, shelf_c, accent = th

    # ---- task geometry
    tgt = S.target(spec) or (spec.products[0] if spec.products else None)
    cz = S.checkout_xy(spec)
    counter = _counter(spec, cz)
    robot_xy = [S.ROBOT_START[0], S.ROBOT_START[1]]
    cart_xy = [robot_xy[0] + CART_AHEAD, robot_xy[1]]
    grip_pose = [cart_xy[0] - GRIP_OFFSET, cart_xy[1]]
    if tgt is not None:
        access = S.product_access_xy(spec, tgt)
        tj = min(tgt.shelf_index, len(rects) - 1)
    else:
        access, tj = (w / 2, S.aisle_center_y(spec, 0)), 0
    goal = (cz[0] - GRIP_OFFSET, cz[1])
    route_pts, grid = _plan(spec, tuple(grip_pose), tuple(access), goal, counter)

    # ---- waypoints: route points first, then aisle end/centre nodes
    wps = [list(p) for p in route_pts]
    route = list(range(len(wps)))
    xs0 = min((r[0] for r in rects), default=1.0)
    xs1 = max((r[2] for r in rects), default=w - 1.0)
    for a in range(spec.aisles):
        ay = S.aisle_center_y(spec, a)
        wps.append([max(0.5, xs0 / 2), ay])
        wps.append([(xs0 + xs1) / 2, ay])
        wps.append([min(w - 0.5, xs1 + (w - xs1) / 2), ay])

    # ---- target product placement
    fd_t = _face(spec, tj) if rects else 1
    x0t, y0t, x1t, y1t = rects[tj] if rects else (1, 1, 2, 2)
    face_y_t = y1t if fd_t > 0 else y0t
    t_ledge = _ledge_z(tgt.height_m if tgt else 0.9)
    t_xy = [access[0], face_y_t + fd_t * LEDGE_HY]
    t_z = t_ledge + PROD_HALF[2] + 0.002

    # ---- spills along explicit spec positions
    spills = [[float(s.x), float(s.y), float(s.radius)] for s in _all_spills(spec)]
    paths = []
    zone_x = cart_xy[0] + 0.9                 # keep shoppers out of the robot/cart staging area
    for pth in S.shopper_paths(spec):
        pth = [[max(q[0], zone_x + 0.4) if q[1] < S.CORRIDOR_Y + 0.2 else q[0], q[1]] for q in pth]
        if abs(pth[0][0] - pth[-1][0]) < 1.0 and pth[0][1] < S.CORRIDOR_Y + 0.2:
            pth[-1][0] = min(w - 0.8, pth[0][0] + 3.0)
        paths.append(pth)
    sh_speed = float(spec.shoppers.speed_mps)
    # ---- v2: per-seed domain randomisation, terrain, obstacles, hazards (all optional)
    dr_spec = getattr(spec, "dr", None)
    dr = T.sample_dr(dr_spec.model_dump() if dr_spec else None, spec.name, seed,
                     horizon_s=spec.success.max_time_s) if T is not None else {"enabled": False}
    dr_on = bool(dr.get("enabled"))
    scale = float(dr.get("mass_scale", 1.0)) if dr_on else 1.0
    floor_fr = f"{float(dr['friction']):.4f} 0.005 0.0001" if dr_on else "1 0.005 0.0001"
    prod_mass = f"{0.3 * scale:.4f}" if dr_on else "0.3"
    mass = float(spec.cart.mass_kg) * scale

    terr = getattr(spec, "terrain", None)
    region = S.terrain_region(spec) if (terr is not None and hasattr(S, "terrain_region")) else None
    ttype = getattr(terr, "type", "flat") if terr is not None else "flat"
    hf, hf_asset, hf_geom, stairs_x = None, "", "", ""
    if T is not None and region is not None and ttype in ("rough", "slope", "ramp"):
        hf = T.hfield_data(terr.model_dump(), region, spec.name, seed)
        if hf is not None:
            try:
                hf_asset, hf_geom = T.hfield_xml(hf, float(floor_fr.split()[0]))
            except TypeError:
                hf_asset, hf_geom = T.hfield_xml(hf)
            hf_asset = _nl(hf_asset, "    ")
            hf_geom = _nl(hf_geom, "    ")
    elif T is not None and region is not None and ttype == "steps":
        stairs_x, _ = T.stairs_xml(terr.model_dump(), region)
        stairs_x = _nl(stairs_x, "    ")
    obstacles = list(getattr(spec, "obstacles", []) or [])
    hazards = list(getattr(spec, "hazards", []) or [])
    lighting = getattr(spec, "lighting", None)
    ambient = float(lighting.ambient) if lighting is not None else 0.45

    robot_x, cart_x, tail, shop_meta = _shop_parts(mass, robot_xy, cart_xy)

    meta = dict(
        seed=int(seed), name=spec.name, venue=spec.venue, dense=spec.shoppers.density == "rush_hour",
        cart_xy=cart_xy, cart_yaw=0.0, robot_xy=robot_xy, robot_yaw=0.0,
        grip_body="robot", grip_offset=GRIP_OFFSET, hand_body="r_hand_b",
        target_product="prod_0", target_name=tgt.name if tgt else "", target_xy=t_xy, target_z=t_z,
        target_aisle=min(tj % rows, spec.aisles - 1), shelf_side=-fd_t, shelf_index=tj, ledge_z=t_ledge,
        pick_xy=list(map(float, access)), pick_yaw=0.0 if fd_t > 0 else math.pi, place_yaw=math.pi / 2,
        waypoints=wps, route=route, checkout_xy=[float(cz[0]), float(cz[1])],
        checkout_robot_xy=[float(goal[0]), float(goal[1])],
        counter=list(counter), spills=spills, shopper_paths=paths, n_shoppers=len(paths),
        shopper_speed=sh_speed, cart_mass_kg=mass, store_size=[w, h], dr=dr,
        aisle_ys=[S.aisle_center_y(spec, a) for a in range(spec.aisles)],
        shelf_geoms=[f"shelf_{j}" for j in range(len(rects))],
        max_time_s=float(spec.success.max_time_s),
    )

    from iz.shop.scene import robot_header_parts
    rb_comp, rb_default, rb_asset = robot_header_parts()
    rb_default = f"  {rb_default}\n" if rb_default else ""
    rb_asset = f"    {rb_asset}\n" if rb_asset else ""

    A = []
    A.append(f"""<mujoco model="{spec.name or 'world'}">
  <compiler angle="radian" autolimits="true"{rb_comp}/>
  <option timestep="0.002" integrator="implicitfast" gravity="0 0 -9.81"/>
  <visual><global offwidth="1920" offheight="1080"/><quality shadowsize="2048"/>
    <headlight ambient="{ambient:.2f} {ambient:.2f} {ambient:.2f}" diffuse="0.5 0.5 0.5"/></visual>
{rb_default}  <asset>
{rb_asset}    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.78 0.86 0.95" rgb2="0.96 0.96 0.98" width="64" height="64"/>
    <texture name="floortex" type="2d" builtin="checker" rgb1="{_f(*fa)}" rgb2="{_f(*fb)}" width="256" height="256"/>
    <material name="floor" texture="floortex" texrepeat="{max(2, int(w))} {max(2, int(h))}" reflectance="0.05"/>
{hf_asset}  </asset>
  <worldbody>
    <light pos="{_f(w / 2, h / 2, 7)}" dir="0 0.2 -1" diffuse="0.6 0.6 0.6" castshadow="false"/>
    <light pos="{_f(w * 0.2, -3, 6)}" dir="0.3 0.5 -1" diffuse="0.35 0.35 0.35"/>
    <geom name="floor" type="plane" pos="{_f(w / 2, h / 2, 0)}" size="{_f(w / 2, h / 2, 0.1)}" material="floor"
          friction="{floor_fr}" contype="1" conaffinity="10" priority="0"/>
{hf_geom}{stairs_x}    <camera name="cam_34" pos="{_f(w * 0.2, -h * 0.55, h * 0.85 + 2)}" xyaxes="1 0 0 0 0.6 0.8" fovy="55"/>
    <camera name="cam_top" pos="{_f(w / 2, h / 2, max(w, h) * 1.1)}" xyaxes="1 0 0 0 1 0" fovy="55"/>
""")
    for nm, p, s in [("wall_s", (w / 2, -0.05, WALL_H), (w / 2, 0.05, WALL_H)),
                     ("wall_n", (w / 2, h + 0.05, WALL_H), (w / 2, 0.05, WALL_H)),
                     ("wall_w", (-0.05, h / 2, WALL_H), (0.05, h / 2, WALL_H)),
                     ("wall_e", (w + 0.05, h / 2, WALL_H), (0.05, h / 2, WALL_H))]:
        A.append(f'    <geom name="{nm}" type="box" pos="{_f(*p)}" size="{_f(*s)}" rgba="{_f(*wall_c)} 1" contype="1" conaffinity="14"/>\n')

    # ---- shelves + ledges + decor
    free_prods = []        # (j, ledge_z, x, name, rgb)
    used = {}
    for i, p in enumerate(spec.products):
        is_t = tgt is not None and p.name == tgt.name
        j = min(p.shelf_index, len(rects) - 1)
        lz = _ledge_z(p.height_m)
        if is_t:
            continue
        free_prods.append((j, lz, p))
    occupied = {(tj, t_ledge): [access[0]]}
    placed = []
    for j, lz, p in free_prods:
        x0, y0, x1, y1 = rects[j]
        cx = (x0 + x1) / 2
        taken = occupied.setdefault((j, lz), [])
        for off in (0.0, -0.4, 0.4, -0.6, 0.6, -0.2, 0.2):
            if all(abs(cx + off - t) > 0.3 for t in taken):
                taken.append(cx + off)
                placed.append((j, lz, cx + off, p))
                break
    for j, (x0, y0, x1, y1) in enumerate(rects):
        row = j % rows
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        hx, hy = (x1 - x0) / 2 - 0.01, (y1 - y0) / 2
        A.append(f'    <geom name="shelf_{j}" type="box" pos="{_f(cx, cy, SHELF_H)}" size="{_f(hx, hy, SHELF_H)}" '
                 f'rgba="{_f(*shelf_c)} 1" contype="1" conaffinity="14"/>\n')
        A.append(f'    <geom name="shelf_{j}_top" type="box" pos="{_f(cx, cy, 1.52)}" size="{_f(hx + 0.01, hy + 0.02, 0.02)}" '
                 f'rgba="{_f(*accent)} 1" contype="0" conaffinity="0"/>\n')
        for fd in (-1, 1):
            if (fd < 0 and row == 0) or (fd > 0 and row == spec.aisles):
                continue
            ly = (y1 + LEDGE_HY) if fd > 0 else (y0 - LEDGE_HY)
            for k, lz in enumerate(LEDGE_ZS):
                A.append(f'    <geom name="shelf_{j}_ledge_{fd}_{k}" type="box" pos="{_f(cx, ly, lz - 0.01)}" '
                         f'size="{_f(hx, LEDGE_HY, 0.01)}" rgba="0.45 0.52 0.62 1" contype="1" conaffinity="14"/>\n')
                drng = random.Random(f"{spec.name}:{seed}:{j}:{fd}:{k}")
                x = x0 + 0.2
                fixed = [t for (jj, zz), ts in occupied.items() if jj == j and abs(zz - lz) < 1e-6 and fd == _face(spec, j) for t in ts]
                while x < x1 - 0.15:
                    if all(abs(x - t) > 0.28 for t in fixed):
                        c = drng.choice(PALETTE)
                        hz = drng.uniform(0.04, 0.09)
                        A.append(f'    <geom type="box" pos="{_f(x, ly, lz + hz)}" size="{_f(0.045, 0.05, hz)}" '
                                 f'rgba="{_f(*c)} 1" contype="0" conaffinity="0"/>\n')
                    x += 0.34

    # ---- checkout
    ccx, ccy, chx, chy = counter
    A.append(f'    <geom name="checkout" type="box" pos="{_f(ccx, ccy, 0.45)}" size="{_f(chx, chy, 0.45)}" '
             f'rgba="0.2 0.55 0.45 1" contype="1" conaffinity="14"/>\n')
    A.append(f'    <geom name="checkout_top" type="box" pos="{_f(ccx, ccy, 0.91)}" size="{_f(chx + 0.03, chy + 0.03, 0.015)}" '
             f'rgba="0.9 0.9 0.92 1" contype="0" conaffinity="0"/>\n')
    A.append(f'    <geom name="checkout_screen" type="box" pos="{_f(ccx - 0.3, ccy, 1.1)}" size="{_f(0.12, 0.04, 0.12)}" '
             f'rgba="0.1 0.12 0.18 1" contype="0" conaffinity="0"/>\n')
    A.append(f'    <site name="checkout_zone" pos="{_f(cz[0], cz[1], 0.02)}" size="0.45 0.01" type="cylinder" rgba="0.2 0.9 0.4 0.35"/>\n')

    # ---- spills
    for i, (sx, sy, r) in enumerate(spills):
        A.append(f'    <geom name="spill_{i}" type="cylinder" pos="{_f(sx, sy, 0.001)}" size="{_f(max(r, 0.15), 0.001)}" '
                 f'friction="0.04 0.005 0.0001" priority="2" rgba="0.25 0.55 0.95 0.55" contype="1" conaffinity="2"/>\n')

    # ---- v2 obstacles / doors / hazards (nothing is emitted for a v1 spec)
    obst_meta, doors_meta, haz_meta = [], [], []
    for i, o in enumerate(obstacles):
        kind = getattr(o, "kind", "box")
        ox, oy = float(o.x), float(o.y)
        ow, ol, oh = float(o.w), float(o.l), float(o.h)
        yaw = float(getattr(o, "yaw", 0.0) or 0.0)
        om = getattr(o, "mass_kg", None)
        is_open = bool(getattr(o, "open", False))
        free = om is not None and kind != "door"
        q = _yaw_quat(yaw)
        if kind == "door":
            ang = -1.6 if is_open else 0.0
            ref = f' ref="{ang}"' if is_open else ""
            A.append(f'    <body name="door_{i}" pos="{_f(ox, oy - ol / 2, 0)}" quat="{_yaw_quat(yaw + ang)}">\n'
                     f'      <joint name="door_{i}_hinge" type="hinge" axis="0 0 1" range="-1.7 0" limited="true" '
                     f'damping="8" stiffness="4" springref="{ang}"{ref}/>\n'
                     f'      <geom name="door_{i}_leaf" type="box" pos="{_f(0, ol / 2, oh / 2)}" '
                     f'size="{_f(ow / 2, ol / 2, oh / 2)}" contype="1" conaffinity="14" mass="20" '
                     f'rgba="0.75 0.25 0.2 1"/>\n    </body>\n')
            doors_meta.append(dict(name=f"door_{i}", joint=f"door_{i}_hinge", range=[-1.7, 0.0], open=is_open,
                                   hinge_xy=[ox, oy - ol / 2]))
        elif free:
            A.append(f'    <body name="obst_{i}" pos="{_f(ox, oy, oh / 2 + 0.002)}" quat="{q}">\n'
                     f'      <freejoint name="obst_{i}_free"/>\n'
                     f'      <geom name="obst_{i}_g" type="box" size="{_f(ow / 2, ol / 2, oh / 2)}" '
                     f'mass="{float(om) * scale:.4f}" contype="1" conaffinity="15" friction="0.8 0.005 0.0001" '
                     f'rgba="0.85 0.55 0.2 1"/>\n    </body>\n')
        else:
            rgba = "0.95 0.8 0.1 1" if kind == "barrier" else "0.55 0.45 0.35 1"
            A.append(f'    <geom name="obst_{i}" type="box" pos="{_f(ox, oy, oh / 2)}" size="{_f(ow / 2, ol / 2, oh / 2)}" '
                     f'quat="{q}" contype="1" conaffinity="14" rgba="{rgba}"/>\n')
            if kind == "barrier":
                A.append(f'    <geom name="obst_{i}_stripe" type="box" pos="{_f(ox, oy, oh * 0.8)}" '
                         f'size="{_f(ow / 2 + 0.002, ol / 2 + 0.002, oh * 0.1)}" quat="{q}" contype="0" '
                         f'conaffinity="0" rgba="0.08 0.08 0.08 1"/>\n')
        obst_meta.append(dict(name=f"door_{i}" if kind == "door" else f"obst_{i}", kind=kind, x=ox, y=oy, w=ow, l=ol,
                              h=oh, yaw=yaw, free=bool(free), open=is_open))
    for i, hz in enumerate(hazards):
        kind = getattr(hz, "kind", "")
        hx_, hy_ = float(hz.x), float(hz.y)
        r = float(getattr(hz, "radius", None) or getattr(hz, "r", None) or 0.4)
        if kind == "cable":
            A.append(f'    <geom name="cable_{i}" type="capsule" fromto="{_f(hx_ - r, hy_, 0.012)} {_f(hx_ + r, hy_, 0.012)}" '
                     f'size="0.012" contype="64" conaffinity="2" rgba="0.1 0.1 0.1 1"/>\n')
        elif kind == "low_light":
            A.append(f'    <site name="lowlight_{i}" type="cylinder" pos="{_f(hx_, hy_, 0.003)}" size="{_f(r)} 0.002" '
                     f'rgba="0.05 0.05 0.1 0.6"/>\n')
        haz_meta.append(dict(name=f"{kind}_{i}", kind=kind, x=hx_, y=hy_, radius=r))
    if terr is not None:
        meta["terrain"] = {**terr.model_dump(), "region": list(region) if region else None,
                           "has_hfield": hf is not None, "collides_with": "cart+products"}
        if hf is not None:
            meta["terrain"]["hf"] = hf
    else:
        meta["terrain"] = {"type": "flat", "region": None, "has_hfield": False, "collides_with": "cart+products"}
    meta["obstacles"], meta["doors"], meta["hazards"] = obst_meta, doors_meta, haz_meta
    meta["people"] = meta["n_shoppers"]
    meta["robots"] = int(getattr(spec, "robots", 1))
    meta["robot_starts"] = [list(map(float, p)) for p in S.robot_starts(spec)] if hasattr(S, "robot_starts") else [list(robot_xy)]
    meta["tasks"] = [t.model_dump() for t in getattr(spec, "tasks", [])]
    meta["spec_version"] = getattr(spec, "spec_version", 1)
    meta["template"] = getattr(spec, "template", "")

    # ---- shoppers
    shirt = [(0.85, 0.25, 0.3), (0.95, 0.65, 0.15), (0.35, 0.65, 0.35), (0.6, 0.35, 0.8), (0.9, 0.5, 0.6)]
    for i, p in enumerate(paths):
        c = shirt[i % len(shirt)]
        A.append(f'    <body name="shopper_{i}" mocap="true" pos="{_f(p[0][0], p[0][1], 0.0)}">\n'
                 f'      <geom name="shopper_{i}_g" type="capsule" fromto="0 0 0.25 0 0 1.4" size="0.19" rgba="{_f(*c)} 1" contype="16" conaffinity="6"/>\n'
                 f'      <geom name="shopper_{i}_legs" type="capsule" fromto="0 0 0.1 0 0 0.35" size="0.13" rgba="0.2 0.25 0.4 1" contype="16" conaffinity="6"/>\n'
                 f'      <geom name="shopper_{i}_head" type="sphere" pos="0 0 1.6" size="0.11" rgba="0.93 0.76 0.62 1" contype="16" conaffinity="6"/>\n'
                 f'    </body>\n')

    # ---- free products (prod_0 = target)
    def prod_body(i, x, y, lz, rgb):
        return (f'    <body name="prod_{i}" pos="{_f(x, y, lz + PROD_HALF[2] + 0.002)}">\n'
                f'      <freejoint name="prod_{i}_free"/>\n'
                f'      <geom name="prod_{i}_g" type="box" size="{_f(*PROD_HALF)}" mass="{prod_mass}" friction="0.9 0.005 0.0001" '
                f'rgba="{_f(*rgb)} 1" contype="8" conaffinity="43"/>\n    </body>\n')
    A.append(prod_body(0, t_xy[0], t_xy[1], t_ledge, TARGET_RGB))
    for n, (j, lz, x, p) in enumerate(placed, start=1):
        x0, y0, x1, y1 = rects[j]
        fd = _face(spec, j)
        y = (y1 if fd > 0 else y0) + fd * LEDGE_HY
        rgb = COLORS.get(p.color.lower(), PALETTE[n % len(PALETTE)])
        A.append(prod_body(n, x, y, lz, rgb))
    meta["n_products"] = 1 + len(placed)

    A.append(robot_x)
    A.append(cart_x)
    A.append("  </worldbody>\n")
    A.append(tail)
    return "".join(A), meta


# ------------------------------------------------------------------ checks
REQUIRED_META = ("target_product", "checkout_xy", "cart_xy", "robot_xy", "waypoints", "route",
                 "cart_mass_kg", "spills", "shopper_paths", "grip_body")
_O = mujoco.mjtObj
REQUIRED = [
    (_O.mjOBJ_BODY, ["robot", "torso", "head", "cart"]),
    (_O.mjOBJ_GEOM, ["checkout", "floor"]),
    (_O.mjOBJ_JOINT, ["base_x", "base_y", "base_yaw", "r_shoulder", "r_elbow", "l_shoulder", "l_elbow", "cart_free"]),
    (_O.mjOBJ_ACTUATOR, ["act_vx", "act_vy", "act_wz", "act_r_shoulder", "act_r_elbow", "act_l_shoulder", "act_l_elbow"]),
    (_O.mjOBJ_SENSOR, ["rf_front", "rf_left", "rf_right", "rf_cart", "touch_r", "f_wrist", "imu_acc", "imu_gyro",
                       "cart_up", "cart_pos", "jp_r_shoulder", "jp_r_elbow", "jp_l_shoulder", "jp_l_elbow"]),
    (_O.mjOBJ_SITE, ["r_hand", "l_hand", "cart_handle"]),
    (_O.mjOBJ_CAMERA, ["head_cam"]),
    (_O.mjOBJ_EQUALITY, ["grip_cart", "grip_prod"]),
]


def _inside(p, r, m):
    return r[0] - m < p[0] < r[2] + m and r[1] - m < p[1] < r[3] + m


def check_world(xml: str, meta: dict) -> list[str]:
    probs: list[str] = []
    try:
        model = compile_world(xml, meta)
    except Exception as e:  # noqa: BLE001
        return [f"compile error: {e}"]
    for k in REQUIRED_META:
        if k not in meta:
            probs.append(f"meta missing '{k}'")
    if probs:
        return probs

    def has(t, n):
        return mujoco.mj_name2id(model, t, n) >= 0

    for t, names in REQUIRED:
        for n in names:
            if not has(t, n):
                probs.append(f"missing {t.name[6:].lower()} '{n}'")
    if not has(_O.mjOBJ_BODY, meta["target_product"]):
        probs.append(f"missing target body '{meta['target_product']}'")
    if not any((mujoco.mj_id2name(model, _O.mjOBJ_GEOM, g) or "").startswith("shelf_") for g in range(model.ngeom)):
        probs.append("no shelf_* geoms")
    for i in range(len(meta["spills"])):
        if not has(_O.mjOBJ_GEOM, f"spill_{i}"):
            probs.append(f"missing spill_{i}")
    for i in range(len(meta["shopper_paths"])):
        if not has(_O.mjOBJ_BODY, f"shopper_{i}") or not has(_O.mjOBJ_GEOM, f"shopper_{i}_g"):
            probs.append(f"missing shopper_{i}")
    if abs(model.opt.timestep - 0.002) > 1e-9:
        probs.append("timestep != 0.002")
    if probs:
        return probs

    # stability with zero control
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    pid = mujoco.mj_name2id(model, _O.mjOBJ_BODY, meta["target_product"])
    cid = mujoco.mj_name2id(model, _O.mjOBJ_BODY, "cart")
    p0 = data.xpos[pid].copy()
    r0 = np.array(meta["robot_xy"], float)
    for _ in range(300):
        mujoco.mj_step(model, data)
    if not (np.all(np.isfinite(data.qpos)) and np.all(np.isfinite(data.sensordata))):
        probs.append("NaN after 300 zero-ctrl steps")
        return probs
    up = data.xmat[cid].reshape(3, 3)[:, 2]
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, float(up[2])))))
    if tilt > 10:
        probs.append(f"cart not upright after settle (tilt {tilt:.1f} deg)")
    if np.linalg.norm(data.xpos[mujoco.mj_name2id(model, _O.mjOBJ_BODY, "robot")][:2] - r0) > 0.3 and \
            np.linalg.norm(data.qpos[model.joint("base_x").qposadr[0]] - r0[0]) > 0.3:
        probs.append("robot drifted with zero control")
    if np.linalg.norm(data.xpos[pid] - p0) > 0.08:
        probs.append("target product moved/fell while settling (not resting on its ledge)")

    # free-space geometry of route / start / cart / spills (shelf footprints from the model)
    rects = []
    for g in range(model.ngeom):
        n = mujoco.mj_id2name(model, _O.mjOBJ_GEOM, g) or ""
        if (n.startswith("shelf_") and "_" not in n[6:]) or n == "checkout":
            c, s = data.geom_xpos[g], model.geom_size[g]
            rects.append((n, (c[0] - s[0], c[1] - s[1], c[0] + s[0], c[1] + s[1])))
        elif re.fullmatch(r"obst_\d+", n) or re.fullmatch(r"door_\d+_leaf", n):
            # static obstacle / door leaf: axis-aligned bound of the (possibly rotated) box at load
            c, s = data.geom_xpos[g], model.geom_size[g]
            ext = np.abs(data.geom_xmat[g].reshape(3, 3)) @ s
            rects.append((n, (c[0] - ext[0], c[1] - ext[1], c[0] + ext[0], c[1] + ext[1])))
    wps = meta["waypoints"]
    W, H = meta.get("store_size", [None, None])
    route = [wps[i] for i in meta["route"]]
    if len(route) < 3:
        probs.append("route has fewer than 3 waypoints")
    for i, p in enumerate(route):
        if W is not None and not (0.3 < p[0] < W - 0.3 and 0.3 < p[1] < H - 0.3):
            probs.append(f"route waypoint {i} {np.round(p, 2).tolist()} outside the room")
        for n, r in rects:
            if _inside(p, r, 0.3):
                probs.append(f"route waypoint {i} {np.round(p, 2).tolist()} overlaps {n}")
    for a, b in zip(route[:-1], route[1:]):
        for t in np.linspace(0, 1, 40):
            q = (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))
            hit = [n for n, r in rects if _inside(q, r, 0.2)]
            if hit:
                probs.append(f"route segment {np.round(a, 2).tolist()}->{np.round(b, 2).tolist()} crosses {hit[0]}")
                break
    for nm, p in (("robot_xy", meta["robot_xy"]), ("cart_xy", meta["cart_xy"])):
        for n, r in rects:
            if _inside(p, r, 0.4):
                probs.append(f"{nm} overlaps {n}")
    # target reachable: the shelf waypoint is the route point nearest to the product
    tp = data.xpos[pid][:2]
    dist = [math.hypot(p[0] - tp[0], p[1] - tp[1]) for p in route]
    k = int(np.argmin(dist))
    if dist[k] > 1.2:
        probs.append(f"route passes {dist[k]:.2f} m from the target product (>1.2 m)")
    if math.hypot(route[-1][0] - meta["checkout_xy"][0], route[-1][1] - meta["checkout_xy"][1]) > 1.6:
        probs.append("route does not end near checkout")
    return probs


# ------------------------------------------------------------------ preview
def render_preview(xml: str, meta: dict, path: str) -> str:
    """1280x720 3/4 overview PNG with the planned route overlaid."""
    import os
    from PIL import Image

    model = compile_world(xml, meta)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    w, h = meta.get("store_size", [12.0, 8.0])
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = (w / 2, h * 0.5, 0.3)
    cam.distance = max(w, h * 1.5) * 0.95
    cam.azimuth, cam.elevation = 72.0, -48.0
    r = mujoco.Renderer(model, 720, 1280)
    try:
        opt = mujoco.MjvOption()
        opt.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = False
        r.update_scene(data, camera=cam, scene_option=opt)
        scn = r.scene

        def add(kind, rgba, size, p0, p1=None):
            if scn.ngeom >= scn.maxgeom:
                return
            g = scn.geoms[scn.ngeom]
            mujoco.mjv_initGeom(g, kind, np.zeros(3), np.zeros(3), np.zeros(9), np.array(rgba, np.float32))
            if p1 is None:
                mujoco.mjv_initGeom(g, kind, np.array(size, float), np.array(p0, float), np.eye(3).ravel(),
                                    np.array(rgba, np.float32))
            else:
                mujoco.mjv_connector(g, kind, size[0], np.array(p0, float), np.array(p1, float))
            scn.ngeom += 1

        pts = [meta["waypoints"][i] for i in meta["route"]]
        for a, b in zip(pts[:-1], pts[1:]):
            add(mujoco.mjtGeom.mjGEOM_CAPSULE, (0.1, 0.8, 0.2, 1), (0.03,), (a[0], a[1], 0.05), (b[0], b[1], 0.05))
        for p in pts:
            add(mujoco.mjtGeom.mjGEOM_SPHERE, (1.0, 0.85, 0.1, 1), (0.09, 0, 0), (p[0], p[1], 0.06))
        tx, ty = meta["target_xy"]
        add(mujoco.mjtGeom.mjGEOM_SPHERE, (1.0, 0.1, 0.6, 0.9), (0.12, 0, 0), (tx, ty, meta["target_z"] + 0.45))
        add(mujoco.mjtGeom.mjGEOM_CAPSULE, (1.0, 0.1, 0.6, 0.9), (0.012,), (tx, ty, meta["target_z"] + 0.1), (tx, ty, meta["target_z"] + 0.4))
        for pth in meta.get("shopper_paths", []):
            for a, b in zip(pth[:-1], pth[1:]):
                add(mujoco.mjtGeom.mjGEOM_CAPSULE, (0.9, 0.3, 0.3, 0.7), (0.015,), (a[0], a[1], 0.04), (b[0], b[1], 0.04))
        for hz in meta.get("hazards", []):
            if hz.get("kind") == "low_light":
                add(mujoco.mjtGeom.mjGEOM_CYLINDER, (0.9, 0.1, 0.1, 0.45), (hz["radius"], 0.004, 0), (hz["x"], hz["y"], 0.01))
        for d in meta.get("doors", []):
            hx_, hy_ = d["hinge_xy"]
            add(mujoco.mjtGeom.mjGEOM_SPHERE, (1.0, 0.0, 1.0, 1), (0.12, 0, 0), (hx_, hy_, 0.25))
        img = r.render()
    finally:
        r.close()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    Image.fromarray(img).save(path)
    return path


if __name__ == "__main__":
    import json
    import os

    from iz.worldgen.generate import demo_specs

    for slug, spec in demo_specs().items():
        xml, meta = build_world(spec, seed=0)
        probs = check_world(xml, meta)
        print(slug, probs)
        d = os.path.join("out", "worldgen", slug)
        os.makedirs(d, exist_ok=True)
        render_preview(xml, meta, os.path.join(d, "preview.png"))
        with open(os.path.join(d, "spec.json"), "w", encoding="utf-8") as fh:
            json.dump(spec.model_dump(), fh, indent=1, default=float)
        with open(os.path.join(d, "meta.json"), "w", encoding="utf-8") as fh:
            json.dump(meta, fh, default=float)
