"""SHOP scene: a MuJoCo retail store with a wheeled humanoid robot, a shopping cart,
shelves with products, spills and walking shoppers.  build_shop(seed) -> (xml, meta).

Collision bit layout (contype = own bit, conaffinity = bits collided with):
  1 static world (floor, shelves, ledges, counter, walls)   2 cart   4 robot body
  8 products   16 shoppers (mocap)   32 robot hands
Robot<->cart collision is disabled on purpose (they are welded when pushing).
Robot joints base_x/base_y/base_yaw have qpos == world x, y, yaw (robot starts at yaw 0).
Hand body names: "r_hand_b", "l_hand_b" (jointless child bodies; meta["hand_body"]).
Cart wheels are CYLINDERS on hinge joints (spheres are omni-directional and could never
tip a cart); wheel friction is 1.0 on the floor and the spill's 0.04 on spills (geom priority).
Waypoints are ROBOT-centre coordinates; when gripped, the cart centre is meta["grip_offset"]
metres ahead of the robot.  meta["checkout_xy"] is the CART target zone.
Seeds >= 200 produce a dense-shopper layout (safety bank).
Arm kinematics: shoulder/elbow axes are along the robot x axis, so each arm swings sideways
(right arm towards robot -y).  Pick: robot yaw meta["pick_yaw"]; place into cart: yaw
meta["place_yaw"] (pi/2, right side faces the cart ahead).
"""
import math
import random

import numpy as np

STORE_HX, STORE_HY = 6.1, 4.1
SHELF_YS = [-3.0, -1.0, 1.0, 3.0]
AISLE_YS = [-2.0, 0.0, 2.0]
SHELF_CX, SHELF_HX, SHELF_HY, SHELF_HZ = 0.0, 2.2, 0.18, 0.75
LEDGE_ZS = [0.60, 0.90, 1.20]
LEDGE_HY = 0.08            # ledge half depth (protrudes 0.16 m from shelf face)
SHELF_FACE = 1.0 - SHELF_HY   # aisle centre -> shelf face distance
PROD_HALF = (0.035, 0.035, 0.07)
GRIP_OFFSET = 1.0          # cart centre is this far ahead of the robot centre when gripped
PALETTE = [(0.9, 0.2, 0.2), (0.2, 0.55, 0.9), (0.95, 0.75, 0.1), (0.25, 0.7, 0.35),
           (0.85, 0.4, 0.75), (0.95, 0.5, 0.15), (0.3, 0.8, 0.8)]

# weld softness (tuned so a 30 kg cart tips at ~1.4 m/s turning but not at 0.6 m/s)
TRACK = 0.20
TIP_TORQUESCALE = 0.05
TIP_SOLREF = "0.06 1"


def _f(*v):
    return " ".join(f"{x:.4f}" for x in v)


def _xyaxes(pos, target):
    d = np.array(target, float) - np.array(pos, float)
    d /= np.linalg.norm(d)
    x = np.cross(d, [0, 0, 1.0])
    x /= np.linalg.norm(x)
    y = np.cross(x, d)
    return _f(*x, *y)


def _seg_point(rng, a, b, lo=0.2, hi=0.8):
    t = rng.uniform(lo, hi)
    return [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])]


def _arm(s, n, y, sh=1.2):
    ax = "1 0 0" if s > 0 else "-1 0 0"
    return f"""      <body name="{n}_upper" pos="0 {y} {sh}">
        <joint name="{n}_shoulder" type="hinge" axis="{ax}" range="-0.5 3.0" damping="2" armature="0.02"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.34" size="0.045" mass="0.8" rgba="0.62 0.72 0.88 1" contype="0" conaffinity="0"/>
        <geom type="sphere" size="0.06" mass="0.2" rgba="0.5 0.6 0.8 1" contype="0" conaffinity="0"/>
        <body name="{n}_fore" pos="0 0 -0.34">
          <joint name="{n}_elbow" type="hinge" axis="{ax}" range="-2.4 2.4" damping="1.5" armature="0.01"/>
          <geom type="capsule" fromto="0 0 0 0 0 -0.36" size="0.037" mass="0.5" rgba="0.62 0.72 0.88 1" contype="0" conaffinity="0"/>
          <geom type="sphere" size="0.045" mass="0.1" rgba="0.5 0.6 0.8 1" contype="0" conaffinity="0"/>
          <body name="{n}_hand_b" pos="0 0 -0.36">
            <geom name="{n}_hand_g" type="sphere" pos="0 0 -0.05" size="0.045" mass="0.15" rgba="0.2 0.22 0.27 1" contype="32" conaffinity="8"/>
            <geom type="box" pos="0.025 0 -0.11" size="0.008 0.02 0.04" mass="0.01" rgba="0.15 0.15 0.18 1" contype="0" conaffinity="0"/>
            <geom type="box" pos="-0.025 0 -0.11" size="0.008 0.02 0.04" mass="0.01" rgba="0.15 0.15 0.18 1" contype="0" conaffinity="0"/>
            <site name="{n}_hand" pos="0 0 -0.06" size="0.02" rgba="1 0.3 0.1 1"/>
            <site name="{n}_wrist" pos="0 0 0" size="0.02" rgba="0.2 1 0.3 0.5"/>
          </body>
        </body>
      </body>
"""


def build_shop(seed: int, cart_mass_kg=None, n_shoppers=None):
    rng = random.Random(int(seed) * 7919 + 13)
    dense = seed >= 200
    aisle = rng.randrange(3)
    ay = AISLE_YS[aisle]
    side = rng.choice([-1, 1])                    # -1: product on lower-y shelf, +1: upper-y shelf
    shelf_i = aisle if side < 0 else aisle + 1
    face_dir = -side                              # direction from shelf centre towards the aisle
    ledge_k = rng.randrange(3)
    ledge_z = LEDGE_ZS[ledge_k]
    pick_x = rng.uniform(-1.4, 1.5)
    prod_y = ay + side * (SHELF_FACE - LEDGE_HY)
    prod_z = ledge_z + PROD_HALF[2] + 0.002
    mass = float(cart_mass_kg) if cart_mass_kg is not None else rng.uniform(12, 35)
    cy = rng.uniform(-1.0, 1.0)
    cart_xy = [-4.3, cy]
    robot_xy = [cart_xy[0] - GRIP_OFFSET - rng.uniform(0.15, 0.45), cy + rng.uniform(-0.8, 0.8)]
    pick_yaw = 0.0 if side < 0 else math.pi

    wps = [[cart_xy[0] - GRIP_OFFSET, cy]]                     # 0 grip pose
    for a in AISLE_YS:                                         # entry nodes 1..3
        wps.append([-4.7, a])
    for a in AISLE_YS:                                         # exit nodes 4..6
        wps.append([3.6, a])
    pick_xy = [pick_x, ay + side * 0.2]   # hug the product side so the arm reaches
    wps.append(pick_xy)                                        # 7 pick point
    checkout_robot = [3.6, -2.3]
    wps.append(checkout_robot)                                 # 8 checkout (robot)
    route = [0, 1 + aisle, 7, 4 + aisle, 8]
    checkout_xy = [3.6, -3.3]                                  # cart zone

    spills = []
    segs = [(wps[i], wps[j]) for i, j in zip(route[:-1], route[1:])]
    for _ in range(rng.choice([1, 2, 2, 3])):
        a, b = rng.choice(segs)
        p = _seg_point(rng, a, b)
        if math.hypot(p[0] - pick_x, p[1] - ay) < 1.2 or p[0] < -4.6:
            continue
        spills.append([p[0], p[1], 0.55, 0.40])

    paths = []
    n_sh = n_shoppers if n_shoppers is not None else (rng.randint(4, 5) if dense else rng.randint(1, 3))
    for k in range(n_sh):
        kind = rng.random()
        if kind < 0.55:
            a = AISLE_YS[aisle] if (k == 0 or dense) and rng.random() < 0.6 else rng.choice(AISLE_YS)
            y = a + rng.uniform(-0.2, 0.2)
            x0, x1 = (-2.2, 2.2) if rng.random() < 0.5 else (2.2, -2.2)
            paths.append([[x0, y], [x1, y]])
        elif kind < 0.8:
            x = rng.uniform(3.2, 5.0)
            paths.append([[x, 3.2], [x, -1.0]] if rng.random() < 0.5 else [[x, -1.0], [x, 3.2]])
        else:
            x = rng.uniform(-3.4, -2.8)          # cross-aisle walker at the aisle mouths
            paths.append([[x, 3.5], [x, -3.5]] if rng.random() < 0.5 else [[x, -3.5], [x, 3.5]])

    meta = dict(
        seed=int(seed), dense=dense, cart_xy=cart_xy, cart_yaw=0.0, robot_xy=robot_xy, robot_yaw=0.0,
        grip_body="robot", grip_offset=GRIP_OFFSET, hand_body="r_hand_b",
        target_product="prod_0", target_xy=[pick_x, prod_y], target_z=prod_z,
        target_aisle=aisle, shelf_side=side, shelf_index=shelf_i, ledge_z=ledge_z,
        pick_xy=pick_xy, pick_yaw=pick_yaw, place_yaw=math.pi / 2,
        waypoints=wps, route=route, checkout_xy=checkout_xy, checkout_robot_xy=checkout_robot,
        checkout_heading=-math.pi / 2, spills=spills, shopper_paths=paths, n_shoppers=n_sh,
        cart_mass_kg=mass, store_half=[STORE_HX, STORE_HY], aisle_ys=AISLE_YS,
        shelf_geoms=[f"shelf_{i}" for i in range(4)], shopper_speed=0.8,
    )

    A = []
    A.append(f"""<mujoco model="shop">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002" integrator="implicitfast" gravity="0 0 -9.81"/>
  <visual><global offwidth="1280" offheight="960"/><quality shadowsize="2048"/>
    <headlight ambient="0.45 0.45 0.45" diffuse="0.5 0.5 0.5"/></visual>
  <asset>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.80 0.88 0.97" rgb2="0.93 0.95 0.98" width="64" height="64"/>
    <texture name="floortex" type="2d" builtin="checker" rgb1="0.90 0.91 0.93" rgb2="0.82 0.84 0.87" width="256" height="256"/>
    <material name="floor" texture="floortex" texrepeat="12 8" reflectance="0.05"/>
  </asset>
  <worldbody>
    <light pos="0 0 6" dir="0 0.2 -1" diffuse="0.6 0.6 0.6" castshadow="false"/>
    <light pos="-3 -4 5" dir="0.4 0.5 -1" diffuse="0.35 0.35 0.35"/>
    <geom name="floor" type="plane" size="{STORE_HX} {STORE_HY} 0.1" material="floor" friction="1 0.005 0.0001"
          contype="1" conaffinity="10" priority="0"/>
    <camera name="cam_34" pos="-1.0 -6.8 8.6" xyaxes="{_xyaxes((-1.0, -6.8, 8.6), (0.2, 0.2, 0.3))}" fovy="58"/>
    <camera name="cam_top" pos="0 -1.0 11" xyaxes="1 0 0 0 1 0" fovy="55"/>
""")
    for nm, p, s in [("wall_n", (0, STORE_HY, 0.25), (STORE_HX, 0.05, 0.25)),
                     ("wall_s", (0, -STORE_HY, 0.25), (STORE_HX, 0.05, 0.25)),
                     ("wall_e", (STORE_HX, 0, 0.25), (0.05, STORE_HY, 0.25)),
                     ("wall_w", (-STORE_HX, 0, 0.25), (0.05, STORE_HY, 0.25))]:
        A.append(f'    <geom name="{nm}" type="box" pos="{_f(*p)}" size="{_f(*s)}" rgba="0.55 0.58 0.64 1" contype="1" conaffinity="14"/>\n')
    for i, sy in enumerate(SHELF_YS):
        A.append(f'    <geom name="shelf_{i}" type="box" pos="{_f(SHELF_CX, sy, SHELF_HZ)}" size="{_f(SHELF_HX, SHELF_HY, SHELF_HZ)}" '
                 f'rgba="0.93 0.92 0.88 1" contype="1" conaffinity="14"/>\n')
        A.append(f'    <geom name="shelf_{i}_top" type="box" pos="{_f(SHELF_CX, sy, 1.52)}" size="{_f(SHELF_HX + 0.02, SHELF_HY + 0.02, 0.02)}" '
                 f'rgba="0.15 0.4 0.7 1" contype="0" conaffinity="0"/>\n')
        for fd in (-1, 1):
            if (i == 0 and fd > 0 and False) or (i == 0 and fd < 0) or (i == 3 and fd > 0):
                continue
            ly = sy + fd * (SHELF_HY + LEDGE_HY)
            for k, lz in enumerate(LEDGE_ZS):
                A.append(f'    <geom name="shelf_{i}_ledge_{fd}_{k}" type="box" pos="{_f(SHELF_CX, ly, lz - 0.01)}" '
                         f'size="{_f(SHELF_HX, LEDGE_HY, 0.01)}" rgba="0.45 0.52 0.62 1" contype="1" conaffinity="14"/>\n')
                drng = random.Random(i * 100 + (fd + 1) * 10 + k + seed)
                x = SHELF_CX - SHELF_HX + 0.25
                while x < SHELF_CX + SHELF_HX - 0.2:
                    on_target = (i == shelf_i and fd == face_dir and k == ledge_k and abs(x - pick_x) < 1.0)
                    if not on_target:
                        c = drng.choice(PALETTE)
                        hz = drng.uniform(0.04, 0.09)
                        A.append(f'    <geom type="box" pos="{_f(x, ly, lz + hz)}" size="{_f(0.045, 0.05, hz)}" '
                                 f'rgba="{_f(*c)} 1" contype="0" conaffinity="0"/>\n')
                    x += 0.34
    A.append('    <geom name="checkout" type="box" pos="5.1 -3.1 0.45" size="0.4 0.8 0.45" rgba="0.2 0.55 0.45 1" contype="1" conaffinity="14"/>\n')
    A.append('    <geom name="checkout_top" type="box" pos="5.1 -3.1 0.91" size="0.43 0.83 0.015" rgba="0.9 0.9 0.92 1" contype="0" conaffinity="0"/>\n')
    A.append('    <geom name="checkout_screen" type="box" pos="5.3 -3.6 1.1" size="0.04 0.14 0.12" rgba="0.1 0.12 0.18 1" contype="0" conaffinity="0"/>\n')
    A.append(f'    <site name="checkout_zone" pos="{_f(*checkout_xy, 0.02)}" size="0.45 0.01" type="cylinder" rgba="0.2 0.9 0.4 0.35"/>\n')
    for i, (sx, sy_, hx, hy) in enumerate(spills):
        A.append(f'    <geom name="spill_{i}" type="box" pos="{_f(sx, sy_, 0.001)}" size="{_f(hx, hy, 0.001)}" '
                 f'friction="0.04 0.005 0.0001" priority="2" rgba="0.2 0.55 1.0 0.85" contype="1" conaffinity="2"/>\n')
    shirt = [(0.85, 0.25, 0.3), (0.95, 0.65, 0.15), (0.35, 0.65, 0.35), (0.6, 0.35, 0.8), (0.9, 0.5, 0.6)]
    for i, p in enumerate(paths):
        c = shirt[i % len(shirt)]
        A.append(f'    <body name="shopper_{i}" mocap="true" pos="{_f(p[0][0], p[0][1], 0.0)}">\n'
                 f'      <geom name="shopper_{i}_g" type="capsule" fromto="0 0 0.25 0 0 1.4" size="0.19" rgba="{_f(*c)} 1" contype="16" conaffinity="6"/>\n'
                 f'      <geom name="shopper_{i}_legs" type="capsule" fromto="0 0 0.1 0 0 0.35" size="0.13" rgba="0.2 0.25 0.4 1" contype="16" conaffinity="6"/>\n'
                 f'      <geom name="shopper_{i}_head" type="sphere" pos="0 0 1.6" size="0.11" rgba="0.93 0.76 0.62 1" contype="16" conaffinity="6"/>\n'
                 f'    </body>\n')
    prods = [(pick_x, ledge_z), (pick_x - 0.4, ledge_z), (pick_x + 0.4, ledge_z),
             (pick_x - 0.2, LEDGE_ZS[(ledge_k + 1) % 3]), (pick_x + 0.3, LEDGE_ZS[(ledge_k + 2) % 3])]
    for i, (px, pz) in enumerate(prods):
        c = (1.0, 0.15, 0.55) if i == 0 else PALETTE[i % len(PALETTE)]
        A.append(f'    <body name="prod_{i}" pos="{_f(px, prod_y, pz + PROD_HALF[2] + 0.002)}">\n'
                 f'      <freejoint name="prod_{i}_free"/>\n'
                 f'      <geom name="prod_{i}_g" type="box" size="{_f(*PROD_HALF)}" mass="0.3" friction="0.9 0.005 0.0001" '
                 f'rgba="{_f(*c)} 1" contype="8" conaffinity="43"/>\n    </body>\n')

    A.append(f"""    <body name="robot" pos="{_f(robot_xy[0], robot_xy[1], 0)}">
      <joint name="base_x" type="slide" axis="1 0 0" ref="{robot_xy[0]:.4f}" damping="5"/>
      <joint name="base_y" type="slide" axis="0 1 0" ref="{robot_xy[1]:.4f}" damping="5"/>
      <joint name="base_yaw" type="hinge" axis="0 0 1" damping="5"/>
      <geom name="base_g" type="box" pos="0 0 0.17" size="0.26 0.22 0.07" mass="14" rgba="0.28 0.33 0.42 1" contype="4" conaffinity="17"/>
      <geom type="cylinder" pos="0 0.25 0.09" euler="1.5708 0 0" size="0.09 0.02" mass="0.5" rgba="0.1 0.1 0.12 1" contype="0" conaffinity="0"/>
      <geom type="cylinder" pos="0 -0.25 0.09" euler="1.5708 0 0" size="0.09 0.02" mass="0.5" rgba="0.1 0.1 0.12 1" contype="0" conaffinity="0"/>
      <geom type="sphere" pos="0.2 0 0.04" size="0.04" mass="0.1" rgba="0.1 0.1 0.12 1" contype="0" conaffinity="0"/>
      <geom type="sphere" pos="-0.2 0 0.04" size="0.04" mass="0.1" rgba="0.1 0.1 0.12 1" contype="0" conaffinity="0"/>
      <geom type="box" pos="0.262 0 0.17" size="0.004 0.1 0.02" mass="0.01" rgba="0.3 0.9 1 1" contype="0" conaffinity="0"/>
      <site name="imu" pos="0 0 0.2" size="0.02"/>
      <body name="torso" pos="0 0 0">
        <geom name="waist_g" type="capsule" fromto="0 0 0.24 0 0 0.72" size="0.09" mass="3" rgba="0.55 0.6 0.7 1" contype="4" conaffinity="17"/>
        <geom name="torso_g" type="box" pos="0 0 1.0" size="0.12 0.21 0.28" mass="8" rgba="0.66 0.75 0.90 1" contype="4" conaffinity="17"/>
        <geom type="box" pos="0.121 0 1.05" size="0.004 0.12 0.1" mass="0.01" rgba="0.95 0.95 0.98 1" contype="0" conaffinity="0"/>
        <geom type="capsule" fromto="0 -0.2 1.22 0 0.2 1.22" size="0.05" mass="0.5" rgba="0.62 0.72 0.88 1" contype="0" conaffinity="0"/>
        <geom type="cylinder" pos="0 0 1.31" size="0.04 0.03" mass="0.1" rgba="0.5 0.55 0.65 1" contype="0" conaffinity="0"/>
        <body name="head" pos="0 0 1.42">
          <geom name="head_g" type="sphere" size="0.105" mass="1" rgba="0.97 0.97 0.98 1" contype="4" conaffinity="17"/>
          <geom type="box" pos="0.085 0 0.01" size="0.025 0.085 0.04" mass="0.01" rgba="0.05 0.06 0.1 1" contype="0" conaffinity="0"/>
          <camera name="head_cam" pos="0.11 0 0.01" xyaxes="0 -1 0 0 0 1" fovy="70"/>
          <site name="rf_front_s" pos="0.105 0 0" zaxis="1 0 0" size="0.01"/>
          <site name="rf_left_s" pos="0 0.105 0" zaxis="0 1 0" size="0.01"/>
          <site name="rf_right_s" pos="0 -0.105 0" zaxis="0 -1 0" size="0.01"/>
        </body>
{_arm(+1, 'l', 0.26)}{_arm(-1, 'r', -0.26)}      </body>
      <camera name="cam_follow" mode="trackcom" pos="-2.8 -3.6 3.0" xyaxes="{_xyaxes((-2.8, -3.6, 3.0), (0, 0, 0.8))}" fovy="55"/>
    </body>
""")
    cmass = mass
    com_z = (8 * 0.42 + max(cmass - 8, 1) * 0.98) / cmass          # loaded -> higher CoM
    ih = 1.0
    Ixx = cmass * (0.56 ** 2 + ih ** 2) / 12 + 0.5
    Iyy = cmass * (0.9 ** 2 + ih ** 2) / 12 + 0.5
    Izz = cmass * (0.9 ** 2 + 0.56 ** 2) / 12 + 0.5
    wheel = ""
    wg = ('friction="1.0 0.005 0.0001" priority="1" rgba="0.1 0.1 0.12 1" contype="2" conaffinity="1"')
    for wi, (wx, wy) in enumerate([(0.34, TRACK), (0.34, -TRACK), (-0.34, TRACK), (-0.34, -TRACK)]):
        axle = (f'<body name="cart_wheel_{wi}" pos="{"-0.04 0 -0.07" if wx > 0 else _f(wx, wy, 0.07)}">'
                f'<joint name="cart_wheel_{wi}_j" type="hinge" axis="0 1 0" damping="0.02"/>'
                f'<geom name="cart_wheel_{wi}_g" type="cylinder" euler="1.5708 0 0" size="0.07 0.02" mass="0.3" {wg}/>'
                f'</body>')
        if wx > 0:   # front wheels are swivel casters (like a real cart); rear wheels are fixed
            wheel += (f'      <body name="cart_caster_{wi}" pos="{_f(wx, wy, 0.14)}">\n'
                      f'        <joint name="cart_caster_{wi}_j" type="hinge" axis="0 0 1" damping="0.05"/>\n'
                      f'        <geom type="box" size="0.01 0.025 0.03" contype="0" conaffinity="0" rgba="0.5 0.5 0.55 1" mass="0.05"/>\n'
                      f'        {axle}\n      </body>\n')
        else:
            wheel += f'      {axle}\n'
    A.append(f"""    <body name="cart" pos="{_f(cart_xy[0], cart_xy[1], 0.0)}">
      <freejoint name="cart_free"/>
      <inertial pos="0 0 {com_z:.4f}" mass="{cmass - 1.2:.3f}" diaginertia="{_f(Ixx, Iyy, Izz)}"/>
      <geom name="cart_floor" type="box" pos="0 0 0.30" size="0.45 0.28 0.012" rgba="0.75 0.77 0.8 1" contype="2" conaffinity="9" mass="0"/>
      <geom name="cart_wall_l" type="box" pos="0 0.28 0.58" size="0.45 0.008 0.28" rgba="0.8 0.82 0.86 0.55" contype="2" conaffinity="9" mass="0"/>
      <geom name="cart_wall_r" type="box" pos="0 -0.28 0.58" size="0.45 0.008 0.28" rgba="0.8 0.82 0.86 0.55" contype="2" conaffinity="9" mass="0"/>
      <geom name="cart_wall_f" type="box" pos="0.45 0 0.58" size="0.008 0.28 0.28" rgba="0.8 0.82 0.86 0.55" contype="2" conaffinity="9" mass="0"/>
      <geom name="cart_wall_b" type="box" pos="-0.45 0 0.58" size="0.008 0.28 0.28" rgba="0.8 0.82 0.86 0.55" contype="2" conaffinity="9" mass="0"/>
      <geom name="cart_handle_g" type="capsule" fromto="-0.5 -0.26 0.95 -0.5 0.26 0.95" size="0.02" rgba="0.9 0.2 0.2 1" contype="0" conaffinity="0" mass="0"/>
      <geom type="capsule" fromto="-0.45 0.26 0.86 -0.5 0.26 0.95" size="0.015" rgba="0.9 0.2 0.2 1" contype="0" conaffinity="0" mass="0"/>
      <geom type="capsule" fromto="-0.45 -0.26 0.86 -0.5 -0.26 0.95" size="0.015" rgba="0.9 0.2 0.2 1" contype="0" conaffinity="0" mass="0"/>
      <geom type="box" pos="0 0 0.20" size="0.4 0.03 0.012" rgba="0.5 0.5 0.55 1" contype="0" conaffinity="0" mass="0"/>
      <site name="cart_handle" pos="-0.5 0 0.95" size="0.03" rgba="1 0.2 0.2 1"/>
      <site name="cart_front" pos="0.46 0 0.45" zaxis="1 0 0" size="0.02" rgba="1 1 0 1"/>
{wheel}    </body>
  </worldbody>
""")
    A.append(f"""  <equality>
    <weld name="grip_cart" body1="robot" body2="cart" anchor="-0.5 0 0.95" relpose="{GRIP_OFFSET - 0.5} 0 0.95 1 0 0 0" torquescale="{TIP_TORQUESCALE}" solref="{TIP_SOLREF}" active="false"/>
    <weld name="grip_prod" body1="r_hand_b" body2="prod_0" relpose="0 0 -0.095 1 0 0 0" solref="0.04 1" active="false"/>
  </equality>
  <actuator>
    <velocity name="act_vx" joint="base_x" kv="900" ctrlrange="-2 2" forcerange="-450 450"/>
    <velocity name="act_vy" joint="base_y" kv="900" ctrlrange="-2 2" forcerange="-450 450"/>
    <velocity name="act_wz" joint="base_yaw" kv="1500" ctrlrange="-3 3" forcerange="-500 500"/>
    <position name="act_r_shoulder" joint="r_shoulder" kp="120" ctrlrange="-0.5 3.0" forcerange="-40 40"/>
    <position name="act_r_elbow" joint="r_elbow" kp="90" ctrlrange="-2.4 2.4" forcerange="-30 30"/>
    <position name="act_l_shoulder" joint="l_shoulder" kp="120" ctrlrange="-0.5 3.0" forcerange="-40 40"/>
    <position name="act_l_elbow" joint="l_elbow" kp="90" ctrlrange="-2.4 2.4" forcerange="-30 30"/>
  </actuator>
  <sensor>
    <rangefinder name="rf_front" site="rf_front_s" cutoff="8"/>
    <rangefinder name="rf_left" site="rf_left_s" cutoff="8"/>
    <rangefinder name="rf_right" site="rf_right_s" cutoff="8"/>
    <rangefinder name="rf_cart" site="cart_front" cutoff="8"/>
    <touch name="touch_r" site="r_hand"/>
    <force name="f_wrist" site="r_wrist"/>
    <accelerometer name="imu_acc" site="imu"/>
    <gyro name="imu_gyro" site="imu"/>
    <framezaxis name="cart_up" objtype="body" objname="cart"/>
    <framepos name="cart_pos" objtype="body" objname="cart"/>
    <jointpos name="jp_r_shoulder" joint="r_shoulder"/>
    <jointpos name="jp_r_elbow" joint="r_elbow"/>
    <jointpos name="jp_l_shoulder" joint="l_shoulder"/>
    <jointpos name="jp_l_elbow" joint="l_elbow"/>
  </sensor>
</mujoco>
""")
    return "".join(A), meta
