"""Unitree G1 (MuJoCo Menagerie) mounted as a KINEMATIC PUPPET on the shop's planar base joints.

HONEST CLAIM: this robot does not learn to walk.  The pelvis is bolted at a fixed height
(PELVIS_Z) onto planar joints base_x / base_y / base_yaw that are driven by velocity
actuators, and the legs are driven by a PARAMETRIC GAIT ENGINE (iz/robot/gait.py), not a
learned locomotion policy.  Legs and feet never collide with anything.  Learning in this
project happens at the behaviour, environment and fleet level.  The simulator is authored by us.

What is built here (xml.etree on Menagerie g1.xml, no MjSpec, output is a plain XML string):
  * every G1 joint except the 8 leg joints (hip pitch/roll, knee, ankle pitch, both sides) and
    the 4 arm joints (shoulder pitch, elbow) is deleted, so those links are rigid;
  * the freejoint is removed, the pelvis hangs off the `robot{sfx}` body;
  * all collision-class geoms are dropped and every remaining G1 geom gets contype=0
    conaffinity=0; collision uses hidden proxies base_g / waist_g / torso_g / head_g
    (bits 4 vs 17) and the hand sphere r_hand_g / l_hand_g (bits 32 vs 8);
  * arm joints are renamed r_shoulder, r_elbow, l_shoulder, l_elbow with axis 0 -1 0: +q raises
    the arm FORWARD.  The G1 forearm points forward at q=0; the forearm frame is rotated by
    +90 deg about y so that the forearm hangs DOWN at q=0 and +q_elbow swings it forward/up
    (matches the old shop robot, whose hand frame has -z along the arm).

Contract names (suffix `_{robot_id}` for robot_id > 0, applied to every body, joint, geom,
site, camera, actuator and sensor of the fragment): robot, torso, head, r_hand_b, l_hand_b,
r_hand, l_hand, r_wrist, l_wrist, head_cam, cam_follow, base_x/base_y/base_yaw, r_/l_shoulder,
r_/l_elbow, act_vx/act_vy/act_wz, act_r_shoulder ..., rf_front/left/right, touch_r, f_wrist,
imu_acc, imu_gyro, jp_*.  Meshes, materials and default classes are shared (never suffixed).

Licence of the G1 model: BSD-3-Clause, see iz/robot/LICENSE_NOTES.md.
"""
from __future__ import annotations

import copy
import functools
import math
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from iz.robot import G1_DIR

PELVIS_Z = 0.79
KP_LEG = 150
KV_LEG = 5
LEG_JOINTS = [f"{s}_{j}_joint" for s in ("left", "right") for j in ("hip_pitch", "hip_roll", "knee", "ankle_pitch")]
ARM_JOINTS = {  # G1 joint -> contract name
    "right_shoulder_pitch_joint": "r_shoulder", "right_elbow_joint": "r_elbow",
    "left_shoulder_pitch_joint": "l_shoulder", "left_elbow_joint": "l_elbow",
}
HEAD_POS = (0.008, 0.0, 0.385)            # head_link mesh centre in the torso_link frame (measured)
HAND_POS = (0.04, 0.0, 0.0)               # hand body in the wrist_yaw frame (forearm = +x)
HAND_QUAT = "0.7071068 0 -0.7071068 0"    # hand frame: -z along the forearm (as the old scene)


def _f(*v):
    return " ".join(f"{x:.4f}".rstrip("0").rstrip(".") if abs(x) > 1e-9 else "0" for x in v)


def _xyaxes(pos, target):
    d = np.array(target, float) - np.array(pos, float)
    d /= np.linalg.norm(d)
    x = np.cross(d, [0, 0, 1.0])
    x /= np.linalg.norm(x)
    y = np.cross(x, d)
    return _f(*x, *y)


def _inner(el: ET.Element) -> str:
    return "".join(ET.tostring(c, encoding="unicode") for c in el)


def _rot_y(deg_rad: float, v):
    c, s = math.cos(deg_rad), math.sin(deg_rad)
    return (v[0] * c + v[2] * s, v[1], -v[0] * s + v[2] * c)


def _g1_root() -> ET.Element:
    return ET.parse(str(G1_DIR / "g1.xml")).getroot()


def _sub(body: ET.Element, sfx: str, rename: dict) -> None:
    for el in body.iter():
        n = el.get("name")
        if n is not None and el.tag in ("body", "joint", "geom", "site", "camera"):
            el.set("name", rename.get(n, n) + sfx)


def _hidden(geom: ET.Element) -> None:
    geom.set("group", "3")
    geom.set("rgba", "0 0 0 0")
    geom.set("mass", "0")
    geom.set("contype", "4")
    geom.set("conaffinity", "17")


def _build(robot_id: int, xy, yaw: float):
    """Return (fragment_xml, assets, leg_joint_ranges, arm_joint_ranges) without arm meta."""
    sfx = "" if robot_id == 0 else f"_{robot_id}"
    root = _g1_root()
    pelvis = root.find("worldbody").find("body[@name='pelvis']")
    pelvis.set("pos", f"0 0 {PELVIS_Z}")
    keep = set(LEG_JOINTS) | set(ARM_JOINTS)
    ranges = {}
    parent_of = {c: p for p in pelvis.iter() for c in p}

    # joints: delete non-kept, tune kept
    for b in list(pelvis.iter("body")):
        for j in list(b.findall("joint")) + list(b.findall("freejoint")):
            if j.tag == "freejoint" or j.get("name") not in keep:
                b.remove(j)
                continue
            n = j.get("name")
            j.attrib.pop("actuatorfrcrange", None)
            if n in ARM_JOINTS:
                j.set("axis", "0 -1 0")
                j.set("range", "-0.5 3.0" if "shoulder" in n else "-1.0 2.4")
                j.set("damping", "3")
            else:
                ranges[n + sfx] = j.get("range")

    # forearm hangs down at q=0: rotate the wrist_roll frame +90 deg about y
    for side in ("left", "right"):
        wr = pelvis.find(f".//body[@name='{side}_wrist_roll_link']")
        p = [float(x) for x in wr.get("pos").split()]
        wr.set("pos", _f(*_rot_y(math.pi / 2, p)))
        wr.set("quat", "0.7071068 0 0.7071068 0")

    # collision geoms out, everything else contype 0
    for b in pelvis.iter("body"):
        for g in list(b.findall("geom")):
            if g.get("class") == "collision":
                b.remove(g)
    for g in pelvis.iter("geom"):
        g.set("contype", "0")
        g.set("conaffinity", "0")

    # torso: head meshes move to the head body (so the head rangefinder does not see them)
    torso = pelvis.find(".//body[@name='torso_link']")
    head = ET.Element("body", name="head", pos=_f(*HEAD_POS))
    for g in list(torso.findall("geom")):
        if g.get("mesh") in ("head_link", "logo_link"):
            torso.remove(g)
            gp = [float(x) for x in g.get("pos", "0 0 0").split()]
            g.set("pos", _f(*[gp[i] - HEAD_POS[i] for i in range(3)]))
            head.append(g)
    ET.SubElement(head, "geom", name="head_g", type="sphere", size="0.1")
    _hidden(head.find("geom[@name='head_g']"))
    ET.SubElement(head, "camera", name="head_cam", pos="0.09 0 0.01", xyaxes="0 -1 0 0 0 1", fovy="70")
    ET.SubElement(head, "site", name="rf_front_s", pos="0.105 0 0", zaxis="1 0 0", size="0.01")
    ET.SubElement(head, "site", name="rf_left_s", pos="0 0.105 0", zaxis="0 1 0", size="0.01")
    ET.SubElement(head, "site", name="rf_right_s", pos="0 -0.105 0", zaxis="0 -1 0", size="0.01")
    torso.append(head)
    wg = ET.SubElement(torso, "geom", name="waist_g", type="capsule", fromto="0 0 -0.35 0 0 0.02", size="0.09")
    tg = ET.SubElement(torso, "geom", name="torso_g", type="box", pos="0.004 0 0.14", size="0.10 0.17 0.14")
    _hidden(wg)
    _hidden(tg)

    # hands
    for side, s in (("right", "r"), ("left", "l")):
        wy = pelvis.find(f".//body[@name='{side}_wrist_yaw_link']")
        hb = ET.SubElement(wy, "body", name=f"{s}_hand_b", pos=_f(*HAND_POS), quat=HAND_QUAT)
        ET.SubElement(hb, "geom", name=f"{s}_hand_g", type="sphere", pos="0 0 -0.05", size="0.045", mass="0.15",
                      rgba="0.2 0.22 0.27 1", contype="32", conaffinity="8")
        ET.SubElement(hb, "site", name=f"{s}_hand", pos="0 0 -0.06", size="0.02", rgba="1 0.3 0.1 1")
        ET.SubElement(hb, "site", name=f"{s}_wrist", pos="0 0 0", size="0.02", rgba="0.2 1 0.3 0.5")

    # rename + suffix
    rename = {"torso_link": "torso", **ARM_JOINTS}
    _sub(pelvis, sfx, rename)

    robot = ET.Element("body", name=f"robot{sfx}", pos=f"{_f(xy[0], xy[1])} 0")
    if yaw != 0:
        robot.set("euler", f"0 0 {_f(yaw)}")
    ET.SubElement(robot, "joint", name=f"base_x{sfx}", type="slide", axis="1 0 0", ref=f"{xy[0]:.4f}", damping="5")
    ET.SubElement(robot, "joint", name=f"base_y{sfx}", type="slide", axis="0 1 0", ref=f"{xy[1]:.4f}", damping="5")
    ET.SubElement(robot, "joint", name=f"base_yaw{sfx}", type="hinge", axis="0 0 1", damping="5")
    ET.SubElement(robot, "inertial", pos="0 0 0.3", mass="5", diaginertia="0.3 0.3 0.3")
    bg = ET.SubElement(robot, "geom", name=f"base_g{sfx}", type="box", pos="0 0 0.5", size="0.12 0.16 0.22")
    _hidden(bg)
    ET.SubElement(robot, "site", name=f"imu{sfx}", pos="0 0 0.2", size="0.02")
    ET.SubElement(robot, "camera", name=f"cam_follow{sfx}", mode="trackcom", pos="-2.8 -3.6 3.0",
                  xyaxes=_xyaxes((-2.8, -3.6, 3.0), (0, 0, 0.8)), fovy="55")
    robot.append(pelvis)
    # head body/sites/camera got renamed by _sub (they were inside pelvis) - good.

    # actuators / sensors
    S = sfx
    act = [
        f'<velocity name="act_vx{S}" joint="base_x{S}" kv="900" ctrlrange="-2 2" forcerange="-450 450"/>',
        f'<velocity name="act_vy{S}" joint="base_y{S}" kv="900" ctrlrange="-2 2" forcerange="-450 450"/>',
        f'<velocity name="act_wz{S}" joint="base_yaw{S}" kv="1500" ctrlrange="-3 3" forcerange="-500 500"/>',
    ]
    for s in ("r", "l"):
        act.append(f'<position name="act_{s}_shoulder{S}" joint="{s}_shoulder{S}" kp="120" ctrlrange="-0.5 3.0" forcerange="-40 40"/>')
        act.append(f'<position name="act_{s}_elbow{S}" joint="{s}_elbow{S}" kp="90" ctrlrange="-1.0 2.4" forcerange="-30 30"/>')
    for j in LEG_JOINTS:
        act.append(f'<position name="act_{j}{S}" joint="{j}{S}" kp="{KP_LEG}" kv="{KV_LEG}" ctrlrange="{ranges[j + S]}"/>')
    sens = [
        f'<rangefinder name="rf_front{S}" site="rf_front_s{S}" cutoff="8"/>',
        f'<rangefinder name="rf_left{S}" site="rf_left_s{S}" cutoff="8"/>',
        f'<rangefinder name="rf_right{S}" site="rf_right_s{S}" cutoff="8"/>',
        f'<touch name="touch_r{S}" site="r_hand{S}"/>',
        f'<force name="f_wrist{S}" site="r_wrist{S}"/>',
        f'<accelerometer name="imu_acc{S}" site="imu{S}"/>',
        f'<gyro name="imu_gyro{S}" site="imu{S}"/>',
    ] + [f'<jointpos name="jp_{s}_{k}{S}" joint="{s}_{k}{S}"/>' for s in ("r", "l") for k in ("shoulder", "elbow")]

    assets = {
        "meshdir": str(G1_DIR / "assets").replace("\\", "/"),
        "default_xml": ET.tostring(root.find("default"), encoding="unicode"),
        "asset_xml": _inner(root.find("asset")),
        "actuator_xml": "\n    ".join(act),
        "sensor_xml": "\n    ".join(sens),
    }
    return ET.tostring(robot, encoding="unicode"), assets, [j + S for j in LEG_JOINTS]


def _standalone(robot_ids, spacing, frags) -> str:
    a = frags[0][1]
    body = "\n".join(f[0] for f in frags)
    act = "\n    ".join(f[1]["actuator_xml"] for f in frags)
    sens = "\n    ".join(f[1]["sensor_xml"] for f in frags)
    return f"""<mujoco model="g1_standalone">
  <compiler angle="radian" autolimits="true" meshdir="{a['meshdir']}"/>
  <option timestep="0.002" integrator="implicitfast" gravity="0 0 -9.81"/>
  <visual><global offwidth="1280" offheight="960"/><quality shadowsize="2048"/>
    <headlight ambient="0.45 0.45 0.45" diffuse="0.5 0.5 0.5"/></visual>
  {a['default_xml']}
  <asset>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.80 0.88 0.97" rgb2="0.93 0.95 0.98" width="64" height="64"/>
    <texture name="floortex" type="2d" builtin="checker" rgb1="0.90 0.91 0.93" rgb2="0.82 0.84 0.87" width="256" height="256"/>
    <material name="floor" texture="floortex" texrepeat="12 8" reflectance="0.0"/>
    {a['asset_xml']}
  </asset>
  <worldbody>
    <light pos="0 0 6" dir="0 0.2 -1" diffuse="0.6 0.6 0.6" castshadow="false"/>
    <light pos="-3 -4 5" dir="0.4 0.5 -1" diffuse="0.35 0.35 0.35"/>
    <geom name="floor" type="plane" size="8 8 0.1" material="floor" friction="1 0.005 0.0001" contype="1" conaffinity="10"/>
    {body}
  </worldbody>
  <actuator>
    {act}
  </actuator>
  <sensor>
    {sens}
  </sensor>
</mujoco>
"""


@functools.lru_cache(maxsize=1)
def _measure_arm() -> dict:
    """Arm geometry at q=0, MEASURED by compiling a standalone model (never hand-typed)."""
    frag = _build(0, (0.0, 0.0), 0.0)
    m = mujoco.MjModel.from_xml_string(_standalone((0,), 1.5, [frag]))
    d = mujoco.MjData(m)
    mujoco.mj_kinematics(m, d)
    sh = d.xanchor[m.joint("r_shoulder").id].copy()
    el = d.xanchor[m.joint("r_elbow").id].copy()
    hd = d.site("r_hand").xpos.copy()
    up, fo = el - sh, hd - el
    a1 = math.atan2(up[0], -up[2])
    a2 = math.atan2(fo[0], -fo[2]) - a1
    return dict(shoulder_xyz=[round(float(x), 5) for x in sh], l1=float(np.linalg.norm(up)),
                l2=float(np.linalg.norm(fo)), a1_0=float(a1), a2_0=float(a2))


def build_robot_xml(robot_id: int = 0, xy=(0, 0), yaw: float = 0.0):
    """-> (xml_fragment, meta, assets).  See module docstring for the contract."""
    frag, assets, legs = _build(robot_id, xy, yaw)
    sfx = "" if robot_id == 0 else f"_{robot_id}"
    meta = dict(robot_body="g1", robot_id=robot_id, pelvis_z=PELVIS_Z, leg_joints=legs,
                leg_acts=["act_" + j for j in legs], arm=copy.deepcopy(_measure_arm()), sfx=sfx)
    return frag, meta, assets


def build_standalone_xml(robot_ids=(0,), spacing: float = 1.5) -> str:
    frags = [_build(i, (0.0, i * spacing), 0.0) for i in robot_ids]
    return _standalone(robot_ids, spacing, frags)


def foot_min_z() -> float:
    """Lowest foot-sphere bottom in the stop pose (all joints 0)."""
    m = mujoco.MjModel.from_xml_string(build_standalone_xml((0,)))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    zs = []
    for g in range(m.ngeom):
        bn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) or ""
        if "ankle_roll" in bn and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE:
            zs.append(d.geom_xpos[g][2] - m.geom_size[g][0])
    return float(min(zs))
