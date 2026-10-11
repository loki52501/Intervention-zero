import math

import mujoco
import numpy as np
import pytest

from iz.shop.controller import (DEFAULT_PARAMS, LOCKED, PARAM_BOUNDS, ShopController)


def fixture_xml(wall_x=None, shopper_xy=(8.0, -4.0)):
    wall = ""
    if wall_x is not None:
        wall = f'<geom name="shelf_1" type="box" pos="{wall_x} 0 0.5" size="0.1 1.5 0.5"/>'
    return f"""
<mujoco>
  <compiler angle="radian"/>
  <option timestep="0.002" integrator="implicitfast"/>
  <worldbody>
    <geom name="floor" type="plane" size="20 20 0.1"/>
    <geom name="shelf_0" type="box" pos="4 2.0 0.45" size="1 0.2 0.45"/>
    {wall}
    <body name="prod_0" pos="3.5 2.0 0.96"><freejoint/>
      <geom name="prod_0_g" type="box" size="0.05 0.05 0.05" mass="0.2"/></body>
    <body name="shopper_0" mocap="true" pos="{shopper_xy[0]} {shopper_xy[1]} 0.85">
      <geom name="shopper_0_g" type="capsule" size="0.2 0.6" contype="0" conaffinity="0"/></body>
    <body name="robot" pos="0 0 0.3">
      <joint name="base_x" type="slide" axis="1 0 0" damping="50"/>
      <joint name="base_y" type="slide" axis="0 1 0" damping="50"/>
      <joint name="base_yaw" type="hinge" axis="0 0 1" damping="20"/>
      <geom name="base_g" type="box" size="0.2 0.25 0.1" mass="20" contype="0" conaffinity="0"/>
      <site name="rf_f" pos="0.25 0 0.1" zaxis="1 0 0"/>
      <site name="rf_l" pos="0 0.26 0.1" zaxis="0 1 0"/>
      <site name="rf_r" pos="0 -0.26 0.1" zaxis="0 -1 0"/>
      <site name="imu" pos="0 0 0"/>
      <body name="torso" pos="0 0 0.4">
        <geom type="box" size="0.12 0.2 0.3" mass="5" contype="0" conaffinity="0"/>
        <body name="r_upper" pos="0 -0.25 0.3">
          <joint name="r_shoulder" type="hinge" axis="0 1 0" range="-2.5 2.5"/>
          <geom type="capsule" fromto="0 0 0 0 0 -0.4" size="0.04" mass="1" contype="0" conaffinity="0"/>
          <body name="r_fore" pos="0 0 -0.4">
            <joint name="r_elbow" type="hinge" axis="0 1 0" range="-2.5 2.5"/>
            <geom type="capsule" fromto="0 0 0 0 0 -0.4" size="0.035" mass="0.8" contype="0" conaffinity="0"/>
            <body name="r_hand_b" pos="0 0 -0.4">
              <geom type="sphere" size="0.04" mass="0.2" contype="0" conaffinity="0"/>
              <site name="r_hand" pos="0 0 0"/>
              <site name="touch_site" pos="0 0 0" size="0.06"/>
              <site name="wrist" pos="0 0 0"/>
            </body>
          </body>
        </body>
        <body name="l_upper" pos="0 0.25 0.3">
          <joint name="l_shoulder" type="hinge" axis="0 1 0" range="-2.5 2.5"/>
          <geom type="capsule" fromto="0 0 0 0 0 -0.4" size="0.04" mass="1" contype="0" conaffinity="0"/>
          <body name="l_fore" pos="0 0 -0.4">
            <joint name="l_elbow" type="hinge" axis="0 1 0" range="-2.5 2.5"/>
            <geom type="capsule" fromto="0 0 0 0 0 -0.4" size="0.035" mass="0.8" contype="0" conaffinity="0"/>
          </body>
        </body>
      </body>
    </body>
    <body name="cart" pos="2.0 0 0.3"><freejoint name="cart_free"/>
      <geom name="cart_basket" type="box" size="0.3 0.25 0.25" mass="15" friction="0.05 0.005 0.0001"/>
      <site name="cart_handle" pos="-0.3 0 0.2"/>
      <site name="cart_front" pos="0.31 0 0" zaxis="1 0 0"/>
    </body>
  </worldbody>
  <equality>
    <weld name="grip_cart" body1="robot" body2="cart" active="false" solref="0.02 1"/>
    <weld name="grip_prod" body1="r_hand_b" body2="prod_0" active="false" solref="0.02 1"/>
  </equality>
  <actuator>
    <velocity name="act_vx" joint="base_x" kv="400" forcerange="-300 300"/>
    <velocity name="act_vy" joint="base_y" kv="400" forcerange="-300 300"/>
    <velocity name="act_wz" joint="base_yaw" kv="100" forcerange="-100 100"/>
    <position name="act_r_shoulder" joint="r_shoulder" kp="100" forcerange="-50 50"/>
    <position name="act_r_elbow" joint="r_elbow" kp="100" forcerange="-50 50"/>
    <position name="act_l_shoulder" joint="l_shoulder" kp="100" forcerange="-50 50"/>
    <position name="act_l_elbow" joint="l_elbow" kp="100" forcerange="-50 50"/>
  </actuator>
  <sensor>
    <rangefinder name="rf_front" site="rf_f"/>
    <rangefinder name="rf_left" site="rf_l"/>
    <rangefinder name="rf_right" site="rf_r"/>
    <rangefinder name="rf_cart" site="cart_front"/>
    <touch name="touch_r" site="touch_site"/>
    <force name="f_wrist" site="wrist"/>
    <accelerometer name="imu_acc" site="imu"/>
    <gyro name="imu_gyro" site="imu"/>
    <framezaxis name="cart_up" objtype="body" objname="cart"/>
    <framepos name="cart_pos" objtype="body" objname="cart"/>
    <jointpos name="jp_x" joint="base_x"/>
    <jointpos name="jp_y" joint="base_y"/>
    <jointpos name="jp_yaw" joint="base_yaw"/>
    <jointpos name="jp_rs" joint="r_shoulder"/>
    <jointpos name="jp_re" joint="r_elbow"/>
  </sensor>
</mujoco>"""


META = {
    "target_product": "prod_0",
    "grip_body": "robot",
    "checkout_xy": [7.0, 0.0],
    "cart_xy": [2.0, 0.0],
    "robot_xy": [0.0, 0.0],
    "waypoints": [[0.0, 0.0], [3.5, 0.0], [7.0, 0.0]],
    "route": [0, 1, 2],
    "spills": [],
}


def make(params=None, **kw):
    model = mujoco.MjModel.from_xml_string(fixture_xml(**kw))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data, ShopController(model, data, dict(META), params)


def run(model, data, c, seconds, stop=None):
    seen, log = [], []
    while data.time < seconds:
        c.step(data.time)
        mujoco.mj_step(model, data)
        if not seen or seen[-1] != c.state:
            seen.append(c.state)
        log.append((data.time, c.state, c.speed_cmd, c.rf("rf_cart"), c.yielding))
        if stop and stop(c):
            break
    return seen, log


def test_sensor_helpers_finite():
    model, data, c = make()
    c.step(0.0)
    mujoco.mj_step(model, data)
    for n in ("rf_front", "rf_left", "rf_right", "rf_cart"):
        assert math.isfinite(c.rf(n)) and 0 < c.rf(n) <= 10.0
    assert math.isfinite(c.wrist_force()) and math.isfinite(c.touch())
    assert 0.0 <= c.cart_tilt_deg() < 5.0
    assert np.all(np.isfinite(c.cart_pos()))
    assert np.all(np.isfinite(c.read("imu_acc"))) and np.all(np.isfinite(c.read("imu_gyro")))
    assert math.isfinite(c.min_person_dist())
    assert c.read("nonexistent") is None


def test_state_machine_advances():
    model, data, c = make()
    seen, _ = run(model, data, c, 8.0, stop=lambda c: c.state == "PUSH")
    assert seen[:3] == ["GO_TO_CART", "GRASP_CART", "PUSH"], seen
    assert data.time < 6.0
    assert data.eq_active[0] == 1      # grip_cart welded
    # keeps pushing: cart moves toward the shelf waypoint
    x0 = data.xpos[model.body("cart").id][0]
    run(model, data, c, data.time + 2.0)
    assert data.xpos[model.body("cart").id][0] > x0 + 0.3
    assert len(c.trace) > 20 and {"t", "state", "speed_cmd", "rf_front", "rf_cart",
                                  "cart_tilt_deg", "wrist_force", "min_shopper_dist"} <= set(c.trace[0])


def test_brake_near_obstacle():
    model, data, c = make(wall_x=4.0, params={"brake_distance_m": 0.5})
    _, log = run(model, data, c, 25.0)
    pushing = [r for r in log if r[1] == "PUSH"]
    assert max(r[2] for r in pushing) > 0.5            # it did drive
    closest = min(pushing, key=lambda r: r[3])
    assert closest[3] < 0.7, "never got close to the wall"
    assert closest[2] < 0.15                          # crept to a near-stop at the brake distance
    assert all(r[2] == 0.0 for r in pushing if r[3] < 0.5)   # fully stopped inside brake distance
    assert closest[3] > 0.35                          # never hit it
    print("min rf_cart", min(r[3] for r in pushing), "peak", max(r[2] for r in pushing))


def test_yield_to_shopper():
    model, data, c = make(shopper_xy=(3.0, 0.45))
    _, log = run(model, data, c, 8.0)
    y = [r for r in log if r[1] == "PUSH" and r[4]]
    assert y, "never yielded"
    assert all(r[2] == 0.0 for r in y)
    data2 = None
    # far shopper: no yield
    model, data, c = make(shopper_xy=(8.0, -4.0))
    _, log = run(model, data, c, 6.0)
    assert not any(r[4] for r in log)


def test_locked_and_bounds():
    locked_before = dict(LOCKED)
    model, data, c = make(params={"min_person_clearance_m": 0.0, "max_wrist_force_n": 9999,
                                  "cruise_speed": 99, "push_force_limit_n": -5})
    assert dict(c.locked) == locked_before == {"min_person_clearance_m": 0.35,
                                               "max_wrist_force_n": 300.0}
    assert set(c.ignored_params) == {"min_person_clearance_m", "max_wrist_force_n"}
    assert "min_person_clearance_m" not in c.params
    assert c.params["cruise_speed"] == PARAM_BOUNDS["cruise_speed"][1]
    assert c.params["push_force_limit_n"] == PARAM_BOUNDS["push_force_limit_n"][0]
    with pytest.raises(TypeError):
        LOCKED["min_person_clearance_m"] = 0.0
    # yield enforcement never below locked clearance
    model, data, c = make(params={"yield_distance_m": 0.0}, shopper_xy=(0.3, 0.0))
    c.step(0.0)
    assert c.yielding
    assert set(DEFAULT_PARAMS) == set(PARAM_BOUNDS)


def test_shelf_reach_and_pick():
    model, data, c = make(params={"reach_height_m": 0.9})
    seen, _ = run(model, data, c, 40.0, stop=lambda c: c.state == "DONE")
    print("states", seen, "t", data.time)
    assert "AT_SHELF" in seen
    assert "RE-GRASP" in seen
