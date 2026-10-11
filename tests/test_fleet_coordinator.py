import os
os.environ['RAINDROP_ENABLED'] = '0'
import threading
import numpy as np
import pytest

from iz.fleet.coordinator import Coordinator, ROBOT_RADIUS
from iz.fleet.scene import add_fleet_robots
from iz.shop.episode import point_polyline_dist


class _FakeBB:
    def __init__(self):
        self.docs, self.claims, self.lock = [], {}, threading.Lock()

    def publish(self, world, kind, x, y, radius, robot_id, ttl=60.0, sim_t=None, parent_id=None):
        with self.lock:
            d = dict(_id=f'h{len(self.docs)}', world=world, kind=kind, x=x, y=y, radius=radius,
                     robot_id=robot_id, ttl=ttl, sim_t=sim_t, parent_id=parent_id)
            self.docs.append(d)
            return d

    def query_near(self, world, xy, r, kinds=None, now_sim=None):
        with self.lock:
            return [dict(d) for d in self.docs if d['world'] == world
                    and (kinds is None or d['kind'] in kinds)
                    and np.hypot(d['x'] - xy[0], d['y'] - xy[1]) <= r]

    def claim_task(self, world, task_id, robot_id):
        with self.lock:
            return self.claims.setdefault((world, task_id), robot_id) == robot_id


def _meta(target_aisle=1):
    ays = [-2.0, 0.0, 2.0]
    wps = [[-5.0, 0.0]] + [[-4.7, a] for a in ays] + [[3.6, a] for a in ays] + [[0, 0], [3.6, -2.3]]
    return dict(aisle_ys=ays, waypoints=wps, checkout_robot_xy=[3.6, -2.3], spills=[[0.0, 0.0, 0.55, 0.40]],
                target_aisle=target_aisle, robot_xy=[-5.6, 1.0], store_half=[6.1, 4.1],
                fleet_starts=[[-5.6, 1.0], [-4.7, 0.0], [-4.7, 1.0]])


def _co(ids=('robot', 'robot_1'), meta=None):
    return Coordinator(meta or _meta(), _FakeBB(), 'w', list(ids), {})


def test_hazard_shared_and_reroute_before_1m():
    c = _co()
    c.tick(5.0, poses={'robot': (0.1, 0.0, 0.0), 'robot_1': (-4.7, 0.0, 0.0)},
           vels={'robot': (0.5, 0), 'robot_1': (0.8, 0)}, detections={'robot': [(0.0, 0.0, 0.55)]})
    assert any(e['type'] == 'hazard_shared' and e['robot_id'] == 'robot' for e in c.events)
    rr = [e for e in c.events if e['type'] == 'reroute' and e['robot_id'] == 'robot_1']
    assert len(rr) == 1 and rr[0]['dist_to_hazard_m'] > 1.0
    assert point_polyline_dist((0, 0), c.route('robot_1')) >= 0.55 + 0.3
    # same detection again is deduped
    c.tick(5.1, poses={'robot': (0.1, 0.0, 0.0), 'robot_1': (-4.7, -0.2, 0.0)},
           vels={'robot': (0.5, 0), 'robot_1': (0.8, 0)}, detections={'robot': [(0.05, 0.0, 0.55)]})
    assert c.counts()['hazard_shared'] == 1 and c.counts()['reroute'] == 1


def test_reroute_out_of_range():
    c = _co()
    c.tick(5.0, poses={'robot': (0.1, 0.0, 0.0), 'robot_1': (-7.0, 0.0, 0.0)},
           vels={'robot': (0.5, 0), 'robot_1': (0.8, 0)}, detections={'robot': [(0.0, 0.0, 0.55)]})
    assert any(e['type'] == 'hazard_shared' for e in c.events)
    assert not any(e['type'] == 'reroute' for e in c.events)


def test_right_of_way():
    c = _co(('robot_1', 'robot_2'))
    c.tick(1.0, poses={'robot_1': (0, 0, 0), 'robot_2': (0.6, 0.2, 1.57)},
           vels={'robot_1': (0.8, 0), 'robot_2': (0, 0.8)}, detections={})
    assert c.should_stop('robot_2') and not c.should_stop('robot_1')
    ys = [e for e in c.events if e['type'] == 'yield']
    assert len(ys) == 1 and ys[0]['robot_id'] == 'robot_2'
    c.tick(1.1, poses={'robot_1': (0.05, 0, 0), 'robot_2': (0.6, 0.2, 1.57)},
           vels={'robot_1': (0.8, 0), 'robot_2': (0, 0.8)}, detections={})
    assert len([e for e in c.events if e['type'] == 'yield']) == 1  # same episode
    c2 = _co(('robot_1', 'robot_2'))
    c2.tick(1.0, poses={'robot_1': (0, 0, 0), 'robot_2': (0.7, 0.2, 0)},
            vels={'robot_1': (0.8, 0), 'robot_2': (0.8, 0)}, detections={})
    assert not c2.should_stop('robot_2') and c2.counts()['yield'] == 0


def test_clearance_violation():
    c = _co()
    d = 2 * ROBOT_RADIUS + 0.2
    c.tick(1.0, poses={'robot': (0, 0, 0), 'robot_1': (d, 0, 0)}, vels={}, detections={})
    assert len(c.safety_violations) == 1
    v = c.safety_violations[0]
    assert v['type'] == 'too_close_to_robot' and v['robots'] == ['robot', 'robot_1']
    assert v['clearance_m'] == pytest.approx(0.2, abs=1e-6)
    c.tick(1.1, poses={'robot': (0, 0, 0), 'robot_1': (d, 0, 0)}, vels={}, detections={})
    assert len(c.safety_violations) == 1  # once per pair
    c2 = _co()
    c2.tick(1.0, poses={'robot': (0, 0, 0), 'robot_1': (2 * ROBOT_RADIUS + 0.5, 0, 0)}, vels={}, detections={})
    assert c2.safety_violations == []


def test_task_claim_single_winner():
    c = _co(('robot', 'robot_1', 'robot_2'))
    a = c.assign_tasks(0)
    assert a['robot'] == 1 and a['robot_1'] == 1
    assert a['robot_2'] != a['robot_1'] and a['robot_2'] in (0, 2)
    assert c.counts()['task_claimed'] == 2
    # a second coordinator on the same blackboard cannot re-claim those aisles
    c3 = Coordinator(_meta(), c.bb, 'w', ['robot', 'robot_7', 'robot_8'], {})
    a3 = c3.assign_tasks(0)
    assert a3['robot_7'] == 2 and a3['robot_8'] is None


def test_helper_command_follows_route():
    c = _co()
    c.assign_tasks(0)
    vx, vy, wz = c.helper_command('robot_1', (-4.7, 0.0, 0.0))
    assert np.hypot(vx, vy) == pytest.approx(0.8, abs=1e-6)


def test_add_fleet_robots_compiles():
    import mujoco
    from iz.shop.scene import build_shop
    xml, meta = build_shop(0)
    xml2, meta2 = add_fleet_robots(xml, meta, 3)
    m = mujoco.MjModel.from_xml_string(xml2)
    O = mujoco.mjtObj
    for n in ('robot', 'robot_1', 'robot_2'):
        assert mujoco.mj_name2id(m, O.mjOBJ_BODY, n) >= 0
    for n in ('base_x_1', 'base_y_1', 'base_yaw_1'):
        assert mujoco.mj_name2id(m, O.mjOBJ_JOINT, n) >= 0
    for n in ('act_vx_1', 'act_vy_1', 'act_wz_1'):
        assert mujoco.mj_name2id(m, O.mjOBJ_ACTUATOR, n) >= 0
    assert meta2['fleet_robots'] == ['robot', 'robot_1', 'robot_2']
    assert len(meta2['fleet_starts']) == 3 and all(len(s) == 2 for s in meta2['fleet_starts'])
    mujoco.mj_step(m, mujoco.MjData(m))
    x1, m1 = add_fleet_robots(xml, meta, 1)
    assert x1 == xml
    assert 'fleet_robots' not in m1
