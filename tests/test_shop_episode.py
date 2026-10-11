import os

import numpy as np
import pytest

from iz.shop import episode as ep

_here = os.path.dirname(__file__)
HAVE = all(os.path.exists(os.path.join(_here, "..", "iz", "shop", f)) for f in ("scene.py", "controller.py"))


def test_tilt():
    assert ep.tilt_deg_from_zaxis([0, 0, 1]) == pytest.approx(0)
    assert ep.tilt_deg_from_zaxis([1, 0, 0]) == pytest.approx(90)


def test_polyline():
    assert ep.point_polyline_dist([1, 1], [[0, 0], [2, 0]]) == pytest.approx(1)


def test_shopper_pingpong():
    p = [[0, 0], [4, 0]]
    assert np.allclose(ep.shopper_position(p, 0), [0, 0])
    assert np.allclose(ep.shopper_position(p, 2.5), [2.0, 0])
    assert np.allclose(ep.shopper_position(p, 6.25), [3.0, 0])


def test_classify():
    assert ep.classify_failure("person_contact")[0] == "safety_violation"
    assert ep.classify_failure("cart_tipped")[0] == "takeover"
    assert ep.classify_failure(None)[0] == "success"


def test_tips():
    t = ep.make_tip("cart_tipped", dict(speed_mps=1.4, tilt_deg=41, near_wp=2, yaw_rate=1.5))
    assert "1.4 m/s" in t and "41 deg" in t and "0.6" in t
    assert "0.4 m/s" in ep.make_tip("spill_slide", dict(speed_mps=1.2, slide_m=0.5))
    assert "0.28 m" in ep.make_tip("too_close_to_person", dict(dist_m=0.28, speed_mps=1))
    for f in ("person_contact", "collision_shelf", "dropped_product", "timeout"):
        assert ep.make_tip(f, {})


@pytest.mark.skipif(not HAVE, reason="iz.shop.scene/controller not landed yet")
def test_full_episode_deterministic():
    from iz.shop.controller import DEFAULT_PARAMS
    a = ep.run_episode(DEFAULT_PARAMS, 0)
    b = ep.run_episode(DEFAULT_PARAMS, 0)
    assert a["status"] in ("success", "takeover", "safety_violation")
    for k in ("success", "failure", "t_end", "summary"):
        assert a[k] == b[k]
    assert set(a) >= {"seed", "success", "status", "failure", "t_end", "summary", "operator_tip", "trace"}
