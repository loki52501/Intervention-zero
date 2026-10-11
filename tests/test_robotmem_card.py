import pytest

from iz.robotmem.card import _ntok, card_text, robot_card, store_card

INLINE = """<mujoco><worldbody><body name="robot"><joint name="base_x" type="slide" axis="1 0 0"/>
<geom name="base_g" type="box" size="0.2 0.2 0.1" mass="5"/><site name="imu" size="0.02"/>
<site name="rf_s" size="0.01" zaxis="1 0 0"/></body></worldbody>
<actuator><velocity name="act_vx" joint="base_x" kv="100" ctrlrange="-2 2" forcerange="-400 400"/></actuator>
<sensor><rangefinder name="rf_front" site="rf_s" cutoff="8"/><gyro name="g" site="imu"/></sensor></mujoco>"""


def _xml():
    try:
        from iz.shop.scene import build_shop
        return build_shop(0)[0], True
    except Exception:
        return INLINE, False


def test_card_shape():
    xml, real = _xml()
    c = robot_card(xml)
    assert c["name"] == "IZ-1 retail humanoid" and c["hash"]
    assert c["actuators"] and c["sensors"]
    assert "get_cart" in c["skills"] and "cruise_speed" in c["tunable_params"]
    assert "min_person_clearance_m" in c["locked"]
    if real:
        assert len(c["actuators"]) == 7
        assert any(s["type"] == "rangefinder" and s["site"] == "rf_front_s" for s in c["sensors"])
        a = next(a for a in c["actuators"] if a["name"] == "act_vx")
        assert a["limit"] == [-2.0, 2.0] and a["kind"] == "velocity"
        assert next(a for a in c["actuators"] if a["name"] == "act_r_elbow")["kind"] == "position"


def test_hash_stable_and_text_budget():
    xml, _ = _xml()
    c = robot_card(xml)
    assert robot_card(xml)["hash"] == c["hash"]
    for b in (350, 200, 120):
        assert _ntok(card_text(c, b)) <= b
    assert "LOCKED" in card_text(c, 350)


def test_default_xml():
    pytest.importorskip("iz.shop.scene")
    assert robot_card()["name"].startswith("IZ-1")


def test_store_upsert_fake():
    class Col:
        def __init__(self):
            self.calls = []

        def update_one(self, f, u, upsert=False):
            self.calls.append((f, upsert))

    class DB(dict):
        def __missing__(self, k):
            self[k] = Col()
            return self[k]

    db = DB()
    c = robot_card(_xml()[0])
    r = store_card(c, db)
    assert db["robot_cards"].calls == [({"name": c["name"], "hash": c["hash"]}, True)]
    assert r["hash"] == c["hash"]
