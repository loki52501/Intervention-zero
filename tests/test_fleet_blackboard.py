import os
os.environ["RAINDROP_ENABLED"] = "0"
import threading
import uuid

import pytest

from iz.store.blackboard import MemoryBlackboard, AtlasBlackboard, get_blackboard


def _race(bb, world, task):
    res = {}
    bar = threading.Barrier(2)

    def go(r):
        bar.wait()
        res[r] = bb.claim_task(world, task, r)

    ts = [threading.Thread(target=go, args=(r,)) for r in ("robot_1", "robot_2")]
    [t.start() for t in ts]
    [t.join() for t in ts]
    return res


def test_memory_publish_query():
    bb = MemoryBlackboard()
    bb.publish("w1", "hazard", 1.0, 2.0, 0.55, "robot", ttl=60, sim_t=3.0)
    got = bb.query_near("w1", (1.5, 2.0), 1.0)
    assert len(got) == 1
    for k in ("world", "kind", "x", "y", "radius", "robot_id", "ts", "ttl"):
        assert k in got[0]
    assert bb.query_near("w1", (9, 9), 1.0) == []
    assert bb.query_near("w2", (1, 2), 1.0) == []
    assert bb.query_near("w1", (1.5, 2.0), 1.0, kinds=["blocked_lane"]) == []


def test_memory_ttl_expiry():
    bb = MemoryBlackboard()
    bb.publish("w1", "hazard", 1.0, 2.0, 0.5, "r", ttl=5, sim_t=0.0)
    assert len(bb.query_near("w1", (1, 2), 1.0, now_sim=4.0)) == 1
    assert bb.query_near("w1", (1, 2), 1.0, now_sim=6.0) == []


def test_claim_single_winner_threads():
    bb = MemoryBlackboard()
    res = _race(bb, "w1", "aisle_0_sweep")
    assert sum(res.values()) == 1
    winner = [r for r, v in res.items() if v][0]
    loser = [r for r, v in res.items() if not v][0]
    assert bb.claim_task("w1", "aisle_0_sweep", winner) is True
    assert bb.claim_task("w1", "aisle_0_sweep", loser) is False
    assert bb.task_owner("w1", "aisle_0_sweep") == winner


def test_bad_kind_raises():
    with pytest.raises(ValueError):
        MemoryBlackboard().publish("w1", "banana", 0, 0, 1, "r")


def test_factory_offline(monkeypatch):
    monkeypatch.delenv("MONGODB_URI", raising=False)
    assert isinstance(get_blackboard(), MemoryBlackboard)


def _live():
    try:
        from iz.store.db import mongo_available
        return mongo_available()
    except Exception:
        return False


@pytest.mark.skipif(not _live(), reason="Atlas not available")
def test_live_atlas_claim():
    bb = AtlasBlackboard(db_name="iz_test")
    world = f"test_world_{uuid.uuid4().hex[:8]}"
    bb.clear(world)
    try:
        res = _race(bb, world, "aisle_0_sweep")
        assert sum(res.values()) == 1
        bb.publish(world, "hazard", 1.0, 2.0, 0.5, "r", ttl=60)
        assert len(bb.query_near(world, (1.2, 2.0), 1.0, kinds=["hazard"])) == 1
    finally:
        bb.clear(world)


def test_schema_appended():
    from iz.store.schemas import VALIDATORS, FLEET_BLACKBOARD
    assert FLEET_BLACKBOARD == "fleet_blackboard"
    v = VALIDATORS[FLEET_BLACKBOARD]
    assert "$jsonSchema" in v
    s = v["$jsonSchema"]
    for k in ("world", "kind", "x", "y", "radius", "robot_id", "ts", "ttl"):
        assert k in s["required"]
    assert s["properties"]["kind"]["enum"] == ["hazard", "blocked_lane", "task_claim"]
