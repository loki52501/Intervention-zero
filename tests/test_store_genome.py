import threading
import time
import uuid
from datetime import datetime, timezone

import pytest

from iz.store.db import ACTIVE, DECISIONS, GENOMES, RUNS, get_db, mongo_available
from iz.store.genome_store import GenomeStore, ValidatorRefusal
from iz.types import Decision, GateResult, Genome, Prediction, SetScore

pytestmark = pytest.mark.skipif(not mongo_available(), reason="MongoDB not available")


@pytest.fixture(scope="module")
def db():
    from scripts.setup_atlas import setup
    d = get_db("iz_test")
    setup(d, wait_search=False)
    return d


@pytest.fixture()
def store(db):
    orig = db[ACTIVE].find_one({"_id": "robot_genome"})
    yield GenomeStore(db)
    db[ACTIVE].replace_one({"_id": "robot_genome"}, orig, upsert=True)


def _decision(did=None, parent=0, new=None):
    s = SetScore(n=1, takeover_free=1, safety_violations=0)
    gate = GateResult(candidate_id="c", parent_version=parent, train_before=s, train_after=s,
                      heldout_before=s, heldout_after=s, regression_after=s, regressions=0,
                      safety_violations=0, heldout_gain=0, verdict="PROMOTE", reason="ok")
    p = Prediction(train_fixed=0, heldout_fixed=0)
    return Decision(id=did or f"d-{uuid.uuid4().hex}", run_id="r", candidate_id="c", parent_version=parent,
                    new_version=new, verdict="PROMOTE", reason="ok", gate=gate, predicted=p, actual=p,
                    ts=datetime.now(timezone.utc))


def test_promote_and_lineage(db, store):
    p = store.next_version()
    v1 = Genome(version=p, parent=0 if p else None)
    store.promote(v1, _decision(new=p, parent=0))
    assert store.get_active().version == p
    v2 = store.next_version()
    dec = _decision(new=v2, parent=p)
    store.promote(Genome(version=v2, parent=p), dec)
    assert store.get_active().version == v2
    lin = store.lineage(v2)
    assert lin[:2] == [v2, p]
    assert db[DECISIONS].count_documents({"_id": dec.id}) == 1


def test_validator_refusal_leaves_state(db, store):
    v = store.next_version()
    bad = Genome(version=v).model_dump(mode="json")
    bad["guardrails"]["locked"]["battery_floor_pct"] = 5
    before = db[ACTIVE].find_one({"_id": "robot_genome"})["version"]
    dec = _decision(new=v)
    with pytest.raises(ValidatorRefusal) as ei:
        store.promote(bad, dec)
    assert ei.value.message
    assert db[ACTIVE].find_one({"_id": "robot_genome"})["version"] == before
    assert db[GENOMES].count_documents({"_id": v}) == 0
    assert db[DECISIONS].count_documents({"_id": dec.id}) == 0


def test_atomicity(db, store):
    v = store.next_version()
    dec = _decision(new=v)
    db[DECISIONS].insert_one({**dec.model_dump(mode="python"), "_id": dec.id})
    before = db[ACTIVE].find_one({"_id": "robot_genome"})["version"]
    with pytest.raises(Exception):
        store.promote(Genome(version=v, parent=before), dec)
    assert db[ACTIVE].find_one({"_id": "robot_genome"})["version"] == before
    assert db[GENOMES].count_documents({"_id": v}) == 0


def test_watch_active(db, store):
    got = []
    stop = threading.Event()
    name = "t-" + uuid.uuid4().hex
    t = threading.Thread(target=store.watch_active, args=(lambda k, v: got.append((k, v)), stop, name))
    t.start()
    try:
        time.sleep(1.5)
        v = store.next_version()
        store.promote(Genome(version=v, parent=0), _decision(new=v))
        deadline = time.time() + 10
        while time.time() < deadline and ("robot_genome", v) not in got:
            time.sleep(0.2)
        assert ("robot_genome", v) in got
        time.sleep(0.8)
    finally:
        stop.set()
        t.join(10)
    assert db[RUNS].find_one({"_id": f"watch:{name}"})["token"]
    db[RUNS].delete_one({"_id": f"watch:{name}"})
