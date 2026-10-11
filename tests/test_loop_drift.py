import random
from datetime import datetime

from iz.loop import consolidate, drift
from iz.types import Genome, OutcomeStatus, RuleEntry, Scope, Takeover


def _series(seed, segs):
    rng = random.Random(seed)
    return [1 if rng.random() < p else 0 for p, n in segs for _ in range(n)]


def test_drift_fires_on_drop():
    h = _series(1, [(0.85, 60), (0.55, 30)])
    r = drift.check(h, "A")
    assert r and r["alarm"] and r["window_tfmr"] < r["baseline_tfmr"]
    assert drift.detection_latency(h, 60) is not None


def test_drift_quiet_when_stable():
    h = _series(2, [(0.85, 90)])
    r = drift.check(h, "A")
    assert r and not r["alarm"]


def test_z_direction():
    z, p = drift.two_proportion_z(10, 30, 50, 60)
    assert z < 0 and p < 0.01


def _tk(i, day, site="A", typ="fire_door"):
    return Takeover(id=i, run_id="r", mission_id="m", robot_id="r1", site=site, day=day, shift=0,
                    situation_id="s", situation_type=typ, zone="z", zone_class="ward", observable="o", tried=[], outcome=OutcomeStatus.TAKEOVER,
                    reason="x", operator_tip="t", policy_version=1, ts=datetime.now())


def test_consolidate_merges_and_retires():
    sc = Scope(site="A", situation_type="fire_door")
    g = Genome(version=3, rulebook=[
        RuleEntry(id="r1", scope=sc, text="Wait for the fire door to open then proceed", provenance=["t1"]),
        RuleEntry(id="r2", scope=sc, text="Wait for the fire door to open then proceed.", provenance=["t2"]),
        RuleEntry(id="r3", scope=Scope(site="B", situation_type="spill"), text="clean spill slowly",
                  provenance=["t0"]),
    ])
    new, summary = consolidate.run(g, [_tk("t1", 8), _tk("t2", 8), _tk("t0", 1, "B", "spill")], day=9)
    assert new and new.version == 4 and new.parent == 3
    st = {r.id: r.status for r in new.rulebook}
    assert "r2" not in st and st["r1"] == "active" and st["r3"] == "retired"
    assert set(new.rulebook[0].provenance) == {"t1", "t2"}
    assert "merged 1" in summary


def test_consolidate_noop():
    assert consolidate.run(Genome(version=1), [], 3)[0] is None
