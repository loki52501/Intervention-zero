import pytest

from iz.types import Action, RobotView
from iz.world.naive import NAIVE_TABLE, naive_actions

TYPES = ["fire_door", "office_door", "cart_blocking", "crowd_blocking", "spill",
         "restricted_shortcut", "elevator_out_of_service", "person_down",
         "low_battery", "construction_detour"]


def test_naive_covers_all_types():
    assert set(NAIVE_TABLE) == set(TYPES)
    for t in TYPES:
        v = RobotView(situation_id="x", site="A", type=t, zone="Z", zone_class="corridor", observable="o")
        acts = naive_actions(v)
        assert 1 <= len(acts) <= 3 and all(isinstance(a, Action) for a in acts)


def test_run_baseline_rule():
    pytest.importorskip("iz.world.generator")
    pytest.importorskip("iz.world.stats")
    from scripts.baseline import run_baseline
    r = run_baseline(judge_kind="rule", n=3)
    lo, hi = r["ci"]
    assert 0.0 <= r["tfmr"] <= 1.0 and lo <= r["tfmr"] <= hi
    assert r == run_baseline(judge_kind="rule", n=3)
