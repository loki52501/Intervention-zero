import json
import sys
import types

import pytest

from iz.llm.client import FakeLLM
from iz.shop import learn


@pytest.fixture(autouse=True)
def fake_controller(monkeypatch):
    c = types.SimpleNamespace(
        DEFAULT_PARAMS={"cruise_speed": 1.4, "turn_speed": 1.3, "spill_speed": 1.4},
        PARAM_BOUNDS={"cruise_speed": (0.3, 1.6), "turn_speed": (0.3, 1.5), "spill_speed": (0.2, 1.5)},
        LOCKED={"min_person_clearance_m": 0.35, "max_wrist_force_n": 300})
    monkeypatch.setitem(sys.modules, "iz.shop.controller", c)
    import iz.shop
    monkeypatch.setattr(iz.shop, "controller", c, raising=False)
    monkeypatch.setattr(learn, "_atlas_collection", lambda: None)


def test_propose_clamps_and_refuses_locked():
    llm = FakeLLM(responses=[{"edits": {"cruise_speed": 9.0, "turn_speed": 0.0,
                                        "min_person_clearance_m": 0.0, "max_wrist_force_n": 999,
                                        "bogus": 1},
                              "rationale": "r", "prediction": {"train_fixed": 2, "heldout_fixed": 1}}])
    out = learn.propose(llm, {"cruise_speed": 1.4}, [{"seed": 0, "failure": "cart_tipped"}], None)
    assert out["edits"] == {"cruise_speed": 1.6, "turn_speed": 0.3}
    assert sum("LOCKED" in r for r in out["refused"]) == 2
    assert out["prediction"] == {"train_fixed": 2, "heldout_fixed": 1}


def _fake_eval(rule):
    def ev(params, seeds):
        res = []
        for s in seeds:
            st = rule(params, s)
            res.append({"seed": s, "success": st == "success", "status": st,
                        "failure": None if st == "success" else "cart_tipped",
                        "operator_tip": "tip", "summary": {}})
        return res
    return ev


def _gate(monkeypatch, rule):
    monkeypatch.setattr(learn, "evaluate", _fake_eval(rule))
    return learn.gate({"cruise_speed": 1.4}, {"cruise_speed": 0.8}, range(4), range(100, 104), range(200, 203))


def _base(p, s):
    return "success" if p["cruise_speed"] < 1 or s % 2 else "takeover"


def test_gate_promotes(monkeypatch):
    g = _gate(monkeypatch, _base)
    assert g["verdict"] == "PROMOTE" and g["heldout"]["after"] > g["heldout"]["before"]
    assert g["safety_violations"] == 0 and g["regressions"] == 0


def test_gate_rejects_safety(monkeypatch):
    g = _gate(monkeypatch, lambda p, s: "safety_violation" if p["cruise_speed"] < 1 and s == 201 else _base(p, s))
    assert g["verdict"] == "REJECT" and "safety" in g["reason"]


def test_gate_rejects_regression(monkeypatch):
    g = _gate(monkeypatch, lambda p, s: "takeover" if p["cruise_speed"] < 1 and s == 1 else _base(p, s))
    assert g["verdict"] == "REJECT" and "regression" in g["reason"]


def test_gate_rejects_no_gain(monkeypatch):
    g = _gate(monkeypatch, lambda p, s: "success" if s % 2 else "takeover")
    assert g["verdict"] == "REJECT" and "no held-out gain" in g["reason"]


def test_run_learning_history_and_events(monkeypatch, tmp_path):
    monkeypatch.setattr(learn, "evaluate", _fake_eval(_base))
    llm = FakeLLM(responses=[{"edits": {"cruise_speed": 0.8}, "rationale": "slow",
                              "prediction": {"train_fixed": 2, "heldout_fixed": 2}},
                             {"edits": {"cruise_speed": 0.7}, "rationale": "again",
                              "prediction": {"train_fixed": 0, "heldout_fixed": 0}}])
    hist = learn.run_learning(generations=2, llm=llm, out_dir=str(tmp_path))
    assert [h["gen"] for h in hist] == [0, 1, 2]
    assert hist[1]["verdict"] == "PROMOTE" and hist[1]["params"]["cruise_speed"] == 0.8
    assert hist[1]["heldout_rate"] > hist[0]["heldout_rate"]
    assert hist[2]["verdict"] == "REJECT"
    assert hist[2]["params"]["cruise_speed"] == 0.8
    assert json.loads((tmp_path / "shop_history.json").read_text())[1]["gen"] == 1
    evs = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert {"takeover", "fix_proposed", "gate_result", "promoted", "metrics"} <= {e["kind"] for e in evs}
    assert all(e["payload"]["world"] == "shop" and {"id", "ts", "kind", "run_id", "payload"} <= set(e) for e in evs)
