from __future__ import annotations

from datetime import datetime

from iz.loop.coach import Coach
from iz.loop.fixer import SEED_PLAYBOOK, Fixer, cluster_takeovers, placebo, random_edit
from iz.types import Genome, LLMResponse, OutcomeStatus, Takeover


class FakeLLM:
    def __init__(self, data):
        self.data, self.calls = data, []

    def complete_json(self, model, system, user, schema):
        self.calls.append((model, system, user))
        return LLMResponse(data=self.data, model=model)


def tk(i, door="D-17", site="A", typ="fire_door", zone="z1"):
    return Takeover(id=f"t{i}", run_id="r", mission_id="m", robot_id="R1", site=site, day=1, shift=1,
                    situation_id=f"s{i}", situation_type=typ, zone=zone, zone_class="corridor",
                    observable=f"FIRE DOOR {door} closed" if door else "cart in the way", tried=[],
                    outcome=OutcomeStatus.TAKEOVER, reason="r", operator_tip="use 25N push",
                    policy_version=0, ts=datetime(2026, 1, 1))


def edit(kind="add_rule", payload=None):
    return {"kind": kind, "payload": payload if payload is not None else {"text": "push gently"},
            "rationale": "why", "scope": {"situation_type": "fire_door", "door_id": "D-17"},
            "prediction": {"train_fixed": 2, "heldout_fixed": 1}}


def test_cluster():
    cl = cluster_takeovers([tk(1), tk(2), tk(3), tk(4, door=None, typ="cart_blocking", zone="z9")])
    assert len(cl) == 2 and cl[0]["count"] == 3 and cl[0]["door_id"] == "D-17"
    assert cl[0]["takeover_ids"] == ["t1", "t2", "t3"] and cl[0]["operator_tips"] == ["use 25N push"]


def test_fixer_filters_and_caps():
    llm = FakeLLM({"edits": [edit(), edit(payload={"locked": {"x": 1}}), edit(), edit(), edit(),
                             {"kind": "bogus"}]})
    clusters = cluster_takeovers([tk(1), tk(2), tk(3)])
    c = Fixer(llm, SEED_PLAYBOOK, model="m").propose(clusters, None, Genome(version=4))
    assert len(c.edits) == 3 and c.author == "fixer" and c.parent_version == 4
    assert all("locked" not in e.payload for e in c.edits)
    p = c.edits[0].payload
    assert p["id"] and p["provenance"] == ["t1", "t2", "t3"] and p["scope"]["door_id"] == "D-17"


def test_min_evidence_in_prompt():
    llm = FakeLLM({"edits": []})
    clusters = cluster_takeovers([tk(1), tk(2), tk(3), tk(9, door=None, typ="spill", zone="zSPILL")])
    Fixer(llm, SEED_PLAYBOOK, model="m").propose(clusters, "summary", Genome(version=0), ["no gain"])
    user = llm.calls[0][2]
    assert "D-17" in user and "zSPILL" not in user and "no gain" in user and "summary" in user


def test_llm_failure_gives_empty():
    class Bad:
        def complete_json(self, *a):
            raise RuntimeError("x")
    assert Fixer(Bad(), SEED_PLAYBOOK, model="m").propose([], None, Genome(version=0)).edits == []


def test_placebo_and_random():
    llm = FakeLLM({"edits": [edit(), edit()]})
    c = Fixer(llm, SEED_PLAYBOOK, model="m").propose(cluster_takeovers([tk(1), tk(2)]), None, Genome(version=0))
    p = placebo(c)
    assert len(p.edits) == len(c.edits) == 2 and p.author == "placebo"
    assert len({e.payload["id"] for e in p.edits}) == 2
    g = Genome(version=1)
    assert random_edit(g, 5).model_dump() == random_edit(g, 5).model_dump()
    assert random_edit(g, 5).author == "random"


def test_coach():
    llm = FakeLLM({"instructions": "new", "max_edits": 99, "min_evidence": 0, "lessons": ["a"] * 12,
                   "gate_threshold": 0})
    tr = [{"verdict": "REJECT", "reason": "no held-out gain", "predicted": {"train_fixed": 1, "heldout_fixed": 2},
           "actual": {"train_fixed": 0, "heldout_fixed": 0}}]
    n = Coach(llm, model="m").evolve(SEED_PLAYBOOK, tr)
    assert n.version == 1 and n.parent == 0 and n.created_by == "coach"
    assert n.max_edits == 5 and n.min_evidence == 1 and len(n.lessons) == 8 and n.instructions == "new"
    assert not hasattr(n, "gate_threshold")
    assert "no held-out gain" in llm.calls[0][2]


def test_coach_failure():
    class Bad:
        def complete_json(self, *a):
            raise RuntimeError("x")
    n = Coach(Bad(), model="m").evolve(SEED_PLAYBOOK, [])
    assert n.version == 1 and n.created_by == "coach" and n.lessons
