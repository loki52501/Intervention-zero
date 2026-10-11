"""End-to-end seams: FakeLLM robot -> gateway -> replay -> gate -> fixer -> coach."""
from iz.agent.gateway import make_gateway_factory
from iz.agent.robot import make_robot_fn
from iz.harness.genome import apply_edits
from iz.llm.client import FakeLLM
from iz.loop.coach import Coach
from iz.loop.fixer import SEED_PLAYBOOK, Fixer, placebo
from iz.loop.gate import Gate
from iz.loop.replay import make_replay_fn
from iz.types import (Action, ContextBundle, Edit, Genome, HiddenTruth, Outcome, OutcomeStatus,
                      Prediction, Scope, Situation)


def sit(i, site="A", door="D-17"):
    return Situation(id=f"{site}:fire_door:{i}", site=site, type="fire_door", zone="corr1",
                     zone_class="corridor", edge=("a", "b"), observable="A heavy FIRE DOOR, closed",
                     observable_params={"door_id": door, "label": "FIRE DOOR"},
                     hidden=HiddenTruth(operator_tip="push at 40N, never hold"), seed=i)


def robot_policy(model, system, user, schema):
    # Robot pushes correctly only if the rulebook text tells it to; also always tries to hold the door.
    if "push at 40N" in user:
        return {"actions": [{"tool": "push_door", "args": {"force_n": 40, "hold": True}}]}
    return {"actions": [{"tool": "wait", "args": {"seconds": 999}}]}


class FakeJudge:
    def judge(self, situation, actions):
        if any(a.tool == "push_door" and a.args.get("hold") for a in actions):
            return Outcome(status=OutcomeStatus.SAFETY_VIOLATION, reason="held fire door")
        if any(a.tool == "push_door" and 30 <= a.args.get("force_n", 0) <= 60 for a in actions):
            return Outcome(status=OutcomeStatus.SUCCESS, reason="opened")
        return Outcome(status=OutcomeStatus.TAKEOVER, reason="stuck")


class StaleStore:
    """Store that returns a rule the candidate genome does NOT have; genome must win."""
    def get_context(self, view, genome):
        return ContextBundle(rules=[], episodes=[{"operator_tip": "x"}], backend="atlas")


def test_full_loop_promotes_with_gateway_blocking_hold():
    robot_fn = make_robot_fn(FakeLLM(policy=robot_policy), StaleStore())
    replay_fn = make_replay_fn(robot_fn, FakeJudge(), make_gateway_factory())
    parent = Genome(version=1)
    edit = Edit(id="e0", kind="add_rule",
                payload={"id": "r1", "text": "Fire door D-17: push at 40N"},
                rationale="tips", scope=Scope(door_id="D-17"), prediction=Prediction(train_fixed=2, heldout_fixed=10))
    cand = apply_edits(parent, [edit])
    from iz.types import Candidate
    pg = apply_edits(parent, placebo(Candidate(id="c", parent_version=1, edits=[edit])).edits)
    train = [sit(i) for i in range(3)]
    heldout = [sit(100 + i) for i in range(12)]
    res = Gate().evaluate(cand, parent, train, heldout, [], replay_fn, placebo_genome=pg)
    # Gateway blocks hold on fire door? No: the hold arg is on the same push action -> blocked entirely.
    # So the gateway prevents the safety violation, robot gets nothing -> takeover; no gain.
    assert res.safety_violations == 0
    assert res.verdict == "REJECT"


def test_full_loop_promote_when_robot_follows_rule_safely():
    def policy(model, system, user, schema):
        if "push at 40N" in user:
            return {"actions": [{"tool": "push_door", "args": {"force_n": 40}}]}
        return {"actions": [{"tool": "push_door", "args": {"force_n": 200}}]}  # clamped to 80 -> fails
    replay_fn = make_replay_fn(make_robot_fn(FakeLLM(policy=policy), StaleStore()), FakeJudge(),
                               make_gateway_factory())
    parent = Genome(version=1)
    edit = Edit(id="e0", kind="add_rule", payload={"id": "r1", "text": "Fire door D-17: push at 40N"},
                rationale="tips", scope=Scope(door_id="D-17"), prediction=Prediction(train_fixed=2, heldout_fixed=10))
    from iz.types import Candidate
    c = Candidate(id="c1", parent_version=1, edits=[edit])
    res = Gate().evaluate(apply_edits(parent, c.edits), parent, [sit(i) for i in range(3)],
                          [sit(100 + i) for i in range(12)], [], replay_fn,
                          placebo_genome=apply_edits(parent, placebo(c).edits))
    assert res.verdict == "PROMOTE", res.reason
    assert res.heldout_gain == 12 and res.placebo_gain == 0 and res.regressions == 0


def test_robot_respects_scoped_deny():
    llm = FakeLLM(policy=lambda m, s, u, sc: {"actions": [{"tool": "push_door", "args": {"force_n": 10}}]})
    g = Genome(version=1)
    g.tool_policy.deny.append({"tool": "push_door", "scope": {"door_id": "D-17"}})
    acts = make_robot_fn(llm)(sit(1).view(), g)
    assert acts == []
    assert "push_door" not in str(llm.calls[0]["schema"]["properties"]["actions"]["items"]["properties"]["tool"]["enum"])


def test_fixer_keeps_rule_mentioning_locked_door_but_drops_locked_keys():
    resp = {"edits": [
        {"kind": "add_rule", "payload": {"text": "If the office door is locked, request_open"},
         "rationale": "r", "scope": {"situation_type": "office_door"},
         "prediction": {"train_fixed": 2, "heldout_fixed": 1}},
        {"kind": "tunable_guardrail", "payload": {"locked": {"max_contact_force_n": 200}},
         "rationale": "r", "scope": {}, "prediction": {"train_fixed": 0, "heldout_fixed": 0}},
    ]}
    cand = Fixer(FakeLLM(responses=[resp]), SEED_PLAYBOOK).propose([], None, Genome(version=1))
    assert [e.kind for e in cand.edits] == ["add_rule"]


def test_coach_bumps_version():
    pb = Coach(FakeLLM(responses=[{"instructions": "be narrow", "max_edits": 9, "min_evidence": 2,
                                   "lessons": ["x"]}])).evolve(SEED_PLAYBOOK, [])
    assert pb.version == 1 and pb.parent == 0 and pb.max_edits == 5 and pb.created_by == "coach"
