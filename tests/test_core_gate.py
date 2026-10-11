from __future__ import annotations

import pytest

from iz.loop.gate import Gate, MIN_HELDOUT_FOR_PROMOTE, gate_fixer
from iz.types import (Candidate, Edit, FixerPlaybook, Genome, HiddenTruth, Outcome,
                      OutcomeStatus, Prediction, Scope, Situation)

OK = Outcome(status=OutcomeStatus.SUCCESS, reason="ok")
TK = Outcome(status=OutcomeStatus.TAKEOVER, reason="tk")
UN = Outcome(status=OutcomeStatus.SAFETY_VIOLATION, reason="unsafe")


def sits(prefix: str, n: int) -> list[Situation]:
    return [Situation(id=f"{prefix}{i}", site="A", type="fire_door", zone="z", zone_class="corridor",
                      edge=("a", "b"), observable="o", hidden=HiddenTruth(operator_tip="x"), seed=i)
            for i in range(n)]


def make_replay(table: dict[tuple[int, str], Outcome], default=TK, calls=None):
    def fn(genome, ss):
        if calls is not None:
            calls.append(genome.version)
        return [table.get((genome.version, s.id), default) for s in ss]
    return fn


PARENT = Genome(version=1)
CAND = Genome(version=2, parent=1)
PLACEBO = Genome(version=3, parent=1)


def setup(n_held=12, gain=2):
    train, held, reg = sits("t", 4), sits("h", n_held), sits("r", 3)
    table = {}
    for s in reg:
        table[(1, s.id)] = OK
        table[(2, s.id)] = OK
    for s in held[:gain]:
        table[(2, s.id)] = OK
    return train, held, reg, table


def run(table, train, held, reg, validator=None, placebo=None):
    return Gate(validator=validator or (lambda g: [])).evaluate(
        CAND, PARENT, train, held, reg, make_replay(table), placebo_genome=placebo)


def test_promote():
    train, held, reg, table = setup()
    r = run(table, train, held, reg)
    assert r.verdict == "PROMOTE" and r.heldout_gain == 2 and r.regressions == 0
    assert r.regression_after.takeover_free == 3


def test_reject_safety():
    train, held, reg, table = setup()
    table[(2, reg[0].id)] = UN
    r = run(table, train, held, reg)
    assert r.verdict == "REJECT" and r.reason.startswith("safety")
    assert "regression bank" in r.reason and r.safety_violations == 1


def test_reject_regression():
    train, held, reg, table = setup()
    table[(1, train[0].id)] = OK
    r = run(table, train, held, reg)
    assert r.verdict == "REJECT" and r.reason.startswith("regression") and r.regressions == 1


def test_reject_no_gain():
    train, held, reg, table = setup(gain=0)
    r = run(table, train, held, reg)
    assert r.verdict == "REJECT" and r.reason == "no held-out gain"


def test_reject_placebo():
    train, held, reg, table = setup(gain=2)
    for s in held[:2]:
        table[(3, s.id)] = OK
    for s in reg:
        table[(3, s.id)] = OK
    r = run(table, train, held, reg, placebo=PLACEBO)
    assert r.verdict == "REJECT" and r.reason.startswith("did not beat placebo")
    assert r.placebo_gain == 2


def test_promote_beats_placebo():
    train, held, reg, table = setup(gain=2)
    table[(3, held[0].id)] = OK
    r = run(table, train, held, reg, placebo=PLACEBO)
    assert r.verdict == "PROMOTE" and r.placebo_gain == 1


def test_quarantine_small_heldout():
    train, held, reg, table = setup(n_held=5, gain=1)
    assert len(held) < MIN_HELDOUT_FOR_PROMOTE
    r = run(table, train, held, reg)
    assert r.verdict == "QUARANTINE"


def test_locked_guardrail_no_replay():
    train, held, reg, _ = setup()

    def boom(g, ss):
        raise AssertionError("replay must not be called")
    r = Gate(validator=lambda g: ["x"]).evaluate(CAND, PARENT, train, held, reg, boom)
    assert r.verdict == "REJECT" and r.reason == "locked guardrail: x"
    assert r.heldout_after.n == len(held) and r.heldout_after.takeover_free == 0


def test_new_unsafe_on_train_blocks_but_preexisting_does_not():
    train, held, reg, table = setup()
    table[(1, train[0].id)] = UN
    table[(2, train[0].id)] = UN
    assert run(table, train, held, reg).verdict == "PROMOTE"
    table[(2, train[1].id)] = UN
    assert run(table, train, held, reg).reason.startswith("safety")


def test_gate_fixer_keeps_new_only_if_strictly_better():
    train, held, reg, table = setup(gain=2)
    old_pb = FixerPlaybook(version=0, instructions="old")
    new_pb = FixerPlaybook(version=1, parent=0, instructions="new", created_by="coach")
    g_old, g_new = Genome(version=2, parent=1), Genome(version=3, parent=1)
    for s in held[:1]:
        table[(2, s.id)] = OK
    table = {k: v for k, v in table.items() if k[0] != 2 or k[1].startswith("r") or k[1] == held[0].id}
    for s in held[:3]:
        table[(3, s.id)] = OK
    for s in reg:
        table[(3, s.id)] = OK

    def propose(pb, clusters, genome):
        return Candidate(id=pb.instructions, parent_version=1, edits=[
            Edit(id="e0", kind="dispatcher", payload={}, rationale="r", scope=Scope(),
                 prediction=Prediction(train_fixed=0, heldout_fixed=0))])

    def apply(genome, edits):
        return g_new if edits and getattr(apply, "cur", None) == "new" else g_old

    def propose2(pb, clusters, genome):
        apply.cur = pb.instructions
        return propose(pb, clusters, genome)

    gate = Gate(validator=lambda g: [])
    out = gate_fixer(new_pb, old_pb, [], PARENT, propose2, apply, gate, train, held, reg, make_replay(table))
    assert out["score_new"] == 3 and out["score_old"] == 1 and out["keep_new"] is True
    out2 = gate_fixer(old_pb, new_pb, [], PARENT, propose2, apply, gate, train, held, reg, make_replay(table))
    assert out2["keep_new"] is False
    out3 = gate_fixer(old_pb, old_pb, [], PARENT, propose2, apply, gate, train, held, reg, make_replay(table))
    assert out3["keep_new"] is False
