import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import ablate  # noqa: E402
import kill_resume  # noqa: E402
from iz.types import Genome, Outcome, OutcomeStatus  # noqa: E402
from iz.world.stats import wilson  # noqa: E402


def outs(free, tk, unsafe):
    return ([Outcome(status=OutcomeStatus.SUCCESS, reason="ok")] * free
            + [Outcome(status=OutcomeStatus.TAKEOVER, reason="x")] * tk
            + [Outcome(status=OutcomeStatus.SAFETY_VIOLATION, reason="u")] * unsafe)


def test_arm_row_wilson():
    r = ablate.arm_row("evolved", outs(8, 1, 1), [1, 2])
    assert (r["n"], r["takeover_free"], r["safety_violations"]) == (10, 8, 1)
    assert (r["ci_lo"], r["ci_hi"]) == wilson(8, 10)
    assert r["tfmr"] == 0.8


def test_empty_and_md():
    r0 = ablate.arm_row("baseline", [], [1])
    assert r0["n"] == 0 and r0["ci_hi"] == 1.0
    md = ablate.render_md([r0, ablate.arm_row("x", outs(1, 1, 0), [1, 2])])
    assert md.count("\n") == 4 and "| x | 2 | 1 | 50.0%" in md


def test_arm_genomes():
    base = Genome(version=0)
    p = ablate.placebo_genome(base, 3)
    assert len(p.rulebook) == 3
    r = ablate.random_genome(base, 3, 5)
    assert r.version == 3


def test_end_to_end_fake_rule_judge():
    ev = ablate.placebo_genome(Genome(version=0), 2)
    res = ablate.run_ablation("t", [9100], True, genomes=(Genome(version=0), ev), judge_kind="rule")
    assert [r["arm"] for r in res["rows"]] == ["baseline", "evolved", "placebo", "random-edit"]
    assert all(r["n"] == res["n_incidents"] > 0 for r in res["rows"])
    assert "D" in res["sites"]


def test_kill_resume_verifiers():
    ev = [{"id": "1", "kind": "promoted", "payload": {"version": 1, "candidate_id": "a"}},
          {"id": "2", "kind": "promoted", "payload": {"version": 2, "candidate_id": "b"}},
          {"id": "3", "kind": "decision", "payload": {}}]
    assert kill_resume.check_events(ev)["no_duplicate_promotions"]
    assert kill_resume.check_events(ev)["decisions_unique"]
    bad = ev + [{"id": "1", "kind": "promoted", "payload": {"version": 1, "candidate_id": "a"}}]
    assert not kill_resume.check_events(bad)["no_duplicate_promotions"]
    assert kill_resume.count_steps({"steps": {"d1s1": {}, "fix_cycle_1": {}}}) == (2, True)
    assert kill_resume.count_steps({"completed": ["a"]}) == (1, False)
