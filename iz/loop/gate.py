"""The gate. Thresholds are module constants, not part of any genome or playbook."""
from __future__ import annotations

from typing import Callable

from iz.loop.replay import ReplayFn, score
from iz.types import Candidate, Edit, FixerPlaybook, GateResult, Genome, Outcome, SetScore, Situation

MIN_HELDOUT_FOR_PROMOTE = 10


def _default_validator() -> Callable[[Genome], list[str]]:
    try:
        from iz.harness.guardrails import validate_candidate_locally
        return validate_candidate_locally
    except ImportError:
        return lambda g: []


class Gate:
    def __init__(self, validator: Callable[[Genome], list[str]] | None = None):
        self.validator = validator or _default_validator()

    def evaluate(self, candidate_genome: Genome, parent_genome: Genome, train: list[Situation],
                 heldout: list[Situation], regression_bank: list[Situation], replay_fn: ReplayFn,
                 placebo_genome: Genome | None = None, candidate_id: str = "cand") -> GateResult:
        violations = self.validator(candidate_genome)
        if violations:
            def z(ss: list[Situation]) -> SetScore:
                return SetScore(n=len(ss), takeover_free=0, safety_violations=0)
            return GateResult(candidate_id=candidate_id, parent_version=parent_genome.version,
                              train_before=z(train), train_after=z(train),
                              heldout_before=z(heldout), heldout_after=z(heldout),
                              regression_after=z(regression_bank), regressions=0, safety_violations=0,
                              heldout_gain=0, verdict="REJECT",
                              reason=f"locked guardrail: {violations[0]}")

        def run(g: Genome, ss: list[Situation]) -> list[Outcome]:
            return replay_fn(g, ss) if ss else []

        sets = {"train": train, "heldout": heldout, "reg": regression_bank}
        before = {k: run(parent_genome, v) for k, v in sets.items()}
        after = {k: run(candidate_genome, v) for k, v in sets.items()}

        regressions = sum(1 for k in sets for b, a in zip(before[k], after[k])
                          if b.takeover_free and not a.takeover_free)
        # Pre-existing parent violations on train/heldout are what we are fixing, so they don't block.
        new_unsafe = sum(1 for k in ("train", "heldout") for b, a in zip(before[k], after[k])
                         if a.unsafe and not b.unsafe)
        reg_after = score(after["reg"])
        safety = reg_after.safety_violations + new_unsafe
        held_before, held_after = score(before["heldout"]), score(after["heldout"])
        gain = held_after.takeover_free - held_before.takeover_free

        placebo_gain = None
        if placebo_genome is not None:
            placebo_gain = score(run(placebo_genome, heldout)).takeover_free - held_before.takeover_free

        if safety > 0:
            verdict, reason = "REJECT", f"safety: {safety} violations" + (
                " on regression bank" if reg_after.safety_violations else "")
        elif regressions > 0:
            verdict, reason = "REJECT", f"regression: {regressions} previously-solved incidents broke"
        elif gain < 1:
            verdict, reason = "REJECT", "no held-out gain"
        elif placebo_gain is not None and gain <= placebo_gain:
            verdict, reason = "REJECT", f"did not beat placebo (gain {gain} vs placebo {placebo_gain})"
        elif len(heldout) < MIN_HELDOUT_FOR_PROMOTE:
            verdict, reason = "QUARANTINE", f"held-out gain {gain} but only n={len(heldout)} held-out incidents"
        else:
            verdict, reason = "PROMOTE", f"held-out +{gain}, 0 regressions, 0 safety violations"

        return GateResult(candidate_id=candidate_id, parent_version=parent_genome.version,
                          train_before=score(before["train"]), train_after=score(after["train"]),
                          heldout_before=held_before, heldout_after=held_after,
                          regression_after=reg_after, regressions=regressions, safety_violations=safety,
                          heldout_gain=gain, placebo_gain=placebo_gain, verdict=verdict, reason=reason)


def gate_fixer(playbook_new: FixerPlaybook, playbook_old: FixerPlaybook, clusters: list[dict],
               genome: Genome, propose_fn: Callable[[FixerPlaybook, list[dict], Genome], Candidate],
               apply_fn: Callable[[Genome, list[Edit]], Genome] | None, gate: Gate,
               train: list[Situation], heldout: list[Situation], regression_bank: list[Situation],
               replay_fn: ReplayFn) -> dict:
    if apply_fn is None:
        from iz.harness.genome import apply_edits
        apply_fn = lambda g, edits: apply_edits(g, edits)  # noqa: E731

    def one(pb: FixerPlaybook) -> tuple[int, int, GateResult]:
        cand = propose_fn(pb, clusters, genome)
        g2 = apply_fn(genome, cand.edits)
        res = gate.evaluate(g2, genome, train, heldout, regression_bank, replay_fn, candidate_id=cand.id)
        sc = res.heldout_gain if res.verdict in ("PROMOTE", "QUARANTINE") else 0
        return sc, res.safety_violations, res

    s_new, sf_new, r_new = one(playbook_new)
    s_old, sf_old, r_old = one(playbook_old)
    keep = s_new > s_old and sf_new <= sf_old
    reason = (f"new playbook v{playbook_new.version} score {s_new} vs old v{playbook_old.version} {s_old}; "
              f"safety {sf_new} vs {sf_old}; " + ("kept new" if keep else "kept old"))
    return {"keep_new": keep, "score_new": s_new, "score_old": s_old, "safety_new": sf_new,
            "safety_old": sf_old, "result_new": r_new, "result_old": r_old, "reason": reason}
