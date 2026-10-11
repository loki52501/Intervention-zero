"""Naive-robot baseline: TFMR with Wilson 95% CI over 3 days x 3 shifts x train sites."""
from __future__ import annotations

import argparse
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from iz.types import TRAIN_SITES, OutcomeStatus  # noqa: E402


def run_baseline(seed: int = 0, judge_kind: str = "physics", n: int = 10) -> dict:
    from iz.world.generator import generate_missions
    from iz.world.judge import get_judge
    from iz.world.naive import naive_actions
    from iz.world.stats import tfmr, wilson

    judge = get_judge(judge_kind)
    outcomes: list[list] = []
    per: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    statuses: Counter = Counter()
    by: Counter = Counter()
    for site in TRAIN_SITES:
        for day in (1, 2, 3):
            for shift in (1, 2, 3):
                for m in generate_missions(f"baseline-{seed}", site, day, shift, seed, n):
                    outs = []
                    for sit in m.situations:
                        o = judge.judge(sit, naive_actions(sit.view()))
                        outs.append(o)
                        per[sit.type][1] += 1
                        per[sit.type][0] += int(o.takeover_free)
                        statuses[o.status.value] += 1
                        by[o.judged_by] += 1
                    outcomes.append(outs)
    free = sum(all(o.takeover_free for o in outs) for outs in outcomes)
    total = len(outcomes)
    return {"missions": total, "free": free, "tfmr": tfmr(outcomes),
            "ci": wilson(free, total),
            "per_type": {k: (v[0], v[1]) for k, v in sorted(per.items())},
            "statuses": dict(statuses), "judged_by": dict(by)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--judge", choices=["physics", "rule"], default="physics")
    ap.add_argument("--n", type=int, default=10)
    a = ap.parse_args()
    t0 = time.time()
    r = run_baseline(a.seed, a.judge, a.n)
    lo, hi = r["ci"]
    nsit = sum(v[1] for v in r["per_type"].values())
    print(f"missions={r['missions']} situations={nsit}")
    print(f"TFMR = {r['free']}/{r['missions']} = {r['tfmr']:.1%}  (Wilson 95% CI {lo:.1%} - {hi:.1%})")
    print("per-type naive success:")
    for k, (s, n) in r["per_type"].items():
        print(f"  {k:26s} {s:4d}/{n:<4d} {s / n if n else 0:.0%}")
    st = r["statuses"]
    print(f"SAFETY_VIOLATION={st.get(OutcomeStatus.SAFETY_VIOLATION.value, 0)} "
          f"ESCALATION_MISSED={st.get(OutcomeStatus.ESCALATION_MISSED.value, 0)}")
    print(f"judged_by: {r['judged_by']}")
    print(f"wall time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
