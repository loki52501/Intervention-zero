r"""Ablation on FRESH held-out incidents: baseline vs evolved vs placebo vs random-edit.

Usage: .venv\Scripts\python.exe scripts/ablate.py --run-id RUN [--seeds 3] [--fake-llm]
Writes out/ablation.json and out/ablation.md.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from iz.types import Genome, Outcome, RuleEntry, Scope  # noqa: E402
from iz.world.stats import wilson  # noqa: E402

OUT = ROOT / "out"
PLACEBO_TEXTS = ["Keep a log of observed conditions.", "Note the time of day when starting a mission.",
                 "Remember that facilities are generally busy on weekdays.",
                 "Stay polite when speaking to staff.", "Battery status should be checked regularly."]


# ---------- table building (pure, tested) ----------
def arm_row(name: str, outcomes: list[Outcome], seeds: list[int], n_rules: int | None = None) -> dict:
    n = len(outcomes)
    free = sum(1 for o in outcomes if o.takeover_free)
    lo, hi = wilson(free, n)
    return {"arm": name, "n": n, "takeover_free": free, "tfmr": (free / n) if n else 0.0,
            "ci_lo": lo, "ci_hi": hi, "safety_violations": sum(1 for o in outcomes if o.unsafe),
            "seeds": list(seeds), "n_rules": n_rules}


def render_md(rows: list[dict]) -> str:
    out = ["| arm | n | takeover_free | TFMR | Wilson 95% CI | safety violations | seeds |",
           "|---|---:|---:|---:|---|---:|---|"]
    for r in rows:
        out.append(f"| {r['arm']} | {r['n']} | {r['takeover_free']} | {r['tfmr']:.1%} | "
                   f"{r['ci_lo']:.1%} - {r['ci_hi']:.1%} | {r['safety_violations']} | "
                   f"{','.join(map(str, r['seeds']))} |")
    return "\n".join(out) + "\n"


# ---------- genome loading ----------
def _find_genomes(obj, acc: list[dict]) -> None:
    if isinstance(obj, dict):
        if "rulebook" in obj and "version" in obj:
            acc.append(obj)
            return
        for v in obj.values():
            _find_genomes(v, acc)
    elif isinstance(obj, list):
        for v in obj:
            _find_genomes(v, acc)


def genomes_from_json(path: Path) -> list[Genome]:
    acc: list[dict] = []
    _find_genomes(json.loads(path.read_text(encoding="utf-8")), acc)
    res = []
    for d in acc:
        try:
            res.append(Genome.model_validate(d))
        except Exception:  # noqa: BLE001
            pass
    return res


def load_evolved(run_id: str) -> tuple[Genome, str]:
    """Return (evolved genome, source). The run's own local checkpoint first (it is the campaign's source of
    truth); Atlas active genome only when no local file exists for this run id."""
    for p in (OUT / "checkpoints" / f"{run_id}.json", OUT / f"campaign_{run_id}.json"):
        if p.exists():
            gs = genomes_from_json(p)
            if gs:
                return max(gs, key=lambda g: g.version), str(p)
    try:
        from iz.store.db import mongo_available
        if mongo_available():
            from iz.store.genome_store import GenomeStore
            return GenomeStore().get_active(), "atlas"
    except Exception:  # noqa: BLE001
        pass
    raise SystemExit(f"no evolved genome found for run {run_id} (no local checkpoint, Atlas unreachable)")


def baseline_genome() -> Genome:
    """The campaign's v0 harness (seed rules), so baseline vs evolved isolates what the loop added."""
    try:
        from iz.loop.campaign import seed_genome
        return seed_genome()
    except Exception:  # noqa: BLE001
        return Genome(version=0)


# ---------- arms ----------
def placebo_genome(base: Genome, k: int) -> Genome:
    g = Genome.model_validate(base.model_dump())
    for i in range(k):
        g.rulebook.append(RuleEntry(id=f"placebo-{i}", scope=Scope(), text=PLACEBO_TEXTS[i % len(PLACEBO_TEXTS)]))
    g.version = base.version + 1
    g.parent = base.version
    g.created_by = "placebo"
    return g


def random_genome(base: Genome, k: int, seed: int) -> Genome:
    from iz.harness.genome import apply_edits
    from iz.loop.fixer import random_edit
    g = base
    for i in range(k):
        g = apply_edits(g, random_edit(g, seed * 1000 + i).edits, created_by="random")
    return g


def fake_policy(model, system, user, schema):
    """Deterministic smoke policy: follows a force number from rule text if present."""
    m = re.search(r"(\d{2,3})\s*N\b", user)
    if m:
        return {"actions": [{"tool": "push_door", "args": {"force_n": float(m.group(1)), "hold": False}}]}
    return {"actions": [{"tool": "wait", "args": {"seconds": 5}}]}


def run_ablation(run_id: str, seeds: list[int], fake: bool, genomes: tuple[Genome, Genome] | None = None,
                 judge_kind: str = "physics") -> dict:
    from iz.agent.gateway import make_gateway_factory
    from iz.agent.robot import make_robot_fn
    from iz.loop.replay import replay
    from iz.world.generator import heldout_incidents
    from iz.world.judge import get_judge

    if genomes:
        base, evolved = genomes
        src = "provided"
    else:
        evolved, src = load_evolved(run_id)
        base = baseline_genome()
    if fake:
        try:  # same deterministic offline policy the campaign uses
            from iz.loop.campaign import fake_llm
            llm = fake_llm()
        except Exception:  # noqa: BLE001
            from iz.llm.client import FakeLLM
            llm = FakeLLM(policy=fake_policy)
    else:
        from iz.llm.client import make_llm
        llm = make_llm(cached=True)
    robot_fn = make_robot_fn(llm, None)
    judge = get_judge(judge_kind)
    gw = make_gateway_factory()
    incidents = []
    for s in seeds:
        incidents += heldout_incidents(seed=s)
    seen, inc = set(), []
    for x in incidents:
        if x.id not in seen:
            seen.add(x.id)
            inc.append(x)
    k = max(1, sum(1 for r in evolved.rulebook if r.status == "active")
            - sum(1 for r in base.rulebook if r.status == "active"))
    arms = [("baseline", base), ("evolved", evolved), ("placebo", placebo_genome(base, k)),
            ("random-edit", random_genome(base, k, seeds[0]))]
    rows = []
    for name, g in arms:
        outs = replay(inc, g, robot_fn, judge, gw(g))
        rows.append(arm_row(name, outs, seeds, sum(1 for r in g.rulebook if r.status == "active")))
    return {"run_id": run_id, "genome_source": src, "fake_llm": fake, "seeds": seeds,
            "sites": sorted({x.site for x in inc}), "n_incidents": len(inc), "k_edits": k,
            "evolved_version": evolved.version, "rows": rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--seed-base", type=int, default=9100, help="fresh seeds, not used by the campaign")
    ap.add_argument("--fake-llm", action="store_true")
    ap.add_argument("--judge", choices=["physics", "rule"], default="physics")
    a = ap.parse_args()
    seeds = [a.seed_base + i for i in range(a.seeds)]
    res = run_ablation(a.run_id, seeds, a.fake_llm, judge_kind=a.judge)
    OUT.mkdir(exist_ok=True)
    (OUT / "ablation.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    md = (f"Ablation run={res['run_id']} evolved=v{res['evolved_version']} ({res['genome_source']}) "
          f"incidents={res['n_incidents']} sites={','.join(res['sites'])} fake_llm={res['fake_llm']}\n\n"
          + render_md(res["rows"]))
    (OUT / "ablation.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
