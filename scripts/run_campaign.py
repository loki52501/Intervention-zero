"""Run (or resume) a long-horizon campaign. Usage: python scripts/run_campaign.py --run-id demo --fake-llm"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="campaign")
    ap.add_argument("--days", type=int, default=14)
    ap.add_argument("--shifts", type=int, default=3)
    ap.add_argument("--missions", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fake-llm", action="store_true")
    ap.add_argument("--raindrop", action="store_true")
    ap.add_argument("--stop-after", type=int, default=None, help="stop after N executed steps (resume later)")
    ap.add_argument("--no-skills", action="store_true")
    ap.add_argument("--coach-every", type=int, default=0)
    ap.add_argument("--drift-day", type=int, default=8)
    ap.add_argument("--atlas", choices=["auto", "on", "off"], default="auto",
                    help="auto: use Atlas if a 3s ping works (never with --fake-llm)")
    ap.add_argument("--rule-judge", action="store_true", help="skip MuJoCo, use the rule judge")
    a = ap.parse_args(argv)

    from dotenv import load_dotenv
    load_dotenv()
    if not a.raindrop:
        os.environ["RAINDROP_ENABLED"] = "0"
    from iz.obs import rd
    if a.raindrop:
        rd.force_enable()
    from iz.loop.campaign import fake_llm, run_campaign

    llm = fake_llm() if a.fake_llm else None
    judge = None
    if a.rule_judge:
        from iz.world.judge import RuleJudge
        judge = RuleJudge()
    try:
        summary = run_campaign(a.run_id, days=a.days, shifts=a.shifts, missions_per_shift=a.missions,
                               seed=a.seed, drift_day=a.drift_day, llm=llm, judge=judge,
                               coach_every=a.coach_every, skills=not a.no_skills, stop_after=a.stop_after,
                               use_atlas={'auto': None, 'on': True, 'off': False}[a.atlas])
    finally:
        rd.flush()
    out = Path("out") / f"campaign_{a.run_id}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    brief = {k: v for k, v in summary.items() if k not in ("decision_log", "tfmr_per_day_per_site")}
    print(json.dumps(brief, indent=2, default=str))
    print("tfmr_per_day_per_site:", json.dumps(summary["tfmr_per_day_per_site"]))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
