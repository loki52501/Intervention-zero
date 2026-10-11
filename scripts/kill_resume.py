r"""Crash-resume proof: run the campaign, kill -9 mid fix-cycle, restart, verify exactly-once effects.

Usage: .venv\Scripts\python.exe scripts/kill_resume.py [--run-id kr-test] [--campaign-args "..."]
Writes out/kill_resume.json. Exit 0 only if every verification passes.

Assumptions about scripts/run_campaign.py (adapt via --campaign-args / --checkpoint):
  - accepts --run-id; checkpoint JSON at out/checkpoints/<run_id>.json listing completed steps under one of
    done / steps / completed_steps / done_steps / completed (list or dict); a step whose key mentions "fix" is a fix cycle
  - events appended to out/events.jsonl as {"id","ts","kind","run_id","payload"}
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
PY = ROOT / ".venv" / "Scripts" / "python.exe"
_STEP_KEYS = ("done", "steps", "completed_steps", "done_steps", "completed")


def read_json(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def count_steps(cp) -> tuple[int, bool]:
    """(completed step count, fix_cycle seen in progress or done)."""
    if not isinstance(cp, dict):
        return 0, False
    for k in _STEP_KEYS:
        v = cp.get(k)
        if isinstance(v, (list, dict)) and v:
            keys = [str(x) for x in (v.keys() if isinstance(v, dict) else v)]
            fix = any("fix" in s.lower() for s in keys) or bool(cp.get("fix_cycle") or cp.get("fix_in_progress"))
            return len(keys), fix
    return 0, bool(cp.get("fix_cycle") or cp.get("fix_in_progress"))


def events_size() -> int:
    p = OUT / "events.jsonl"
    return p.stat().st_size if p.exists() else 0


def read_events(run_id: str, offset: int = 0) -> list[dict]:
    """Events for run_id appended after byte `offset` (so old runs with the same id are ignored)."""
    p = OUT / "events.jsonl"
    evs = []
    if p.exists():
        with open(p, "rb") as f:
            f.seek(offset)
            data = f.read().decode("utf-8", errors="replace")
        for line in data.splitlines():
            try:
                e = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            if e.get("run_id") == run_id:
                evs.append(e)
    return evs


def check_events(evs: list[dict]) -> dict:
    promos = [e for e in evs if e.get("kind") == "promoted"]
    vers = [(e.get("payload") or {}).get("version") for e in promos]
    cids = [(e.get("payload") or {}).get("candidate_id") for e in promos]
    decs = [e.get("id") for e in evs if e.get("kind") == "decision"]
    return {"no_duplicate_promotions": len(set(vers)) == len(vers) and len(set(cids)) == len(cids),
            "decisions_unique": len(set(decs)) == len(decs),
            "n_promotions": len(promos), "n_decisions": len(decs),
            "max_promoted_version": max([v for v in vers if isinstance(v, int)], default=None)}


def final_version(run_id: str):
    sys.path.insert(0, str(ROOT / "scripts"))
    from ablate import genomes_from_json
    for p in (OUT / "checkpoints" / f"{run_id}.json", OUT / f"campaign_{run_id}.json"):
        if p.exists():
            gs = genomes_from_json(p)
            if gs:
                return max(g.version for g in gs)
    return None


def launch(cmd: list[str], log: Path) -> subprocess.Popen:
    f = open(log, "ab")
    return subprocess.Popen(cmd, cwd=str(ROOT), stdout=f, stderr=subprocess.STDOUT)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="kr-test")
    ap.add_argument("--campaign-args", default="--fake-llm")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--wait-kill-s", type=float, default=600)
    ap.add_argument("--wait-done-s", type=float, default=900)
    ap.add_argument("--fresh", action="store_true", help="delete old checkpoint/events for this run id first")
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    cp_path = Path(a.checkpoint) if a.checkpoint else OUT / "checkpoints" / f"{a.run_id}.json"
    if a.fresh and cp_path.exists():
        cp_path.unlink()
    cmd = [str(PY), str(ROOT / "scripts" / "run_campaign.py"), "--run-id", a.run_id, *shlex.split(a.campaign_args)]
    log = OUT / f"kill_resume_{a.run_id}.log"
    res: dict = {"run_id": a.run_id, "cmd": cmd}
    ev_offset = events_size()
    t0 = time.time()
    p1 = launch(cmd, log)
    killed = False
    steps_at_kill, fix_at_kill = 0, False
    while time.time() - t0 < a.wait_kill_s:
        if p1.poll() is not None:
            break
        steps_at_kill, fix_at_kill = count_steps(read_json(cp_path))
        if steps_at_kill >= 2 and fix_at_kill:
            p1.kill()  # TerminateProcess on Windows
            p1.wait()
            killed = True
            break
        time.sleep(0.25)
    else:
        p1.kill()
        p1.wait()
    res.update(killed_mid_run=killed, steps_at_kill=steps_at_kill, fix_cycle_at_kill=fix_at_kill,
               time_to_kill_s=round(time.time() - t0, 2), first_exit=p1.returncode)
    t1 = time.time()
    p2 = launch(cmd, log)
    try:
        rc = p2.wait(timeout=a.wait_done_s)
    except subprocess.TimeoutExpired:
        p2.kill()
        p2.wait()
        rc = None
    res.update(second_exit=rc, resume_time_s=round(time.time() - t1, 2))
    cp = read_json(cp_path)
    steps_end, _ = count_steps(cp)
    ev = check_events(read_events(a.run_id, ev_offset))
    fv = final_version(a.run_id)
    mv = ev["max_promoted_version"]
    consistent = (fv == mv) if (fv is not None and mv is not None) else (fv is None and mv is None) or (fv is not None)
    checks = {
        "killed_mid_run": killed,
        "completed": rc == 0 and steps_end >= steps_at_kill,
        "no_duplicate_promotions": ev["no_duplicate_promotions"],
        "decisions_unique": ev["decisions_unique"],
        "final_version_consistent": bool(consistent),
    }
    res.update(steps_end=steps_end, events=ev, final_version=fv, checks=checks, passed=all(checks.values()))
    (OUT / "kill_resume.json").write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"checks": checks, "passed": res["passed"]}, indent=2))
    return 0 if res["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
