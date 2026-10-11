"""SHOP learning loop: takeover records -> LLM fixer -> gate -> promote, on controller PARAMETERS.

Honest claim: the robot learns how to drive (speeds, distances, force limits) from its own
sensor evidence; it does not learn motor control from scratch.

Gate: iz.loop.gate.Gate works on Genome/Situation/replay objects, which do not map cleanly onto
per-seed episode results, so the SAME rule is re-implemented here on aggregates:
PROMOTE iff safety_violations == 0 on all sets, regressions == 0, held-out gain >= 1.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import uuid
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from iz.llm.client import fixer_model

log = logging.getLogger(__name__)

_FALLBACK_LOCKED = {"min_person_clearance_m": 0.35, "max_wrist_force_n": 300}


def _controller():
    from iz.shop import controller
    return controller


def _bounds() -> dict:
    return dict(_controller().PARAM_BOUNDS)


def _locked() -> dict:
    try:
        v = getattr(_controller(), "LOCKED", None)
        if v:
            return dict(v) if isinstance(v, dict) else {k: _FALLBACK_LOCKED.get(k) for k in v}
    except Exception:  # noqa: BLE001
        pass
    return dict(_FALLBACK_LOCKED)


def default_params() -> dict:
    return dict(_controller().DEFAULT_PARAMS)


# ---------------------------------------------------------------- evaluation
def _worker(args: tuple) -> dict:
    params, seed = args
    from iz.shop.episode import run_episode
    return run_episode(params, seed)


def evaluate(params: dict, seeds) -> list[dict]:
    seeds = list(seeds)
    if not seeds:
        return []
    if len(seeds) == 1:
        return [_worker((params, seeds[0]))]
    with ProcessPoolExecutor(4) as ex:
        return list(ex.map(_worker, [(params, s) for s in seeds]))


# ---------------------------------------------------------------- fixer
_SCHEMA = {
    "type": "object",
    "properties": {
        "edits": {"type": "object", "additionalProperties": {"type": "number"}},
        "rationale": {"type": "string"},
        "prediction": {"type": "object", "properties": {
            "train_fixed": {"type": "integer"}, "heldout_fixed": {"type": "integer"}}},
    },
    "required": ["edits", "rationale", "prediction"],
}

_SYSTEM = (
    "You are the harness fixer for a retail robot that pushes a shopping cart. Its behaviour is a "
    "sensor-driven controller whose numeric PARAMETERS are evolvable. You see takeover records "
    "(failure, operator tip, sensor-trace summary) and propose a minimal edit to the parameters. "
    "Respect the given bounds. NEVER edit locked keys. Gate feedback from the previous attempt is "
    "provided; do not repeat a rejected edit. Return JSON: edits {param: new_value}, rationale, "
    "prediction {train_fixed, heldout_fixed}."
)


def failure_counts(records: list[dict]) -> dict:
    out: dict[str, int] = {}
    for r in records:
        f = r.get("failure") or r.get("status")
        out[f] = out.get(f, 0) + 1
    return out


def propose(llm, params: dict, takeover_records: list[dict], gate_feedback: dict | None = None) -> dict:
    bounds, locked = _bounds(), _locked()
    recs = [{"seed": r.get("seed"), "failure": r.get("failure"), "status": r.get("status"),
             "operator_tip": r.get("operator_tip"), "summary": r.get("summary")}
            for r in takeover_records[:8]]
    user = (
        f"CURRENT_PARAMS: {json.dumps(params, sort_keys=True)}\n"
        f"BOUNDS: {json.dumps(bounds)}\nLOCKED (never edit): {json.dumps(sorted(locked))}\n"
        f"FAILURE_COUNTS: {json.dumps(failure_counts(takeover_records), sort_keys=True)}\n"
        f"TAKEOVER_RECORDS: {json.dumps(recs, default=str)}\n"
        f"GATE_FEEDBACK: {json.dumps(gate_feedback, default=str) if gate_feedback else 'none'}"
    )
    resp = llm.complete_json(model=fixer_model(), system=_SYSTEM, user=user, schema=_SCHEMA)
    data = getattr(resp, "data", resp) or {}
    edits: dict[str, float] = {}
    refused: list[str] = []
    for k, v in (data.get("edits") or {}).items():
        if k in locked:
            msg = f"REFUSED edit to LOCKED key {k!r}"
            log.warning(msg)
            refused.append(msg)
            continue
        if k not in bounds:
            refused.append(f"REFUSED unknown param {k!r}")
            continue
        try:
            v = float(v)
        except (TypeError, ValueError):
            refused.append(f"REFUSED non-numeric value for {k!r}")
            continue
        lo, hi = bounds[k]
        edits[k] = min(max(v, lo), hi)
    pred = data.get("prediction") or {}
    return {"edits": edits, "rationale": str(data.get("rationale", "")),
            "prediction": {"train_fixed": int(pred.get("train_fixed", 0) or 0),
                           "heldout_fixed": int(pred.get("heldout_fixed", 0) or 0)},
            "refused": refused}


# ---------------------------------------------------------------- gate
def _ok(r: dict) -> bool:
    return bool(r.get("success"))


def _unsafe(r: dict) -> bool:
    return r.get("status") == "safety_violation"


def gate(parent_params, cand_params, train_seeds, heldout_seeds, safety_seeds) -> dict:
    train_seeds, heldout_seeds, safety_seeds = list(train_seeds), list(heldout_seeds), list(safety_seeds)
    sets = {"train": train_seeds, "heldout": heldout_seeds, "safety": safety_seeds}
    before = {k: evaluate(parent_params, s) for k, s in sets.items()}
    after = {k: evaluate(cand_params, s) for k, s in sets.items()}
    regressions = sum(1 for k in sets for b, a in zip(before[k], after[k]) if _ok(b) and not _ok(a))
    safety = sum(1 for k in sets for a in after[k] if _unsafe(a))
    tb, ta = sum(map(_ok, before["train"])), sum(map(_ok, after["train"]))
    hb, ha = sum(map(_ok, before["heldout"])), sum(map(_ok, after["heldout"]))
    gain = ha - hb
    if safety > 0:
        verdict, reason = "REJECT", f"safety: {safety} violation(s) across train/held-out/safety sets"
    elif regressions > 0:
        verdict, reason = "REJECT", f"regression: {regressions} previously-solved seed(s) now fail"
    elif gain < 1:
        verdict, reason = "REJECT", f"no held-out gain ({hb}->{ha})"
    else:
        verdict, reason = "PROMOTE", f"held-out +{gain}, 0 regressions, 0 safety violations"
    return {"verdict": verdict, "reason": reason,
            "train": {"before": tb, "after": ta, "n": len(train_seeds)},
            "heldout": {"before": hb, "after": ha, "n": len(heldout_seeds)},
            "safety_violations": safety, "regressions": regressions, "heldout_gain": gain,
            "_after": {"train": after["train"], "heldout": after["heldout"]}}


# ---------------------------------------------------------------- events / atlas
class _Events:
    def __init__(self, run_id: str, path: Path):
        self.run_id, self.path, self._seq = run_id, path, 0
        self._prefix = uuid.uuid4().hex[:8]
        self._lock = threading.Lock()
        path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, kind: str, payload: dict) -> dict:
        payload = {**payload, "world": "shop"}
        with self._lock:
            self._seq += 1
            ev = {"id": f"{self._prefix}-{self._seq:06d}",
                  "ts": datetime.now(timezone.utc).isoformat(),
                  "kind": kind, "run_id": self.run_id, "payload": payload}
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(ev, default=str) + "\n")
        return ev


def _atlas_collection():
    uri = os.getenv("MONGODB_URI")
    if not uri:
        try:
            from dotenv import load_dotenv
            load_dotenv()
            uri = os.getenv("MONGODB_URI")
        except Exception:  # noqa: BLE001
            pass
    if not uri:
        return None
    try:
        from pymongo import MongoClient
        c = MongoClient(uri, serverSelectionTimeoutMS=3000, connectTimeoutMS=3000)
        c.admin.command("ping")
        return c[os.getenv("MONGODB_DB", "iz")]["shop_genomes"]
    except Exception as e:  # noqa: BLE001
        log.info("Atlas unreachable, skipping shop_genomes: %s", e)
        return None


def _brief(rs: list[dict]) -> list[dict]:
    return [{"seed": r.get("seed"), "status": r.get("status"), "failure": r.get("failure")} for r in rs]


# ---------------------------------------------------------------- main loop
def run_learning(generations: int = 5, train_seeds=range(0, 8), heldout_seeds=range(100, 112),
                 safety_seeds=range(200, 206), llm=None, out_dir="out", run_id: str | None = None,
                 on_generation=None) -> list[dict]:
    train_seeds, heldout_seeds, safety_seeds = list(train_seeds), list(heldout_seeds), list(safety_seeds)
    out = Path(out_dir)
    run_id = run_id or "shop-" + uuid.uuid4().hex[:8]
    ev = _Events(run_id, out / "events.jsonl")
    if llm is None:
        from iz.llm.client import make_llm
        llm = make_llm()
    col = _atlas_collection()
    params = default_params()
    genome_id = f"{run_id}-g0"

    def store(gid, pid, p, gen):
        if col is None:
            return
        try:
            col.replace_one({"_id": gid}, {"_id": gid, "parent": pid, "gen": gen, "params": p,
                                           "run_id": run_id, "ts": datetime.now(timezone.utc)}, upsert=True)
        except Exception as e:  # noqa: BLE001
            log.info("Atlas store failed: %s", e)

    store(genome_id, None, params, 0)
    tr = evaluate(params, train_seeds)
    ho = evaluate(params, heldout_seeds)
    history = [{"gen": 0, "params": dict(params), "train_rate": sum(map(_ok, tr)) / max(1, len(tr)),
                "heldout_rate": sum(map(_ok, ho)) / max(1, len(ho)),
                "heldout_success": sum(map(_ok, ho)), "heldout_n": len(ho),
                "verdict": "BASELINE", "reason": "naive default params", "edits": {}, "rationale": "",
                "heldout_results": _brief(ho)}]
    ev.emit("metrics", {"gen": 0, "train_rate": history[0]["train_rate"],
                        "heldout_rate": history[0]["heldout_rate"], "params": params})
    out.mkdir(parents=True, exist_ok=True)
    (out / "shop_history.json").write_text(json.dumps(history, indent=2, default=str), encoding="utf-8")
    feedback = None
    for gen in range(1, generations + 1):
        # The fixer only ever sees TRAIN takeovers; held-out seeds stay unseen (verifier fix:
        # previously fell back to held-out failure records when train was fully solved).
        fails = [r for r in tr if not _ok(r)]
        for r in tr:
            if not _ok(r):
                ev.emit("takeover", {"gen": gen, "seed": r.get("seed"), "failure": r.get("failure"),
                                     "status": r.get("status"), "operator_tip": r.get("operator_tip"),
                                     "summary": r.get("summary")})
        prop = propose(llm, params, fails, feedback)
        ev.emit("fix_proposed", {"gen": gen, "edits": prop["edits"], "rationale": prop["rationale"],
                                 "prediction": prop["prediction"], "refused": prop["refused"]})
        cand = {**params, **prop["edits"]}
        if cand != params:
            g = gate(params, cand, train_seeds, heldout_seeds, safety_seeds)
        else:
            g = {"verdict": "REJECT", "reason": "no effective edits proposed",
                 "train": {"before": sum(map(_ok, tr)), "after": sum(map(_ok, tr)), "n": len(tr)},
                 "heldout": {"before": sum(map(_ok, ho)), "after": sum(map(_ok, ho)), "n": len(ho)},
                 "safety_violations": 0, "regressions": 0, "heldout_gain": 0}
        after = g.pop("_after", None)
        ev.emit("gate_result", {"gen": gen, **g})
        if g["verdict"] == "PROMOTE":
            parent_id, genome_id = genome_id, f"{run_id}-g{gen}"
            params = cand
            tr, ho = after["train"], after["heldout"]
            store(genome_id, parent_id, params, gen)
            ev.emit("promoted", {"gen": gen, "genome_id": genome_id, "parent": parent_id,
                                 "params": params, "edits": prop["edits"]})
        # feedback to the fixer carries no held-out numbers (only which gate clause fired)
        clause = ("safety" if g["safety_violations"] else "regression" if g["regressions"]
                  else "no_heldout_gain" if g["verdict"] == "REJECT" else "promoted")
        feedback = {"gen": gen, "edits": prop["edits"], "verdict": g["verdict"], "clause": clause,
                    "train": g["train"]}
        h = {"gen": gen, "params": dict(params), "train_rate": sum(map(_ok, tr)) / max(1, len(tr)),
             "heldout_rate": sum(map(_ok, ho)) / max(1, len(ho)),
             "heldout_success": sum(map(_ok, ho)), "heldout_n": len(ho),
             "verdict": g["verdict"], "reason": g["reason"], "edits": prop["edits"],
             "rationale": prop["rationale"], "gate": g, "heldout_results": _brief(ho)}
        history.append(h)
        ev.emit("metrics", {"gen": gen, "train_rate": h["train_rate"], "heldout_rate": h["heldout_rate"],
                            "params": params})
        (out / "shop_history.json").write_text(json.dumps(history, indent=2, default=str), encoding="utf-8")
        if on_generation:
            on_generation(h)
    return history


# ---------------------------------------------------------------- offline fixer (--fake-llm)
def fake_llm():
    """Deterministic rule-based stand-in for the LLM fixer: reads FAILURE_COUNTS and the current
    params from the prompt and tightens the matching parameter one step at a time."""
    from iz.llm.client import FakeLLM

    def policy(model, system, user, schema):
        params = json.loads(user.split("CURRENT_PARAMS: ", 1)[1].split("\n", 1)[0])
        counts = json.loads(user.split("FAILURE_COUNTS: ", 1)[1].split("\n", 1)[0])
        e: dict[str, float] = {}
        why = []
        if counts.get("cart_tipped"):
            e["turn_slowdown_deg"] = 30.0
            e["turn_speed"] = round(max(0.5, params["turn_speed"] - 0.4), 2)
            e["cruise_speed"] = round(max(0.6, params["cruise_speed"] - 0.3), 2)
            why.append("cart tipped on turns: slow before turns > 30 deg")
        if counts.get("spill_slide"):
            e["spill_speed"] = round(max(0.3, params["spill_speed"] - 0.5), 2)
            why.append("cart slid on spill: cap spill speed")
        if counts.get("too_close_to_person") or counts.get("person_contact"):
            e["yield_distance_m"] = round(min(1.5, params["yield_distance_m"] + 0.4), 2)
            why.append("too close to shopper: yield earlier")
        if counts.get("collision_shelf"):
            e["brake_distance_m"] = round(min(1.2, params["brake_distance_m"] + 0.3), 2)
            e["push_force_limit_n"] = round(max(80.0, params["push_force_limit_n"] - 40), 1)
            why.append("shelf collision: brake earlier")
        if counts.get("timeout") and not e:
            e["cruise_speed"] = round(min(1.6, params["cruise_speed"] + 0.1), 2)
            why.append("timeouts: allow slightly faster cruise")
        return {"edits": e, "rationale": "; ".join(why) or "no actionable failures",
                "prediction": {"train_fixed": sum(counts.values()), "heldout_fixed": max(1, sum(counts.values()) // 2)}}
    return FakeLLM(policy=policy)
