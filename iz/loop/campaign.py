"""Long-horizon campaign runner: checkpointed, idempotent, resumable state machine.

Steps per (day, shift): missions -> fix_cycle -> [consolidate after last shift] -> [coach every N days].
Everything runs locally (LocalMemory, out/checkpoints/<run_id>.json, out/events.jsonl) and auto-uses
Atlas when a 3 s ping succeeds (memory, promotion transaction, steps/metrics mirrors).
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import re
import tempfile
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

from iz.agent.gateway import Gateway, make_gateway_factory
from iz.agent.robot import Robot
from iz.harness import dispatcher
from iz.harness.context import build_context_text
from iz.harness.genome import apply_edits, rules_for
from iz.harness.guardrails import validate_candidate_locally, validate_edits
from iz.loop.fixer import SEED_PLAYBOOK, Fixer, cluster_takeovers, placebo
from iz.loop.gate import Gate, gate_fixer
from iz.loop.replay import make_replay_fn
from iz.store.memory import LocalMemory
from iz.types import (Action, Candidate, ContextBundle, Decision, Edit, FixerPlaybook, GateResult, Genome,
                      Outcome, OutcomeStatus, Prediction, Scope, SetScore, Situation, Takeover)
from iz.world import generator
from iz.world.sites import SITES, get_site

log = logging.getLogger("iz.loop.campaign")

ROOT = Path(__file__).resolve().parents[2]
N_ROBOTS = 6
TOP_RULES = 2


# ------------------------------------------------------------------ small utils

def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def atlas_reachable(timeout_s: float = 3.0) -> bool:
    """3 s ping that does not use the shared 8 s client."""
    try:
        uri = os.environ.get("MONGODB_URI")
        if not uri:
            return False
        from pymongo import MongoClient
        c = MongoClient(uri, serverSelectionTimeoutMS=int(timeout_s * 1000),
                        connectTimeoutMS=int(timeout_s * 1000))
        try:
            c.admin.command("ping")
        finally:
            c.close()
        return True
    except Exception:  # noqa: BLE001
        return False


class Emitter:
    """UI event stream: out/events.jsonl (+ optional callback). Ids continue across resumes."""

    def __init__(self, run_id: str, path: Path, seq: int = 0, cb: Callable[[dict], None] | None = None):
        self.run_id, self.path, self.seq, self.cb = run_id, path, seq, cb
        self._lock = threading.Lock()
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(path, "a", encoding="utf-8")

    def __call__(self, kind: str, payload: dict) -> dict:
        with self._lock:
            self.seq += 1
            ev = {"id": f"{self.run_id}-{self.seq:06d}",
                  "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                  "kind": kind, "run_id": self.run_id, "payload": payload}
            try:
                self._fh.write(json.dumps(ev, default=str) + "\n")
                self._fh.flush()
            except (OSError, ValueError):
                pass
        if self.cb:
            try:
                self.cb(ev)
            except Exception:  # noqa: BLE001
                pass
        return ev

    def ticker(self, feature: str, text: str) -> None:
        self("ticker", {"feature": feature, "text": text})


# ------------------------------------------------------------------ fake LLM (offline demo / tests)

_TIP_N = re.compile(r"about (\d+) N")
_CALL = re.compile(r"call (\w+) instead")


def fake_llm():
    """Deterministic offline LLM: robot follows rules if any are shown; fixer writes one rule per cluster."""
    from iz.llm.client import FakeLLM

    def informed(user: str, ty: str) -> list[dict]:
        if ty == "fire_door":
            return [{"tool": "request_open", "args": {}}, {"tool": "wait", "args": {"seconds": 12}}]
        if ty == "office_door":
            m = _TIP_N.search(user)
            return [{"tool": "push_door", "args": {"force_n": float(m.group(1)) if m else 60.0}}]
        if ty in ("cart_blocking", "construction_detour", "restricted_shortcut"):
            return [{"tool": "reroute", "args": {}}]
        if ty == "crowd_blocking":
            return [{"tool": "yield", "args": {}}, {"tool": "wait", "args": {"seconds": 15}}]
        if ty == "elevator_out_of_service":
            m = _CALL.search(user)
            return [{"tool": "call_elevator", "args": {"elevator": m.group(1) if m else "E1"}}]
        if ty == "low_battery":
            return [{"tool": "reroute", "args": {"via": "CHG"}}]
        if ty == "person_down":
            return [{"tool": "page_staff", "args": {}}]
        return [{"tool": "slow_down", "args": {}}]

    def policy(model, system, user, schema):
        if "coach" in system.lower():
            return {"instructions": "Group takeovers, propose minimal scoped rules; prefer narrow scope.",
                    "max_edits": 3, "min_evidence": 2, "lessons": ["fake coach: keep rules narrow"]}
        if "Takeover clusters:" in user:
            try:
                blob = user.split("Takeover clusters:\n", 1)[1].split("\n\nCurrent rulebook", 1)[0]
                clusters = json.loads(blob)
            except Exception:  # noqa: BLE001
                clusters = []
            edits, seen = [], set()
            for c in clusters:
                ty = c["situation_type"]
                scope = {"situation_type": ty}
                if ty in ("elevator_out_of_service", "office_door", "fire_door"):
                    scope["site"] = c["site"]
                key = json.dumps(scope, sort_keys=True)
                if key in seen:
                    continue
                seen.add(key)
                tip = (c.get("operator_tips") or ["Follow the operator tip."])[0]
                if ty == "restricted_shortcut":
                    tip = "Never use the S-2 shortcut; reroute along the normal path."
                edits.append({"kind": "add_rule", "scope": scope, "payload": {"text": tip},
                              "rationale": f"operator tip for {ty}",
                              "prediction": {"train_fixed": c["count"], "heldout_fixed": 1}})
            return {"edits": edits}
        if "SITUATION:" in user:
            ty = user.split("SITUATION:", 1)[1].split("\n", 1)[0].strip()
            rules = [ln for ln in user.split("RULES:", 1)[1].splitlines() if ln.startswith("- [")]                 if "RULES:" in user else []
            if any("Keep a log of observed conditions" not in ln for ln in rules):
                return {"actions": informed(user, ty)}
            from iz.world.naive import NAIVE_TABLE
            return {"actions": [{"tool": t, "args": dict(a)} for t, a in NAIVE_TABLE.get(ty, [])]}
        return {}

    return FakeLLM(policy=policy)


# ------------------------------------------------------------------ local validator mirror

def _local_violations(g: Genome) -> list[str]:
    out = list(validate_candidate_locally(g))
    for r in g.rulebook:
        if r.status == "active" and "human_only" in ((r.params or {}).get("allow_zone_kinds") or []):
            out.append(f"Document failed validation: rulebook {r.id} params.allow_zone_kinds contains "
                       "'human_only' (not in enum)")
    return out


# ------------------------------------------------------------------ the campaign

def seed_genome() -> Genome:
    """v0 harness: a few global safety-oriented base rules (the regression bank must be solvable by the parent)."""
    from iz.types import RuleEntry
    rules = [
        RuleEntry(id="base-cart", scope=Scope(situation_type="cart_blocking"),
                  text="A cart with a red tag must never be pushed: reroute around it. Otherwise probe_push first."),
        RuleEntry(id="base-person", scope=Scope(situation_type="person_down"),
                  text="Person down: page_staff immediately and wait for staff."),
        RuleEntry(id="base-shortcut", scope=Scope(situation_type="restricted_shortcut"),
                  text="Never enter human_only zones such as S-2: reroute via the normal path."),
    ]
    return Genome(version=0, rulebook=rules)


class Campaign:
    def __init__(self, run_id, days, shifts, missions_per_shift, sites, seed, drift_day, llm, robot_fn, judge,
                 demo_traps, coach_every, skills, stop_after, out_dir, use_atlas, emit_cb, placebo_on):
        self.run_id, self.days, self.shifts, self.mps = run_id, days, shifts, missions_per_shift
        self.sites, self.seed, self.drift_day = tuple(sites), seed, drift_day
        self.demo_traps, self.coach_every, self.skills = demo_traps, coach_every, skills
        self.stop_after, self.placebo_on = stop_after, placebo_on
        self.out_dir = Path(out_dir)
        self.ck_path = self.out_dir / "checkpoints" / f"{run_id}.json"
        if llm is None:
            from iz.llm.client import make_llm
            llm = make_llm()
        self.llm = llm
        self.robot_fn = robot_fn
        if judge is None:
            from iz.world.judge import get_judge
            judge = get_judge("physics")
        self.judge = judge
        self.kinds = {z.id: z.kind for s in SITES.values() for z in s.zones}
        self.robot = Robot()
        self._mcache: dict[tuple, list] = {}
        self.cost0 = self._spent()

        # ---- Atlas (optional)
        self.db = None
        self.store = None
        if use_atlas is None:
            # never write fake-LLM data into the shared Atlas unless explicitly asked to
            use_atlas = False if type(llm).__name__ == "FakeLLM" else atlas_reachable()
        if use_atlas:
            try:
                from iz.store.db import get_db
                from iz.store.genome_store import GenomeStore
                self.db = get_db()
                self.store = GenomeStore(self.db)
            except Exception as e:  # noqa: BLE001
                log.warning("Atlas unavailable, local fallback: %s", str(e)[:120])
                self.db = self.store = None

        # ---- state (load or fresh)
        self.state = self._load()
        self.emit = Emitter(run_id, self.out_dir / "events.jsonl", self.state["event_seq"], emit_cb)
        self.genome = Genome.model_validate(self.state["genome"])
        self.playbook = FixerPlaybook.model_validate(self.state["playbook"])
        self.takeovers = [Takeover.model_validate(t) for t in self.state["takeovers"]]
        self.memory = self._make_memory()
        self.steps_executed = 0

    # ---------------- state
    def _fresh(self) -> dict:
        g = seed_genome()
        if self.store is not None:
            try:
                g = self.store.get_active()
            except KeyError:
                try:
                    self._atlas_seed(g)
                except Exception:  # noqa: BLE001
                    pass
            except Exception:  # noqa: BLE001
                pass
        return {"run_id": self.run_id, "done": [], "genome": g.model_dump(mode="json"),
                "playbook": SEED_PLAYBOOK.model_dump(mode="json"), "takeovers": [], "mission_log": [],
                "decisions": 0, "situations": 0, "context_tokens": 0, "promotions": [], "rejections": [],
                "refusals": [], "decision_log": [], "promoted_keys": [], "drift": None, "cycle": 0,
                "last_cycle_idx": 0, "trap_done": False, "event_seq": 0, "cost_usd": 0.0,
                "consolidations": [], "coach_log": [], "skills_used": []}

    def _atlas_seed(self, g: Genome) -> None:
        from iz.store import db as D
        now = datetime.now(timezone.utc)
        self.db[D.GENOMES].insert_one({**g.model_dump(mode="json"), "_id": g.version, "created_at": now})
        self.db[D.ACTIVE].update_one({"_id": "robot_genome"}, {"$set": {"version": g.version, "updated_at": now}},
                                     upsert=True)

    def _load(self) -> dict:
        if self.ck_path.exists():
            try:
                return json.loads(self.ck_path.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                log.warning("corrupt checkpoint for %s; starting fresh", self.run_id)
        return self._fresh()

    def _save(self) -> None:
        s = self.state
        s["genome"] = self.genome.model_dump(mode="json")
        s["playbook"] = self.playbook.model_dump(mode="json")
        s["takeovers"] = [t.model_dump(mode="json") for t in self.takeovers]
        s["event_seq"] = self.emit.seq
        s["cost_usd"] = round(s.get("cost_usd", 0.0) + max(0.0, self._spent() - self.cost0), 6)
        self.cost0 = self._spent()
        _atomic_write(self.ck_path, json.dumps(s, default=str))

    @staticmethod
    def _spent() -> float:
        try:
            from iz.obs.budget import get_budget
            return float(get_budget().spent)
        except Exception:  # noqa: BLE001
            return 0.0

    def _make_memory(self):
        if self.db is not None:
            try:
                from iz.store.memory import AtlasMemory
                return AtlasMemory(self.db)
            except Exception:  # noqa: BLE001
                pass
        mem = LocalMemory()
        for t in self.takeovers:
            mem.add_takeover(t)
        for r in self.genome.rulebook:
            if r.status == "active":
                mem.add_rule(r, r.scope.site)
        return mem

    # ---------------- step bookkeeping
    def key(self, day: int, shift: int, step: str) -> str:
        return f"{self.run_id}:{day}:{shift}:{step}"

    def _mark_done(self, key: str) -> None:
        if key not in self.state["done"]:
            self.state["done"].append(key)
        self._save()
        if self.db is not None:
            try:
                from iz.store.db import STEPS
                self.db[STEPS].create_index("key", unique=True)
                self.db[STEPS].update_one({"key": key}, {"$setOnInsert": {"key": key, "run_id": self.run_id,
                                                                          "ts": datetime.now(timezone.utc)}},
                                          upsert=True)
            except Exception:  # noqa: BLE001
                pass

    # ---------------- helpers
    def _site_for(self, site: str, day: int) -> str:
        return "A_reno" if (site == "A" and day >= self.drift_day) else site

    def _missions(self, site: str, day: int, shift: int):
        k = (site, day, shift)
        if k not in self._mcache:
            self._mcache[k] = generator.generate_missions(self.run_id, site, day, shift, self.seed, self.mps)
        return self._mcache[k]

    def _situation_for(self, t: Takeover) -> Situation | None:
        for m in self._missions(t.site, t.day, t.shift):
            if m.id == t.mission_id:
                for s in m.situations:
                    if s.id == t.situation_id:
                        return s
        return None

    def _sim_ts(self, day: int, shift: int) -> datetime:
        return datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(days=day - 1, hours=6 * shift)

    def _decide(self, view, genome: Genome, memory) -> tuple[list[Action], ContextBundle]:
        """Robot decision with MINIMAL context: top rules, 1 episode, 1 fact."""
        top = rules_for(genome, view)[:TOP_RULES]
        ctx = ContextBundle(backend="local", rules=top)
        if memory is not None:
            try:
                got = memory.get_context(view, genome)
                ctx = got.model_copy(update={"rules": top, "episodes": list(got.episodes)[:1],
                                             "facts": list(got.facts)[:1]})
            except Exception:  # noqa: BLE001
                pass
        if self.robot_fn is not None:
            try:
                return self.robot_fn(view, genome), ctx
            except Exception:  # noqa: BLE001
                return [], ctx
        return self.robot.decide(view, genome, ctx, self.llm), ctx

    def _replay_fn(self):
        def fn(view, genome):
            return self._decide(view, genome, None)[0]
        return make_replay_fn(fn, self.judge, make_gateway_factory(self.kinds))

    # ---------------- missions step
    def step_missions(self, day: int, shift: int) -> None:
        genome, mem = self.genome, self.memory
        rng = random.Random(int(hashlib.sha256(
            f"{self.run_id}:{self.seed}:{day}:{shift}".encode()).hexdigest()[:8], 16))
        robots = [dispatcher.RobotState(id=f"R{i + 1}", battery_pct=round(rng.uniform(20, 100), 1), xy=(0.0, 0.0))
                  for i in range(N_ROBOTS)]
        batch = []   # (logical_site, actual_site, mission, robot_id)
        for lsite in self.sites:
            asite = self._site_for(lsite, day)
            site = get_site(asite)
            zxy = {z.id: z.xy for z in site.zones}
            ms = self._missions(asite, day, shift)
            assign = dispatcher.assign(
                [{"id": m.id, "kind": m.kind, "xy": zxy.get(m.route[0], (0.0, 0.0))} for m in ms],
                robots, genome.dispatcher, hour=6 + 6 * shift, peak_hours=site.peak_hours)
            for r in robots:
                r.load += sum(1 for m in ms if assign.get(m.id) == r.id)
            for m in ms:
                batch.append((lsite, asite, m, assign.get(m.id, m.robot_id)))

        work = [(bi, s) for bi, (_, _, m, _) in enumerate(batch) for s in m.situations]

        def run_one(item):
            _, sit = item
            view = sit.view()
            actions, ctx = self._decide(view, genome, mem)
            blocked: list[str] = []
            try:
                actions, blocked = Gateway(genome).filter(view, actions, self.kinds)
            except Exception:  # noqa: BLE001
                actions = []
            try:
                out = self.judge.judge(sit, actions)
            except Exception as e:  # noqa: BLE001
                out = Outcome(status=OutcomeStatus.TAKEOVER, reason=f"judge_error: {type(e).__name__}")
            return actions, ctx, blocked, out

        with ThreadPoolExecutor(max_workers=8) as ex:
            results = list(ex.map(run_one, work))   # order == work order -> deterministic
        by_mission: dict[int, list] = {}
        for (bi, sit), res in zip(work, results):
            by_mission.setdefault(bi, []).append((sit, res))

        from iz.obs import rd
        from iz.store import metrics as store_metrics
        shift_rows: dict[str, list[bool]] = {}
        for bi, (lsite, asite, m, rid) in enumerate(batch):
            self.emit("mission_start", {"mission_id": m.id, "robot_id": rid, "site": asite, "day": day,
                                        "shift": shift, "kind": m.kind, "route": m.route})
            zxy = {z.id: z.xy for z in get_site(asite).zones}
            self.emit("robot_move", {"robot_id": rid, "site": asite, "zone": m.route[0],
                                     "xy": list(zxy.get(m.route[0], (0.0, 0.0)))})
            items = by_mission.get(bi, [])
            handle = rd.mission_begin(rid, asite, day, shift, f"{m.kind} mission {m.id}", {
                "policy_version": genome.version, "run_id": self.run_id, "model": self.robot.model,
                "situation_type": items[0][0].type if items else ""})
            m_free = True
            for sit, (actions, ctx, blocked, out) in items:
                self.state["situations"] += 1
                self.state["decisions"] += 1
                tokens = len(build_context_text(ctx, genome.context_policy.token_budget)) // 4
                self.state["context_tokens"] += tokens
                self.emit("robot_move", {"robot_id": rid, "site": asite, "zone": sit.zone,
                                         "xy": list(zxy.get(sit.zone, (0.0, 0.0)))})
                self.emit("situation", {"robot_id": rid, "site": asite, "zone": sit.zone, "type": sit.type,
                                        "observable": sit.observable})
                self.emit("decision", {"robot_id": rid, "actions": [a.model_dump() for a in actions],
                                       "rules_used": [r.id for r in ctx.rules], "context_hits": ctx.hits,
                                       "context_tokens": tokens})
                for a in actions:
                    handle.tool(a.tool, a.args, {"status": out.status.value})
                payload = {"robot_id": rid, "mission_id": m.id, "status": out.status.value, "reason": out.reason,
                           "judged_by": out.judged_by}
                if out.clip_path:
                    payload["clip"] = out.clip_path
                self.emit("outcome", payload)
                if blocked:
                    rd.signal(handle.event_id, "guardrail_blocked", {"blocked": blocked[:3]})
                if out.unsafe:
                    rd.signal(handle.event_id, "safety_violation", {"reason": out.reason})
                if not out.takeover_free:
                    m_free = False
                    t = Takeover(id=f"to-{m.id}-{sit.id}", run_id=self.run_id, mission_id=m.id, robot_id=rid,
                                 site=asite, day=day, shift=shift, situation_id=sit.id, situation_type=sit.type,
                                 zone=sit.zone, zone_class=sit.zone_class, observable=sit.observable,
                                 tried=actions, outcome=out.status, reason=out.reason,
                                 operator_tip=sit.hidden.operator_tip, policy_version=genome.version,
                                 ts=self._sim_ts(day, shift))
                    self.takeovers.append(t)
                    try:
                        mem.add_takeover(t)
                    except Exception:  # noqa: BLE001
                        pass
                    self.emit("takeover", {"robot_id": rid, "site": asite, "type": sit.type,
                                           "operator_tip": t.operator_tip})
                    rd.signal(handle.event_id, "takeover", {"situation_type": sit.type})
                    if out.status == OutcomeStatus.TAKEOVER:
                        rd.signal(handle.event_id, "takeover_avoidable", {"situation_type": sit.type})
            handle.finish("ok" if m_free else "takeover")
            self.state["mission_log"].append({"site": lsite, "variant": asite, "day": day, "shift": shift,
                                              "takeover_free": bool(m_free), "policy_version": genome.version})
            shift_rows.setdefault(lsite, []).append(m_free)
            if self.db is not None:
                try:
                    store_metrics.record_mission(
                        self.run_id, lsite, day, shift, genome.version, m_free,
                        takeovers=sum(1 for _, r in items if not r[3].takeover_free),
                        safety_violations=sum(1 for _, r in items if r[3].unsafe), robot_id=rid,
                        ts=self._sim_ts(day, shift), db=self.db)
                except Exception:  # noqa: BLE001
                    pass

        for lsite, rows in shift_rows.items():
            hist = [h["takeover_free"] for h in self.state["mission_log"] if h["site"] == lsite][-30:]
            self.emit("metrics", {"day": day, "shift": shift, "site": lsite,
                                  "tfmr": round(sum(rows) / len(rows), 4),
                                  "window_tfmr": round(sum(hist) / len(hist), 4) if hist else 0.0,
                                  "policy_version": genome.version,
                                  "cost_usd": round(self.state["cost_usd"] + max(0.0, self._spent() - self.cost0), 6)})
        self._drift(day, shift)

    def _drift(self, day: int, shift: int) -> None:
        if self.state["drift"]:
            return
        try:
            from iz.loop import drift
        except Exception:  # noqa: BLE001
            return
        for lsite in self.sites:
            try:
                res = drift.check(self.state["mission_log"], lsite)
            except Exception:  # noqa: BLE001
                continue
            if res and res.get("alarm"):
                since = sum(1 for h in self.state["mission_log"]
                            if h["site"] == lsite and h["day"] >= self.drift_day)
                info = {"site": lsite, "window_tfmr": res.get("window_tfmr"),
                        "baseline_tfmr": res.get("baseline_tfmr"), "p_value": res.get("p_value"),
                        "missions_since_change": since, "day": day, "shift": shift}
                self.state["drift"] = info
                self.emit("drift_alarm", {k: info[k] for k in ("site", "window_tfmr", "baseline_tfmr", "p_value",
                                                              "missions_since_change")})
                break

    # ---------------- fix cycle
    @staticmethod
    def _gate_obj(g: GateResult) -> dict:
        return {"train": {"before": g.train_before.takeover_free, "after": g.train_after.takeover_free,
                          "n": g.train_after.n},
                "heldout": {"before": g.heldout_before.takeover_free, "after": g.heldout_after.takeover_free,
                            "n": g.heldout_after.n},
                "regression": {"n": g.regression_after.n, "passed": g.regression_after.takeover_free},
                "safety_violations": g.safety_violations}

    @staticmethod
    def _zero_gate(cid: str, parent: int, verdict: str, reason: str) -> GateResult:
        z = SetScore(n=0, takeover_free=0, safety_violations=0)
        return GateResult(candidate_id=cid, parent_version=parent, train_before=z, train_after=z,
                          heldout_before=z, heldout_after=z, regression_after=z, regressions=0,
                          safety_violations=0, heldout_gain=0, verdict=verdict, reason=reason)

    def _decision(self, key: str, cand: Candidate, gate: GateResult, new_version: int | None) -> Decision:
        pred = Prediction(train_fixed=sum(e.prediction.train_fixed for e in cand.edits),
                          heldout_fixed=sum(e.prediction.heldout_fixed for e in cand.edits))
        actual = Prediction(train_fixed=gate.train_after.takeover_free - gate.train_before.takeover_free,
                            heldout_fixed=gate.heldout_gain)
        d = Decision(id=f"dec-{self.run_id}-{cand.id}", run_id=self.run_id, candidate_id=cand.id,
                     parent_version=cand.parent_version, new_version=new_version, verdict=gate.verdict,
                     reason=gate.reason, gate=gate, predicted=pred, actual=actual,
                     ts=datetime.now(timezone.utc))
        self.state["decision_log"].append({"step": key, "candidate_id": cand.id, "verdict": gate.verdict,
                                           "reason": gate.reason, "predicted": pred.model_dump(),
                                           "actual": actual.model_dump()})
        return d

    def _emit_proposal(self, cand: Candidate) -> None:
        self.emit("fix_proposed", {"candidate_id": cand.id, "author": cand.author, "edits": [
            {"kind": e.kind, "scope": e.scope.model_dump(),
             "text_or_change": (e.payload.get("text") if e.kind == "add_rule" else json.dumps(e.payload)),
             "rationale": e.rationale, "prediction": e.prediction.model_dump()} for e in cand.edits]})

    def _db_probe(self, g: Genome) -> str | None:
        """Ask the real Mongo validator without side effects (insert inside a txn, then abort)."""
        if self.db is None:
            return None
        try:
            from pymongo.errors import OperationFailure
            from iz.store import db as D
            from iz.store.genome_store import VALIDATION_FAILED
            doc = {**g.model_dump(mode="json"), "_id": 10**9 + g.version}
            with D.get_client().start_session() as s:
                s.start_transaction()
                try:
                    self.db[D.GENOMES].insert_one(doc, session=s)
                except OperationFailure as e:
                    if e.code == VALIDATION_FAILED:
                        return (e.details or {}).get("errmsg") or "Document failed validation"
                    raise
                finally:
                    try:
                        s.abort_transaction()
                    except Exception:  # noqa: BLE001
                        pass
        except Exception:  # noqa: BLE001
            return None
        return None

    def _refuse(self, key: str, cand: Candidate, g: Genome, violations: list[str]) -> None:
        db_msg = self._db_probe(g)
        msg, src = (db_msg, "db") if db_msg else (violations[0], "local")
        self.emit("validator_refused", {"candidate_id": cand.id, "message": msg})
        self.emit.ticker("validator", f"{src} validator refused {cand.id}: {msg[:90]}")
        self.state["refusals"].append({"candidate_id": cand.id, "message": msg, "step": key, "source": src,
                                       "author": cand.author})
        gate = self._zero_gate(cand.id, cand.parent_version, "REJECT", f"validator refused: {msg[:120]}")
        try:
            d = self._decision(key, cand, gate, None)
            if self.store is not None:
                self.store.record_decision(d)
        except Exception:  # noqa: BLE001
            pass

    def _promote(self, key: str, cand: Candidate, g: Genome, gate: GateResult) -> bool:
        if key in self.state["promoted_keys"]:
            return False
        dec = self._decision(key, cand, gate, g.version)
        if self.store is not None:
            try:
                self.store.promote(g, dec)
            except Exception as e:  # noqa: BLE001
                from iz.store.genome_store import ValidatorRefusal
                if isinstance(e, ValidatorRefusal):
                    self.emit("validator_refused", {"candidate_id": cand.id, "message": e.message})
                    self.emit.ticker("validator", f"db validator refused {cand.id}")
                    self.state["refusals"].append({"candidate_id": cand.id, "message": e.message, "step": key,
                                                   "source": "db", "author": cand.author})
                    return False
                if "E11000" not in str(e) and "duplicate key" not in str(e).lower():
                    log.warning("atlas promote failed (%s); promoting locally", str(e)[:120])
        old = {r.id for r in self.genome.rulebook}
        self.genome = g
        for r in g.rulebook:
            if r.status == "active" and r.id not in old:
                try:
                    self.memory.add_rule(r, r.scope.site)
                except Exception:  # noqa: BLE001
                    pass
        self.state["promoted_keys"].append(key)
        self.state["promotions"].append({"version": g.version, "parent": cand.parent_version,
                                         "candidate_id": cand.id, "step": key, "author": cand.author})
        self.emit("promoted", {"version": g.version, "parent": cand.parent_version, "candidate_id": cand.id})
        self.emit("reload", {"robots": N_ROBOTS, "version": g.version})
        self.emit.ticker("change_stream" if self.store else "transaction",
                         f"{N_ROBOTS} robots reloaded genome v{g.version}")
        self._save()
        return True

    def _raindrop_summary(self, new: list[Takeover]) -> str:
        from iz.obs import rd
        if rd.enabled():
            try:
                from iz.obs import rd_query
                r = rd_query.top_takeover_causes(self._sim_ts(1, 1).replace(year=2000))
                if not r.get("degraded"):
                    return "Raindrop top takeover causes: " + json.dumps(r.get("causes", [])[:5])
            except Exception:  # noqa: BLE001
                pass
            self.emit.ticker("raindrop_query", "raindrop degraded; using local counts")
        c = Counter((t.situation_type, t.site) for t in new)
        return "Local takeover counts: " + json.dumps([{"type": k[0], "site": k[1], "count": n}
                                                       for k, n in c.most_common(5)])

    def _skill_edits(self, clusters: list[dict]) -> list[Edit]:
        if not self.skills:
            return []
        try:
            from iz.skills.distill import skills_for_clusters
        except Exception:  # noqa: BLE001
            return []
        try:
            eds = list(skills_for_clusters(clusters, self.llm))[:2]
        except Exception as e:  # noqa: BLE001
            log.warning("skills_for_clusters failed: %s", str(e)[:120])
            return []
        out = []
        for i, e in enumerate(eds):
            if validate_edits([e]):
                continue
            rat = e.rationale if e.rationale.startswith("video skill:") else f"video skill: {e.rationale}"
            out.append(e.model_copy(update={"id": f"s{i}", "rationale": rat}))
        return out

    def step_fix_cycle(self, day: int, shift: int) -> None:
        key = self.key(day, shift, "fix_cycle")
        s = self.state
        new = self.takeovers[s["last_cycle_idx"]:]
        has_shortcut = any(t.situation_type == "restricted_shortcut" for t in new)
        if self.demo_traps and not s["trap_done"] and (has_shortcut or (day == 1 and shift == self.shifts)):
            self._trap(key)
            s["trap_done"] = True
        if len(new) < self.playbook.min_evidence:
            return
        s["cycle"] += 1
        cycle = s["cycle"]
        from iz.obs import rd
        h = rd.cycle_begin("fix_cycle", self.run_id, f"fix cycle {cycle}",
                           {"policy_version": self.genome.version, "model": "fixer", "day": day, "shift": shift})
        clusters = cluster_takeovers(new)
        if s["drift"]:
            ds = s["drift"]["site"]
            clusters.sort(key=lambda c: 0 if c["site"].startswith(ds) else 1)   # stable: drifted site first
        cand = Fixer(self.llm, self.playbook).propose(clusters, self._raindrop_summary(new), self.genome)
        sk = self._skill_edits(clusters)
        if sk:
            cand = cand.model_copy(update={"edits": cand.edits + sk})
            s["skills_used"] += [e.rationale for e in sk]
        s["last_cycle_idx"] = len(self.takeovers)
        if not cand.edits:
            self.emit.ticker("validator", f"fixer produced no edits for cycle {cycle}")
            s["rejections"].append({"candidate_id": cand.id, "reason": "no edits proposed", "step": key})
            h.finish("no edits")
            return
        self._emit_proposal(cand)
        g2 = apply_edits(self.genome, cand.edits)
        viol = _local_violations(g2)
        if viol:
            self._refuse(key, cand, g2, viol)
            h.finish("refused")
            return
        train = list({x.id: x for x in (self._situation_for(t) for t in new) if x is not None}.values())
        held = generator.heldout_incidents(self.seed + cycle)
        reg = generator.regression_bank(self.seed + 2)
        pg = apply_edits(self.genome, placebo(cand).edits) if self.placebo_on else None
        gate = Gate().evaluate(g2, self.genome, train, held, reg, self._replay_fn(), placebo_genome=pg,
                               candidate_id=cand.id)
        self.emit("gate_result", {"candidate_id": cand.id, **self._gate_obj(gate), "verdict": gate.verdict,
                                  "reason": gate.reason})
        if gate.verdict == "PROMOTE":
            ok = self._promote(key, cand, g2, gate)
            h.finish("promoted" if ok else "promote failed")
            rd.signal(h.event_id, "edit_promoted", {"candidate": cand.id}, sentiment="POSITIVE")
        else:
            self._decision(key, cand, gate, None)
            s["rejections"].append({"candidate_id": cand.id, "reason": gate.reason, "verdict": gate.verdict,
                                    "step": key})
            h.finish(gate.verdict)
            rd.signal(h.event_id, "edit_rejected", {"candidate": cand.id, "reason": gate.reason})

    def _trap(self, key: str) -> None:
        e = Edit(id="e0", kind="add_rule", scope=Scope(situation_type="restricted_shortcut"),
                 payload={"id": f"naive-tip-v{self.genome.version}",
                          "text": "Staff may cut through S-2 when the corridor is closed.",
                          "params": {"allow_zone_kinds": ["human_only"]}},
                 rationale="naive fixer copies operator tip literally: staff may cut through S-2",
                 prediction=Prediction(train_fixed=1, heldout_fixed=1))
        cand = Candidate(id=f"c-v{self.genome.version}-naive-{self.state['cycle']}",
                         parent_version=self.genome.version, edits=[e], author="human")
        self._emit_proposal(cand)
        g = apply_edits(self.genome, cand.edits, created_by="human")
        viol = _local_violations(g) or ["Document failed validation: allow_zone_kinds contains human_only"]
        self._refuse(key, cand, g, viol)

    # ---------------- consolidate / coach
    def step_consolidate(self, day: int, shift: int) -> None:
        try:
            from iz.loop import consolidate
            new_g, summary = consolidate.run(self.genome, self.takeovers, day)
        except Exception as e:  # noqa: BLE001
            log.info("consolidate skipped: %s", str(e)[:100])
            return
        self.state["consolidations"].append({"day": day, "changed": new_g is not None, "summary": summary})
        if summary:
            for site in self.sites:
                try:
                    self.memory.add_fact(site, summary)
                except Exception:  # noqa: BLE001
                    pass
        if new_g is None:
            return
        if new_g.version <= self.genome.version:
            new_g = new_g.model_copy(update={"version": self.genome.version + 1, "parent": self.genome.version})
        key = self.key(day, shift, "consolidate")
        cand = Candidate(id=f"c-v{self.genome.version}-consolidate-d{day}",
                         parent_version=self.genome.version, edits=[], author="fixer")
        viol = _local_violations(new_g)
        if viol:
            self._refuse(key, cand, new_g, viol)
            return
        self._promote(key, cand, new_g,
                      self._zero_gate(cand.id, cand.parent_version, "PROMOTE", f"consolidation: {summary[:100]}"))

    def step_coach(self, day: int, shift: int) -> None:
        from iz.loop.coach import Coach
        old = self.playbook
        new_pb = Coach(self.llm).evolve(old, list(self.state["decision_log"]))
        recent = self.takeovers[-40:]
        verdict, reason = "REJECT", "no recent takeovers to evaluate on"
        if recent:
            clusters = cluster_takeovers(recent)
            train = list({x.id: x for x in (self._situation_for(t) for t in recent) if x is not None}.values())
            held = generator.heldout_incidents(self.seed + 1000 + day)
            reg = generator.regression_bank(self.seed + 2)
            res = gate_fixer(new_pb, old, clusters, self.genome,
                             lambda pb, cl, g: Fixer(self.llm, pb).propose(cl, None, g),
                             None, Gate(), train, held, reg, self._replay_fn())
            verdict = "PROMOTE" if res["keep_new"] else "REJECT"
            reason = res["reason"]
        if verdict == "PROMOTE":
            self.playbook = new_pb
        changes = {"instructions_changed": new_pb.instructions != old.instructions,
                   "max_edits": [old.max_edits, new_pb.max_edits],
                   "min_evidence": [old.min_evidence, new_pb.min_evidence], "lessons": new_pb.lessons[:3]}
        self.state["coach_log"].append({"day": day, "verdict": verdict, "reason": reason,
                                        "playbook_version": new_pb.version})
        self.emit("coach", {"playbook_version": new_pb.version, "parent": old.version, "verdict": verdict,
                            "reason": reason, "changes": changes})

    # ---------------- driver
    def run(self) -> dict:
        for sid in self.sites:
            self._emit_site(sid)
        stopped = False
        for day in range(1, self.days + 1):
            for shift in range(1, self.shifts + 1):
                steps = [("missions", self.step_missions), ("fix_cycle", self.step_fix_cycle)]
                if shift == self.shifts:
                    steps.append(("consolidate", self.step_consolidate))
                    if self.coach_every and day % self.coach_every == 0:
                        steps.append(("coach", self.step_coach))
                for name, fn in steps:
                    k = self.key(day, shift, name)
                    if k in self.state["done"]:
                        continue
                    if self.stop_after is not None and self.steps_executed >= self.stop_after:
                        stopped = True
                        break
                    fn(day, shift)
                    self.steps_executed += 1
                    self._mark_done(k)
                if stopped:
                    break
            if stopped:
                break
        self._save()
        try:
            self.emit._fh.close()
        except Exception:  # noqa: BLE001
            pass
        return self.summary(stopped)

    def _emit_site(self, sid: str) -> None:
        s = get_site(sid)
        self.emit("site_loaded", {"site": sid, "zones": [{"id": z.id, "kind": z.kind, "xy": list(z.xy)}
                                                          for z in s.zones],
                                  "edges": [{"a": e.a, "b": e.b, "door_id": e.door_id, "door_type": e.door_type}
                                            for e in s.edges]})

    def summary(self, stopped: bool = False) -> dict:
        s = self.state
        per: dict[str, dict[int, list[int]]] = {}
        for h in s["mission_log"]:
            per.setdefault(h["site"], {}).setdefault(h["day"], []).append(int(h["takeover_free"]))
        tf = {site: {str(d): round(sum(v) / len(v), 4) for d, v in sorted(days.items())}
              for site, days in per.items()}
        n = len(s["mission_log"])
        return {"run_id": self.run_id, "missions": n, "situations": s["situations"], "decisions": s["decisions"],
                "takeovers": len(self.takeovers),
                "tfmr_overall": round(sum(h["takeover_free"] for h in s["mission_log"]) / n, 4) if n else None,
                "tfmr_per_day_per_site": tf,
                "avg_context_tokens": round(s["context_tokens"] / s["decisions"], 2) if s["decisions"] else 0.0,
                "promotions": s["promotions"], "rejections": s["rejections"], "refusals": s["refusals"],
                "decision_log": s["decision_log"], "coach": s["coach_log"], "consolidations": s["consolidations"],
                "skills_used": s["skills_used"], "drift": s["drift"], "cost_usd": s["cost_usd"],
                "final_version": self.genome.version, "playbook_version": self.playbook.version,
                "steps_done": len(s["done"]), "steps_executed_this_run": self.steps_executed,
                "atlas": self.db is not None, "stopped_early": stopped}


def run_campaign(run_id: str, days: int = 14, shifts: int = 3, missions_per_shift: int = 10,
                 sites=("A", "B", "C"), seed: int = 0, drift_day: int = 8, llm=None, robot_fn=None, judge=None,
                 demo_traps: bool = True, coach_every: int = 0, skills: bool = True, stop_after: int | None = None,
                 *, out_dir: str | os.PathLike | None = None, use_atlas: bool | None = None,
                 emit: Callable[[dict], None] | None = None, placebo_on: bool = True) -> dict:
    """Run (or resume) a campaign. Same run_id + out_dir resumes at the first undone step."""
    c = Campaign(run_id, days, shifts, missions_per_shift, sites, seed, drift_day, llm, robot_fn, judge,
                 demo_traps, coach_every, skills, stop_after, out_dir or (ROOT / "out"), use_atlas, emit,
                 placebo_on)
    return c.run()
