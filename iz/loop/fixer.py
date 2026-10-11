"""Layer 2: turn clustered takeovers into a candidate harness edit. Never sees held-out data."""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
from typing import Any

from iz.types import (Candidate, Edit, FixerPlaybook, Genome, LockedGuardrails, LLM, Scope,
                      Takeover)

SEED_PLAYBOOK = FixerPlaybook(
    version=0,
    instructions=("Group takeovers, propose minimal scoped rules or tool-policy changes that would have "
                  "prevented them; prefer the narrowest scope; never touch locked guardrails or gate thresholds."),
    max_edits=3, min_evidence=2)

EDIT_KINDS = ["add_rule", "retire_rule", "tool_policy", "dispatcher", "tunable_guardrail",
              "context_policy", "model_routing"]

FIXER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"edits": {"type": "array", "items": {
        "type": "object",
        "properties": {
            "kind": {"type": "string", "enum": EDIT_KINDS},
            "payload": {"type": "object"},
            "rationale": {"type": "string"},
            "scope": {"type": "object", "properties": {
                "site": {"type": ["string", "null"]}, "situation_type": {"type": ["string", "null"]},
                "zone_class": {"type": ["string", "null"]}, "door_id": {"type": ["string", "null"]}}},
            "prediction": {"type": "object", "properties": {
                "train_fixed": {"type": "integer"}, "heldout_fixed": {"type": "integer"}}},
        },
        "required": ["kind", "payload", "rationale", "scope", "prediction"]}}},
    "required": ["edits"],
}


def cluster_takeovers(takeovers: list[Takeover]) -> list[dict]:
    groups: dict[tuple, dict] = {}
    for t in takeovers:
        m = re.search(r"D-\d+", t.observable)
        door = m.group(0) if m else None
        key = (t.site, t.situation_type, door or t.zone)
        g = groups.setdefault(key, {
            "key": "|".join(str(k) for k in key), "site": t.site, "situation_type": t.situation_type,
            "zone": t.zone, "zone_class": t.zone_class, "door_id": door, "count": 0,
            "observables": [], "tried": [], "operator_tips": [], "takeover_ids": []})
        g["count"] += 1
        g["takeover_ids"].append(t.id)
        if t.observable not in g["observables"] and len(g["observables"]) < 3:
            g["observables"].append(t.observable)
        dump = json.dumps([a.model_dump() for a in t.tried], sort_keys=True)
        if dump not in g["tried"] and len(g["tried"]) < 3:
            g["tried"].append(dump)
        if t.operator_tip not in g["operator_tips"] and len(g["operator_tips"]) < 5:
            g["operator_tips"].append(t.operator_tip)
    return sorted(groups.values(), key=lambda g: -g["count"])


def _rulebook_summary(genome: Genome) -> str:
    rows = [f"- {r.id} [{r.scope.model_dump(exclude_none=True)}]: {r.text[:100]}"
            for r in genome.rulebook if r.status == "active"]
    return "\n".join(rows) or "(empty)"


class Fixer:
    def __init__(self, llm: LLM, playbook: FixerPlaybook, model: str | None = None):
        self.llm = llm
        self.playbook = playbook
        from iz.llm.client import fixer_model
        self.model = model or fixer_model()

    def _prompt(self, clusters, raindrop_summary, genome, gate_feedback) -> tuple[str, str]:
        pb = self.playbook
        shown = [c for c in clusters if c["count"] >= pb.min_evidence]
        locked = LockedGuardrails().model_dump()
        system = (f"You improve a facilities-robot harness.\nPlaybook instructions: {pb.instructions}\n"
                  f"max_edits: {pb.max_edits}\nmin_evidence: {pb.min_evidence}\n"
                  f"Lessons: {json.dumps(pb.lessons)}\n"
                  f"Locked guardrails you may NOT change: {json.dumps(locked)}. Never grant access to human_only zones. "
                  "Gate thresholds are out of scope. Return JSON {edits:[...]}.")
        user = (f"Takeover clusters:\n{json.dumps(shown, indent=1)}\n\n"
                f"Current rulebook (genome v{genome.version}):\n{_rulebook_summary(genome)}\n\n"
                f"Gate feedback (aggregate only):\n{json.dumps(gate_feedback or [])}\n\n"
                f"Monitoring summary:\n{raindrop_summary or '(none)'}")
        return system, user

    def propose(self, clusters: list[dict], raindrop_summary: str | None, genome: Genome,
                gate_feedback: list[str] | None = None) -> Candidate:
        pb = self.playbook
        system, user = self._prompt(clusters, raindrop_summary, genome, gate_feedback)
        try:
            data = self.llm.complete_json(self.model, system, user, FIXER_SCHEMA).data
            raw = data.get("edits", []) if isinstance(data, dict) else []
        except Exception:  # noqa: BLE001
            raw = []
        edits: list[Edit] = []
        for i, e in enumerate(raw):
            if len(edits) >= pb.max_edits:
                break
            try:
                if not isinstance(e, dict):
                    continue
                edit = Edit.model_validate({**e, "id": f"e{len(edits)}"})
            except Exception:  # noqa: BLE001
                continue
            if _touches_locked(edit):   # key-based: rule TEXT may mention "locked door"
                continue
            if edit.kind == "add_rule":
                p = dict(edit.payload)
                p.setdefault("id", f"r-{genome.version + 1}-{len(edits)}")
                p.setdefault("scope", edit.scope.model_dump())
                p.setdefault("text", edit.rationale)
                if not p.get("provenance"):
                    p["provenance"] = _provenance(clusters, edit.scope)
                edit = edit.model_copy(update={"payload": p})
            edits.append(edit)
        h = hashlib.sha256(json.dumps([e.model_dump() for e in edits], sort_keys=True, default=str)
                           .encode()).hexdigest()[:8]
        return Candidate(id=f"c-v{genome.version}-{h}", parent_version=genome.version, edits=edits,
                         author="fixer", fixer_version=pb.version)


def _touches_locked(edit: Edit) -> bool:
    try:
        from iz.harness.guardrails import validate_edits
        return bool(validate_edits([edit]))
    except ImportError:
        def walk(o: Any) -> bool:
            if isinstance(o, dict):
                return any(str(k).lower() in ("locked", "gate", "threshold") or walk(v) for k, v in o.items())
            return isinstance(o, list) and any(walk(v) for v in o)
        return walk(edit.payload)


def _provenance(clusters: list[dict], scope: Scope) -> list[str]:
    ids: list[str] = []
    for c in clusters:
        if ((scope.door_id and c.get("door_id") == scope.door_id)
                or (scope.situation_type and c["situation_type"] == scope.situation_type
                    and (not scope.site or c["site"] == scope.site))):
            ids += c["takeover_ids"]
    return ids


def placebo(candidate: Candidate) -> Candidate:
    edits = [Edit(id=f"p{i}", kind="add_rule",
                  payload={"id": f"placebo-{candidate.id}-{i}", "scope": Scope().model_dump(),
                           "text": "Keep a log of observed conditions."},
                  rationale="placebo", scope=Scope(), prediction=e.prediction)
             for i, e in enumerate(candidate.edits)]
    return Candidate(id=f"{candidate.id}-placebo", parent_version=candidate.parent_version, edits=edits,
                     author="placebo", fixer_version=candidate.fixer_version)


_TYPES = ["fire_door", "office_door", "cart_blocking", "crowd_blocking", "spill",
          "restricted_shortcut", "elevator_out_of_service", "person_down", "low_battery",
          "construction_detour"]
_WEIGHTS = ["distance", "battery", "capability", "congestion", "peak_avoid"]


def random_edit(genome: Genome, seed: int) -> Candidate:
    rng = random.Random(seed)
    choice = rng.choice(["add_rule", "dispatcher", "tunable_guardrail"])
    pred = {"train_fixed": 0, "heldout_fixed": 0}
    if choice == "add_rule":
        st = rng.choice(_TYPES)
        scope = Scope(situation_type=st)
        payload = {"id": f"rand-{seed}", "scope": scope.model_dump(),
                   "text": f"Proceed carefully when handling {st.replace('_', ' ')} situations."}
    elif choice == "dispatcher":
        scope = Scope()
        w = rng.choice(_WEIGHTS)
        payload = {w: round(max(0.0, getattr(genome.dispatcher, w) + rng.uniform(-0.3, 0.3)), 3)}
    else:
        scope = Scope()
        payload = {"max_wait_s": float(rng.randint(5, 30))}
    e = Edit(id="e0", kind=choice, payload=payload, rationale="random ablation", scope=scope,
             prediction=pred)
    return Candidate(id=f"c-v{genome.version}-rand{seed}", parent_version=genome.version, edits=[e],
                     author="random")
