"""Pure genome edit application and scope matching."""
from __future__ import annotations

from typing import Any

from iz.types import (ContextPolicy, DispatcherWeights, Edit, Genome, ModelRouting, RobotView,
                      RuleEntry, Scope, TunableGuardrails)


def _scope(scope: Scope | dict | None) -> Scope:
    if isinstance(scope, Scope):
        return scope
    return Scope.model_validate(scope or {})


def scope_matches(scope: Scope | dict | None, view: RobotView) -> bool:
    s = _scope(scope)
    if s.site is not None and s.site != view.site:
        return False
    if s.situation_type is not None and s.situation_type != view.type:
        return False
    if s.zone_class is not None and s.zone_class != view.zone_class:
        return False
    if s.door_id is not None and s.door_id != view.observable_params.get("door_id"):
        return False
    return True


def specificity(scope: Scope | dict | None) -> int:
    s = _scope(scope)
    if s.door_id is not None:
        return 4
    if s.site and s.situation_type:
        return 3
    if s.situation_type and s.zone_class:
        return 2
    if s.situation_type:
        return 1
    n = sum(v is not None for v in s.model_dump().values())
    return min(n, 3)


def rules_for(genome: Genome, view: RobotView) -> list[RuleEntry]:
    act = [r for r in genome.rulebook if r.status == "active" and scope_matches(r.scope, view)]
    return sorted(act, key=lambda r: -specificity(r.scope))  # sorted() is stable


def _merge(model: Any, cls: type, payload: dict) -> Any:
    d = model.model_dump()
    d.update({k: v for k, v in payload.items() if k in cls.model_fields})
    return cls.model_validate(d)


def apply_edits(genome: Genome, edits: list[Edit], created_by: str = "fixer") -> Genome:
    g = Genome.model_validate(genome.model_dump())
    for e in edits:
        p = e.payload or {}
        if e.kind == "add_rule":
            d = dict(p)
            d.setdefault("scope", e.scope.model_dump())
            r = RuleEntry.model_validate(d)
            g.rulebook = [x for x in g.rulebook if x.id != r.id] + [r]
        elif e.kind == "retire_rule":
            for r in g.rulebook:
                if r.id == p.get("rule_id"):
                    r.status = "retired"
        elif e.kind == "tool_policy":
            for ent in p.get("deny", []) or []:
                ent = dict(ent)
                sc = ent.get("scope")
                if sc is None:
                    sc = e.scope
                ent["scope"] = _scope(sc).model_dump(exclude_none=True)
                g.tool_policy.deny.append(ent)
            for tool, args in (p.get("clamps") or {}).items():
                tgt = g.tool_policy.clamps.setdefault(tool, {})
                for arg, bounds in args.items():
                    tgt[arg] = [float(b) for b in bounds]
        elif e.kind == "dispatcher":
            g.dispatcher = _merge(g.dispatcher, DispatcherWeights, p)
        elif e.kind == "tunable_guardrail":
            g.guardrails.tunable = _merge(g.guardrails.tunable, TunableGuardrails, p)
        elif e.kind == "context_policy":
            g.context_policy = _merge(g.context_policy, ContextPolicy, p)
        elif e.kind == "model_routing":
            g.model_routing = _merge(g.model_routing, ModelRouting, p)
    g.version = genome.version + 1
    g.parent = genome.version
    g.created_by = created_by
    return g
