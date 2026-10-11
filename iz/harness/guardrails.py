"""Local mirror of the Mongo $jsonSchema validator for candidate genomes."""
from __future__ import annotations

import re
from typing import Any

from pydantic import ValidationError

from iz.types import Edit, Genome, LockedGuardrails

LOCKED = LockedGuardrails()
_LOCKED_VALUES = LOCKED.model_dump()
_GRANT = re.compile(r"(enter|access|go through|reroute via|shortcut through).{0,40}(human[_ ]only|S-2|server room)",
                    re.I | re.S)
_NEG = re.compile(r"\b(never|do not|don't|avoid)\b", re.I)
_DENY_VERBS = {"deny", "avoid", "never", "block", "forbid"}
MAX_FORCE = float(LOCKED.max_contact_force_n)


def validate_candidate_locally(genome: Genome | dict) -> list[str]:
    """Return violation strings; [] means the candidate is acceptable."""
    raw = genome.model_dump() if isinstance(genome, Genome) else dict(genome)
    out: list[str] = []
    locked = ((raw.get("guardrails") or {}).get("locked")) or {}
    for k, want in _LOCKED_VALUES.items():
        if k in locked and locked[k] != want:
            out.append(f"locked guardrail: {k} must be {want}")
    for k in locked:
        if k not in _LOCKED_VALUES:
            out.append(f"locked guardrail: unknown field {k}")
    try:
        g = genome if isinstance(genome, Genome) else Genome.model_validate(raw)
    except ValidationError as e:
        if not out:
            out.append(f"invalid genome: {e.errors()[0].get('loc')}")
        return out
    for r in g.rulebook:
        if r.status != "active":
            continue
        p = r.params or {}
        grants = p.get("allow_zone_kind") == "human_only"
        if (p.get("zone_class") == "human_only"
                and str(p.get("action", p.get("verb", "allow"))).lower() not in _DENY_VERBS
                and not _NEG.search(r.text)):
            grants = True
        if _GRANT.search(r.text) and not _NEG.search(r.text):
            grants = True
        if grants:
            out.append(f"rule {r.id} grants human_only access")
    tp = g.tool_policy
    for tool, args in tp.clamps.items():
        for arg, bounds in args.items():
            if arg in ("hold", "allow", "unlock"):
                out.append(f"tool_policy cannot clamp/allow {tool}.{arg}")
            if "force_n" in arg and bounds and max(bounds) > MAX_FORCE:
                out.append(f"clamp exceeds max_contact_force_n: {tool}.{arg}")
    for ent in tp.deny:
        if any(k in ent for k in ("allow", "unlock")):
            out.append("tool_policy cannot allow/unlock")
    return out


def validate_edits(edits: list[Edit]) -> list[str]:
    """Flag edits whose payload touches locked/gate/threshold keys."""
    out: list[str] = []

    def walk(o: Any) -> bool:
        if isinstance(o, dict):
            return any(str(k).lower() in ("locked", "gate", "threshold") or walk(v) for k, v in o.items())
        if isinstance(o, list):
            return any(walk(v) for v in o)
        return False

    for e in edits:
        if walk(e.payload):
            out.append(f"locked guardrail: edit {e.id} touches locked/gate fields")
    return out
