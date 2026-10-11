"""Nightly rulebook consolidation: merge near-duplicate rules, retire stale ones. Guardrails untouched."""
from __future__ import annotations

import re
from typing import Any

from iz.types import Genome, RuleEntry

STALE_DAYS = 5


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9_\-]+", text.lower()))


def jaccard(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta and not tb:
        return 1.0
    return len(ta & tb) / len(ta | tb)


def _matches(rule: RuleEntry, t: Any) -> bool:
    s = rule.scope
    if s.site is not None and s.site != t.site:
        return False
    if s.situation_type is not None and s.situation_type != t.situation_type:
        return False
    if s.zone_class is not None and s.zone_class != getattr(t, "zone_class", s.zone_class):
        return False
    return True


def run(genome: Genome, takeovers: list, day: int) -> tuple[Genome | None, str]:
    """Returns (new_genome or None if unchanged, summary text for memory)."""
    tday = {t.id: t.day for t in takeovers}
    merged: list[str] = []
    retired: list[str] = []

    keep: list[RuleEntry] = []
    for r in (x.model_copy(deep=True) for x in genome.rulebook):
        if r.status != "active":
            keep.append(r)
            continue
        dup = next((k for k in keep if k.status == "active" and k.scope == r.scope
                    and jaccard(k.text, r.text) >= 0.8), None)
        if dup is None:
            keep.append(r)
        else:
            dup.provenance = list(dict.fromkeys(dup.provenance + r.provenance))
            merged.append(f"{r.id}->{dup.id}")

    for r in keep:
        if r.status != "active" or not r.provenance:
            continue
        days = [tday[p] for p in r.provenance if p in tday]
        if not days or max(days) > day - STALE_DAYS:
            continue
        if any(t.day > day - STALE_DAYS and _matches(r, t) for t in takeovers):
            continue
        r.status = "retired"
        retired.append(r.id)

    if not merged and not retired:
        return None, f"day {day}: consolidation made no changes ({len(genome.rulebook)} rules)."
    new = genome.model_copy(deep=True)
    new.rulebook = keep
    new.parent = genome.version
    new.version = genome.version + 1
    new.created_by = "consolidate"
    summary = (f"day {day}: consolidated rulebook; merged {len(merged)} duplicate rule(s) "
               f"[{', '.join(merged)}], retired {len(retired)} stale rule(s) [{', '.join(retired)}].")
    return new, summary
