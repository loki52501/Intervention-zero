"""Render a ContextBundle as prompt text under a token budget (~4 chars/token)."""
from __future__ import annotations

from iz.types import ContextBundle, RuleEntry


def _scope(r: RuleEntry) -> str:
    s = r.scope
    parts = [p for p in (s.site, s.situation_type, s.zone_class, s.door_id) if p]
    return "/".join(parts) if parts else "global"


def _episode(e: dict) -> str:
    keys = ("observable", "tried", "operator_tip", "what_worked")
    parts = [f"{k}={e[k]}" for k in keys if e.get(k)]
    return "- " + "; ".join(parts) if parts else ""


def build_context_text(bundle: ContextBundle, token_budget: int) -> str:
    limit = max(0, token_budget) * 4
    sections = [
        ("RULES:", [f"- [{_scope(r)}] {r.text}" for r in bundle.rules]),
        ("PAST INCIDENTS:", [_episode(e) for e in bundle.episodes]),
        ("SITE FACTS:", [f"- {f}" for f in bundle.facts]),
    ]
    out = ""
    for header, items in sections:
        started = False
        for it in items:
            if not it:
                continue
            add = ("" if started else (("\n" if out else "") + header + "\n")) + it + "\n"
            if len(out) + len(add) > limit:
                break
            out += add
            started = True
    return out
