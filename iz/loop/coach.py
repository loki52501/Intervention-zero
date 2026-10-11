"""Layer 3: evolve the fixer's playbook. Cannot touch gate thresholds or locked guardrails."""
from __future__ import annotations

import os
from collections import Counter
from typing import Any

from iz.types import FixerPlaybook, LLM

COACH_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "instructions": {"type": "string"},
        "max_edits": {"type": "integer", "minimum": 1, "maximum": 5},
        "min_evidence": {"type": "integer", "minimum": 1, "maximum": 5},
        "lessons": {"type": "array", "items": {"type": "string"}, "maxItems": 8},
    },
    "required": ["instructions", "max_edits", "min_evidence", "lessons"],
}


def _clamp(v: Any, lo: int, hi: int, default: int) -> int:
    try:
        return max(lo, min(hi, int(v)))
    except (TypeError, ValueError):
        return default


def _calibration(track: list[dict]) -> float:
    errs = []
    for t in track:
        p, a = t.get("predicted") or {}, t.get("actual") or {}
        errs.append(abs(p.get("heldout_fixed", 0) - a.get("heldout_fixed", 0)))
    return sum(errs) / len(errs) if errs else 0.0


class Coach:
    def __init__(self, llm: LLM, model: str | None = None):
        self.llm = llm
        from iz.llm.client import fixer_model
        self.model = model or fixer_model()

    def evolve(self, playbook: FixerPlaybook, track_record: list[dict]) -> FixerPlaybook:
        verdicts = Counter(t.get("verdict") for t in track_record)
        reasons = Counter(t.get("reason") for t in track_record if t.get("verdict") == "REJECT")
        system = ("You coach a harness-fixer agent by rewriting its playbook (instructions, max_edits 1-5, "
                  "min_evidence 1-5, up to 8 lessons). Gate thresholds and locked guardrails are out of "
                  "scope and cannot be changed. Return JSON.")
        user = (f"Current playbook v{playbook.version}: instructions={playbook.instructions!r} "
                f"max_edits={playbook.max_edits} min_evidence={playbook.min_evidence} "
                f"lessons={playbook.lessons}\n"
                f"Decisions: {dict(verdicts)}\nTop reject reasons: {reasons.most_common(5)}\n"
                f"Mean abs error of predicted held-out fixes: {_calibration(track_record):.2f}\n"
                f"Track record: {track_record[-20:]}")
        base = dict(version=playbook.version + 1, parent=playbook.version, created_by="coach")
        try:
            d = self.llm.complete_json(self.model, system, user, COACH_SCHEMA).data
            if not isinstance(d, dict):
                raise ValueError("bad response")
        except Exception:  # noqa: BLE001
            return FixerPlaybook(instructions=playbook.instructions, max_edits=playbook.max_edits,
                                 min_evidence=playbook.min_evidence,
                                 lessons=list(playbook.lessons) or ["unchanged (coach unavailable)"], **base)
        lessons = d.get("lessons", playbook.lessons)
        if not isinstance(lessons, list):
            lessons = playbook.lessons
        ins = d.get("instructions")
        return FixerPlaybook(
            instructions=ins if isinstance(ins, str) and ins.strip() else playbook.instructions,
            max_edits=_clamp(d.get("max_edits"), 1, 5, playbook.max_edits),
            min_evidence=_clamp(d.get("min_evidence"), 1, 5, playbook.min_evidence),
            lessons=[str(x) for x in lessons][:8], **base)
