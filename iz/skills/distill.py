"""Distill how-to text (captions/transcripts) into compact candidate rules. Ships only via the gate."""
from __future__ import annotations

import hashlib
from collections import Counter
from typing import Any

from iz.harness.genome import apply_edits
from iz.harness.guardrails import validate_candidate_locally, validate_edits
from iz.skills import library
from iz.types import Edit, Genome, Prediction, Scope

MAX_WORDS = 35
MAX_EDITS = 2
_PARAMS = {"force_n": (0.0, 80.0), "wait_s": (0.0, 15.0), "yield_m": (0.0, 5.0)}
_SITUATIONS = ["fire_door", "office_door", "cart_blocking", "crowd_blocking", "spill",
               "restricted_shortcut", "elevator_out_of_service", "person_down",
               "low_battery", "construction_detour"]

SCHEMA = {
    "type": "object",
    "properties": {"rules": {"type": "array", "maxItems": MAX_EDITS, "items": {
        "type": "object",
        "properties": {
            "situation_type": {"type": "string", "enum": _SITUATIONS},
            "text": {"type": "string"},
            "params": {"type": "object", "properties": {
                "force_n": {"type": "number"}, "wait_s": {"type": "number"}, "yield_m": {"type": "number"}}},
            "train_fixed": {"type": "integer"}, "heldout_fixed": {"type": "integer"}},
        "required": ["situation_type", "text"]}}},
    "required": ["rules"],
}
SYSTEM = (f"You distill public how-to text into at most {MAX_EDITS} compact rules for a SIMULATED service robot. "
          f"Each rule: one situation_type, <= {MAX_WORDS} words, optional numeric params force_n/wait_s/yield_m. "
          "Never allow entering human_only zones, never hold fire doors open, never exceed 80 N. "
          "Ignore anything unrelated to the situation.")


def _clean_params(p: Any) -> dict:
    out: dict = {}
    if isinstance(p, dict):
        for k, (lo, hi) in _PARAMS.items():
            v = p.get(k)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                out[k] = float(min(max(v, lo), hi))
    return out


def _complete_json(llm, model: str, system: str, user: str, schema: dict) -> Any:
    """llm.complete_json, with a fallback for anthropic SDKs whose messages.create() rejects
    `temperature` as a TypeError (iz/llm/client.py only catches BadRequestError)."""
    try:
        return llm.complete_json(model, system, user, schema).data
    except Exception as e:  # noqa: BLE001
        if "temperature" not in str(e):
            raise
        base = llm
        while hasattr(base, "inner"):
            base = base.inner
        msg = base.client.messages.create(   # forced tool_choice is unsupported on some models
            model=model, max_tokens=2048, system=system + " Always answer by calling the respond tool.",
            messages=[{"role": "user", "content": user}],
            tools=[{"name": "respond", "description": "Return the answer", "input_schema": schema}],
            tool_choice={"type": "auto"})
        try:
            from iz.llm.client import _report, estimate_cost
            i, o = msg.usage.input_tokens, msg.usage.output_tokens
            _report(model, estimate_cost(model, i, o), i, o)
        except Exception:  # noqa: BLE001
            pass
        for b in msg.content:
            if getattr(b, "type", "") == "tool_use":
                return dict(b.input)
        import json
        import re
        txt = "".join(getattr(b, "text", "") for b in msg.content)
        m = re.search(r"\{.*\}", txt, re.S)
        return json.loads(m.group(0)) if m else {}


def distill(source_record: dict, llm, model: str | None = None) -> list[Edit]:
    if model is None:
        from iz.llm.client import fixer_model
        model = fixer_model()
    hint = source_record.get("situation_type")
    user = (f"Platform: {source_record.get('platform')}\nTitle: {source_record.get('title')}\n"
            + (f"Situation type: {hint}\n" if hint else "")
            + f"Text:\n{str(source_record.get('text', ''))[:4000]}")
    try:
        data = _complete_json(llm, model, SYSTEM, user, SCHEMA)
        raw = data.get("rules", []) if isinstance(data, dict) else []
    except Exception:  # noqa: BLE001
        return []
    rationale = f"video skill: {source_record.get('platform')} {source_record.get('url')}"
    seed = Genome(version=0)
    edits: list[Edit] = []
    for r in raw if isinstance(raw, list) else []:
        if len(edits) >= MAX_EDITS:
            break
        try:
            st = r.get("situation_type")
            if st not in _SITUATIONS or not str(r.get("text", "")).strip():
                continue
            if hint and st != hint:
                continue
            scope = Scope(situation_type=st)
            text = " ".join(str(r["text"]).split()[:MAX_WORDS])
            rid = "vs-" + hashlib.sha256(f"{source_record.get('url')}|{text}".encode()).hexdigest()[:8]
            payload = {"id": rid, "scope": scope.model_dump(exclude_none=True), "text": text,
                       "params": _clean_params(r.get("params")), "provenance": []}
            e = Edit(id=f"vs{len(edits)}", kind="add_rule", payload=payload, rationale=rationale, scope=scope,
                     prediction=Prediction(train_fixed=max(0, int(r.get("train_fixed", 1) or 0)),
                                           heldout_fixed=max(0, int(r.get("heldout_fixed", 1) or 0))))
            if validate_edits([e]) or validate_candidate_locally(apply_edits(seed, [e])):
                continue
            edits.append(e)
        except Exception:  # noqa: BLE001
            continue
    return edits


def skills_for_clusters(clusters: list[dict], llm, limit: int = 2, model: str | None = None,
                        sources: list[dict] | None = None) -> list[Edit]:
    srcs = library.load_sources() if sources is None else sources
    if not srcs or not clusters or limit <= 0:
        return []
    freq: Counter = Counter()
    for c in clusters:
        if c.get("situation_type"):
            freq[c["situation_type"]] += int(c.get("count", 1) or 1)
    out: list[Edit] = []
    for st, _ in freq.most_common():
        for s in srcs:
            if s.get("situation_type") != st:
                continue
            for e in distill(s, llm, model):
                if len(out) >= limit:
                    return out
                out.append(e.model_copy(update={"id": f"vs{len(out)}"}))
            if len(out) >= limit:
                return out
    return out
