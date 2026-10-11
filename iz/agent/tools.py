"""Tool schemas for the robot agent."""
from __future__ import annotations

import typing
from typing import Any

from iz.types import MAX_ACTIONS_PER_DECISION, Action, ToolName

_N = {"type": "number"}
_S = {"type": "string"}
TOOL_ARGS: dict[str, dict[str, Any]] = {
    "wait": {"seconds": {"type": "number", "minimum": 0, "maximum": 120}},
    "request_open": {"door_id": _S},
    "request_access": {"zone": _S},
    "push_door": {"force_n": {"type": "number", "minimum": 0, "maximum": 200}, "hold": {"type": "boolean"}},
    "call_elevator": {"elevator": _S},
    "reroute": {"via": _S},
    "probe_push": {"force_n": _N},
    "push_cart": {"force_n": _N},
    "yield": {"distance_m": _N},
    "clean_spill": {},
    "slow_down": {"speed_mps": _N},
    "page_staff": {"reason": _S},
    "request_human_takeover": {"reason": _S},
    "report_issue": {"kind": _S, "detail": _S},
}
ALL_TOOLS: list[str] = list(typing.get_args(ToolName))


def decision_schema(allowed: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {
            "actions": {
                "type": "array", "maxItems": MAX_ACTIONS_PER_DECISION,
                "items": {"type": "object",
                          "properties": {"tool": {"type": "string", "enum": list(allowed)},
                                         "args": {"type": "object"}},
                          "required": ["tool", "args"]},
            },
            "report_issue": {"type": "string"},
        },
        "required": ["actions"],
    }


def describe_tools(allowed: list[str]) -> str:
    lines = []
    for t in allowed:
        args = TOOL_ARGS.get(t, {})
        sig = ", ".join(f"{k}: {v.get('type', 'any')}" for k, v in args.items())
        lines.append(f"- {t}({sig})")
    return "\n".join(lines)


def _cast(spec: dict, v: Any) -> Any:
    t = spec.get("type")
    if t == "number":
        if isinstance(v, bool):
            raise ValueError("bool is not a number")
        return float(v)
    if t == "boolean":
        if isinstance(v, str):
            return v.strip().lower() in ("true", "1", "yes")
        return bool(v)
    if t == "string":
        return str(v)
    return v


def coerce_actions(raw: Any, allowed: list[str]) -> list[Action]:
    """Drop unknown/disallowed tools and unknown args; cast types; cap at 3. Never raises."""
    out: list[Action] = []
    try:
        for item in raw or []:
            if len(out) >= MAX_ACTIONS_PER_DECISION:
                break
            try:
                tool = item.get("tool")
                if tool not in allowed or tool not in TOOL_ARGS:
                    continue
                spec = TOOL_ARGS[tool]
                args: dict[str, Any] = {}
                for k, v in (item.get("args") or {}).items():
                    if k in spec:
                        try:
                            args[k] = _cast(spec[k], v)
                        except (ValueError, TypeError):
                            pass
                out.append(Action(tool=tool, args=args))
            except Exception:
                continue
    except Exception:
        pass
    return out
