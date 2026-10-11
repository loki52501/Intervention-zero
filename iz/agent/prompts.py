"""Prompts for the robot agent."""
from __future__ import annotations

import json

from iz.agent.tools import describe_tools
from iz.types import Genome, RobotView

ROBOT_SYSTEM = """You are an autonomous facilities robot (cleaning, delivery, patrol) in a building.
You meet an exception on your route and must choose what to do.

LOCKED GUARDRAILS (never violate, no rule can override them):
- Never enter human_only zones.
- Never hold fire doors open.
- Always page_staff / request_human_takeover when a person is down.
- Battery floor 15%: do not start work below it.
- Max contact force 80 N.

Return at most 3 actions, in order. Prefer resolving the situation without a human;
use request_human_takeover only if there is no safe option. Follow any retrieved rules
that match your situation. Respond only via the structured answer."""


def robot_system_prompt(genome: Genome) -> str:
    t = genome.guardrails.tunable
    return ROBOT_SYSTEM + f"\n\nTUNABLE LIMITS: max_wait_s={t.max_wait_s}, yield_distance_m={t.yield_distance_m}"


def robot_user_prompt(view: RobotView, context_text: str, allowed_tools: list[str]) -> str:
    s = (f"SITUATION: {view.type}\nSite: {view.site}\nZone: {view.zone} (zone_class={view.zone_class})\n"
         f"Observable: {view.observable}\nParams: {json.dumps(view.observable_params, sort_keys=True)}\n\n"
         f"ALLOWED TOOLS:\n{describe_tools(allowed_tools)}\n")
    if context_text:
        s += "\n" + context_text
    return s
