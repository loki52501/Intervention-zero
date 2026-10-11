"""Robot agent: view + genome + context -> up to 3 actions."""
from __future__ import annotations

from typing import Callable

from iz.agent.prompts import robot_system_prompt, robot_user_prompt
from iz.agent.tools import ALL_TOOLS, coerce_actions, decision_schema
from iz.harness.context import build_context_text
from iz.llm import client
from iz.types import MAX_ACTIONS_PER_DECISION, LLM, Action, ContextBundle, Genome, RobotView


class Robot:
    def __init__(self, model: str | None = None) -> None:
        self.model = model or client.robot_model()

    def decide(self, view: RobotView, genome: Genome, context: ContextBundle | None, llm: LLM) -> list[Action]:
        """Up to 3 actions; [] on any failure (caller counts an agent_error takeover)."""
        try:
            denied = {d.get("tool") for d in genome.tool_policy.deny
                      if not d.get("scope") or _scope_ok(d["scope"], view)}
            allowed = [t for t in ALL_TOOLS if t not in denied]
            ctx = build_context_text(context or ContextBundle(), genome.context_policy.token_budget)
            resp = llm.complete_json(self.model, robot_system_prompt(genome),
                                     robot_user_prompt(view, ctx, allowed), decision_schema(allowed))
            data = resp.data or {}
            actions = coerce_actions(data.get("actions", []), allowed)
            rep = data.get("report_issue")
            if (isinstance(rep, str) and rep and len(actions) < MAX_ACTIONS_PER_DECISION
                    and "report_issue" in allowed):
                actions.append(Action(tool="report_issue", args={"kind": "agent", "detail": rep}))
            return actions
        except Exception:
            return []


def _scope_ok(scope, view: RobotView) -> bool:
    try:
        from iz.harness.genome import scope_matches
        return scope_matches(scope, view)
    except Exception:
        return False


def _rules(genome: Genome, view: RobotView):
    try:
        from iz.harness.genome import rules_for
        return rules_for(genome, view)
    except Exception:
        return [r for r in genome.rulebook if r.status == "active"]


def make_robot_fn(llm: LLM, store=None, model: str | None = None) -> Callable[[RobotView, Genome], list[Action]]:
    robot = Robot(model)

    def fn(view: RobotView, genome: Genome) -> list[Action]:
        ctx = None
        if store is not None:
            try:
                ctx = store.get_context(view, genome)
            except Exception:
                ctx = None
        if ctx is None:
            ctx = ContextBundle(backend="local")
        # The genome under evaluation is the authority for rules (candidate/placebo genomes in the
        # gate must be tested with THEIR rulebook, not whatever the store has promoted).
        ctx = ctx.model_copy(update={"rules": _rules(genome, view)})
        return robot.decide(view, genome, ctx, llm)

    return fn
