"""Gateway: enforces tool policy, argument clamps and locked guardrails on robot actions."""
from __future__ import annotations

from typing import Callable

from iz.harness.genome import scope_matches
from iz.types import MAX_ACTIONS_PER_DECISION, Action, Genome, LockedGuardrails, RobotView

MAX_FORCE = float(LockedGuardrails().max_contact_force_n)


def _num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _fmt(v: float) -> str:
    return str(int(v)) if float(v) == int(v) else str(v)


def make_gateway_factory(site_zone_kinds: dict[str, str] | None = None,
                         on_call: Callable[[dict], None] | None = None):
    """genome -> (view, actions) -> allowed actions. Plug into iz.loop.replay.make_replay_fn."""
    def factory(genome: Genome):
        # new Gateway per call: last_clamps is per-instance state and replay runs threaded
        return lambda view, actions: Gateway(genome, on_call).filter(view, actions, site_zone_kinds)[0]
    return factory


class Gateway:
    def __init__(self, genome: Genome, on_call: Callable[[dict], None] | None = None):
        self.genome = genome
        self.on_call = on_call
        self.last_clamps: list[str] = []

    @staticmethod
    def _is_fire_door(view: RobotView) -> bool:
        p = view.observable_params
        return (view.type == "fire_door" or p.get("door_type") == "fire"
                or "FIRE" in str(p.get("label", "")).upper())

    def _note(self, arg: str, old, new) -> None:
        self.last_clamps.append(f"clamped {arg} {_fmt(old)}->{_fmt(new)}")

    def _clamp(self, tool: str, args: dict) -> None:
        for arg, b in self.genome.tool_policy.clamps.get(tool, {}).items():
            if arg in args and _num(args[arg]) and len(b) == 2:
                new = min(max(args[arg], b[0]), b[1])
                if new != args[arg]:
                    self._note(arg, args[arg], new)
                    args[arg] = new
        if "force_n" in args and _num(args["force_n"]):
            new = min(max(args["force_n"], 0), MAX_FORCE)
            if new != args["force_n"]:
                self._note("force_n", args["force_n"], new)
                args["force_n"] = new
        if tool == "wait" and _num(args.get("seconds")):
            mx = self.genome.guardrails.tunable.max_wait_s
            if args["seconds"] > mx:
                self._note("seconds", args["seconds"], mx)
                args["seconds"] = mx

    def filter(self, view: RobotView, actions: list[Action],
               site_zone_kinds: dict[str, str] | None = None) -> tuple[list[Action], list[str]]:
        self.last_clamps = []
        allowed: list[Action] = []
        blocked: list[str] = []
        kinds = site_zone_kinds or {}
        for i, a in enumerate(actions):
            if i >= MAX_ACTIONS_PER_DECISION:
                blocked.append(f"max actions: {a.tool}")
                continue
            reason = self._block_reason(view, a, kinds)
            if reason:
                blocked.append(reason)
                continue
            c = a.model_copy(deep=True)
            self._clamp(c.tool, c.args)
            allowed.append(c)
        if self.on_call:
            try:
                self.on_call({"situation_id": view.situation_id,
                              "tool_in": [a.tool for a in actions],
                              "allowed": [a.model_dump() for a in allowed],
                              "blocked": list(blocked), "clamps": list(self.last_clamps)})
            except Exception:
                pass
        return allowed, blocked

    def _block_reason(self, view: RobotView, a: Action, kinds: dict[str, str]) -> str | None:
        for d in self.genome.tool_policy.deny:
            if d.get("tool") == a.tool and (not d.get("scope") or scope_matches(d["scope"], view)):
                return f"denied by tool_policy: {a.tool}"
        if a.tool in ("reroute", "request_access"):
            tgt = a.args.get("via") if a.tool == "reroute" else (a.args.get("zone") or a.args.get("via"))
            if tgt and (kinds.get(tgt) == "human_only" or str(tgt).upper().startswith("S-2")):
                return "locked: never_enter_human_only"
        if a.tool == "push_door" and a.args.get("hold") and self._is_fire_door(view):
            return "locked: never_hold_fire_doors"
        return None
