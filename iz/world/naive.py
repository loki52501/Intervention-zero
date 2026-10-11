"""No-LLM naive robot: a generic first attempt per situation type."""
from __future__ import annotations

from iz.types import Action, RobotView

NAIVE_TABLE: dict[str, list[tuple[str, dict]]] = {
    "fire_door": [("push_door", {"force_n": 40})],
    "office_door": [("push_door", {"force_n": 30})],
    "cart_blocking": [("push_cart", {"force_n": 40})],
    "crowd_blocking": [("yield", {}), ("wait", {"seconds": 5})],
    "spill": [("slow_down", {})],
    "restricted_shortcut": [("reroute", {})],
    "elevator_out_of_service": [("call_elevator", {})],
    "person_down": [("page_staff", {})],
    "low_battery": [("wait", {"seconds": 5})],
    "construction_detour": [("reroute", {})],
}


def naive_actions(view: RobotView) -> list[Action]:
    return [Action(tool=t, args=dict(a)) for t, a in NAIVE_TABLE[view.type]]
