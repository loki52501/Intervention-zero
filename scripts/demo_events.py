"""Generate a clearly SYNTHETIC demo event stream (out/demo_events.jsonl).

Every payload carries "synthetic": true. Numbers here are made up for UI demos
and are NOT results of any real run.
"""
from __future__ import annotations

import json
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO / "out" / "demo_events.jsonl"
RUN_ID = "demo-synthetic"

ZONES_A = [
    ("lobby", "public", [4, 12]), ("corridor1", "corridor", [12, 12]),
    ("corridor2", "corridor", [24, 12]), ("ward_3", "ward", [30, 20]),
    ("office_12", "office", [20, 20]), ("kitchen", "kitchen", [12, 3]),
    ("elevator_1", "elevator", [36, 12]), ("elevator_2", "elevator", [36, 6]),
    ("loading", "service", [24, 3]), ("S-2", "human_only", [38, 21]),
]
EDGES_A = [
    ("lobby", "corridor1", None, None), ("corridor1", "corridor2", "D-17", "fire"),
    ("corridor2", "office_12", "D-21", "office"), ("corridor2", "ward_3", None, None),
    ("corridor1", "kitchen", None, None), ("kitchen", "loading", None, None),
    ("corridor2", "elevator_1", None, None), ("corridor2", "elevator_2", None, None),
    ("ward_3", "S-2", "D-30", "badge"), ("office_12", "S-2", None, "shortcut"),
]


class _Gen:
    def __init__(self):
        self.rng = random.Random(7)
        self.t = datetime(2026, 10, 11, 16, 0, 0, tzinfo=timezone.utc)
        self.events: list[dict] = []

    def emit(self, kind: str, payload: dict):
        self.t += timedelta(seconds=round(self.rng.uniform(0.4, 2.5), 2))
        payload = dict(payload, synthetic=True)
        self.events.append({
            "id": f"ev-{len(self.events) + 1:04d}",
            "ts": self.t.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
            "kind": kind, "run_id": RUN_ID, "payload": payload,
        })

    def tick(self, feature: str, text: str):
        self.emit("ticker", {"feature": feature, "text": text})


def build_events() -> list[dict]:
    g = _Gen()
    xy = {z[0]: z[2] for z in ZONES_A}
    kind_of = {z[0]: z[1] for z in ZONES_A}

    # (1) site
    g.emit("site_loaded", {
        "site": "A",
        "zones": [{"id": i, "kind": k, "xy": p} for i, k, p in ZONES_A],
        "edges": [{"a": a, "b": b, "door_id": d, "door_type": t} for a, b, d, t in EDGES_A],
    })
    # (2) baseline metrics
    g.emit("metrics", {"day": 1, "shift": "day", "site": "A", "tfmr": 0.20,
                       "window_tfmr": 0.20, "policy_version": 1, "cost_usd": 0.0})
    g.tick("time_series", "metrics bucket written (day 1, day shift)")

    robots = [("R-01", "clean", ["lobby", "corridor1", "kitchen", "loading"]),
              ("R-02", "deliver", ["lobby", "corridor1", "corridor2", "ward_3"]),
              ("R-03", "patrol", ["corridor1", "corridor2", "elevator_1", "office_12"])]

    def run_mission(n, rid, mk, route, day, site, xymap, fire=None):
        mid = f"M-{n:03d}"
        g.emit("mission_start", {"mission_id": mid, "robot_id": rid, "site": site, "day": day,
                                 "shift": "day", "kind": mk, "route": route})
        for z in route[:3]:
            g.emit("robot_move", {"robot_id": rid, "site": site, "zone": z, "xy": xymap[z]})
        if fire:
            typ, zone, tool, args, rules, status, reason, judge, clip = fire
            g.emit("situation", {"robot_id": rid, "site": site, "zone": zone, "type": typ,
                                 "observable": {"door_id": args.get("door_id"), "closed": True}})
            g.tick("vector_search", "hybrid hits 5")
            g.tick("atlas_search", "text hits 4 (door_type=fire)")
            g.tick("rank_fusion", "fused 5 -> 3 candidates")
            g.tick("rerank", "reranked top 3")
            g.emit("decision", {"robot_id": rid, "actions": [{"tool": tool, "args": args}],
                                "rules_used": rules,
                                "context_hits": {"vector": 5, "text": 4, "fused": 3, "path": "hybrid"}})
            o = {"robot_id": rid, "mission_id": mid, "status": status, "reason": reason,
                 "judged_by": judge}
            if clip:
                o["clip"] = clip
            g.emit("outcome", o)
            return
        g.emit("situation", {"robot_id": rid, "site": site, "zone": route[-1], "type": "spill",
                             "observable": {"size": "small"}})
        g.emit("decision", {"robot_id": rid, "actions": [{"tool": "clean_spill", "args": {}}],
                            "rules_used": ["base-01"],
                            "context_hits": {"vector": 2, "text": 1, "fused": 2, "path": "hybrid"}})
        g.emit("outcome", {"robot_id": rid, "mission_id": mid, "status": "success",
                           "reason": "handled by base rule", "judged_by": "rule"})

    # (3) first missions, one fire-door takeover
    run_mission(1, *robots[0], 1, "A", xy)
    run_mission(2, *robots[1], 1, "A", xy, fire=(
        "fire_door", "corridor1", "push_door", {"door_id": "D-17", "force": 0.8}, ["base-03"],
        "takeover", "pushed a closed fire door", "mujoco", "out/clips/demo_fire_door.mp4"))
    g.emit("takeover", {"robot_id": "R-02", "site": "A", "type": "fire_door",
                        "operator_tip": "Fire doors close on their own; never prop or push. "
                                        "Request open and wait."})
    g.tick("raindrop_event", "takeover event sent")
    g.tick("raindrop_signal", "takeover signal")
    run_mission(3, *robots[2], 1, "A", xy)
    g.tick("raindrop_query", "fixer read 3 signals")

    # (4) C-001 promoted
    g.emit("fix_proposed", {"candidate_id": "C-001", "author": "fixer", "edits": [{
        "kind": "add_rule",
        "scope": {"site": None, "situation_type": "fire_door", "zone_class": None, "door_id": None},
        "text_or_change": "At fire doors: request_open, wait 10s, never push_door.",
        "rationale": "Operator tip: fire doors close on their own; pushing is unsafe.",
        "prediction": {"train_fixed": 9, "heldout_fixed": 7}}]})
    g.tick("validator", "schema + safety validator passed C-001")
    g.emit("gate_result", {"candidate_id": "C-001", "train": {"before": 3, "after": 9, "n": 10},
                           "heldout": {"before": 2, "after": 7, "n": 8},
                           "regression": {"n": 40, "passed": 40}, "safety_violations": 0,
                           "verdict": "PROMOTE", "reason": "0 safety violations, 0 regressions"})
    g.tick("transaction", "promotion committed in one transaction")
    g.emit("promoted", {"version": 2, "parent": 1, "candidate_id": "C-001"})
    g.tick("change_stream", "policy v2 delivered to 3 robots")
    g.emit("reload", {"robots": 3, "version": 2})

    # (5) placebo rejected
    g.emit("fix_proposed", {"candidate_id": "C-002", "author": "placebo", "edits": [{
        "kind": "add_rule",
        "scope": {"site": None, "situation_type": "fire_door", "zone_class": None, "door_id": None},
        "text_or_change": "Be careful near doors.", "rationale": "Generic advice (placebo control).",
        "prediction": {"train_fixed": 0, "heldout_fixed": 0}}]})
    g.emit("gate_result", {"candidate_id": "C-002", "train": {"before": 3, "after": 3, "n": 10},
                           "heldout": {"before": 2, "after": 2, "n": 8},
                           "regression": {"n": 40, "passed": 40}, "safety_violations": 0,
                           "verdict": "REJECT", "reason": "no improvement over parent"})

    # (6) S-2 refusal
    g.emit("fix_proposed", {"candidate_id": "C-003", "author": "fixer", "edits": [{
        "kind": "add_rule",
        "scope": {"site": "A", "situation_type": "restricted_shortcut", "zone_class": "human_only",
                  "door_id": None},
        "text_or_change": "Allow shortcut through S-2 when late",
        "rationale": "Operator wanted faster deliveries.",
        "prediction": {"train_fixed": 4, "heldout_fixed": 3}}]})
    g.emit("validator_refused", {"candidate_id": "C-003", "message":
           "Document failed validation: rules may not permit entry to human_only zone S-2"})
    g.tick("validator", "C-003 refused by database validator")

    # (7) quarantine
    g.emit("fix_proposed", {"candidate_id": "C-004", "author": "random", "edits": [{
        "kind": "tunable_guardrail",
        "scope": {"site": "A", "situation_type": "crowd_blocking", "zone_class": None, "door_id": None},
        "text_or_change": "slow_down threshold 0.5 -> 0.9", "rationale": "Random mutation control.",
        "prediction": {"train_fixed": 1, "heldout_fixed": 0}}]})
    g.emit("gate_result", {"candidate_id": "C-004", "train": {"before": 3, "after": 6, "n": 10},
                           "heldout": {"before": 2, "after": 2, "n": 8},
                           "regression": {"n": 40, "passed": 39}, "safety_violations": 1,
                           "verdict": "QUARANTINE", "reason": "1 safety-bank violation"})

    # (8) coach
    g.emit("coach", {"playbook_version": 2, "parent": 1, "verdict": "PROMOTE",
                     "reason": "Fixer playbook v2 beat v1 on held-out replays through the same gate",
                     "changes": ["Prefer scoped rules over global ones",
                                 "Never propose entry to human_only zones",
                                 "Cite the operator tip in the rationale"]})

    # (9) declining metrics days 2..7
    tf = [0.17, 0.13, 0.10, 0.08, 0.07, 0.06]
    for i, v in enumerate(tf):
        d = i + 2
        n = 4 + d
        run_mission(n, *robots[d % 3], d, "A", xy)
        g.emit("metrics", {"day": d, "shift": "day", "site": "A", "tfmr": v, "window_tfmr": v,
                           "policy_version": 2, "cost_usd": round(0.05 * d, 2)})
        g.tick("time_series", f"metrics bucket written (day {d})")
        if d % 2 == 0:
            g.tick("raindrop_signal", "no takeover signal")

    # (10) day 8 renovation
    xy2 = {k: list(v) for k, v in xy.items()}
    zones_b = [z for z in ZONES_A if z[0] != "elevator_2"]
    edges_b = [e for e in EDGES_A if "elevator_2" not in (e[0], e[1])]
    edges_b = [("corridor1", "corridor2", None, None) if e[2] == "D-17" else e for e in edges_b]
    edges_b.append(("kitchen", "corridor2", "D-17", "fire"))
    g.emit("site_loaded", {
        "site": "A_reno",
        "zones": [{"id": i, "kind": k, "xy": p} for i, k, p in zones_b],
        "edges": [{"a": a, "b": b, "door_id": d, "door_type": t} for a, b, d, t in edges_b]})
    for n, (typ, zone, rid, mk) in enumerate([
            ("elevator_out_of_service", "elevator_1", "R-02", "deliver"),
            ("construction_detour", "corridor2", "R-01", "clean"),
            ("elevator_out_of_service", "elevator_1", "R-03", "patrol"),
            ("construction_detour", "corridor1", "R-02", "deliver")]):
        mid = f"M-{40 + n}"
        g.emit("mission_start", {"mission_id": mid, "robot_id": rid, "site": "A_reno", "day": 8,
                                 "shift": "day", "kind": mk, "route": ["lobby", "corridor1", zone]})
        g.emit("robot_move", {"robot_id": rid, "site": "A_reno", "zone": zone, "xy": xy2[zone]})
        g.emit("situation", {"robot_id": rid, "site": "A_reno", "zone": zone, "type": typ,
                             "observable": {"detail": "layout changed"}})
        g.emit("decision", {"robot_id": rid, "actions": [{"tool": "call_elevator", "args": {"id": 2}}],
                            "rules_used": ["base-07"],
                            "context_hits": {"vector": 1, "text": 0, "fused": 1, "path": "vector"}})
        g.emit("outcome", {"robot_id": rid, "mission_id": mid, "status": "takeover",
                           "reason": "plan assumed removed layout", "judged_by": "rule"})
        g.emit("takeover", {"robot_id": rid, "site": "A_reno", "type": typ,
                            "operator_tip": "Elevator 2 is gone; use elevator 1 or reroute."})
    g.tick("time_series", "window TFMR spike detected in time-series bucket")
    g.emit("metrics", {"day": 8, "shift": "day", "site": "A_reno", "tfmr": 0.35,
                       "window_tfmr": 0.35, "policy_version": 2, "cost_usd": 0.5})
    g.emit("drift_alarm", {"site": "A", "window_tfmr": 0.35, "baseline_tfmr": 0.06,
                           "p_value": 0.003, "missions_since_change": 20})
    g.emit("fix_proposed", {"candidate_id": "C-005", "author": "fixer", "edits": [{
        "kind": "add_rule",
        "scope": {"site": "A_reno", "situation_type": "elevator_out_of_service", "zone_class": None,
                  "door_id": None},
        "text_or_change": "If elevator is out of service: reroute via remaining elevator, else page_staff.",
        "rationale": "Four takeovers after renovation share this cause.",
        "prediction": {"train_fixed": 6, "heldout_fixed": 5}}]})
    g.tick("validator", "schema + safety validator passed C-005")
    g.emit("gate_result", {"candidate_id": "C-005", "train": {"before": 2, "after": 7, "n": 8},
                           "heldout": {"before": 1, "after": 5, "n": 6},
                           "regression": {"n": 40, "passed": 40}, "safety_violations": 0,
                           "verdict": "PROMOTE", "reason": "0 safety violations, 0 regressions"})
    g.tick("transaction", "promotion committed in one transaction")
    g.emit("promoted", {"version": 3, "parent": 2, "candidate_id": "C-005"})
    g.tick("change_stream", "policy v3 delivered to 3 robots")
    g.emit("reload", {"robots": 3, "version": 3})
    for d, v in [(9, 0.18), (10, 0.10), (11, 0.07)]:
        run_mission(60 + d, *robots[d % 3], d, "A_reno", xy2)
        g.emit("metrics", {"day": d, "shift": "day", "site": "A_reno", "tfmr": v, "window_tfmr": v,
                           "policy_version": 3, "cost_usd": round(0.06 * d, 2)})
        g.tick("time_series", f"metrics bucket written (day {d})")
    return g.events


def write(path: Path | str = DEFAULT_OUT) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    evs = build_events()
    with open(path, "w", encoding="utf-8") as f:
        for e in evs:
            f.write(json.dumps(e) + "\n")
    return len(evs)


def main():
    n = write(DEFAULT_OUT)
    print(f"wrote {n} synthetic events to {DEFAULT_OUT}")


if __name__ == "__main__":
    main()
