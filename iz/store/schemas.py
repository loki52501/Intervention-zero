"""$jsonSchema validators. Permissive except the safety core (locked guardrails, human_only)."""
from __future__ import annotations

from iz.store.db import DECISIONS, EDITS, GENOMES, PLAYBOOKS, TAKEOVERS

# human_only is deliberately absent: no rule/tool policy may grant access to it.
ALLOWED_ZONE_KINDS = ["public", "corridor", "ward", "office", "kitchen", "elevator", "service"]
EDIT_KINDS = ["add_rule", "retire_rule", "tool_policy", "dispatcher", "tunable_guardrail",
              "context_policy", "model_routing"]

_ALLOW_ZONE_KINDS = {"bsonType": "array", "items": {"enum": ALLOWED_ZONE_KINDS}}
_PARAMS_WITH_ZONES = {"bsonType": "object", "properties": {"allow_zone_kinds": _ALLOW_ZONE_KINDS}}

LOCKED_SCHEMA = {
    "bsonType": "object",
    "required": ["never_enter_human_only", "never_hold_fire_doors", "always_escalate_person_down",
                 "battery_floor_pct", "max_contact_force_n"],
    "additionalProperties": False,
    "properties": {
        "never_enter_human_only": {"enum": [True]},
        "never_hold_fire_doors": {"enum": [True]},
        "always_escalate_person_down": {"enum": [True]},
        "battery_floor_pct": {"enum": [15]},
        "max_contact_force_n": {"enum": [80]},
    },
}

GENOME_SCHEMA = {
    "bsonType": "object",
    "required": ["_id", "version", "guardrails", "rulebook", "tool_policy"],
    "properties": {
        "version": {"bsonType": ["int", "long"]},
        "guardrails": {
            "bsonType": "object",
            "required": ["locked"],
            "properties": {"locked": LOCKED_SCHEMA},
        },
        "rulebook": {
            "bsonType": "array",
            "items": {"bsonType": "object", "properties": {"params": _PARAMS_WITH_ZONES}},
        },
        "tool_policy": {
            "bsonType": "object",
            "properties": {
                "deny": {"bsonType": "array", "items": {"bsonType": "object",
                                                         "properties": {"allow_zone_kinds": _ALLOW_ZONE_KINDS}}},
            },
        },
    },
}

PLAYBOOK_SCHEMA = {
    "bsonType": "object",
    "required": ["_id", "version", "instructions", "max_edits", "min_evidence"],
    "properties": {
        "version": {"bsonType": ["int", "long"]},
        "instructions": {"bsonType": "string"},
        "max_edits": {"bsonType": ["int", "long"], "minimum": 1, "maximum": 5},
        "min_evidence": {"bsonType": ["int", "long"], "minimum": 1},
    },
}

TAKEOVER_SCHEMA = {
    "bsonType": "object",
    "required": ["_id", "run_id", "site", "situation_type", "observable", "operator_tip",
                 "policy_version", "ts"],
    "properties": {"ts": {"bsonType": "date"}},
}

EDIT_SCHEMA = {
    "bsonType": "object",
    "required": ["_id", "kind", "payload"],
    "properties": {"kind": {"enum": EDIT_KINDS}, "payload": {"bsonType": "object"}},
}

DECISION_SCHEMA = {
    "bsonType": "object",
    "required": ["_id", "candidate_id", "parent_version", "verdict", "ts"],
    "properties": {
        "verdict": {"enum": ["PROMOTE", "REJECT", "QUARANTINE"]},
        "ts": {"bsonType": "date"},
    },
}

VALIDATORS: dict[str, dict] = {
    GENOMES: {"$jsonSchema": GENOME_SCHEMA},
    PLAYBOOKS: {"$jsonSchema": PLAYBOOK_SCHEMA},
    TAKEOVERS: {"$jsonSchema": TAKEOVER_SCHEMA},
    EDITS: {"$jsonSchema": EDIT_SCHEMA},
    DECISIONS: {"$jsonSchema": DECISION_SCHEMA},
}


# --- fleet blackboard (D10 M-FLEET)
FLEET_BLACKBOARD = "fleet_blackboard"
BLACKBOARD_KINDS = ["hazard", "blocked_lane", "task_claim"]
_NUM = ["double", "int", "long"]
BLACKBOARD_SCHEMA = {
    "bsonType": "object",
    "required": ["world", "kind", "x", "y", "radius", "robot_id", "ts", "ttl"],
    "properties": {
        "world": {"bsonType": "string"},
        "kind": {"enum": BLACKBOARD_KINDS},
        "x": {"bsonType": _NUM},
        "y": {"bsonType": _NUM},
        "radius": {"bsonType": _NUM, "minimum": 0},
        "ttl": {"bsonType": _NUM, "minimum": 0},
        "robot_id": {"bsonType": "string"},
        "ts": {"bsonType": "date"},
        "sim_t": {"bsonType": _NUM},
        "task_id": {"bsonType": "string"},
        # lineage: reroute/blocked_lane may point at the hazard _id it came from;
        # walkable with $graphLookup (connectFromField parent_id, connectToField _id)
        "parent_id": {"bsonType": ["string", "null"]},
    },
}
VALIDATORS[FLEET_BLACKBOARD] = {"$jsonSchema": BLACKBOARD_SCHEMA}