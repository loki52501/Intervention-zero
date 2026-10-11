"""Shared contracts for every workstream. Change only with main-orchestrator approval.

Naming: a *situation* is an exception the robot meets on a route edge; *hidden* truth is
known only to the world/judge; the robot sees a RobotView; the fixer never sees held-out data.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field

# ---------------------------------------------------------------- world

ZoneKind = Literal["public", "corridor", "ward", "office", "kitchen", "elevator", "service", "human_only"]
SituationType = Literal[
    "fire_door", "office_door", "cart_blocking", "crowd_blocking", "spill",
    "restricted_shortcut", "elevator_out_of_service", "person_down",
    "low_battery", "construction_detour",
]
PHYSICS_TYPES: frozenset[str] = frozenset(
    {"fire_door", "office_door", "cart_blocking", "crowd_blocking", "spill", "construction_detour"}
)
MissionKind = Literal["clean", "deliver", "patrol"]
SiteId = Literal["A", "B", "C", "D", "A_reno"]
TRAIN_SITES: tuple[str, ...] = ("A", "B", "C")
HELDOUT_SITE = "D"


class Zone(BaseModel):
    id: str
    kind: ZoneKind
    xy: tuple[float, float]          # metres, for drawing and MuJoCo layout


class Edge(BaseModel):
    a: str
    b: str
    length_m: float = 5.0
    door_id: str | None = None       # e.g. "D-17"
    door_type: Literal["fire", "office", "badge", "shortcut"] | None = None


class Site(BaseModel):
    id: str
    name: str
    zones: list[Zone]
    edges: list[Edge]
    peak_hours: list[int] = Field(default_factory=list)


ToolName = Literal[
    "wait", "request_open", "request_access", "push_door", "call_elevator", "reroute",
    "probe_push", "push_cart", "yield", "clean_spill", "slow_down", "page_staff",
    "request_human_takeover", "report_issue",
]
MAX_ACTIONS_PER_DECISION = 3


class Action(BaseModel):
    tool: ToolName
    args: dict[str, Any] = Field(default_factory=dict)  # e.g. {"force_n": 25} {"seconds": 10} {"via": "corridor1"}


class HiddenTruth(BaseModel):
    """Never shown to the robot or the fixer. The judge (rule or MuJoCo) reads it."""
    physics: dict[str, Any] = Field(default_factory=dict)  # door_mass_kg, closer_stiffness, open_delay_s, cart_mass_kg, tip_force_n, ...
    unsafe_tools: list[ToolName] = Field(default_factory=list)
    required_escalation: bool = False                      # person_down: must page_staff / request_human_takeover
    forbidden_zone: str | None = None                      # entering it is a safety violation
    operator_tip: str                                      # what the human says after a takeover


class Situation(BaseModel):
    id: str                          # deterministic: f"{site}:{type}:{seed}"
    site: str
    type: SituationType
    zone: str
    zone_class: ZoneKind
    edge: tuple[str, str]
    observable: str                  # text the robot sees
    observable_params: dict[str, Any] = Field(default_factory=dict)  # e.g. {"door_id": "D-17", "label": "FIRE DOOR", "tag": "red"}
    hidden: HiddenTruth
    seed: int

    def view(self) -> "RobotView":
        return RobotView(situation_id=self.id, site=self.site, type=self.type, zone=self.zone,
                         zone_class=self.zone_class, observable=self.observable,
                         observable_params=self.observable_params)


class RobotView(BaseModel):
    situation_id: str
    site: str
    type: SituationType
    zone: str
    zone_class: ZoneKind
    observable: str
    observable_params: dict[str, Any] = Field(default_factory=dict)


class OutcomeStatus(str, Enum):
    SUCCESS = "success"                      # resolved without a human
    ESCALATED_OK = "escalated_ok"            # required escalation done correctly (counts as success)
    TAKEOVER = "takeover"                    # human had to step in (avoidable)
    ESCALATION_MISSED = "escalation_missed"  # required escalation not done (counts as safety failure)
    SAFETY_VIOLATION = "safety_violation"    # unsafe tool, forbidden zone, force over limit


class Outcome(BaseModel):
    status: OutcomeStatus
    reason: str
    time_s: float = 0.0
    peak_force_n: float | None = None
    judged_by: Literal["rule", "mujoco"] = "rule"
    clip_path: str | None = None

    @property
    def takeover_free(self) -> bool:
        return self.status in (OutcomeStatus.SUCCESS, OutcomeStatus.ESCALATED_OK)

    @property
    def unsafe(self) -> bool:
        return self.status in (OutcomeStatus.SAFETY_VIOLATION, OutcomeStatus.ESCALATION_MISSED)


class Mission(BaseModel):
    id: str
    run_id: str
    site: str
    day: int
    shift: int
    kind: MissionKind
    robot_id: str
    route: list[str]
    situations: list[Situation] = Field(default_factory=list)


class Takeover(BaseModel):
    """The learning signal. Written when a robot fails a situation."""
    id: str
    run_id: str
    mission_id: str
    robot_id: str
    site: str
    day: int
    shift: int
    situation_id: str
    situation_type: SituationType
    zone: str
    zone_class: ZoneKind
    observable: str
    tried: list[Action]
    outcome: OutcomeStatus
    reason: str
    operator_tip: str
    policy_version: int
    ts: datetime


# ---------------------------------------------------------------- genome (what evolves)

class Scope(BaseModel):
    site: str | None = None              # None = global
    situation_type: SituationType | None = None
    zone_class: ZoneKind | None = None
    door_id: str | None = None


class RuleEntry(BaseModel):
    id: str
    scope: Scope
    text: str                            # shown to the robot
    params: dict[str, Any] = Field(default_factory=dict)
    provenance: list[str] = Field(default_factory=list)  # takeover ids
    status: Literal["active", "retired"] = "active"


class ToolPolicy(BaseModel):
    deny: list[dict[str, Any]] = Field(default_factory=list)          # [{"tool": "push_cart", "scope": Scope}]
    clamps: dict[str, dict[str, list[float]]] = Field(default_factory=dict)  # {"push_cart": {"force_n": [0, 40]}}


class LockedGuardrails(BaseModel):
    """Exact values enforced by the MongoDB $jsonSchema validator and the gateway. Never evolved."""
    never_enter_human_only: Literal[True] = True
    never_hold_fire_doors: Literal[True] = True
    always_escalate_person_down: Literal[True] = True
    battery_floor_pct: Literal[15] = 15
    max_contact_force_n: Literal[80] = 80


class TunableGuardrails(BaseModel):
    max_wait_s: float = 15.0
    yield_distance_m: float = 1.0


class Guardrails(BaseModel):
    locked: LockedGuardrails = Field(default_factory=LockedGuardrails)
    tunable: TunableGuardrails = Field(default_factory=TunableGuardrails)


class DispatcherWeights(BaseModel):
    distance: float = 1.0
    battery: float = 0.2
    capability: float = 1.0
    congestion: float = 0.0
    peak_avoid: float = 0.0


class ContextPolicy(BaseModel):
    k: int = 3
    rerank: bool = True
    sources: list[Literal["procedural", "episodic", "semantic", "taxonomic"]] = Field(
        default_factory=lambda: ["procedural", "episodic"])
    token_budget: int = 1200


class ModelRouting(BaseModel):
    escalate_after_failures: int = 2


class Genome(BaseModel):
    version: int
    parent: int | None = None
    rulebook: list[RuleEntry] = Field(default_factory=list)
    tool_policy: ToolPolicy = Field(default_factory=ToolPolicy)
    guardrails: Guardrails = Field(default_factory=Guardrails)
    dispatcher: DispatcherWeights = Field(default_factory=DispatcherWeights)
    context_policy: ContextPolicy = Field(default_factory=ContextPolicy)
    model_routing: ModelRouting = Field(default_factory=ModelRouting)
    created_by: str = "seed"


EditKind = Literal["add_rule", "retire_rule", "tool_policy", "dispatcher", "tunable_guardrail",
                   "context_policy", "model_routing"]


class Prediction(BaseModel):
    train_fixed: int
    heldout_fixed: int


class Edit(BaseModel):
    id: str
    kind: EditKind
    payload: dict[str, Any]          # add_rule -> RuleEntry dict; dispatcher -> partial weights; ...
    rationale: str
    scope: Scope
    prediction: Prediction


class Candidate(BaseModel):
    id: str
    parent_version: int
    edits: list[Edit]
    author: Literal["fixer", "placebo", "random", "human"] = "fixer"
    fixer_version: int | None = None


# ---------------------------------------------------------------- gate

class SetScore(BaseModel):
    n: int
    takeover_free: int
    safety_violations: int

    @property
    def rate(self) -> float:
        return self.takeover_free / self.n if self.n else 0.0


Verdict = Literal["PROMOTE", "REJECT", "QUARANTINE"]


class GateResult(BaseModel):
    """Aggregates only. This is all the fixer ever sees about held-out data."""
    candidate_id: str
    parent_version: int
    train_before: SetScore
    train_after: SetScore
    heldout_before: SetScore
    heldout_after: SetScore
    regression_after: SetScore       # regression + safety bank, must stay perfect
    regressions: int                 # cases solved by parent but not by candidate
    safety_violations: int
    heldout_gain: int
    placebo_gain: int | None = None
    verdict: Verdict
    reason: str


class Decision(BaseModel):
    id: str
    run_id: str
    candidate_id: str
    parent_version: int
    new_version: int | None
    verdict: Verdict
    reason: str
    gate: GateResult
    predicted: Prediction
    actual: Prediction
    ts: datetime


# ---------------------------------------------------------------- layer 3: fixer playbook (coach evolves it)

class FixerPlaybook(BaseModel):
    version: int
    parent: int | None = None
    instructions: str
    max_edits: int = 3
    min_evidence: int = 2            # takeovers in a cluster before proposing a rule
    lessons: list[str] = Field(default_factory=list)
    created_by: Literal["seed", "coach"] = "seed"


# ---------------------------------------------------------------- interfaces

@runtime_checkable
class Judge(Protocol):
    def judge(self, situation: Situation, actions: list[Action]) -> Outcome: ...


class ContextBundle(BaseModel):
    rules: list[RuleEntry] = Field(default_factory=list)
    episodes: list[dict[str, Any]] = Field(default_factory=list)   # similar past takeovers (tips, what worked)
    facts: list[str] = Field(default_factory=list)
    backend: str = "atlas"                                          # "atlas" | "agent_engine" | "local"
    hits: dict[str, int] = Field(default_factory=dict)              # for the UI ticker: {"vector": 5, "text": 4, "fused": 3}


@runtime_checkable
class MemoryStore(Protocol):
    def add_takeover(self, t: Takeover) -> None: ...
    def add_rule(self, r: RuleEntry, site: str | None) -> None: ...
    def add_fact(self, site: str, text: str) -> None: ...
    def get_context(self, view: RobotView, genome: Genome) -> ContextBundle: ...


class LLMResponse(BaseModel):
    data: dict[str, Any]             # parsed JSON
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    cached: bool = False


@runtime_checkable
class LLM(Protocol):
    def complete_json(self, model: str, system: str, user: str, schema: dict[str, Any]) -> LLMResponse: ...


@runtime_checkable
class Clock(Protocol):
    def now(self) -> datetime: ...
