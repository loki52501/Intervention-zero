from iz.types import Genome, Edit, Scope, Prediction, RuleEntry, RobotView, DispatcherWeights
from iz.harness.genome import apply_edits, rules_for, scope_matches, specificity
from iz.harness.guardrails import validate_candidate_locally, validate_edits, LOCKED
from iz.harness.dispatcher import RobotState, assign

P = Prediction(train_fixed=0, heldout_fixed=0)


def E(kind, payload, scope=None, id="e1"):
    return Edit(id=id, kind=kind, payload=payload, rationale="r", scope=scope or Scope(), prediction=P)


def rule(id, scope, text="x", status="active"):
    return RuleEntry(id=id, scope=scope, text=text, status=status)


def view(**kw):
    d = dict(situation_id="s", site="A", type="fire_door", zone="z", zone_class="corridor",
             observable="o", observable_params={"door_id": "D-17"})
    d.update(kw)
    return RobotView(**d)


def test_apply_edits_pure_and_versioning():
    g = Genome(version=3)
    before = g.model_dump()
    g2 = apply_edits(g, [
        E("add_rule", {"id": "r1", "text": "hi", "scope": {"site": "A"}}),
        E("dispatcher", {"distance": 2.5}),
        E("tool_policy", {"deny": [{"tool": "push_cart", "scope": Scope(site="A")}],
                          "clamps": {"push_cart": {"force_n": [0, 40]}}}),
    ])
    assert g.model_dump() == before
    assert g2.version == 4 and g2.parent == 3 and g2.created_by == "fixer"
    assert g2.rulebook[0].id == "r1"
    assert g2.dispatcher.distance == 2.5 and g2.dispatcher.battery == 0.2
    assert g2.tool_policy.deny[0]["scope"] == {"site": "A"}
    assert g2.tool_policy.clamps["push_cart"]["force_n"] == [0, 40]
    g3 = apply_edits(g2, [E("retire_rule", {"rule_id": "r1"})])
    assert g3.rulebook[0].status == "retired" and g2.rulebook[0].status == "active"


def test_apply_edits_ignores_locked():
    g2 = apply_edits(Genome(version=0), [E("tunable_guardrail", {"locked": {"max_contact_force_n": 120}, "max_wait_s": 9})])
    assert g2.guardrails.tunable.max_wait_s == 9
    assert g2.guardrails.locked.max_contact_force_n == 80


def test_rules_for_precedence():
    rb = [
        rule("glob", Scope()),
        rule("type", Scope(situation_type="fire_door")),
        rule("tz", Scope(situation_type="fire_door", zone_class="corridor")),
        rule("st", Scope(site="A", situation_type="fire_door")),
        rule("door", Scope(door_id="D-17")),
        rule("other", Scope(site="B", situation_type="fire_door")),
        rule("ret", Scope(door_id="D-17"), status="retired"),
    ]
    g = Genome(version=0, rulebook=rb)
    assert [r.id for r in rules_for(g, view())] == ["door", "st", "tz", "type", "glob"]
    assert specificity(Scope()) == 0 and scope_matches({"site": "A"}, view())


def test_validator():
    assert validate_candidate_locally(Genome(version=0)) == []
    d = Genome(version=0).model_dump()
    d["guardrails"]["locked"]["max_contact_force_n"] = 120
    v = validate_candidate_locally(d)
    assert v and "locked" in v[0]
    bad = Genome(version=0, rulebook=[rule("r", Scope(), "Reroute via S-2 server room shortcut")])
    assert any("grants human_only" in x for x in validate_candidate_locally(bad))
    ok = Genome(version=0, rulebook=[rule("r", Scope(), "Never enter S-2")])
    assert validate_candidate_locally(ok) == []
    cl = apply_edits(Genome(version=0), [E("tool_policy", {"clamps": {"push_cart": {"force_n": [0, 100]}}})])
    assert any("clamp exceeds" in x for x in validate_candidate_locally(cl))
    assert validate_edits([E("tunable_guardrail", {"locked": {}})])
    assert LOCKED.max_contact_force_n == 80


def test_dispatcher():
    robots = [RobotState(id="r1", battery_pct=10, xy=(0, 0)), RobotState(id="r2", battery_pct=80, xy=(50, 0)),
              RobotState(id="r3", battery_pct=80, xy=(5, 0))]
    out = assign([{"id": "m1", "kind": "clean", "xy": (0, 0)}], robots, DispatcherWeights())
    assert out == {"m1": "r3"}
    out = assign([{"id": "m1", "kind": "clean", "xy": (0, 0)}], [robots[0]], DispatcherWeights())
    assert out == {}
