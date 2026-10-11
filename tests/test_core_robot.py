from iz.agent.robot import Robot, make_robot_fn
from iz.agent.tools import ALL_TOOLS
from iz.harness.context import build_context_text
from iz.llm.client import FakeLLM
from iz.types import ContextBundle, Genome, RobotView, RuleEntry, Scope

VIEW = RobotView(situation_id="A:fire_door:1", site="A", type="fire_door", zone="z1",
                 zone_class="corridor", observable="red door D-17")


def test_decide_truncates_and_validates():
    acts = [{"tool": "wait", "args": {"seconds": "5", "bogus": 1}}, {"tool": "nope", "args": {}},
            {"tool": "reroute", "args": {"via": "c1"}}, {"tool": "yield", "args": {}},
            {"tool": "page_staff", "args": {}}]
    r = Robot("m").decide(VIEW, Genome(version=0), None, FakeLLM(responses=[{"actions": acts}]))
    assert 0 < len(r) <= 3 and all(a.tool in ALL_TOOLS for a in r)
    assert r[0].args == {"seconds": 5.0}


def test_failure_returns_empty():
    f = FakeLLM(responses=[{"actions": []}])
    f.fail_times = 99
    assert Robot("m").decide(VIEW, Genome(version=0), None, f) == []


def test_prompt_contains_rule_and_denied_tool_absent():
    g = Genome(version=0)
    g.tool_policy.deny = [{"tool": "push_cart"}, {"tool": "wait", "scope": {"site": "B"}}]  # scoped to another site -> still offered
    rule = RuleEntry(id="r1", scope=Scope(), text="UNIQUE_RULE_TEXT")
    llm = FakeLLM(responses=[{"actions": []}])
    Robot("m").decide(VIEW, g, ContextBundle(rules=[rule]), llm)
    c = llm.calls[0]
    assert "UNIQUE_RULE_TEXT" in c["user"]
    enum = c["schema"]["properties"]["actions"]["items"]["properties"]["tool"]["enum"]
    assert "push_cart" not in enum and "wait" in enum


def test_context_budget():
    b = ContextBundle(rules=[RuleEntry(id=str(i), scope=Scope(), text="x" * 50) for i in range(100)],
                      facts=["f" * 50] * 50)
    assert len(build_context_text(b, 50)) <= 200
    assert build_context_text(ContextBundle(), 100) == ""


def test_make_robot_fn_local():
    g = Genome(version=0, rulebook=[RuleEntry(id="r", scope=Scope(), text="LOCALRULE")])
    llm = FakeLLM(responses=[{"actions": [{"tool": "wait", "args": {}}]}])
    out = make_robot_fn(llm)(VIEW, g)
    assert out[0].tool == "wait" and "LOCALRULE" in llm.calls[0]["user"]
