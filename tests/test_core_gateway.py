from iz.types import Genome, Action, RobotView, ToolPolicy
from iz.agent.gateway import Gateway


def view(**kw):
    d = dict(situation_id="s", site="A", type="cart_blocking", zone="z", zone_class="corridor",
             observable="o", observable_params={})
    d.update(kw)
    return RobotView(**d)


def test_reroute_human_only_blocked():
    gw = Gateway(Genome(version=0))
    ok, bl = gw.filter(view(), [Action(tool="reroute", args={"via": "S-2"})], {"S-2": "human_only"})
    assert ok == [] and "never_enter_human_only" in bl[0]
    ok, bl = gw.filter(view(), [Action(tool="reroute", args={"via": "corridor1"})], {"corridor1": "corridor"})
    assert len(ok) == 1 and bl == []


def test_fire_door_hold():
    gw = Gateway(Genome(version=0))
    v = view(type="fire_door")
    ok, bl = gw.filter(v, [Action(tool="push_door", args={"hold": True})])
    assert ok == [] and "never_hold_fire_doors" in bl[0]
    ok, bl = gw.filter(v, [Action(tool="push_door", args={"hold": False})])
    assert len(ok) == 1


def test_clamps():
    a = Action(tool="push_cart", args={"force_n": 120})
    gw = Gateway(Genome(version=0))
    ok, _ = gw.filter(view(), [a])
    assert ok[0].args["force_n"] == 80 and a.args["force_n"] == 120 and gw.last_clamps
    g = Genome(version=0, tool_policy=ToolPolicy(clamps={"push_cart": {"force_n": [0, 40]}}))
    ok, _ = Gateway(g).filter(view(), [a])
    assert ok[0].args["force_n"] == 40
    ok, _ = gw.filter(view(), [Action(tool="wait", args={"seconds": 99})])
    assert ok[0].args["seconds"] == 15.0


def test_deny_scope_and_limits():
    g = Genome(version=0, tool_policy=ToolPolicy(
        deny=[{"tool": "push_cart", "scope": {"site": "A", "situation_type": "cart_blocking"}}]))
    gw = Gateway(g)
    a = [Action(tool="push_cart", args={"force_n": 10})]
    ok, bl = gw.filter(view(site="A"), a)
    assert ok == [] and "denied by tool_policy" in bl[0]
    ok, _ = gw.filter(view(site="B"), a)
    assert len(ok) == 1


def test_max_actions_and_hook():
    calls = []
    gw = Gateway(Genome(version=0), on_call=calls.append)
    acts = [Action(tool="wait", args={"seconds": 1}) for _ in range(5)]
    ok, bl = gw.filter(view(), acts)
    assert len(ok) <= 3 and len(bl) == 2 and calls and calls[0]["situation_id"] == "s"
    assert len(acts) == 5
