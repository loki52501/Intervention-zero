import pytest

pytest.importorskip("mcp")

from iz.agent import mcp_body as m  # noqa: E402
from iz.types import Outcome, OutcomeStatus  # noqa: E402


class _OkJudge:
    def judge(self, situation, actions):
        return Outcome(status=OutcomeStatus.SUCCESS, reason=",".join(a.tool for a in actions))


def test_pending_accumulates_and_commit_returns_outcome():
    m.configure(judge=_OkJudge())
    view = m.observe()
    assert view["situation_id"]
    m.request_open("D-17")
    m.push_door(20, hold=False)
    m.yield_(1.5)
    assert [a.tool for a in m.STATE.pending] == ["request_open", "push_door", "yield"]
    assert m.STATE.pending[2].args == {"distance_m": 1.5}
    out = m.commit()
    assert out["status"] == "success" and out["reason"] == "request_open,push_door,yield"
    assert m.STATE.pending == []


def test_default_judge_commit_returns_outcome_dict():
    m.configure()
    m.observe()
    m.slow_down()
    out = m.commit()
    assert Outcome(**out).status in set(OutcomeStatus)


def test_injected_provider_and_registered_tool_names():
    from iz.types import RobotView
    v = RobotView(situation_id="x", site="A", type="spill", zone="z", zone_class="public", observable="o")
    m.configure(provider=lambda sid: v, judge=_OkJudge())
    assert m.observe("x")["type"] == "spill"
    import asyncio
    names = {t.name for t in asyncio.run(m.mcp.list_tools())}
    assert {"yield", "observe", "commit", "page_staff", "report_issue"} <= names


def test_all_toolnames_registered():
    import asyncio
    import typing
    from iz.types import ToolName
    names = {t.name for t in asyncio.run(m.mcp.list_tools())}
    assert set(typing.get_args(ToolName)) <= names


def test_real_generator_and_judge_end_to_end():
    m.configure()
    view = m.observe()
    assert m.STATE.situation is not None and m.STATE.situation.id == view["situation_id"]
    assert m.observe(view["situation_id"])["situation_id"] == view["situation_id"]
    m.request_human_takeover("unsure")
    out = m.commit()
    assert Outcome(**out).status in set(OutcomeStatus)
