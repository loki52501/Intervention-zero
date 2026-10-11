from datetime import datetime, timezone
from types import SimpleNamespace as NS

import pytest

from iz.obs import rd_query

SINCE = datetime(2026, 1, 1, tzinfo=timezone.utc)


def ev(signals, **props):
    return NS(signals=[NS(id="i", name=s) for s in signals], properties=props,
              timestamp=datetime(2026, 1, 2, 3, 4, tzinfo=timezone.utc))


class StubClient:
    def __init__(self, events):
        self.events = self
        self._e = events

    def list(self, **kw):
        return NS(data=self._e, meta=NS(has_more=False, cursor=None))


def test_missing_key_degrades(monkeypatch):
    monkeypatch.delenv("RAINDROP_QUERY_API_KEY", raising=False)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)
    for r in (rd_query.top_takeover_causes(SINCE), rd_query.agent_reports(SINCE),
              rd_query.signal_timeseries("takeover")):
        assert r["degraded"] is True and r["source"] == "raindrop"


def test_client_raises(monkeypatch):
    def boom():
        raise RuntimeError("net down")
    monkeypatch.setattr(rd_query, "_client", boom)
    assert rd_query.top_takeover_causes(SINCE)["degraded"] is True
    assert rd_query.agent_reports(SINCE)["degraded"] is True
    assert rd_query.signal_timeseries("takeover")["degraded"] is True


def test_grouping(monkeypatch):
    events = [
        ev(["takeover"], situation_type="fire_door", site="A"),
        ev(["takeover_avoidable"], situation_type="fire_door", site="A"),
        ev(["takeover"], situation_type="spill", site="B"),
        ev(["agent_report"], kind="capability_gap", detail="cannot open"),
        ev(["agent_report"], kind="capability_gap", detail="again"),
        ev(["agent_report"], kind="missing_context", detail="no map"),
        ev([]),
    ]
    monkeypatch.setattr(rd_query, "_client", lambda: StubClient(events))
    t = rd_query.top_takeover_causes(SINCE)
    assert t["degraded"] is False
    assert t["causes"][0] == {"situation_type": "fire_door", "site": "A", "count": 2}
    assert len(t["causes"]) == 2
    a = rd_query.agent_reports(SINCE)
    assert a["reports"][0]["kind"] == "capability_gap" and a["reports"][0]["count"] == 2
    assert a["reports"][0]["examples"] == ["cannot open", "again"]
    s = rd_query.signal_timeseries("takeover", SINCE, "hour")
    assert s["series"] == [{"t": "2026-01-02T03:00:00Z", "count": 2}]
    assert rd_query.signal_timeseries("takeover", SINCE, "bogus")["degraded"] is True
