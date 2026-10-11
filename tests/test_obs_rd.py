import pytest

from iz.obs import rd


class StubInteraction:
    def __init__(self, log, eid):
        self.log, self.id = log, eid

    def track_tool(self, **kw):
        self.log.append(("tool", kw))

    def set_properties(self, p):
        self.log.append(("props", p))

    def finish(self, **kw):
        self.log.append(("finish", kw))


class StubSDK:
    def __init__(self):
        self.log = []
        self.n = 0

    def init(self, key, **kw):
        self.log.append(("init", kw))

    def begin(self, **kw):
        self.n += 1
        self.log.append(("begin", kw))
        return StubInteraction(self.log, f"evt{self.n}")

    def track_signal(self, **kw):
        self.log.append(("signal", kw))

    def flush(self):
        self.log.append(("flush", {}))


@pytest.fixture()
def clean(monkeypatch):
    monkeypatch.setattr(rd, "_count", 0)
    monkeypatch.setattr(rd, "_initialized", False)
    monkeypatch.setattr(rd, "_cap_logged", False)
    monkeypatch.setattr(rd, "_SDK", None)


def test_disabled_is_noop(clean, monkeypatch):
    import raindrop.analytics as an
    monkeypatch.setattr(an, "init", lambda *a, **k: (_ for _ in ()).throw(AssertionError("init called")))
    monkeypatch.setattr(rd, "_enabled", False)
    h = rd.mission_begin("r1", "A", 1, 0, "brief", {})
    assert isinstance(h, rd.NullHandle) and h.event_id is None
    h.tool("wait", {}, {})
    h.finish("x")
    assert rd.signal("e1", "takeover") is False
    assert rd.agent_report("e1", "capability_gap", "d") is False
    rd.flush()
    assert rd.events_sent() == 0


def test_base_props_has_all_keys():
    p = rd.base_props(site="A")
    for k in rd.REQUIRED_PROPS:
        assert k in p
    assert p["site"] == "A"


def test_invalid_signal_name():
    with pytest.raises(ValueError):
        rd.signal("e1", "nope")
    with pytest.raises(ValueError):
        rd.agent_report("e1", "bad_kind", "d")


def test_cap_and_agent_report(clean, monkeypatch):
    sdk = StubSDK()
    monkeypatch.setattr(rd, "_SDK", sdk)
    monkeypatch.setattr(rd, "_enabled", True)
    monkeypatch.setattr(rd, "CAP", 2)
    monkeypatch.setattr(rd, "feed", None, raising=False)
    h = rd.mission_begin("r1", "A", 1, 0, "brief", {"run_id": "x"})
    assert h.event_id == "evt1"
    h.tool("wait", {"s": 1}, "ok", 5.0)
    assert rd.agent_report(h.event_id, "missing_context", "no map") is True
    sig = [kw for k, kw in sdk.log if k == "signal"][0]
    assert sig["name"] == "agent_report" and sig["signal_type"] == "default"
    assert sig["properties"]["kind"] == "missing_context"
    # third event refused
    assert rd.signal(h.event_id, "takeover") is False
    assert isinstance(rd.mission_begin("r1", "A", 1, 1, "b", {}), rd.NullHandle)
    assert rd.events_sent() == 2
    begin = [kw for k, kw in sdk.log if k == "begin"][0]
    assert begin["user_id"] == "r1" and begin["convo_id"] == "A:1:0" and begin["event"] == "mission"
    for k in rd.REQUIRED_PROPS:
        assert k in begin["properties"]
    rd.flush()
    assert ("flush", {}) in sdk.log
