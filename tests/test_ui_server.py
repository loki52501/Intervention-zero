import importlib.util
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from iz.ui import server

REPO = Path(__file__).resolve().parents[1]

KINDS = {"site_loaded", "mission_start", "robot_move", "situation", "decision", "outcome",
         "takeover", "fix_proposed", "validator_refused", "gate_result", "promoted", "reload",
         "drift_alarm", "coach", "metrics", "ticker"}


def _demo():
    spec = importlib.util.spec_from_file_location("demo_events", REPO / "scripts" / "demo_events.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_build_events_cover_contract():
    evs = _demo().build_events()
    assert 100 <= len(evs) <= 400
    assert KINDS <= {e["kind"] for e in evs}
    assert all(e["payload"].get("synthetic") is True for e in evs)
    assert all(e["run_id"] == "demo-synthetic" for e in evs)
    assert len({e["id"] for e in evs}) == len(evs)
    feats = {e["payload"]["feature"] for e in evs if e["kind"] == "ticker"}
    assert {"vector_search", "atlas_search", "rank_fusion", "rerank", "transaction",
            "change_stream", "validator", "time_series", "raindrop_event", "raindrop_signal",
            "raindrop_query"} <= feats


def test_index(tmp_path, monkeypatch):
    (tmp_path / "index.html").write_text("<html><title>Intervention Zero</title></html>")
    monkeypatch.setattr(server, "STATIC", tmp_path)
    r = TestClient(server.app).get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "Intervention Zero" in r.text


def test_replay_streams_and_terminates():
    p = REPO / "out" / "test_demo_events.jsonl"
    _demo().write(p)
    try:
        c = TestClient(server.app)
        r = c.get("/events", params={"replay": "out/test_demo_events.jsonl", "speed": 1000})
        assert r.status_code == 200
        for k in ("site_loaded", "gate_result", "validator_refused", "drift_alarm"):
            assert k in r.text
        datas = [json.loads(l[5:].strip()) for l in r.text.splitlines() if l.startswith("data:")]
        assert datas and datas[0]["id"] == "ev-0001"
    finally:
        p.unlink(missing_ok=True)


def test_replay_outside_repo_400():
    c = TestClient(server.app)
    assert c.get("/events", params={"replay": "../outside.jsonl"}).status_code == 400


def test_clip_rejects_traversal():
    c = TestClient(server.app)
    assert c.get("/clips/..%5Cx.mp4").status_code == 404
    assert c.get("/clips/nope.mp4").status_code == 404


def test_replay_resumes_after_last_event_id():
    p = REPO / "out" / "test_demo_events2.jsonl"
    _demo().write(p)
    try:
        c = TestClient(server.app)
        r = c.get("/events", params={"replay": "out/test_demo_events2.jsonl", "speed": 1000},
                  headers={"Last-Event-ID": "ev-0002"})
        datas = [json.loads(l[5:].strip()) for l in r.text.splitlines() if l.startswith("data:")]
        assert datas and datas[0]["id"] == "ev-0003"
    finally:
        p.unlink(missing_ok=True)


def test_static_dom_ids_match_app_js():
    import re
    static = REPO / "iz" / "ui" / "static"
    html = (static / "index.html").read_text(encoding="utf-8")
    js = (static / "app.js").read_text(encoding="utf-8")
    ids = set(re.findall(r"(?:getElementById|\$)\(['\"]([\w-]+)['\"]\)", js))
    assert ids
    missing = [i for i in ids if f'id="{i}"' not in html]
    assert not missing, missing
