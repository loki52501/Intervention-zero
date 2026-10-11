import json
import threading

import pytest

from iz.obs import events as ev


def test_emits_valid_lines(tmp_path):
    p = tmp_path / "e.jsonl"
    em = ev.Emitter("r1", path=str(p), atlas=False)
    e = em.emit("takeover", {"robot_id": "r", "site": "s", "type": "t", "operator_tip": "x"})
    em.ticker("mujoco", "hello")
    lines = [json.loads(l) for l in p.read_text().splitlines()]
    assert len(lines) == 2
    assert set(lines[0]) == {"id", "ts", "kind", "run_id", "payload"}
    assert lines[0] == e and lines[0]["run_id"] == "r1"
    assert lines[1]["kind"] == "ticker" and lines[1]["payload"] == {"feature": "mujoco", "text": "hello"}
    assert lines[0]["id"] != lines[1]["id"]


def test_unknown_kind_raises(tmp_path):
    em = ev.Emitter("r", path=str(tmp_path / "e.jsonl"), atlas=False)
    with pytest.raises(ValueError):
        em.emit("nope", {})
    with pytest.raises(ValueError):
        em.ticker("bogus_feature", "x")


def test_atlas_unreachable_and_feed_errors_never_raise(tmp_path, monkeypatch):
    monkeypatch.delenv("MONGODB_URI", raising=False)
    em = ev.Emitter("r", path=str(tmp_path / "e.jsonl"))
    assert em.emit("reload", {"robots": 1, "version": 2})["kind"] == "reload"
    em2 = ev.Emitter("r", path=str(tmp_path / "f.jsonl"), atlas=True)
    import iz.store.feed as feed
    monkeypatch.setattr(feed, "emit", lambda *a, **k: 1 / 0)
    em2.emit("reload", {"robots": 1, "version": 2})


def test_threadsafe_and_get_emitter(tmp_path):
    em = ev.Emitter("r", path=str(tmp_path / "e.jsonl"), atlas=False)
    ts = [threading.Thread(target=lambda: [em.emit("reload", {"i": i}) for i in range(20)]) for _ in range(5)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    lines = (tmp_path / "e.jsonl").read_text().splitlines()
    assert len(lines) == 100 and all(json.loads(l) for l in lines)
    assert ev.get_emitter("same", path=str(tmp_path / "g.jsonl"), atlas=False) is ev.get_emitter("same")
