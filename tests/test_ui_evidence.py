from fastapi import FastAPI
from fastapi.testclient import TestClient

from iz.ui import evidence


def _client():
    app = FastAPI()
    app.include_router(evidence.router)
    return TestClient(app)


def _boom(*a, **k):
    raise RuntimeError("mongo unreachable")


def test_page_is_html():
    r = _client().get("/evidence")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "learning loop" in r.text.lower()


def test_summary_degrades_when_mongo_down(monkeypatch):
    import iz.store.db as D
    monkeypatch.setattr(D, "get_db", _boom)
    evidence._cache.update(t=0.0, data=None)
    evidence._probe_cache.update(t=0.0, data=None)
    evidence._rd_cache.update(t=0.0, data=None, busy=True)  # no network in tests
    j = _client().get("/api/evidence/summary").json()
    for k in evidence.SECTIONS:
        assert k in j
    for k in ("loop", "safety", "memory", "recall"):
        assert j[k]["ok"] is False and j[k]["error"]
    assert "ok" in j["raindrop"]
    evidence._cache.update(t=0.0, data=None)


def test_recall_returns_list(monkeypatch):
    import iz.store.db as D
    monkeypatch.setattr(D, "get_db", _boom)
    j = _client().get("/api/evidence/recall", params={"q": "wet floor"}).json()
    assert isinstance(j["hits"], list)
    assert _client().get("/api/evidence/recall").json()["hits"] == []
