import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import iz.ui.studio as studio


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(studio, "STUDIO", tmp_path / "studio")
    monkeypatch.setattr(studio, "OUT", tmp_path)
    studio._WORLDS.clear()
    app = FastAPI()
    app.include_router(studio.router)
    return TestClient(app)


def test_world_endpoint_fake(client):
    r = client.post("/api/studio/world?fake=1", json={"prompt": "a small grocery store with a spill"})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["slug"] and d["spec"]["venue"] is not None
    assert set(d["counts"]) == {"train", "heldout", "safety"}
    assert d["counts"]["train"] > 0
    assert d["preview_url"]
    assert client.get(d["preview_url"]).status_code == 200


def test_events_stream_and_learn(client, monkeypatch):
    d = client.post("/api/studio/world?fake=1", json={"prompt": "a small grocery store"}).json()
    slug = d["slug"]

    def fake_learn(s, generations=3, fake=False):
        studio._emit(s, "generation", {"gen": 0, "verdict": "BASELINE", "edits": {}, "rationale": "",
                                       "heldout_success": 0, "heldout_n": 8, "safety_violations": 0})
        studio._emit(s, "generation", {"gen": 1, "verdict": "PROMOTE", "edits": {"cruise_speed": 0.7},
                                       "rationale": "slow down", "heldout_success": 3, "heldout_n": 8,
                                       "safety_violations": 0})
        return [{}, {}]

    monkeypatch.setattr(studio, "run_learning_for", fake_learn)
    r = client.post("/api/studio/learn", json={"slug": slug, "generations": 1})
    assert r.json()["started"] is True
    studio._RUNNING.get(slug) and studio._RUNNING[slug].join(5)
    kinds = []
    with client.stream("GET", f"/api/studio/events/{slug}") as resp:
        for line in resp.iter_lines():
            if line.startswith("data:"):
                ev = json.loads(line[5:])
                kinds.append((ev["kind"], ev["payload"].get("gen")))
                if ev["kind"] == "done":
                    break
    assert ("generation", 1) in kinds and kinds[-1][0] == "done"


def test_page_and_bad_slug(client):
    assert client.get("/studio").status_code == 200
    assert client.get("/api/studio/events/..%2Fx").status_code in (400, 404)
