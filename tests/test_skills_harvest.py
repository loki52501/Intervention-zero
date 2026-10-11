"""Harvest tests. API-shaped dicts below are TEST FIXTURES, not real harvested data."""
from iz.skills import harvest as h


def test_match_situation():
    assert h.match_situation("Never prop open a fire door") == "fire_door"
    assert h.match_situation("wet floor spill safety") == "spill"
    assert h.match_situation("elevator out of service tips") == "elevator_out_of_service"
    assert h.match_situation("someone collapsed, call for help") == "person_down"
    assert h.match_situation("cooking pasta") == "general"


def test_platform_from_url():
    assert h.platform_from_url("https://www.youtube.com/watch?v=1") == "youtube"
    assert h.platform_from_url("https://youtu.be/x") == "youtube"
    assert h.platform_from_url("https://www.tiktok.com/@a/video/1") == "tiktok"
    assert h.platform_from_url("https://www.instagram.com/reel/x") == "instagram"
    assert h.platform_from_url("https://osha.gov/x") == "web"


def test_normalize_brightdata_fixture():
    raw = {"url": "https://www.youtube.com/watch?v=abc", "title": "Fire door safety",
           "description": "d" * 3000, "transcript": "never prop open a fire door"}
    r = h.normalize_brightdata(raw, "youtube", now="T")
    assert r["source"] == "brightdata" and r["platform"] == "youtube"
    assert len(r["text"]) <= 1500 and r["situation_type"] == "fire_door"
    assert r["fetched_at"] == "T" and r["id"]
    assert h.normalize_brightdata({"title": "x"}, "tiktok") is None


def test_normalize_firecrawl_fixture():
    raw = {"url": "https://www.tiktok.com/@a/video/1", "title": "Spill",
           "markdown": "wet floor " * 400, "metadata": {}}
    r = h.normalize_firecrawl(raw, now="T")
    assert r["source"] == "firecrawl" and r["platform"] == "tiktok"
    assert len(r["text"]) == 1500 and r["situation_type"] == "spill"
    raw2 = {"url": "https://a.com", "metadata": {"title": "T"}, "description": "desc"}
    assert h.normalize_firecrawl(raw2)["title"] == "T"


def test_cache_roundtrip_and_dedup(tmp_path):
    p = tmp_path / "c.json"
    assert h.load_cache(p) == []
    a = h.normalize_firecrawl({"url": "https://a.com/1", "title": "t", "markdown": "fire door"}, now="T")
    h.save_cache([a, dict(a)], p)
    got = h.load_cache(p)
    assert len(got) == 1 and got[0]["id"] == a["id"]
    assert all(c["source"] == "cache" for c in h.load_cache(p, as_cache=True))


def test_harvest_no_keys_uses_cache(tmp_path, monkeypatch):
    monkeypatch.delenv("BRIGHTDATA_API_KEY", raising=False)
    monkeypatch.delenv("FIRECRAWL_API_KEY", raising=False)
    monkeypatch.setattr(h, "_load_env", lambda: None)
    assert h.harvest(cache_path=tmp_path / "c.json", time_cap=5) == []
