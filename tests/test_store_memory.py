import os
import uuid
from datetime import datetime, timezone

import pytest

from iz.store.memory import AtlasMemory, LocalMemory, scope_matches
from iz.store.voyage import FakeEmbedder
from iz.types import (Action, ContextPolicy, Genome, OutcomeStatus, RobotView, RuleEntry, Scope,
                      Takeover)


def view(site="A", typ="fire_door", door="D-17", zc="corridor"):
    return RobotView(situation_id="s", site=site, type=typ, zone="z1", zone_class=zc,
                     observable="red FIRE DOOR", observable_params={"door_id": door})


def rule(id, scope, status="active", text="r"):
    return RuleEntry(id=id, scope=scope, text=text, status=status)


def tk(id, site="A", typ="fire_door", obs="red FIRE DOOR", tip="push 25N"):
    return Takeover(id=id, run_id="r", mission_id="m", robot_id="R1", site=site, day=1, shift=1,
                    situation_id="s" + id, situation_type=typ, zone="z1", zone_class="corridor",
                    observable=obs, tried=[Action(tool="wait")], outcome=OutcomeStatus.TAKEOVER,
                    reason="x", operator_tip=tip, policy_version=0, ts=datetime.now(timezone.utc))


def test_scope_matches():
    v = view()
    assert scope_matches(Scope(), v)
    assert scope_matches(Scope(site="A", situation_type="fire_door", zone_class="corridor", door_id="D-17"), v)
    assert not scope_matches(Scope(site="B"), v)
    assert not scope_matches(Scope(situation_type="spill"), v)
    assert not scope_matches(Scope(zone_class="ward"), v)
    assert not scope_matches(Scope(door_id="D-99"), v)


def test_local_rules_filtering():
    g = Genome(version=1, rulebook=[
        rule("ok_site", Scope(site="A", situation_type="fire_door")),
        rule("global", Scope(situation_type="fire_door")),
        rule("other_site", Scope(site="B", situation_type="fire_door")),
        rule("retired", Scope(site="A"), status="retired"),
        rule("wrong_type", Scope(situation_type="spill")),
    ])
    ctx = LocalMemory().get_context(view(), g)
    assert sorted(r.id for r in ctx.rules) == ["global", "ok_site"]
    assert ctx.backend == "local"


def test_local_episodes_site_and_k():
    m = LocalMemory()
    for i in range(5):
        m.add_takeover(tk(f"a{i}"))
    m.add_takeover(tk("b0", site="B"))
    m.add_takeover(tk("a_spill", typ="spill", obs="wet spill", tip="clean"))
    g = Genome(version=0, context_policy=ContextPolicy(k=3))
    ctx = m.get_context(view(), g)
    assert len(ctx.episodes) == 3
    assert all(e["situation_type"] == "fire_door" for e in ctx.episodes)
    assert "embedding" not in ctx.episodes[0]
    m.add_fact("A", "fact1")
    m.add_fact("B", "fact2")
    g2 = Genome(version=0, context_policy=ContextPolicy(sources=["semantic"]))
    assert m.get_context(view(), g2).facts == ["fact1"]


def _mongo():
    try:
        from iz.store.db import mongo_available
        return mongo_available()
    except Exception:
        return False


needs_mongo = pytest.mark.skipif(not _mongo(), reason="mongo unavailable")


@needs_mongo
def test_atlas_roundtrip_fake_embedder():
    from iz.store.db import EMB_CACHE, RULES, TAKEOVERS, get_db
    db = get_db("iz_test")
    mem = AtlasMemory(db, FakeEmbedder())
    t = tk("rt-" + uuid.uuid4().hex[:6], site="RT")
    r = rule("rt-rule-" + uuid.uuid4().hex[:6], Scope(situation_type="fire_door"), text="push gently")
    before = db[EMB_CACHE].count_documents({})
    try:
        mem.add_takeover(t)
        mem.add_rule(r, None)
        mem.add_fact("RT", "hello")
        d = db[TAKEOVERS].find_one({"_id": t.id})
        assert len(d["embedding"]) == 1024 and d["outcome"] == "takeover" and d["text"].startswith("fire_door")
        rd = db[RULES].find_one({"_id": r.id})
        assert rd["site"] == "global" and rd["situation_type"] == "fire_door"
        assert db[EMB_CACHE].count_documents({}) == before  # fake embedder never touches the cache
    finally:
        db[TAKEOVERS].delete_one({"_id": t.id})
        db[RULES].delete_one({"_id": r.id})
        db["facts"].delete_many({"site": "RT"})


@needs_mongo
def test_get_context_never_raises_without_indexes():
    from iz.store.db import get_client
    name = "iz_test_noidx"
    db = get_client()[name]
    try:
        mem = AtlasMemory(db, FakeEmbedder())
        mem.add_takeover(tk("n1"))
        g = Genome(version=1, rulebook=[rule("r1", Scope(situation_type="fire_door"))])
        ctx = mem.get_context(view(), g)
        assert [r.id for r in ctx.rules] == ["r1"]
        assert ctx.episodes == []  # degraded (error) or empty, never raises
    finally:
        get_client().drop_database(name)


@pytest.mark.skipif(not (_mongo() and os.getenv("VOYAGE_API_KEY")), reason="needs mongo + voyage")
def test_voyage_embedder_populates_cache():
    from iz.store.db import EMB_CACHE, get_db
    from iz.store.voyage import VoyageEmbedder
    db = get_db("iz_test")
    e = VoyageEmbedder(db)
    text = "cache probe " + uuid.uuid4().hex
    v = e.embed([text], "document")[0]
    assert len(v) == 1024
    assert db[EMB_CACHE].find_one({"_id": e._key(text, "document")}) is not None
    assert e.embed([text], "document")[0] == v
    db[EMB_CACHE].delete_one({"_id": e._key(text, "document")})
