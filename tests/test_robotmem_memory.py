"""Live Atlas tests (DB iz_test); skipped when Atlas is unreachable."""
from __future__ import annotations

import time
import uuid

import pytest

from iz.store.db import get_db, mongo_available

pytestmark = pytest.mark.skipif(not mongo_available(), reason="Atlas unreachable")

from iz.robotmem import memory as M  # noqa: E402


@pytest.fixture(scope="module")
def mem():
    db = get_db("iz_test")
    m = M.EnvMemory(db)
    M.ensure_search_indexes(db)
    m.indexes_ready = M.wait_queryable(db, timeout_s=150, poll_s=5)
    yield m
    for c in (M.EPISODES, M.FACTS, M.PROCEDURES):
        db[c].delete_many({"run": m.run})


@pytest.fixture(autouse=True)
def _run(mem):
    mem.run = getattr(mem, "run", None) or uuid.uuid4().hex[:8]


def _env(mem, name, venue, text):
    return {"env_id": f"{mem.run}-{name}", "name": name, "venue": venue, "text": text}


def _wait_for(fn, timeout=40):
    end = time.time() + timeout
    out = fn()
    while not out and time.time() < end:
        time.sleep(3)
        out = fn()
    return out


def test_episode_fact_round_trip(mem):
    env = _env(mem, "rt", "pharmacy", "pharmacy at rush hour with heavy cart")
    eid = mem.remember_episode(env, {"success": False, "status": "fail", "failure": "cart_tipped on turn",
                                     "operator_tip": "slow the turn near the counter", "seed": 3,
                                     "summary": {"max_tilt": 0.4}}, params_version=2)
    assert eid
    assert mem.remember_facts(env, ["carts here weigh about 32 kg", "carts here weigh about 32 kg"]) == 1
    rec = mem.recall(env, k=3)
    assert [f["fact"] for f in rec["facts"]] == ["carts here weigh about 32 kg"]
    doc = mem.db[M.EPISODES].find_one({"_id": __import__("bson").ObjectId(eid)})
    assert doc["taxonomy_node"] == "cart_tipped" and doc["params_version"] == 2
    assert len(doc["embedding"]) == 1024


def test_taxonomy_classify_and_counts(mem):
    assert mem.classify("cart_tipped") == "cart_tipped"
    assert mem.classify({"reason": "cart slid on the spill"}) == "cart_slide_on_spill"
    assert mem.classify("person_contact at aisle 2") == "person_contact"
    assert mem.classify("ran out of time: timeout") == "timeout"
    assert mem.classify("weird") == "manoeuvre"
    assert mem.classify(None) is None
    before = {t["_id"]: t["examples_count"] for t in mem.db[M.TAXONOMY].find()}
    assert len(before) >= 14 and "manoeuvre" in before
    chain = mem.bump_taxonomy("collision_shelf")
    assert chain == ["collision_shelf", "collision", "manoeuvre"]
    after = {t["_id"]: t["examples_count"] for t in mem.db[M.TAXONOMY].find()}
    assert all(after[n] == before[n] + 1 for n in chain)


def test_warm_start_default_when_empty_and_clamped(mem):
    from iz.shop.controller import DEFAULT_PARAMS, PARAM_BOUNDS
    empty_env = _env(mem, "lonely", "zzz-nothing", "unique unseen place")
    # empty-memory behaviour is exercised via an empty recall payload
    params, prov = mem.warm_start(empty_env, rec={"procedures": []})
    assert params == DEFAULT_PARAMS and prov["source"] == "default"

    src = _env(mem, "src", "warehouse", "warehouse aisle with narrow shelves and fast workers")
    wild = {"cruise_speed": 99.0, "yield_distance_m": -5.0, "brake_distance_m": 0.9, "bogus": 1}
    mem.learn_procedure(src, wild, {"heldout_rate": 0.8, "n": 10}, taxonomy_nodes_fixed=["collision_shelf"])
    tgt = _env(mem, "tgt", "warehouse", "a warehouse aisle with narrow shelves and fast forklifts")

    def go():
        r = mem.recall(tgt, k=3)
        return r if any(p["env_id"] == src["env_id"] for p in r["procedures"]) else None
    rec = _wait_for(go)
    assert rec, "procedure from other environment not recalled"
    # same-env procedures are never returned for the env itself
    self_rec = mem.recall(src, k=3)
    assert all(p["env_id"] != src["env_id"] for p in self_rec["procedures"])
    # the shared test DB accumulates procedures from earlier runs; isolate this run's source procedure
    rec = {**rec, "procedures": [p for p in rec["procedures"] if p["env_id"] == src["env_id"]]}
    params, prov = mem.warm_start(tgt, rec=rec)
    lo, hi = PARAM_BOUNDS["cruise_speed"]
    assert params["cruise_speed"] == hi
    assert params["yield_distance_m"] == PARAM_BOUNDS["yield_distance_m"][0]
    assert params["brake_distance_m"] == 0.9 and "bogus" not in params
    assert prov["source"] == "procedure" and prov["from_env_id"] == src["env_id"]


def test_lineage_graphlookup(mem):
    env = _env(mem, "lin", "grocery", "small grocery")
    a = mem.learn_procedure(env, {"cruise_speed": 1.0}, {"heldout_rate": 0.5, "n": 8})
    b = mem.learn_procedure(env, {"cruise_speed": 0.9}, {"heldout_rate": 0.7, "n": 8}, parent=a)
    anc = mem.lineage(b)
    assert [str(x["_id"]) for x in anc] == [a]


def test_context_under_budget_and_hybrid_path(mem):
    env = _env(mem, "ctx", "warehouse", "warehouse aisle narrow shelves")
    mem.remember_facts(env, [f"fact number {i} about aisle width and worker speed" for i in range(40)])
    tgt = _env(mem, "ctx2", "warehouse", "warehouse aisle narrow shelves fast workers")
    mem.remember_facts(tgt, [f"fact number {i} about aisle width and worker speed" for i in range(40)])
    rec = mem.recall(tgt, k=3)
    text = mem.context_text(rec, budget_tokens=120)
    assert text and mem.last_context_tokens <= 120
    big = mem.context_text(rec, budget_tokens=600)
    assert mem.last_context_tokens <= 600
    assert rec["path"], "recall must report its retrieval path"
    print("hybrid path:", rec["path"], "| episode path:", rec["episode_path"],
          "| indexes_ready:", mem.indexes_ready, "| ctx tokens:", mem.last_context_tokens)
    if mem.indexes_ready:
        assert rec["path"] in ("rankFusion", "rrf_python", "vector_only", "text_only") or \
            rec["path"].endswith("cosine_fallback")
