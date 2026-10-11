import os
import time
import uuid

import pytest
from pymongo.errors import OperationFailure

import iz.store.hybrid as hy
from iz.store.hybrid import hybrid_search, rrf_fuse
from iz.store.voyage import FakeEmbedder


def D(i):
    return {"_id": i, "text": f"doc {i}"}


def test_both_lists_outranks():
    out = rrf_fuse([[D("a"), D("b")], [D("c"), D("b")]])
    assert out[0]["_id"] == "b"
    assert len(out) == 3


def test_k60_numeric():
    out = rrf_fuse([[D("a")], [D("a")]])
    assert out[0]["_rrf"] == pytest.approx(2 / 61)


def test_weights_change_order():
    lists = [[D("a")], [D("b")]]
    assert rrf_fuse(lists, weights=[1.0, 0.1])[0]["_id"] == "a"
    assert rrf_fuse(lists, weights=[0.1, 1.0])[0]["_id"] == "b"


def test_ties_deterministic():
    out = rrf_fuse([[D("z")], [D("m")]])
    assert [d["_id"] for d in out] == ["m", "z"]


class FakeColl:
    name = "fake"

    def __init__(self, vec, txt):
        self.vec, self.txt = vec, txt
        self.calls = []

    def aggregate(self, pipe):
        stage = next(iter(pipe[0]))
        self.calls.append(stage)
        if stage == "$rankFusion":
            raise OperationFailure("Unrecognized pipeline stage name: '$rankFusion'")
        return iter(self.vec if stage == "$vectorSearch" else self.txt)


def test_fallback_rrf_python_and_rerank(monkeypatch):
    monkeypatch.setattr(hy, "_PATH", None)
    vec = [{"_id": "1", "text": "fire door red"}, {"_id": "2", "text": "cart"}, {"_id": "3", "text": "spill"}]
    txt = [{"_id": "3", "text": "spill"}, {"_id": "1", "text": "fire door red"}]
    c = FakeColl(vec, txt)
    res = hybrid_search(c, "fire door", [0.0], {"site": "A"}, 2, vector_index="v", text_index="t",
                        text_paths=["text"], embedder=FakeEmbedder())
    assert res.path == "rrf_python"
    assert len(res.docs) == 2
    assert res.docs[0]["_id"] == "1" and "_rerank" in res.docs[0]
    assert res.hits["vector"] == 3 and res.hits["text"] == 2 and res.hits["fused"] == 3
    # fused order without rerank: 1 (both, top vec) first
    res2 = hybrid_search(c, "fire door", [0.0], {"site": "A"}, 3, vector_index="v", text_index="t",
                         text_paths=["text"], rerank=False)
    assert [d["_id"] for d in res2.docs] == ["1", "3", "2"]
    assert c.calls.count("$rankFusion") == 1  # cached after first failure
    assert hy.last_path() == "rrf_python"


def test_single_leg_degrades(monkeypatch):
    monkeypatch.setattr(hy, "_PATH", "rrf_python")

    class C(FakeColl):
        def aggregate(self, pipe):
            if next(iter(pipe[0])) == "$search":
                raise OperationFailure("index not ready")
            return super().aggregate(pipe)

    res = hybrid_search(C([D("a")], []), "q", [0.0], {}, 3, vector_index="v", text_index="t",
                        text_paths=["text"], rerank=False)
    assert res.path == "vector_only" and len(res.docs) == 1


def _live_ready():
    try:
        from iz.store.db import get_db, mongo_available
        if not (mongo_available() and os.getenv("VOYAGE_API_KEY")):
            return False
        idx = {i["name"]: i.get("queryable") for i in get_db("iz_test")["takeovers"].list_search_indexes()}
        return bool(idx.get("takeovers_vec") and idx.get("takeovers_text"))
    except Exception:
        return False


@pytest.mark.skipif(not _live_ready(), reason="live mongo/voyage/search indexes unavailable")
def test_live_hybrid():
    from datetime import datetime, timezone
    from iz.store.db import TAKEOVERS, TAKEOVERS_TEXT, TAKEOVERS_VEC, get_db
    from iz.store.memory import AtlasMemory
    from iz.store.voyage import VoyageEmbedder
    from iz.types import Action, OutcomeStatus, Takeover
    db = get_db("iz_test")
    emb = VoyageEmbedder(db)
    mem = AtlasMemory(db, emb)
    site = "HY" + uuid.uuid4().hex[:6]
    ids = []
    for n, (typ, obs, tip) in enumerate([
            ("fire_door", "red FIRE DOOR D-17 closed", "push with 25N, never hold it open"),
            ("cart_blocking", "supply cart in corridor", "push cart gently 30N"),
            ("spill", "wet floor spill near kitchen", "call cleaning, slow down")]):
        t = Takeover(id=f"{site}-{n}", run_id="r", mission_id="m", robot_id="R1", site=site, day=1, shift=1,
                     situation_id=f"s{n}", situation_type=typ, zone="z", zone_class="corridor",
                     observable=obs, tried=[Action(tool="wait")], outcome=OutcomeStatus.TAKEOVER,
                     reason="x", operator_tip=tip, policy_version=0, ts=datetime.now(timezone.utc))
        mem.add_takeover(t)
        ids.append(t.id)
    try:
        qv = emb.embed(["fire door red"], "query")[0]
        res = None
        for _ in range(30):
            res = hybrid_search(db[TAKEOVERS], "fire door red", qv, {"site": site}, 3,
                                vector_index=TAKEOVERS_VEC, text_index=TAKEOVERS_TEXT,
                                text_paths=["observable", "operator_tip", "zone"], embedder=emb)
            if res.docs:
                break
            time.sleep(2)
        print("hybrid path:", res.path, "rerank model:", emb.rerank_model_used)
        assert res.docs
    finally:
        db[TAKEOVERS].delete_many({"_id": {"$in": ids}})
