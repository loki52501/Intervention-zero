"""/evidence: live proof page showing how the robot learns through MongoDB Atlas and Raindrop.

Every number comes from a live query; each section is isolated so one failure never blanks the page.
"""
from __future__ import annotations

import logging
import os
import re
import threading
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

from bson import ObjectId
from fastapi import APIRouter
from fastapi.responses import FileResponse

log = logging.getLogger("iz.ui.evidence")
STATIC = Path(__file__).parent / "static"
router = APIRouter()
SECTIONS = ("loop", "safety", "memory", "recall", "raindrop")
CACHE_TTL = 2.0
_cache: dict[str, Any] = {"t": 0.0, "data": None}
_lock = threading.Lock()
_probe_cache: dict[str, Any] = {"t": 0.0, "data": None}
_rd_cache: dict[str, Any] = {"t": 0.0, "data": None}


def _clean(v: Any) -> Any:
    if isinstance(v, dict):
        return {str(k): _clean(x) for k, x in v.items() if k != "embedding"}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v[:50]]
    if isinstance(v, datetime):
        return v.isoformat()
    if isinstance(v, ObjectId):
        return str(v)
    if isinstance(v, bytes):
        return "<bytes>"
    if isinstance(v, str):
        return v[:300]
    if isinstance(v, (int, float, bool)) or v is None:
        return v
    return str(v)[:300]


def _err(e: Exception) -> dict:
    return {"ok": False, "error": f"{type(e).__name__}: {str(e)[:200]}"}


def _safe(fn: Callable[[], dict]) -> dict:
    try:
        out = fn()
        out.setdefault("ok", True)
        return out
    except Exception as e:  # noqa: BLE001
        return _err(e)


def _newest(coll: Any, sort_field: str = "_id", proj: dict | None = None) -> dict | None:
    d = coll.find_one(sort=[(sort_field, -1)], projection=proj)
    return _clean(d) if d else None


# ------------------------------------------------------------------ section 1
def _loop() -> dict:
    from iz.store import db as D
    db = D.get_db()
    tk_new = _newest(db[D.TAKEOVERS], "ts", {"operator_tip": 1, "site": 1, "situation_type": 1, "ts": 1})
    edits_new = _newest(db[D.EDITS], "_id", {"kind": 1, "payload": 1})
    dec_new = _newest(db[D.DECISIONS], "ts", {"verdict": 1, "reason": 1, "candidate_id": 1,
                                              "parent_version": 1, "new_version": 1, "ts": 1})
    verdicts = {str(r["_id"]): r["n"] for r in db[D.DECISIONS].aggregate(
        [{"$group": {"_id": "$verdict", "n": {"$sum": 1}}}])}
    gen_new = db[D.GENOMES].find_one(sort=[("_id", -1)], projection={"_id": 1, "parent": 1, "created_at": 1})
    ptr = db[D.ACTIVE].find_one({"_id": "robot_genome"})
    active_v = ptr.get("version") if ptr else None
    lineage: list = []
    if active_v is not None:
        r = list(db[D.GENOMES].aggregate([
            {"$match": {"_id": active_v}},
            {"$graphLookup": {"from": D.GENOMES, "startWith": "$parent", "connectFromField": "parent",
                              "connectToField": "_id", "as": "anc", "depthField": "d"}},
            {"$project": {"anc._id": 1, "anc.d": 1}}]))
        if r:
            anc = sorted(r[0].get("anc", []), key=lambda a: a["d"])
            lineage = [a["_id"] for a in anc]
            lineage.reverse()
            lineage.append(active_v)
    ue = db[D.UI_EVENTS]
    return {"cards": [
        {"key": "takeovers", "title": "Takeovers", "count": db[D.TAKEOVERS].estimated_document_count(),
         "newest": tk_new, "pill": "$jsonSchema validator"},
        {"key": "edits", "title": "Edits / candidates", "count": db[D.EDITS].estimated_document_count(),
         "newest": edits_new, "pill": "Vector Search + Atlas Search -> $rankFusion -> Voyage rerank"},
        {"key": "decisions", "title": "Decisions", "count": db[D.DECISIONS].estimated_document_count(),
         "newest": dec_new, "verdicts": verdicts, "pill": "multi-document transaction",
         "extra": {"transactions_committed": ue.count_documents({"kind": "transaction_committed"})}},
        {"key": "genomes", "title": "Genomes", "count": db[D.GENOMES].estimated_document_count(),
         "newest": _clean(gen_new) if gen_new else None, "active_version": active_v, "lineage": lineage,
         "pill": "$graphLookup"},
        {"key": "active", "title": "Active pointer", "count": None, "active_version": active_v,
         "newest": _clean(ptr) if ptr else None, "pill": "change stream",
         "extra": {"change_stream_reloads": ue.count_documents({"kind": "change_stream_reloaded"})}},
    ], "metrics_pill": "time series + $setWindowFields (metrics collection)"}


# ------------------------------------------------------------------ section 2
def _probe() -> dict:
    """Live harmless probe: insert an unsafe genome into iz_test; the server must refuse (code 121)."""
    from pymongo.errors import CollectionInvalid, WriteError
    from iz.store.db import get_db
    from iz.store.schemas import VALIDATORS
    db = get_db("iz_test")
    name = "evidence_probe_genomes"
    try:
        db.create_collection(name, validator=VALIDATORS["genomes"], validationAction="error")
    except CollectionInvalid:
        pass
    doc = {"_id": "evidence_probe", "version": 999, "guardrails": {"locked": {
        "never_enter_human_only": True, "never_hold_fire_doors": True, "always_escalate_person_down": True,
        "battery_floor_pct": 15, "max_contact_force_n": 80}},
        "rulebook": [{"params": {"allow_zone_kinds": ["human_only"]}}], "tool_policy": {}}
    try:
        db[name].insert_one(doc)
        db[name].delete_one({"_id": "evidence_probe"})
        return {"refused": False, "note": "probe was accepted (validator missing)", "db": "iz_test"}
    except WriteError as e:
        return {"refused": True, "code": e.code,
                "errmsg": str(e.details.get("errmsg") if e.details else e)[:200],
                "db": "iz_test", "attempted": 'allow_zone_kinds ["human_only"]', "source": "live probe"}


def _safety() -> dict:
    from iz.store import db as D
    db = D.get_db()
    ev = db[D.UI_EVENTS].find_one({"kind": "validator_refused"}, sort=[("_id", -1)])
    out: dict = {"event": _clean(ev) if ev else None,
                 "refusal_events": db[D.UI_EVENTS].count_documents({"kind": "validator_refused"})}
    now = time.time()
    if _probe_cache["data"] is None or now - _probe_cache["t"] > 30:
        try:
            _probe_cache["data"] = _probe()
        except Exception as e:  # noqa: BLE001
            _probe_cache["data"] = {"refused": None, "error": f"{type(e).__name__}: {str(e)[:160]}"}
        _probe_cache["t"] = now
    out["probe"] = _probe_cache["data"]
    return out


# ------------------------------------------------------------------ section 3
def _memory() -> dict:
    from iz.robotmem import memory as M
    from iz.store.db import get_db
    db = get_db()
    proj = {"embedding": 0}

    def block(coll: str) -> dict:
        c = db[coll]
        return {"collection": coll, "count": c.estimated_document_count(),
                "newest": _clean(c.find_one(sort=[("_id", -1)], projection=proj))}
    tax = list(db[M.TAXONOMY].find({}, proj))
    nodes = [{"name": t.get("name") or t.get("node") or str(t.get("_id")), "parent": t.get("parent"),
              "count": t.get("count", 0)} for t in tax]
    g = db["genomes"].find_one(sort=[("_id", -1)], projection={"rulebook": 1, "_id": 1})
    card = db[M.CARDS].find_one(sort=[("_id", -1)], projection=proj)
    cs = None
    if card:
        def n(k: str) -> int | None:
            v = card.get(k)
            return len(v) if isinstance(v, (list, dict)) else None
        cs = {"robot": str(card.get("robot") or card.get("name") or card.get("_id")),
              "sensors": n("sensors"), "actuators": n("actuators"), "fields": list(card.keys())[:12]}
    return {"episodic": block(M.EPISODES), "semantic": block(M.FACTS),
            "taxonomic": {"collection": M.TAXONOMY, "count": len(nodes), "nodes": _clean(nodes),
                          "shared_across_worlds": True},
            "procedural": {**block(M.PROCEDURES),
                           "rulebook_rules_in_newest_genome": len(g.get("rulebook") or []) if g else 0,
                           "newest_genome": g["_id"] if g else None},
            "robot_card": cs, "robot_cards_count": db[M.CARDS].estimated_document_count()}


# ------------------------------------------------------------------ section 4
def recall(q: str, k: int = 5) -> dict:
    q = (q or "").strip()
    if not q:
        return {"ok": True, "hits": [], "path": None, "rerank": None}
    try:
        from iz.store import db as D
        from iz.store.hybrid import hybrid_search
        from iz.store.voyage import FakeEmbedder, VoyageEmbedder
        db = D.get_db()
        try:
            emb: Any = VoyageEmbedder(db)
            qv = emb.embed([q], "query")[0]
            embedder = "voyage"
        except Exception:  # noqa: BLE001
            emb = FakeEmbedder()
            qv = emb.embed([q], "query")[0]
            embedder = "fake (Voyage unavailable)"
        try:
            r = hybrid_search(db[D.TAKEOVERS], q, qv, {}, k, vector_index=D.TAKEOVERS_VEC,
                              text_index=D.TAKEOVERS_TEXT, text_paths=["observable", "operator_tip", "zone"],
                              embedder=emb, rerank=True, rerank_text_field="operator_tip")
            path, docs = r.path, r.docs
        except Exception as e:  # noqa: BLE001
            rx = {"$regex": re.escape(q.split()[0]), "$options": "i"}
            docs = list(db[D.TAKEOVERS].find({"$or": [{"operator_tip": rx}, {"observable": rx}]},
                                             {"embedding": 0}).sort("_id", -1).limit(k))
            path = f"fallback_regex ({type(e).__name__})"
        hits = [{"id": d.get("_id"), "site": d.get("site"), "situation_type": d.get("situation_type"),
                 "operator_tip": d.get("operator_tip"), "score": d.get("_rerank")} for d in docs]
        reranked = any(d.get("_rerank") is not None for d in docs)
        return {"ok": True, "hits": _clean(hits), "path": path, "rerank": reranked,
                "embedder": embedder, "query": q}
    except Exception as e:  # noqa: BLE001
        return {**_err(e), "hits": [], "path": None, "rerank": None}


def _recall_section() -> dict:
    from iz.store import db as D
    from iz.store.hybrid import last_path
    n = D.get_db()[D.TAKEOVERS].estimated_document_count()
    return {"takeovers_searchable": n, "last_path": last_path(),
            "hint": "calls iz.store.hybrid.hybrid_search over takeovers"}


# ------------------------------------------------------------------ section 5
def _raindrop() -> dict:
    from iz.obs import rd
    out: dict = {"local": {"enabled": rd.enabled(), "events_sent_this_session": rd.events_sent()},
                 "fixer_reads": ("The fixer reads the 'takeover' / 'takeover_avoidable' signals through "
                                 "rd_query.top_takeover_causes before proposing (iz/loop/campaign.py, "
                                 "_raindrop_summary). Drift detection reads the 'takeover' timeseries."),
                 "degraded": False}
    now = time.time()
    if _rd_cache["data"] is None or (now - _rd_cache["t"] > 20 and not _rd_cache.get("busy")):
        _rd_cache["busy"] = True
        threading.Thread(target=_rd_refresh, daemon=True).start()
    if _rd_cache["data"] is None:
        return {**out, "loading": True, "window": "last 7 days", "total_events": 0, "events": {}, "signals": {},
                "takeover_series": []}
    return {**out, **_rd_cache["data"]}


def _rd_refresh() -> None:
    try:
        from iz.obs import rd_query
        since = datetime.now(timezone.utc) - timedelta(days=7)
        evs = rd_query._events_since(rd_query._client(), since)
        ev_names: Counter = Counter()
        sigs: Counter = Counter()
        series: Counter = Counter()
        for ev in evs:
            ev_names[getattr(ev, "event_name", None) or "unknown"] += 1
            names = rd_query._sig_names(ev)
            sigs.update(names)
            if "takeover" in names:
                ts = ev.timestamp
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                series[ts.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:00Z")] += 1
        data = {"window": "last 7 days", "total_events": len(evs), "events": dict(ev_names.most_common(12)),
                "signals": dict(sigs.most_common(16)),
                "takeover_series": [{"t": t, "count": n} for t, n in sorted(series.items())]}
    except Exception as e:  # noqa: BLE001
        msg = str(e).replace("\n", " ")[:140]
        key = os.getenv("RAINDROP_QUERY_API_KEY")
        if key:
            msg = msg.replace(key, "***")
        data = {"degraded": True, "error": f"{type(e).__name__}: {msg}",
                "message": "Raindrop degraded - falling back to MongoDB"}
        try:
            from iz.store import db as D
            data["mongo_fallback"] = {"takeovers": D.get_db()[D.TAKEOVERS].estimated_document_count()}
        except Exception as e2:  # noqa: BLE001
            data["mongo_fallback"] = _err(e2)
    _rd_cache.update(t=time.time(), data=data, busy=False)


# ------------------------------------------------------------------ routes
def build_summary() -> dict:
    fns = {"loop": _loop, "safety": _safety, "memory": _memory, "recall": _recall_section,
           "raindrop": _raindrop}
    out: dict = {k: _safe(f) for k, f in fns.items()}
    out["generated_at"] = datetime.now(timezone.utc).isoformat()
    out["db"] = os.getenv("MONGODB_DB", "intervention_zero")
    return out


@router.get("/evidence")
async def evidence_page():
    return FileResponse(STATIC / "evidence.html")


@router.get("/api/evidence/summary")
def evidence_summary():
    with _lock:
        if _cache["data"] is not None and time.time() - _cache["t"] < CACHE_TTL:
            return _cache["data"]
        data = build_summary()
        _cache.update(t=time.time(), data=data)
        return data


@router.get("/api/evidence/recall")
def evidence_recall(q: str = "", k: int = 5):
    return recall(q, max(1, min(k, 10)))
