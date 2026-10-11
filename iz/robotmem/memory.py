"""Per-environment robot memory on MongoDB Atlas: episodic, semantic, taxonomic, procedural.

Collections: env_episodes, env_facts, taxonomy (shared), env_procedures.
Retrieval: hybrid ($rankFusion vector+text, then Voyage rerank) via iz.store.hybrid, with a
Python-cosine fallback over a capped find when the search indexes are not queryable.
"""
from __future__ import annotations

import logging
import math
import os
import time
from datetime import datetime, timezone
from typing import Any

from bson import ObjectId
from pymongo.database import Database
from pymongo.operations import SearchIndexModel

from iz.store.db import EMBED_DIMS, get_db
from iz.store.voyage import Embedder, FakeEmbedder

log = logging.getLogger(__name__)

EPISODES, FACTS, TAXONOMY, PROCEDURES = "env_episodes", "env_facts", "taxonomy", "env_procedures"
CARDS = "robot_cards"
FILTER_FIELDS = ("venue", "env_id", "kind")

# collection -> (vector index, text index, text fields)
INDEXED = {
    EPISODES: ("env_episodes_vec", "env_episodes_text", ["text", "failure", "operator_tip", "venue"]),
    FACTS: ("env_facts_vec", "env_facts_text", ["text", "fact", "venue"]),
    PROCEDURES: ("env_procedures_vec", "env_procedures_text", ["text", "venue"]),
}

TAXONOMY_SEED = [
    ("manoeuvre", None, "Root of all cart-pushing and shop-floor manoeuvre failures"),
    ("cart_dynamics", "manoeuvre", "Cart behaves badly: tips, slides, drifts"),
    ("cart_tipped", "cart_dynamics", "Cart tipped over while turning or braking"),
    ("cart_slide_on_spill", "cart_dynamics", "Cart slid or lost traction on a floor spill"),
    ("clearance", "manoeuvre", "Not enough distance kept from people or obstacles"),
    ("too_close_to_person", "clearance", "Robot or cart came closer to a person than allowed"),
    ("person_contact", "clearance", "Robot or cart touched a person"),
    ("collision", "manoeuvre", "Unintended contact with the environment"),
    ("collision_shelf", "collision", "Cart or robot hit a shelf or fixture"),
    ("manipulation", "manoeuvre", "Arm or grasp failures"),
    ("dropped_product", "manipulation", "Product dropped while carrying or placing"),
    ("missed_grasp", "manipulation", "Grasp missed the product"),
    ("time", "manoeuvre", "Task not finished in time"),
    ("timeout", "time", "Episode timed out before completion"),
]
# keyword -> node, first match wins
_KEYWORDS = [
    ("tip", "cart_tipped"), ("spill", "cart_slide_on_spill"), ("slide", "cart_slide_on_spill"),
    ("slip", "cart_slide_on_spill"), ("person_contact", "person_contact"), ("touch", "person_contact"),
    ("too_close", "too_close_to_person"), ("clearance", "too_close_to_person"),
    ("person", "too_close_to_person"), ("shelf", "collision_shelf"), ("collision", "collision"),
    ("drop", "dropped_product"), ("grasp", "missed_grasp"), ("miss", "missed_grasp"),
    ("timeout", "timeout"), ("time", "timeout"),
]


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _est_tokens(text: str) -> int:
    return max(1, math.ceil(len(text) / 4))


def _controller_consts() -> tuple[dict, dict]:
    try:
        from iz.shop.controller import DEFAULT_PARAMS, PARAM_BOUNDS
        return dict(DEFAULT_PARAMS), dict(PARAM_BOUNDS)
    except Exception:  # noqa: BLE001
        return {}, {}


def env_fields(env: Any) -> dict:
    """Normalise an env spec (dict / object / plain text) to env_id, env_name, venue, text."""
    if isinstance(env, str):
        return {"env_id": "text:" + str(abs(hash(env)) % 10**8), "env_name": env[:60],
                "venue": "", "text": env}
    if isinstance(env, dict):
        get = lambda k, d=None: env.get(k, d)  # noqa: E731
    else:
        get = lambda k, d=None: getattr(env, k, d)  # noqa: E731
    env_id = get("env_id") or get("id") or get("name") or "env"
    name = get("env_name") or get("name") or str(env_id)
    venue = get("venue") or get("kind") or ""
    text = get("text") or get("prompt") or get("description") or ""
    return {"env_id": str(env_id), "env_name": str(name), "venue": str(venue), "text": str(text)}


def _cos(a: list[float], b: list[float]) -> float:
    num = sum(x * y for x, y in zip(a, b))
    da = math.sqrt(sum(x * x for x in a)) or 1.0
    db = math.sqrt(sum(y * y for y in b)) or 1.0
    return num / (da * db)


# ---------------------------------------------------------------- schema / indexes
def _validator(required: list[str]) -> dict:
    return {"$jsonSchema": {"bsonType": "object", "required": required,
                            "properties": {"env_id": {"bsonType": "string"}}}}


VALIDATORS = {
    EPISODES: _validator(["env_id", "kind", "status", "ts"]),
    FACTS: _validator(["env_id", "kind", "fact", "source"]),
    PROCEDURES: _validator(["env_id", "kind", "params", "version", "scope"]),
    TAXONOMY: {"$jsonSchema": {"bsonType": "object", "required": ["node", "examples_count"]}},
}


def _vec_def() -> dict:
    return {"fields": [{"type": "vector", "path": "embedding", "numDimensions": EMBED_DIMS,
                        "similarity": "cosine"}] + [{"type": "filter", "path": f} for f in FILTER_FIELDS]}


def _text_def(fields: list[str]) -> dict:
    f = {k: {"type": "string"} for k in fields}
    for k in FILTER_FIELDS:
        f[k] = {"type": "token"}
    return {"analyzer": "lucene.standard", "mappings": {"dynamic": False, "fields": f}}


def ensure_collections(db: Database) -> list[str]:
    have = set(db.list_collection_names())
    made = []
    for name, val in VALIDATORS.items():
        opts = dict(validator=val, validationLevel="moderate", validationAction="error")
        if name in have:
            db.command("collMod", name, **opts)
        else:
            db.create_collection(name, **opts)
            made.append(name)
    if CARDS not in have:
        db.create_collection(CARDS)
    db[EPISODES].create_index([("env_id", 1), ("ts", -1)], name="env_ts")
    db[FACTS].create_index([("env_id", 1)], name="env")
    db[PROCEDURES].create_index([("env_id", 1), ("version", -1)], name="env_ver")
    db[CARDS].create_index([("name", 1), ("hash", 1)], unique=True, name="name_hash")
    return made


def seed_taxonomy(db: Database, embedder: Embedder | None = None) -> int:
    n = 0
    texts = [f"{node}: {desc}" for node, _, desc in TAXONOMY_SEED]
    embs = embedder.embed(texts, "document") if embedder else [None] * len(texts)
    for (node, parent, desc), emb in zip(TAXONOMY_SEED, embs):
        doc = {"node": node, "parent": parent, "description": desc, "examples_count": 0}
        if emb is not None:
            doc["embedding"] = emb
        r = db[TAXONOMY].update_one({"_id": node}, {"$setOnInsert": doc}, upsert=True)
        n += 1 if r.upserted_id else 0
    return n


def ensure_search_indexes(db: Database) -> dict[str, str]:
    out = {}
    for coll, (vname, tname, tfields) in INDEXED.items():
        existing = {i["name"]: i for i in db[coll].list_search_indexes()}
        for name, typ, d in ((vname, "vectorSearch", _vec_def()), (tname, "search", _text_def(tfields))):
            if name not in existing:
                db[coll].create_search_index(SearchIndexModel(definition=d, name=name, type=typ))
                out[name] = "created"
            else:
                cur = existing[name].get("latestDefinition") or existing[name].get("definition")
                if cur != d:
                    db[coll].update_search_index(name, d)
                    out[name] = "updated"
                else:
                    out[name] = "exists"
    return out


def search_status(db: Database) -> dict[str, bool]:
    st = {}
    for coll, (vname, tname, _) in INDEXED.items():
        have = {i["name"]: bool(i.get("queryable")) for i in db[coll].list_search_indexes()}
        st[vname] = have.get(vname, False)
        st[tname] = have.get(tname, False)
    return st


def wait_queryable(db: Database, timeout_s: int = 120, poll_s: float = 4.0, log_fn=None) -> bool:
    deadline = time.time() + timeout_s
    while True:
        st = search_status(db)
        if log_fn:
            log_fn(st)
        if all(st.values()):
            return True
        if time.time() > deadline:
            return False
        time.sleep(poll_s)


def _default_embedder(db: Database) -> Embedder:
    if os.getenv("VOYAGE_API_KEY"):
        try:
            from iz.store.voyage import VoyageEmbedder
            return VoyageEmbedder(db)
        except Exception as e:  # noqa: BLE001
            log.warning("Voyage unavailable (%s); using FakeEmbedder", type(e).__name__)
    return FakeEmbedder()


# ---------------------------------------------------------------- EnvMemory
class EnvMemory:
    def __init__(self, db: Database | None = None, embedder: Embedder | None = None,
                 ensure: bool = True) -> None:
        self.db = db if db is not None else get_db()
        self.embedder = embedder or _default_embedder(self.db)
        self.last_path: str | None = None
        self.scope: str | None = None  # env_id prefix; recall only sees docs from this namespace
        self.last_context_tokens = 0
        if ensure:
            ensure_collections(self.db)
            if self.db[TAXONOMY].estimated_document_count() < len(TAXONOMY_SEED):
                seed_taxonomy(self.db, self.embedder)

    def _embed(self, text: str, kind: str = "document") -> list[float]:
        return self.embedder.embed([text], kind)[0]  # type: ignore[arg-type]

    # -- taxonomy
    def classify(self, failure: Any) -> str | None:
        """Map an episode failure (str or dict) to a taxonomy node; None if no failure."""
        if not failure:
            return None
        s = (" ".join(str(v) for v in failure.values()) if isinstance(failure, dict)
             else str(failure)).lower()
        for n, _, _ in sorted(TAXONOMY_SEED, key=lambda t: -len(t[0])):
            if n != "time" and n in s:
                return n
        for kw, node in _KEYWORDS:
            if kw in s:
                return node
        return "manoeuvre"

    def bump_taxonomy(self, node: str) -> list[str]:
        chain, cur = [], node
        while cur:
            doc = self.db[TAXONOMY].find_one({"_id": cur}, {"parent": 1})
            if not doc:
                break
            chain.append(cur)
            cur = doc.get("parent")
        if chain:
            self.db[TAXONOMY].update_many({"_id": {"$in": chain}}, {"$inc": {"examples_count": 1}})
        return chain

    def taxonomy_stats(self, min_count: int = 0) -> list[dict]:
        return list(self.db[TAXONOMY].find(
            {"examples_count": {"$gt": min_count}}, {"embedding": 0}).sort("examples_count", -1))

    # -- episodic
    def remember_episode(self, env: Any, result: dict, params_version: Any = None) -> str:
        e = env_fields(env)
        failure = result.get("failure")
        failure_s = str(failure) if isinstance(failure, (dict, list)) else (failure or "")
        tip = result.get("operator_tip") or ""
        node = self.classify(failure) if failure_s else None
        text = f"{e['venue']} {e['text']} {failure_s} {tip}".strip()
        doc = {"kind": "episode", "env_id": e["env_id"], "env_name": e["env_name"], "venue": e["venue"],
               "seed": result.get("seed"), "params_version": params_version,
               "status": str(result.get("status", "success" if result.get("success") else "failed")),
               "success": bool(result.get("success")), "failure": failure_s,
               "taxonomy_node": node, "summary": result.get("summary") or {},
               "operator_tip": tip, "text": text, "ts": _now(), "embedding": self._embed(text)}
        _id = self.db[EPISODES].insert_one(doc).inserted_id
        if node:
            self.bump_taxonomy(node)
        return str(_id)

    # -- semantic
    def remember_facts(self, env: Any, facts: list, source: str = "learned") -> int:
        e = env_fields(env)
        items = []
        for f in facts:
            if isinstance(f, dict):
                items.append((f.get("fact") or f.get("text") or "", f.get("source", source)))
            else:
                items.append((str(f), source))
        items = [(f, s) for f, s in items if f]
        if not items:
            return 0
        embs = self.embedder.embed([f"{e['venue']} {f}" for f, _ in items], "document")
        n = 0
        for (fact, src), emb in zip(items, embs):
            r = self.db[FACTS].update_one(
                {"env_id": e["env_id"], "fact": fact},
                {"$setOnInsert": {"kind": "fact", "env_id": e["env_id"], "env_name": e["env_name"],
                                  "venue": e["venue"], "fact": fact, "text": f"{e['venue']} {fact}",
                                  "source": src if src in ("spec", "learned") else "learned",
                                  "ts": _now(), "embedding": emb}}, upsert=True)
            n += 1 if r.upserted_id else 0
        return n

    # -- procedural
    def learn_procedure(self, env: Any, params: dict, metrics: dict | None = None,
                        parent: Any = None, scope: str = "env",
                        taxonomy_nodes_fixed: list[str] | None = None) -> str:
        e = env_fields(env)
        metrics = metrics or {}
        last = self.db[PROCEDURES].find_one({"env_id": e["env_id"]}, {"version": 1},
                                            sort=[("version", -1)])
        version = (last["version"] + 1) if last else 1
        fixed = list(taxonomy_nodes_fixed or [])
        desc = (f"{e['venue']} {e['env_name']} {e['text']} fixes {' '.join(fixed)} "
                f"params {' '.join(f'{k}={v}' for k, v in params.items())}").strip()
        pid = parent
        if isinstance(pid, str):
            try:
                pid = ObjectId(pid)
            except Exception:  # noqa: BLE001
                pass
        doc = {"kind": "procedure", "env_id": e["env_id"], "env_name": e["env_name"],
               "venue": e["venue"], "scope": scope, "params": dict(params), "version": version,
               "parent": pid, "heldout_rate": metrics.get("heldout_rate"), "n": metrics.get("n"),
               "metrics": metrics, "promoted_at": _now(), "taxonomy_nodes_fixed": fixed,
               "text": desc, "embedding": self._embed(desc)}
        return str(self.db[PROCEDURES].insert_one(doc).inserted_id)

    def lineage(self, procedure_id: Any) -> list[dict]:
        pid = ObjectId(procedure_id) if isinstance(procedure_id, str) else procedure_id
        res = list(self.db[PROCEDURES].aggregate([
            {"$match": {"_id": pid}},
            {"$graphLookup": {"from": PROCEDURES, "startWith": "$parent", "connectFromField": "parent",
                              "connectToField": "_id", "as": "ancestors", "depthField": "depth"}},
            {"$project": {"ancestors.embedding": 0, "embedding": 0}}]))
        return res[0]["ancestors"] if res else []

    # -- retrieval
    def _search(self, coll: str, query: str, kind: str, k: int) -> tuple[list[dict], str]:
        """Hybrid search; Python cosine fallback. Returns (docs, path)."""
        vname, tname, _ = INDEXED[coll]
        qv = self._embed(query, "query")
        path = ""
        try:
            from iz.store.hybrid import hybrid_search
            r = hybrid_search(self.db[coll], query, qv, {"kind": kind}, k, vector_index=vname,
                              text_index=tname, text_paths=["text"], embedder=self.embedder,
                              rerank_text_field="text", limit=max(20, k * 4))
            path = r.path
            if r.docs:
                return r.docs, path
        except Exception as ex:  # noqa: BLE001
            log.info("hybrid failed (%s); python cosine fallback", str(ex)[:100])
            path = "hybrid_error"
        cands = list(self.db[coll].find({"kind": kind}).sort("_id", -1).limit(500))
        for d in cands:
            d["_cos"] = _cos(qv, d.get("embedding") or [0.0] * EMBED_DIMS)
            d.pop("embedding", None)
        cands.sort(key=lambda d: -d["_cos"])
        return cands[: max(k * 4, 20)], f"{path}+cosine_fallback"

    def recall(self, env: Any, k: int = 3) -> dict:
        e = env_fields(env)
        query = f"{e['venue']} {e['env_name']} {e['text']}".strip() or e["env_id"]
        pdocs, ppath = self._search(PROCEDURES, query, "procedure", k * 4 + 4)
        sc = self.scope
        inscope = lambda d: not sc or str(d.get("env_id", "")).startswith(sc)  # noqa: E731
        procs = [d for d in pdocs if d.get("env_id") != e["env_id"] and inscope(d)][:k]
        edocs, epath = self._search(EPISODES, query, "episode", k * 4 + 4)
        eps = [d for d in edocs if (d.get("operator_tip") or d.get("failure")) and inscope(d)][:k]
        facts = list(self.db[FACTS].find({"env_id": e["env_id"]}, {"embedding": 0}).limit(20))
        for lst in (procs, eps):
            for d in lst:
                d.pop("embedding", None)
        self.last_path = ppath
        return {"procedures": procs, "facts": facts, "episodes": eps,
                "taxonomy": self.taxonomy_stats(), "path": ppath, "episode_path": epath, "env": e}

    def warm_start(self, env: Any, rec: dict | None = None) -> tuple[dict, dict]:
        defaults, bounds = _controller_consts()
        rec = rec or self.recall(env, k=3)
        procs = rec.get("procedures") or []
        if not procs:
            return dict(defaults), {"source": "default", "reason": "memory empty"}
        top = procs[0]
        params = dict(defaults)
        for key, v in (top.get("params") or {}).items():
            if key not in bounds or not isinstance(v, (int, float)):
                continue
            lo, hi = bounds[key]
            params[key] = min(max(v, lo), hi)
        return params, {"source": "procedure", "procedure_id": str(top["_id"]),
                        "from_env": top.get("env_name"), "from_env_id": top.get("env_id"),
                        "venue": top.get("venue"), "heldout_rate": top.get("heldout_rate"),
                        "version": top.get("version"), "path": rec.get("path")}

    def context_text(self, rec: dict, budget_tokens: int = 600) -> str:
        lines = []
        for p in rec.get("procedures", []):
            lines.append(f"[proc] from '{p.get('env_name')}' ({p.get('venue')}), held-out "
                         f"{p.get('heldout_rate')}: {p.get('params')} fixed {p.get('taxonomy_nodes_fixed')}")
        for f in rec.get("facts", [])[:8]:
            lines.append(f"[fact] {f.get('fact')}")
        for ep in rec.get("episodes", []):
            lines.append(f"[past] {ep.get('failure')} -> {ep.get('operator_tip') or ''}"[:200])
        tax = [f"{t['node']}={t['examples_count']}" for t in rec.get("taxonomy", [])
               if t.get("examples_count") and t.get("parent")][:8]
        if tax:
            lines.append("[failure taxonomy counts] " + ", ".join(tax))
        out, used = [], 0
        for ln in lines:
            t = _est_tokens(ln)
            if used + t > budget_tokens:
                break
            out.append(ln)
            used += t
        text = "\n".join(out)
        self.last_context_tokens = _est_tokens(text) if text else 0
        return text

    def clear_env(self, env_ids: list[str]) -> dict:
        """Delete only docs belonging to the given env_ids (taxonomy untouched)."""
        return {c: self.db[c].delete_many({"env_id": {"$in": list(env_ids)}}).deleted_count
                for c in (EPISODES, FACTS, PROCEDURES)}
