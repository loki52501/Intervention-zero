"""Idempotent Atlas setup: collections + validators, indexes, search indexes, seed genome/playbook."""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pymongo import ASCENDING, DESCENDING  # noqa: E402
from pymongo.database import Database  # noqa: E402
from pymongo.errors import PyMongoError  # noqa: E402
from pymongo.operations import SearchIndexModel  # noqa: E402

from iz.store import db as D  # noqa: E402
from iz.store.schemas import VALIDATORS  # noqa: E402
from iz.types import FixerPlaybook, Genome  # noqa: E402

SEED_PLAYBOOK = (
    "Cluster takeovers by situation_type and zone. Propose at most max_edits scoped edits, and only "
    "for clusters with at least min_evidence takeovers. Prefer the narrowest scope that explains the "
    "evidence (site, situation_type, zone_class, door_id). Never touch locked guardrails or gate "
    "thresholds. Cite the takeover ids that justify each edit as provenance."
)

PLAIN = [D.ACTIVE, D.RULES, D.FACTS, D.RUNS, D.LLM_CACHE, D.EMB_CACHE, D.STEPS, D.METRICS_DAILY]


def _vec_def() -> dict:
    return {"fields": [
        {"type": "vector", "path": "embedding", "numDimensions": D.EMBED_DIMS, "similarity": "cosine"},
        {"type": "filter", "path": "site"},
        {"type": "filter", "path": "situation_type"},
    ]}


def _text_def(fields: dict) -> dict:
    f = {k: {"type": "string"} for k in fields}
    f["site"] = {"type": "token"}
    f["situation_type"] = {"type": "token"}
    return {"analyzer": "lucene.standard", "mappings": {"dynamic": False, "fields": f}}


SEARCH_INDEXES = [
    (D.TAKEOVERS, D.TAKEOVERS_VEC, "vectorSearch", _vec_def()),
    (D.RULES, D.RULES_VEC, "vectorSearch", _vec_def()),
    (D.TAKEOVERS, D.TAKEOVERS_TEXT, "search", _text_def(["observable", "operator_tip", "zone"])),
    (D.RULES, D.RULES_TEXT, "search", _text_def(["text"])),
]


def _ensure_search_index(db: Database, coll: str, name: str, typ: str, definition: dict) -> str:
    c = db[coll]
    existing = {i["name"]: i for i in c.list_search_indexes()}
    if name not in existing:
        c.create_search_index(SearchIndexModel(definition=definition, name=name, type=typ))
        return "created"
    cur = existing[name].get("latestDefinition") or existing[name].get("definition")
    if cur != definition:
        c.update_search_index(name, definition)
        return "updated"
    return "exists"


def setup(db: Database, wait_search: bool = True, timeout_s: int = 300) -> dict:
    report: dict = {"db": db.name, "collections": [], "search_indexes": {}, "queryable": {}}
    have = set(db.list_collection_names())

    for name, validator in VALIDATORS.items():
        opts = dict(validator=validator, validationLevel="strict", validationAction="error")
        if name in have:
            db.command("collMod", name, **opts)
        else:
            db.create_collection(name, **opts)
        report["collections"].append(name)

    for name in PLAIN:
        if name not in have:
            db.create_collection(name)
        report["collections"].append(name)

    if D.UI_EVENTS not in have:
        db.create_collection(D.UI_EVENTS, capped=True, size=16 * 1024 * 1024)
    report["collections"].append(D.UI_EVENTS)

    if D.METRICS not in have:
        db.create_collection(D.METRICS, timeseries={"timeField": "ts", "metaField": "meta",
                                                    "granularity": "minutes"})
    report["collections"].append(D.METRICS)

    db[D.STEPS].create_index([("run_id", ASCENDING), ("day", ASCENDING), ("shift", ASCENDING),
                              ("step", ASCENDING)], unique=True, name="steps_idem")
    db[D.TAKEOVERS].create_index([("site", ASCENDING), ("situation_type", ASCENDING)], name="site_type")
    db[D.DECISIONS].create_index([("ts", DESCENDING)], name="ts_desc")
    db[D.EDITS].create_index([("created", DESCENDING)], name="created_desc")

    # seed (only if missing)
    now = datetime.now(timezone.utc)
    g = {**Genome(version=0).model_dump(mode="json"), "_id": 0, "created_at": now}
    p = {**FixerPlaybook(version=0, instructions=SEED_PLAYBOOK).model_dump(mode="json"),
         "_id": 0, "created_at": now}
    db[D.GENOMES].update_one({"_id": 0}, {"$setOnInsert": g}, upsert=True)
    db[D.PLAYBOOKS].update_one({"_id": 0}, {"$setOnInsert": p}, upsert=True)
    for kind in ("robot_genome", "fixer_playbook"):
        db[D.ACTIVE].update_one({"_id": kind}, {"$setOnInsert": {"version": 0, "updated_at": now}},
                                upsert=True)

    for coll, name, typ, definition in SEARCH_INDEXES:
        report["search_indexes"][name] = _ensure_search_index(db, coll, name, typ, definition)

    if wait_search:
        deadline = time.time() + timeout_s
        names = [n for _, n, _, _ in SEARCH_INDEXES]
        while True:
            status = {}
            for coll in (D.TAKEOVERS, D.RULES):
                for i in db[coll].list_search_indexes():
                    status[i["name"]] = bool(i.get("queryable"))
            report["queryable"] = {n: status.get(n, False) for n in names}
            print("search indexes:", ", ".join(f"{n}={'ready' if v else 'building'}"
                                               for n, v in report["queryable"].items()), flush=True)
            if all(report["queryable"].values()):
                break
            if time.time() > deadline:
                report["timeout"] = True
                print("WARNING: timed out waiting for search indexes", flush=True)
                break
            time.sleep(5)
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=os.getenv("MONGODB_DB", "intervention_zero"))
    ap.add_argument("--no-wait", action="store_true")
    ap.add_argument("--timeout", type=int, default=300)
    a = ap.parse_args()
    try:
        rep = setup(D.get_db(a.db), wait_search=not a.no_wait, timeout_s=a.timeout)
    except (PyMongoError, RuntimeError) as e:
        print(f"setup failed: {type(e).__name__}: {str(e)[:150]}")
        return 1
    print(f"db={rep['db']} collections={len(rep['collections'])} indexes={rep['search_indexes']}")
    return 2 if rep.get("timeout") else 0


if __name__ == "__main__":
    sys.exit(main())
