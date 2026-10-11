"""Time-series metrics (takeover-free mission rate) + rolling/daily aggregation."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pymongo.errors import CollectionInvalid, OperationFailure

from iz.store.db import METRICS, METRICS_DAILY, get_db


def ensure_metrics(db: Any) -> None:
    if METRICS in db.list_collection_names():
        return
    try:
        db.create_collection(METRICS, timeseries={"timeField": "ts", "metaField": "meta",
                                                  "granularity": "minutes"})
    except (CollectionInvalid, OperationFailure) as e:
        if isinstance(e, OperationFailure) and e.code != 48:  # 48 = NamespaceExists
            raise


def _row(run_id: str, site: str, day: int, shift: int, policy_version: int, takeover_free: bool,
         takeovers: int = 0, safety_violations: int = 0, time_s: float = 0.0, robot_id: str = "",
         ts: datetime | None = None) -> dict:
    return {"ts": ts or datetime.now(timezone.utc),
            "meta": {"site": site, "policy_version": policy_version, "run_id": run_id},
            "day": day, "shift": shift, "takeover_free": int(bool(takeover_free)),
            "takeovers": takeovers, "safety_violations": safety_violations,
            "time_s": float(time_s), "robot_id": robot_id}


def record_mission(run_id: str, site: str, day: int, shift: int, policy_version: int,
                   takeover_free: bool, *, takeovers: int = 0, safety_violations: int = 0,
                   time_s: float = 0.0, robot_id: str = "", ts: datetime | None = None, db: Any = None) -> None:
    db = db if db is not None else get_db()
    ensure_metrics(db)
    db[METRICS].insert_one(_row(run_id, site, day, shift, policy_version, takeover_free,
                                takeovers, safety_violations, time_s, robot_id, ts))


def record_many(rows: list[dict], db: Any = None) -> None:
    """rows: dicts of record_mission kwargs (run_id, site, day, shift, policy_version, takeover_free, ...)."""
    if not rows:
        return
    db = db if db is not None else get_db()
    ensure_metrics(db)
    docs = []
    for r in rows:
        if "meta" in r:  # already shaped
            docs.append({"ts": datetime.now(timezone.utc), **r})
        else:
            docs.append(_row(**r))
    db[METRICS].insert_many(docs)


def rolling_tfmr(run_id: str, site: str | None = None, window: int = 20, db: Any = None) -> list[dict]:
    db = db if db is not None else get_db()
    match: dict = {"meta.run_id": run_id}
    if site:
        match["meta.site"] = site
    pipe = [
        {"$match": match},
        {"$setWindowFields": {
            "partitionBy": "$meta.site",
            "sortBy": {"ts": 1},
            "output": {"tfmr": {"$avg": "$takeover_free",
                                "window": {"documents": [-(max(window, 1) - 1), 0]}}},
        }},
        {"$sort": {"ts": 1}},
        {"$project": {"_id": 0, "ts": 1, "site": "$meta.site", "day": 1, "shift": 1,
                      "policy_version": "$meta.policy_version", "tfmr": 1}},
    ]
    return list(db[METRICS].aggregate(pipe))


def daily_rollup(run_id: str, db: Any = None) -> int:
    db = db if db is not None else get_db()
    pipe = [
        {"$match": {"meta.run_id": run_id}},
        {"$group": {"_id": {"run_id": "$meta.run_id", "site": "$meta.site", "day": "$day"},
                    "missions": {"$sum": 1}, "takeover_free": {"$sum": "$takeover_free"},
                    "safety_violations": {"$sum": "$safety_violations"},
                    "policy_version": {"$max": "$meta.policy_version"}}},
        {"$project": {
            "_id": {"$concat": ["$_id.run_id", ":", "$_id.site", ":", {"$toString": "$_id.day"}]},
            "run_id": "$_id.run_id", "site": "$_id.site", "day": "$_id.day",
            "missions": 1, "takeover_free": 1, "safety_violations": 1, "policy_version": 1,
            "rate": {"$cond": [{"$gt": ["$missions", 0]}, {"$divide": ["$takeover_free", "$missions"]}, 0]}}},
        {"$merge": {"into": METRICS_DAILY, "on": "_id", "whenMatched": "replace", "whenNotMatched": "insert"}},
    ]
    list(db[METRICS].aggregate(pipe))
    return db[METRICS_DAILY].count_documents({"run_id": run_id})


def tfmr_by_day(run_id: str, db: Any = None) -> list[dict]:
    db = db if db is not None else get_db()
    return list(db[METRICS_DAILY].find({"run_id": run_id}, {"_id": 0}).sort([("site", 1), ("day", 1)]))
