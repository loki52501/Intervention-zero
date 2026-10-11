"""UI ticker feed backed by the ui_events collection."""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Iterator

from bson import ObjectId
from pymongo.database import Database

from iz.store.db import UI_EVENTS, get_db

log = logging.getLogger(__name__)

HYBRID_HITS = "hybrid_search_hits"
TXN_COMMITTED = "transaction_committed"
STREAM_RELOAD = "change_stream_reloaded"
RAINDROP_SENT = "raindrop_signal_sent"
VALIDATOR_REFUSED = "validator_refused"
ACTIVE_CHANGED = "active_pointer_changed"


def emit(kind: str, payload: dict, db: Database | None = None) -> None:
    """Insert a ticker event. Never raises."""
    try:
        if db is None:
            if not os.environ.get("MONGODB_URI"):
                return
            db = get_db()
        db[UI_EVENTS].insert_one({"kind": kind, "payload": payload, "ts": datetime.now(timezone.utc)})
    except Exception as e:  # noqa: BLE001
        log.debug("feed.emit failed: %s", e)


def _jsonable(doc: dict) -> dict:
    out = dict(doc)
    out["_id"] = str(out["_id"])
    ts = out.get("ts")
    if isinstance(ts, datetime):
        out["ts"] = ts.isoformat()
    return out


def tail(after_id: str | None = None, db: Database | None = None,
         max_await_ms: int = 1000) -> Iterator[dict]:
    """Yield events after after_id (if given), then follow new inserts forever."""
    db = db if db is not None else get_db()
    coll = db[UI_EVENTS]
    # open the stream first so nothing inserted during the replay is missed
    with coll.watch([{"$match": {"operationType": "insert"}}],
                    max_await_time_ms=max_await_ms) as stream:
        last = ObjectId(after_id) if after_id else None
        if after_id:
            for d in coll.find({"_id": {"$gt": last}}).sort("_id", 1):
                last = d["_id"]
                yield _jsonable(d)
        while stream.alive:
            change = stream.try_next()
            if change is None:
                continue
            doc = change["fullDocument"]
            if last is not None and doc["_id"] <= last:
                continue
            yield _jsonable(doc)
