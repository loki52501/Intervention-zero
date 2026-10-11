"""Fleet blackboard (D10 M-FLEET): shared hazards, blocked lanes and task claims.

Honest note: episodes use MemoryBlackboard by default (deterministic, sim-time
TTL). The Atlas backend is used for live fleet runs and the live test; there
claims are single-winner via an upsert on a deterministic _id.
"""
from __future__ import annotations

import logging
import math
import os
import threading
import uuid
from datetime import datetime, timedelta, timezone

from iz.store.schemas import FLEET_BLACKBOARD, VALIDATORS

log = logging.getLogger(__name__)

KINDS = ("hazard", "blocked_lane", "task_claim")
DEFAULT_TTL_S = 60.0
TTL_INDEX_S = 3600


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _aware(ts: datetime) -> datetime:
    return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)


def _check_kind(kind: str) -> None:
    if kind not in KINDS:
        raise ValueError(f"bad blackboard kind {kind!r}; expected one of {KINDS}")


def _filter(docs, xy, r, now_sim=None, kinds=None):
    """Distance + expiry filter shared by both backends; sorted by distance."""
    kinds = tuple(kinds) if kinds else ("hazard", "blocked_lane")
    now = _now()
    out = []
    for d in docs:
        if d.get("kind") not in kinds:
            continue
        ttl = float(d.get("ttl", DEFAULT_TTL_S))
        sim_t = d.get("sim_t")
        if now_sim is not None and sim_t is not None:
            if float(sim_t) + ttl < now_sim:
                continue
        elif _aware(d["ts"]) + timedelta(seconds=ttl) < now:
            continue
        dist = math.hypot(float(d["x"]) - xy[0], float(d["y"]) - xy[1])
        if dist <= r + float(d.get("radius", 0.0)):
            out.append((dist, d))
    out.sort(key=lambda t: t[0])
    return [dict(d) for _, d in out]


def _make_doc(world, kind, x, y, radius, robot_id, ttl, sim_t, parent_id, extra):
    _check_kind(kind)
    doc = dict(extra)
    doc.update(_id=uuid.uuid4().hex, world=world, kind=kind, x=float(x), y=float(y),
               radius=float(radius), robot_id=robot_id, ts=_now(), ttl=float(ttl))
    if sim_t is not None:
        doc["sim_t"] = float(sim_t)
    if parent_id is not None:
        doc["parent_id"] = parent_id
    return doc


class MemoryBlackboard:
    def __init__(self):
        self._lock = threading.Lock()
        self._docs: dict[str, dict] = {}

    def publish(self, world, kind, x, y, radius, robot_id, ttl=DEFAULT_TTL_S,
                sim_t=None, parent_id=None, **extra) -> dict:
        doc = _make_doc(world, kind, x, y, radius, robot_id, ttl, sim_t, parent_id, extra)
        with self._lock:
            self._docs[doc["_id"]] = doc
        return dict(doc)

    def query_near(self, world, xy, r, kinds=None, now_sim=None) -> list[dict]:
        with self._lock:
            docs = [d for d in self._docs.values() if d["world"] == world]
        return _filter(docs, xy, r, now_sim, kinds)

    def claim_task(self, world, task_id, robot_id, ttl=DEFAULT_TTL_S) -> bool:
        key = f"{world}:task:{task_id}"
        with self._lock:
            cur = self._docs.get(key)
            if cur is None:
                self._docs[key] = dict(_id=key, world=world, kind="task_claim",
                                       task_id=task_id, robot_id=robot_id, x=0.0, y=0.0,
                                       radius=0.0, ts=_now(), ttl=float(ttl))
                return True
            return cur["robot_id"] == robot_id

    def task_owner(self, world, task_id):
        with self._lock:
            cur = self._docs.get(f"{world}:task:{task_id}")
        return cur["robot_id"] if cur else None

    def clear(self, world=None) -> int:
        with self._lock:
            ids = [k for k, d in self._docs.items() if world is None or d["world"] == world]
            for k in ids:
                del self._docs[k]
        return len(ids)


class AtlasBlackboard:
    def __init__(self, db_name=None):
        from iz.store.db import get_client
        db = get_client()[db_name or os.getenv("MONGODB_DB", "intervention_zero")]
        self.db = db
        self.coll = db[FLEET_BLACKBOARD]
        self.ensure()

    def ensure(self) -> None:
        try:
            if FLEET_BLACKBOARD not in self.db.list_collection_names():
                self.db.create_collection(FLEET_BLACKBOARD,
                                          validator=VALIDATORS[FLEET_BLACKBOARD])
        except Exception as e:  # already exists / race
            if "already exists" not in str(e):
                log.warning("blackboard ensure: %s", type(e).__name__)
        self.coll.create_index([("ts", 1)], expireAfterSeconds=TTL_INDEX_S, name="ttl_ts")
        self.coll.create_index([("world", 1), ("kind", 1), ("x", 1), ("y", 1)],
                               name="world_kind_xy")
        self.coll.create_index([("parent_id", 1)], name="parent")

    def publish(self, world, kind, x, y, radius, robot_id, ttl=DEFAULT_TTL_S,
                sim_t=None, parent_id=None, **extra) -> dict:
        doc = _make_doc(world, kind, x, y, radius, robot_id, ttl, sim_t, parent_id, extra)
        self.coll.insert_one(dict(doc))
        return doc

    def query_near(self, world, xy, r, kinds=None, now_sim=None) -> list[dict]:
        ks = list(kinds) if kinds else ["hazard", "blocked_lane"]
        h = float(r) + 5.0
        docs = list(self.coll.find({
            "world": world, "kind": {"$in": ks},
            "x": {"$gte": xy[0] - h, "$lte": xy[0] + h},
            "y": {"$gte": xy[1] - h, "$lte": xy[1] + h},
        }))
        return _filter(docs, xy, r, now_sim, ks)

    def claim_task(self, world, task_id, robot_id, ttl=DEFAULT_TTL_S) -> bool:
        from pymongo import ReturnDocument
        from pymongo.errors import DuplicateKeyError
        key = f"{world}:task:{task_id}"
        try:
            doc = self.coll.find_one_and_update(
                {"_id": key},
                {"$setOnInsert": {"world": world, "kind": "task_claim", "task_id": task_id,
                                  "robot_id": robot_id, "x": 0.0, "y": 0.0, "radius": 0.0,
                                  "ts": _now(), "ttl": float(ttl)}},
                upsert=True, return_document=ReturnDocument.AFTER)
        except DuplicateKeyError:
            doc = self.coll.find_one({"_id": key})
        return bool(doc) and doc["robot_id"] == robot_id

    def task_owner(self, world, task_id):
        doc = self.coll.find_one({"_id": f"{world}:task:{task_id}"})
        return doc["robot_id"] if doc else None

    def clear(self, world=None) -> int:
        if world is None:
            raise ValueError("AtlasBlackboard.clear requires a world")
        return self.coll.delete_many({"world": world}).deleted_count


def get_blackboard(db_name=None, offline=None):
    if offline or not os.environ.get("MONGODB_URI"):
        return MemoryBlackboard()
    try:
        from iz.store.db import mongo_available
        if not mongo_available():
            return MemoryBlackboard()
        return AtlasBlackboard(db_name)
    except Exception as e:
        log.warning("blackboard falling back to memory: %s", type(e).__name__)
        return MemoryBlackboard()
