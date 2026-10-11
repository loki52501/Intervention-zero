"""Genome / playbook versioning with transactional promotion."""
from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import Any, Callable, Literal

from pymongo.database import Database
from pymongo.errors import OperationFailure

from iz.store import db as D
from iz.store import feed
from iz.types import Decision, Edit, FixerPlaybook, Genome

Kind = Literal["robot_genome", "fixer_playbook"]
_COLL = {"robot_genome": D.GENOMES, "fixer_playbook": D.PLAYBOOKS}
_MODEL = {"robot_genome": Genome, "fixer_playbook": FixerPlaybook}
VALIDATION_FAILED = 121


class ValidatorRefusal(Exception):
    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


def _refusal(e: OperationFailure) -> ValidatorRefusal:
    det = e.details or {}
    return ValidatorRefusal(det.get("errmsg") or str(e), det.get("errInfo") or {})


class GenomeStore:
    def __init__(self, db: Database | None = None):
        self.db = db if db is not None else D.get_db()

    # ---- reads
    def _load(self, kind: Kind, v: int):
        doc = self.db[_COLL[kind]].find_one({"_id": v})
        if doc is None:
            raise KeyError(f"{kind} version {v} not found")
        doc.pop("_id", None)
        doc.pop("created_at", None)
        return _MODEL[kind].model_validate(doc)

    def get_active(self, kind: Kind = "robot_genome"):
        ptr = self.db[D.ACTIVE].find_one({"_id": kind})
        if ptr is None:
            raise KeyError(f"no active pointer for {kind}")
        return self._load(kind, ptr["version"])

    def get_version(self, v: int, kind: Kind = "robot_genome"):
        return self._load(kind, v)

    def next_version(self, kind: Kind = "robot_genome") -> int:
        top = self.db[_COLL[kind]].find_one(sort=[("_id", -1)], projection={"_id": 1})
        return (top["_id"] + 1) if top else 0

    def lineage(self, v: int, kind: Kind = "robot_genome") -> list[int]:
        coll = _COLL[kind]
        res = list(self.db[coll].aggregate([
            {"$match": {"_id": v}},
            {"$graphLookup": {"from": coll, "startWith": "$parent", "connectFromField": "parent",
                              "connectToField": "_id", "as": "anc", "depthField": "d"}},
            {"$project": {"anc._id": 1, "anc.d": 1}},
        ]))
        if not res:
            raise KeyError(f"{kind} version {v} not found")
        anc = sorted(res[0].get("anc", []), key=lambda a: a["d"])
        return [v] + [a["_id"] for a in anc]

    # ---- writes
    def promote(self, candidate: Genome | FixerPlaybook | dict, decision: Decision,
                kind: Kind = "robot_genome") -> int:
        raw: dict[str, Any] = (dict(candidate) if isinstance(candidate, dict)
                               else candidate.model_dump(mode="json"))
        version = raw["version"]
        now = datetime.now(timezone.utc)
        doc = {**raw, "_id": version, "created_at": now}
        dec = {**decision.model_dump(mode="python"), "_id": decision.id}
        client = D.get_client()

        def cb(session):
            self.db[_COLL[kind]].insert_one(doc, session=session)
            self.db[D.ACTIVE].update_one({"_id": kind}, {"$set": {"version": version, "updated_at": now}},
                                         upsert=True, session=session)
            self.db[D.DECISIONS].insert_one(dec, session=session)

        try:
            with client.start_session() as s:
                s.with_transaction(cb)
        except OperationFailure as e:
            if e.code == VALIDATION_FAILED:
                r = _refusal(e)
                feed.emit(feed.VALIDATOR_REFUSED, {"kind": kind, "version": version,
                                                   "message": r.message}, db=self.db)
                raise r from e
            raise
        feed.emit(feed.TXN_COMMITTED, {"kind": kind, "version": version, "parent": raw.get("parent"),
                                       "decision_id": decision.id}, db=self.db)
        return version

    def record_decision(self, decision: Decision) -> None:
        self.db[D.DECISIONS].insert_one({**decision.model_dump(mode="python"), "_id": decision.id})

    def record_edit(self, edit: Edit, candidate_id: str) -> None:
        self.db[D.EDITS].insert_one({**edit.model_dump(mode="python"), "_id": edit.id,
                                     "candidate_id": candidate_id, "created": datetime.now(timezone.utc)})

    def rollback(self, to_version: int, decision: Decision, kind: Kind = "robot_genome") -> None:
        if self.db[_COLL[kind]].count_documents({"_id": to_version}, limit=1) == 0:
            raise KeyError(f"{kind} version {to_version} not found")
        now = datetime.now(timezone.utc)
        dec = {**decision.model_dump(mode="python"), "_id": decision.id}

        def cb(session):
            self.db[D.ACTIVE].update_one({"_id": kind}, {"$set": {"version": to_version, "updated_at": now}},
                                         upsert=True, session=session)
            self.db[D.DECISIONS].insert_one(dec, session=session)

        with D.get_client().start_session() as s:
            s.with_transaction(cb)
        feed.emit(feed.TXN_COMMITTED, {"kind": kind, "version": to_version, "rollback": True,
                                       "decision_id": decision.id}, db=self.db)

    def watch_active(self, callback: Callable[[str, int], None], stop: threading.Event | None = None,
                     name: str = "default") -> None:
        runs = self.db[D.RUNS]
        key = f"watch:{name}"
        saved = runs.find_one({"_id": key})
        kwargs: dict[str, Any] = {"full_document": "updateLookup", "max_await_time_ms": 500}
        if saved and saved.get("token"):
            kwargs["resume_after"] = saved["token"]
        with self.db[D.ACTIVE].watch([{"$match": {"operationType": {"$in": ["insert", "update", "replace"]}}}],
                                     **kwargs) as stream:
            while stream.alive and not (stop and stop.is_set()):
                ch = stream.try_next()
                if ch is None:
                    continue
                full = ch.get("fullDocument") or {}
                kind, ver = full.get("_id"), full.get("version")
                if kind is not None and ver is not None:
                    callback(kind, ver)
                    feed.emit(feed.ACTIVE_CHANGED, {"kind": kind, "version": ver}, db=self.db)
                runs.update_one({"_id": key}, {"$set": {"token": ch["_id"],
                                                        "ts": datetime.now(timezone.utc)}}, upsert=True)
