"""UI event emitter: append-only JSONL (always) + Atlas ui_events feed (when reachable)."""
from __future__ import annotations

import itertools
import json
import logging
import os
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)

KINDS = frozenset({
    "site_loaded", "mission_start", "robot_move", "situation", "decision", "outcome",
    "takeover", "fix_proposed", "validator_refused", "gate_result", "promoted", "reload",
    "drift_alarm", "coach", "skill_harvested", "metrics", "ticker",
})
FEATURES = frozenset({
    "vector_search", "atlas_search", "rank_fusion", "rerank", "transaction", "change_stream",
    "validator", "time_series", "raindrop_event", "raindrop_signal", "raindrop_query",
    "bright_data", "firecrawl", "mujoco",
})

_ping_lock = threading.Lock()
_ping_cache: tuple[float, bool] = (0.0, False)


def atlas_reachable(ttl: float = 30.0) -> bool:
    """3s ping, cached for ttl seconds (negative results too, so offline runs stay fast)."""
    global _ping_cache
    with _ping_lock:
        now = time.monotonic()
        if _ping_cache[0] and now - _ping_cache[0] < ttl:
            return _ping_cache[1]
        ok = False
        uri = os.environ.get("MONGODB_URI")
        if uri:
            try:
                from pymongo import MongoClient
                c = MongoClient(uri, serverSelectionTimeoutMS=3000, connectTimeoutMS=3000)
                try:
                    c.admin.command("ping")
                    ok = True
                finally:
                    c.close()
            except Exception:  # noqa: BLE001
                ok = False
        _ping_cache = (now, ok)
        return ok


class Emitter:
    def __init__(self, run_id: str, path: str = "out/events.jsonl", atlas: bool | None = None):
        """atlas: None=auto-detect, False=never, True=try feed (still guarded)."""
        self.run_id = run_id
        self.path = Path(path)
        self.atlas = atlas
        self._lock = threading.Lock()
        self._seq = itertools.count(1)
        self._prefix = uuid.uuid4().hex[:8]
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, kind: str, payload: dict | None = None) -> dict:
        if kind not in KINDS:
            raise ValueError(f"unknown event kind: {kind!r}")
        payload = payload or {}
        with self._lock:
            ev = {"id": f"{self._prefix}-{next(self._seq):06d}",
                  "ts": datetime.now(timezone.utc).isoformat(),
                  "kind": kind, "run_id": self.run_id, "payload": payload}
            try:
                line = json.dumps(ev, default=str)
            except Exception:  # noqa: BLE001
                ev["payload"] = json.loads(json.dumps(payload, default=str))
                line = json.dumps(ev, default=str)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
                f.flush()
        self._to_atlas(ev)
        return ev

    def _to_atlas(self, ev: dict) -> None:
        try:
            if self.atlas is False:
                return
            if self.atlas is None and not atlas_reachable():
                return
            if self.atlas is True and not os.environ.get("MONGODB_URI"):
                return
            from iz.store import feed
            feed.emit(ev["kind"], {**ev["payload"], "run_id": ev["run_id"], "event_id": ev["id"]})
        except Exception as e:  # noqa: BLE001
            log.debug("atlas feed emit failed: %s", e)

    def ticker(self, feature: str, text: str) -> dict:
        if feature not in FEATURES:
            raise ValueError(f"unknown ticker feature: {feature!r}")
        return self.emit("ticker", {"feature": feature, "text": text})


_emitters: dict[str, Emitter] = {}
_reg_lock = threading.Lock()
_last: Emitter | None = None


def get_emitter(run_id: str | None = None, path: str = "out/events.jsonl",
                atlas: bool | None = None) -> Emitter:
    """Per-run singleton; with no run_id returns the most recently created emitter."""
    global _last
    with _reg_lock:
        if run_id is None:
            if _last is None:
                _last = _emitters.setdefault("default", Emitter("default", path, atlas))
            return _last
        em = _emitters.get(run_id)
        if em is None:
            em = _emitters[run_id] = Emitter(run_id, path, atlas)
        _last = em
        return em


def ticker(feature: str, text: str, run_id: str | None = None) -> dict:
    return get_emitter(run_id).ticker(feature, text)
