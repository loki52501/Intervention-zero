"""Replay cache for LLM calls: Mongo collection llm_cache, else local JSON-lines."""
from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Any

from iz.types import LLM, LLMResponse


def cache_key(model: str, system: str, user: str, schema: dict) -> str:
    return hashlib.sha256(json.dumps([model, system, user, schema], sort_keys=True).encode()).hexdigest()


class JsonlCache:
    def __init__(self, path: str | Path = "out/llm_cache.jsonl") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._d: dict[str, dict] = {}
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    try:
                        rec = json.loads(line)
                        self._d[rec["key"]] = rec["response"]
                    except Exception:
                        continue

    def get(self, key: str) -> dict | None:
        with self._lock:
            return self._d.get(key)

    def put(self, key: str, response: dict) -> None:
        with self._lock:
            self._d[key] = response
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"key": key, "response": response}) + "\n")


class MongoCache:
    def __init__(self, uri: str, db: str) -> None:
        from pymongo import MongoClient
        self.client = MongoClient(uri, serverSelectionTimeoutMS=3000)
        self.col = self.client[db]["llm_cache"]

    def ping(self) -> None:
        self.client.admin.command("ping")

    def get(self, key: str) -> dict | None:
        doc = self.col.find_one({"_id": key})
        return doc["response"] if doc else None

    def put(self, key: str, response: dict) -> None:
        self.col.replace_one({"_id": key}, {"_id": key, "response": response}, upsert=True)


def _default_backend() -> Any:
    from dotenv import load_dotenv
    load_dotenv()
    uri = os.getenv("MONGODB_URI")
    if uri:
        try:
            m = MongoCache(uri, os.getenv("MONGODB_DB", "iz"))
            m.ping()
            return m
        except Exception:
            pass
    return JsonlCache()


class CachedLLM:
    def __init__(self, inner: LLM, backend: Any = None) -> None:
        self.inner = inner
        self.backend = backend if backend is not None else _default_backend()
        self.hits = 0
        self.misses = 0
        self._lock = threading.Lock()

    def complete_json(self, model: str, system: str, user: str, schema: dict[str, Any]) -> LLMResponse:
        key = cache_key(model, system, user, schema)
        stored = self.backend.get(key)
        if stored is not None:
            with self._lock:
                self.hits += 1
            d = dict(stored)
            d.pop("cached", None)
            d.pop("cost_usd", None)
            return LLMResponse(**d, cached=True, cost_usd=0.0)
        resp = self.inner.complete_json(model, system, user, schema)
        with self._lock:
            self.misses += 1
        self.backend.put(key, resp.model_dump())
        return resp
