"""Embedding + rerank clients (Voyage via Atlas AI Model APIs) and a deterministic offline fake.

voyageai 0.5.0 auto-routes keys starting with 'al-' (Atlas AI Model API keys) to
https://ai.mongodb.com/v1, so no base_url is needed.
"""
from __future__ import annotations

import hashlib
import logging
import math
import os
import re
import time
from typing import Any, Literal, Protocol

from dotenv import load_dotenv

from iz.store.db import EMB_CACHE, EMBED_DIMS

load_dotenv()
log = logging.getLogger(__name__)

InputType = Literal["document", "query"]
_TOKEN = re.compile(r"[a-z0-9_\-]+")


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class Embedder(Protocol):
    def embed(self, texts: list[str], input_type: InputType) -> list[list[float]]: ...
    def rerank(self, query: str, docs: list[str], top_k: int) -> list[tuple[int, float]]: ...


class FakeEmbedder:
    """Deterministic, no network. Hashed bag-of-words, 1024 dims, L2-normalised."""

    def __init__(self, dims: int = EMBED_DIMS) -> None:
        self.dims = dims

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.dims
        for tok in _tokens(text):
            h = int.from_bytes(hashlib.sha256(tok.encode()).digest()[:8], "big")
            v[h % self.dims] += 1.0
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]

    def embed(self, texts: list[str], input_type: InputType) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def rerank(self, query: str, docs: list[str], top_k: int) -> list[tuple[int, float]]:
        q = set(_tokens(query))
        scored = []
        for i, d in enumerate(docs):
            dt = set(_tokens(d))
            scored.append((i, len(q & dt) / (len(q | dt) or 1)))
        scored.sort(key=lambda p: (-p[1], p[0]))
        return scored[:top_k]


class VoyageEmbedder:
    def __init__(self, db: Any = None, model: str | None = None, rerank_model: str | None = None) -> None:
        import voyageai
        self.db = db
        self.model = model or os.getenv("VOYAGE_EMBED_MODEL") or "voyage-4-large"
        self.rerank_model = rerank_model or os.getenv("VOYAGE_RERANK_MODEL") or "rerank-2.5"
        self.rerank_model_used: str | None = None
        self.client = voyageai.Client(api_key=os.environ.get("VOYAGE_API_KEY"))

    def _retry(self, fn, attempts: int = 3):
        for i in range(attempts):
            try:
                return fn()
            except Exception as e:  # noqa: BLE001
                rate = "ratelimit" in type(e).__name__.lower() or "429" in str(e)
                if not rate or i == attempts - 1:
                    raise
                time.sleep(1.5 * (2 ** i))

    def _key(self, text: str, input_type: str) -> str:
        return hashlib.sha256(f"{self.model}|{input_type}|{text}".encode()).hexdigest()

    def embed(self, texts: list[str], input_type: InputType) -> list[list[float]]:
        out: list[list[float] | None] = [None] * len(texts)
        keys = [self._key(t, input_type) for t in texts]
        if self.db is not None and texts:
            try:
                cached = {d["_id"]: d["v"] for d in
                          self.db[EMB_CACHE].find({"_id": {"$in": list(set(keys))}})}
                for i, k in enumerate(keys):
                    if k in cached:
                        out[i] = cached[k]
            except Exception as e:  # noqa: BLE001
                log.warning("embedding cache read failed: %s", e)
        todo = [i for i, v in enumerate(out) if v is None]
        for s in range(0, len(todo), 128):
            idx = todo[s:s + 128]
            batch = [texts[i] for i in idx]
            res = self._retry(lambda: self.client.embed(
                batch, model=self.model, input_type=input_type, output_dimension=EMBED_DIMS))
            for i, emb in zip(idx, res.embeddings):
                out[i] = list(emb)
            if self.db is not None:
                try:
                    for i in idx:
                        self.db[EMB_CACHE].replace_one({"_id": keys[i]}, {"_id": keys[i], "v": out[i]},
                                                       upsert=True)
                except Exception as e:  # noqa: BLE001
                    log.warning("embedding cache write failed: %s", e)
        return out  # type: ignore[return-value]

    def rerank(self, query: str, docs: list[str], top_k: int) -> list[tuple[int, float]]:
        if not docs:
            return []
        candidates = [self.rerank_model_used or self.rerank_model]
        for m in ("rerank-2.5", "rerank-2"):
            if m not in candidates:
                candidates.append(m)
        last: Exception | None = None
        for m in candidates:
            try:
                res = self._retry(lambda: self.client.rerank(query, docs, model=m, top_k=top_k))
                self.rerank_model_used = m
                return [(r.index, float(r.relevance_score)) for r in res.results]
            except Exception as e:  # noqa: BLE001
                last = e
                log.warning("rerank model %s failed: %s", m, type(e).__name__)
        assert last is not None
        raise last
