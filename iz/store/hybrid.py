"""Hybrid retrieval: $rankFusion if the server supports it, else Python RRF; then Voyage rerank."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Literal

from pymongo.errors import OperationFailure

from iz.store.feed import HYBRID_HITS, emit
from iz.store.voyage import Embedder

log = logging.getLogger(__name__)

_PATH: str | None = None   # per-process cached: "rankFusion" | "rrf_python"
_LAST: str | None = None


def last_path() -> str | None:
    return _LAST


def rrf_fuse(lists: list[list[dict]], k: int = 60, weights: list[float] | None = None,
             key: str = "_id") -> list[dict]:
    weights = weights or [1.0] * len(lists)
    score: dict[str, float] = {}
    first: dict[str, dict] = {}
    for w, lst in zip(weights, lists):
        for rank, d in enumerate(lst, start=1):
            kk = str(d[key])
            score[kk] = score.get(kk, 0.0) + w / (k + rank)
            if kk not in first:
                first[kk] = dict(d)
    out = []
    for kk in sorted(score, key=lambda x: (-score[x], x)):
        d = first[kk]
        d["_rrf"] = score[kk]
        out.append(d)
    return out


@dataclass
class HybridResult:
    docs: list[dict]
    path: Literal["rankFusion", "rrf_python", "vector_only", "text_only"]
    hits: dict[str, int] = field(default_factory=dict)


def _mql(filter: dict) -> dict:
    parts = []
    for f, v in filter.items():
        parts.append({f: {"$in": list(v)}} if isinstance(v, (list, tuple, set)) else {f: {"$eq": v}})
    if not parts:
        return {}
    return parts[0] if len(parts) == 1 else {"$and": parts}


def hybrid_search(coll: Any, query_text: str, query_vec: list[float], filter: dict,
                  k: int, *, vector_index: str, text_index: str, text_paths: list[str],
                  embedder: Embedder | None = None, rerank: bool = True,
                  rerank_text_field: str = "text", num_candidates: int = 100, limit: int = 20,
                  weights: tuple[float, float] = (0.6, 0.4)) -> HybridResult:
    global _PATH, _LAST
    vec_stage: dict[str, Any] = {"index": vector_index, "path": "embedding", "queryVector": query_vec,
                                 "numCandidates": num_candidates, "limit": limit}
    mql = _mql(filter)
    if mql:
        vec_stage["filter"] = mql
    vecpipe: list[dict] = [{"$vectorSearch": vec_stage}]
    flt = [{"in": {"path": f, "value": list(v) if isinstance(v, (list, tuple, set)) else v}}
           for f, v in filter.items()]
    txtpipe: list[dict] = [{"$search": {"index": text_index, "compound": {
        "must": [{"text": {"query": query_text, "path": text_paths}}], "filter": flt}}},
        {"$limit": limit}]
    proj = {"$project": {"embedding": 0}}

    docs: list[dict] | None = None
    path: str = ""
    hits = {"vector": -1, "text": -1, "fused": 0, "reranked": 0}

    if _PATH != "rrf_python":
        try:
            pipe = [{"$rankFusion": {"input": {"pipelines": {"vec": vecpipe, "txt": txtpipe}},
                                     "combination": {"weights": {"vec": weights[0], "txt": weights[1]}}}},
                    {"$limit": limit}, proj]
            docs = list(coll.aggregate(pipe))
            _PATH = path = "rankFusion"
        except OperationFailure as e:
            log.info("$rankFusion unavailable (%s); using python RRF", str(e)[:120])
            _PATH = "rrf_python"
            docs = None

    if docs is None:
        vdocs = tdocs = None
        try:
            vdocs = list(coll.aggregate(vecpipe + [proj]))
        except Exception as e:  # noqa: BLE001
            log.warning("vector leg failed: %s", str(e)[:120])
        try:
            tdocs = list(coll.aggregate(txtpipe + [proj]))
        except Exception as e:  # noqa: BLE001
            log.warning("text leg failed: %s", str(e)[:120])
        if vdocs is None and tdocs is None:
            raise RuntimeError("both hybrid legs failed")
        if vdocs is not None and tdocs is not None:
            path, docs = "rrf_python", rrf_fuse([vdocs, tdocs], weights=list(weights))
        elif vdocs is not None:
            path, docs = "vector_only", vdocs
        else:
            path, docs = "text_only", tdocs or []
        hits["vector"] = len(vdocs) if vdocs is not None else 0
        hits["text"] = len(tdocs) if tdocs is not None else 0

    hits["fused"] = len(docs)
    top = docs[:20]
    if rerank and embedder is not None and top:
        order = embedder.rerank(query_text, [str(d.get(rerank_text_field, "")) for d in top], k)
        final = []
        for idx, sc in order:
            d = top[idx]
            d["_rerank"] = sc
            final.append(d)
        docs = final
    else:
        docs = top[:k]
    hits["reranked"] = len(docs)
    _LAST = path
    emit(HYBRID_HITS, {"coll": coll.name, "path": path, **hits})
    return HybridResult(docs=docs, path=path, hits=hits)  # type: ignore[arg-type]
