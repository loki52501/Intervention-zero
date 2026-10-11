"""Memory stores: AtlasMemory (Mongo + Voyage hybrid retrieval) and LocalMemory (offline).

Document shapes (collections owned here):
- takeovers: {**Takeover.model_dump(python), _id: t.id, tried: [Action dicts], outcome: str,
              text: "<situation_type> <zone_class> <observable> TIP: <operator_tip>", embedding: [1024]}
- rules:     {_id: r.id, site: site|"global", situation_type: type|"any", text, rule: RuleEntry json,
              embedding: [1024]}  (upsert)
- facts:     {_id: sha1(site+text), site, text}  (upsert)
Only rules in the ACTIVE genome's rulebook are ever returned to a robot; the rules collection is
just an index used to rank them.
"""
from __future__ import annotations

import hashlib
import logging
from typing import Any

from iz.store.db import (ANY_TYPE, FACTS, GLOBAL_SITE, RULES, RULES_TEXT, RULES_VEC, TAKEOVERS,
                         TAKEOVERS_TEXT, TAKEOVERS_VEC, get_db)
from iz.store.hybrid import HybridResult, hybrid_search
from iz.store.voyage import Embedder, FakeEmbedder, VoyageEmbedder, _tokens
from iz.types import (ContextBundle, Genome, RobotView, RuleEntry, Scope, Takeover)

log = logging.getLogger(__name__)
MAX_RULES = 8
MAX_FACTS = 3
EPISODE_FIELDS = ("situation_type", "zone", "observable", "operator_tip", "tried", "outcome")


def scope_matches(scope: Scope, view: RobotView) -> bool:
    if scope.site is not None and scope.site != view.site:
        return False
    if scope.situation_type is not None and scope.situation_type != view.type:
        return False
    if scope.zone_class is not None and scope.zone_class != view.zone_class:
        return False
    if scope.door_id is not None and scope.door_id != view.observable_params.get("door_id"):
        return False
    return True


def query_text(view: RobotView) -> str:
    return f"{view.type} {view.zone_class} {view.zone} {view.observable}"


def takeover_text(t: Takeover) -> str:
    return f"{t.situation_type} {t.zone_class} {t.observable} TIP: {t.operator_tip}"


def _episode(d: dict) -> dict:
    return {k: d.get(k) for k in EPISODE_FIELDS}


def _matching_rules(view: RobotView, genome: Genome) -> list[RuleEntry]:
    return [r for r in genome.rulebook if r.status == "active" and scope_matches(r.scope, view)]


class AtlasMemory:
    def __init__(self, db: Any = None, embedder: Embedder | None = None) -> None:
        self.db = db if db is not None else get_db()
        self.embedder: Embedder = embedder if embedder is not None else VoyageEmbedder(self.db)

    # ---- writes
    def add_takeover(self, t: Takeover) -> None:
        doc = t.model_dump(mode="python")
        doc["tried"] = [a.model_dump() for a in t.tried]
        doc["outcome"] = t.outcome.value
        doc["_id"] = t.id
        doc["text"] = takeover_text(t)
        doc["embedding"] = self.embedder.embed([doc["text"]], "document")[0]
        self.db[TAKEOVERS].replace_one({"_id": t.id}, doc, upsert=True)

    def add_rule(self, r: RuleEntry, site: str | None) -> None:
        doc = {"_id": r.id, "site": site or GLOBAL_SITE,
               "situation_type": r.scope.situation_type or ANY_TYPE, "text": r.text,
               "rule": r.model_dump(mode="json"),
               "embedding": self.embedder.embed([r.text], "document")[0]}
        self.db[RULES].replace_one({"_id": r.id}, doc, upsert=True)

    def add_fact(self, site: str, text: str) -> None:
        _id = hashlib.sha1((site + text).encode()).hexdigest()
        self.db[FACTS].replace_one({"_id": _id}, {"_id": _id, "site": site, "text": text}, upsert=True)

    # ---- read
    def get_context(self, view: RobotView, genome: Genome) -> ContextBundle:
        sources = genome.context_policy.sources
        matching = _matching_rules(view, genome) if "procedural" in sources else []
        bundle = ContextBundle(backend="atlas", rules=matching[:MAX_RULES], hits={"rules": len(matching[:MAX_RULES])})
        try:
            qtext = query_text(view)
            qvec = self.embedder.embed([qtext], "query")[0]
            hits: dict[str, int] = {}
            if "procedural" in sources and len(matching) > MAX_RULES:
                by_id = {r.id: r for r in matching}
                res = hybrid_search(
                    self.db[RULES], qtext, qvec,
                    {"site": [view.site, GLOBAL_SITE], "situation_type": [view.type, ANY_TYPE]},
                    len(matching), vector_index=RULES_VEC, text_index=RULES_TEXT, text_paths=["text"],
                    embedder=self.embedder, rerank=genome.context_policy.rerank)
                ranked = [by_id[d["_id"]] for d in res.docs if d["_id"] in by_id]
                ranked += [r for r in matching if r.id not in {x.id for x in ranked}]
                bundle.rules = ranked[:MAX_RULES]
            if "episodic" in sources:
                cp = genome.context_policy
                kw: dict[str, Any] = dict(vector_index=TAKEOVERS_VEC, text_index=TAKEOVERS_TEXT,
                                          text_paths=["observable", "operator_tip", "zone"],
                                          embedder=self.embedder, rerank=cp.rerank)
                res = hybrid_search(self.db[TAKEOVERS], qtext, qvec,
                                    {"site": view.site, "situation_type": view.type}, cp.k, **kw)
                if not res.docs:
                    res = hybrid_search(self.db[TAKEOVERS], qtext, qvec, {"site": view.site}, cp.k, **kw)
                bundle.episodes = [_episode(d) for d in res.docs]
                hits.update({k: v for k, v in res.hits.items()})
            if "semantic" in sources:
                bundle.facts = [d["text"] for d in
                                self.db[FACTS].find({"site": view.site}).sort("_id", 1).limit(MAX_FACTS)]
            hits["rules"] = len(bundle.rules)
            bundle.hits = hits
        except Exception as e:  # noqa: BLE001 - the robot must keep running
            log.warning("AtlasMemory.get_context degraded: %s", str(e)[:200])
            bundle.episodes, bundle.facts = [], []
            bundle.hits = {"error": 1}
        return bundle


class LocalMemory:
    """In-process, deterministic, no network."""

    def __init__(self) -> None:
        self.takeovers: list[Takeover] = []
        self.rules: list[tuple[RuleEntry, str]] = []
        self.facts: list[tuple[str, str]] = []

    def add_takeover(self, t: Takeover) -> None:
        self.takeovers = [x for x in self.takeovers if x.id != t.id] + [t]

    def add_rule(self, r: RuleEntry, site: str | None) -> None:
        self.rules = [x for x in self.rules if x[0].id != r.id] + [(r, site or GLOBAL_SITE)]

    def add_fact(self, site: str, text: str) -> None:
        if (site, text) not in self.facts:
            self.facts.append((site, text))

    def get_context(self, view: RobotView, genome: Genome) -> ContextBundle:
        cp = genome.context_policy
        rules = _matching_rules(view, genome)[:MAX_RULES] if "procedural" in cp.sources else []
        episodes: list[dict] = []
        n = 0
        if "episodic" in cp.sources:
            q = set(_tokens(query_text(view)))
            scored = []
            for t in self.takeovers:
                if t.site != view.site:
                    continue
                score = len(q & set(_tokens(takeover_text(t))))
                scored.append((-score, t.id, t))
            scored.sort(key=lambda p: (p[0], p[1]))
            n = len(scored)
            for _, _, t in scored[:cp.k]:
                d = t.model_dump(mode="python")
                d["tried"] = [a.model_dump() for a in t.tried]
                d["outcome"] = t.outcome.value
                episodes.append(_episode(d))
        facts = ([x for s, x in self.facts if s == view.site][:MAX_FACTS]
                 if "semantic" in cp.sources else [])
        return ContextBundle(rules=rules, episodes=episodes, facts=facts, backend="local",
                             hits={"text": n, "fused": len(episodes), "rules": len(rules)})
