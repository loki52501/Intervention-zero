"""MongoDB connection + collection/index name constants."""
from __future__ import annotations

import functools
import os

from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.database import Database

load_dotenv()

GENOMES = "genomes"
ACTIVE = "active"
TAKEOVERS = "takeovers"
RULES = "rules"
FACTS = "facts"
EDITS = "edits"
DECISIONS = "decisions"
PLAYBOOKS = "fixer_playbooks"
RUNS = "runs"
LLM_CACHE = "llm_cache"
UI_EVENTS = "ui_events"
EMB_CACHE = "embeddings_cache"
METRICS = "metrics"
METRICS_DAILY = "metrics_daily"
STEPS = "steps"

TAKEOVERS_VEC = "takeovers_vec"
TAKEOVERS_TEXT = "takeovers_text"
RULES_VEC = "rules_vec"
RULES_TEXT = "rules_text"

EMBED_DIMS = 1024
GLOBAL_SITE = "global"
ANY_TYPE = "any"


@functools.lru_cache(maxsize=1)
def get_client() -> MongoClient:
    uri = os.environ.get("MONGODB_URI")
    if not uri:
        raise RuntimeError("MONGODB_URI missing")
    return MongoClient(uri, serverSelectionTimeoutMS=8000, retryWrites=True,
                       appname="intervention-zero")


def get_db(name: str | None = None) -> Database:
    return get_client()[name or os.getenv("MONGODB_DB", "intervention_zero")]


def mongo_available() -> bool:
    try:
        if not os.environ.get("MONGODB_URI"):
            return False
        get_client().admin.command("ping")
        return True
    except Exception:
        return False
