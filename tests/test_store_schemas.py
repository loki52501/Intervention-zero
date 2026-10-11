import copy
import uuid

import pytest
from pymongo.errors import WriteError

from iz.store.db import GENOMES, get_db, mongo_available
from iz.types import Genome

pytestmark = pytest.mark.skipif(not mongo_available(), reason="MongoDB not available")


@pytest.fixture(scope="module")
def db():
    from scripts.setup_atlas import setup
    d = get_db("iz_test")
    setup(d, wait_search=False)
    return d


def _doc():
    return {**Genome(version=900000 + uuid.uuid4().int % 90000).model_dump(mode="json")}


def _insert(db, doc):
    doc["_id"] = doc["version"]
    try:
        db[GENOMES].insert_one(doc)
    finally:
        db[GENOMES].delete_one({"_id": doc["_id"]})


def _code(db, doc):
    with pytest.raises(WriteError) as ei:
        _insert(db, doc)
    return ei.value.code


def test_valid_genome_passes(db):
    _insert(db, _doc())


def test_battery_floor_changed(db):
    d = _doc(); d["guardrails"]["locked"]["battery_floor_pct"] = 10
    assert _code(db, d) == 121


def test_never_enter_human_only_false(db):
    d = _doc(); d["guardrails"]["locked"]["never_enter_human_only"] = False
    assert _code(db, d) == 121


def test_missing_locked(db):
    d = _doc(); del d["guardrails"]["locked"]
    assert _code(db, d) == 121


def test_human_only_in_rule_params(db):
    d = _doc()
    d["rulebook"] = [{"id": "r1", "params": {"allow_zone_kinds": ["corridor", "human_only"]}}]
    assert _code(db, d) == 121


def test_human_only_in_tool_policy_deny(db):
    d = _doc()
    d["tool_policy"]["deny"] = [{"tool": "wait", "allow_zone_kinds": ["human_only"]}]
    assert _code(db, d) == 121


def test_allowed_zone_kinds_pass(db):
    d = _doc()
    d["rulebook"] = [{"id": "r1", "params": {"allow_zone_kinds": ["corridor"]}}]
    _insert(db, d)


def test_setup_idempotent(db):
    from scripts.setup_atlas import setup
    setup(db, wait_search=False)
    setup(db, wait_search=False)
