import time
import uuid

import pytest

from iz.store import feed
from iz.store.db import UI_EVENTS, get_db, mongo_available


class Boom:
    def __getitem__(self, k):
        raise RuntimeError("boom")


def test_emit_never_raises():
    feed.emit("x", {"a": 1}, db=Boom())


@pytest.mark.skipif(not mongo_available(), reason="MongoDB not available")
def test_emit_then_tail():
    from scripts.setup_atlas import setup
    db = get_db("iz_test")
    setup(db, wait_search=False)
    last = db[UI_EVENTS].find_one(sort=[("_id", -1)])
    if last is None:
        feed.emit("seed", {}, db=db)
        last = db[UI_EVENTS].find_one(sort=[("_id", -1)])
    marker = uuid.uuid4().hex
    feed.emit(feed.HYBRID_HITS, {"m": marker}, db=db)
    gen = feed.tail(after_id=str(last["_id"]), db=db)
    try:
        item = next(gen)
    finally:
        gen.close()
    assert item["payload"]["m"] == marker
    assert isinstance(item["_id"], str) and isinstance(item["ts"], str)
