import uuid
from datetime import datetime, timedelta, timezone

import pytest

from iz.store import metrics
from iz.store.db import METRICS_DAILY, get_db, mongo_available

pytestmark = pytest.mark.skipif(not mongo_available(), reason="MongoDB not available")


@pytest.fixture()
def db():
    return get_db("iz_test")


def test_rolling_and_daily(db):
    run = f"t-{uuid.uuid4().hex[:8]}"
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rows = []
    # site A: 15 missions, free on i%3 != 0 ; site B: 15 missions, free after the 5th
    for i in range(15):
        rows.append(dict(run_id=run, site="A", day=i // 5, shift=i % 5, policy_version=0,
                         takeover_free=(i % 3 != 0), ts=t0 + timedelta(minutes=i)))
        rows.append(dict(run_id=run, site="B", day=i // 5, shift=i % 5, policy_version=1,
                         takeover_free=(i >= 5), ts=t0 + timedelta(minutes=i, seconds=30)))
    metrics.ensure_metrics(db)
    metrics.record_many(rows, db=db)
    metrics.record_mission(run, "A", 3, 0, 0, True, ts=t0 + timedelta(minutes=60), db=db)

    a = metrics.rolling_tfmr(run, "A", window=10, db=db)
    assert len(a) == 16
    last10 = [i % 3 != 0 for i in range(15)][-9:] + [True]
    assert a[-1]["tfmr"] == pytest.approx(sum(last10) / 10)
    b = metrics.rolling_tfmr(run, "B", window=10, db=db)
    assert b[-1]["tfmr"] == pytest.approx(1.0)
    assert len(metrics.rolling_tfmr(run, window=10, db=db)) == 31

    n = metrics.daily_rollup(run, db=db)
    assert n == 7  # A: days 0..3, B: days 0..2
    n2 = metrics.daily_rollup(run, db=db)
    assert n2 == n
    by_day = metrics.tfmr_by_day(run, db=db)
    assert [(d["site"], d["day"]) for d in by_day][:2] == [("A", 0), ("A", 1)]
    b0 = [d for d in by_day if d["site"] == "B" and d["day"] == 0][0]
    assert b0["missions"] == 5 and b0["rate"] == 0.0
    b2 = [d for d in by_day if d["site"] == "B" and d["day"] == 2][0]
    assert b2["rate"] == 1.0
    a0 = [d for d in by_day if d["site"] == "A" and d["day"] == 0][0]
    assert a0["takeover_free"] == 3  # i=0..4 -> free at 1,2,4

    db[METRICS_DAILY].delete_many({"run_id": run})
