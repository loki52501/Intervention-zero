import logging
import threading

import pytest

from iz.obs.budget import Budget, BudgetExceeded, get_budget


def test_warns_once_and_raises(caplog):
    b = Budget(10)
    with caplog.at_level(logging.WARNING, logger="iz.obs.budget"):
        b.add(7, "m")
        assert not caplog.records
        b.add(1.5, "m")
        b.add(0.5, "m")
    assert len(caplog.records) == 1
    assert b.fraction == pytest.approx(0.9)
    with pytest.raises(BudgetExceeded):
        b.add(5, "x")
    assert b.spent == pytest.approx(14.0)
    assert b.remaining == 0
    assert b.snapshot()["by_model"]["x"] == 5


def test_threadsafe():
    b = Budget(1e9)
    ts = [threading.Thread(target=lambda: [b.add(0.01) for _ in range(1000)]) for _ in range(8)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert b.spent == pytest.approx(80.0)


def test_singleton():
    assert get_budget() is get_budget()
