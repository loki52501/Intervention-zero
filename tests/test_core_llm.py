import pytest

from iz.llm.cache import CachedLLM, JsonlCache
from iz.llm.client import FakeLLM, LLMError, estimate_cost


def _counter_policy():
    n = {"i": 0}

    def p(model, system, user, schema):
        n["i"] += 1
        return {"n": n["i"]}
    return p, n


def test_cache_replay_identical(tmp_path):
    f = tmp_path / "c.jsonl"
    p, n = _counter_policy()
    c = CachedLLM(FakeLLM(policy=p), backend=JsonlCache(f))
    a = c.complete_json("m", "s", "u", {"x": 1})
    b = c.complete_json("m", "s", "u", {"x": 1})
    assert a.data == b.data and not a.cached and b.cached and n["i"] == 1
    p2, n2 = _counter_policy()
    c2 = CachedLLM(FakeLLM(policy=p2), backend=JsonlCache(f))
    r = c2.complete_json("m", "s", "u", {"x": 1})
    assert r.data == a.data and r.cached and n2["i"] == 0 and c2.hits == 1


def test_fake_cycles_and_fails():
    f = FakeLLM(responses=[{"a": 1}, {"a": 2}])
    assert [f.complete_json("m", "", "", {}).data["a"] for _ in range(4)] == [1, 2, 2, 2]
    g = FakeLLM(responses=[{}])
    g.fail_times = 1
    with pytest.raises(LLMError):
        g.complete_json("m", "", "", {})
    assert g.complete_json("m", "", "", {}).data == {}


def test_estimate_cost():
    assert estimate_cost("claude-sonnet-4-5", 1000, 1000) > 0
    assert estimate_cost("claude-opus-4-1", 1000, 1000) > estimate_cost("claude-sonnet-4-5", 1000, 1000)
    assert estimate_cost("unknown-model", 1_000_000, 0) == 3.0
