import typing

from iz.types import SituationType
from iz.world.generator import generate_missions, heldout_incidents, regression_bank, train_incidents
from iz.world.sites import get_site

ALL = set(typing.get_args(SituationType))


def dump(ms):
    return [m.model_dump() for m in ms]


def test_deterministic():
    a = generate_missions("r", "A", 1, 1, 5)
    assert dump(a) == dump(generate_missions("r", "A", 1, 1, 5))
    assert dump(a) != dump(generate_missions("r", "A", 1, 1, 6))
    assert len(a) == 10 and len(generate_missions("r", "B", 0, 0, 1, n=4)) == 4


def test_mission_validity():
    for sid in ("A", "B", "C", "D", "A_reno"):
        s = get_site(sid)
        edges = {(e.a, e.b) for e in s.edges}
        for m in generate_missions("r", sid, 2, 0, 3, n=30):
            assert "S-2" not in m.route and len(set(m.route)) >= 2
            assert 0 <= len(m.situations) <= 2
            assert len({x.id for x in m.situations}) == len(m.situations)
            for x in m.situations:
                assert x.edge in edges
            assert m.id.startswith(f"r:{sid}:d2s0:m")


def test_situation_counts_vary():
    ms = generate_missions("r", "A", 0, 0, 1, n=60)
    assert {len(m.situations) for m in ms} == {0, 1, 2}


def test_splits():
    tr, ho = train_incidents(0), heldout_incidents(1)
    assert {x.id for x in tr}.isdisjoint({x.id for x in ho})
    assert "D" in {x.site for x in ho} and "D" not in {x.site for x in tr}
    assert {x.type for x in tr} == ALL and {x.type for x in ho} == ALL
    assert [x.id for x in tr] == [x.id for x in train_incidents(0)]


def test_regression_bank():
    rb = regression_bank()
    for sid in ("A", "B", "C", "D", "A_reno"):
        assert any(x.site == sid and x.type == "person_down" for x in rb)
        assert any(x.site == sid and x.type == "restricted_shortcut" for x in rb)
    assert any(x.type == "cart_blocking" and x.hidden.physics["red_tag"] for x in rb)
    extra = train_incidents(0)[:3]
    rb2 = regression_bank(solved=extra + extra)
    assert len(rb2) == len({x.id for x in rb2})
    assert all(x.id in {y.id for y in rb2} for x in extra)
