import pytest

from iz.world.sites import RENO_BROKEN_ELEVATOR, get_site, path_edges, shortest_path, zone_kind

IDS = ["A", "B", "C", "D", "A_reno"]
REQ = {"lobby": "public", "C1": "corridor", "C2": "corridor", "C3": "corridor", "EB": "elevator",
       "K": "kitchen", "SVC": "service", "CHG": "service", "S-2": "human_only"}


def _connected(s):
    seen, todo = set(), [s.zones[0].id]
    while todo:
        z = todo.pop()
        if z in seen:
            continue
        seen.add(z)
        todo += [e.b if e.a == z else e.a for e in s.edges if z in (e.a, e.b)]
    return seen == {z.id for z in s.zones}


@pytest.mark.parametrize("sid", IDS)
def test_site_basics(sid):
    s = get_site(sid)
    assert s.id == sid and 9 <= len(s.zones) <= 11
    for z, k in REQ.items():
        assert zone_kind(s, z) == k
    assert _connected(s)
    xys = [z.xy for z in s.zones]
    assert len(set(xys)) == len(xys)
    types = {e.door_type for e in s.edges}
    assert "fire" in types and "office" in types
    s2 = [e for e in s.edges if "S-2" in (e.a, e.b)]
    assert len(s2) == 2 and all(e.door_type == "shortcut" and e.door_id == "S-2-door" and e.length_m == 2.0 for e in s2)
    eb = {e.a if e.b == "EB" else e.b for e in s.edges if "EB" in (e.a, e.b)}
    assert "lobby" in eb and len(eb) >= 2
    for e in s.edges:
        if e.door_type == "office":
            assert zone_kind(s, e.a) in ("ward", "office") or zone_kind(s, e.b) in ("ward", "office")


def test_unknown():
    with pytest.raises(KeyError):
        get_site("Z")


def test_b_fire_door():
    assert any(e.door_id == "D-17" and e.door_type == "fire" for e in get_site("B").edges)


def test_a_c2_edges():
    assert any("C2" in (e.a, e.b) for e in get_site("A").edges)


@pytest.mark.parametrize("sid", IDS)
def test_paths(sid):
    s = get_site(sid)
    assert "S-2" not in shortest_path(s, "C1", "C3")
    assert "S-2" in shortest_path(s, "C1", "C3", avoid_kinds=set())
    p = shortest_path(s, "lobby", "K")
    assert len(path_edges(s, p)) == len(p) - 1


def test_reno_diff():
    a, r = get_site("A"), get_site("A_reno")
    ka = {(e.a, e.b, e.door_id) for e in a.edges}
    kr = {(e.a, e.b, e.door_id) for e in r.edges}
    assert len(ka - kr) == 1 and len(kr - ka) == 1
    (x,), (y,) = ka - kr, kr - ka
    assert x[2] == y[2] and {x[0], x[1]} != {y[0], y[1]}
    assert RENO_BROKEN_ELEVATOR == "E2"
    assert [z.id for z in a.zones] == [z.id for z in r.zones]
