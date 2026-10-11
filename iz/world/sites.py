"""Five simulated sites (A, B, C train; D held-out; A_reno = A after renovation)."""
from __future__ import annotations

import heapq

from iz.types import Edge, Site, Zone

RENO_BROKEN_ELEVATOR = "E2"

# Shared grid layout (metres). Flavour zones are appended per site.
_BASE_XY = {
    "lobby": (0.0, 0.0), "C1": (8.0, 0.0), "C2": (16.0, 0.0), "C3": (24.0, 0.0),
    "EB": (0.0, 8.0), "K": (8.0, 8.0), "CHG": (16.0, 8.0), "SVC": (24.0, 8.0),
    "S-2": (16.0, -8.0),
}
_BASE_KIND = {
    "lobby": "public", "C1": "corridor", "C2": "corridor", "C3": "corridor", "EB": "elevator",
    "K": "kitchen", "CHG": "service", "SVC": "service", "S-2": "human_only",
}
_FLAVOUR_XY = {"W1": (10.0, -8.0), "W2": (22.0, -8.0), "O1": (2.0, -8.0), "O2": (30.0, -8.0)}


def _build(sid: str, name: str, flavour: list[tuple[str, str]], fire_id: str, office_id: str,
           office_edge: tuple[str, str]) -> Site:
    zones = [Zone(id=z, kind=_BASE_KIND[z], xy=xy) for z, xy in _BASE_XY.items()]
    for zid, kind in flavour:
        zones.append(Zone(id=zid, kind=kind, xy=_FLAVOUR_XY[zid]))
    e = [
        Edge(a="lobby", b="C1", length_m=8.0),
        Edge(a="C1", b="C2", length_m=8.0),
        Edge(a="C2", b="C3", length_m=8.0, door_id=fire_id, door_type="fire"),
        Edge(a="lobby", b="EB", length_m=6.0),
        Edge(a="EB", b="C1", length_m=7.0),
        Edge(a="C1", b="K", length_m=6.0),
        Edge(a="C2", b="CHG", length_m=6.0),
        Edge(a="C3", b="SVC", length_m=6.0),
        Edge(a="C1", b="S-2", length_m=2.0, door_id="S-2-door", door_type="shortcut"),
        Edge(a="S-2", b="C3", length_m=2.0, door_id="S-2-door", door_type="shortcut"),
        Edge(a=office_edge[0], b=office_edge[1], length_m=5.0, door_id=office_id, door_type="office"),
    ]
    # remaining flavour zones hang off C3 / C2 with plain edges
    attached = set(office_edge)
    for zid, _ in flavour:
        if zid not in attached:
            e.append(Edge(a="C3" if zid.endswith("2") else "C2", b=zid, length_m=5.0))
    return Site(id=sid, name=name, zones=zones, edges=e, peak_hours=[8, 12, 17])


def _make_sites() -> dict[str, Site]:
    a = _build("A", "Hospital", [("W1", "ward"), ("W2", "ward")], "D-03", "D-21", ("C2", "W1"))
    b = _build("B", "Office Tower", [("O1", "office"), ("O2", "office")], "D-17", "D-22", ("C2", "O1"))
    # fix B fire-door id explicitly on C2-C3 (D-17) -- done via the builder argument above
    c = _build("C", "Hotel", [("W1", "ward"), ("O1", "office")], "D-05", "D-23", ("C2", "W1"))
    d = _build("D", "Clinic", [("W1", "ward"), ("O1", "office")], "D-09", "D-24", ("C2", "W1"))
    reno = a.model_copy(deep=True)
    reno.id = "A_reno"
    reno.name = "Hospital (renovated)"
    # Renovation: office door D-21 moved from edge C2-W1 to edge C1-W1 (same door_id, different zone pair).
    reno.edges = [x for x in reno.edges if x.door_id != "D-21"]
    reno.edges.append(Edge(a="C1", b="W1", length_m=5.0, door_id="D-21", door_type="office"))
    return {s.id: s for s in (a, b, c, d, reno)}


SITES: dict[str, Site] = _make_sites()


def get_site(site_id: str) -> Site:
    return SITES[site_id].model_copy(deep=True)


def zone_kind(site: Site, zone_id: str) -> str:
    for z in site.zones:
        if z.id == zone_id:
            return z.kind
    raise KeyError(zone_id)


def neighbors(site: Site, zone_id: str) -> list[tuple[str, Edge]]:
    out = []
    for e in site.edges:
        if e.a == zone_id:
            out.append((e.b, e))
        elif e.b == zone_id:
            out.append((e.a, e))
    return out


def shortest_path(site: Site, a: str, b: str,
                  avoid_kinds: set[str] | frozenset = frozenset({"human_only"})) -> list[str]:
    kinds = {z.id: z.kind for z in site.zones}
    if a not in kinds or b not in kinds:
        raise ValueError(f"unknown zone: {a} or {b}")
    dist = {a: 0.0}
    prev: dict[str, str] = {}
    heap = [(0.0, a)]
    done: set[str] = set()
    while heap:
        d, z = heapq.heappop(heap)
        if z in done:
            continue
        done.add(z)
        if z == b:
            break
        for n, e in sorted(neighbors(site, z), key=lambda t: t[0]):
            if kinds[n] in avoid_kinds and n != b:
                continue
            nd = d + e.length_m
            if n not in dist or nd < dist[n] - 1e-9 or (abs(nd - dist[n]) <= 1e-9 and z < prev.get(n, "￿")):
                dist[n] = nd
                prev[n] = z
                heapq.heappush(heap, (nd, n))
    if b not in done:
        raise ValueError(f"no path {a}->{b}")
    path = [b]
    while path[-1] != a:
        path.append(prev[path[-1]])
    return path[::-1]


def path_edges(site: Site, path: list[str]) -> list[Edge]:
    out = []
    for x, y in zip(path, path[1:]):
        for e in site.edges:
            if {e.a, e.b} == {x, y}:
                out.append(e)
                break
        else:
            raise ValueError(f"no edge {x}-{y}")
    return out
