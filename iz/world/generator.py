"""Deterministic mission and incident generation."""
from __future__ import annotations

import random

from iz.types import TRAIN_SITES, Mission, Situation
from iz.world.sites import get_site, path_edges, shortest_path
from iz.world.situations import TEMPLATES, eligible_edges, make_situation

TYPES = list(TEMPLATES)
TYPE_WEIGHTS: dict[str, float] = {t: 1.0 for t in TYPES}
_ALL_SITES = ("A", "B", "C", "D", "A_reno")


def _weights(site_id: str) -> dict[str, float]:
    w = dict(TYPE_WEIGHTS)
    if site_id == "A_reno":
        w["elevator_out_of_service"] *= 3
    return w


def generate_missions(run_id: str, site: str, day: int, shift: int, seed: int, n: int = 10) -> list[Mission]:
    s = get_site(site)
    rng = random.Random(f"{run_id}:{site}:{day}:{shift}:{seed}")
    w = _weights(site)
    elig = {t: eligible_edges(s, t) for t in TYPES}
    elig = {t: v for t, v in elig.items() if v}
    zones = [z.id for z in s.zones if z.kind != "human_only"]
    missions = []
    for i in range(n):
        kind = rng.choice(["clean", "deliver", "patrol"])
        a, b = rng.sample(zones, 2)
        route = shortest_path(s, a, b)
        redges = path_edges(s, route)
        rkeys = {frozenset((e.a, e.b)) for e in redges}
        on_route = {t: [e for e in v if frozenset((e.a, e.b)) in rkeys] for t, v in elig.items()}
        on_route = {t: v for t, v in on_route.items() if v}
        r = rng.random()
        count = 0 if r < 0.2 else (1 if r < 0.7 else 2)
        sits: list[Situation] = []
        for _ in range(count):
            pool = on_route or elig
            types = sorted(pool)
            t = rng.choices(types, weights=[w[x] for x in types])[0]
            edge = rng.choice(pool[t])
            while True:
                sd = rng.randrange(1, 10**9)
                sit = make_situation(t, s, edge, sd)
                if sit.id not in {x.id for x in sits}:
                    break
            sits.append(sit)
        missions.append(Mission(id=f"{run_id}:{site}:d{day}s{shift}:m{i:02d}", run_id=run_id, site=site,
                                day=day, shift=shift, kind=kind, robot_id=f"R{i % 4 + 1}",
                                route=route, situations=sits))
    return missions


def _incidents(sites, tag: str, seed: int, per: int, offset: int) -> list[Situation]:
    rng = random.Random(f"{tag}:{seed}")
    out = []
    for sid in sites:
        s = get_site(sid)
        for t in TYPES:
            edges = eligible_edges(s, t)
            if not edges:
                continue
            for _ in range(per):
                sd = offset + rng.randrange(1, 400_000_000)
                out.append(make_situation(t, s, rng.choice(edges), sd))
    return out


def train_incidents(seed: int = 0, per_type_per_site: int = 2) -> list[Situation]:
    return _incidents(TRAIN_SITES, "train", seed, per_type_per_site, 0)


def heldout_incidents(seed: int = 1, per_type_per_site: int = 1) -> list[Situation]:
    out = _incidents(("A", "B", "C", "D"), "heldout", seed, per_type_per_site, 500_000_000)
    assert not {x.id for x in out} & {x.id for x in train_incidents(0)}
    return out


def regression_bank(seed: int = 2, solved: list[Situation] | None = None) -> list[Situation]:
    rng = random.Random(f"regression:{seed}")
    out: list[Situation] = []
    for sid in _ALL_SITES:
        s = get_site(sid)
        for t in ("person_down", "restricted_shortcut"):
            out.append(make_situation(t, s, rng.choice(eligible_edges(s, t)), 900_000_000 + rng.randrange(1, 10**8)))
    for sid in TRAIN_SITES:
        s = get_site(sid)
        edge = eligible_edges(s, "cart_blocking")[0]
        while True:
            sit = make_situation("cart_blocking", s, edge, 900_000_000 + rng.randrange(1, 10**8))
            if sit.hidden.physics["red_tag"]:
                break
        out.append(sit)
    seen, res = set(), []
    for x in out + list(solved or []):
        if x.id not in seen:
            seen.add(x.id)
            res.append(x)
    return res
