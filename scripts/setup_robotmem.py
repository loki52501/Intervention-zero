"""Idempotent Atlas setup for robot env-memory: collections, validators, search indexes, taxonomy seed."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from iz.robotmem import memory as M  # noqa: E402
from iz.store.db import get_db  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=os.getenv("MONGODB_DB", "intervention_zero"))
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--no-wait", action="store_true")
    a = ap.parse_args()
    db = get_db(a.db)
    made = M.ensure_collections(db)
    seeded = M.seed_taxonomy(db, M._default_embedder(db))
    idx = M.ensure_search_indexes(db)
    print(f"db={db.name} created={made} taxonomy_seeded={seeded} indexes={idx}", flush=True)
    if a.no_wait:
        return 0
    ok = M.wait_queryable(db, a.timeout, log_fn=lambda st: print(
        "search:", ", ".join(f"{n}={'ready' if v else 'building'}" for n, v in st.items()), flush=True))
    if not ok:
        print("WARNING: not queryable in time; recall falls back to Python cosine")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
