"""Live smoke test. Run: .venv\\Scripts\\python.exe scripts\\smoke.py [--db iz_test] [--raindrop]"""
from __future__ import annotations

import argparse
import sys
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

FAILS: list[str] = []


def step(name, fn):
    try:
        detail = fn()
        print(f"OK {name} {detail if detail is not None else ''}".rstrip(), flush=True)
    except Exception as e:  # noqa: BLE001
        FAILS.append(name)
        print(f"FAIL {name} {type(e).__name__}: {str(e)[:200]}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="iz_test")
    ap.add_argument("--raindrop", action="store_true")
    args = ap.parse_args()

    from iz.store.db import ACTIVE, get_db
    from iz.types import Genome

    ctx: dict = {}

    def ping():
        ctx["db"] = get_db(args.db)
        ctx["db"].client.admin.command("ping")
        return f"db={args.db}"

    step("mongo_ping", ping)
    db = ctx.get("db")

    def voyage_embed():
        from iz.store.voyage import VoyageEmbedder
        emb = VoyageEmbedder(db)
        ctx["emb"] = emb
        d = emb.embed(["fire door held closed by magnet"], "document")
        q = emb.embed(["door with red tag"], "query")
        assert len(d[0]) == 1024 and len(q[0]) == 1024, (len(d[0]), len(q[0]))
        return f"dims={len(d[0])}"

    step("voyage_embed", voyage_embed)

    def voyage_rerank():
        r = ctx["emb"].rerank("fire door", ["a fire door with red tag", "a wet floor spill"], 2)
        return f"model={ctx['emb'].rerank_model_used} top={r[0][0]}"

    step("voyage_rerank", voyage_rerank)

    def hybrid():
        from iz.store.memory import AtlasMemory
        from iz.types import Action, OutcomeStatus, RobotView, Takeover
        mem = AtlasMemory(db, ctx["emb"])
        site = f"smoke-{uuid.uuid4().hex[:6]}"
        now = datetime.now(timezone.utc)
        for i, (st, obs, tip) in enumerate([
                ("fire_door", "Red FIRE DOOR D-17 closed, magnet hold", "never hold; wait for staff to open"),
                ("spill", "Wet floor spill in corridor", "clean_spill with low force")]):
            mem.add_takeover(Takeover(
                id=f"smoke-{site}-{i}", run_id="smoke", mission_id="m", robot_id="r1", site=site, day=0, shift=0,
                situation_id=f"{site}:{st}:{i}", situation_type=st, zone="z", zone_class="corridor",
                observable=obs, tried=[Action(tool="wait")], outcome=OutcomeStatus.TAKEOVER, reason="x",
                operator_tip=tip, policy_version=0, ts=now))
        ctx["smoke_site"] = site
        view = RobotView(situation_id="v", site=site, type="fire_door", zone="z", zone_class="corridor",
                         observable="Red fire door closed")
        b = mem.get_context(view, Genome(version=0))
        return f"backend={b.backend} hits={b.hits} episodes={len(b.episodes)}"

    if db is not None and "emb" in ctx:
        step("hybrid_search", hybrid)

    def refusal():
        from iz.store.genome_store import GenomeStore, ValidatorRefusal
        g = Genome(version=9_999_001).model_dump(mode="json")
        g["guardrails"]["locked"]["battery_floor_pct"] = 5
        dec = _decision()
        try:
            GenomeStore(db).promote(g, dec)
        except ValidatorRefusal as e:
            return f"refused: {e.message[:160]}"
        raise AssertionError("validator did NOT refuse a changed locked guardrail")

    def promote_txn():
        if args.db != "iz_test":
            return "SKIPPED (promote only runs on iz_test)"
        from iz.store.genome_store import GenomeStore
        store = GenomeStore(db)
        orig = db[ACTIVE].find_one({"_id": "robot_genome"})
        try:
            v = store.next_version()
            store.promote(Genome(version=v, parent=orig["version"] if orig else None), _decision(new=v))
            assert store.get_active().version == v
            return f"promoted v{v}"
        finally:
            if orig:
                db[ACTIVE].replace_one({"_id": "robot_genome"}, orig, upsert=True)

    if db is not None:
        if args.db == "iz_test":
            step("validator_refusal", refusal)
        else:
            print("OK validator_refusal SKIPPED (only on iz_test)")
        step("transaction_promote", promote_txn)

    if args.raindrop:
        raindrop_steps()

    print("RESULT", "FAIL " + ",".join(FAILS) if FAILS else "all OK")
    return 1 if FAILS else 0


def _decision(new=None):
    from iz.types import Decision, GateResult, Prediction, SetScore
    s = SetScore(n=1, takeover_free=1, safety_violations=0)
    gate = GateResult(candidate_id="smoke", parent_version=0, train_before=s, train_after=s, heldout_before=s,
                      heldout_after=s, regression_after=s, regressions=0, safety_violations=0, heldout_gain=0,
                      verdict="PROMOTE", reason="smoke")
    p = Prediction(train_fixed=0, heldout_fixed=0)
    return Decision(id=f"smoke-{uuid.uuid4().hex}", run_id="smoke", candidate_id="smoke", parent_version=0,
                    new_version=new, verdict="PROMOTE", reason="smoke", gate=gate, predicted=p, actual=p,
                    ts=datetime.now(timezone.utc))


def raindrop_steps() -> None:
    from iz.obs import rd, rd_query
    rd.force_enable()
    since = datetime.now(timezone.utc) - timedelta(minutes=10)
    ctx: dict = {}

    def send():
        if not rd.enabled():
            raise RuntimeError("RAINDROP_WRITE_KEY missing")
        h = rd.mission_begin("smoke-robot", "A", 0, 0, "smoke mission", {"run_id": "smoke", "policy_version": 0,
                                                                         "situation_type": "fire_door",
                                                                         "model": "smoke"})
        h.tool("wait", {"seconds": 1}, {"ok": True})
        h.finish("smoke done")
        ok = rd.signal(h.event_id, "takeover", {"situation_type": "fire_door", "site": "A", "run_id": "smoke"})
        rd.flush()
        assert h.event_id and ok, "event/signal not sent"
        return f"event sent, events_sent={rd.events_sent()}"

    step("raindrop_send", send)

    def readback():
        deadline = time.time() + 60
        r: dict = {}
        while time.time() < deadline:
            r = rd_query.top_takeover_causes(since)
            if not r.get("degraded") and r.get("causes"):
                return f"degraded=False causes={len(r['causes'])}"
            time.sleep(10)
        if r.get("degraded"):
            return f"WARN degraded=True {r.get('error')}"
        return "WARN degraded=False causes=0 (read-back lag)"

    step("raindrop_readback", readback)


if __name__ == "__main__":
    sys.exit(main())
