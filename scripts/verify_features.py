"""Live verification of the MongoDB Atlas features used by Intervention Zero.

Writes only to <db>.verify_scratch, iz_test.verify_scratch and (rejected) validator
inserts on iz_test.genomes. Outputs out/features.json and docs/FEATURES.md.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ["RAINDROP_ENABLED"] = "0"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")
os.environ["RAINDROP_ENABLED"] = "0"

import argparse  # noqa: E402
import json  # noqa: E402
import random  # noqa: E402
import re  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from uuid import uuid4  # noqa: E402


def _redact(s) -> str:
    s = str(s)
    uri = os.environ.get("MONGODB_URI")
    if uri:
        s = s.replace(uri, "<uri>")
    s = re.sub(r"mongodb(\+srv)?://\S+", "<uri>", s)
    s = re.sub(r"[\w.-]+\.mongodb\.net", "<cluster>.mongodb.net", s)
    for k in ("VOYAGE_API_KEY", "ANTHROPIC_API_KEY", "BRIGHTDATA_API_KEY", "FIRECRAWL_API_KEY",
              "RAINDROP_WRITE_KEY", "RAINDROP_API_KEY"):
        v = os.environ.get(k)
        if v and len(v) > 6:
            s = s.replace(v, "<key>")
    return s.encode("ascii", "replace").decode("ascii")


class _Abort(Exception):
    pass


JOBS = {
    "atlas_connection": "single Atlas cluster backs every loop",
    "collections_and_validators": "schema-validated store for genomes, playbooks, takeovers, edits, decisions",
    "locked_guardrails_exact": "safety core the fixer can never evolve, enforced by the database",
    "validator_rejects_locked_change": "a fixer edit that weakens a locked guardrail is refused at write time",
    "validator_rejects_human_only": "no rule may grant access to human_only zones",
    "vector_search_index": "semantic recall of similar past takeovers and rules (Voyage 1024-d)",
    "atlas_search_index": "keyword recall of takeovers and rules",
    "hybrid_search": "fuses vector + keyword recall into robot context, then Voyage rerank",
    "multi_doc_transaction": "promotion: new genome + active pointer + decision commit atomically",
    "change_stream": "robots hot-reload the promoted genome; UI ticker",
    "time_series_metrics": "takeover-free mission rate per mission",
    "set_window_fields": "rolling takeover-free rate for drift detection",
    "graph_lookup_lineage": "genome ancestry for audit/rollback",
    "llm_cache": "replay cache so demos and gates are reproducible",
    "capped_ui_events": "bounded live event feed",
}
FILES = {
    "atlas_connection": "iz/store/db.py",
    "collections_and_validators": "scripts/setup_atlas.py, iz/store/schemas.py",
    "locked_guardrails_exact": "iz/store/schemas.py",
    "validator_rejects_locked_change": "iz/store/genome_store.py, iz/store/schemas.py",
    "validator_rejects_human_only": "iz/store/schemas.py",
    "vector_search_index": "iz/store/hybrid.py, iz/store/memory.py, scripts/setup_atlas.py",
    "atlas_search_index": "iz/store/hybrid.py, iz/store/memory.py, scripts/setup_atlas.py",
    "hybrid_search": "iz/store/hybrid.py, iz/store/memory.py",
    "multi_doc_transaction": "iz/store/genome_store.py",
    "change_stream": "iz/store/genome_store.py, iz/store/feed.py",
    "time_series_metrics": "iz/store/metrics.py",
    "set_window_fields": "iz/store/metrics.py, iz/loop/drift.py",
    "graph_lookup_lineage": "iz/store/genome_store.py",
    "llm_cache": "iz/llm/cache.py",
    "capped_ui_events": "iz/store/feed.py",
}

RESULTS: dict[str, dict] = {}
SEARCH_NAMES = [("takeovers", "takeovers_vec", "vectorSearch"), ("rules", "rules_vec", "vectorSearch"),
                ("takeovers", "takeovers_text", "search"), ("rules", "rules_text", "search")]


def run_check(name, fn):
    try:
        ok, ev = fn()
    except Exception as e:  # noqa: BLE001
        ok, ev = False, f"{type(e).__name__}: {e}"
    ev = _redact(ev)
    RESULTS[name] = {"ok": bool(ok), "evidence": ev, "file_that_uses_it": FILES[name]}
    print(f"{'OK' if ok else 'FAIL'} {name} {ev[:160]}", flush=True)


def _idx_state(db, coll, name):
    for i in db[coll].list_search_indexes():
        if i.get("name") == name:
            return i
    return None


def _setup(args, db_names, VALIDATORS):
    if args.skip_setup:
        return "setup skipped (--skip-setup)"
    notes = []
    required = set(VALIDATORS) | {"metrics", "llm_cache", "ui_events", "active", "rules", "takeovers"}
    from iz.store.db import get_db
    for n in db_names:
        d = get_db(n)

        def need():
            if required - set(d.list_collection_names()):
                return True
            for c, i, _t in SEARCH_NAMES:
                st = _idx_state(d, c, i)
                if not st or not st.get("queryable"):
                    return True
            return False
        if not need():
            notes.append(f"{n}: already applied")
            continue
        r = subprocess.run([sys.executable, "scripts/setup_atlas.py", "--db", n, "--timeout", "300"],
                           cwd=str(ROOT), capture_output=True, text=True)
        notes.append(f"{n}: setup_atlas ran, exit {r.returncode}")
    return "setup_atlas: " + "; ".join(notes)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=os.environ.get("MONGODB_DB") or "intervention_zero")
    ap.add_argument("--test-db", default="iz_test")
    ap.add_argument("--out", default="out/features.json")
    ap.add_argument("--md", default="docs/FEATURES.md")
    ap.add_argument("--skip-setup", action="store_true")
    args = ap.parse_args()
    os.environ["MONGODB_DB"] = args.db
    tag = f"verify-{uuid4().hex[:8]}"
    state: dict = {"version": "unknown", "setup": ""}
    scratch = []

    try:
        from iz.store import db as D
        from iz.store import schemas as S
        from iz.store.db import get_client, get_db
        db = get_db(args.db)
        tdb = get_db(args.test_db)
        client = get_client()
        VALIDATORS = S.VALIDATORS
    except Exception as e:  # noqa: BLE001
        msg = _redact(f"{type(e).__name__}: {e}")
        for n in JOBS:
            RESULTS[n] = {"ok": False, "evidence": "import/connect failed: " + msg, "file_that_uses_it": FILES[n]}
            print(f"FAIL {n} {msg[:160]}")
        return _write(args, state, 1)

    try:
        try:
            state["setup"] = _setup(args, [args.db, args.test_db], VALIDATORS)
        except Exception as e:  # noqa: BLE001
            state["setup"] = _redact(f"setup error: {type(e).__name__}: {e}")

        def c_conn():
            client.admin.command("ping")
            v = client.server_info()["version"]
            state["version"] = v
            return True, f"server {v}, db={args.db}, test_db={args.test_db}, {state['setup']}"

        def c_colls():
            cols = {c["name"]: c for c in db.list_collections()}
            bad, good = [], []
            for n in VALIDATORS:
                o = cols.get(n, {}).get("options", {})
                if o.get("validator") and o.get("validationLevel") == "strict" and o.get("validationAction") == "error":
                    good.append(n)
                else:
                    bad.append(n)
            types = []
            if "metrics" in cols:
                types.append(f"metrics={cols['metrics'].get('type')}")
            if "ui_events" in cols:
                types.append(f"ui_events capped={cols['ui_events'].get('options', {}).get('capped')}")
            return (not bad), (f"{len(cols)} collections; validated(strict/error)={good}; "
                               f"NOT validated={bad}; {', '.join(types)}")

        def c_locked():
            cols = {c["name"]: c for c in db.list_collections()}
            live_v = cols["genomes"]["options"]["validator"]
            lk = live_v["$jsonSchema"]["properties"]["guardrails"]["properties"]["locked"]
            same_locked = lk == S.LOCKED_SCHEMA
            same_all = live_v == VALIDATORS["genomes"]
            props = lk.get("properties", {})
            parts = [f"{k} enum {v.get('enum')}" for k, v in props.items()]
            seed_ok = db.genomes.count_documents({"_id": 0, "$jsonSchema": S.GENOME_SCHEMA})
            ev = (f"live locked == LOCKED_SCHEMA: {same_locked}; whole validator == VALIDATORS['genomes']: {same_all}; "
                  f"{'; '.join(parts)}; additionalProperties {lk.get('additionalProperties')}; "
                  f"seed genome _id 0 passes schema: {seed_ok == 1}")
            return same_locked and same_all, ev

        def _reject(mutate, label):
            from pymongo.errors import OperationFailure
            from iz.types import Genome
            V = 990_000_000 + random.randint(0, 999_999)
            doc = Genome(version=V).model_dump(mode="json")
            doc["_id"] = V
            mutate(doc)
            try:
                try:
                    tdb.genomes.insert_one(doc)
                except OperationFailure as e:
                    det = e.details or {}
                    info = json.dumps(det.get("errInfo", {}), default=str)[:1200]
                    return e.code == 121, (f"{label}: refused code={e.code} errmsg={det.get('errmsg')!r} "
                                           f"errInfo={info}")
                return False, f"{label}: validator did NOT refuse"
            finally:
                tdb.genomes.delete_one({"_id": V})

        def c_rej_locked():
            def m(d):
                d["guardrails"]["locked"]["battery_floor_pct"] = 5
            return _reject(m, "battery_floor_pct=5")

        def c_rej_human():
            def m(d):
                d["rulebook"] = [{"id": "verify-r", "text": "verify",
                                  "params": {"allow_zone_kinds": ["human_only"]}}]
            return _reject(m, "allow_zone_kinds=[human_only]")

        def _idx(want_type, names):
            out, ok = [], True
            for d in (db, tdb):
                for coll, name, typ in SEARCH_NAMES:
                    if typ != want_type:
                        continue
                    st = _idx_state(d, coll, name)
                    if not st:
                        ok = False
                        out.append(f"{d.name}.{name}: MISSING")
                        continue
                    df = st.get("latestDefinition") or st.get("definition") or {}
                    good = st.get("type") == typ and st.get("status") == "READY" and bool(st.get("queryable"))
                    if typ == "vectorSearch":
                        vf = [f for f in df.get("fields", []) if f.get("type") == "vector"]
                        dims = vf[0].get("numDimensions") if vf else None
                        sim = vf[0].get("similarity") if vf else None
                        good = good and dims == 1024 and sim == "cosine"
                        extra = f"dims={dims} sim={sim}"
                    else:
                        an = df.get("analyzer")
                        extra = f"analyzer={an}"
                        good = good and bool(an)
                    ok = ok and good
                    out.append(f"{d.name}.{name}: {st.get('type')} {st.get('status')} "
                               f"queryable={st.get('queryable')} {extra}")
            return ok, "; ".join(out)

        def c_vec():
            return _idx("vectorSearch", None)

        def c_text():
            return _idx("search", None)

        def c_hybrid():
            import iz.store.hybrid as H
            from iz.store.voyage import FakeEmbedder, VoyageEmbedder
            captured: list = []
            H.emit = lambda kind, payload, db=None: captured.append(payload)
            q = "red fire door closed, magnet hold"
            emb_name = "VoyageEmbedder"
            try:
                emb = VoyageEmbedder(db=None)
                vec = emb.embed([q], "query")[0]
            except Exception as e:  # noqa: BLE001
                emb = FakeEmbedder()
                vec = emb.embed([q], "query")[0]
                emb_name = f"FakeEmbedder (Voyage failed: {type(e).__name__})"
            parts, ok = [], True
            for d, rr in ((db, False), (tdb, True)):
                H._PATH = None
                res = H.hybrid_search(d["takeovers"], q, vec, {}, 3, vector_index=D.TAKEOVERS_VEC,
                                      text_index=D.TAKEOVERS_TEXT,
                                      text_paths=["observable", "operator_tip", "zone"],
                                      embedder=emb, rerank=rr)
                label = {"rankFusion": "$rankFusion (server-side)",
                         "rrf_python": "python RRF fallback"}.get(res.path, res.path)
                ok = ok and res.path in ("rankFusion", "rrf_python")
                n = d["takeovers"].estimated_document_count()
                parts.append(f"{d.name}: path={label}, hits={res.hits}, docs={len(res.docs)}, "
                             f"rerank={'on' if rr else 'off'}, _rerank present="
                             f"{any('_rerank' in x for x in res.docs)}"
                             + (f", 0 docs in {d.name}" if n == 0 else f", {n} takeovers in {d.name}"))
            return ok, f"embedder={emb_name}; " + "; ".join(parts)

        def c_txn():
            c = tdb.verify_scratch
            c.insert_one({"run": tag, "k": "init"})
            c.delete_many({"run": tag, "k": "init"})
            with client.start_session() as s:
                s.with_transaction(lambda sess: [c.insert_one({"run": tag, "k": "txn-a"}, session=sess),
                                                 c.insert_one({"run": tag, "k": "txn-b"}, session=sess)])
            a = c.count_documents({"run": tag, "k": {"$in": ["txn-a", "txn-b"]}})

            def cb(sess):
                c.insert_one({"run": tag, "k": "txn-abort"}, session=sess)
                raise _Abort()
            try:
                with client.start_session() as s:
                    s.with_transaction(cb)
            except _Abort:
                pass
            b = c.count_documents({"run": tag, "k": "txn-abort"})
            return a == 2 and b == 0, f"commit: {a}/2 visible; abort: {b} visible"

        def c_cs():
            c = db.verify_scratch
            if "verify_scratch" not in db.list_collection_names():
                db.create_collection("verify_scratch")
            with c.watch([{"$match": {"operationType": "insert", "fullDocument.run": tag}}],
                         max_await_time_ms=500) as st:
                t0 = time.time()
                c.insert_one({"run": tag, "k": "cs"})
                ch = None
                while time.time() - t0 < 20:
                    ch = st.try_next()
                    if ch:
                        break
                lat = int((time.time() - t0) * 1000)
                if not ch:
                    return False, "no change event within 20 s"
                return True, (f"operationType={ch['operationType']}, ns.coll={ch['ns']['coll']}, "
                              f"resume token present={bool(ch.get('_id'))}, latency {lat} ms")

        def c_ts():
            cols = list(db.list_collections(filter={"name": "metrics"}))
            if not cols:
                return False, "metrics collection missing"
            o = cols[0]
            ts = o.get("options", {}).get("timeseries", {})
            ok = (o.get("type") == "timeseries" and ts.get("timeField") == "ts"
                  and ts.get("metaField") == "meta")
            return ok, (f"type={o.get('type')}, timeField={ts.get('timeField')}, metaField={ts.get('metaField')}, "
                        f"granularity={ts.get('granularity')}, docs={db.metrics.estimated_document_count()}")

        def c_win():
            last = db.metrics.find_one(sort=[("ts", -1)])
            if last:
                from iz.store.metrics import rolling_tfmr
                rid = last["meta"]["run_id"]
                rows = rolling_tfmr(rid, db=db, window=20)
                return True, (f"rolling_tfmr(run_id={rid}) on {db.name}: {len(rows)} rows, "
                              f"last tfmr={rows[-1].get('tfmr') if rows else None}")
            docs = [{"s": "a", "t": i, "v": i % 2} for i in range(4)]
            rows = list(db.aggregate([{"$documents": docs}, {"$setWindowFields": {
                "partitionBy": "$s", "sortBy": {"t": 1},
                "output": {"r": {"$avg": "$v", "window": {"documents": [-1, 0]}}}}}]))
            return True, f"metrics empty in {db.name}; operator verified on $documents: r={[r['r'] for r in rows]}"

        def c_lin():
            from iz.store.genome_store import GenomeStore
            parts, ok = [], True
            for d in (db, tdb):
                top = d.genomes.find_one(sort=[("_id", -1)])
                if not top:
                    ok = False
                    parts.append(f"{d.name}: no genomes")
                    continue
                v = top["_id"]
                lin = GenomeStore(d).lineage(v)
                ok = ok and bool(lin) and lin[0] == v
                parts.append(f"{d.name}: lineage({v})={lin}")
            if tdb.genomes.count_documents({"_id": 2}):
                l2 = GenomeStore(tdb).lineage(2)
                ok = ok and l2 == [2, 1, 0]
                parts.append(f"{tdb.name}: lineage(2)={l2} (expected [2, 1, 0])")
            return ok, "; ".join(parts)

        def c_cache():
            names = db.list_collection_names()
            if "llm_cache" not in names:
                return False, f"llm_cache missing in {db.name}"
            n = db.llm_cache.count_documents({})
            tn = tdb.llm_cache.count_documents({}) if "llm_cache" in tdb.list_collection_names() else None
            note = " (replay cache not yet populated)" if n == 0 else ""
            return True, f"{n} entries so far in {db.name}{note}; {args.test_db}: {tn}"

        def c_cap():
            cols = list(db.list_collections(filter={"name": "ui_events"}))
            if not cols:
                return False, "ui_events missing"
            o = cols[0].get("options", {})
            t = list(tdb.list_collections(filter={"name": "ui_events"}))
            to = t[0].get("options", {}) if t else {}
            ev = (f"{db.name}: capped={bool(o.get('capped'))}, size={o.get('size')}, max={o.get('max')}; "
                  f"{tdb.name}: capped={bool(to.get('capped'))}, size={to.get('size')}")
            if not o.get("capped"):
                ev += " (not capped: likely created implicitly by an insert before setup_atlas ran; drop the empty collection and rerun setup_atlas)"
            return bool(o.get("capped")), ev

        for name, fn in [("atlas_connection", c_conn), ("collections_and_validators", c_colls),
                         ("locked_guardrails_exact", c_locked),
                         ("validator_rejects_locked_change", c_rej_locked),
                         ("validator_rejects_human_only", c_rej_human),
                         ("vector_search_index", c_vec), ("atlas_search_index", c_text),
                         ("hybrid_search", c_hybrid), ("multi_doc_transaction", c_txn),
                         ("change_stream", c_cs), ("time_series_metrics", c_ts),
                         ("set_window_fields", c_win), ("graph_lookup_lineage", c_lin),
                         ("llm_cache", c_cache), ("capped_ui_events", c_cap)]:
            run_check(name, fn)
    finally:
        for d in (db, tdb):
            try:
                d.verify_scratch.delete_many({"run": tag})
            except Exception:  # noqa: BLE001
                pass
    return _write(args, state, 0)


def _write(args, state, _rc) -> int:
    ordered = {k: RESULTS[k] for k in JOBS if k in RESULTS}
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(ordered, indent=2), encoding="utf-8")
    n_ok = sum(1 for v in ordered.values() if v["ok"])
    total = len(JOBS)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    def cell(s, limit=None):
        s = str(s).replace("\r", " ").replace("\n", " ").replace("|", "\\|")
        if limit and len(s) > limit:
            s = s[:limit - 3] + "..."
        return s
    lines = ["# MongoDB Atlas features: live verification", "",
             f"Generated {ts} by `.venv\\Scripts\\python.exe scripts\\verify_features.py` against db "
             f"`{args.db}` (writes only to `verify_scratch` and `{args.test_db}`). Server {state['version']}. "
             f"Result: {n_ok}/{total} features verified.", "",
             "| feature | job in the system | file | verified evidence |", "|---|---|---|---|"]
    for k in JOBS:
        r = ordered.get(k)
        if r is None:
            ev = "FAIL not run"
        else:
            ev = ("PASS " if r["ok"] else "FAIL ") + r["evidence"]
        lines.append(f"| {k} | {cell(JOBS[k])} | {cell(FILES[k])} | {cell(ev, 240)} |")
    bad = [k for k in JOBS if not ordered.get(k, {}).get("ok")]
    if bad:
        lines += ["", "Not verified here: " + ", ".join(bad) + " (see out/features.json for full evidence)."]
    md = ROOT / args.md
    md.parent.mkdir(parents=True, exist_ok=True)
    md.write_text(_redact("\n".join(lines) + "\n"), encoding="utf-8")
    print(f"{n_ok}/{total} features verified")
    return 0 if n_ok == total else 1


if __name__ == "__main__":
    sys.exit(main())
