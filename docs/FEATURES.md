# MongoDB Atlas features: live verification

Generated 2026-10-11T21:47:59Z by `.venv\Scripts\python.exe scripts\verify_features.py` against db `intervention_zero` (writes only to `verify_scratch` and `iz_test`). Server 8.0.34. Result: 14/15 features verified.

| feature | job in the system | file | verified evidence |
|---|---|---|---|
| atlas_connection | single Atlas cluster backs every loop | iz/store/db.py | PASS server 8.0.34, db=intervention_zero, test_db=iz_test, setup_atlas: intervention_zero: already applied; iz_test: already applied |
| collections_and_validators | schema-validated store for genomes, playbooks, takeovers, edits, decisions | scripts/setup_atlas.py, iz/store/schemas.py | PASS 18 collections; validated(strict/error)=['genomes', 'fixer_playbooks', 'takeovers', 'edits', 'decisions']; NOT validated=[]; metrics=timeseries, ui_events capped=None |
| locked_guardrails_exact | safety core the fixer can never evolve, enforced by the database | iz/store/schemas.py | PASS live locked == LOCKED_SCHEMA: True; whole validator == VALIDATORS['genomes']: True; never_enter_human_only enum [True]; never_hold_fire_doors enum [True]; always_escalate_person_down enum [True]; battery_floor_pct enum [15]; max_con... |
| validator_rejects_locked_change | a fixer edit that weakens a locked guardrail is refused at write time | iz/store/genome_store.py, iz/store/schemas.py | PASS battery_floor_pct=5: refused code=121 errmsg='Document failed validation' errInfo={"failingDocumentId": 990978375, "details": {"operatorName": "$jsonSchema", "schemaRulesNotSatisfied": [{"operatorName": "properties", "propertiesNotS... |
| validator_rejects_human_only | no rule may grant access to human_only zones | iz/store/schemas.py | PASS allow_zone_kinds=[human_only]: refused code=121 errmsg='Document failed validation' errInfo={"failingDocumentId": 990777528, "details": {"operatorName": "$jsonSchema", "schemaRulesNotSatisfied": [{"operatorName": "properties", "prop... |
| vector_search_index | semantic recall of similar past takeovers and rules (Voyage 1024-d) | iz/store/hybrid.py, iz/store/memory.py, scripts/setup_atlas.py | PASS intervention_zero.takeovers_vec: vectorSearch READY queryable=True dims=1024 sim=cosine; intervention_zero.rules_vec: vectorSearch READY queryable=True dims=1024 sim=cosine; iz_test.takeovers_vec: vectorSearch READY queryable=True d... |
| atlas_search_index | keyword recall of takeovers and rules | iz/store/hybrid.py, iz/store/memory.py, scripts/setup_atlas.py | PASS intervention_zero.takeovers_text: search READY queryable=True analyzer=lucene.standard; intervention_zero.rules_text: search READY queryable=True analyzer=lucene.standard; iz_test.takeovers_text: search READY queryable=True analyzer... |
| hybrid_search | fuses vector + keyword recall into robot context, then Voyage rerank | iz/store/hybrid.py, iz/store/memory.py | PASS embedder=VoyageEmbedder; intervention_zero: path=$rankFusion (server-side), hits={'vector': -1, 'text': -1, 'fused': 0, 'reranked': 0}, docs=0, rerank=off, _rerank present=False, 0 docs in intervention_zero; iz_test: path=$rankFusio... |
| multi_doc_transaction | promotion: new genome + active pointer + decision commit atomically | iz/store/genome_store.py | PASS commit: 2/2 visible; abort: 0 visible |
| change_stream | robots hot-reload the promoted genome; UI ticker | iz/store/genome_store.py, iz/store/feed.py | PASS operationType=insert, ns.coll=verify_scratch, resume token present=True, latency 98 ms |
| time_series_metrics | takeover-free mission rate per mission | iz/store/metrics.py | PASS type=timeseries, timeField=ts, metaField=meta, granularity=minutes, docs=0 |
| set_window_fields | rolling takeover-free rate for drift detection | iz/store/metrics.py, iz/loop/drift.py | PASS metrics empty in intervention_zero; operator verified on $documents: r=[0.0, 0.5, 0.5, 0.5] |
| graph_lookup_lineage | genome ancestry for audit/rollback | iz/store/genome_store.py | PASS intervention_zero: lineage(0)=[0]; iz_test: lineage(4)=[4, 0] |
| llm_cache | replay cache so demos and gates are reproducible | iz/llm/cache.py | PASS 0 entries so far in intervention_zero (replay cache not yet populated); iz_test: 0 |
| capped_ui_events | bounded live event feed | iz/store/feed.py | FAIL intervention_zero: capped=False, size=None, max=None; iz_test: capped=True, size=16777216 (not capped: likely created implicitly by an insert before setup_atlas ran; drop the empty collection and rerun setup_atlas) |

Not verified here: capped_ui_events (see out/features.json for full evidence).
