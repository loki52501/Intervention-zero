# Raindrop in Intervention Zero

## What Raindrop does here

Every simulated mission is one Raindrop event: `user_id` is the robot id, event name `mission`, `convo_id` is `site:day:shift`, and properties are `site`, `policy_version`, `run_id`, `situation_type`, `model`, `day`, `shift`. Robot actions are recorded as tool spans on that event. Fix cycles and coach cycles are events named `fix_cycle` / `coach_cycle` (users `fixer` / `coach`, `convo_id` = run id). Signals attach to these events.

## Signals

All signals are sent with `signal_type="default"` (`iz/obs/rd.py`). Emitters below were grepped from `iz/` (`iz/loop/campaign.py`).

| Signal | Emitted where |
|---|---|
| `takeover` | per mission, on operator takeover |
| `takeover_avoidable` | per mission, when the takeover was avoidable |
| `guardrail_blocked` | per mission, when the guardrail blocks an action |
| `safety_violation` | per mission, on a safety violation |
| `edit_promoted` | per fix cycle (POSITIVE), on promotion |
| `edit_rejected` | per fix cycle, when the gate rejects |
| `slow_mission`, `drift_detected`, `rollback`, `coach_promoted` | defined, not yet emitted |
| `agent_report` | helper `rd.agent_report` exists; no emitter in `iz/` yet (only used by the check script) |

`agent_report` kinds: `missing_context`, `capability_gap`, `repeatedly_broken_tool`, `complete_task_failure`.

The SDK type only allows `default` / `feedback` / `edit`. Verified probe with `signal_type="agent"`: pydantic rejects it (`DefaultSignal.signal_type` is `Literal['default']`); `track_signal` logs "Invalid data passed to track_signal", does not raise, and drops the signal before buffering (nothing sent).

## Read-back (why it is load-bearing)

- Fixer: `campaign._raindrop_summary` calls `rd_query.top_takeover_causes` (Query SDK `events.list`) and puts "Raindrop top takeover causes" into the fixer prompt when Raindrop is enabled.
- Drift: `drift._raindrop_rate` calls `signal_timeseries("takeover")` and attaches `raindrop_rate` to the drift result. This is informational only; the drift alarm itself is a two-proportion z-test on local history.
- Known gap: `events.list` does not return signal properties, so `agent_reports()` reads `kind` from event properties and typically reports `unknown`.

## Degraded fallback

Any Query failure returns `{"degraded": True}`. The fixer then uses local takeover counts (Mongo / in-memory takeovers) and a "raindrop degraded; using local counts" ticker is emitted. The robot loop never raises on Raindrop errors. Raindrop is off unless `RAINDROP_ENABLED=1` and a write key are present.

## Budget

In-process cap `RAINDROP_EVENT_CAP` (default 900, counting events plus signals), kept below the free tier. Further sends are dropped with one warning. No Pro-only features are used.

## Verified live check (2026-10-11 14:46 PT)

Command: `.venv\Scripts\python.exe scripts\raindrop_check.py` (run once; output `out/raindrop_check.json`). Sent 1 event (+1 tool span) and 2 signals (`events_sent_counter` = 3).

| Field | Value |
|---|---|
| sent | true |
| event_visible | true |
| signals_visible | agent_report, takeover |
| id_match | true |
| tool_calls_n | 1 |
| latency_event_s | 7.2 |
| latency_s (both signals) | 7.2 |
| poll_attempts | 1 |
| events_get ok | true (signals: agent_report, takeover) |
| agent probe | rejected (pydantic Literal['default']); raised=False, logged_error=True; network_sent=false |
| readback top_takeover_causes | degraded=false, check_row_present=true |
| readback agent_reports | degraded=false, kinds=["unknown"] (the gap above) |

Readback-only rerun (no sends) at 2026-10-11 14:51 PT: `scripts\raindrop_check.py --readback-only --timeout 30` found the same event again (visible=true, both signals, id_match=true, 1 attempt). It is stored under `readback_rerun` in the JSON, and the live-run fields above were left unchanged.
