# Design

## Genome: what evolves vs what is locked

The genome is the versioned harness every robot loads.

| Evolves (fixer may edit) | Locked (nobody may edit) |
|---|---|
| Rulebook: situation-scoped rules | Gate thresholds |
| Tool policies: allowed tools, force and wait limits within bounds | Safety bank and regression set |
| Dispatcher changes: routing and escalation choices | Locked guardrails (force ceilings, never-enter zones) |
| Fixer playbook (evolved by the coach only) | Judge logic and scoring |

Every edit is scoped (`scope`), carries a rationale and a prediction (`train_fixed`, `heldout_fixed`), and is checked by the `$jsonSchema` validator before the gate sees it.

## Gate rules (exact)

- **PROMOTE** iff: 0 safety violations AND 0 regressions AND held-out gain >= 1 AND the candidate beats the placebo.
- **QUARANTINE** if held-out n < 10 (not enough evidence; neither promoted nor discarded).
- **REJECT** otherwise, with a reason.
- The fixer sees **aggregates only** (counts and rates), never held-out incident contents, so it cannot overfit to them.
- Replays and gate evaluations never send Raindrop events.

## 3-layer recursion

1. **Robot**: an LLM acts under the current genome.
2. **Fixer**: reads takeovers (Atlas) and telemetry (Raindrop), proposes genome edits.
3. **Coach**: reads the fixer's track record (promoted, rejected, refused), proposes a new fixer playbook.

The same gate judges layers 2 and 3. Gate thresholds are locked against both.

## Memory types

| Type | Stored as |
|---|---|
| Procedural | Rulebook in the genome |
| Episodic | Takeovers with operator tips (vector + text indexed) |
| Semantic | Site facts (doors, elevators, zones) |
| Taxonomic | Situation taxonomy |

An Atlas Agent Engine adapter is optional and off the critical path.

## Drift detector

Two-proportion z-test: rolling window takeover rate (from `$setWindowFields` over the time series) against the baseline rate. Alarm when the p-value falls below threshold and enough missions have passed since the last change. The alarm emits `drift_alarm` and Raindrop signals, and wakes the fixer. Latency is measured in missions since the change.

## Long-horizon runner

A checkpointed state machine. Each step is idempotent, keyed by `run/day/shift/step`. After a kill -9 the runner reloads the checkpoint, skips completed keys and continues. Results from resumed and uninterrupted runs should match; that claim stays a placeholder until tested.

## Raindrop event budget

- Hard cap of 900 events per process (a module-level counter), enforced in `iz/obs/rd.py` (`CAP`, override with `RAINDROP_EVENT_CAP`). `iz/obs/budget.py` is the separate LLM dollar budget.
- `user_id` = robot id, `convo_id` = `site:day:shift`, event = `mission | fix_cycle | coach_cycle`.
- Self-reports use the default signal `agent_report`.
- Replays and gate runs never log. `RAINDROP_ENABLED=0` by default.
