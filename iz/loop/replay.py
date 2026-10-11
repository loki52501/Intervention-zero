"""Replay incidents under a genome. robot_fn / judge / gateway are injected."""
from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

from iz.types import Action, Genome, Judge, Outcome, OutcomeStatus, RobotView, SetScore, Situation

RobotFn = Callable[[RobotView, Genome], list[Action]]
ReplayFn = Callable[[Genome, list[Situation]], list[Outcome]]


def _one(sit: Situation, genome: Genome, robot_fn: RobotFn, judge: Judge,
         gateway_fn: Callable[[RobotView, list[Action]], list[Action]] | None) -> Outcome:
    view = sit.view()
    try:
        actions = robot_fn(view, genome)
        if gateway_fn:
            actions = gateway_fn(view, actions)
    except Exception as e:  # noqa: BLE001
        return Outcome(status=OutcomeStatus.TAKEOVER, reason=f"agent_error: {type(e).__name__}")
    try:
        return judge.judge(sit, actions)
    except Exception as e:  # noqa: BLE001
        return Outcome(status=OutcomeStatus.TAKEOVER, reason=f"judge_error: {type(e).__name__}: {e}")


def replay(incidents: list[Situation], genome: Genome, robot_fn: RobotFn, judge: Judge,
           gateway_fn: Callable[[RobotView, list[Action]], list[Action]] | None = None,
           max_workers: int = 8) -> list[Outcome]:
    if not incidents:
        return []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        return list(ex.map(lambda s: _one(s, genome, robot_fn, judge, gateway_fn), incidents))


def make_replay_fn(robot_fn: RobotFn, judge: Judge,
                   gateway_factory: Callable[[Genome], Callable] | None = None) -> ReplayFn:
    memo: dict[tuple[str, tuple[str, ...]], list[Outcome]] = {}

    def fn(genome: Genome, sits: list[Situation]) -> list[Outcome]:
        key = (hashlib.sha256(genome.model_dump_json().encode()).hexdigest(), tuple(s.id for s in sits))
        if key not in memo:
            memo[key] = replay(sits, genome, robot_fn, judge, gateway_factory(genome) if gateway_factory else None)
        return list(memo[key])

    return fn


def score(outcomes: list[Outcome]) -> SetScore:
    return SetScore(n=len(outcomes), takeover_free=sum(o.takeover_free for o in outcomes),
                    safety_violations=sum(o.unsafe for o in outcomes))
