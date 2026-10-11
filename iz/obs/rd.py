"""Raindrop wrapper. No-op unless RAINDROP_ENABLED=1 and RAINDROP_WRITE_KEY is set.

Real SDK API used (raindrop-ai 0.0.75): raindrop.analytics.init(api_key, tracing_enabled),
begin(user_id, event, properties, input, convo_id, model) -> Interaction,
Interaction.track_tool(name=, input=, output=, duration_ms=) / .finish(output=) / .id,
track_signal(event_id, name, signal_type in default|feedback|edit, properties, sentiment), flush().
Never raises into the robot loop.
"""
from __future__ import annotations

import atexit
import logging
import os
import threading
from typing import Any, Literal

log = logging.getLogger("iz.obs.rd")

SIGNAL_NAMES = {
    "takeover", "takeover_avoidable", "guardrail_blocked", "safety_violation", "slow_mission",
    "edit_promoted", "edit_rejected", "drift_detected", "rollback", "coach_promoted", "agent_report",
}
AGENT_REPORT_KINDS = {"missing_context", "capability_gap", "repeatedly_broken_tool", "complete_task_failure"}
REQUIRED_PROPS = ("site", "policy_version", "run_id", "situation_type", "model", "day", "shift")

_lock = threading.RLock()
_enabled = False
_count = 0
CAP = 900
_initialized = False
_cap_logged = False
_SDK: Any = None  # raindrop.analytics module (or a stub in tests)


def _load_env() -> None:
    global _enabled, CAP
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:  # noqa: BLE001
        pass
    _enabled = os.getenv("RAINDROP_ENABLED") == "1" and bool(os.getenv("RAINDROP_WRITE_KEY"))
    try:
        CAP = int(os.getenv("RAINDROP_EVENT_CAP") or 900)
    except ValueError:
        CAP = 900


_load_env()


def force_enable() -> None:
    """Used by smoke.py --raindrop only."""
    global _enabled
    os.environ["RAINDROP_ENABLED"] = "1"
    _enabled = bool(os.getenv("RAINDROP_WRITE_KEY"))


def enabled() -> bool:
    return _enabled


def events_sent() -> int:
    return _count


def _take(n: int = 1) -> bool:
    global _count, _cap_logged
    with _lock:
        if _count + n > CAP:
            if not _cap_logged:
                log.warning("Raindrop event cap %d reached; further events are dropped", CAP)
                _cap_logged = True
            return False
        _count += n
        return True


def base_props(**kw: Any) -> dict:
    out = dict(kw)
    for k in REQUIRED_PROPS:
        if k not in out or out[k] is None:
            out[k] = "" if k in ("site", "run_id", "situation_type", "model") else None
    return out


def _sdk() -> Any:
    """Initialise lazily; returns the SDK or None on failure."""
    global _SDK, _initialized
    with _lock:
        if _initialized:
            return _SDK
        try:
            if _SDK is None:
                import raindrop.analytics as sdk
                _SDK = sdk
            _SDK.init(os.getenv("RAINDROP_WRITE_KEY"), tracing_enabled=True)
            atexit.register(flush)
        except Exception as e:  # noqa: BLE001
            log.warning("Raindrop init failed: %s", type(e).__name__)
            _SDK = None
        _initialized = True
        return _SDK


class NullHandle:
    event_id: str | None = None

    def tool(self, name: str, input: Any, output: Any, duration_ms: float | None = None) -> None:
        return None

    def finish(self, output: str, properties: dict | None = None) -> None:
        return None


class MissionHandle:
    def __init__(self, interaction: Any):
        self._i = interaction
        try:
            self.event_id: str | None = str(interaction.id)
        except Exception:  # noqa: BLE001
            self.event_id = None

    def tool(self, name: str, input: Any, output: Any, duration_ms: float | None = None) -> None:
        try:
            if hasattr(self._i, "track_tool"):
                self._i.track_tool(name=name, input=input, output=output, duration_ms=duration_ms)
            else:
                sp = self._i.start_span(kind="tool", name=name)
                sp.record_input(input)
                sp.record_output(output)
                sp.end()
        except Exception as e:  # noqa: BLE001
            log.warning("raindrop tool span failed: %s", type(e).__name__)

    def finish(self, output: str, properties: dict | None = None) -> None:
        try:
            if properties:
                self._i.set_properties(properties)
            self._i.finish(output=output)
        except Exception as e:  # noqa: BLE001
            log.warning("raindrop finish failed: %s", type(e).__name__)


def _begin(user_id: str, event: str, convo_id: str | None, input: str, props: dict) -> "MissionHandle | NullHandle":
    if not _enabled or not _take():
        return NullHandle()
    sdk = _sdk()
    if sdk is None:
        return NullHandle()
    try:
        i = sdk.begin(user_id=user_id, event=event, properties=props, input=input,
                      convo_id=convo_id, model=props.get("model") or None)
        return MissionHandle(i)
    except Exception as e:  # noqa: BLE001
        log.warning("raindrop begin failed: %s", type(e).__name__)
        return NullHandle()


def mission_begin(robot_id: str, site: str, day: int, shift: int, brief: str,
                  properties: dict, event: str = "mission") -> "MissionHandle | NullHandle":
    props = base_props(**{**properties, "site": site, "day": day, "shift": shift})
    return _begin(robot_id, event, f"{site}:{day}:{shift}", brief, props)


def cycle_begin(kind: Literal["fix_cycle", "coach_cycle"], run_id: str, input: str,
                properties: dict) -> "MissionHandle | NullHandle":
    if kind not in ("fix_cycle", "coach_cycle"):
        raise ValueError(f"bad cycle kind {kind!r}")
    user = "fixer" if kind == "fix_cycle" else "coach"
    props = base_props(**{**properties, "run_id": run_id})
    return _begin(user, kind, run_id, input, props)


def signal(event_id: str | None, name: str, properties: dict | None = None,
           sentiment: Literal["POSITIVE", "NEGATIVE"] = "NEGATIVE", comment: str | None = None) -> bool:
    if name not in SIGNAL_NAMES:
        raise ValueError(f"unknown signal name {name!r}")
    if not _enabled or not event_id or not _take():
        return False
    sdk = _sdk()
    if sdk is None:
        return False
    props = dict(properties or {})
    if comment:
        props["comment"] = comment
    try:
        sdk.track_signal(event_id=event_id, name=name, signal_type="default",
                         properties=props, sentiment=sentiment)
    except Exception as e:  # noqa: BLE001
        log.warning("raindrop signal failed: %s", type(e).__name__)
        return False
    try:
        from iz.store import feed
        feed.emit(feed.RAINDROP_SENT, {"name": name, "event_id": event_id})
    except Exception:  # noqa: BLE001
        pass
    return True


def agent_report(event_id: str | None, kind: str, detail: str, properties: dict | None = None) -> bool:
    if kind not in AGENT_REPORT_KINDS:
        raise ValueError(f"unknown agent_report kind {kind!r}")
    props = {**(properties or {}), "kind": kind, "detail": detail}
    return signal(event_id, "agent_report", props, sentiment="NEGATIVE")


def flush() -> None:
    if not _initialized or _SDK is None:
        return
    try:
        _SDK.flush()
    except Exception as e:  # noqa: BLE001
        log.warning("raindrop flush failed: %s", type(e).__name__)
