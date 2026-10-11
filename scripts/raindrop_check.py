"""Live Raindrop round-trip check. Sends EXACTLY 1 mission event (+1 tool span) and 2 signals, then polls the Query SDK.
Run: .venv\\Scripts\\python.exe scripts\\raindrop_check.py [--readback-only] [--timeout 60]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

USER_ID = "robot-check"
SITE = "check"
DAY = 0
SHIFT = 0
EVENT = "mission"
OUT = ROOT / "out" / "raindrop_check.json"


def _scrub(s: str) -> str:
    for name in ("RAINDROP_WRITE_KEY", "RAINDROP_QUERY_API_KEY"):
        v = os.getenv(name)
        if v:
            s = s.replace(v, "***")
    return s


def _write(res: dict) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")


class _Collect(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.ERROR)
        self.records: list = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


def _agent_probe(rd, event_id) -> dict:
    """Zero-network probe of signal_type='agent'. track_signal only runs if the model rejects it."""
    from pydantic import ValidationError
    from raindrop.models import DefaultSignal

    info: dict = {"model_rejected": None, "model_error": None, "track_signal_called": False,
                  "track_signal_raised": None, "track_signal_logged_error": None,
                  "network_sent": False, "conclusion": None}
    try:
        DefaultSignal(event_id=event_id or "probe", signal_name="agent_report",
                      properties={"kind": "missing_context"}, signal_type="agent")
        info["model_rejected"] = False
    except ValidationError as e:
        info["model_rejected"] = True
        info["model_error"] = _scrub(" | ".join(str(e).splitlines()[:3]))[:200]
    if info["model_rejected"]:
        lg = logging.getLogger("raindrop.analytics")
        h = _Collect()
        lg.addHandler(h)
        try:
            info["track_signal_called"] = True
            try:
                rd._sdk().track_signal(event_id=event_id, name="agent_report", signal_type="agent",
                                       properties={"kind": "missing_context", "probe": True})
                info["track_signal_raised"] = False
            except Exception as e:  # noqa: BLE001
                info["track_signal_raised"] = True
                info["raise_type"] = type(e).__name__
        finally:
            lg.removeHandler(h)
        info["track_signal_logged_error"] = any(
            "Invalid data passed to track_signal" in r.getMessage() for r in h.records)
        info["conclusion"] = (
            f"rejected (pydantic Literal['default']); raised={info['track_signal_raised']}, "
            f"logged_error={info['track_signal_logged_error']}; dropped before buffering")
    else:
        info["conclusion"] = "model ACCEPTED 'agent'; track_signal not called to avoid a third signal"
    return info


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--interval", type=float, default=5.0)
    ap.add_argument("--readback-only", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)

    from iz.obs import rd, rd_query

    t0 = datetime.now(timezone.utc)
    res: dict = {"ts": t0.isoformat(), "user_id": USER_ID, "convo_id": f"{SITE}:{DAY}:{SHIFT}",
                 "event_name": EVENT, "poll_timeout_s": args.timeout}
    event_id = None

    if args.readback_only:
        prev = json.loads(OUT.read_text(encoding="utf-8"))
        res = prev
        res["poll_timeout_s"] = args.timeout
        run_id = prev["run_id"]
        event_id = prev.get("event_id")
        t_sent = prev.get("t_sent_epoch") or time.time()
        t0 = datetime.fromisoformat(prev["ts"])
        res["readback_only_rerun"] = True
    else:
        run_id = f"raindrop-check-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
        res["run_id"] = run_id
        rd.force_enable()
        if not rd.enabled():
            res.update(sent=False, visible=False, latency_s=None, error="RAINDROP_WRITE_KEY missing")
            _write(res)
            print("FAIL RAINDROP_WRITE_KEY missing")
            return 1
        props = {"site": SITE, "policy_version": 0, "run_id": run_id, "situation_type": "fire_door",
                 "model": "raindrop-check", "day": DAY, "shift": SHIFT}
        h = rd.mission_begin(USER_ID, SITE, DAY, SHIFT, "raindrop check mission: fire door on route", props)
        event_id = h.event_id
        if event_id is None:
            res.update(sent=False, event_sent=False, visible=False, latency_s=None, event_id=None,
                       events_sent_counter=rd.events_sent(), error="mission_begin returned no event id")
            _write(res)
            print("FAIL mission_begin returned no event id")
            return 1
        h.tool("wait_for_staff", {"door": "D-17", "seconds": 5},
               {"ok": True, "door_opened_by": "staff"}, duration_ms=5000.0)
        h.finish("mission complete after operator takeover")
        sig_takeover = rd.signal(event_id, "takeover",
                                 {"situation_type": "fire_door", "site": SITE, "run_id": run_id})
        sig_report = rd.agent_report(event_id, "missing_context", "no rule for magnet-held fire doors",
                                     {"site": SITE, "run_id": run_id})
        rd.flush()
        t_sent = time.time()
        n = rd.events_sent()
        res.update(sent=bool(event_id and sig_takeover and sig_report), event_sent=True,
                   signals_sent={"takeover": bool(sig_takeover), "agent_report": bool(sig_report)},
                   events_sent_counter=n, event_id=event_id, t_sent_epoch=t_sent)
        print(f"SENT event_id={event_id} run_id={run_id} events_sent={n} "
              f"takeover={sig_takeover} agent_report={sig_report}")
        if n > 3:
            print(f"FAIL events_sent={n} > 3")
        # Persist the send record immediately so a crash while polling cannot lose it.
        res.update(visible=False, latency_s=None)
        _write(res)
        res["agent_signal_type"] = _agent_probe(rd, event_id)
        a = res["agent_signal_type"]
        print(f"AGENT_TYPE model_rejected={a['model_rejected']} logged_error={a['track_signal_logged_error']} "
              f"raised={a['track_signal_raised']}")

    # STEP 3 - poll
    since = t0 - timedelta(minutes=2)
    ev_vis = False
    names: list = []
    id_match = None
    tool_n = None
    lat_event = None
    lat = None
    attempts = 0
    last_error = None
    query_degraded = False
    query_error = None
    client = None
    try:
        client = rd_query._client()
    except Exception as e:  # noqa: BLE001
        query_degraded = True
        query_error = _scrub(f"{type(e).__name__}: {e}")[:200]
    poll_start = time.time()  # deadline is relative to poll start, so --readback-only reruns actually poll
    while client is not None and time.time() - poll_start <= args.timeout:
        attempts += 1
        try:
            evs = client.events.list(limit=25, user_id={"eq": USER_ID}, timestamp={"gte": since}).data or []
        except Exception as e:  # noqa: BLE001
            last_error = _scrub(f"{type(e).__name__}: {e}")[:200]
            try:
                evs = [x for x in rd_query._events_since(client, since) if getattr(x, "user_id", None) == USER_ID]
            except Exception as e2:  # noqa: BLE001
                last_error = _scrub(f"{type(e2).__name__}: {e2}")[:200]
                time.sleep(args.interval)
                continue
        match = next((x for x in evs if getattr(x, "id", None) == event_id
                      or (getattr(x, "properties", None) or {}).get("run_id") == run_id), None)
        if match is not None:
            if not ev_vis:
                ev_vis = True
                lat_event = round(time.time() - t_sent, 1)
            id_match = (match.id == event_id)
            names = sorted({s.name for s in (match.signals or [])})
            tool_n = len(match.tool_calls or [])
        print(f"POLL attempt {attempts}: event_visible={ev_vis} signals={names}", flush=True)
        if {"takeover", "agent_report"} <= set(names):
            lat = round(time.time() - t_sent, 1)
            break
        time.sleep(args.interval)

    eg: dict = {"ok": False, "error": None, "signals": None}
    if client is not None and event_id:
        try:
            r = client.events.get(id=event_id)
            d = getattr(r, "data", r)
            sigs = getattr(d, "signals", None)
            eg["ok"] = True
            if sigs is not None:
                eg["signals"] = sorted({getattr(s, "name", str(s)) for s in sigs})
        except Exception as e:  # noqa: BLE001
            eg["error"] = _scrub(f"{type(e).__name__}: {e}")[:200]

    readback: dict = {}
    try:
        r1 = rd_query.top_takeover_causes(since)
        r2 = rd_query.agent_reports(since)
        readback = {
            "top_takeover_causes": {
                "degraded": r1["degraded"],
                "check_row_present": any(c["site"] == SITE and c["situation_type"] == "fire_door"
                                         for c in r1.get("causes", [])),
                "error": _scrub(str(r1["error"])) if r1.get("error") else None},
            "agent_reports": {"degraded": r2["degraded"],
                              "kinds": [x["kind"] for x in r2.get("reports", [])],
                              "error": _scrub(str(r2["error"])) if r2.get("error") else None}}
    except Exception as e:  # noqa: BLE001
        readback = {"error": _scrub(f"{type(e).__name__}: {e}")[:200]}

    visible = bool(ev_vis and {"takeover", "agent_report"} <= set(names))
    poll = dict(visible=visible, event_visible=ev_vis, signals_visible=names, id_match=id_match,
                tool_calls_n=tool_n, latency_s=lat, latency_event_s=lat_event, poll_attempts=attempts,
                poll_last_error=last_error, query_degraded=query_degraded, query_error=query_error,
                events_get=eg, readback=readback)
    if args.readback_only:
        # Keep the original live-run record intact; a rerun's latencies are measured from the
        # original send time and so are not comparable. Record the rerun separately.
        poll["ts"] = datetime.now(timezone.utc).isoformat()
        res["readback_rerun"] = poll
    else:
        res.update(poll)
    _write(res)
    print(f"VISIBLE event={ev_vis} signals={names} latency_s={lat} latency_event_s={lat_event} "
          f"id_match={id_match} attempts={attempts}")
    print(f"READBACK {json.dumps(readback, default=str)}")
    if args.readback_only:
        return 0 if visible else 2
    if not res.get("sent"):
        return 1
    return 0 if visible else 2


if __name__ == "__main__":
    try:
        rc = main()
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        msg = _scrub(f"{type(exc).__name__}: {exc}")[:300]
        print("ERROR", msg)
        try:
            cur = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
        except Exception:  # noqa: BLE001
            cur = {}
        cur.setdefault("sent", False)
        cur.setdefault("visible", False)
        cur.setdefault("latency_s", None)
        cur["error"] = msg
        _write(cur)
        rc = 1 if not cur.get("sent") else 2
    sys.exit(rc)
