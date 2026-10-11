"""Raindrop read-back (raindrop_query 0.1.14).

Real API: RaindropQuery(api_key=...).events.list(limit, cursor, timestamp={'gte': dt}, event_name={'eq': ..})
-> resp.data[] (id, event_name, timestamp, properties, signals[{id,name}]), resp.meta.{cursor,has_more}.
Signals are attached to events (signals.list only returns signal definitions), so we list events and
group by the signal names present on them, using event properties. Any failure -> {'degraded': True}.
"""
from __future__ import annotations

import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any

MAX_PAGES = 20


def _client() -> Any:
    key = os.getenv("RAINDROP_QUERY_API_KEY")
    if not key:
        try:
            from dotenv import load_dotenv
            load_dotenv()
            key = os.getenv("RAINDROP_QUERY_API_KEY")
        except Exception:  # noqa: BLE001
            pass
    if not key:
        raise RuntimeError("RAINDROP_QUERY_API_KEY missing")
    from raindrop_query import RaindropQuery
    return RaindropQuery(api_key=key)


def _degraded(e: Exception) -> dict:
    msg = str(e).replace("\n", " ")[:120]
    key = os.getenv("RAINDROP_QUERY_API_KEY")
    if key:
        msg = msg.replace(key, "***")
    return {"degraded": True, "error": f"{type(e).__name__}: {msg}", "source": "raindrop"}


def _events_since(client: Any, since: datetime) -> list:
    out: list = []
    cursor = None
    for _ in range(MAX_PAGES):
        resp = client.events.list(limit=100, cursor=cursor, timestamp={"gte": since})
        out.extend(resp.data or [])
        if not resp.meta.has_more or not resp.meta.cursor:
            break
        cursor = resp.meta.cursor
    return out


def _sig_names(ev: Any) -> set:
    return {s.name for s in (getattr(ev, "signals", None) or [])}


def _props(ev: Any) -> dict:
    return dict(getattr(ev, "properties", None) or {})


def top_takeover_causes(since: datetime, limit: int = 10) -> dict:
    try:
        c: Counter = Counter()
        for ev in _events_since(_client(), since):
            if _sig_names(ev) & {"takeover", "takeover_avoidable"}:
                p = _props(ev)
                c[(p.get("situation_type") or "unknown", p.get("site") or "")] += 1
        causes = [{"situation_type": k[0], "site": k[1], "count": n} for k, n in c.most_common(limit)]
        return {"degraded": False, "source": "raindrop", "causes": causes}
    except Exception as e:  # noqa: BLE001
        return _degraded(e)


def agent_reports(since: datetime) -> dict:
    try:
        counts: Counter = Counter()
        examples: dict = defaultdict(list)
        for ev in _events_since(_client(), since):
            if "agent_report" in _sig_names(ev):
                p = _props(ev)
                kind = p.get("agent_report_kind") or p.get("kind") or "unknown"
                counts[kind] += 1
                detail = p.get("agent_report_detail") or p.get("detail")
                if detail and len(examples[kind]) < 3:
                    examples[kind].append(str(detail)[:200])
        reports = [{"kind": k, "count": n, "examples": examples[k]} for k, n in counts.most_common()]
        return {"degraded": False, "source": "raindrop", "reports": reports}
    except Exception as e:  # noqa: BLE001
        return _degraded(e)


_BUCKET_FMT = {"minute": "%Y-%m-%dT%H:%M:00Z", "hour": "%Y-%m-%dT%H:00:00Z", "day": "%Y-%m-%dT00:00:00Z"}


def signal_timeseries(name: str, since: datetime | None = None, bucket: str = "hour") -> dict:
    try:
        fmt = _BUCKET_FMT.get(bucket)
        if fmt is None:
            raise ValueError(f"bucket must be one of {sorted(_BUCKET_FMT)}")
        since = since or datetime(1970, 1, 1, tzinfo=timezone.utc)
        c: Counter = Counter()
        for ev in _events_since(_client(), since):
            if name in _sig_names(ev):
                ts = ev.timestamp
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                c[ts.astimezone(timezone.utc).strftime(fmt)] += 1
        return {"degraded": False, "source": "raindrop",
                "series": [{"t": t, "count": n} for t, n in sorted(c.items())]}
    except Exception as e:  # noqa: BLE001
        return _degraded(e)
