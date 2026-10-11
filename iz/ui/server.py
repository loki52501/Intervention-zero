"""Intervention Zero UI server: static page + SSE event stream."""
from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse

log = logging.getLogger("iz.ui")

REPO = Path(__file__).resolve().parents[2]
STATIC = Path(__file__).parent / "static"
OUT = REPO / "out"

app = FastAPI(title="Intervention Zero UI")
STATIC.mkdir(parents=True, exist_ok=True)
from iz.ui.studio import router as studio_router  # noqa: E402
app.include_router(studio_router)
from iz.ui.evidence import router as evidence_router  # noqa: E402
app.include_router(evidence_router)
app.mount("/static", StaticFiles(directory=STATIC, check_dir=False), name="static")


@app.get("/")
async def index():
    return FileResponse(STATIC / "index.html")


@app.get("/clips/{name}")
async def clip(name: str):
    if not name or "/" in name or "\\" in name or ".." in name:
        raise HTTPException(404)
    p = OUT / "clips" / name
    if not p.is_file():
        raise HTTPException(404)
    return FileResponse(p)


def _parse_ts(ts) -> float | None:
    try:
        return datetime.fromisoformat(str(ts).replace("Z", "+00:00")).timestamp()
    except Exception:
        return None


def _msg(ev: dict) -> dict:
    return {"data": json.dumps(ev), "id": str(ev.get("id", ""))}


def _load_jsonl(path: Path) -> list[dict]:
    out = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except Exception:
            continue
        if isinstance(ev, dict):
            out.append(ev)
    return out


async def _replay_gen(request: Request, path: Path, speed: float, after: str | None = None):
    speed = speed if speed > 0 else 1.0
    evs = _load_jsonl(path)
    if after:
        ids = [str(e.get("id")) for e in evs]
        if after in ids:
            evs = evs[ids.index(after) + 1:]
            if not evs:
                # EventSource reconnected after the replay finished: hold the
                # connection open instead of letting it restart from the top.
                while not await request.is_disconnected():
                    await asyncio.sleep(1.0)
                return
    prev = None
    for ev in evs:
        if await request.is_disconnected():
            return
        t = _parse_ts(ev.get("ts"))
        if prev is not None:
            if t is not None and prev[0] is not None:
                delay = min(max(t - prev[0], 0) / speed, 2.0)
            else:
                delay = 0.5 / speed
            if delay > 0:
                await asyncio.sleep(delay)
        prev = (t,)
        yield _msg(ev)


def _normalise(doc: dict) -> dict:
    doc = dict(doc)
    _id = doc.pop("_id", None)
    return {
        "id": str(doc.get("id") or _id or ""),
        "ts": doc.get("ts"),
        "kind": doc.get("kind"),
        "run_id": doc.get("run_id", ""),
        "payload": doc.get("payload", {}),
    }


async def _mongo_gen(request: Request, after: str | None):
    from anyio import to_thread

    from iz.store import feed  # lazy

    from bson import ObjectId

    if after and not ObjectId.is_valid(after):
        after = None
    it = iter(feed.tail(after))
    sentinel = object()
    while True:
        if await request.is_disconnected():
            return
        doc = await to_thread.run_sync(next, it, sentinel)
        if doc is sentinel:
            return
        yield _msg(_normalise(doc))


async def _file_gen(request: Request, path: Path, after: str | None):
    pos = 0
    skipping = after is not None
    buf = b""
    while True:
        if await request.is_disconnected():
            return
        try:
            size = path.stat().st_size
        except OSError:
            size = None
        if size is None:
            await asyncio.sleep(0.5)
            continue
        if size < pos:
            pos, buf = 0, b""
        if size > pos:
            with open(path, "rb") as f:
                f.seek(pos)
                chunk = f.read()
            pos += len(chunk)
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line.decode("utf-8", "replace"))
                except Exception:
                    continue
                if not isinstance(ev, dict):
                    continue
                if skipping:
                    if str(ev.get("id")) == after:
                        skipping = False
                    continue
                yield _msg(ev)
        else:
            await asyncio.sleep(0.5)


async def _live_gen(request: Request, after: str | None):
    if os.environ.get("UI_SOURCE") == "mongo":
        started = False
        try:
            async for m in _mongo_gen(request, after):
                started = True
                yield m
            return
        except Exception as e:  # fall back to file mode
            log.warning("mongo feed unavailable (%s); falling back to file", e)
            if started:
                return
    path = Path(os.environ.get("UI_EVENTS_FILE") or (OUT / "events.jsonl"))
    async for m in _file_gen(request, path, after):
        yield m


@app.get("/events")
async def events(request: Request, replay: str | None = None, speed: float = 4.0,
                 after: str | None = None):
    after = after or request.headers.get("last-event-id") or None
    if replay:
        p = Path(replay)
        if not p.is_absolute():
            p = REPO / p
        p = p.resolve()
        if not p.is_relative_to(REPO.resolve()):
            raise HTTPException(400, "replay path must be inside the repo")
        if not p.is_file():
            raise HTTPException(404, "replay file not found")
        gen = _replay_gen(request, p, speed, after)
    else:
        gen = _live_gen(request, after)
    return EventSourceResponse(gen, ping=15)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
