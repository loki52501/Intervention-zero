"""Prompt Studio: one screen - prompt -> world -> learning. FastAPI APIRouter (see static/studio.html)."""
from __future__ import annotations

import asyncio
import json
import re
import threading
import uuid
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

REPO = Path(__file__).resolve().parents[2]
STATIC = Path(__file__).parent / "static"
OUT = REPO / "out"
STUDIO = OUT / "studio"

router = APIRouter()

N_TRAIN, N_HELD, N_SAFETY = 8, 8, 4
KEY_BASE = {"train": 0, "heldout": 100, "safety": 200}
_WORLDS: dict[str, dict] = {}       # slug -> {"spec": dict, "registry": {key: spec dict}, "counts": ...}
_RUNNING: dict[str, threading.Thread] = {}
_LOCK = threading.Lock()


class WorldReq(BaseModel):
    prompt: str


class LearnReq(BaseModel):
    slug: str
    generations: int = 3
    fake: bool = False


def _slug(prompt: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", prompt.lower()).strip("-")[:36].strip("-") or "world"


def _safe_slug(slug: str) -> str:
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,60}", slug or ""):
        raise HTTPException(400, "bad slug")
    return slug


def _emit(slug: str, kind: str, payload: dict) -> dict:
    d = STUDIO / slug
    d.mkdir(parents=True, exist_ok=True)
    with _LOCK:
        ev = {"id": uuid.uuid4().hex[:10], "ts": datetime.now(timezone.utc).isoformat(),
              "kind": kind, "payload": payload}
        with open(d / "events.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(ev, default=str) + "\n")
    return ev


# ------------------------------------------------------------------ world
def create_world(prompt: str, fake: bool = False) -> dict:
    from iz.worldgen import build as B
    from iz.worldgen import generate as G
    llm = None
    if fake:
        from iz.llm.client import FakeLLM
        llm = FakeLLM(responses=[{}])
    spec = G.prompt_to_spec(prompt, llm)
    repairs = list(spec.notes)
    slug = _slug(prompt)
    wdir = STUDIO / slug
    wdir.mkdir(parents=True, exist_ok=True)
    (wdir / "events.jsonl").write_text("", encoding="utf-8")
    _emit(slug, "ticker", {"msg": "LLM wrote the world spec; validator repaired it", "stage": "spec"})

    registry: dict[int, dict] = {}
    counts = {"train": 0, "heldout": 0, "safety": 0}
    sizes = {"train": N_TRAIN, "heldout": N_HELD, "safety": N_SAFETY}
    first = None
    for split, n in sizes.items():
        base = spec
        if split == "safety":  # dense shoppers
            base = spec.model_copy(deep=True)
            base.shoppers.count = min(8, max(base.shoppers.count + 2, 4))
            base.shoppers.density = "rush_hour"
        seed_off = {"train": 1, "heldout": 2, "safety": 3}[split]
        for i, v in enumerate(G.variations(base, n, seed=seed_off)):
            try:
                xml, meta = B.build_world(v, i)
                probs = B.check_world(xml, meta)
            except Exception as e:  # noqa: BLE001
                probs, xml, meta = [f"build error: {e}"], None, None
            if probs:
                continue
            registry[KEY_BASE[split] + i] = v.model_dump(mode="json")
            counts[split] += 1
            if first is None:
                first = (xml, meta)
    preview_url = None
    if first:
        try:
            B.render_preview(first[0], first[1], str(wdir / "preview.png"))
            preview_url = f"/api/studio/file/{slug}/preview.png"
        except Exception as e:  # noqa: BLE001
            _emit(slug, "ticker", {"msg": f"preview failed: {e}", "stage": "validator"})
    spec_d = spec.model_dump(mode="json")
    _WORLDS[slug] = {"spec": spec_d, "registry": registry, "counts": counts}
    (wdir / "world.json").write_text(json.dumps({"spec": spec_d, "registry": registry, "counts": counts}),
                                     encoding="utf-8")
    _emit(slug, "ticker", {"msg": f"validator: {sum(counts.values())} valid worlds "
                                  f"(train {counts['train']} / held-out {counts['heldout']} / safety {counts['safety']})",
                           "stage": "validator"})
    _emit(slug, "world", {"counts": counts, "preview_url": preview_url})
    return {"slug": slug, "spec": spec_d, "repairs": repairs, "counts": counts, "preview_url": preview_url}


def _load_world(slug: str) -> dict:
    w = _WORLDS.get(slug)
    if w:
        return w
    p = STUDIO / slug / "world.json"
    if not p.is_file():
        raise HTTPException(404, "unknown world; create it first")
    d = json.loads(p.read_text(encoding="utf-8"))
    d["registry"] = {int(k): v for k, v in d["registry"].items()}
    _WORLDS[slug] = d
    return d


# ------------------------------------------------------------------ learning
def _worker(args: tuple) -> dict:
    params, spec, key = args
    from iz.shop.episode import run_episode
    return run_episode(params, key, spec=spec, world_id=str(spec.get("name")))


def make_evaluator(registry: dict, workers: int = 4):
    def evaluate(params: dict, seeds) -> list[dict]:
        seeds = list(seeds)
        if not seeds:
            return []
        jobs = [(params, registry[s], s) for s in seeds]
        if len(jobs) == 1:
            return [_worker(jobs[0])]
        with ProcessPoolExecutor(workers) as ex:
            return list(ex.map(_worker, jobs))
    return evaluate


class _StudioEvents:
    """Replaces iz.shop.learn._Events: writes to the studio stream and derives ticker lines."""

    def __init__(self, slug: str):
        self.slug = slug

    def emit(self, kind: str, payload: dict) -> dict:
        _emit(self.slug, kind, payload)
        msg = None
        if kind == "takeover":
            msg = f"takeover record: {payload.get('failure')} (seed {payload.get('seed')})"
        elif kind == "fix_proposed":
            msg = f"LLM fixer proposed {len(payload.get('edits') or {})} parameter edit(s)"
        elif kind == "gate_result":
            msg = f"gate on held-out + safety: {payload.get('verdict')}"
        elif kind == "promoted":
            msg = "Atlas write: promoted genome stored"
        if msg:
            _emit(self.slug, "ticker", {"msg": msg, "stage": kind})
        return {}


def _gen_payload(h: dict) -> dict:
    g = h.get("gate") or {}
    return {"gen": h["gen"], "verdict": h["verdict"], "edits": h.get("edits", {}),
            "rationale": h.get("rationale", ""), "reason": h.get("reason", ""),
            "heldout_success": h["heldout_success"], "heldout_n": h["heldout_n"],
            "train_rate": h.get("train_rate"), "safety_violations": g.get("safety_violations", 0)}


def run_learning_for(slug: str, generations: int = 3, fake: bool = False) -> list[dict]:
    """Learn in the generated worlds with the real fixer + promotion gate (iz.shop.learn)."""
    from iz.shop import learn
    w = _load_world(slug)
    reg = w["registry"]
    tr = sorted(k for k in reg if k < 100)
    ho = sorted(k for k in reg if 100 <= k < 200)
    sf = sorted(k for k in reg if k >= 200)
    if not tr or not ho:
        raise RuntimeError("not enough valid worlds to learn")
    if fake:
        llm = learn.fake_llm()
    else:
        from iz.llm.client import make_llm
        llm = make_llm()
    old_eval, old_ev = learn.evaluate, learn._Events
    learn.evaluate = make_evaluator(reg)
    learn._Events = lambda rid, path: _StudioEvents(slug)
    try:
        _emit(slug, "ticker", {"msg": "memory recall: loading prior genomes", "stage": "memory"})
        return learn.run_learning(generations=generations, train_seeds=tr, heldout_seeds=ho,
                                  safety_seeds=sf, llm=llm, out_dir=str(STUDIO / slug),
                                  run_id="studio-" + uuid.uuid4().hex[:6],
                                  on_generation=lambda h: _emit(slug, "generation", _gen_payload(h)))
    finally:
        learn.evaluate, learn._Events = old_eval, old_ev


def _learn_thread(slug: str, generations: int, fake: bool) -> None:
    try:
        hist = run_learning_for(slug, generations, fake)
        _emit(slug, "done", {"generations": max(0, len(hist or []) - 1)})
    except Exception as e:  # noqa: BLE001
        _emit(slug, "error", {"message": f"{type(e).__name__}: {e}"})
    finally:
        _RUNNING.pop(slug, None)


# ------------------------------------------------------------------ routes
@router.get("/studio")
async def studio_page():
    return FileResponse(STATIC / "studio.html")


@router.post("/api/studio/world")
async def api_world(req: WorldReq, fake: int = 0):
    if not req.prompt.strip():
        raise HTTPException(400, "empty prompt")
    try:
        return await asyncio.to_thread(create_world, req.prompt.strip(), bool(fake))
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"{type(e).__name__}: {e}")


@router.post("/api/studio/learn")
async def api_learn(req: LearnReq):
    slug = _safe_slug(req.slug)
    _load_world(slug)
    if slug in _RUNNING:
        return {"started": False, "reason": "already running"}
    gens = max(1, min(8, req.generations))
    t = threading.Thread(target=_learn_thread, args=(slug, gens, req.fake), daemon=True)
    _RUNNING[slug] = t
    t.start()
    return {"started": True, "slug": slug, "generations": gens}


@router.get("/api/studio/events/{slug}")
async def api_events(slug: str, request: Request):
    slug = _safe_slug(slug)
    path = STUDIO / slug / "events.jsonl"

    async def gen():
        pos = 0
        while True:
            if await request.is_disconnected():
                return
            if path.is_file():
                with open(path, "rb") as f:
                    f.seek(pos)
                    chunk = f.read()
                end = chunk.rfind(b"\n")
                if end >= 0:
                    pos += end + 1
                    for line in chunk[:end].decode("utf-8").splitlines():
                        if line.strip():
                            ev = json.loads(line)
                            yield {"event": ev["kind"], "id": ev["id"], "data": json.dumps(ev)}
                            if ev["kind"] in ("done", "error"):
                                return
            await asyncio.sleep(0.2)

    return EventSourceResponse(gen(), ping=15)


@router.get("/api/studio/file/{slug}/{name}")
async def api_file(slug: str, name: str):
    slug = _safe_slug(slug)
    if "/" in name or "\\" in name or ".." in name:
        raise HTTPException(404)
    p = STUDIO / slug / name
    if not p.is_file():
        raise HTTPException(404)
    return FileResponse(p)


@router.get("/api/studio/clips/{slug}")
async def api_clips(slug: str):
    slug = _safe_slug(slug)
    out = []
    for suffix in ("gen0", "final"):
        p = OUT / "clips" / f"{slug}_{suffix}.mp4"
        if p.is_file():
            out.append({"name": suffix, "url": f"/clips/{p.name}"})
    return {"clips": out}
