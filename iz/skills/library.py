"""Skill source library: local JSON (+ Atlas 'skills' collection when reachable)."""
from __future__ import annotations

import json
import re
from pathlib import Path

from iz.types import RobotView

SOURCES_PATH = Path(__file__).resolve().parents[2] / "data" / "skill_sources.json"


def count_tokens(text: str) -> int:
    return len(text) // 4 if text else 0


def _atlas_sources() -> list[dict]:
    try:
        from iz.store.db import get_db, mongo_available
        if not mongo_available():
            return []
        return [{k: v for k, v in d.items() if k not in ("_id", "embedding")}
                for d in get_db()["skills"].find({}).limit(200)]
    except Exception:  # noqa: BLE001
        return []


def load_sources(path: str | Path | None = None, use_atlas: bool = True) -> list[dict]:
    """Source records: {platform,url,title,text,situation_type,source}. [] when none."""
    out: list[dict] = []
    p = Path(path) if path else SOURCES_PATH
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, dict):          # harvest cache format: {"records": [...]}
            data = data.get("records") or []
        out = [d for d in data if isinstance(d, dict) and d.get("text")]
    except Exception:  # noqa: BLE001
        out = []
    if use_atlas and path is None:
        seen = {d.get("url") for d in out}
        out += [d for d in _atlas_sources() if d.get("url") not in seen and d.get("text")]
    return out


def _words(s: str) -> set[str]:
    return set(re.findall(r"[a-z0-9_]+", (s or "").lower()))


def top_skills(view: RobotView, k: int = 2, sources: list[dict] | None = None) -> list[dict]:
    """Minimal-context retrieval: sources for the view's situation type, ranked by text overlap."""
    srcs = load_sources() if sources is None else sources
    q = _words(view.observable) | _words(view.type)
    scored = [(len(q & _words(s.get("text", "") + " " + s.get("title", ""))), s)
              for s in srcs if s.get("situation_type") == view.type]
    scored.sort(key=lambda t: -t[0])
    return [s for _, s in scored[:max(0, k)]]
