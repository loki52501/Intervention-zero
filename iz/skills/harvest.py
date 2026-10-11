"""Harvest public how-to TEXT (captions/transcripts/descriptions) via Bright Data / Firecrawl / cache.

Records: {id, platform, url, title, text(<=1500), situation_type, source, fetched_at}.
Never invents sources; with no keys and an empty cache, returns [].
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

CACHE_PATH = Path(__file__).resolve().parents[2] / "data" / "skill_sources.json"
MAX_TEXT = 1500

QUERIES = [
    "how to push a heavy cart safely",
    "fire door safety never prop open",
    "wet floor spill safety walking",
    "elevator out of service what to do",
    "hospital corridor yield to patients",
    "construction zone detour safety",
    "someone collapsed what to do call for help",
]

KEYWORDS = {
    "fire_door": ["fire door", "prop open", "propped", "fire safety"],
    "office_door": ["office door", "badge", "access door"],
    "cart_blocking": ["cart", "trolley", "heavy load"],
    "crowd_blocking": ["crowd", "yield", "corridor", "pedestrian", "patients"],
    "spill": ["spill", "wet floor", "wet surface", "slip", "slippery"],
    "restricted_shortcut": ["restricted", "shortcut", "no entry", "authorized only"],
    "elevator_out_of_service": ["elevator", "lift", "out of service"],
    "person_down": ["collapsed", "person down", "unconscious", "call for help", "fallen", "cpr"],
    "low_battery": ["low battery", "battery", "charging"],
    "construction_detour": ["construction", "detour", "roadwork", "work zone"],
}


def _load_env() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def match_situation(text: str) -> str:
    t = (text or "").lower()
    best, score = "general", 0
    for typ, kws in KEYWORDS.items():
        s = sum(1 for k in kws if k in t)
        if s > score:
            best, score = typ, s
    return best


def platform_from_url(url: str) -> str:
    host = (urlparse(url or "").netloc or "").lower()
    if "youtube.com" in host or "youtu.be" in host:
        return "youtube"
    if "tiktok.com" in host:
        return "tiktok"
    if "instagram.com" in host:
        return "instagram"
    return "web"


def _record(platform, url, title, text, source, now) -> dict:
    text = (text or "").strip()[:MAX_TEXT]
    return {
        "id": hashlib.sha1(url.encode()).hexdigest()[:12], "platform": platform, "url": url,
        "title": (title or "").strip()[:200], "text": text,
        "situation_type": match_situation(f"{title} {urlparse(url).path.replace('-', ' ').replace('_', ' ')} {text}"),
        "source": source, "fetched_at": now or now_iso(),
    }


def normalize_brightdata(raw: dict, platform: str, now: str | None = None) -> dict | None:
    url = raw.get("url") or raw.get("video_url") or raw.get("post_url") or ""
    if not url:
        return None
    title = raw.get("title") or raw.get("post_title") or ""
    parts = [raw.get("transcript") or raw.get("formatted_transcript"),
             raw.get("description") or raw.get("caption") or raw.get("post_content")]
    text = " ".join(p if isinstance(p, str) else json.dumps(p) for p in parts if p)
    if not (title or text):
        return None
    return _record(platform, url, title, text, "brightdata", now)


def normalize_firecrawl(raw: dict, now: str | None = None) -> dict | None:
    meta = raw.get("metadata") or {}
    url = raw.get("url") or meta.get("sourceURL") or ""
    if not url:
        return None
    title = raw.get("title") or meta.get("title") or ""
    text = raw.get("markdown") or raw.get("description") or ""
    if not (title or text):
        return None
    return _record(platform_from_url(url), url, title, text, "firecrawl", now)


def dedup(recs: list[dict]) -> list[dict]:
    seen, out = set(), []
    for r in recs:
        k = r.get("url")
        if k and k not in seen:
            seen.add(k)
            out.append(r)
    return out


def load_cache(path=CACHE_PATH, as_cache: bool = False) -> list[dict]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        recs = data["records"] if isinstance(data, dict) else data
    except Exception:
        return []
    recs = dedup(recs)
    if as_cache:
        recs = [{**r, "source": "cache"} for r in recs]
    return recs


def save_cache(recs: list[dict], path=CACHE_PATH) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"records": dedup(recs)}, indent=1, ensure_ascii=False), encoding="utf-8")


def _is_seed(r: dict) -> bool:
    return ".invalid" in (r.get("url") or "")


# ---------------------------------------------------------------- network
def fetch_firecrawl(key: str, deadline: float, queries=QUERIES) -> list[dict]:
    import requests
    out = []
    for q in queries:
        left = deadline - time.time()
        if left < 3:
            break
        try:
            r = requests.post("https://api.firecrawl.dev/v1/search", timeout=min(30, left),
                              headers={"Authorization": f"Bearer {key}"},
                              json={"query": q, "limit": 3, "scrapeOptions": {"formats": ["markdown"]}})
            r.raise_for_status()
            for it in r.json().get("data") or []:
                rec = normalize_firecrawl(it)
                if rec and rec["situation_type"] == "general":
                    rec["situation_type"] = match_situation(q)   # fall back to the query's intent
                if rec:
                    out.append(rec)
        except Exception:
            continue
    return out


def _bd_pick_datasets(datasets: list[dict]) -> dict[str, str]:
    """Pick by name. YouTube 'Videos posts' (keyword discovery), TikTok 'posts by keyword' (search url)."""
    want = {"youtube": ("youtube", "videos posts"), "tiktok": ("tiktok", "posts by keyword")}
    picks = {}
    for plat, (a, b) in want.items():
        for d in datasets:
            name = (d.get("name") or "").lower()
            if a in name and b in name and "internal" not in name and "test" not in name:
                picks[plat] = d.get("id")
                break
    return picks


def _bd_trigger(requests, base, H, plat, did, queries):
    from urllib.parse import quote
    params = {"dataset_id": did, "include_errors": "true", "limit_per_input": 2}
    if plat == "youtube":
        params.update({"type": "discover_new", "discover_by": "keyword"})
        body = [{"keyword": q} for q in queries[:4]]
    else:
        body = [{"url": "https://www.tiktok.com/search?q=" + quote(q)} for q in queries[:3]]
    return requests.post(f"{base}/v3/trigger", headers=H, timeout=20, params=params, json=body).json().get("snapshot_id")


def fetch_brightdata(key: str, deadline: float, queries=QUERIES) -> list[dict]:
    import requests
    H = {"Authorization": f"Bearer {key}"}
    base = "https://api.brightdata.com/datasets"
    out = []
    try:
        ds = requests.get(f"{base}/list", headers=H, timeout=15).json()
        picks = _bd_pick_datasets(ds if isinstance(ds, list) else [])
    except Exception:
        return out
    pending = []
    for plat, did in picks.items():
        try:
            sid = _bd_trigger(requests, base, H, plat, did, queries)
            if sid:
                pending.append((plat, sid))
        except Exception:
            continue
    while pending and time.time() < deadline - 5:
        for plat, sid in list(pending):
            try:
                st = requests.get(f"{base}/v3/progress/{sid}", headers=H, timeout=15).json().get("status")
                if st == "ready":
                    rows = requests.get(f"{base}/v3/snapshot/{sid}", headers=H, timeout=30,
                                        params={"format": "json"}).json()
                    for row in rows if isinstance(rows, list) else []:
                        rec = normalize_brightdata(row, plat)
                        if rec:
                            out.append(rec)
                    pending.remove((plat, sid))
                elif st == "failed":
                    pending.remove((plat, sid))
            except Exception:
                pass
        if pending:
            time.sleep(4)
    return out


def harvest(cache_path=CACHE_PATH, time_cap: float = 90.0) -> list[dict]:
    """Bright Data -> Firecrawl -> cache. Merges into cache and returns all records."""
    _load_env()
    t0 = time.time()
    bd, fc = os.getenv("BRIGHTDATA_API_KEY"), os.getenv("FIRECRAWL_API_KEY")
    new: list[dict] = []
    if bd:
        new += fetch_brightdata(bd, t0 + (0.55 * time_cap if fc else time_cap))
    if fc:
        new += fetch_firecrawl(fc, t0 + time_cap)
    if not new:
        return load_cache(cache_path, as_cache=True)
    old = [r for r in load_cache(cache_path) if not _is_seed(r)]
    merged = dedup(new + old)
    save_cache(merged, cache_path)
    return merged
