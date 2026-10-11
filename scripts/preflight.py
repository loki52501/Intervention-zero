"""Preflight: toolchain, packages, env keys (presence only, never values), live service checks.

Run: .venv\\Scripts\\python.exe scripts\\preflight.py
"""
from __future__ import annotations

import importlib
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

rows: list[tuple[str, str, str]] = []


def check(name: str, fn) -> None:
    t = time.time()
    try:
        detail = fn() or ""
        rows.append(("OK  ", name, f"{detail} ({time.time() - t:.1f}s)"))
    except Exception as e:  # noqa: BLE001 - preflight reports every failure
        msg = str(e).splitlines()[0][:140] if str(e) else type(e).__name__
        rows.append(("FAIL", name, msg))


def skip(name: str, why: str) -> None:
    rows.append(("SKIP", name, why))


def has(key: str) -> bool:
    v = os.getenv(key, "")
    return bool(v) and not v.startswith("<")


# ---------- toolchain
check("python 3.13", lambda: f"{sys.version.split()[0]}" if sys.version_info[:2] == (3, 13) else (_ for _ in ()).throw(RuntimeError(sys.version)))
for tool in ("git", "node", "npx", "docker"):
    check(f"{tool} on PATH", lambda tool=tool: shutil.which(tool) or (_ for _ in ()).throw(RuntimeError("not found")))
check("docker daemon", lambda: subprocess.run(["docker", "version", "--format", "{{.Server.Version}}"],
                                              capture_output=True, text=True, timeout=15, check=True).stdout.strip())
check(".env gitignored", lambda: "yes" if subprocess.run(["git", "check-ignore", ".env"], cwd=ROOT,
                                                          capture_output=True).returncode == 0 else (_ for _ in ()).throw(RuntimeError(".env NOT ignored")))

# ---------- packages
for mod in ("mujoco", "numpy", "imageio", "pymongo", "voyageai", "fastapi", "uvicorn", "sse_starlette",
            "pydantic", "pytest", "yaml", "raindrop.analytics", "raindrop_query", "anthropic", "openai"):
    check(f"import {mod}", lambda mod=mod: getattr(importlib.import_module(mod), "__version__", "ok"))

# ---------- env keys (presence only)
for key in ("MONGODB_URI", "VOYAGE_API_KEY", "RAINDROP_WRITE_KEY", "RAINDROP_QUERY_API_KEY",
            "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "ROBOT_MODEL", "FIXER_MODEL"):
    rows.append(("OK  " if has(key) else "MISS", f"env {key}", "set" if has(key) else "empty"))


# ---------- MuJoCo physics + render
def mujoco_check() -> str:
    import mujoco
    m = mujoco.MjModel.from_xml_string(
        "<mujoco><worldbody><light pos='0 0 3'/><geom type='plane' size='1 1 .1'/>"
        "<body pos='0 0 .5'><freejoint/><geom type='box' size='.1 .1 .1'/></body></worldbody></mujoco>")
    d = mujoco.MjData(m)
    for _ in range(500):
        mujoco.mj_step(m, d)
    z = float(d.qpos[2])
    try:
        r = mujoco.Renderer(m, 120, 160)
        r.update_scene(d)
        r.render()
        rend = "render ok"
    except Exception as e:  # noqa: BLE001
        rend = f"render FAILED ({type(e).__name__})"
    return f"mujoco {mujoco.__version__}, box settled z={z:.2f}, {rend}"


check("mujoco physics", mujoco_check)


# ---------- live services
def mongo_check() -> str:
    from pymongo import MongoClient
    c = MongoClient(os.environ["MONGODB_URI"], serverSelectionTimeoutMS=8000)
    info = c.admin.command("buildInfo")
    return f"server {info['version']}"


def voyage_check() -> str:
    import voyageai
    vo = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])
    e = vo.embed(["fire door D-17"], model=os.getenv("VOYAGE_EMBED_MODEL", "voyage-4-large"), input_type="document")
    return f"embed dim {len(e.embeddings[0])}"


def raindrop_check() -> str:
    import raindrop.analytics as rd
    rd.init(os.environ["RAINDROP_WRITE_KEY"])
    if os.getenv("PREFLIGHT_SEND_RAINDROP") == "1":
        i = rd.begin(user_id="preflight", event="preflight", input="preflight ping")
        i.finish(output="pong")
        rd.flush()
        return "init ok, 1 event sent"
    return "init ok (no event sent; set PREFLIGHT_SEND_RAINDROP=1 to send one)"


check("mongodb atlas", mongo_check) if has("MONGODB_URI") else skip("mongodb atlas", "MONGODB_URI empty")
check("voyage embed", voyage_check) if has("VOYAGE_API_KEY") else skip("voyage embed", "VOYAGE_API_KEY empty")
check("raindrop sdk", raindrop_check) if has("RAINDROP_WRITE_KEY") else skip("raindrop sdk", "key empty")

width = max(len(r[1]) for r in rows)
for status, name, detail in rows:
    print(f"[{status}] {name.ljust(width)}  {detail}")
fails = sum(r[0] == "FAIL" for r in rows)
print(f"\n{fails} fail, {sum(r[0] == 'MISS' for r in rows)} missing keys, {sum(r[0] == 'SKIP' for r in rows)} skipped")
sys.exit(1 if fails else 0)
