"""One prompt -> a generated simulation world -> robot learns in it.

Usage:
  python scripts/prompt_world.py "a pharmacy with a wet floor near aisle 2, a heavy cart and rush-hour shoppers"
         [--generations 3] [--fake-llm] [--live] [--no-learn] [--out out]

Pipeline:
  1. prompt_to_spec   LLM (or keyword fallback) -> validated/repaired world spec
  2. make_variants    8 train + 8 held-out + 4 safety (dense shoppers) variants of that spec
  3. build_world / check_world   MuJoCo store (iz.shop.scene robot, cart, sensors) per variant;
                      invalid worlds are dropped; preview.png is rendered
  4. learning         iz.shop.learn.propose + gate (the real fixer + the real promotion gate),
                      evaluated on the GENERATED worlds (not the stock shop seeds)
  5. out/worldgen/<slug>/history.json and UI events in out/events.jsonl (world: "generated")
  6. --live           plays a held-out world in the MuJoCo viewer: gen0 params, then final params

Honest claim: the AI model writes the world spec and the controller-parameter edits; the robot's
low-level motion is the sensor-driven controller. Nothing is learned from scratch.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
import threading
import time
import uuid
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

N_TRAIN, N_HELD, N_SAFETY = 8, 8, 4
KEY_BASE = {"train": 0, "heldout": 100, "safety": 200}

SPEC_BOUNDS = {
    "n_spills": (0, 3),
    "cart_mass_kg": (5.0, 60.0),
    "n_shoppers": (0, 6),
}

SPEC_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "store_type": {"type": "string"},
        "aisle": {"type": ["integer", "null"], "description": "1-based aisle (1..3) the pick happens in, or null"},
        "n_spills": {"type": "integer", "description": "wet-floor patches 0..3"},
        "spill_near_aisle": {"type": "boolean"},
        "cart_mass_kg": {"type": "number", "description": "loaded cart mass 5..60 kg"},
        "n_shoppers": {"type": "integer", "description": "walking shoppers 0..6"},
        "product": {"type": "string"},
        "summary": {"type": "string"},
    },
    "required": ["name", "store_type", "n_spills", "cart_mass_kg", "n_shoppers"],
}

SPEC_SYSTEM = (
    "You turn a one-line description of a retail environment into a JSON world spec for a MuJoCo "
    "store simulation (3 aisles, a wheeled humanoid robot pushing a shopping cart, a checkout). "
    "Fields: name, store_type, aisle (1-based 1..3 or null), n_spills (0..3 wet-floor patches), "
    "spill_near_aisle (bool), cart_mass_kg (5..60; 'heavy' ~45, 'light' ~10, default 25), "
    "n_shoppers (0..6; 'rush hour'/'crowded' ~5, 'quiet' ~1, default 2), product, summary. "
    "Stay inside the ranges. Return JSON only."
)


# ====================================================================== 1. prompt -> spec
def slugify(text: str, n: int = 40) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return (s[:n].strip("-") or "world")


def keyword_spec(prompt: str) -> dict:
    """Deterministic prompt parser (the --fake-llm brain and the real-LLM fallback)."""
    p = prompt.lower()
    kinds = ["pharmacy", "grocery", "supermarket", "hardware", "bookstore", "electronics",
             "convenience", "bakery", "clothing", "warehouse", "store", "shop"]
    kind = next((k for k in kinds if k in p), "store")
    m = re.search(r"aisle\s*(?:no\.?\s*)?(\d)", p)
    aisle = int(m.group(1)) if m else None
    wet = any(w in p for w in ("wet", "spill", "slippery", "puddle", "slick"))
    nsp = 0
    if wet and not re.search(r"\b(no|without|dry)\b[^,]*\b(spill|wet|puddle)", p):
        nsp = 3 if any(w in p for w in ("many", "several", "puddles", "spills")) else 1
        if re.search(r"\b(two|2)\s+(\w+\s+)?(spills|puddles|wet)", p):  # not "two narrow aisles"
            nsp = 2
    mass = 25.0
    mkg = re.search(r"(\d+(?:\.\d+)?)\s*(?:kg|kilo)", p)
    if mkg:  # explicit mass is clamped to the safe envelope by repair_spec
        mass = float(mkg.group(1))
    elif any(w in p for w in ("heavy", "loaded", "full cart", "overloaded")):
        mass = 45.0
    elif any(w in p for w in ("light", "empty cart")):
        mass = 10.0
    ns = 2
    if any(w in p for w in ("rush", "crowd", "busy", "packed", "dense")):
        ns = 5
    elif any(w in p for w in ("quiet", "empty store", "no shoppers", "deserted")):
        ns = 0 if "no shoppers" in p else 1
    return {"name": f"{kind} run", "store_type": kind, "aisle": aisle, "n_spills": nsp,
            "spill_near_aisle": bool(aisle and nsp), "cart_mass_kg": mass, "n_shoppers": ns,
            "product": "item", "summary": prompt.strip()}


def repair_spec(raw: dict) -> tuple[dict, list[str]]:
    """Validate/clamp an LLM spec. Returns (spec, repairs) - every change is reported."""
    raw = dict(raw or {})
    repairs: list[str] = []
    spec: dict = {}
    spec["name"] = str(raw.get("name") or "generated world")[:60]
    spec["store_type"] = str(raw.get("store_type") or "store")[:30]
    spec["product"] = str(raw.get("product") or "item")[:30]
    spec["summary"] = str(raw.get("summary") or "")[:300]
    for k, (lo, hi) in SPEC_BOUNDS.items():
        v = raw.get(k)
        try:
            v = float(v)
            if not math.isfinite(v):
                raise ValueError
        except (TypeError, ValueError):
            default = {"n_spills": 0, "cart_mass_kg": 25.0, "n_shoppers": 2}[k]
            repairs.append(f"{k}: missing/invalid ({raw.get(k)!r}) -> {default}")
            v = default
        c = min(max(v, lo), hi)
        if c != v:
            repairs.append(f"{k}: {v:g} out of range [{lo:g},{hi:g}] -> {c:g}")
        spec[k] = c if k == "cart_mass_kg" else int(round(c))
    a = raw.get("aisle")
    if a is None:
        spec["aisle"] = None
    else:
        try:
            ai = int(a)
        except (TypeError, ValueError):
            ai = None
            repairs.append(f"aisle: invalid ({a!r}) -> any")
        if ai is not None and not 1 <= ai <= 3:
            repairs.append(f"aisle: {ai} not in 1..3 -> clamped")
            ai = min(max(ai, 1), 3)
        spec["aisle"] = ai
    spec["spill_near_aisle"] = bool(raw.get("spill_near_aisle", False)) and spec["aisle"] is not None \
        and spec["n_spills"] > 0
    if bool(raw.get("spill_near_aisle", False)) and not spec["spill_near_aisle"]:
        repairs.append("spill_near_aisle dropped (needs an aisle and n_spills > 0)")
    return spec, repairs


def prompt_to_spec(prompt: str, llm=None) -> tuple[dict, list[str]]:
    repairs: list[str] = []
    raw = None
    if llm is not None:
        try:
            from iz.llm.client import fixer_model
            resp = llm.complete_json(model=fixer_model(), system=SPEC_SYSTEM,
                                     user=f"PROMPT: {prompt}", schema=SPEC_SCHEMA)
            raw = getattr(resp, "data", resp)
            if not isinstance(raw, dict) or not raw:
                raw = None
                repairs.append("LLM returned no usable spec -> keyword parser used")
        except Exception as e:  # noqa: BLE001
            repairs.append(f"LLM spec call failed ({type(e).__name__}) -> keyword parser used")
    if raw is None:
        raw = keyword_spec(prompt)
    spec, rep = repair_spec(raw)
    spec["prompt"] = prompt
    return spec, repairs + rep


def fake_spec_llm():
    """FakeLLM whose 'model' is the keyword parser (offline, deterministic)."""
    from iz.llm.client import FakeLLM

    def policy(model, system, user, schema):
        return keyword_spec(user.split("PROMPT: ", 1)[-1])
    return FakeLLM(policy=policy)


# ====================================================================== 2. variants
_REAL_BUILD_SHOP = None


class _SceneProxy:
    """Always the ORIGINAL iz.shop.scene.build_shop, even while an episode patches the module."""

    @staticmethod
    def build_shop(*a, **k):
        global _REAL_BUILD_SHOP
        if _REAL_BUILD_SHOP is None:
            from iz.shop import scene
            _REAL_BUILD_SHOP = scene.build_shop
        return _REAL_BUILD_SHOP(*a, **k)


def _scene():
    return _SceneProxy


def _find_seeds(base: int, count: int, aisle0: int | None) -> list[int]:
    """Scene seeds >= base whose target aisle matches the spec (0-based aisle0, or any)."""
    out, s = [], base
    sc = _scene()
    while len(out) < count and s < base + 400:
        if aisle0 is None:
            out.append(s)
        else:
            rng = random.Random(int(s) * 7919 + 13)
            if rng.randrange(3) == aisle0:   # same first draw build_shop uses for the aisle
                out.append(s)
        s += 1
    del sc
    return out


def make_variants(spec: dict, n_train=N_TRAIN, n_held=N_HELD, n_safety=N_SAFETY) -> list[dict]:
    aisle0 = None if spec.get("aisle") is None else int(spec["aisle"]) - 1
    rng = random.Random(zlib_crc(spec.get("prompt", "") + spec.get("name", "")))
    out = []
    plan = [("train", n_train, 0.2, 1), ("heldout", n_held, 0.35, 1), ("safety", n_safety, 0.15, 3)]
    for split, n, mass_jit, extra_sh in plan:
        for i, seed in enumerate(_find_seeds(KEY_BASE[split], n, aisle0)):
            mass = spec["cart_mass_kg"] * (1 + rng.uniform(-mass_jit, mass_jit))
            nsh = spec["n_shoppers"] + rng.choice([-1, 0, 1]) * (1 if split != "safety" else 0)
            if split == "safety":
                nsh = max(4, spec["n_shoppers"] + extra_sh)
            nsh = int(min(max(nsh, 0), 8))
            out.append({"id": f"{split}-{i}", "key": KEY_BASE[split] + i, "split": split,
                        "scene_seed": seed, "cart_mass_kg": round(min(max(mass, 5.0), 60.0), 1),
                        "n_shoppers": nsh, "n_spills": spec["n_spills"],
                        "spill_seed": rng.randrange(10 ** 6)})
    return out


def zlib_crc(s: str) -> int:
    import zlib
    return zlib.crc32(s.encode("utf-8"))


# ====================================================================== 3. build / check
_SPILL_RE = re.compile(r'^    <geom name="spill_\d+".*?/>\n', re.M)


def _spill_positions(meta: dict, spec: dict, v: dict) -> list[list[float]]:
    rng = random.Random(v["spill_seed"])
    wps, route = meta["waypoints"], meta["route"]
    segs = [(wps[i], wps[j]) for i, j in zip(route[:-1], route[1:])]
    out: list[list[float]] = []
    pick = meta["pick_xy"]
    tries = 0
    while len(out) < v["n_spills"] and tries < 60:
        tries += 1
        if spec.get("spill_near_aisle") and not out:
            a, b = segs[1]                      # aisle entry -> pick point (the named aisle)
        else:
            a, b = rng.choice(segs)
        t = rng.uniform(0.25, 0.75)
        p = [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])]
        if math.hypot(p[0] - pick[0], p[1] - pick[1]) < 1.2 or p[0] < -4.4:
            continue
        if any(math.hypot(p[0] - q[0], p[1] - q[1]) < 1.2 for q in out):
            continue
        out.append([round(p[0], 3), round(p[1], 3), 0.55, 0.40])
    return out


def build_world(v: dict, spec: dict) -> tuple[str, dict]:
    """Variant -> (xml, meta) with the shared robot/cart/sensor contract of iz.shop.scene."""
    xml, meta = _scene().build_shop(v["scene_seed"], cart_mass_kg=v["cart_mass_kg"],
                                    n_shoppers=v["n_shoppers"])
    spills = _spill_positions(meta, spec, v)
    xml = _SPILL_RE.sub("", xml)
    lines = "".join(
        f'    <geom name="spill_{i}" type="box" pos="{sx:.4f} {sy:.4f} 0.0010" size="{hx:.4f} {hy:.4f} 0.0010" '
        f'friction="0.04 0.005 0.0001" priority="2" rgba="0.25 0.55 0.95 0.55" contype="1" conaffinity="2"/>\n'
        for i, (sx, sy, hx, hy) in enumerate(spills))
    marker = '    <body name="prod_0"'
    if marker not in xml:
        raise ValueError("scene layout changed: prod_0 marker not found")
    xml = xml.replace(marker, lines + marker, 1)
    meta["spills"] = spills
    meta["world_id"] = v["id"]
    meta["world_prompt"] = spec.get("prompt", "")
    return xml, meta


def check_world(xml: str, meta: dict) -> list[str]:
    """Return a list of problems ([] == valid). Compiles, checks the contract, probes stability."""
    import mujoco
    probs: list[str] = []
    try:
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
    except Exception as e:  # noqa: BLE001
        return [f"compile: {str(e)[:120]}"]
    O = mujoco.mjtObj
    need = [(O.mjOBJ_BODY, n) for n in ("robot", "cart", meta["target_product"])]
    need += [(O.mjOBJ_GEOM, n) for n in ("checkout", "shelf_0", "floor")]
    need += [(O.mjOBJ_ACTUATOR, n) for n in ("act_vx", "act_vy", "act_wz", "act_r_shoulder", "act_r_elbow")]
    need += [(O.mjOBJ_SENSOR, n) for n in ("rf_front", "rf_cart", "cart_up", "cart_pos", "f_wrist")]
    need += [(O.mjOBJ_EQUALITY, n) for n in ("grip_cart", "grip_prod")]
    need += [(O.mjOBJ_SITE, n) for n in ("r_hand", "cart_handle")]
    for t, n in need:
        if mujoco.mj_name2id(model, t, n) < 0:
            probs.append(f"missing {n}")
    for k in ("waypoints", "route", "checkout_xy", "cart_xy", "robot_xy", "shopper_paths", "spills"):
        if k not in meta:
            probs.append(f"meta missing {k}")
    if probs:
        return probs
    wps = meta["waypoints"]
    if any(i < 0 or i >= len(wps) for i in meta["route"]) or len(meta["route"]) < 3:
        probs.append("bad route")
    hx, hy = meta.get("store_half", [6.1, 4.1])
    for w in wps:
        if abs(w[0]) > hx or abs(w[1]) > hy:
            probs.append(f"waypoint outside store {w}")
            break
    for s in meta["spills"]:
        for nm, q in (("robot", meta["robot_xy"]), ("cart", meta["cart_xy"])):
            if math.hypot(s[0] - q[0], s[1] - q[1]) < 0.9:
                probs.append(f"spill on {nm} start")
        if math.hypot(s[0] - meta["checkout_xy"][0], s[1] - meta["checkout_xy"][1]) < 0.9:
            probs.append("spill on checkout")
    for i, path in enumerate(meta["shopper_paths"]):
        for nm, q in (("robot", meta["robot_xy"]), ("cart", meta["cart_xy"])):
            if math.hypot(path[0][0] - q[0], path[0][1] - q[1]) < 0.8:
                probs.append(f"shopper_{i} spawns on {nm} start")
    try:
        for _ in range(250):
            mujoco.mj_step(model, data)
        import numpy as np
        if not np.all(np.isfinite(data.qpos)):
            probs.append("unstable physics (NaN)")
        z = data.xmat[mujoco.mj_name2id(model, O.mjOBJ_BODY, "cart")].reshape(3, 3)[:, 2]
        if z[2] < 0.9:
            probs.append("cart not upright at rest")
    except Exception as e:  # noqa: BLE001
        probs.append(f"step: {str(e)[:100]}")
    return probs


def build_all(spec: dict, variants: list[dict]) -> tuple[list[dict], list[dict]]:
    """Build + check every variant. Returns (valid, dropped)."""
    valid, dropped = [], []
    for v in variants:
        try:
            xml, meta = build_world(v, spec)
            probs = check_world(xml, meta)
        except Exception as e:  # noqa: BLE001
            probs = [f"build: {type(e).__name__}: {str(e)[:100]}"]
        (dropped if probs else valid).append({**v, "problems": probs} if probs else v)
    return valid, dropped


def render_preview(spec: dict, variants: list[dict], path: Path) -> bool:
    """2x2 montage of up to 4 worlds (train/held-out/safety) from the store overview camera."""
    import mujoco
    import numpy as np
    pick = []
    for split in ("train", "heldout", "heldout", "safety"):
        cands = [v for v in variants if v["split"] == split and v not in pick]
        if cands:
            pick.append(cands[0])
    tiles = []
    for v in pick[:4]:
        xml, meta = build_world(v, spec)
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        r = mujoco.Renderer(model, 360, 480)
        try:
            r.update_scene(data, camera="cam_34")
            tiles.append(r.render().copy())
        finally:
            r.close()
    if not tiles:
        return False
    while len(tiles) < 4:
        tiles.append(np.zeros_like(tiles[0]))
    img = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:4])])
    import imageio
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(str(path), img)
    return True


# ====================================================================== 4. learning on generated worlds
def _world_worker(args: tuple) -> dict:
    """Process-pool worker: run one episode in a generated world (patches scene.build_shop)."""
    params, variant, spec, key = args
    import iz.shop.scene as sc
    from iz.shop.episode import run_episode
    _SceneProxy.build_shop(0)          # capture the original before patching
    sc.build_shop = lambda seed, *a, **k: build_world(variant, spec)
    r = run_episode(params, key)
    r["world_id"] = variant["id"]
    return r


def _importable_worker():
    """The worker as seen under its package path, so it pickles even when run as a script."""
    import importlib
    return importlib.import_module("scripts.prompt_world")._world_worker


def make_evaluator(registry: dict, spec: dict, workers: int = 4):
    def evaluate(params: dict, seeds) -> list[dict]:
        seeds = list(seeds)
        if not seeds:
            return []
        jobs = [(params, registry[s], spec, s) for s in seeds]
        if len(jobs) == 1:
            return [_world_worker(jobs[0])]
        with ProcessPoolExecutor(workers) as ex:
            return list(ex.map(_importable_worker(), jobs))
    return evaluate


class GenEvents:
    """Event writer for the UI contract: world == "generated" plus the prompt on every event."""

    def __init__(self, run_id: str, path: Path, prompt: str = "", slug: str = ""):
        self.run_id, self.prompt, self.slug = run_id, prompt, slug
        self.path = Path(path)
        self._seq = 0
        self._prefix = uuid.uuid4().hex[:8]
        self._lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, kind: str, payload: dict) -> dict:
        payload = {**payload, "world": "generated", "prompt": self.prompt, "slug": self.slug}
        with self._lock:
            self._seq += 1
            ev = {"id": f"{self._prefix}-{self._seq:06d}",
                  "ts": datetime.now(timezone.utc).isoformat(),
                  "kind": kind, "run_id": self.run_id, "payload": payload}
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(ev, default=str) + "\n")
        return ev


def run_generated_learning(spec: dict, valid: list[dict], generations: int, llm, out_dir: Path,
                           events_path: Path, slug: str, run_id: str, on_generation=None) -> list[dict]:
    from iz.shop import learn
    registry = {v["key"]: v for v in valid}
    tr = [v["key"] for v in valid if v["split"] == "train"]
    ho = [v["key"] for v in valid if v["split"] == "heldout"]
    sf = [v["key"] for v in valid if v["split"] == "safety"]
    if not tr or not ho:
        raise RuntimeError(f"not enough valid worlds to learn (train={len(tr)}, held-out={len(ho)})")
    old_eval, old_ev = learn.evaluate, learn._Events
    learn.evaluate = make_evaluator(registry, spec)
    learn._Events = lambda rid, path: GenEvents(rid, events_path, spec["prompt"], slug)
    try:
        return learn.run_learning(generations=generations, train_seeds=tr, heldout_seeds=ho,
                                  safety_seeds=sf, llm=llm, out_dir=str(out_dir), run_id=run_id,
                                  on_generation=on_generation)
    finally:
        learn.evaluate, learn._Events = old_eval, old_ev


# ====================================================================== 6. live viewer
def _text(v, left: str, right: str) -> None:
    try:
        import mujoco
        v.set_texts((mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT, left, right))
    except Exception:  # noqa: BLE001
        pass


def play_live(spec: dict, variant: dict, labelled_params: list[tuple[str, dict]]) -> None:
    import types

    import mujoco
    import mujoco.viewer
    import iz.shop.episode as ep
    import iz.shop.scene as sc
    xml, _ = build_world(variant, spec)       # captures the original build_shop first
    real_bs = sc.build_shop
    sc.build_shop = lambda seed, *a, **k: build_world(variant, spec)
    try:
        for label, params in labelled_params:
            m = mujoco.MjModel.from_xml_string(xml)
            d = mujoco.MjData(m)
            left = f"{label} | world: {spec['name']}"
            print(f"\n=== {left}", flush=True)
            with mujoco.viewer.launch_passive(m, d) as v:
                _text(v, left, "running...")
                v.cam.distance, v.cam.azimuth, v.cam.elevation = 11.0, 135.0, -40.0
                v.cam.lookat[:] = (0.0, 0.0, 0.5)
                shim = types.SimpleNamespace(**{k: getattr(mujoco, k) for k in dir(mujoco) if not k.startswith("__")})
                shim.MjModel = types.SimpleNamespace(from_xml_string=lambda s, _m=m: _m)
                shim.MjData = lambda model, _d=d: _d
                real = ep.mujoco
                ep.mujoco = shim
                try:
                    res = ep.run_episode(params, variant["key"], viewer=v)
                finally:
                    ep.mujoco = real
                _text(v, left, f"result: {res['status']} {res.get('failure') or ''}")
                print(f"    -> {res['status']} {res.get('failure')}  {res.get('operator_tip') or ''}", flush=True)
                t0 = time.time()
                while v.is_running() and time.time() - t0 < 2.0:
                    v.sync()
                    time.sleep(0.03)
    finally:
        sc.build_shop = real_bs


# ====================================================================== main
def run(prompt: str, generations: int = 3, fake_llm: bool = False, live: bool = False,
        learn_enabled: bool = True, out: str = "out", say=print) -> dict:
    out_root = Path(out)
    slug = slugify(prompt)
    wdir = out_root / "worldgen" / slug
    wdir.mkdir(parents=True, exist_ok=True)
    events_path = out_root / "events.jsonl"
    run_id = "gen-" + uuid.uuid4().hex[:8]

    # 1. spec
    llm = None
    if fake_llm:
        llm = fake_spec_llm()
    else:
        try:
            from iz.llm.client import make_llm
            llm = make_llm()
        except Exception as e:  # noqa: BLE001
            say(f"[spec] no LLM available ({type(e).__name__}); using keyword parser")
    spec, repairs = prompt_to_spec(prompt, llm)
    say(f"[1] SPEC  {spec['name']!r} type={spec['store_type']} aisle={spec['aisle']} spills={spec['n_spills']} "
        f"(near aisle: {spec['spill_near_aisle']}) cart={spec['cart_mass_kg']:g} kg shoppers={spec['n_shoppers']}")
    say("    repairs: " + ("; ".join(repairs) if repairs else "none"))

    # 2-3. variants, build, check
    variants = make_variants(spec)
    valid, dropped = build_all(spec, variants)
    counts = {s: sum(1 for v in valid if v["split"] == s) for s in ("train", "heldout", "safety")}
    say(f"[2] VARIANTS {len(variants)} generated -> {len(valid)} valid, {len(dropped)} dropped  "
        f"(train {counts['train']}/{N_TRAIN}, held-out {counts['heldout']}/{N_HELD}, safety {counts['safety']}/{N_SAFETY})")
    for d in dropped:
        say(f"    dropped {d['id']}: {'; '.join(d['problems'])}")
    preview = wdir / "preview.png"
    ok = render_preview(spec, valid, preview) if valid else False
    say(f"[3] PREVIEW {preview if ok else 'not rendered'}")

    gen_ev = GenEvents(run_id, events_path, prompt, slug)
    gen_ev.emit("world_generated", {"spec": spec, "repairs": repairs, "counts": counts,
                                    "dropped": len(dropped), "preview": str(preview) if ok else None})

    # 4. learning
    history: list[dict] = []
    if learn_enabled:
        if llm is None or fake_llm:
            from iz.shop import learn
            fixer = learn.fake_llm() if fake_llm else None
        else:
            fixer = llm
        if fixer is None:
            from iz.llm.client import make_llm
            fixer = make_llm()
        say(f"[4] LEARNING {generations} generation(s) on generated worlds ...")
        history = run_generated_learning(
            spec, valid, generations, fixer, wdir, events_path, slug, run_id,
            on_generation=lambda h: say(f"    gen {h['gen']} {h['verdict']}: {h['reason']}"))
        say(f"    {'gen':>3} {'verdict':9} {'train':>6} {'held-out':>9}  edits")
        for h in history:
            say(f"    {h['gen']:>3} {h['verdict']:9} {h['train_rate']*100:5.0f}% "
                f"{h['heldout_success']:>4}/{h['heldout_n']:<4} {h['edits']}")
    else:
        say("[4] LEARNING skipped (--no-learn)")

    # 5. history
    doc = {"prompt": prompt, "slug": slug, "run_id": run_id, "spec": spec, "repairs": repairs,
           "counts": counts, "dropped": dropped, "variants": valid, "history": history,
           "preview": str(preview) if ok else None}
    hist_path = wdir / "history.json"
    hist_path.write_text(json.dumps(doc, indent=2, default=str), encoding="utf-8")
    say(f"[5] WROTE {hist_path} and events -> {events_path}")

    # 6. live
    if live and valid:
        held = next((v for v in valid if v["split"] == "heldout"), valid[0])
        from iz.shop.controller import DEFAULT_PARAMS
        seq = [("Generation 0 (naive params)", history[0]["params"] if history else dict(DEFAULT_PARAMS))]
        if len(history) > 1:
            seq.append((f"Final generation {history[-1]['gen']} (learned params)", history[-1]["params"]))
        play_live(spec, held, seq)
    return doc


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Generate a simulation world from one prompt and let the robot learn in it.")
    ap.add_argument("prompt")
    ap.add_argument("--generations", type=int, default=3)
    ap.add_argument("--fake-llm", action="store_true")
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--no-learn", action="store_true")
    ap.add_argument("--out", default="out")
    a = ap.parse_args(argv)
    try:  # verifier fix: logs were empty when stdout was redirected (block buffering)
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    run(a.prompt, a.generations, a.fake_llm, a.live, not a.no_learn, a.out)


if __name__ == "__main__":
    main()
