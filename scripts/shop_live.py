"""LIVE SHOW: plays one held-out episode per learned generation in the native MuJoCo window.

Usage: python scripts/shop_live.py [--seed 100] [--live-learn] [--fake-llm] [--generations 5]
One viewer window is opened per generation: the episode builds its own model, which is shared
with the window through a small mujoco shim (episode.run_episode only receives the viewer handle).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mujoco  # noqa: E402
import mujoco.viewer  # noqa: E402


def caption(h: dict) -> tuple[str, str]:
    edits = h.get("edits") or {}
    if h["gen"] == 0:
        changed = "none (naive baseline)"
    elif h["verdict"] != "PROMOTE":
        changed = "none (candidate rejected: " + h["reason"] + ")"
    else:
        changed = ", ".join(f"{k}={v}" for k, v in edits.items()) or "none"
    left = f"Generation {h['gen']} | params changed: {changed}"
    right = f"verdict {h['verdict']} | held-out success {h['heldout_success']}/{h['heldout_n']}"
    return left, right


def _text(v, left: str, right: str) -> None:
    try:
        v.set_texts((mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT, left, right))
    except Exception:  # noqa: BLE001
        pass  # console print is the fallback


def play(h: dict, seed: int) -> dict:
    import iz.shop.episode as ep
    from iz.shop.scene import build_shop
    xml, _ = build_shop(seed)
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    left, right = caption(h)
    print(f"\n=== {left}\n    {right}", flush=True)
    with mujoco.viewer.launch_passive(m, d) as v:
        _text(v, left, right)
        v.cam.distance, v.cam.azimuth, v.cam.elevation = 11.0, 135.0, -40.0
        v.cam.lookat[:] = (0.0, 0.0, 0.5)
        shim = types.SimpleNamespace(**{k: getattr(mujoco, k) for k in dir(mujoco) if not k.startswith("__")})
        shim.MjModel = types.SimpleNamespace(from_xml_string=lambda s: m)
        shim.MjData = lambda model: d
        real = ep.mujoco
        ep.mujoco = shim
        try:
            res = ep.run_episode(h["params"], seed, viewer=v)
        finally:
            ep.mujoco = real
        _text(v, left, f"{right} | this run: {res['status']} {res.get('failure') or ''}")
        print(f"    episode -> {res['status']} {res.get('failure')}  {res.get('operator_tip') or ''}", flush=True)
        t0 = time.time()
        while v.is_running() and time.time() - t0 < 2.0:  # 2 s pause between generations
            v.sync()
            time.sleep(0.03)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--live-learn", action="store_true")
    ap.add_argument("--fake-llm", action="store_true")
    ap.add_argument("--generations", type=int, default=5)
    ap.add_argument("--out", default="out")
    a = ap.parse_args()
    from iz.shop import learn
    path = Path(a.out) / "shop_history.json"
    llm = None
    if a.live_learn or not path.exists():
        llm = learn.fake_llm() if a.fake_llm else None
        if llm is None:
            from iz.llm.client import make_llm
            llm = make_llm()
    if a.live_learn:
        # baseline first, then each generation plays as soon as the gate has decided it
        first = {"done": False}

        def on_gen(h):
            if not first["done"]:
                first["done"] = True
                base = json.loads(path.read_text(encoding="utf-8"))[0]
                play(base, a.seed)
            play(h, a.seed)

        learn.run_learning(generations=a.generations, llm=llm, out_dir=a.out, on_generation=on_gen)
        return
    if not path.exists():
        print("no shop_history.json: running learning first...", flush=True)
        learn.run_learning(generations=a.generations, llm=llm, out_dir=a.out)
    for h in json.loads(path.read_text(encoding="utf-8")):
        play(h, a.seed)


if __name__ == "__main__":
    main()
