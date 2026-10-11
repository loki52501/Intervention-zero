"""Render 1280x720 stills of a SHOP held-out episode at 4 moments (start, turning with
cart, at shelf picking, end/failure) for gen0 params and final params.
Usage: python scripts/shop_frames.py [--seed 100]  -> out/shop_frames/*.png
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import imageio.v2 as imageio  # noqa: E402

from iz.shop.controller import DEFAULT_PARAMS  # noqa: E402
from iz.shop.episode import run_episode  # noqa: E402

CAUTIOUS = dict(DEFAULT_PARAMS, cruise_speed=0.7, turn_speed=0.4, turn_slowdown_deg=25,
                brake_distance_m=0.9, yield_distance_m=1.2, spill_speed=0.35)


class Stills:
    width, height, every_s = 1280, 720, 0.25

    def __init__(self):
        self.first = None
        self.push = []
        self.shelf = []
        self.last = None

    def on_frame(self, frame, t, state):
        f = frame.copy()
        if self.first is None:
            self.first = (t, state, f)
        s = state.upper()
        if "PUSH" in s and len(self.push) < 40:
            self.push.append((t, state, f))
        if "SHELF" in s:
            self.shelf.append((t, state, f))
            if len(self.shelf) > 60:
                self.shelf = self.shelf[::2]
        self.last = (t, state, f)

    def moments(self):
        out = [("1_start", self.first)]
        if self.push:
            out.append(("2_turning", self.push[min(len(self.push) - 1, 12)]))
        if self.shelf:
            out.append(("3_shelf", self.shelf[len(self.shelf) // 2]))
        out.append(("4_end", self.last))
        return out


def params_final():
    p = ROOT / "out" / "shop_history.json"
    if p.exists():
        h = json.loads(p.read_text())
        if isinstance(h, dict):
            h = h.get("history", [])
        if h:
            return dict(h[-1]["params"]), "history"
    return CAUTIOUS, "cautious(fallback)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--max-time", type=float, default=70)
    a = ap.parse_args()
    od = ROOT / "out" / "shop_frames"
    od.mkdir(parents=True, exist_ok=True)
    fin, src = params_final()
    for tag, params in [("gen0", dict(DEFAULT_PARAMS)), ("final", fin)]:
        rec = Stills()
        r = run_episode(params, a.seed, max_time_s=a.max_time, recorder=rec)
        print(tag, src if tag == "final" else "", "seed", a.seed, r["status"], r["failure"], r["t_end"],
              json.dumps(r["summary"]), r["operator_tip"])
        for name, m in rec.moments():
            if m is None:
                continue
            t, st, f = m
            fn = od / f"{tag}_s{a.seed}_{name}.png"
            imageio.imwrite(fn, f)
            print("  ", fn.name, f"t={t:.2f}", st)


if __name__ == "__main__":
    main()
