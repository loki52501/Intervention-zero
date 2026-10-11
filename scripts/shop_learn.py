"""Headless SHOP learning run. Usage: python scripts/shop_learn.py [--generations 5] [--fake-llm] [--no-clips]"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from iz.world.stats import wilson  # noqa: E402


def render_clip(params: dict, seed: int, path: Path) -> dict:
    import imageio
    from iz.shop.episode import run_episode
    path.parent.mkdir(parents=True, exist_ok=True)
    w = imageio.get_writer(str(path), fps=30, macro_block_size=1)
    try:
        return run_episode(params, seed, recorder=w)
    finally:
        w.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", type=int, default=5)
    ap.add_argument("--fake-llm", action="store_true")
    ap.add_argument("--no-clips", action="store_true")
    ap.add_argument("--clip-seed", type=int, default=100)
    ap.add_argument("--out", default="out")
    a = ap.parse_args()
    from iz.shop import learn
    llm = learn.fake_llm() if a.fake_llm else None
    if llm is None:
        from iz.llm.client import make_llm
        llm = make_llm()
    hist = learn.run_learning(generations=a.generations, llm=llm, out_dir=a.out,
                              on_generation=lambda h: print(
                                  f"  gen {h['gen']} {h['verdict']}: {h['reason']}", flush=True))
    print(f"\n{'gen':>3} {'verdict':9} {'train':>6} {'held-out':>9} {'Wilson 95% CI (held-out)':>26}  edits")
    for h in hist:
        k, n = h["heldout_success"], h["heldout_n"]
        lo, hi = wilson(k, n)
        print(f"{h['gen']:>3} {h['verdict']:9} {h['train_rate']*100:5.0f}% {k:>4}/{n:<4} "
              f"[{lo*100:5.1f}%, {hi*100:5.1f}%]   {h['edits']}")
    if not a.no_clips:
        for h in hist:
            p = Path(a.out) / "clips" / f"shop_gen{h['gen']}.mp4"
            try:
                r = render_clip(h["params"], a.clip_seed, p)
                print(f"clip {p} -> {r['status']} {r.get('failure')}")
            except Exception as e:  # noqa: BLE001
                print(f"clip gen {h['gen']} failed: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
