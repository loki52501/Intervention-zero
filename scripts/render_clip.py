"""Render before/after MuJoCo clips into out/clips/. Run from repo root: python scripts/render_clip.py"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from iz.types import Action  # noqa: E402

FIRE = {"door_type": "fire", "door_mass_kg": 40.0, "min_push_n": 160.0, "open_delay_s": 9.0}
CART = {"cart_mass_kg": 20.0, "move_force_n": 25.0, "tip_force_n": 60.0, "red_tag": False}


def _report(name, r):
    if r.clip_path:
        print(f"{name}: success={r.success} t={r.time_s}s peak={r.peak_force_n}N reason='{r.reason}' -> {r.clip_path}")
    else:
        print(f"{name}: success={r.success} reason='{r.reason}' render skipped: {r.extra.get('render_error', 'unknown')}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join("out", "clips"))
    ap.add_argument("--site", default="B", help="site id for the overview clip")
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    try:
        from iz.world.mujoco_cart import simulate_cart
        from iz.world.mujoco_door import simulate_door
    except Exception as e:  # noqa: BLE001
        print(f"render skipped: {e!r}")
        return 0
    p = lambda n: os.path.join(args.out, n)  # noqa: E731
    try:
        _report("fire_door_before", simulate_door(FIRE, [Action(tool="push_door", args={"force_n": 60})], record_path=p("fire_door_before.mp4")))
        _report("fire_door_after", simulate_door(FIRE, [Action(tool="request_open"), Action(tool="wait", args={"seconds": 10})], record_path=p("fire_door_after.mp4")))
        _report("cart_before", simulate_cart(CART, 75.0, record_path=p("cart_before.mp4")))
        _report("cart_after", simulate_cart(CART, 40.0, record_path=p("cart_after.mp4")))
    except Exception as e:  # noqa: BLE001
        print(f"render skipped: {e!r}")
    try:
        from iz.world.sites import get_site
    except ImportError as e:
        print(f"site_overview: skipped ({e})")
        return 0
    try:
        from iz.world.scene3d import render_site_clip
        site = get_site(args.site)
        ids = [z.id for z in site.zones]
        path = ids[: min(len(ids), 5)]
        try:
            from iz.world.sites import shortest_path
            lobby = next((z.id for z in site.zones if z.kind == "public"), ids[0])
            far = next((z.id for z in reversed(site.zones) if z.id != lobby and z.kind in ("ward", "office")), ids[-1])
            path = shortest_path(site, lobby, far) or path
        except Exception:  # noqa: BLE001
            pass
        out = render_site_clip(site, list(path), p("site_overview.mp4"))
        print(f"site_overview: {'-> ' + out if out else 'render skipped: render_site_clip returned None'}")
    except Exception as e:  # noqa: BLE001
        print(f"site_overview: render skipped: {e!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
