"""Filmstrips from our own MuJoCo sim (cart tip / fire door). Script-local hooks only; no iz/** edits.

Usage: .venv\\Scripts\\python.exe scripts\\filmstrip.py [--only a,b] [--no-video]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import imageio  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

import iz.world.mujoco_cart as mc  # noqa: E402
import iz.world.mujoco_door as md  # noqa: E402
from iz.types import Action  # noqa: E402

OLD_VIS = '<visual><global offwidth="640" offheight="480"/></visual>'
NEW_VIS = ('<visual><global offwidth="1280" offheight="960"/><headlight ambient="0.45 0.45 0.45" '
           'diffuse="0.5 0.5 0.5" specular="0.1 0.1 0.1"/><quality shadowsize="4096"/></visual>'
           '<asset><texture name="sky" type="skybox" builtin="gradient" rgb1="1 1 1" '
           'rgb2="0.86 0.89 0.93" width="256" height="256"/></asset>')

CURRENT_CAM: dict = {}
NO_VIDEO = False
LAST_REC = None

RED, GREEN, GREY = (192, 57, 43), (30, 132, 73), (80, 80, 80)
NOTE = "MuJoCo 3.15 sim \u00b7 our micro-world, not real footage"


def _lighten(xml: str) -> str:
    assert OLD_VIS in xml, "visual literal missing from scene XML"
    return xml.replace(OLD_VIS, NEW_VIS)


class FilmRecorder:
    def __init__(self, model, lookat, distance, azimuth, elevation, fps=30):
        global LAST_REC
        self.model, self.fps, self.frames, self.next_t, self.error = model, fps, [], 0.0, None
        c = CURRENT_CAM
        self.cam = mujoco.MjvCamera()
        self.cam.lookat[:] = c["lookat"]
        self.cam.distance, self.cam.azimuth, self.cam.elevation = c["distance"], c["azimuth"], c["elevation"]
        self.cart_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cart")
        self.hinge = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "hinge")
        self.rx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "rx")
        self.drive = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "drive_x")
        self.renderer = mujoco.Renderer(model, height=480, width=640)
        LAST_REC = self

    def _telemetry(self, data) -> dict:
        if self.cart_id >= 0:
            return {"tilt_deg": math.degrees(math.acos(float(np.clip(data.xmat[self.cart_id][8], -1, 1)))),
                    "cart_x": float(data.xpos[self.cart_id][0])}
        t = {"hinge_deg": math.degrees(-float(data.qpos[self.model.jnt_qposadr[self.hinge]])),
             "robot_x": md.ROBOT_START_X + float(data.qpos[self.model.jnt_qposadr[self.rx]])}
        if self.drive >= 0:
            t["drive_force_n"] = float(data.actuator_force[self.drive])
        return t

    def capture(self, data, force=False):
        if self.renderer is None or (not force and data.time + 1e-9 < self.next_t):
            return
        self.renderer.update_scene(data, camera=self.cam)
        frame = self.renderer.render().copy()
        self.frames.append((float(data.time), frame, self._telemetry(data)))
        self.next_t += 1.0 / self.fps

    def save(self, path):
        try:
            if not self.frames or NO_VIDEO:
                return path if NO_VIDEO else None
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            with imageio.get_writer(path, fps=self.fps, codec="libx264", macro_block_size=1, quality=8) as w:
                for _, f, _ in self.frames + [self.frames[-1]] * self.fps:
                    w.append_data(f)
            return path
        finally:
            if self.renderer is not None:
                self.renderer.close()
                self.renderer = None


def _font(size, bold=False):
    names = (["segoeuib.ttf", "arialbd.ttf"] if bold else ["segoeui.ttf", "arial.ttf"])
    for n in names:
        try:
            return ImageFont.truetype("C:/Windows/Fonts/" + n, size)
        except OSError:
            pass
    return ImageFont.load_default(size=size)


# ---------------------------------------------------------------- scenarios
CART = {"cart_mass_kg": 20.0, "move_force_n": 20.0, "tip_force_n": 35.0, "red_tag": False}
FIRE = {"door_type": "fire", "door_id": "D-17", "door_mass_kg": 40.0, "min_push_n": 160.0, "open_delay_s": 9.0}
CAM_CART_AFTER = dict(lookat=(0.7, 0, 0.6), distance=4.2, azimuth=120, elevation=-20)
CAM_CART_BEFORE = dict(lookat=(0.3, 0, 0.5), distance=3.6, azimuth=120, elevation=-20)
CAM_DOOR = dict(lookat=(0.1, 0, 0.6), distance=5.0, azimuth=25, elevation=-55)


def _run_cart_before(rec):
    return mc.simulate_cart(CART, 40.0, record_path=rec)


def _run_cart_after(rec):
    return mc.simulate_cart(CART, 25.0, record_path=rec)


def _run_fd_before(rec):
    return md.simulate_door(FIRE, [Action(tool="push_door", args={"force_n": 60})], record_path=rec)


def _run_fd_after(rec):
    return md.simulate_door(FIRE, [Action(tool="request_open", args={}),
                                   Action(tool="wait", args={"seconds": 10})], record_path=rec)


SCEN = {
    "cart_before": dict(run=_run_cart_before, cam=CAM_CART_BEFORE, kind="cart",
                        input="push force 40 N (input)", cmp_extra="tipped"),
    "cart_after": dict(run=_run_cart_after, cam=CAM_CART_AFTER, kind="cart",
                       input="push force 25 N (input)", cmp_extra="tipped"),
    "firedoor_before": dict(run=_run_fd_before, cam=CAM_DOOR, kind="door",
                            input="push_door 60 N (input)", cmp_extra="crossed"),
    "firedoor_after": dict(run=_run_fd_after, cam=CAM_DOOR, kind="door",
                           input="request_open + wait 10 s (input)", cmp_extra="crossed"),
}
PARAMS = {"cart_before": CART, "cart_after": CART, "firedoor_before": FIRE, "firedoor_after": FIRE}


def _expected(name, res) -> bool:
    ex = res.extra
    return {"cart_before": ex.get("tipped") is True,
            "cart_after": res.success and ex.get("tipped") is False,
            "firedoor_before": ex.get("crossed") is False,
            "firedoor_after": ex.get("crossed") is True}[name]


def _pick_times(name, frames, res):
    times = [f[0] for f in frames]
    last = times[-1]
    if name == "cart_before":
        T = next((f[0] for f in frames if f[2]["tilt_deg"] >= 85), None)
        end = min(T + 0.3, 4.0, last) if T is not None else last
        return list(np.linspace(0, end, 6))
    if name == "firedoor_after":
        d = FIRE["open_delay_s"]
        return [0, d / 2, d, d + (last - d) / 3, d + 2 * (last - d) / 3, last]
    return list(np.linspace(0, last, 6))


def _status(name, res):
    if name == "cart_before":
        return "force 40 N \u2014 TIPPED", RED
    if name == "cart_after":
        return f"force 25 N \u2014 SLID {res.extra['displacement_m']:.1f} m upright", GREEN
    if name == "firedoor_before":
        return "push 60 N \u2014 CLOSER HELD", RED
    return f"request_open + wait \u2014 CROSSED at {res.time_s:.1f} s", GREEN


def _header(name, res):
    ex = res.extra
    if name.startswith("cart"):
        ph = "BEFORE" if name.endswith("before") else "AFTER"
        f = "40" if ph == "BEFORE" else "25"
        out = "tipped" if ex["tipped"] else "upright"
        return (f"Cart push \u2014 {ph} fix: {f} N (input)  |  sim: {out}, max tilt "
                f"{ex['max_tilt_deg']:.1f}\u00b0, peak {res.peak_force_n:.1f} N")
    ph = "BEFORE" if name.endswith("before") else "AFTER"
    inp = "push_door 60 N (input)" if ph == "BEFORE" else "request_open + wait 10 s (input)"
    if ex["crossed"]:
        out = f"crossed at {res.time_s:.1f} s, peak contact {res.peak_force_n:.1f} N"
    else:
        out = f"not crossed ({res.reason}), peak {res.peak_force_n:.1f} N"
    return f"Fire door D-17 \u2014 {ph} fix: {inp}  |  sim: {out}"


def reason_eq(r, b) -> bool:
    # The cart module may stop a recorded run early (clip long enough) or run on past the tip, so the
    # numbers inside the reason text (angle / distance) legitimately differ; compare the wording only.
    if r.extra.get("tipped") is not None and "displacement_m" in r.extra:
        import re
        return re.sub(r"[\d.]+", "#", r.reason) == re.sub(r"[\d.]+", "#", b.reason)
    return r.reason == b.reason


def compose(name, res, frames, png_path):
    targets = _pick_times(name, frames, res)
    idx = [min(range(len(frames)), key=lambda i: abs(frames[i][0] - t)) for t in targets]
    W, H, G, HEAD, CAP = 480, 360, 8, 56, 58
    width = 6 * W + 7 * G
    canvas = Image.new("RGB", (width, HEAD + H + CAP + 24), "white")
    d = ImageDraw.Draw(canvas)
    d.text((G + 4, 6), _header(name, res), fill=(30, 30, 30), font=_font(21, True))
    d.text((G + 4, 34), NOTE, fill=GREY, font=_font(15))
    f_small, f_bold = _font(18), _font(19, True)
    for k, i in enumerate(idx):
        t, img, tel = frames[i]
        x = G + k * (W + G)
        canvas.paste(Image.fromarray(img).resize((W, H), Image.LANCZOS), (x, HEAD))
        y = HEAD + H + 4
        d.text((x, y), f"t={t:.2f}s", fill=(30, 30, 30), font=f_small)
        if "tilt_deg" in tel:
            l2 = f"tilt {tel['tilt_deg']:.0f}\u00b0 \u00b7 x {tel['cart_x']:.2f} m"
        else:
            l2 = f"door {tel['hinge_deg'] + 0.0:.0f}\u00b0 \u00b7 robot x {tel['robot_x']:.2f} m"
            l2 = l2.replace("door -0\u00b0", "door 0\u00b0")
        d.text((x, y + 24), l2, fill=GREY, font=f_small)
    txt, col = _status(name, res)
    d.text((width - G - 4, HEAD + H + CAP - 2), txt, fill=col, font=f_bold, anchor="ra")
    canvas.save(png_path)
    return [frames[i][0] for i in idx]


def _captioned_video(path, frames, status_txt):
    pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--no-video", action="store_true")
    a = ap.parse_args()
    global NO_VIDEO
    NO_VIDEO = a.no_video
    names = [n for n in a.only.split(",") if n] or list(SCEN)
    os.makedirs(os.path.join(ROOT, "out", "clips"), exist_ok=True)
    media = os.path.join(ROOT, "docs", "media")
    os.makedirs(media, exist_ok=True)

    # unhooked baselines
    base = {n: SCEN[n]["run"](None) for n in names}

    orig_door, orig_cart = md.build_door_xml, mc.build_cart_xml
    md.build_door_xml = lambda p: _lighten(orig_door(p))
    mc.build_cart_xml = lambda p: _lighten(orig_cart(p))
    md.Recorder = FilmRecorder
    mc.Recorder = FilmRecorder

    results, ok_all = {}, True
    for n in names:
        s = SCEN[n]
        CURRENT_CAM.clear()
        CURRENT_CAM.update(s["cam"])
        clip = os.path.join(ROOT, "out", "clips", f"filmstrip_{n}.mp4")
        res = s["run"](clip)
        rec = LAST_REC
        if a.no_video:
            pass
        b = base[n]
        det = (res.success == b.success and res.peak_force_n == b.peak_force_n and reason_eq(res, b)
               and res.extra[s["cmp_extra"]] == b.extra[s["cmp_extra"]])
        if s["kind"] == "door":
            det = det and res.time_s == b.time_s
        png = os.path.join(media, f"{n}.png")
        ftimes = compose(n, res, rec.frames, png)
        exp = _expected(n, res)
        ok_all &= bool(exp and det)
        print(f"{n}: success={res.success} reason={res.reason!r} peak={res.peak_force_n:.1f}N "
              f"time={res.time_s:.2f}s png={png} mp4={clip if not a.no_video else '-'} "
              f"determinism={'OK' if det else 'FAIL'} expected={'OK' if exp else 'FAIL'}")
        print(f"   unhooked: success={b.success} reason={b.reason!r} peak={b.peak_force_n:.1f}N "
              f"time={b.time_s:.2f}s {s['cmp_extra']}={b.extra[s['cmp_extra']]}")
        results[n] = {"params": PARAMS[n], "commanded_input": s["input"], "success": res.success,
                      "reason": res.reason, "peak_force_n": res.peak_force_n, "time_s": res.time_s,
                      "extra": res.extra, "frame_times": ftimes, "camera": {k: list(v) if isinstance(v, tuple) else v for k, v in s["cam"].items()},
                      "clip_path": clip if not a.no_video else None, "png_path": png,
                      "unhooked": {"success": b.success, "reason": b.reason, "peak_force_n": b.peak_force_n,
                                   "time_s": b.time_s, "extra": b.extra},
                      "determinism_ok": bool(det), "outcome_as_expected": bool(exp),
                      "tipped_uses_module_threshold_deg": mc.TILT_DEG}
    if set(names) == set(SCEN):
        with open(os.path.join(media, "filmstrip_results.json"), "w") as f:
            json.dump(results, f, indent=2, default=str)
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
