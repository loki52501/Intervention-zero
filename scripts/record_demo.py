"""Intervention Zero demo-video kit.

What it does:
  * makes sure the UI server is up (reuses a running one, else starts uvicorn
    with RAINDROP_ENABLED=0 and stops it afterwards),
  * checks the page and the SSE replay stream,
  * writes out/video/cues.txt (on-screen time of each beat, from the server's
    replay pacing formula),
  * records the replay in headless Chromium (Playwright) to out/video/raw.webm,
  * optionally renders an ablation card from out/ablation.json,
  * optionally synthesizes narration through ElevenLabs (one POST only),
  * transcodes / muxes with the imageio-ffmpeg binary.

HONESTY NOTE: the demo replay (out/demo_events.jsonl) is a SYNTHETIC event
stream with made-up numbers (the UI shows a SYNTHETIC badge). Ablation numbers
are placeholders while out/ablation.json has fake_llm=true.

Everything written goes under out/video/.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.parse
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "out"
VID = OUT / "video"
PY = sys.executable

CUE_KINDS = {"site_loaded", "takeover", "fix_proposed", "gate_result",
             "validator_refused", "promoted", "reload", "drift_alarm", "coach"}


# ---------------------------------------------------------------- server
def server_up(base_url: str, timeout: float = 2.0) -> bool:
    import httpx
    try:
        return httpx.get(base_url + "/", timeout=timeout).status_code == 200
    except Exception:
        return False


def ensure_server(base_url: str, port: int):
    if server_up(base_url):
        print("UI server already running at", base_url)
        return None
    VID.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["RAINDROP_ENABLED"] = "0"
    log = open(VID / "server.log", "ab")
    kw = {}
    if os.name == "nt":
        kw["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    proc = subprocess.Popen(
        [PY, "-m", "uvicorn", "iz.ui.server:app", "--host", "127.0.0.1",
         "--port", str(port)],
        cwd=str(REPO), env=env, stdout=log, stderr=log, **kw)
    print(f"Started UI server (pid {proc.pid}) on port {port}")
    deadline = time.time() + 25
    while time.time() < deadline:
        if server_up(base_url):
            return proc
        if proc.poll() is not None:
            break
        time.sleep(0.5)
    stop_server(proc)
    raise RuntimeError(f"UI server did not come up; see {VID / 'server.log'}")


def stop_server(proc) -> None:
    if proc is None:
        return
    try:
        proc.terminate()
        try:
            proc.wait(10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(10)
        print("Stopped the UI server this script started")
    except Exception as e:  # pragma: no cover
        print("stop_server error:", e)


# ---------------------------------------------------------------- events
def replay_url(base_url: str, events_rel: str, speed: float) -> str:
    return f"{base_url}/?replay={urllib.parse.quote(events_rel)}&speed={speed:g}"


def load_events(path) -> list[dict]:
    out = []
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except Exception:
            continue
        if isinstance(ev, dict):
            out.append(ev)
    return out


def _ts(ev: dict):
    try:
        return datetime.fromisoformat(str(ev.get("ts")).replace("Z", "+00:00")).timestamp()
    except Exception:
        return None


def _delays(events: list[dict], speed: float) -> list[float]:
    """Delay before each event (first is 0), mirroring server _replay_gen."""
    res, prev = [], None
    for i, ev in enumerate(events):
        t = _ts(ev)
        if i == 0:
            res.append(0.0)
        else:
            if t is not None and prev is not None:
                res.append(min(max(t - prev, 0) / speed, 2.0))
            else:
                res.append(0.5 / speed)
        prev = t
    return res


def estimate_replay_seconds(path, speed: float) -> float:
    return float(sum(_delays(load_events(path), speed)))


def _summary(ev: dict) -> str:
    p = ev.get("payload")
    if not isinstance(p, dict):
        return ""
    parts = []
    for k, v in p.items():
        if k == "synthetic":
            continue
        parts.append(f"{k}={v}")
    s = " ".join(parts).replace("\n", " ")
    return s[:100]


def write_cues(events, speed, path=None) -> Path:
    path = Path(path) if path else VID / "cues.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["Times are relative to page load; add ~1-2 s for page and SSE connect. "
             "Source events are SYNTHETIC."]
    t = 0.0
    for ev, d in zip(events, _delays(events, speed)):
        t += d
        p = ev.get("payload") if isinstance(ev.get("payload"), dict) else {}
        if ev.get("kind") in CUE_KINDS or "clip" in p:
            mm, ss = divmod(t, 60)
            lines.append(f"{int(mm):02d}:{ss:04.1f}  {ev.get('id')}  {ev.get('kind')}  {_summary(ev)}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Wrote cues:", path)
    return path


# ---------------------------------------------------------------- checks
def check_page(base_url, events_rel, speed) -> bool:
    import httpx
    ok = True
    try:
        r = httpx.get(replay_url(base_url, events_rel, speed), timeout=10)
        good = r.status_code == 200 and 'id="gate-panel"' in r.text
    except Exception as e:
        good = False
        print("  page error:", type(e).__name__)
    print("page check (200 + id=\"gate-panel\"):", "PASS" if good else "FAIL")
    ok &= good
    good = False
    try:
        url = f"{base_url}/events?replay={urllib.parse.quote(events_rel)}&speed=50"
        deadline = time.time() + 10
        with httpx.stream("GET", url, timeout=10) as r:
            for line in r.iter_lines():
                if line.startswith("data:"):
                    d = json.loads(line[5:].strip())
                    good = isinstance(d, dict) and "kind" in d
                    break
                if time.time() > deadline:
                    break
    except Exception as e:
        print("  sse error:", type(e).__name__)
    print("SSE check (first data: line has kind):", "PASS" if good else "FAIL")
    return bool(ok and good)


# ---------------------------------------------------------------- playwright
def ensure_playwright(allow_install: bool) -> bool:
    try:
        if importlib.util.find_spec("playwright") is None:
            if not allow_install:
                print("playwright not installed and --no-install given; skipping capture")
                return False
            print("Installing playwright ...")
            r = subprocess.run([PY, "-m", "pip", "install", "playwright"],
                               capture_output=True, text=True, timeout=600)
            if r.returncode != 0:
                print("pip install playwright failed; skipping capture:\n",
                      "\n".join(r.stderr.splitlines()[-5:]))
                return False
            importlib.invalidate_caches()
            if importlib.util.find_spec("playwright") is None:
                print("playwright still not importable; skipping capture")
                return False
        r = subprocess.run([PY, "-m", "playwright", "install", "chromium"],
                           capture_output=True, text=True, timeout=900)
        if r.returncode != 0:
            print("playwright install chromium failed; skipping capture:\n",
                  "\n".join((r.stderr or r.stdout).splitlines()[-5:]))
            return False
        return True
    except Exception as e:
        print(f"playwright setup failed ({type(e).__name__}: {e}); skipping capture")
        return False


def _record_page(prepare, seconds: float, out_path: Path, shots: bool) -> Path | None:
    """Shared Playwright recorder. prepare(page) navigates / sets content."""
    from playwright.sync_api import sync_playwright
    tmpdir = VID / "_pw_tmp"
    shutil.rmtree(tmpdir, ignore_errors=True)
    tmpdir.mkdir(parents=True, exist_ok=True)
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            ctx = browser.new_context(
                viewport={"width": 1920, "height": 1080}, device_scale_factor=1,
                record_video_dir=str(tmpdir),
                record_video_size={"width": 1920, "height": 1080})
            page = ctx.new_page()
            prepare(page)
            if shots:
                page.wait_for_timeout(3000)
                page.screenshot(path=str(VID / "first_frame.png"))
                page.wait_for_timeout(max(seconds * 1000 - 3000, 0))
                page.screenshot(path=str(VID / "last_frame.png"))
            else:
                page.wait_for_timeout(seconds * 1000)
            video = page.video
            ctx.close()  # finalizes the video
            src = Path(video.path())
            browser.close()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.exists():
            out_path.unlink()
        shutil.move(str(src), str(out_path))
        return out_path
    except Exception as e:
        print(f"recording failed: {type(e).__name__}: {e}")
        return None
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def record(url: str, seconds: float, out_path=None) -> Path | None:
    out_path = Path(out_path) if out_path else VID / "raw.webm"

    def prep(page):
        page.goto(url, wait_until="domcontentloaded")
        page.wait_for_selector("#app", timeout=15000)

    return _record_page(prep, seconds, out_path, shots=True)


# ---------------------------------------------------------------- ablation card
def _ablation_html(d: dict) -> str:
    from html import escape
    rows = d.get("rows") or []
    placeholder = bool(d.get("fake_llm")) or d.get("run_id") == "nonexist"
    bars = []
    for r in rows:
        tfmr = float(r.get("tfmr") or 0)
        lo = float(r.get("ci_lo") or 0)
        hi = float(r.get("ci_hi") or 0)
        label = (f"{r.get('arm')}  {tfmr:.1%}  (95% CI {lo:.1%}-{hi:.1%}, n={r.get('n')})")
        bars.append(
            f'<div class="row"><div class="lab">{escape(label)}</div>'
            f'<div class="track"><div class="bar" style="width:{tfmr*100:.2f}%"></div>'
            f'<div class="wh" style="left:{lo*100:.2f}%;width:{max(hi-lo,0)*100:.2f}%"></div>'
            f'<div class="cap" style="left:{lo*100:.2f}%"></div>'
            f'<div class="cap" style="left:{hi*100:.2f}%"></div></div></div>')
    sub = (f"seeds: {escape(str(d.get('seeds')))} | sites: {escape(str(d.get('sites')))} | "
           f"incidents: {escape(str(d.get('n_incidents')))}")
    banner = ('<div class="banner">PLACEHOLDER: fake-LLM smoke run, not a result</div>'
              if placeholder else "")
    return f"""<!doctype html><html><head><meta charset="utf-8"><style>
html,body{{margin:0;width:1920px;height:1080px;background:#0e1116;color:#e6edf3;font-family:Segoe UI,Arial,sans-serif}}
h1{{margin:70px 0 6px 100px;font-size:54px}}
.sub{{margin:0 0 50px 100px;font-size:26px;color:#8b98a5}}
.row{{margin:0 100px 40px 100px}}
.lab{{font-size:30px;margin-bottom:10px}}
.track{{position:relative;height:44px;background:#1b222c;border-radius:6px}}
.bar{{position:absolute;left:0;top:0;height:100%;background:#3fb68b;border-radius:6px}}
.wh{{position:absolute;top:20px;height:4px;background:#f0f6fc}}
.cap{{position:absolute;top:10px;width:3px;height:24px;background:#f0f6fc}}
.banner{{position:absolute;left:100px;right:100px;bottom:70px;padding:24px;text-align:center;
 font-size:56px;font-weight:800;color:#fff;background:rgba(200,30,30,.92);
 border:6px solid #ff6b6b}}
</style></head><body>
<h1>Ablation: takeover-free mission rate</h1><div class="sub">{sub}</div>
{''.join(bars)}{banner}</body></html>"""


def ablation_card(base_url_unused=None, out_path=None, seconds=10) -> Path | None:
    out_path = Path(out_path) if out_path else VID / "ablation.webm"
    src = OUT / "ablation.json"
    if not src.exists():
        print("out/ablation.json missing; skipping ablation card")
        return None
    try:
        d = json.loads(src.read_text(encoding="utf-8"))
    except Exception as e:
        print("ablation.json unreadable; skipping ablation card:", e)
        return None
    html = _ablation_html(d)
    (VID / "ablation_card.html").write_text(html, encoding="utf-8")
    res = _record_page(lambda page: page.set_content(html), seconds, out_path, shots=False)
    if res:
        print("Wrote ablation card:", res)
    return res


# ---------------------------------------------------------------- audio / video
def narration_text(md_path=None) -> str:
    md_path = Path(md_path) if md_path else REPO / "docs" / "VIDEO_SCRIPT.md"
    text = md_path.read_text(encoding="utf-8")
    m = re.search(r"<!-- TTS-BEGIN -->(.*?)<!-- TTS-END -->", text, re.S)
    if not m:
        raise ValueError("TTS-BEGIN/TTS-END markers missing in " + str(md_path))
    return m.group(1).strip()


def synthesize(text, voice_id, out_path=None) -> Path | None:
    out_path = Path(out_path) if out_path else VID / "narration.mp3"
    key = os.environ.get("ELEVENLABS_API_KEY")
    if not key:
        print("ELEVENLABS_API_KEY not set; skipping TTS "
              "(see docs/VIDEO_SCRIPT.md 'record your own voice')")
        return None
    import httpx
    url = (f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
           "?output_format=mp3_44100_128")
    try:
        r = httpx.post(
            url,
            headers={"xi-api-key": key, "Content-Type": "application/json",
                     "Accept": "audio/mpeg"},
            json={"text": text, "model_id": "eleven_multilingual_v2",
                  "voice_settings": {"stability": 0.6, "similarity_boost": 0.75,
                                     "style": 0.0, "use_speaker_boost": True}},
            timeout=120)
    except Exception as e:
        print("TTS request failed:", type(e).__name__)
        return None
    if r.status_code != 200:
        print("TTS error", r.status_code, r.text[:200].replace(key, "***"))
        return None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(r.content)
    print("Wrote narration:", out_path)
    return out_path


def _ffmpeg(args: list[str], out_path: Path) -> Path | None:
    import imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    r = subprocess.run([ff, "-y", *args, str(out_path)], capture_output=True, text=True)
    if r.returncode != 0:
        print("ffmpeg failed:\n" + "\n".join(r.stderr.splitlines()[-20:]))
        return None
    print("Wrote", out_path)
    return out_path


def mux(video, audio, out_path=None) -> Path | None:
    out_path = Path(out_path) if out_path else VID / "demo.mp4"
    return _ffmpeg(["-i", str(video), "-i", str(audio), "-map", "0:v:0", "-map", "1:a:0",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast",
                    "-crf", "20", "-r", "30", "-c:a", "aac", "-b:a", "160k", "-shortest"],
                   out_path)


def transcode_silent(video, out_path=None) -> Path | None:
    out_path = Path(out_path) if out_path else VID / "demo_silent.mp4"
    return _ffmpeg(["-i", str(video), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-preset", "veryfast", "-crf", "20", "-r", "30", "-an"], out_path)


# ---------------------------------------------------------------- main
def main() -> None:
    from dotenv import load_dotenv
    load_dotenv(REPO / ".env")
    ap = argparse.ArgumentParser(description="Intervention Zero demo-video kit")
    ap.add_argument("--events")
    ap.add_argument("--speed", type=float, default=4.0)
    ap.add_argument("--seconds", type=float)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-install", action="store_true")
    ap.add_argument("--no-capture", action="store_true")
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--tts", action="store_true")
    ap.add_argument("--voice-id", default=os.environ.get("ELEVENLABS_VOICE_ID") or "21m00Tcm4TlvDq8ikWAM")
    ap.add_argument("--ablation-card", action="store_true")
    ap.add_argument("--check-script", action="store_true")
    a = ap.parse_args()
    VID.mkdir(parents=True, exist_ok=True)

    if a.check_script:
        n = len(narration_text().split())
        print(f"narration word count: {n}")
        sys.exit(0 if 120 <= n <= 160 else 1)

    if a.events:
        ev_path = Path(a.events)
        if not ev_path.is_absolute():
            ev_path = REPO / ev_path
    else:
        ev_path = OUT / "demo_events.jsonl"
        if not ev_path.exists():
            ev_path = OUT / "events.jsonl"
    if not ev_path.exists():
        print("No events file found (out/demo_events.jsonl or out/events.jsonl)")
        sys.exit(2)
    try:
        events_rel = ev_path.resolve().relative_to(REPO).as_posix()
    except ValueError:
        print("Events file must be inside the repo:", ev_path)
        sys.exit(2)

    events = load_events(ev_path)
    est = estimate_replay_seconds(ev_path, a.speed)
    auto = min(max(est + 4, 10), 120)
    seconds = a.seconds if a.seconds else auto
    print(f"events: {events_rel} ({len(events)}), estimated replay {est:.1f}s, "
          f"auto seconds {auto:.1f}, recording {seconds:.1f}s")
    write_cues(events, a.speed)

    summary: list[tuple[str, str]] = []
    exit_code = 0
    proc = None
    try:
        try:
            proc = ensure_server(a.base_url, a.port)
        except RuntimeError as e:
            print("ERROR:", e)
            sys.exit(1)
        if not check_page(a.base_url, events_rel, a.speed):
            exit_code = 1
        if a.check_only or exit_code:
            sys.exit(exit_code)
        raw = VID / "raw.webm"
        if a.no_capture:
            summary.append(("raw.webm", "skipped (--no-capture)"))
        elif ensure_playwright(not a.no_install):
            res = record(replay_url(a.base_url, events_rel, a.speed), seconds, raw)
            summary.append(("raw.webm", "produced" if res else "FAILED (see above)"))
            if a.ablation_card:
                res = ablation_card(a.base_url)
                summary.append(("ablation.webm", "produced" if res else "skipped (see above)"))
        else:
            summary.append(("raw.webm", "skipped (playwright unavailable)"))
    finally:
        stop_server(proc)

    audio = None
    if a.tts:
        audio = synthesize(narration_text(), a.voice_id)
        summary.append(("narration.mp3", "produced" if audio else "skipped (no key or TTS error)"))
    elif (VID / "narration.mp3").exists():
        audio = VID / "narration.mp3"
        summary.append(("narration.mp3", "existing file reused"))
    raw = VID / "raw.webm"
    if raw.exists():
        r = transcode_silent(raw)
        summary.append(("demo_silent.mp4", "produced" if r else "FAILED"))
        if audio and audio.exists():
            r = mux(raw, audio)
            summary.append(("demo.mp4", "produced" if r else "FAILED"))
        else:
            summary.append(("demo.mp4", "skipped (no narration.mp3)"))
    summary.insert(0, ("cues.txt", "produced"))
    print("\nSummary (out/video/):")
    for name, st in summary:
        print(f"  {name:18s} {st}")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
