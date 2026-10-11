"""Fill the Intervention Zero deck's results table and media block.

Stdlib only. Reads out/results.json and docs/media/*.png, rewrites the text
between the RESULTS and MEDIA markers in index.html. Never invents a number:
anything missing renders as "pending".
"""
import argparse
import base64
import datetime
import html
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

ROWS = [
    ("baseline_tfmr", "Baseline TFMR (held-out)", ["baseline_tfmr"]),
    ("evolved_tfmr", "Evolved TFMR (held-out)", ["evolved_tfmr"]),
    ("placebo_tfmr", "Placebo TFMR (held-out)", ["placebo_tfmr"]),
    ("n", "Held-out missions n", ["n", "heldout_n"]),
    ("seeds", "Seeds", ["seeds"]),
    ("ci", "95% CI (Wilson)", ["ci", "wilson_ci"]),
    ("safety_violations", "Safety violations after promotion", ["safety_violations"]),
    ("drift_latency", "Drift detection latency", ["drift_latency"]),
    ("kill_resume", "Kill -9 resume", ["kill_resume", "kill9_resume"]),
]
TFMR_KEYS = {"baseline_tfmr", "evolved_tfmr", "placebo_tfmr"}
DEFAULT_MEDIA = ["cart_before.png", "cart_after.png"]
MAX_IMG_BYTES = 3 * 1024 * 1024
PENDING_MEDIA = ('<p class="pending media-ph">MuJoCo cart clip frames: pending '
                 '(docs/media/cart_before.png, cart_after.png)</p>')
RES_RE = re.compile(r"(<!-- RESULTS:BEGIN -->)(.*?)(<!-- RESULTS:END -->)", re.S)
MEDIA_RE = re.compile(r"(<!-- MEDIA:BEGIN -->)(.*?)(<!-- MEDIA:END -->)", re.S)


def warn(msg):
    print("warning: " + msg, file=sys.stderr)


def trunc(s, n=80):
    return s if len(s) <= n else s[: n - 1] + "…"


def is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def fmt_num(v, pct):
    if isinstance(v, bool):
        return "pass" if v else "fail"
    if isinstance(v, float):
        if pct and 0.0 <= v <= 1.0:
            return f"{v:.1%}"
        return f"{v:.3g}"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, str):
        return trunc(v)
    return trunc(json.dumps(v, sort_keys=True))


def fmt_raw(key, v):
    """Return an unescaped display string."""
    pct = key in TFMR_KEYS or key == "ci"
    if isinstance(v, bool):
        return "pass" if v else "fail"
    if isinstance(v, float):
        return fmt_num(v, key in TFMR_KEYS)
    if isinstance(v, int):
        return str(v)
    if isinstance(v, str):
        return trunc(v)
    if isinstance(v, (list, tuple)):
        if key == "ci" and len(v) == 2 and all(is_num(x) for x in v):
            return f"[{fmt_num(v[0], True)}, {fmt_num(v[1], True)}]"
        return ", ".join(fmt_raw(key, x) for x in v)
    if isinstance(v, dict):
        if key == "ci":
            for lo_k, hi_k in (("lo", "hi"), ("ci_lo", "ci_hi"), ("low", "high")):
                if lo_k in v and hi_k in v and is_num(v[lo_k]) and is_num(v[hi_k]):
                    return f"[{fmt_num(v[lo_k], True)}, {fmt_num(v[hi_k], True)}]"
        if key in ("drift_latency", "kill_resume"):
            if "value" in v and "unit" in v and v["value"] is not None:
                return trunc(f"{fmt_raw(key, v['value'])} {v['unit']}")
            for ok_k in ("ok", "pass"):
                if isinstance(v.get(ok_k), bool):
                    out = "pass" if v[ok_k] else "fail"
                    for dk in ("detail", "note"):
                        if isinstance(v.get(dk), str) and v[dk]:
                            out += " (" + v[dk] + ")"
                            break
                    return trunc(out)
        return trunc(json.dumps(v, sort_keys=True))
    return trunc(str(v))


def fmt(key, v):
    return html.escape(fmt_raw(key, v))


def lookup(results, aliases):
    nested = results.get("metrics") if isinstance(results.get("metrics"), dict) else {}
    for a in aliases:
        if results.get(a) is not None:
            return results[a]
        if nested.get(a) is not None:
            return nested[a]
    return None


def load_results(path):
    """Return (dict or None, status) with status in loaded/missing/unreadable."""
    if not path.exists():
        warn(f"{path} not found; every row stays pending")
        return None, "missing"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        warn(f"{path} unreadable ({e}); every row stays pending")
        return None, "unreadable"
    if not isinstance(data, dict):
        warn(f"{path} top level is not an object; every row stays pending")
        return None, "unreadable"
    return data, "loaded"


def build_results(results, status, path):
    rows, filled = [], 0
    for key, label, aliases in ROWS:
        v = lookup(results, aliases) if results else None
        if v is None:
            cell = '<td class="pending">pending</td>'
        else:
            cell = f'<td class="val">{fmt(key, v)}</td>'
            filled += 1
        rows.append(f'      <tr data-key="{key}"><th>{html.escape(label)}</th>{cell}</tr>')
    if status == "loaded":
        parts = ["Source: out/results.json"]
        for k in ("run_id",):
            if results.get(k) is not None:
                parts.append(f"run {fmt_raw('s', str(results[k]))}")
        for k in ("generated_at", "timestamp"):
            if results.get(k) is not None:
                parts.append(fmt_raw("s", str(results[k])))
                break
        try:
            parts.append("file modified " + datetime.datetime.fromtimestamp(
                path.stat().st_mtime).isoformat(timespec="seconds"))
        except OSError:
            pass
        src = html.escape(" · ".join(parts))
        if results.get("fake_llm") is True:
            src += ' <span class="badge-red">fake LLM run (smoke test, not a result)</span>'
    elif status == "unreadable":
        src = "Source: out/results.json (unreadable)"
    else:
        src = "Source: out/results.json (not yet generated)"
    block = ('\n    <table class="res">\n' + "\n".join(rows) + "\n    </table>\n"
             f'    <p class="src">{src}</p>\n')
    return block, filled


def build_media(media_dir):
    names = list(DEFAULT_MEDIA)
    if media_dir.is_dir():
        for p in sorted(media_dir.glob("*.png")):
            if p.name not in names:
                names.append(p.name)
    figs = []
    for name in names:
        if len(figs) >= 2:  # the slide has room for one before/after pair
            break
        p = media_dir / name
        if not p.is_file():
            continue
        if p.stat().st_size > MAX_IMG_BYTES:
            warn(f"skipping {p} (over 3 MB)")
            continue
        b64 = base64.b64encode(p.read_bytes()).decode("ascii")
        stem = html.escape(p.stem.replace("_", " "))
        figs.append(f'<figure><img src="data:image/png;base64,{b64}" alt="{stem}">'
                    f"<figcaption>{stem}</figcaption></figure>")
    if not figs:
        return "\n" + PENDING_MEDIA + "\n", 0
    return '\n<div class="media">\n' + "\n".join(figs) + "\n</div>\n", len(figs)


def check(html_path, results_path):
    problems = []
    text = html_path.read_text(encoding="utf-8")
    if re.search(r"<script[^>]*\ssrc\s*=", text, re.I):
        problems.append("external <script src=> found")
    for m in re.finditer(r'\b(?:src|href)\s*=\s*"(https?:[^"]*)"', text, re.I):
        url = m.group(1)
        if not re.match(r"https?://(fonts\.googleapis\.com|fonts\.gstatic\.com)(/|$)", url):
            problems.append(f"external url: {url[:80]}")
    n = len(re.findall(r'<section[^>]*class="slide"', text))
    if n != 7:
        problems.append(f"slide count is {n}, expected 7")
    rm = RES_RE.search(text)
    if not rm:
        problems.append("RESULTS marker pair missing")
    if not MEDIA_RE.search(text):
        problems.append("MEDIA marker pair missing")
    if rm and not results_path.exists():
        for cls, txt in re.findall(r'<td class="([^"]*)">(.*?)</td>', rm.group(2), re.S):
            if cls != "pending" or txt.strip() != "pending":
                problems.append(f"value cell not pending while results.json is absent: {txt[:40]}")
    return problems


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, default=ROOT / "out" / "results.json")
    ap.add_argument("--html", type=Path, default=ROOT / "docs" / "slides" / "index.html")
    ap.add_argument("--media", type=Path, default=ROOT / "docs" / "media")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)

    if not a.html.exists():
        print(f"error: {a.html} not found", file=sys.stderr)
        return 2
    if a.check:
        problems = check(a.html, a.results)
        if problems:
            for p in problems:
                print("CHECK FAILED: " + p)
            return 1
        print("OK")
        return 0

    text = a.html.read_text(encoding="utf-8")
    if not RES_RE.search(text) or not MEDIA_RE.search(text):
        print("error: RESULTS or MEDIA marker pair missing in " + str(a.html), file=sys.stderr)
        return 2
    results, status = load_results(a.results)
    rblock, filled = build_results(results, status, a.results)
    mblock, imgs = build_media(a.media)
    text = RES_RE.sub(lambda m: m.group(1) + rblock + m.group(3), text, count=1)
    text = MEDIA_RE.sub(lambda m: m.group(1) + mblock + m.group(3), text, count=1)
    with open(a.html, "w", encoding="utf-8", newline="") as f:
        f.write(text)
    print(f"rows filled {filled}/9, images embedded {imgs}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
