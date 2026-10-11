"""ENV TOUR: one robot, three environments, memory-assisted learning vs a cold control.

env1 is learned from scratch (memory has nothing from this tour). env2 and env3 are each learned
twice on the SAME world spec and seeds: WARM (episodic/semantic/taxonomic/procedural recall +
warm start) and COLD (use_memory=False). TRANSFER result = generation-0 held-out success and
generations-to-first-promotion, warm vs cold, with Wilson 95% CIs.

Usage: python scripts/env_tour.py [--fresh] [--live] [--fake-llm] [--generations 3] [--run-id X]
Writes out/env_tour.json, out/env_tour.md, UI events to out/events.jsonl (world: "env").
Caveat: recall is global over the Atlas collections; run-scoped env_ids only isolate what THIS
tour writes, so older tours' procedures can also be recalled (shown in the provenance).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PROMPTS = [
    "a small grocery store with a spill near the checkout",
    "a pharmacy at rush hour with a heavy cart",
    "a warehouse aisle with narrow shelves and fast workers",
]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def _fmt(k, n):
    lo, hi = wilson(k, n)
    return f"{k}/{n} = {k / max(1, n):.0%} (95% CI {lo:.0%}-{hi:.0%})"


def summarize(res: dict) -> dict:
    g0 = res["gen0_heldout"]
    lo, hi = wilson(g0["success"], g0["n"])
    return {"env_name": res["env"]["env_name"], "env_id": res["env"]["env_id"],
            "gen0_heldout_success": g0["success"], "n": g0["n"], "gen0_rate": g0["success"] / max(1, g0["n"]),
            "gen0_ci": [lo, hi], "first_promotion_gen": res["first_promotion_gen"],
            "final_heldout_rate": res["final_heldout_rate"], "params": res["params"],
            "warm_start": res["memory_info"]["warm_start"], "memory_used": res["memory_info"]["used"],
            "context_tokens": res["memory_info"]["context_tokens"],
            "memory_context_tokens": res["memory_info"]["memory_context_tokens"]}


def render_md(out: dict) -> str:
    L = ["# Environment tour: transfer result", "",
         f"run_id `{out['run_id']}` | generations {out['generations']} | fake_llm={out['fake_llm']}", "",
         "| env | arm | gen-0 held-out | first promotion (gen) | final held-out | warm start from |",
         "|---|---|---|---|---|---|"]
    for e in out["envs"]:
        for arm in ("scratch", "warm", "cold"):
            s = e.get(arm)
            if not s:
                continue
            fp = s["first_promotion_gen"] if s["first_promotion_gen"] is not None else "none"
            ws = s["warm_start"] or {}
            src = ws.get("from_env") or ws.get("source", "-")
            L.append(f"| {e['prompt'][:45]} | {arm} | {_fmt(s['gen0_heldout_success'], s['n'])} | {fp} | "
                     f"{s['final_heldout_rate']:.0%} | {src} |")
    L += ["", "## Transfer (warm vs cold, same worlds and seeds)", ""]
    for e in out["envs"]:
        if e.get("warm") and e.get("cold"):
            w, c = e["warm"], e["cold"]
            L.append(f"- {e['prompt']}: gen-0 held-out warm {_fmt(w['gen0_heldout_success'], w['n'])} vs cold "
                     f"{_fmt(c['gen0_heldout_success'], c['n'])}; first promotion warm "
                     f"{w['first_promotion_gen']} vs cold {c['first_promotion_gen']}; fixer context "
                     f"{w['context_tokens']} tokens ({w['memory_context_tokens']} from memory).")
    L += ["", "Honest note: n is small; overlapping Wilson CIs mean the transfer gain is not statistically "
          "established. The robot learns controller parameters, not motor control."]
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fresh", action="store_true", help="clear this run's docs before starting")
    ap.add_argument("--live", action="store_true", help="play env3 final params in the MuJoCo viewer")
    ap.add_argument("--fake-llm", action="store_true")
    ap.add_argument("--generations", type=int, default=3)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--out", default="out")
    ap.add_argument("--n-train", type=int, default=4)
    ap.add_argument("--n-heldout", type=int, default=6)
    ap.add_argument("--n-safety", type=int, default=3)
    ap.add_argument("--prompt", action="append", help="override env sentences (repeat 3x)")
    a = ap.parse_args(argv)

    from dotenv import load_dotenv
    load_dotenv()
    from iz.robotmem import curriculum as cur
    from iz.robotmem.memory import EnvMemory

    prompts = a.prompt or PROMPTS
    run_id = a.run_id or "tour-" + uuid.uuid4().hex[:6]
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ev_path = out_dir / "events.jsonl"

    if a.fake_llm:
        from iz.shop.learn import fake_llm
        llm = fake_llm()
    else:
        from iz.llm.client import make_llm
        llm = make_llm()
    memory = EnvMemory()
    memory.scope = run_id + ":"  # env1 truly starts from empty memory; no leakage from older runs
    try:
        from iz.robotmem.card import robot_card, store_card
        store_card(robot_card(), db=memory.db)
    except Exception as exc:  # card persistence must not block the tour
        print("robot card not stored:", type(exc).__name__, exc)
    ids = [f"{run_id}:env{i + 1}" for i in range(len(prompts))]
    ids += [i + ":cold" for i in ids]
    if a.fresh:
        print("cleared:", memory.clear_env(ids))

    kw = dict(generations=a.generations, llm=llm, n_train=a.n_train, n_heldout=a.n_heldout,
              n_safety=a.n_safety, events_path=ev_path, run_id=run_id)
    result = {"run_id": run_id, "generations": a.generations, "fake_llm": a.fake_llm, "envs": []}
    finals = []
    for i, prompt in enumerate(prompts):
        spec = cur._default_spec(prompt, None)  # one spec per env, shared by warm and cold arms
        entry = {"prompt": prompt, "env_id": ids[i]}
        if i == 0:
            r = cur.learn_env(spec, memory, use_memory=True, env_id=ids[i], env_name=f"env1 {spec.name}", **kw)
            entry["scratch"] = summarize(r)
        else:
            r = cur.learn_env(spec, memory, use_memory=True, env_id=ids[i], env_name=f"env{i + 1} {spec.name}", **kw)
            entry["warm"] = summarize(r)
            c = cur.learn_env(spec, memory, use_memory=False, env_id=ids[i] + ":cold",
                              env_name=f"env{i + 1} {spec.name} (cold)", **kw)
            entry["cold"] = summarize(c)
        entry["spec_name"] = spec.name
        finals.append((spec, r["params"]))
        result["envs"].append(entry)
        print(f"[{i + 1}/{len(prompts)}] {prompt}: done")

    (out_dir / "env_tour.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    (out_dir / "env_tour.md").write_text(render_md(result), encoding="utf-8")
    print(render_md(result))

    if a.live:
        import mujoco
        import mujoco.viewer
        from iz.shop.episode import run_episode
        spec, params = finals[-1]
        world = cur._default_build(cur._default_variations(spec, 1, 9)[0], 300)
        m = mujoco.MjModel.from_xml_string(world[0])
        d = mujoco.MjData(m)
        with mujoco.viewer.launch_passive(m, d) as v:
            with cur._patched_shop(world):
                print("live:", {k: run_episode(params, 300, viewer=v).get(k) for k in ("status", "success")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
