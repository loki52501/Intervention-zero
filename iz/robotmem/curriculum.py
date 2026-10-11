"""Cross-environment curriculum: the robot learns each new environment, optionally warm-started
from memory of earlier environments (episodic / semantic / taxonomic / procedural on Atlas).

How memory reaches the fixer: iz.shop.learn.propose() has no context parameter, so we call it
UNCHANGED (keeping its bounds/locked validation) through a thin LLM proxy (_ContextLLM) that
appends a ROBOT_AND_MEMORY_CONTEXT block to the user prompt. The held-out set is never passed
to propose(), and held-out/safety episodes are only written to episodic memory after learning ends.

Gate rule (same as iz.shop.learn.gate, but over (world, seed) items):
PROMOTE iff 0 safety violations, 0 regressions, held-out gain >= 1.
"""
from __future__ import annotations

import json
import logging
import re
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)
_patch_lock = threading.Lock()


def _ok(r: dict) -> bool:
    return bool(r.get("success"))


def _unsafe(r: dict) -> bool:
    return r.get("status") == "safety_violation"


def _rate(rs) -> float:
    return sum(map(_ok, rs)) / max(1, len(rs))


def _tok(text: str) -> int:
    return max(1, len(text) // 4)


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(s).lower()).strip("-")[:40] or "env"


# ------------------------------------------------------------------ default hooks (real stack)
def _default_build(spec, seed):
    from iz.worldgen.build import build_world
    return build_world(spec, seed)


@contextmanager
def _patched_shop(world):
    """run_episode() builds its scene via iz.shop.scene.build_shop(seed) at call time; swap it."""
    import iz.shop.scene as scene
    with _patch_lock:
        orig = scene.build_shop
        scene.build_shop = lambda seed, *a, **k: world
        try:
            yield
        finally:
            scene.build_shop = orig


def _default_run(params, world, seed):
    import inspect
    from iz.shop.episode import run_episode
    sig = inspect.signature(run_episode).parameters
    for kw in ("world", "xml_meta", "scene"):
        if kw in sig:
            return run_episode(params, seed, **{kw: world})
    with _patched_shop(world):
        return run_episode(params, seed)


def _default_spec(prompt, llm):
    from iz.worldgen.generate import prompt_to_spec
    return prompt_to_spec(prompt, llm=llm)


def _default_variations(spec, n, seed):
    from iz.worldgen.generate import variations
    return variations(spec, n, seed)


class _ContextLLM:
    """Proxy that appends robot-card + memory context to the fixer's user prompt."""

    def __init__(self, inner, context: str):
        self.inner, self.context = inner, context

    def complete_json(self, model, system, user, schema):
        if self.context:
            user = user + "\nROBOT_AND_MEMORY_CONTEXT:\n" + self.context
        return self.inner.complete_json(model, system, user, schema)


class _Events:
    def __init__(self, run_id, path, env_name):
        self.run_id, self.env_name = run_id, env_name
        self.path = Path(path) if path else None
        self._seq, self._prefix, self._lock = 0, uuid.uuid4().hex[:8], threading.Lock()
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, kind, payload, memory):
        if not self.path:
            return
        payload = {**payload, "world": "env", "env_name": self.env_name, "memory": memory}
        with self._lock:
            self._seq += 1
            ev = {"id": f"{self._prefix}-{self._seq:06d}", "ts": datetime.now(timezone.utc).isoformat(),
                  "kind": kind, "run_id": self.run_id, "payload": payload}
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(ev, default=str) + "\n")


def spec_facts(spec) -> list[str]:
    """Semantic facts derivable from the world spec (source='spec')."""
    def g(o, k, d=None):
        return o.get(k, d) if isinstance(o, dict) else getattr(o, k, d)
    out = [f"venue is a {g(spec, 'venue', 'store')}"]
    for s in g(spec, "spills", []) or []:
        out.append(f"a spill (radius {g(s, 'radius', 0.3):.2f} m) sits near x={g(s, 'x', 0):.1f}, y={g(s, 'y', 0):.1f}")
    cart = g(spec, "cart")
    if cart is not None:
        out.append(f"carts here weigh about {g(cart, 'mass_kg', 15):.0f} kg")
    sh = g(spec, "shoppers")
    if sh is not None and g(sh, "count", 0):
        out.append(f"{g(sh, 'count')} shoppers walk at ~{g(sh, 'speed_mps', 0.8):.1f} m/s "
                   f"({g(sh, 'density', 'low')} density)")
    aisles, size = g(spec, "aisles"), g(spec, "size_m")
    if aisles:
        out.append(f"{aisles} aisles in a {size[0]:.0f}x{size[1]:.0f} m floor" if size else f"{aisles} aisles")
    return out


def _sig(r: dict) -> str:
    return str(r.get("failure") or r.get("status"))


def _prov(w):
    return (w.get("procedure_id") or w.get("from")) if isinstance(w, dict) else None


def _pid(x, default=None):
    if isinstance(x, dict):
        return x.get("_id") or x.get("id") or default
    return x or default


# ------------------------------------------------------------------ main
def learn_env(env_spec, memory, generations: int = 3, use_memory: bool = True, llm=None, *,
              world_llm=None, n_train: int = 4, n_heldout: int = 6, n_safety: int = 3,
              env_id: str | None = None, env_name: str | None = None, run_id: str | None = None,
              write_memory: bool | None = None, events_path="out/events.jsonl",
              build_fn=None, run_fn=None, spec_fn=None, variations_fn=None, propose_fn=None,
              default_params: dict | None = None, card_text: str | None = None,
              context_budget_tokens: int = 600, k: int = 3) -> dict:
    """Learn one environment. env_spec: one-sentence prompt (str) or a WorldSpec.

    write_memory defaults to use_memory so a cold control never pollutes what warm runs read.
    """
    build_fn = build_fn or _default_build
    run_fn = run_fn or _default_run
    spec_fn = spec_fn or _default_spec
    variations_fn = variations_fn or _default_variations
    write_memory = use_memory if write_memory is None else write_memory
    run_id = run_id or "env-" + uuid.uuid4().hex[:8]

    prompt = env_spec if isinstance(env_spec, str) else None
    spec = spec_fn(prompt, world_llm) if prompt is not None else env_spec
    name = env_name or getattr(spec, "name", None) or _slug(prompt or "env")
    venue = spec.get("venue") if isinstance(spec, dict) else getattr(spec, "venue", "store")
    env_id = env_id or f"{run_id}:{_slug(name)}"
    env = {"env_id": env_id, "env_name": name, "venue": venue, "text": prompt or name,
           "prompt": prompt, "run_id": run_id}

    if card_text is None:
        try:
            from iz.robotmem.card import card_text as _ct, robot_card
            card_text = _ct(robot_card())
        except Exception as e:  # noqa: BLE001
            log.info("robot card unavailable: %s", e)
            card_text = ""
    if propose_fn is None:
        from iz.shop.learn import propose as propose_fn
    if llm is None:
        from iz.llm.client import make_llm
        llm = make_llm()
    if default_params is None:
        from iz.shop.controller import DEFAULT_PARAMS
        default_params = DEFAULT_PARAMS

    def make_set(vseed, n, base):
        return [(build_fn(v, base + i), base + i) for i, v in enumerate(variations_fn(spec, n, vseed))]
    train, heldout, safety = make_set(1, n_train, 0), make_set(2, n_heldout, 100), make_set(3, n_safety, 200)

    def evaluate(params, items):
        out = []
        for world, seed in items:
            r = dict(run_fn(params, world, seed))
            r.setdefault("seed", seed)
            out.append(r)
        return out

    # ---- memory: facts, recall, warm start
    facts = spec_facts(spec)
    recalled, warm_prov, ctx_mem = {}, {"source": "default", "from": None}, ""
    if use_memory:
        recalled = memory.recall(env, k=k) or {}
        params, warm_prov = memory.warm_start(env)
        params = dict(params)
        ctx_mem = memory.context_text(recalled, budget_tokens=context_budget_tokens) or ""
    else:
        params = dict(default_params)
    if write_memory:
        memory.remember_facts(env, [{"fact": f, "source": "spec"} for f in facts])
    n_recalled = sum(len(recalled.get(x) or []) for x in ("procedures", "facts", "episodes"))
    context = "\n".join(x for x in (card_text, "ENV_FACTS: " + "; ".join(facts), ctx_mem) if x)
    mem_info = {"recalled": n_recalled, "used": bool(use_memory and (ctx_mem or _prov(warm_prov))),
                "warm_start": warm_prov, "context_tokens": _tok(context),
                "memory_context_tokens": _tok(ctx_mem) if ctx_mem else 0}
    ev = _Events(run_id, events_path, name)
    fixer_llm = _ContextLLM(llm, context)
    version = lambda gen: f"{run_id}:{_slug(name)}:g{gen}"  # noqa: E731

    tr, ho, sf = evaluate(params, train), evaluate(params, heldout), evaluate(params, safety)
    if write_memory:
        for r in tr:
            memory.remember_episode(env, {**r, "split": "train"}, version(0))
    parent_proc = _prov(warm_prov)
    history = [{"gen": 0, "params": dict(params), "train_rate": _rate(tr), "heldout_rate": _rate(ho),
                "heldout_success": sum(map(_ok, ho)), "heldout_n": len(ho), "verdict": "BASELINE",
                "promoted": False, "edits": {}, "memory": mem_info}]
    ev.emit("metrics", {"gen": 0, "train_rate": history[0]["train_rate"],
                        "heldout_rate": history[0]["heldout_rate"], "params": params}, mem_info)
    feedback, first_promotion = None, None
    for gen in range(1, generations + 1):
        fails = [r for r in tr if not _ok(r)]  # TRAIN only; held-out never shown to the fixer
        for r in fails:
            ev.emit("takeover", {"gen": gen, "seed": r.get("seed"), "failure": r.get("failure"),
                                 "status": r.get("status"), "operator_tip": r.get("operator_tip"),
                                 "summary": r.get("summary")}, mem_info)
        prop = propose_fn(fixer_llm, params, fails, feedback)
        cand = {**params, **prop["edits"]}
        a_tr = a_ho = a_sf = None
        if cand == params:
            g = {"verdict": "REJECT", "reason": "no effective edits proposed", "safety_violations": 0,
                 "regressions": 0, "heldout_gain": 0}
        else:
            a_tr, a_ho, a_sf = evaluate(cand, train), evaluate(cand, heldout), evaluate(cand, safety)
            b, a = {"train": tr, "heldout": ho, "safety": sf}, {"train": a_tr, "heldout": a_ho, "safety": a_sf}
            reg = sum(1 for n in b for x, y in zip(b[n], a[n]) if _ok(x) and not _ok(y))
            safe = sum(1 for n in a for y in a[n] if _unsafe(y))
            hb, ha = sum(map(_ok, ho)), sum(map(_ok, a_ho))
            gain = ha - hb
            if safe:
                v, why = "REJECT", f"safety: {safe} violation(s)"
            elif reg:
                v, why = "REJECT", f"regression: {reg} previously-solved seed(s) now fail"
            elif gain < 1:
                v, why = "REJECT", f"no held-out gain ({hb}->{ha})"
            else:
                v, why = "PROMOTE", f"held-out +{gain}, 0 regressions, 0 safety violations"
            g = {"verdict": v, "reason": why, "safety_violations": safe, "regressions": reg,
                 "heldout_gain": gain,
                 "train": {"before": sum(map(_ok, tr)), "after": sum(map(_ok, a_tr)), "n": len(train)},
                 "heldout": {"before": hb, "after": ha, "n": len(heldout)}}
        ev.emit("gate_result", {"gen": gen, **g}, mem_info)
        promoted = g["verdict"] == "PROMOTE"
        if promoted:
            fixed = sorted({_sig(r) for r in fails} - {_sig(r) for r in a_tr if not _ok(r)})
            params, tr, ho, sf = cand, a_tr, a_ho, a_sf
            first_promotion = first_promotion or gen
            if write_memory:
                for r in tr:
                    memory.remember_episode(env, {**r, "split": "train"}, version(gen))
                memory.remember_facts(env, [{"fact": f"gen {gen}: edits {prop['edits']} fixed "
                                                     f"{', '.join(fixed) or 'nothing new'}", "source": "learned"}])
                pid = memory.learn_procedure(
                    env, params, {"heldout_rate": _rate(ho), "n": len(ho), "gen": gen, "train_rate": _rate(tr),
                                  "taxonomy_nodes_fixed": fixed, "edits": prop["edits"],
                                  "rationale": prop.get("rationale", "")}, parent_proc,
                    taxonomy_nodes_fixed=fixed)
                parent_proc = _pid(pid, parent_proc)
            ev.emit("promoted", {"gen": gen, "params": params, "edits": prop["edits"], "parent": parent_proc},
                    mem_info)
        feedback = {"gen": gen, "edits": prop["edits"], "verdict": g["verdict"], "reason": g["reason"]}
        history.append({"gen": gen, "params": dict(params), "train_rate": _rate(tr), "heldout_rate": _rate(ho),
                        "heldout_success": sum(map(_ok, ho)), "heldout_n": len(ho), "verdict": g["verdict"],
                        "reason": g["reason"], "promoted": promoted, "edits": prop["edits"],
                        "rationale": prop.get("rationale", ""), "memory": mem_info})
        ev.emit("metrics", {"gen": gen, "train_rate": _rate(tr), "heldout_rate": _rate(ho), "params": params},
                mem_info)

    if write_memory:  # held-out / safety episodes enter memory only after learning is over
        for r in ho:
            memory.remember_episode(env, {**r, "split": "heldout"}, version(generations))
        for r in sf:
            memory.remember_episode(env, {**r, "split": "safety"}, version(generations))
    return {"env": env, "history": history, "params": params, "first_promotion_gen": first_promotion,
            "gen0_heldout": {"success": history[0]["heldout_success"], "n": history[0]["heldout_n"]},
            "memory_info": mem_info, "facts": facts, "final_heldout_rate": history[-1]["heldout_rate"]}
