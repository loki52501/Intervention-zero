import json
import types

from iz.llm.client import FakeLLM
from iz.robotmem import curriculum as cur

BOUNDS = {"cruise_speed": (0.3, 1.6)}


class FakeMemory:
    def __init__(self, procs=None):
        self.procs = procs or []
        self.episodes, self.facts, self.learned = [], [], []
        self.recall_calls = 0

    def recall(self, env, k=3):
        self.recall_calls += 1
        return {"procedures": self.procs, "facts": [], "episodes": [], "taxonomy": []}

    def warm_start(self, env):
        if self.procs:
            return dict(self.procs[0]["params"]), {"source": "procedure", "procedure_id": self.procs[0]["_id"]}
        return {"cruise_speed": 1.4}, {"source": "default", "from": None}

    def context_text(self, recall, budget_tokens=600):
        return "MEMORY: slow down" if recall.get("procedures") else ""

    def remember_episode(self, env, result, params_version):
        self.episodes.append((env["env_id"], result, params_version))

    def remember_facts(self, env, facts):
        self.facts.append((env["env_id"], facts))

    def learn_procedure(self, env, params, metrics, parent, taxonomy_nodes_fixed=None):
        pid = f"p{len(self.learned)}"
        self.learned.append({"_id": pid, "params": params, "metrics": metrics, "parent": parent})
        return pid


def fake_run(params, world, seed):
    ok = params["cruise_speed"] <= world["limit"]
    return {"seed": seed, "success": ok, "status": "success" if ok else "takeover",
            "failure": None if ok else "cart_tipped", "operator_tip": "slow", "summary": {}}


def build(spec, seed):
    return {"limit": 1.0, "seed": seed}


spec = types.SimpleNamespace(name="grocery", venue="store", spills=[], cart=None, shoppers=None,
                             aisles=3, size_m=[12, 8])


def make_propose(seen):
    def propose(llm, params, records, feedback=None):
        seen.append([r["seed"] for r in records])
        llm.complete_json("m", "s", "CURRENT_PARAMS: x\n", {})
        return {"edits": {"cruise_speed": 0.8}, "rationale": "r", "prediction": {}, "refused": []}
    return propose


def run(mem, use_memory=True, llm=None, seen=None, tmp=None, **kw):
    return cur.learn_env(spec, mem, generations=2, use_memory=use_memory, llm=llm or FakeLLM(responses=[{}]),
                         build_fn=build, run_fn=fake_run, variations_fn=lambda s, n, seed: list(range(n)),
                         propose_fn=make_propose(seen if seen is not None else []),
                         default_params={"cruise_speed": 1.4}, card_text="CARD", run_id="t",
                         events_path=tmp, **kw)


def test_warm_start_used_when_memory_has_procedure(tmp_path):
    mem = FakeMemory([{"_id": "p9", "params": {"cruise_speed": 0.9}}])
    out = run(mem, tmp=tmp_path / "e.jsonl")
    assert out["history"][0]["params"]["cruise_speed"] == 0.9
    assert out["gen0_heldout"] == {"success": 6, "n": 6}
    assert out["memory_info"]["used"] and mem.recall_calls == 1


def test_cold_control_ignores_memory_and_does_not_write(tmp_path):
    mem = FakeMemory([{"_id": "p9", "params": {"cruise_speed": 0.9}}])
    out = run(mem, use_memory=False, tmp=tmp_path / "e.jsonl")
    assert out["history"][0]["params"]["cruise_speed"] == 1.4
    assert out["gen0_heldout"]["success"] == 0
    assert mem.recall_calls == 0 and not mem.episodes and not mem.learned
    assert out["first_promotion_gen"] == 1


def test_heldout_never_passed_to_propose_and_flushed_after(tmp_path):
    seen, mem = [], FakeMemory()
    run(mem, seen=seen, tmp=tmp_path / "e.jsonl")
    assert seen and all(s < 100 for ss in seen for s in ss)  # train seeds are 0..3
    seeds = [r["seed"] for _, r, _ in mem.episodes]
    assert any(100 <= s < 200 for s in seeds)  # held-out written to episodic, but only...
    first_held = min(i for i, (_, r, _) in enumerate(mem.episodes) if r["split"] == "heldout")
    assert all(r["split"] == "train" for _, r, _ in mem.episodes[:first_held])  # ...after training episodes


def test_episodes_and_procedure_lineage(tmp_path):
    mem = FakeMemory([{"_id": "p9", "params": {"cruise_speed": 1.4}}])
    run(mem, tmp=tmp_path / "e.jsonl")
    splits = {r["split"] for _, r, _ in mem.episodes}
    assert splits == {"train", "heldout", "safety"}
    assert len(mem.episodes) >= 4 + 6 + 3 + 4
    assert len(mem.learned) == 1  # gen 1 promotes, gen 2 has no effective edit
    assert mem.learned[0]["parent"] == "p9" and mem.learned[0]["params"]["cruise_speed"] == 0.8
    assert mem.facts and mem.facts[0][1][0]["source"] == "spec"


def test_context_reaches_fixer_and_events(tmp_path):
    mem = FakeMemory([{"_id": "p9", "params": {"cruise_speed": 1.4}}])
    llm = FakeLLM(responses=[{}])
    ev = tmp_path / "e.jsonl"
    run(mem, llm=llm, tmp=ev)
    assert "CARD" in llm.calls[0]["user"] and "MEMORY: slow down" in llm.calls[0]["user"]
    evs = [json.loads(line) for line in ev.read_text().splitlines()]
    kinds = {e["kind"] for e in evs}
    assert {"metrics", "gate_result", "promoted", "takeover"} <= kinds
    assert all(e["payload"]["world"] == "env" and e["payload"]["memory"]["used"] for e in evs)


def test_gate_rejects_on_safety(tmp_path):
    def run_unsafe(params, world, seed):
        r = fake_run(params, world, seed)
        if params["cruise_speed"] < 1.0:
            r.update(success=False, status="safety_violation")
        return r
    mem = FakeMemory()
    out = cur.learn_env(spec, mem, generations=1, llm=FakeLLM(responses=[{}]), build_fn=build,
                        run_fn=run_unsafe, variations_fn=lambda s, n, seed: list(range(n)),
                        propose_fn=make_propose([]), default_params={"cruise_speed": 1.4},
                        card_text="", events_path=None)
    assert out["history"][1]["verdict"] == "REJECT" and not mem.learned


def test_env_tour_wilson_and_markdown():
    import importlib.util
    import pathlib
    p = pathlib.Path(__file__).resolve().parent.parent / "scripts" / "env_tour.py"
    sp = importlib.util.spec_from_file_location("env_tour", p)
    m = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(m)
    lo, hi = m.wilson(3, 6)
    assert 0.15 < lo < 0.2 and 0.8 < hi < 0.85
    assert m.wilson(0, 0) == (0.0, 1.0)
    s = lambda k: {"gen0_heldout_success": k, "n": 6, "first_promotion_gen": 1, "final_heldout_rate": 1.0,  # noqa: E731
                   "warm_start": {"source": "default"}, "context_tokens": 10, "memory_context_tokens": 4}
    md = m.render_md({"run_id": "r", "generations": 3, "fake_llm": True,
                      "envs": [{"prompt": "p", "warm": s(5), "cold": s(1)}]})
    assert "Transfer" in md and "5/6" in md and "1/6" in md
