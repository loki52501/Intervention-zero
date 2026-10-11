import json

from iz.loop.campaign import fake_llm, run_campaign
from iz.world.judge import RuleJudge


def _run(tmp_path, **kw):
    args = dict(run_id="t1", days=2, shifts=3, missions_per_shift=6, seed=0, llm=fake_llm(), judge=RuleJudge(),
                out_dir=tmp_path, use_atlas=False, drift_day=2, skills=False)
    args.update(kw)
    return run_campaign(**args)


def test_runs_and_emits(tmp_path):
    s = _run(tmp_path)
    assert s["missions"] == 2 * 3 * 3 * 6
    assert s["decisions"] > 0 and s["takeovers"] > 0
    assert (tmp_path / "checkpoints" / "t1.json").exists()
    kinds = {json.loads(l)["kind"] for l in (tmp_path / "events.jsonl").read_text().splitlines()}
    assert {"site_loaded", "mission_start", "decision", "outcome", "takeover", "fix_proposed",
            "metrics"} <= kinds
    # trap candidate is refused by the (local) validator
    assert any(r["author"] == "human" for r in s["refusals"])
    assert "validator_refused" in kinds
    assert s["promotions"] and s["final_version"] == len(s["promotions"])
    assert s["avg_context_tokens"] >= 0


def test_resume_is_idempotent(tmp_path):
    s1 = _run(tmp_path)
    n_events = len((tmp_path / "events.jsonl").read_text().splitlines())
    s2 = _run(tmp_path)
    assert s2["decisions"] == s1["decisions"]
    assert s2["steps_executed_this_run"] == 0
    assert s2["promotions"] == s1["promotions"]
    versions = [p["version"] for p in s2["promotions"]]
    assert len(versions) == len(set(versions))
    # only the per-run site_loaded events were re-emitted
    assert len((tmp_path / "events.jsonl").read_text().splitlines()) - n_events == 3


def test_stop_after_then_resume_matches_full(tmp_path):
    full = _run(tmp_path / "full", run_id="r")
    part = _run(tmp_path / "part", run_id="r", stop_after=5)
    assert part["stopped_early"] and part["steps_executed_this_run"] == 5
    done = _run(tmp_path / "part", run_id="r")
    assert done["decisions"] == full["decisions"]
    assert done["missions"] == full["missions"]
    assert [p["version"] for p in done["promotions"]] == [p["version"] for p in full["promotions"]]
