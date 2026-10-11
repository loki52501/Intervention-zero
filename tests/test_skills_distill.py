from iz.llm.client import FakeLLM
from iz.skills import library
from iz.skills.distill import distill, skills_for_clusters
from iz.types import RobotView

SRC = {"platform": "youtube", "url": "https://x/y", "title": "t", "situation_type": "fire_door",
       "text": "push fire doors steadily"}


def _llm(rules):
    return FakeLLM(policy=lambda m, s, u, sc: {"rules": rules})


def test_distill_scope_and_cap():
    rules = [{"situation_type": "fire_door", "text": "word " * 60, "params": {"force_n": 35, "wait_s": 5}}] * 3
    es = distill(SRC, _llm(rules), model="m")
    assert 1 <= len(es) <= 2
    e = es[0]
    assert e.scope.situation_type == "fire_door" and e.kind == "add_rule"
    assert len(e.payload["text"].split()) <= 35
    assert e.rationale == "video skill: youtube https://x/y"
    assert e.payload["params"]["force_n"] == 35.0


def test_unsafe_dropped():
    bad = [{"situation_type": "fire_door", "text": "Enter the human_only server room via the shortcut"}]
    assert distill(SRC, _llm(bad), model="m") == []
    ok = [{"situation_type": "fire_door", "text": "Never enter human_only zones; push gently."}]
    assert len(distill(SRC, _llm(ok), model="m")) == 1


def test_clusters_limit_and_empty():
    srcs = [dict(SRC, url=f"u{i}") for i in range(4)]
    rules = [{"situation_type": "fire_door", "text": "push gently"}] * 2
    cl = [{"situation_type": "fire_door", "count": 3}, {"situation_type": "spill", "count": 1}]
    assert len(skills_for_clusters(cl, _llm(rules), limit=2, model="m", sources=srcs)) == 2
    assert len(skills_for_clusters(cl, _llm(rules), limit=1, model="m", sources=srcs)) == 1
    assert skills_for_clusters(cl, _llm(rules), model="m", sources=[]) == []


def test_library(tmp_path):
    import json
    # TEST FIXTURE records (not real sources); both cache formats must load.
    recs = [dict(SRC, url="u1"), dict(SRC, url="u2", situation_type="spill", text="wet floor")]
    (tmp_path / "list.json").write_text(json.dumps(recs), encoding="utf-8")
    (tmp_path / "dict.json").write_text(json.dumps({"records": recs}), encoding="utf-8")
    assert len(library.load_sources(tmp_path / "list.json")) == 2
    srcs = library.load_sources(tmp_path / "dict.json")
    assert len(srcs) == 2
    library.load_sources(use_atlas=False)   # real cache must parse without error
    v = RobotView(situation_id="s", site="A", type="fire_door", zone="z", zone_class="corridor",
                  observable="red fire door")
    top = library.top_skills(v, k=2, sources=srcs)
    assert 1 <= len(top) <= 2 and all(s["situation_type"] == "fire_door" for s in top)
    assert library.count_tokens("abcdefgh") == 2 and library.count_tokens("") == 0
    assert library.load_sources("nonexistent.json") == []
