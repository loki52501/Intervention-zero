import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import scripts.prompt_world as pw  # noqa: E402

PROMPT = "a pharmacy with a wet floor near aisle 2, a heavy cart and rush-hour shoppers"


def test_keyword_spec_and_repair():
    s, rep = pw.prompt_to_spec(PROMPT, pw.fake_spec_llm())
    assert s["store_type"] == "pharmacy" and s["aisle"] == 2
    assert s["n_spills"] >= 1 and s["spill_near_aisle"]
    assert s["cart_mass_kg"] >= 40 and s["n_shoppers"] >= 4
    bad, rep = pw.repair_spec({"n_spills": 99, "cart_mass_kg": "x", "n_shoppers": -3, "aisle": 9})
    assert bad["n_spills"] == 3 and bad["cart_mass_kg"] == 25.0 and bad["n_shoppers"] == 0 and bad["aisle"] == 3
    assert len(rep) >= 4


def test_variants_respect_aisle():
    s, _ = pw.prompt_to_spec(PROMPT, pw.fake_spec_llm())
    vs = pw.make_variants(s)
    assert [sum(v["split"] == k for v in vs) for k in ("train", "heldout", "safety")] == [8, 8, 4]
    assert len({v["key"] for v in vs}) == 20
    assert all(v["n_shoppers"] >= 4 for v in vs if v["split"] == "safety")


def test_cli_fake_no_learn(tmp_path):
    out = tmp_path / "out"
    doc = pw.run(PROMPT, fake_llm=True, learn_enabled=False, out=str(out), say=lambda *_: None)
    wdir = out / "worldgen" / pw.slugify(PROMPT)
    assert (wdir / "preview.png").stat().st_size > 1000
    hist = json.loads((wdir / "history.json").read_text(encoding="utf-8"))
    assert hist["spec"]["aisle"] == 2 and hist["history"] == []
    assert sum(hist["counts"].values()) >= 12
    assert hist["counts"]["train"] >= 1 and hist["counts"]["heldout"] >= 1
    ev = [json.loads(line) for line in (out / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert ev[0]["payload"]["world"] == "generated" and ev[0]["payload"]["prompt"] == PROMPT
    assert doc["variants"]


def test_built_world_has_requested_spills():
    s, _ = pw.prompt_to_spec(PROMPT, pw.fake_spec_llm())
    v = pw.make_variants(s)[0]
    xml, meta = pw.build_world(v, s)
    assert len(meta["spills"]) == s["n_spills"]
    assert xml.count('name="spill_') == s["n_spills"]
    assert pw.check_world(xml, meta) == []


def test_check_world_flags_broken():
    assert pw.check_world("<mujoco><bad/></mujoco>", {})[0].startswith("compile")
