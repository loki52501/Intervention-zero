from __future__ import annotations

import pytest

mujoco = pytest.importorskip("mujoco")

from iz.types import Edge, Site, Zone  # noqa: E402
from iz.world.scene3d import build_site_xml, render_site_clip  # noqa: E402


def small_site() -> Site:
    return Site(
        id="T", name="Test",
        zones=[Zone(id="LOBBY", kind="public", xy=(0, 0)), Zone(id="C1", kind="corridor", xy=(6, 0)),
               Zone(id="S-2", kind="human_only", xy=(6, 6)), Zone(id="EB", kind="elevator", xy=(12, 0))],
        edges=[Edge(a="LOBBY", b="C1", length_m=6, door_id="D-17", door_type="fire"),
               Edge(a="C1", b="S-2", length_m=6, door_id="X", door_type="shortcut"),
               Edge(a="C1", b="EB", length_m=6)],
    )


def test_xml_compiles():
    s = small_site()
    m = mujoco.MjModel.from_xml_string(build_site_xml(s, robot_at="LOBBY"))
    assert m.ngeom >= len(s.zones)


def test_xml_deterministic():
    s = small_site()
    assert build_site_xml(s, "C1") == build_site_xml(s, "C1")


def test_render_clip(tmp_path):
    out = str(tmp_path / "c.mp4")
    r = render_site_clip(small_site(), ["LOBBY", "C1", "EB"], out, seconds_per_edge=0.3, fps=10)
    if r is None:
        pytest.skip("rendering unavailable")
    import os
    assert os.path.getsize(r) > 0
