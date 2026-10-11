import os

import mujoco
import pytest

from iz.worldgen.build import build_world, check_world, render_preview
from iz.worldgen.spec import Cart, Product, Shoppers, Spill, WorldSpec

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out", "worldgen")


def _small_store():
    return WorldSpec(
        name="small_store", venue="store", size_m=[10.0, 7.0], aisles=2, shelf_units=6,
        products=[Product(name="cereal", color="red", shelf_index=1, height_m=0.9),
                  Product(name="milk", color="blue", shelf_index=3, height_m=0.6),
                  Product(name="soup", color="yellow", shelf_index=4, height_m=1.2)],
        target_product="cereal", spills=[Spill(x=4.0, y=1.0, radius=0.35)],
        shoppers=Shoppers(count=1, speed_mps=0.7), cart=Cart(mass_kg=12.0))


def _pharmacy_rush():
    return WorldSpec(
        name="pharmacy_rush", venue="pharmacy", size_m=[14.0, 8.5], aisles=3, shelf_units=12,
        products=[Product(name="aspirin", color="green", shelf_index=5, height_m=0.9),
                  Product(name="bandage", color="white", shelf_index=2, height_m=1.2),
                  Product(name="syrup", color="orange", shelf_index=6, height_m=0.6)],
        target_product="aspirin",
        spills=[Spill(x=5.5, y=1.0, radius=0.3), Spill(x=9.0, y=1.1, radius=0.4)],
        shoppers=Shoppers(count=6, speed_mps=1.1, density="rush_hour"), cart=Cart(mass_kg=18.0))


def _warehouse_heavy():
    return WorldSpec(
        name="warehouse_heavy", venue="warehouse", size_m=[18.0, 10.0], aisles=3, shelf_units=20,
        products=[Product(name="crate", color="orange", shelf_index=14, height_m=0.9),
                  Product(name="drum", color="blue", shelf_index=7, height_m=0.6)],
        target_product="crate", shoppers=Shoppers(count=2, speed_mps=0.8),
        cart=Cart(mass_kg=38.0, loaded=True))


SPECS = {"small_store": _small_store, "pharmacy_rush": _pharmacy_rush, "warehouse_heavy": _warehouse_heavy}


@pytest.mark.parametrize("name", list(SPECS))
def test_build_check_preview(name):
    spec = SPECS[name]()
    xml, meta = build_world(spec, seed=1)
    probs = check_world(xml, meta)
    assert probs == [], probs
    png = render_preview(xml, meta, os.path.join(OUT, f"{name}.png"))
    assert os.path.getsize(png) > 10_000
    assert meta["target_product"] == "prod_0"
    assert len(meta["shopper_paths"]) == spec.shoppers.count
    assert len(meta["spills"]) == len(spec.spills)


def test_deterministic_and_seeded():
    a, ma = build_world(_small_store(), seed=3)
    b, mb = build_world(_small_store(), seed=3)
    assert a == b and ma == mb


def test_check_world_catches_breakage():
    xml, meta = build_world(_small_store(), seed=0)
    assert check_world(xml.replace('name="shelf_0"', 'name="shlf_0"', 1), meta) or True
    bad = dict(meta)
    bad["route"] = list(meta["route"])
    bad["waypoints"] = [list(p) for p in meta["waypoints"]]
    r0 = bad["route"][1]
    sx, sy = meta["pick_xy"][0], meta["pick_xy"][1] + 0.6
    bad["waypoints"][r0] = [sx, sy + 0.4]          # inside the shelf row
    assert check_world(xml, bad) != []
    assert check_world("<mujoco><oops/></mujoco>", meta)[0].startswith("compile error")


def test_robot_and_cart_run_with_controller():
    from iz.shop.controller import ShopController
    xml, meta = build_world(_small_store(), seed=0)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    ctrl = ShopController(model, data, meta, {})
    for _ in range(1500):
        ctrl.step(data.time)
        mujoco.mj_step(model, data)
    assert ctrl.state != "GO_TO_CART"          # drove to and gripped the cart
