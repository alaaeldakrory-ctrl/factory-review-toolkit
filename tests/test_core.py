"""Core library: catalog, tags, findings, rules, validation, safe writing, revision diff."""
import os

import pytest
from pxr import Usd, UsdGeom, UsdUtils

from conftest import drawing, needs_packs
from frt_core import catalog as CAT
from frt_core import findings as F
from frt_core import revision, rules, tags, usdio, validate
from frt_core.geom import Shapes
from frt_core.materials import MaterialLibrary

PARAMETRIC = ["industrial_table", "lift_table", "flow_rack", "conveyor", "pallet_rack"]


def _bound(prim):
    bc = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
    return bc.ComputeWorldBound(prim).ComputeAlignedRange()


@pytest.mark.parametrize("key", PARAMETRIC)
def test_parametric_item_sits_on_floor_and_matches_footprint(stage, key):
    prim, p = CAT.place(stage, "/World", key, "X1", (0, 0, 0), 0.0)
    r = _bound(prim)
    assert r.GetMin()[1] == pytest.approx(0, abs=1.0), "item must stand on the floor"
    L, D = CAT.CATALOG[key].footprint(p)
    # the drawing footprint is the working envelope; small parts may stick out (lift table power pack, conveyor legs)
    assert 0.9 * L <= r.GetSize()[0] <= L + 70
    assert 0.8 * D <= r.GetSize()[2] <= D + 40
    assert tags.role_of(prim) == CAT.CATALOG[key].role and tags.id_of(prim) == "X1"


def test_params_are_clamped_and_checked():
    it = CAT.CATALOG["pallet_rack"]
    p = it.resolve({"levels": 2})
    assert p["levels"] == 2 and p["bays"] == it.params["bays"].default
    with pytest.raises((ValueError, KeyError)):
        it.resolve({"no_such_param": 1})


@needs_packs
@pytest.mark.parametrize("key", ["pallet_boxes", "crate", "forklift"])
def test_pack_items(stage, key):
    prim, _ = CAT.place(stage, "/World", key, "P1", (500, 0, 0), 90.0)
    r = _bound(prim)
    assert not r.IsEmpty() and r.GetMin()[1] == pytest.approx(0, abs=2.0)


def test_findings_roundtrip(tmp_path):
    fl = F.FindingList()
    fl.add(F.Finding("layout.aisle_width", "warning", "Aisle too narrow", location=(10, 0, 20), value=180, limit=240))
    fl.add(F.Finding("layout.aisle_width", "critical", "Blocked"))
    assert [f.id for f in fl] == ["layout.aisle_width#1", "layout.aisle_width#2"]
    assert fl.items[0].view["at"] == [10, 0, 20]
    fl.save(tmp_path / "f.json")
    back = F.FindingList.load(tmp_path / "f.json")
    assert back.counts() == fl.counts() and back.items[0].limit == 240
    with pytest.raises(ValueError):
        F.Finding("x", "bad", "y")


def test_rules_override(tmp_path):
    pack = tmp_path / "plant.toml"
    pack.write_text("[validate]\nfloating_tolerance_cm = 7.5\n")
    r = rules.load([str(pack)])
    assert r.get("validate.floating_tolerance_cm") == 7.5
    assert r.get("validate.sunk_tolerance_cm") is not None              # defaults still there
    assert r.get("no.such.key", 3) == 3


def test_validate_finds_floating_and_overlap(stage):
    lib = MaterialLibrary(stage)
    for name, lo, hi in (("A", (0, 40, 0), (100, 140, 100)),              # floating 40 cm
                         ("B", (300, 0, 0), (400, 100, 100)),
                         ("C", (350, 0, 50), (450, 100, 150))):           # overlaps B
        s = Shapes()
        s.box(lo, hi, "powder_grey")
        UsdGeom.Xform.Define(stage, f"/World/{name}")
        s.build(stage, f"/World/{name}/Geom", lib.refs(["powder_grey"]))
        tags.tag(stage.GetPrimAtPath(f"/World/{name}"), "machine", id=name)
    fl = validate.validate(stage, checks=("units", "support", "overlaps"))
    checks = [f.check for f in fl]
    assert any("float" in c for c in checks)
    assert any("overlap" in c for c in checks)
    assert not any("units" in c for c in checks)


def test_auto_tag_skips_materials(stage):
    lib = MaterialLibrary(stage)
    lib.get("powder_blue")
    UsdGeom.Xform.Define(stage, "/World/Conveyor_01")
    got = tags.auto_tag(stage)
    assert ("/World/Conveyor_01", "conveyor") in got
    assert all(not path.startswith("/World/Looks") for path, _ in got)


def test_safe_stage_swaps(tmp_path):
    out = str(tmp_path / "scene.usd")
    for n in (1, 2):
        with usdio.safe_stage(out) as st:
            UsdGeom.Xform.Define(st, f"/Build{n}")
    st = Usd.Stage.Open(out)
    assert st.GetPrimAtPath("/Build2") and not st.GetPrimAtPath("/Build1")
    assert not os.path.exists(usdio.building_path(out))


def test_revision_diff():
    from frt_core import drawing as D
    a = D.read_plant(drawing("demo_plant_revA.dxf"))
    b = D.read_plant(drawing("demo_plant_revB.pdf"))                    # across formats on purpose
    ch = {(c.kind, c.id) for c in revision.diff(a, b)}
    assert ("removed", "P4") in ch and ("added", "FR5") in ch
    assert ("resized", "Station 3") in ch and ("moved", "Station 4") in ch
    assert ("changed", "R1") in ch and ("moved", "T3") in ch
    assert not any(k in ("Station 1", "T1", "R2", "C1") for _, k in ch)
    assert revision.diff(a, a) == []
    fl = revision.to_findings(revision.diff(a, b), b)
    assert len(fl) == len(ch) and all(f.severity == "info" for f in fl)


@needs_packs
def test_package_is_self_contained(tmp_path):
    from frt_core import package
    scene = str(tmp_path / "src" / "s.usd")
    with usdio.safe_stage(scene) as st:
        from frt_core import units
        units.setup_stage(st)
        CAT.place(st, "/World", "forklift", "FL1")                     # UDIM textures, sub-layers
        CAT.place(st, "/World", "crate", "CR1", (300, 0, 0))           # MDL materials
    rep = package.package(scene, str(tmp_path / "pkg"))
    assert not rep["missing"] and rep["files"] > 5
    assert package.check(rep["scene"]) == []
    _, assets, _ = UsdUtils.ComputeAllDependencies(rep["scene"])
    root = os.path.normcase(str(tmp_path / "pkg"))
    assert assets and all(os.path.normcase(a).startswith(root) for a in assets)
