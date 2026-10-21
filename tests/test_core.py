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

PARAMETRIC = ["industrial_table", "lift_table", "flow_rack", "conveyor", "pallet_rack", "bin_rack", "bridge_crane"]


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


def test_jib_crane_hook_and_swing(stage):
    prim, p = CAT.place(stage, "/World", "jib_crane", "JC1", params={"boom": 400, "height": 450, "slew": 90})
    r = _bound(prim)
    assert r.GetMin()[1] == pytest.approx(0, abs=1.0)
    assert r.GetMax()[2] < 50 and r.GetMin()[2] < -350             # slew 90 swings the boom toward -Z
    assert CAT.CATALOG["jib_crane"].footprint(p) == (800, 800)
    hook = stage.GetPrimAtPath("/World/JC1/Attach/hook")
    assert hook and UsdGeom.Xformable(hook).GetLocalTransformation().ExtractTranslation()[1] == pytest.approx(180)


def test_symbol_priority_by_legend_order():
    import importlib
    I = importlib.import_module("frt_core.drawing.interpret")        # the package re-exports a function of that name
    lg = I.load_legend()
    assert I._symbol("LIFT TABLE", lg, labels=True) == "lift_table"
    assert I._symbol("BIN RACK", lg, labels=True) == "bin_rack"
    assert I._symbol("BRIDGE CRANE | TABLE | T7", lg, labels=True) == "bridge_crane"
    assert I._symbol("TABLE", lg, labels=True) == "industrial_table"
    assert I._symbol("PALLET RACK", lg, labels=True) == "pallet_rack"
    assert I._symbol("PALLETISER", lg, labels=True) is None             # whole words only


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
    assert ("removed", "P4") in ch and ("added", "FR5") in ch and ("added", "P5") in ch
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


def test_safe_stage_refreshes_cached_layer(tmp_path):
    out = str(tmp_path / "scene.usd")
    with usdio.safe_stage(out) as st:
        UsdGeom.Xform.Define(st, "/Old")
    held = Usd.Stage.Open(out)                       # like an app that had the scene open
    with usdio.safe_stage(out) as st:
        UsdGeom.Xform.Define(st, "/New")
    again = Usd.Stage.Open(out)
    assert again.GetPrimAtPath("/New") and not again.GetPrimAtPath("/Old")
    del held


def test_safe_stage_refuses_scene_open_elsewhere(tmp_path):
    out = str(tmp_path / "scene.usd")
    with usdio.safe_stage(out) as st:
        UsdGeom.Xform.Define(st, "/A")
    usdio.mark_open(out)                                   # this (live) process "has it open"
    with pytest.raises(usdio.SwapFailed):
        with usdio.safe_stage(out) as st:
            UsdGeom.Xform.Define(st, "/B")
    with open(usdio.marker_path(out), "w") as fh:          # a marker left by a process that is gone: ignored
        fh.write("999999")
    with usdio.safe_stage(out) as st:
        UsdGeom.Xform.Define(st, "/C")
    assert not os.path.exists(usdio.marker_path(out))


def test_layout_checks():
    from frt_core import drawing as D
    from frt_core import layout, rules as RL
    a = D.read_plant(drawing("demo_plant_revA.dxf"))
    b = D.read_plant(drawing("demo_plant_revB.pdf"))
    fa, fb = layout.check(a), layout.check(b)
    assert list(fa) == []                                                # rev A: a clean layout
    # move the main aisle onto the y = 20 m column line and drop the east exit: both are caught
    from dataclasses import asdict
    bad = asdict(a)
    bad["aisles"][0] = [0.0, 1800.0, 6000.0, 2200.0]
    bad["doors"] = [d for d in bad["doors"] if not (d["kind"] == "door" and d["wall"] == "E")]
    fbad = layout.check(bad)
    col = next(f for f in fbad if f.check == "layout.column_in_aisle")
    assert col.value == pytest.approx(180) and col.severity == "critical" and "5 building column" in col.message
    assert any(f.check == "layout.exit_travel" for f in fbad)
    blocked = [f for f in fb if f.check == "layout.in_aisle"]
    assert len(blocked) == 1 and "P5" in blocked[0].message and blocked[0].severity == "critical"
    assert blocked[0].location is not None and blocked[0].view
    # a stricter plant rule pack: the 4 m main aisle now fails, the 3.5 m cross aisle too
    strict = RL.Rules({"layout": {"main_aisle_min_cm": 450, "cross_aisle_min_cm": 380, "exit_travel_max_cm": 10000}}, [])
    widths = [f for f in layout.check(a, strict) if f.check == "layout.aisle_width"]
    assert len(widths) == 2
    assert not any(f.check == "layout.exit_travel" for f in layout.check(a, strict))


def test_smooth_primitives_make_consistent_meshes(stage):
    import numpy as np
    from frt_core.geom import Shapes
    s = Shapes()
    s.rbox((0, 50, 0), (40, 100, 30), 4, "m")
    s.lathe((0, 0, 0), (30, 0, 0), [(0, 0), (0.05, 10), (0.9, 10), (0.95, 11), (1, 6)], "m")
    s.sweep([(0, 0, 0), (50, -20, 0), (100, 0, 0)], 1.0, "m", radii=[1.0, 1.5, 1.0])
    s.torus((0, 0, 0), (0, 0, 1), 3, 0.8, "m", stretch=4)
    s.bolt((0, 0, 0), (0, 1, 0), 2.0, "m", length=5)
    P, C, I, N = s.groups["m"]
    assert sum(C) == len(I) == len(N)
    assert max(I) < len(P) and min(C) >= 3
    n = np.asarray(N, float)
    assert np.allclose(np.linalg.norm(n, axis=1), 1.0, atol=1e-6)
    pts = np.asarray(P, float)
    rb = pts[: (2 * (3 + 1)) ** 2 * 6]                            # the rounded box comes first (8 x 8 per face)
    assert rb[:, 0].max() == pytest.approx(20) and rb[:, 1].min() == pytest.approx(0)
    corner = np.linalg.norm(rb - np.array([20, 100, 15]), axis=1).min()
    assert corner > 1.0                                           # the corner is rounded off
    m = s.build(stage, "/World/T", None)[0]
    st = m.GetPrim().GetAttribute("primvars:st").Get()
    assert st is not None and len(st) == len(I)                   # UVs for every face corner


def test_flow_rack_modular(stage):
    from pxr import UsdGeom
    prim, p = CAT.place(stage, "/World", "flow_rack", "FR", params={
        "bay_widths": "120,90", "levels": 3, "lanes": 2, "bed": "wheel,roller,ball", "return_top": True})
    item = CAT.CATALOG["flow_rack"]
    assert item.footprint(p) == (218.0, 180.0)                         # 120 + 90 + one upright
    att = {c.GetName(): UsdGeom.Xformable(c).GetOrderedXformOps()[0].Get()
           for c in stage.GetPrimAtPath("/World/FR/Attach").GetChildren()}
    assert {f"lane_L{l}_B{b}_N{n}_pick" for l in (1, 2, 3) for b in (1, 2) for n in (1, 2)} <= set(att)
    lo, hi = att["lane_L1_B1_N1_pick"], att["lane_L1_B1_N1_load"]
    assert lo[1] < hi[1] and lo[2] > hi[2]                              # lanes roll down to the pick side (+Z)
    r_pick, r_load = att["lane_L3_B1_N1_pick"], att["lane_L3_B1_N1_load"]
    assert r_pick[1] < r_load[1] and r_pick[2] < r_load[2]              # the return level rolls back
    pi = UsdGeom.PointInstancer(stage.GetPrimAtPath("/World/FR/Rolling"))
    kinds = {t.name.split("_")[0] for t in pi.GetPrototypesRel().GetTargets()}
    assert {"wheel", "roller", "nut", "ball"} <= kinds
    assert stage.GetPrimAtPath("/World/FR/Load/L1_B1_N1")             # each lane's load can be animated


def test_generated_textures_travel_with_the_scene(tmp_path, monkeypatch):
    """Finishes use generated maps referenced relative to the scene; saving copies them next to it, so a scene
    folder opens on any computer."""
    from pxr import UsdShade
    from frt_core import textures as GT
    from frt_core.materials import MaterialLibrary
    from frt_core.usdio import safe_stage
    monkeypatch.setattr(GT, "ROOT", str(tmp_path / "cache"))
    out = str(tmp_path / "scene" / "t.usd")
    with safe_stage(out) as st:
        lib = MaterialLibrary(st, "/World/Looks")
        lib.get("powder_yellow")
        lib.get("galvanised")
    saved = Usd.Stage.Open(out)
    sh = UsdShade.Shader(saved.GetPrimAtPath("/World/Looks/powder_yellow/OmniPBR"))
    p = sh.GetInput("albedo_texture").Get().path
    assert p.startswith("./textures/powder/")
    assert os.path.isfile(os.path.join(str(tmp_path / "scene"), p))
    assert os.path.isfile(str(tmp_path / "scene" / "textures" / "galv" / "galv_normal.png"))
    assert os.path.isfile(str(tmp_path / "scene" / "materials" / "frt_worn.mdl"))            # the wear material too
    assert os.path.isfile(str(tmp_path / "scene" / "textures" / "wear" / "wear_chip.png"))
