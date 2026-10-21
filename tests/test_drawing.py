"""Drawing import and interpretation: the DXF and the PDF of the same layout must say the same thing."""
import pytest

from conftest import drawing
from frt_core import drawing as D
from frt_core.drawing import align

REVS = ["demo_plant_revA.dxf", "demo_plant_revA.pdf", "demo_plant_revB.dxf", "demo_plant_revB.pdf"]


@pytest.fixture(scope="module", params=REVS)
def plant(request):
    return D.read_plant(drawing(request.param))


def test_building_outline(plant):
    assert (plant.width, plant.depth) == pytest.approx((6000, 4000), abs=1)
    assert plant.grid_x == pytest.approx([0, 1000, 2000, 3000, 4000, 5000, 6000], abs=1)
    assert plant.grid_y == pytest.approx([0, 1000, 2000, 3000, 4000], abs=1)


def test_doors(plant):
    kinds = sorted((d.kind, d.wall) for d in plant.doors)
    assert kinds == [("dock", "S")] * 3 + [("door", "N"), ("door", "W"), ("rollup", "E")]


def test_floor(plant):
    assert len(plant.aisles) == 2 and len(plant.walkways) == 1


def test_stations(plant):
    names = [s.name for s in plant.stations]
    assert names == [f"Station {n}" for n in range(1, 8)]
    inside = {s.name: s.label_inside for s in plant.stations}
    assert all(inside[f"Station {n}"] for n in (1, 2, 3, 4))          # "STATION n" written inside the box
    assert not any(inside[f"Station {n}"] for n in (5, 6, 7))         # "STN n" written beside it
    assert not plant.unknown


def test_equipment(plant):
    ids = {e.id: e for e in plant.equipment}
    assert len(ids) == 34
    assert {k for k, e in ids.items() if e.catalog == "pallet_rack"} == {"R1", "R2", "R3"}
    assert ids["R2"].depth > 200                                      # double row, back to back
    assert ids["FL1"].catalog == "forklift"
    assert ids["CR1"].rot == pytest.approx(90)
    assert {k for k, e in ids.items() if e.catalog == "bin_rack"} == {"BR1", "BR2", "BR3", "BR4"}
    assert ids["JC5"].catalog == ids["JC6"].catalog == "jib_crane"      # swing circles (curves in the PDF)
    assert ids["JC5"].length == pytest.approx(500, abs=5)
    assert ids["BC1"].catalog == "bridge_crane"                          # dashed, drawn around table T7
    assert ids["T7"].catalog == "industrial_table"


def test_dxf_and_pdf_agree():
    a = D.read_plant(drawing("demo_plant_revA.dxf"))
    b = D.read_plant(drawing("demo_plant_revA.pdf"))
    ea, eb = {e.id: e for e in a.equipment}, {e.id: e for e in b.equipment}
    assert ea.keys() == eb.keys()
    for k in ea:
        assert (ea[k].x, ea[k].y) == pytest.approx((eb[k].x, eb[k].y), abs=3), k
        assert ea[k].catalog == eb[k].catalog
    for s, t in zip(a.stations, b.stations):
        assert (s.x0, s.y0, s.x1, s.y1) == pytest.approx((t.x0, t.y0, t.x1, t.y1), abs=3)


def test_dxf_units_and_cleaning():
    dr = D.load(drawing("demo_plant_revA.dxf"))
    assert any("INSUNITS" in n for n in dr.notes)
    assert any("duplicate" in n for n in dr.notes)


def test_phrases_join_pdf_words():
    from frt_core.drawing.interpret import phrases
    from frt_core.drawing.model import Text
    ts = [Text("STATION", (0, 0), 10), Text("3", (50, 0), 10), Text("far", (5000, 0), 10)]
    got = [t for t, _ in phrases(ts)]
    assert "STATION 3" in got and "far" in got


def test_two_point_alignment():
    dr = D.load(drawing("demo_plant_revA.dxf"))
    a, b = (21500, 8700), (27500, 8700)                              # SW and SE column centres, drawing cm
    e0 = dr.extents()
    align.two_point(dr, (a, b), ((0, 0), (0, 6000)))                 # rotate the plan 90 degrees
    e1 = dr.extents()
    assert e1[2] - e1[0] == pytest.approx(e0[3] - e0[1], abs=1)     # width and height swap
    assert e1[3] - e1[1] == pytest.approx(e0[2] - e0[0], abs=1)
    assert any("rotated 90.00" in n for n in dr.notes)
