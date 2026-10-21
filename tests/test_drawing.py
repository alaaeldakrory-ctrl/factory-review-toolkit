"""Drawing import and interpretation: the DXF and the PDF of the same layout must say the same thing."""
import pytest

from conftest import drawing
from frt_core import drawing as D
from frt_core.drawing import align

REVS = ["demo_plant_revA.dxf", "demo_plant_revA.pdf", "demo_plant_revB.dxf", "demo_plant_revB.pdf"]


@pytest.fixture(scope="module", params=REVS)
def plant(request):
    p = D.read_plant(drawing(request.param))
    p.rev = "B" if "revB" in request.param else "A"
    return p


def test_building_outline(plant):
    assert (plant.width, plant.depth) == pytest.approx((6000, 4000), abs=1)
    assert plant.grid_x == pytest.approx([0, 1000, 2000, 3000, 4000, 5000, 6000], abs=1)
    assert plant.grid_y == pytest.approx([0, 1000, 2000, 3000, 4000], abs=1)


def test_doors(plant):
    kinds = sorted((d.kind, d.wall) for d in plant.doors)
    assert kinds == [("dock", "S")] * 3 + [("door", "E"), ("door", "N"), ("door", "W"), ("rollup", "E")]


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
    assert len(ids) == (35 if plant.rev == "B" else 34)                 # B: +FR5 +P5 -P4
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


def test_lengths():
    from frt_core.drawing.lengths import parse_length as L
    assert L("40'") == pytest.approx(1219.2)
    assert L("16'-10\"") == pytest.approx(513.08)
    assert L("16' 10 1/2\"") == pytest.approx(514.35)
    assert L("12.5 m") == 1250 and L("3500 mm") == 350 and L("250") == 250
    with pytest.raises(ValueError):
        L("forty feet")


def test_pdf_autocad_polylines(tmp_path):
    """AutoCAD plots closed polylines as one path of joined lines, and wide polylines as triangle strips; both must
    come back as closed shapes, with the PDF's CAD layer and a calibration written in feet."""
    import fitz
    from frt_core.drawing import pdf_import
    doc = fitz.open()
    pg = doc.new_page(width=400, height=300)
    ocg = doc.add_ocg("Stations")
    sh = pg.new_shape()
    sh.draw_polyline([(50, 50), (150, 50), (150, 120), (50, 120), (50, 50)])      # joined lines, closed
    sh.finish(color=(1, 0.5, 0), oc=ocg)
    x0, y0, x1, y1, w = 200, 50, 300, 120, 1.0                                      # wide polyline as triangles
    for (a, b, c) in [((x0, y0), (x1, y0), (x1, y0 + w)), ((x1, y0), (x1, y1), (x1 - w, y1)),
                      ((x1, y1), (x0, y1), (x0, y1 - w)), ((x0, y1), (x0, y0), (x0 + w, y0))]:
        sh.draw_polyline([a, b, c, a])
    sh.finish(color=(0, 0, 1), closePath=False)
    sh.commit()
    path = str(tmp_path / "t.pdf")
    doc.save(path)
    dr = pdf_import.load(path, calibration=((50, 50), (150, 50), "10'"))
    sizes = sorted((round(p.bbox()[2] - p.bbox()[0]), round(p.bbox()[3] - p.bbox()[1]), p.layer) for p in dr.polys)
    assert (305, 213, "Stations") in sizes                          # 100 pt = 10 ft = 304.8 cm
    assert any(abs(w - 305) < 2 and abs(h - 213) < 2 and layer != "Stations" for w, h, layer in sizes)
    assert "Stations" in dr.layers
