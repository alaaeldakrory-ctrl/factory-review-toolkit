"""The drawing reader on a small synthetic AutoCAD-style PDF: layers, a detailed unit with a station label and task
bullets under it, a jib crane circle with its capacity, a staged kit on a model layer, a symbol with no readable
label that is taught once and then found everywhere."""
import math

import pytest


def _make_pdf(path):
    import fitz
    doc = fitz.open()
    pg = doc.new_page(width=800, height=500)
    L = {n: doc.add_ocg(n) for n in ("0", "Overheads", "KIT100", "Racks", "TEXT")}

    def rect(sh, x0, y0, x1, y1):
        sh.draw_polyline([(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)])

    # two detailed "units" (many strokes) on layer 0, 40 x 13 pt
    for ux in (100, 400):
        sh = pg.new_shape()
        for k in range(60):
            sh.draw_line((ux + k * 0.65, 200), (ux + k * 0.65, 213))
        sh.draw_line((ux, 200), (ux + 40, 200))
        sh.draw_line((ux, 213), (ux + 40, 213))
        sh.finish(color=(0, 0.8, 0), oc=L["0"])
        sh.commit()
    # jib: a circle on Overheads
    sh = pg.new_shape()
    sh.draw_circle((600, 150), 50)
    sh.finish(color=(0, 0, 1), oc=L["Overheads"])
    sh.commit()
    # staged kit on a model layer, and three copies of an unlabelled rack symbol
    sh = pg.new_shape()
    rect(sh, 300, 60, 314, 100)
    sh.finish(color=(1, 0, 0), oc=L["KIT100"])
    sh.commit()
    for rx in (100, 200, 400):
        sh = pg.new_shape()
        rect(sh, rx, 300, rx + 25, 310)
        sh.draw_line((rx, 300), (rx + 25, 310))
        sh.finish(color=(0, 0.6, 0), oc=L["Racks"])
        sh.commit()
    # labels: station names under the units, bullets, crane capacity
    pg.insert_text((100, 230), "ST 3", fontsize=6)
    pg.insert_text((100, 237), "• Battery Install", fontsize=6)
    pg.insert_text((100, 244), "• Decals", fontsize=6)
    pg.insert_text((400, 230), "TEST 1 Scale", fontsize=6)
    pg.insert_text((585, 150), "2000lb Jib", fontsize=6)
    doc.save(path)


@pytest.fixture
def team(tmp_path, monkeypatch):
    from frt_core.drawing import classify
    d = tmp_path / "profiles"
    d.mkdir()
    (d / "team.toml").write_text('''[profile]
name = "Test team"
extends = ["base"]

[[rules]]
type = "staged_kit"
layer = ["KIT*"]
shape = "rect"
param = { model = "layer" }

[[rules]]
type = "product_unit"
layer = ["0"]
shape = "group"
min_parts = 40
''', encoding="utf-8")
    monkeypatch.setattr(classify, "USER_PROFILES", str(d))
    pdf = str(tmp_path / "layout.pdf")
    _make_pdf(pdf)
    return pdf


def test_read_and_teach(team):
    from frt_core.drawing import classify
    from frt_core.drawing.read import read, save_settings
    save_settings(team, profile="team", calibration={"a": [100, 200], "b": [140, 200], "distance": "8'"})
    r = read(team)
    assert r.drawing.layers and "Overheads" in r.drawing.layers
    assert r.drawing.cm_per_pt == pytest.approx(8 * 30.48 / 40)
    types = {c.type for c in r.classes}
    assert {"product_unit", "staged_kit", "jib_crane"} <= types
    jib = next((o, c) for o, c in r.items if c.type == "jib_crane")
    assert jib[1].params["capacity"] == "2000lb" and "circle" in jib[1].reason
    kit = next(c for c in r.classes if c.type == "staged_kit")
    assert kit.params["model"] == "KIT100"
    st = {s.name: s for s in r.stations}
    assert set(st) == {"Station 3", "Test 1"}
    assert st["Station 3"].tasks == ["Battery Install", "Decals"]
    assert st["Test 1"].tasks == ["Scale"]
    unit_ids = {o.id for o, c in r.items if c.type == "product_unit"}
    assert {st["Station 3"].anchor, st["Test 1"].anchor} == unit_ids     # each label finds the unit above it
    # the rack symbols have no label: unknown until taught once
    racks = [o for o, c in r.items if o.layer == "Racks"]
    assert len(racks) == 3 and all(c.type == "unknown" for o, c in r.items if o.layer == "Racks")
    classify.teach_symbol(r.profile, racks[0], "flow_rack", "test")
    r2 = read(team)
    assert sum(1 for o, c in r2.items if c.type == "flow_rack") == 3
    assert all("taught symbol" in c.reason for c in r2.classes if c.type == "flow_rack")


def test_base_profile_is_read_only(team):
    from frt_core.drawing import classify
    from frt_core.drawing.read import read
    r = read(team, "base", calibration=((100, 200), (140, 200), "8'"))
    with pytest.raises(PermissionError):
        classify.teach_symbol(r.profile, r.objects[0], "flow_rack")


def test_answer_key_scores(team):
    from frt_core.drawing import answers
    from frt_core.drawing.read import read, save_settings
    save_settings(team, profile="team", calibration={"a": [100, 200], "b": [140, 200], "distance": "8'"})
    r = read(team)
    jib = next(o for o, c in r.items if c.type == "jib_crane")
    rack = next(o for o, c in r.items if o.layer == "Racks")
    # the reviewer adds the rack the reader could not name, and confirms the rest
    answers.save_key(r, corrections=[{"center": list(rack.center), "type": "flow_rack", "size": [max(rack.size), min(rack.size)]}])
    sc = answers.score(read(team))
    assert sc["types"]["flow_rack"]["missed"] == 1          # the reader still leaves it unknown: a miss
    assert sc["types"]["jib_crane"]["found"] == 1 and sc["stations"]["tasks_ok"] == 2
    assert 0 < sc["f1"] < 1
