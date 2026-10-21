"""Draw the generic test plant ("Demo Plant") as a DXF and a PDF, revisions A and B, with the kinds of mess real
drawings have: units in mm with a far-away origin, the outline as loose duplicated lines with a small gap, junk text,
stray lines, equipment as named blocks (DXF) or labelled rectangles (PDF).

    python tools/make_sample_drawings.py [out_dir]       -> demo_plant_revA.dxf / .pdf, demo_plant_revB.dxf / .pdf

Plant: 60 x 40 m, 10 m column grid A-G x 1-5, three docks and a roll-up door, two personnel doors, a 4 m main
aisle, a walkway, an assembly line (conveyors, tables, lift tables, flow racks) and a pallet / crate area.
Revision B moves table T3, lengthens conveyor C2, adds flow rack FR3 and removes pallet P4.
"""
import os
import sys

import ezdxf
from reportlab.lib.colors import Color, black
from reportlab.lib.pagesizes import A3, landscape
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas

OX, OY = 215000.0, 87000.0          # the drawing's own origin offset (mm), as in a real site plan
W, H = 60000.0, 40000.0
GRID = 10000.0


def equipment(rev):
    """(id, kind, x, y, length, depth, rotation deg) in plant mm from the SW corner; x east, y north."""
    eq = [
        ("C1", "CONVEYOR", 12000, 26000, 6000, 720, 0),
        ("C2", "CONVEYOR", 19000, 26000, 6000, 720, 0),
        ("T1", "TABLE", 12500, 24000, 1800, 800, 0),
        ("T2", "TABLE", 15500, 24000, 1800, 800, 0),
        ("T3", "TABLE", 20000, 24000, 1800, 800, 0),
        ("T4", "TABLE", 23000, 24000, 1800, 800, 0),
        ("LT1", "LIFTTABLE", 14000, 28200, 1300, 800, 0),
        ("LT2", "LIFTTABLE", 21500, 28200, 1300, 800, 0),
        ("FR1", "FLOWRACK", 13000, 31000, 3680, 1800, 180),
        ("FR2", "FLOWRACK", 21000, 31000, 3680, 1800, 180),
        ("P1", "PALLET", 42000, 8000, 1200, 1000, 0),
        ("P2", "PALLET", 43800, 8000, 1200, 1000, 0),
        ("P3", "PALLET", 45600, 8000, 1200, 1000, 0),
        ("P4", "PALLET", 47400, 8000, 1200, 1000, 0),
        ("CR1", "CRATE", 42000, 11500, 1200, 1000, 90),
        ("CR2", "CRATE", 44200, 11500, 1200, 1000, 90),
        ("FL1", "FORKLIFT", 52000, 9000, 1370, 3730, 0),
    ]
    if rev == "B":
        eq = [e for e in eq if e[0] != "P4"]
        eq = [("T3", "TABLE", 21000, 23500, 1800, 800, 0) if e[0] == "T3" else e for e in eq]
        eq = [("C2", "CONVEYOR", 19500, 26000, 7000, 720, 0) if e[0] == "C2" else e for e in eq]
        eq.append(("FR3", "FLOWRACK", 29000, 31000, 3680, 1800, 180))
    return eq


DOORS = [  # (kind, wall, position along the wall from its start (mm), width)
    ("DOCK", "S", 15000, 3000), ("DOCK", "S", 25000, 3000), ("DOCK", "S", 35000, 3000),
    ("ROLLUP", "E", 20000, 4000), ("DOOR", "N", 5000, 1000), ("DOOR", "W", 30000, 1000),
]
AISLES = [("MAIN", 0, 18000, 60000, 22000), ("CROSS", 38000, 0, 41000, 18000)]           # x0, y0, x1, y1
WALKWAYS = [("WALK", 1000, 33000, 59000, 34500)]


def door_point(wall, s):
    return {"S": (s, 0), "N": (s, H), "W": (0, s), "E": (W, s)}[wall]


# ------------------------------------------------------------------------------------------ DXF
def write_dxf(path, rev):
    doc = ezdxf.new("R2018", setup=True)
    doc.header["$INSUNITS"] = 4                                # millimetres
    msp = doc.modelspace()
    for name, col in (("A-WALL", 7), ("S-GRID", 8), ("S-COLS", 5), ("A-DOOR", 3), ("F-AISLE", 2), ("F-WALK", 3),
                      ("E-EQUIP", 1), ("TEXT", 7)):
        doc.layers.add(name, color=col)
    P = lambda x, y: (OX + x, OY + y)
    # blocks
    col = doc.blocks.new("COL-W14")
    col.add_lwpolyline([(-200, -200), (200, -200), (200, 200), (-200, 200)], close=True)
    for kind, w in (("DOCK", 3000), ("ROLLUP", 4000), ("DOOR", 1000)):
        b = doc.blocks.new(f"{kind}-{w}")
        b.add_line((-w / 2, 0), (w / 2, 0))
        b.add_line((-w / 2, 0), (-w / 2, 600 if kind != "DOOR" else w))
        b.add_line((w / 2, 0), (w / 2, 600 if kind != "DOOR" else 0))
        if kind == "DOOR":
            b.add_arc((-w / 2, 0), w, 0, 90)
    blocks = {}
    for _, kind, _, _, L, D, _ in equipment(rev):
        name = f"{kind}_{int(L)}x{int(D)}"
        if name in blocks:
            continue
        b = doc.blocks.new(name)
        b.add_lwpolyline([(-L / 2, -D / 2), (L / 2, -D / 2), (L / 2, D / 2), (-L / 2, D / 2)], close=True)
        b.add_line((-L / 2, -D / 2), (L / 2, D / 2))
        b.add_text(kind, dxfattribs={"height": 150}).set_placement((0, 0))
        blocks[name] = b
    # outline as loose lines: one duplicated, one corner 3 mm short (a gap)
    wl = {"layer": "A-WALL"}
    msp.add_line(P(0, 0), P(W, 0), dxfattribs=wl)
    msp.add_line(P(0, 0), P(W, 0), dxfattribs=wl)                            # duplicate
    msp.add_line(P(W, 0), P(W, H), dxfattribs=wl)
    msp.add_line(P(W, H), P(3, H), dxfattribs=wl)                            # gap at the NW corner
    msp.add_line(P(0, H), P(0, 0), dxfattribs=wl)
    msp.add_line(P(-300, -300), P(W + 300, -300), dxfattribs=wl)             # outer face of the wall
    # grid and columns
    for i in range(7):
        x = i * GRID
        msp.add_line(P(x, -3000), P(x, H + 3000), dxfattribs={"layer": "S-GRID", "linetype": "CENTER"})
        msp.add_text("ABCDEFG"[i], dxfattribs={"layer": "S-GRID", "height": 800}).set_placement(P(x, H + 3800))
    for j in range(5):
        y = j * GRID
        msp.add_line(P(-3000, y), P(W + 3000, y), dxfattribs={"layer": "S-GRID", "linetype": "CENTER"})
        msp.add_text(str(j + 1), dxfattribs={"layer": "S-GRID", "height": 800}).set_placement(P(-4200, y))
    for i in range(7):
        for j in range(5):
            msp.add_blockref("COL-W14", P(i * GRID, j * GRID), dxfattribs={"layer": "S-COLS"})
    # doors
    for kind, wall, s, w in DOORS:
        x, y = door_point(wall, s)
        rot = {"S": 0, "N": 180, "W": 270, "E": 90}[wall]
        msp.add_blockref(f"{kind}-{w}", P(x, y), dxfattribs={"layer": "A-DOOR", "rotation": rot})
    # floor marking
    for _, x0, y0, x1, y1 in AISLES:
        msp.add_lwpolyline([P(x0, y0), P(x1, y0), P(x1, y1), P(x0, y1)], close=True, dxfattribs={"layer": "F-AISLE"})
    for _, x0, y0, x1, y1 in WALKWAYS:
        msp.add_lwpolyline([P(x0, y0), P(x1, y0), P(x1, y1), P(x0, y1)], close=True, dxfattribs={"layer": "F-WALK"})
    # equipment
    for eid, kind, x, y, L, D, rot in equipment(rev):
        msp.add_blockref(f"{kind}_{int(L)}x{int(D)}", P(x, y), dxfattribs={"layer": "E-EQUIP", "rotation": rot})
        msp.add_text(eid, dxfattribs={"layer": "TEXT", "height": 250}).set_placement(P(x - L / 2, y + D / 2 + 150))
    # mess: stray short line, junk text, title block
    msp.add_line(P(30500, 12000), P(30512, 12004), dxfattribs={"layer": "0"})
    msp.add_text("SEE NOTE 4", dxfattribs={"layer": "TEXT", "height": 300}).set_placement(P(8000, 15000))
    msp.add_text(f"DEMO PLANT - LAYOUT REV {rev}", dxfattribs={"layer": "TEXT", "height": 900}).set_placement(P(0, -6500))
    msp.add_text("SCALE 1:200  UNITS MM", dxfattribs={"layer": "TEXT", "height": 500}).set_placement(P(0, -8000))
    doc.saveas(path)


# ------------------------------------------------------------------------------------------ PDF
def write_pdf(path, rev, scale=200.0):
    page = landscape(A3)
    c = canvas.Canvas(path, pagesize=page)
    ox, oy = 40 * mm, 45 * mm                                     # page position of the plant's SW corner
    S = lambda v: v / scale * mm                                   # plant mm -> page points
    P = lambda x, y: (ox + S(x), oy + S(y))
    c.setTitle(f"Demo Plant layout rev {rev}")
    # grid
    c.setStrokeColor(Color(0.6, 0.6, 0.6))
    c.setDash(6, 3)
    c.setLineWidth(0.3)
    for i in range(7):
        x0, y0 = P(i * GRID, -2000)
        x1, y1 = P(i * GRID, H + 2000)
        c.line(x0, y0, x1, y1)
        c.drawCentredString(x0, y1 + 6, "ABCDEFG"[i])
    for j in range(5):
        x0, y0 = P(-2000, j * GRID)
        x1, y1 = P(W + 2000, j * GRID)
        c.line(x0, y0, x1, y1)
        c.drawRightString(x0 - 4, y0 - 3, str(j + 1))
    c.setDash()
    # walls
    c.setStrokeColor(black)
    c.setLineWidth(1.6)
    c.rect(*P(0, 0), S(W), S(H))
    # columns
    c.setLineWidth(0.6)
    for i in range(7):
        for j in range(5):
            x, y = P(i * GRID - 200, j * GRID - 200)
            c.rect(x, y, S(400), S(400), fill=1)
    # doors: white gap + tick marks
    for kind, wall, s, w in DOORS:
        x, y = door_point(wall, s)
        horiz = wall in "SN"
        a = P(x - w / 2, y) if horiz else P(x, y - w / 2)
        b = P(x + w / 2, y) if horiz else P(x, y + w / 2)
        c.setStrokeColor(Color(1, 1, 1))
        c.setLineWidth(2.4)
        c.line(*a, *b)
        c.setStrokeColor(black)
        c.setLineWidth(0.5)
        c.line(*a, *b)
        c.setFont("Helvetica", 5)
        c.drawString(a[0] + 2, a[1] + (-8 if wall == "S" else 3), kind)
    # floor marking
    c.setLineWidth(0.8)
    c.setStrokeColor(Color(0.85, 0.7, 0.0))
    for _, x0, y0, x1, y1 in AISLES:
        c.rect(*P(x0, y0), S(x1 - x0), S(y1 - y0))
    c.setStrokeColor(Color(0.2, 0.6, 0.3))
    for _, x0, y0, x1, y1 in WALKWAYS:
        c.rect(*P(x0, y0), S(x1 - x0), S(y1 - y0))
    # equipment: rectangles + labels (no layers or blocks in a PDF)
    c.setStrokeColor(Color(0.1, 0.1, 0.6))
    c.setLineWidth(0.5)
    label = {"CONVEYOR": "CONVEYOR", "TABLE": "TABLE", "LIFTTABLE": "LIFT TABLE", "FLOWRACK": "FLOW RACK",
             "PALLET": "PALLET", "CRATE": "CRATE", "FORKLIFT": "FORKLIFT"}
    for eid, kind, x, y, L, D, rot in equipment(rev):
        if rot % 180 == 90:
            L, D = D, L
        c.rect(*P(x - L / 2, y - D / 2), S(L), S(D))
        c.setFont("Helvetica", 4)
        cx, cy = P(x, y)
        c.drawCentredString(cx, cy - 1.5, label[kind])
        c.drawString(*P(x - L / 2, y + D / 2 + 120), eid)
    # title block
    c.setFillColor(black)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(40 * mm, 20 * mm, f"DEMO PLANT - LAYOUT REV {rev}")
    c.setFont("Helvetica", 9)
    c.drawString(40 * mm, 14 * mm, "SCALE 1:200   DIMENSIONS IN MM   A3")
    c.drawString(250 * mm, 14 * mm, "SEE NOTE 4")
    c.showPage()
    c.save()


def main(out):
    os.makedirs(out, exist_ok=True)
    for rev in ("A", "B"):
        write_dxf(os.path.join(out, f"demo_plant_rev{rev}.dxf"), rev)
        write_pdf(os.path.join(out, f"demo_plant_rev{rev}.pdf"), rev)
    print("->", out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "samples", "drawings"))
