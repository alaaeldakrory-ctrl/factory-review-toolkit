"""Freestanding workstation / gantry bridge crane, modelled the way they are built:

- columns: box section with rounded corners, a bolted splice at mid height, base plates with anchors, nuts and
  gussets on a grout plinth, hazard bands, lifting lugs at the top
- frame: box runway beams with a running rail and bolted end caps, a box tie beam across each end frame, long
  diagonal braces with bolted end plates (to the tie beam and along the runway)
- bridges: box girder with a running flange, end trucks with flanged wheels, hub bolts, rubber bumpers and a finned
  drive motor; capacity sign
- hoists: electric chain hoists (single girder: underhung on the running flange) - trolley plates and wheels, finned
  motor with end caps, gear housing, electrical box with cable glands, load chain links, chain bag, bottom block and
  forged hook with latch; a top-running crab with a rope winch on double girders
- pendant control on a sagging cable, festoon cable loops on a C-track along the bridge

Every part has its finish (textured painted steel, zinc bolts, machined wheels, blued chain, rubber, PVC cable,
cast housings, concrete). Parameters as before (span, runway, height, bridges, bridge_positions, girders, trolley,
hook_height, capacity) plus hoists per bridge. Local frame: span along X, runway along Z, floor at Y = 0.

Rigged for animation: Geom (structure), Bridge_N (travels along Z), Bridge_N/Hoist_M (along X, carries its
pendant), Hoist_M/Block (hook block, along Y), Hoist_M/Chain (a point instancer of links, re-laid for every hook
height so the link pitch stays true), Bridge_N/Festoon (trolleys and cable loops that open and close). Travel
limits are stored as frt:rig:* attributes; frt_core.rig.crane.pose() moves it (optionally at a time code).
"""
import math

from pxr import Gf, Sdf, UsdGeom, Vt

from .base import Item, Param, new_shapes, register, shapes_to_stage

SEG = {"0": "abcdef", "1": "bc", "2": "abdeg", "3": "abcdg", "4": "bcfg", "5": "acdfg", "6": "acdefg", "7": "abc",
       "8": "abcdefg", "9": "abcdfg", "L": "def", "B": "cdefg", "T": "aef"}
BOLT = "bolt_zinc"


def _n(v):
    L = math.sqrt(sum(c * c for c in v)) or 1.0
    return tuple(c / L for c in v)


def _x(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _frame(a, b, up=(0.0, 1.0, 0.0)):
    ax = _n(tuple(q - p for p, q in zip(a, b)))
    if abs(sum(x * y for x, y in zip(ax, up))) > 0.95:
        up = (1.0, 0.0, 0.0)
    side = _n(_x(up, ax))
    upv = _n(_x(ax, side))
    return ax, side, upv


def member(s, a, b, w, h, r, mat, up=(0.0, 1.0, 0.0)):
    """Box section from a to b (w across, h deep): rounded corners along it, square-cut ends."""
    s.rprism(a, b, w, h, r, mat, up=up, seg=3)


def end_plate_bolted(s, p, ax, w, h, mat, nb=2, d=1.2):
    """A small plate at p across direction ax, with bolts."""
    _, side, upv = _frame((0, 0, 0), ax)
    s.rbox(p, (1.6, w, h), 0.4, mat, axes=(ax, side, upv), seg=1)
    for k in range(nb):
        off = (k - (nb - 1) / 2) * (h * 0.55)
        q = tuple(pi + upv_i * off + ax_i * 0.8 for pi, upv_i, ax_i in zip(p, upv, ax))
        s.bolt(q, ax, d, BOLT)


def weld_ring(s, centre, axis, w, h, r, mat, bead=0.45):
    """A weld bead around a w x h section (corner radius r) at centre, in the plane across axis."""
    _, side, upv = _frame((0, 0, 0), axis)
    path = []
    for cx, cy, a0 in ((w / 2 - r, h / 2 - r, 0), (-w / 2 + r, h / 2 - r, 90), (-w / 2 + r, -h / 2 + r, 180),
                       (w / 2 - r, -h / 2 + r, 270)):
        for k in range(4):
            t = math.radians(a0 + 90 * k / 3)
            u, v = cx + (r + bead * 0.6) * math.cos(t), cy + (r + bead * 0.6) * math.sin(t)
            path.append(tuple(c + side_i * u + up_i * v for c, side_i, up_i in zip(centre, side, upv)))
    s.sweep(path, bead, mat, sides=6, caps=False, closed=True)


def nameplate(s, centre, normal, right, w=10.0, h=6.0):
    """Riveted aluminium nameplate on a surface."""
    up = _n(_x(normal, right))
    s.rbox(centre, (w, h, 0.25), 0.3, "aluminium", axes=(right, up, normal), seg=1)
    for du in (-w / 2 + 0.8, w / 2 - 0.8):
        for dv in (-h / 2 + 0.8, h / 2 - 0.8):
            q = tuple(c + r_ * du + u_ * dv + n_ * 0.15 for c, r_, u_, n_ in zip(centre, right, up, normal))
            s.lathe(q, tuple(qi + ni * 0.3 for qi, ni in zip(q, normal)), [(0, 0.35), (1, 0.25)], "stainless", sides=8)
    for k in range(3):                                           # engraved lines
        q = tuple(c + u_ * (h / 2 - 1.8 - k * 1.4) + n_ * 0.15 for c, u_, n_ in zip(centre, up, normal))
        s.obox(q, (w * 0.7, 0.35, 0.05), (right, up, normal), "powder_dark")


def warning_label(s, centre, normal, right, size=6.0):
    """Yellow warning label with a black triangle."""
    up = _n(_x(normal, right))
    s.obox(centre, (size, size, 0.12), (right, up, normal), "floor_yellow")
    c = tuple(ci + ni * 0.1 for ci, ni in zip(centre, normal))
    pts = [(0.0, 0.36 * size), (-0.32 * size, -0.22 * size), (0.32 * size, -0.22 * size)]
    for (u0, v0), (u1, v1) in zip(pts, pts[1:] + pts[:1]):
        mid = tuple(ci + r_ * (u0 + u1) / 2 + u_ * (v0 + v1) / 2 for ci, r_, u_ in zip(c, right, up))
        d = _n(tuple(r_ * (u1 - u0) + u_ * (v1 - v0) for r_, u_ in zip(right, up)))
        s.obox(mid, (math.hypot(u1 - u0, v1 - v0), 0.07 * size, 0.06), (d, _n(_x(normal, d)), normal), "powder_black")
    s.obox(tuple(ci + u_ * 0.03 * size for ci, u_ in zip(c, up)), (0.06 * size, 0.22 * size, 0.06), (right, up, normal),
           "powder_black")


FONT5x7 = {      # rows top to bottom, 5 columns: enough for signs (capacities, units, short words)
    "0": ["01110", "10001", "10011", "10101", "11001", "10001", "01110"],
    "1": ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
    "2": ["01110", "10001", "00001", "00010", "00100", "01000", "11111"],
    "3": ["11110", "00001", "00001", "01110", "00001", "00001", "11110"],
    "4": ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
    "5": ["11111", "10000", "11110", "00001", "00001", "10001", "01110"],
    "6": ["00110", "01000", "10000", "11110", "10001", "10001", "01110"],
    "7": ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
    "8": ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
    "9": ["01110", "10001", "10001", "01111", "00001", "00010", "01100"],
    "L": ["10000", "10000", "10000", "10000", "10000", "10000", "11111"],
    "B": ["11110", "10001", "10001", "11110", "10001", "10001", "11110"],
    "S": ["01111", "10000", "10000", "01110", "00001", "00001", "11110"],
    "T": ["11111", "00100", "00100", "00100", "00100", "00100", "00100"],
    "K": ["10001", "10010", "10100", "11000", "10100", "10010", "10001"],
    "G": ["01110", "10001", "10000", "10111", "10001", "10001", "01111"],
    "W": ["10001", "10001", "10001", "10101", "10101", "11011", "10001"],
    "M": ["10001", "11011", "10101", "10101", "10001", "10001", "10001"],
    "A": ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
    "X": ["10001", "10001", "01010", "00100", "01010", "10001", "10001"],
    "C": ["01110", "10001", "10000", "10000", "10000", "10001", "01110"],
    "D": ["11110", "10001", "10001", "10001", "10001", "10001", "11110"],
    "E": ["11111", "10000", "10000", "11110", "10000", "10000", "11111"],
    "F": ["11111", "10000", "10000", "11110", "10000", "10000", "10000"],
    "H": ["10001", "10001", "10001", "11111", "10001", "10001", "10001"],
    "-": ["00000", "00000", "00000", "11111", "00000", "00000", "00000"],
    "U": ["10001", "10001", "10001", "10001", "10001", "10001", "01110"],
    "P": ["11110", "10001", "10001", "11110", "10000", "10000", "10000"],
    "N": ["10001", "11001", "10101", "10011", "10001", "10001", "10001"],
    "O": ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
    "R": ["11110", "10001", "10001", "11110", "10100", "10010", "10001"],
    "I": ["01110", "00100", "00100", "00100", "00100", "00100", "01110"],
    "V": ["10001", "10001", "10001", "10001", "10001", "01010", "00100"],
    ".": ["00000", "00000", "00000", "00000", "00000", "01100", "01100"],
    " ": ["00000"] * 7,
}


def sign_text(s, text, origin, right, down, h, mat, out):
    """Dot-matrix lettering (5 x 7) on a plane: origin top-left, `right` / `down` unit vectors, letter height h."""
    px = h / 7.0
    x0 = 0.0
    for ch in text.upper():
        rows = FONT5x7.get(ch)
        if rows is None:
            continue
        for r_, row in enumerate(rows):
            for c_, bit in enumerate(row):
                if bit == "1":
                    cu, cv = x0 + (c_ + 0.5) * px, (r_ + 0.5) * px
                    c = tuple(o + cu * ri + cv * di + 0.25 * ni for o, ri, di, ni in zip(origin, right, down, out))
                    s.obox(c, (px * 0.92, px * 0.92, 0.3), (right, down, out), mat)
        x0 += 6 * px


def digits(s, text, origin, right, down, h, t, mat, out):
    w, gap = h * 0.55, h * 0.25
    for i, ch in enumerate(text):
        if ch not in SEG:
            continue
        gx = i * (w + gap)
        seg = {"a": (gx, 0, gx + w, t), "d": (gx, h - t, gx + w, h), "g": (gx, h / 2 - t / 2, gx + w, h / 2 + t / 2),
               "f": (gx, 0, gx + t, h / 2), "b": (gx + w - t, 0, gx + w, h / 2), "e": (gx, h / 2, gx + t, h),
               "c": (gx + w - t, h / 2, gx + w, h)}
        for k in SEG[ch]:
            u0, v0, u1, v1 = seg[k]
            cu, cv = (u0 + u1) / 2, (v0 + v1) / 2
            c = tuple(o + cu * r + cv * d_ + 0.25 * n for o, r, d_, n in zip(origin, right, down, out))
            s.obox(c, (u1 - u0, v1 - v0, 0.3), (right, down, out), mat)


def finned_motor(s, a, b, r, mat, cap_mat):
    """Motor along a -> b: cooling fins along the body, domed fan cover at b, flange at a."""
    L = math.dist(a, b)
    prof = [(0.0, r * 0.85), (0.04, r * 1.05), (0.08, r * 1.05), (0.1, r)]
    nf = max(4, int(L / 2.2))
    for k in range(nf):                                          # fins: radius steps along the body
        t0 = 0.12 + 0.68 * k / nf
        t1 = t0 + 0.68 / nf * 0.5
        prof += [(t0, r), (t0 + 0.001, r * 1.12), (t1, r * 1.12), (t1 + 0.001, r)]
    prof += [(0.82, r), (0.84, r * 1.06), (0.92, r * 1.06), (0.97, r * 0.85), (1.0, r * 0.45)]
    s.lathe(a, b, prof, mat, sides=28)
    ax = _n(tuple(q - p for p, q in zip(a, b)))
    tip = tuple(bi + ai * 0.4 for bi, ai in zip(b, ax))
    s.lathe(b, tip, [(0, r * 0.45), (1, r * 0.2)], cap_mat, sides=20)
    for k in range(3):                                           # fan-cover grille rings
        rr = r * (0.55 + 0.15 * k)
        q = tuple(bi - ai * (0.15 + 0.05 * k) * r for bi, ai in zip(b, ax))
        s.torus(q, ax, rr, 0.18, cap_mat, seg=24, sides=4)
    _, side, upv = _frame(a, b)
    mid = tuple((pa + pb) / 2 + u * (r * 1.12 + 2.5) for pa, pb, u in zip(a, b, upv))
    s.rbox(mid, (8, 5, 7), 1.2, mat, axes=(ax, upv, side), seg=1)                     # terminal box
    s.lathe(tuple(m + sd * 3.5 for m, sd in zip(mid, side)), tuple(m + sd * 5.5 for m, sd in zip(mid, side)),
            [(0, 1.2), (0.6, 1.2), (1, 0.8)], "plastic_black", sides=10)


def chain(s, top, bot, link=3.6, wire=0.55, mat="steel_blued"):
    """Load chain from top down to bot (straight, vertical), alternating link planes."""
    x, y, z = top
    k = 0
    while y - link > bot[1]:
        s.torus((x, y - link / 2, z), (1, 0, 0) if k % 2 == 0 else (0, 0, 1), 0.85, wire, mat, seg=14, sides=6,
                stretch=link - 2.2, up=(0, 1, 0))
        y -= link * 0.78
        k += 1


def hook(s, x, y, z, size=1.0, mat="powder_yellow", latch="stainless"):
    """Forged crane hook: shank at the top (y), tapered C-shaped body below, safety latch."""
    k = size
    path, radii = [(x, y, z), (x, y - 6 * k, z)], [2.2 * k, 2.4 * k]
    for i in range(1, 15):
        ang = math.radians(90 + i * 235 / 14)
        path.append((x - 4.5 * k + 7.5 * k * math.cos(ang) + 4.5 * k, y - 13.5 * k + 7.5 * k * math.sin(ang), z))
        radii.append(k * (2.6 - 1.7 * (i / 14) ** 1.2))
    s.sweep(path, 2.0, mat, sides=12, radii=radii)
    s.lathe((x, y + 0.5 * k, z), (x, y - 2 * k, z), [(0, 3.2 * k), (1, 3.2 * k)], "steel_dark", sides=6)   # nut
    tip = path[-1]
    s.sweep([(x + 2.4 * k, y - 7 * k, z), tip], 0.35 * k, latch, sides=6)                                  # latch


def chain_hoist(s, sk, tx, fb, z, gfw, hook_y, body, attach_key, attach):
    """Electric chain hoist under a running flange at height fb (flange bottom), centred on z."""
    # trolley: side plates both sides of the flange, flanged wheels on the flange top, cross bolts
    for uz in (-1, 1):
        zz = z + uz * (gfw / 2 + 3)
        s.rbox((tx, fb - 9, zz), (40, 26, 2.2), 1.0, body, seg=2)
        for wx in (tx - 12, tx + 12):
            s.lathe((wx, fb + 5.2, zz - uz * 1.2), (wx, fb + 5.2, zz - uz * 5.5),
                    [(0, 3.6), (0.2, 4.6), (0.35, 4.6), (0.4, 3.8), (1, 3.8)], "steel_machined", sides=18)
            s.bolt((wx, fb + 5.2, zz + uz * 1.1), (0, 0, uz), 1.0, BOLT)
    for bx in (tx - 15, tx + 15):
        s.cylinder((bx, fb - 18, z - gfw / 2 - 5), (bx, fb - 18, z + gfw / 2 + 5), 1.1, BOLT, 8)
        for uz in (-1, 1):
            s.bolt((bx, fb - 18, z + uz * (gfw / 2 + 4.2)), (0, 0, uz), 1.3, BOLT)
    # hoist body: gear case, motor, end cover, electrics
    hb = fb - 26                                              # top of the hoist body (suspension)
    s.rbox((tx, hb - 4, z), (14, 8, 10), 2.0, body, seg=2)   # suspension lug
    s.rbox((tx, hb - 22, z), (30, 30, 26), 5.0, body, seg=3)  # main gear case
    finned_motor(s, (tx + 14, hb - 22, z), (tx + 42, hb - 22, z), 12.0, "powder_anthracite", "powder_black")
    nameplate(s, (tx - 2, hb - 20, z - 13.2), (0, 0, -1), (1, 0, 0), 11, 7)
    warning_label(s, (tx - 2, hb - 30, z + 13.2), (0, 0, 1), (1, 0, 0), 6)
    s.lathe((tx - 14, hb - 22, z), (tx - 24, hb - 22, z), [(0, 12.5), (0.6, 12.5), (0.85, 11.0), (1, 7.0)], body, sides=28)
    for k in range(6):                                        # cover screws
        ang = 2 * math.pi * k / 6
        s.bolt((tx - 24.2, hb - 22 + 9 * math.cos(ang), z + 9 * math.sin(ang)), (-1, 0, 0), 0.7, BOLT)
    s.rbox((tx + 2, hb - 22, z + 17), (24, 22, 7), 2.0, "powder_lightgrey", seg=2)   # electrical box
    for dx_ in (-9, 9):                                       # cover screws
        for dy_ in (-8, 8):
            s.bolt((tx + 2 + dx_, hb - 22 + dy_, z + 20.6), (0, 0, 1), 0.45, "stainless")
    for k, gx in enumerate((tx - 6, tx + 2)):                 # cable glands, cables up to the festoon
        s.lathe((gx, hb - 11, z + 17), (gx, hb - 8, z + 17), [(0, 1.6), (0.5, 1.6), (1, 1.1)], "plastic_black", sides=12)
        s.sweep([(gx, hb - 8, z + 17), (gx, hb + 2, z + 19 + k), (gx + 4, fb + 14, z + 21 + 2 * k)], 0.6, "cable_pvc", sides=8)
    # chain outlet, chain, chain bag
    s.lathe((tx + 4, hb - 37, z), (tx + 4, hb - 43, z), [(0, 4.5), (0.6, 3.2), (1, 2.6)], "powder_dark", sides=16)
    cx = tx + 4
    bot_block = hook_y + 34
    s.sweep([(tx - 8, hb - 36, z - 9), (tx - 8, hb - 40, z - 12)], 0.6, BOLT, sides=6)                  # bag hook
    s.lathe((tx - 8, hb - 40, z - 14), (tx - 8, hb - 70, z - 14),
            [(0, 5), (0.15, 7.5), (0.6, 8.5), (0.9, 7.5), (1, 4)], "plastic_black", sides=18)               # chain bag
    # bottom block: cheek plates around a round sheave guard, cross-head, swivel, hook
    yb = bot_block + 6
    sk.lathe((cx, yb, z - 5), (cx, yb, z + 5), [(0, 8), (0.1, 9), (0.9, 9), (1, 8)], body, sides=24)
    for uz in (-1, 1):
        sk.rbox((cx, yb - 2, z + uz * 5.8), (14, 22, 1.6), 0.6, "powder_dark", seg=1)
        sk.bolt((cx, yb, z + uz * 6.6), (0, 0, uz), 1.0, BOLT)
    sk.rbox((cx, yb - 12, z), (13, 4, 13), 1.2, "powder_dark", seg=1)
    hook(sk, cx, yb - 14, z, 1.0)
    attach[attach_key] = (cx, yb - 28, z)
    return cx, hb


def pendant(s, x_top, y_top, z, attach, key):
    """Pendant control box on a sagging cable from the hoist."""
    px, pz = x_top - 12, z + 30
    path = [(x_top, y_top, z + 20)]
    for i in range(1, 10):
        t = i / 9
        path.append((x_top + (px - x_top) * t, y_top + (128 - y_top) * t, z + 20 + (pz - z - 20) * t + 3 * math.sin(math.pi * t)))
    s.sweep(path, 0.55, "cable_pvc", sides=8)
    s.rbox((px, 113, pz), (9, 28, 8), 2.5, "powder_yellow", seg=2)
    for j in range(5):
        s.lathe((px, 121 - j * 4.4, pz + 4), (px, 121 - j * 4.4, pz + 5.2), [(0, 1.6), (0.7, 1.6), (1, 1.2)],
                "plastic_black", sides=12)
    s.lathe((px, 127, pz), (px, 131, pz), [(0, 1.5), (0.4, 1.5), (0.5, 3.4), (1, 3.0)], "powder_red", sides=16)
    attach[key] = (px, 113, pz)


def festoon(s, x0, x1, y, z, park_x, run_x, loops=6):
    """C-track on brackets with cable trolleys parked near x0 and one at run_x, flat cables draped between."""
    s.rbox(((x0 + x1) / 2, y, z), (x1 - x0, 5.0, 4.0), 0.6, "galvanised", seg=1)
    xs = [park_x + k * 16 for k in range(loops - 1)] + [run_x]
    xs.sort()
    for x in xs:
        s.rbox((x, y - 5, z), (7, 6, 5), 1.0, "steel_dark", seg=1)
        for wx in (x - 2.2, x + 2.2):
            s.lathe((wx, y - 1, z - 2.6), (wx, y - 1, z + 2.6), [(0, 1.1), (1, 1.1)], "plastic_black", sides=10)
    for a, b in zip(xs, xs[1:]):
        sag = min(70.0, 0.18 * (b - a) + 14)
        for dz in (-1.1, 1.1):                                   # a flat cable as two touching round cores
            path = []
            for j in range(17):
                t = j / 16
                path.append((a + (b - a) * t, y - 8 - sag * math.sin(math.pi * t) ** 0.9, z + dz))
            s.sweep(path, 1.0, "cable_pvc", sides=8)


LINK, PITCH = 3.6, 3.6 * 0.78


def rig_xform(stage, path, **limits):
    """An Xform with an identity translate (the rig writes deltas from the built pose) and its frt:rig:* values."""
    x = UsdGeom.Xform.Define(stage, path)
    x.ClearXformOpOrder()
    x.AddTranslateOp().Set(Gf.Vec3d(0, 0, 0))
    for k, v in limits.items():
        if isinstance(v, str):
            x.GetPrim().CreateAttribute(f"frt:rig:{k}", Sdf.ValueTypeNames.String).Set(v)
        elif isinstance(v, (list, tuple)):
            x.GetPrim().CreateAttribute(f"frt:rig:{k}", Sdf.ValueTypeNames.DoubleArray).Set(Vt.DoubleArray(list(v)))
        else:
            x.GetPrim().CreateAttribute(f"frt:rig:{k}", Sdf.ValueTypeNames.Double).Set(float(v))
    return x


def chain_links(top, bot):
    """Link centres (y) from the outlet at top down to bot, the way chain() lays them."""
    ys, y = [], top
    while y - LINK > bot:
        ys.append(y - LINK / 2)
        y -= PITCH
    return ys


def lay_chain(pi, bot, time=None):
    """Re-lay a chain instancer down to height bot (crane-local)."""
    p = pi.GetPrim()
    x, z, top = (p.GetAttribute(f"frt:rig:{k}").Get() for k in ("x", "z", "top"))
    ys = chain_links(top, bot)
    t = time if time is not None else _default_time()
    pi.GetPositionsAttr().Set(Vt.Vec3fArray([Gf.Vec3f(x, y, z) for y in ys]), t)
    pi.GetProtoIndicesAttr().Set(Vt.IntArray([k % 2 for k in range(len(ys))]), t)


def _default_time():
    from pxr import Usd
    return Usd.TimeCode.Default()


def chain_instancer(stage, path, lib, x, z, top, bot, mat="steel_blued"):
    """Load chain as a PointInstancer of two link prototypes (the link planes alternate)."""
    pi = UsdGeom.PointInstancer.Define(stage, path)
    UsdGeom.Scope.Define(stage, f"{path}/Prototypes")
    protos = []
    for k, ax in enumerate(((1, 0, 0), (0, 0, 1))):
        pp = f"{path}/Prototypes/Link_{k}"
        UsdGeom.Xform.Define(stage, pp)
        sp = new_shapes()
        sp.torus((0, 0, 0), ax, 0.85, 0.55, mat, seg=14, sides=6, stretch=LINK - 2.2, up=(0, 1, 0))
        shapes_to_stage(stage, pp, sp, lib)
        protos.append(pp)
    pi.CreatePrototypesRel().SetTargets([Sdf.Path(q) for q in protos])
    for k, v in (("x", x), ("z", z), ("top", top)):
        pi.GetPrim().CreateAttribute(f"frt:rig:{k}", Sdf.ValueTypeNames.Double).Set(float(v))
    lay_chain(pi, bot)
    return pi


def festoon_xs(park, run, n):
    """Festoon trolley positions for the running trolley at run: spread evenly, never closer than a trolley."""
    gap = max(7.0, (run - park) / (n - 1))
    return [park + gap * k for k in range(n - 1)] + [max(run, park + 7.0 * (n - 1))]


def festoon_rig(stage, path, lib, track, x0, x1, y, z, park_x, run_x, hoist, loops=6):
    """Festoon: C-track (into the bridge shapes `track`), then one Xform per cable trolley and per cable loop so
    the rig can move the trolleys and stretch each loop about its parked end."""
    track.rbox(((x0 + x1) / 2, y, z), (x1 - x0, 5.0, 4.0), 0.6, "galvanised", seg=1)
    xs = festoon_xs(park_x, run_x, loops)
    rig_xform(stage, path, park=park_x, rest=xs, hoist=hoist)
    for k, x in enumerate(xs):
        st = new_shapes()
        st.rbox((x, y - 5, z), (7, 6, 5), 1.0, "steel_dark", seg=1)
        for wx in (x - 2.2, x + 2.2):
            st.lathe((wx, y - 1, z - 2.6), (wx, y - 1, z + 2.6), [(0, 1.1), (1, 1.1)], "plastic_black", sides=10)
        rig_xform(stage, f"{path}/Trolley_{k}")
        shapes_to_stage(stage, f"{path}/Trolley_{k}", st, lib)
    for k, (a, b) in enumerate(zip(xs, xs[1:])):
        sl = new_shapes()
        sag = min(70.0, 0.18 * (b - a) + 14)
        for dz in (-1.1, 1.1):                                   # a flat cable as two touching round cores
            sl.sweep([(a + (b - a) * j / 16, y - 8 - sag * math.sin(math.pi * j / 16) ** 0.9, z + dz) for j in range(17)],
                     1.0, "cable_pvc", sides=8)
        lx = rig_xform(stage, f"{path}/Loop_{k}")
        lx.AddTranslateOp(opSuffix="pivot").Set(Gf.Vec3d(a, 0, 0))
        lx.AddScaleOp().Set(Gf.Vec3f(1, 1, 1))
        lx.AddTranslateOp(opSuffix="pivot", isInverseOp=True)
        shapes_to_stage(stage, f"{path}/Loop_{k}", sl, lib)


class WorkstationCrane(Item):
    def footprint(self, p):
        return p["span"] + 50.0, p["runway"]

    def build(self, stage, path, p, lib):
        S, Rw, H = p["span"], p["runway"], p["height"]
        yel, col = p["colour"], p["column_colour"]
        s = new_shapes()
        heavy = S > 900 or Rw > 1800
        a = 28.0 if heavy else 24.0                               # column section
        rw, rd = (26.0, 44.0) if heavy else (22.0, 36.0)          # runway box: width, depth
        nbay = max(1, math.ceil(Rw / p["column_spacing"]))
        zs = [-Rw / 2 + a / 2 + 4 + k * (Rw - a - 8) / nbay for k in range(nbay + 1)]
        xs = (-S / 2, S / 2)
        splice = H * 0.45
        # ------------------------------------------------------------------ columns
        for x in xs:
            for z in zs:
                s.rbox((x, 2.5, z), (70, 5, 70), 1.0, "concrete", seg=1)                         # grout plinth
                s.rbox((x, 6.5, z), (56, 3, 56), 0.6, "steel_dark", seg=1)                       # base plate
                for dx in (-21, 21):
                    for dz in (-21, 21):
                        s.cylinder((x + dx, 8, z + dz), (x + dx, 14, z + dz), 1.0, BOLT, 10)    # anchor
                        s.bolt((x + dx, 8, z + dz), (0, 1, 0), 1.6, BOLT)                        # nut + washer
                for ux, uz in ((1, 0), (-1, 0), (0, 1), (0, -1)):                                # gussets
                    gx, gz = x + ux * (a / 2 + 7), z + uz * (a / 2 + 7)
                    axd = _n((ux, 1.0, uz))
                    side = (uz, 0.0, -ux)
                    s.rbox((gx - ux * 1.5, 14.5, gz - uz * 1.5), (19, 1.4, 12), 0.4, col,
                           axes=(axd, side, _n(_x(axd, side))), seg=1)
                bands = [(18 + k * 14, 32 + k * 14, "floor_yellow" if k % 2 == 0 else "powder_black") for k in range(6)]
                pieces = [(8.0, 18.0, col)] + bands + [(bands[-1][1], splice - 1.4, col), (splice + 1.4, H - 2.4, col)]
                for y0, y1, m_ in pieces:                                                        # column: flush pieces
                    s.rprism((x, y0, z), (x, y1, z), a, a, 2.5, m_, up=(1, 0, 0), seg=3)
                for y in (splice - 1.4, splice + 1.4):                                           # bolted splice
                    s.rbox((x, y, z), (a + 14, 2.6, a + 14), 0.5, col, seg=1)
                for ux, uz in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                    bx, bz = x + ux * (a / 2 + 4), z + uz * (a / 2 + 4)
                    s.bolt((bx, splice + 2.7, bz), (0, 1, 0), 1.4, BOLT)
                    s.bolt((bx, splice - 2.7, bz), (0, -1, 0), 1.4, BOLT, length=1.2)           # nut, thread showing
                s.rbox((x, H - 1.2, z), (a + 8, 2.4, a + 8), 0.5, col, seg=1)                     # cap plate
                weld_ring(s, (x, 8.4, z), (0, 1, 0), a, a, 2.5, col)                             # base weld
                weld_ring(s, (x, H - 2.8, z), (0, 1, 0), a, a, 2.5, col)                         # cap weld
                for y_ in (splice - 3.0, splice + 3.0):
                    weld_ring(s, (x, y_, z), (0, 1, 0), a, a, 2.5, col, bead=0.4)
        # ------------------------------------------------------------------ runways, tie beams, braces
        rail_top = H + rd + 4.0
        for x in xs:
            s.rprism((x, H + rd / 2, -Rw / 2), (x, H + rd / 2, Rw / 2), rw, rd, 2.0, col)          # runway box
            s.rbox((x, H + rd + 2, 0), (5, 4, Rw - 8), 0.6, "steel_dark", seg=1)                 # running rail
            s.rbox((x, H + rd + 4.05, 0), (2.6, 0.12, Rw - 12), 0.05, "steel_machined", seg=1)  # worn wheel track
            for z in zs:
                s.rbox((x, rail_top + 9, z), (2.4, 16, 14), 0.8, col, seg=1)                     # lifting lug
                s.torus((x, rail_top + 12, z), (1, 0, 0), 3.0, 1.0, "galvanised", seg=14, sides=6)
            for zend, uz in ((-Rw / 2, 1), (Rw / 2, -1)):                                        # end caps + stops
                s.rbox((x, H + rd / 2, zend + uz * 0.9), (rw + 4, rd + 4, 1.8), 0.4, col, seg=1)
                for dy in (-rd / 3, rd / 3):
                    for dxx in (-rw / 3, rw / 3):
                        s.bolt((x + dxx, H + rd / 2 + dy, zend - uz * 0.1), (0, 0, -uz), 1.1, BOLT)
                s.lathe((x, rail_top + 10, zend + uz * 2), (x, rail_top + 10, zend + uz * 13),
                        [(0, 7), (0.7, 7), (0.85, 6.4), (1, 4.5)], "rubber_black", sides=18)
        tie_y = H * 0.82
        for z in (zs if heavy else (zs[0], zs[-1])):
            member(s, (xs[0] + a / 2, tie_y, z), (xs[1] - a / 2, tie_y, z), 16, 24, 1.5, col)       # tie beam
            for x, ux in ((xs[0], 1), (xs[1], -1)):
                lo = (x + ux * a / 2, H * 0.40, z)
                hi = (x + ux * min(S * 0.28, 250), tie_y - 12, z)
                member(s, lo, hi, 11, 11, 1.2, col)                                                  # lateral brace
                end_plate_bolted(s, lo, (ux, 0.0, 0.0), 16, 14, col)
                end_plate_bolted(s, (hi[0], hi[1] - 1, z), (0.0, 1.0, 0.0), 14, 16, col)
        for x in xs:                                                                                 # runway braces
            for z in zs:
                for uz in (-1, 1):
                    if (uz < 0 and z == zs[0]) or (uz > 0 and z == zs[-1]):
                        continue
                    reach = min(260.0, (Rw - a) / nbay * 0.4)
                    lo, hi = (x, H * 0.45, z + uz * a / 2), (x, H - 1, z + uz * reach)
                    member(s, lo, hi, 10, 10, 1.2, col)
                    end_plate_bolted(s, lo, (0.0, 0.0, uz), 14, 14, col)
        # ------------------------------------------------------------------ bridges
        n = max(1, int(p["bridges"]))
        if p["bridge_positions"].strip():
            fr = [min(1.0, max(0.0, float(v))) for v in p["bridge_positions"].split(",") if v.strip()][:n] or [0.5]
        elif n == 1:
            fr = [p["bridge"]]
        else:
            fr = [(k + 0.5) / n for k in range(n)]
        dbl = int(p["girders"]) >= 2
        gd, gw = (60.0, 30.0) if (heavy or dbl) else (46.0, 24.0)
        ggap = 95.0 if dbl else 0.0
        et = 160.0 + ggap
        rwh = 10.0
        attach = {}
        nh = max(1, int(p["hoists"]))
        shapes_to_stage(stage, path, s, lib)                                          # structure
        z_lo, z_hi = -Rw / 2 + 30 + et / 2, Rw / 2 - 30 - et / 2
        for bi, f in enumerate(fr):
            bz = -Rw / 2 + 30 + et / 2 + (Rw - 60 - et) * f
            bp = f"{path}/Bridge_{bi + 1}"
            rig_xform(stage, bp, z_min=z_lo, z_max=z_hi, z_rest=bz, key=f"bridge{bi + 1}" if n > 1 else "bridge")
            s = new_shapes()                                                              # bridge parts
            wy = rail_top + rwh
            t_lo, t_hi = wy - 13, wy + 15
            for x in xs:                                                                          # end trucks
                for ux in (-1, 1):                                                                # side plates
                    s.rprism((x + ux * 9, (t_lo + t_hi) / 2, bz - et / 2), (x + ux * 9, (t_lo + t_hi) / 2, bz + et / 2),
                             2.2, t_hi - t_lo, 0.8, yel, up=(0, 1, 0), seg=2)
                s.rprism((x, t_hi - 1.2, bz - et / 2), (x, t_hi - 1.2, bz + et / 2), 20, 2.4, 0.6, yel, seg=1)   # cover
                for zend in (bz - et / 2 + 1.2, bz + et / 2 - 1.2):                               # end plates
                    s.rbox((x, (t_lo + t_hi) / 2, zend), (20, t_hi - t_lo - 2, 2.4), 0.5, yel, seg=1)
                    weld_ring(s, (x, (t_lo + t_hi) / 2, zend), (0, 0, 1), 20, t_hi - t_lo - 2, 0.5, yel, bead=0.35)
                for wz in (bz - et / 2 + 28, bz + et / 2 - 28):
                    s.lathe((x - 7.5, wy, wz), (x + 7.5, wy, wz),                                 # double-flanged wheel
                            [(0, rwh + 1.6), (0.12, rwh + 1.6), (0.18, rwh), (0.82, rwh), (0.88, rwh + 1.6), (1, rwh + 1.6)],
                            "steel_machined", sides=28)
                    s.lathe((x - 8, wy, wz), (x + 8, wy, wz), [(0, 3.2), (1, 3.2)], "steel_dark", sides=14)   # hub
                    for ux in (-1, 1):                                                            # bearing housings
                        s.lathe((x + ux * 10.1, wy, wz), (x + ux * 13.5, wy, wz),
                                [(0, 6.5), (0.3, 6.5), (0.7, 5.5), (1, 3.8)], "powder_anthracite", sides=20)
                        for k in range(4):
                            ang = 2 * math.pi * k / 4 + math.pi / 4
                            s.bolt((x + ux * 10.1, wy + 5.3 * math.cos(ang), wz + 5.3 * math.sin(ang)), (ux, 0, 0), 0.8, BOLT)
                for zend, uz in ((bz - et / 2, -1), (bz + et / 2, 1)):
                    s.lathe((x, wy, zend), (x, wy, zend + uz * 9), [(0, 5.5), (0.6, 5.5), (0.8, 5.0), (1, 3.5)],
                            "rubber_black", sides=16)
            dx = xs[1] + 13                                                                       # drive
            finned_motor(s, (dx + 14, wy, bz - et / 2 + 28), (dx + 40, wy, bz - et / 2 + 28), 9.0, "powder_anthracite",
                         "powder_black")
            s.rbox((dx + 7, wy, bz - et / 2 + 28), (14, 26, 26), 3.0, "powder_anthracite", seg=2)        # gearbox
            nameplate(s, (dx + 7, wy, bz - et / 2 + 41.2), (0, 0, 1), (1, 0, 0), 9, 5)
            s.rbox((dx + 22, wy + 13, bz - et / 2 + 28), (12, 6, 14), 1.5, "powder_dark", seg=1)
            gy = t_hi + gd / 2                                                                    # girders
            gz = (bz - ggap / 2, bz + ggap / 2) if dbl else (bz,)
            for z in gz:
                s.rprism((-S / 2 - 25, gy, z), (S / 2 + 25, gy, z), gw, gd, 2.5, yel)
                s.rbox((0, gy - gd / 2 - 1.3, z), (S + 30, 2.6, gw + 14), 0.6, yel, seg=1)        # running flange
                for x in xs:                                                                      # bolted connection
                    s.rbox((x, t_hi + 1.2, z), (30, 2.4, gw + 10), 0.6, "steel_dark", seg=1)
                    s.bolts_grid((x, t_hi + 2.4, z), (1, 0, 0), (0, 0, 1), 2, 2, 20, gw + 4, (0, 1, 0), 1.3, BOLT)
            cap = "".join(ch for ch in str(p["capacity"]) if ch.isdigit())
            low = str(p["capacity"]).lower().strip()
            unit = "LB" if "lb" in low else ("T" if low.endswith("t") else "")
            cap = (cap + unit) if cap else ""
            if cap:                                                                               # capacity sign
                zf = gz[-1] + gw / 2 + 0.2
                cap = cap.replace("LB", " LB").replace("T", " T") if cap[-1:].isalpha() else cap
                wsign = max(44.0, 14 * 6 / 7 * len(cap) + 14)
                s.rbox((0, gy, zf + 0.5), (wsign, 26, 1.0), 0.8, "floor_white", seg=1)
                sign_text(s, cap, (-wsign / 2 + 7, gy + 7, zf + 1.0), (1, 0, 0), (0, -1, 0), 14.0, "plastic_black", (0, 0, 1))
                for k in range(4):                                                              # sign screws
                    s.bolt((-wsign / 2 + 3 + (k % 2) * (wsign - 6), gy + (9 if k < 2 else -9), zf + 1.0), (0, 0, 1), 0.5,
                           "stainless")
            fz = gz[0] - gw / 2 - 8                                                               # festoon
            fy = gy + gd / 2 - 4
            for bxx in range(int(-S / 2), int(S / 2) + 1, 150):
                s.rbox((bxx, fy + 2, fz + 4), (3, 8, 10), 0.5, "galvanised", seg=1)
            if not dbl:
                hx = [(-S / 2 + 60 + (S - 120) * ((k + 0.5) / nh if nh > 1 else p["trolley"])) for k in range(nh)]
                fb = gy - gd / 2 - 2.6
                hk = max(30.0, min(H - 120.0, p["hook_height"]))
                for hi_, tx in enumerate(hx):
                    key = (f"hook{bi + 1}_{hi_ + 1}" if nh > 1 else f"hook{bi + 1}") if n > 1 else \
                          (f"hook{hi_ + 1}" if nh > 1 else "hook")
                    hp = f"{bp}/Hoist_{hi_ + 1}"
                    sh, sk = new_shapes(), new_shapes()
                    cx, hb = chain_hoist(sh, sk, tx, fb, bz, gw + 14, hk, "powder_orange", key, attach)
                    pendant(sh, cx, hb - 30, bz, attach, key.replace("hook", "pendant"))
                    hmax = min(H - 120.0, hb - 43 - 50)
                    rig_xform(stage, hp, x_min=-S / 2 + 60, x_max=S / 2 - 60, x_rest=tx, kind="chain_hoist",
                              key=key, pendant=key.replace("hook", "pendant"))
                    shapes_to_stage(stage, hp, sh, lib)
                    rig_xform(stage, f"{hp}/Block", hook_min=30.0, hook_max=max(30.0, hmax), hook_rest=hk)
                    shapes_to_stage(stage, f"{hp}/Block", sk, lib)
                    chain_instancer(stage, f"{hp}/Chain", lib, cx, bz, hb - 43, hk + 42)
                festoon_rig(stage, f"{bp}/Festoon", lib, s, -S / 2 - 15, S / 2 + 15, fy, fz, -S / 2 + 10, hx[-1],
                            f"Hoist_{nh}")
            else:                                                                                 # crab + winch
                top_g = gy + gd / 2
                tx = -S / 2 + 70 + (S - 140) * p["trolley"]
                for z in gz:
                    s.rbox((0, top_g + 2, z), (S - 40, 4, 5), 0.6, "steel_dark", seg=1)
                hp = f"{bp}/Hoist_1"
                sh, sk, sr = new_shapes(), new_shapes(), new_shapes()
                cl = 120.0
                sh.rbox((tx, top_g + 20, bz), (cl, 14, ggap + 34), 2.5, yel, seg=2)
                for wx in (tx - cl / 2 + 20, tx + cl / 2 - 20):
                    for z in gz:
                        sh.lathe((wx, top_g + 11, z - 5), (wx, top_g + 11, z + 5),
                                 [(0, 6), (0.2, 7.5), (0.3, 7.5), (0.35, 6), (1, 6)], "steel_machined", sides=20)
                drum = [(0, 19), (0.04, 19)] + [v for k in range(16) for v in ((0.06 + k * 0.055, 17.8), (0.085 + k * 0.055, 18.4))] + [(0.95, 19), (1, 19)]
                sh.lathe((tx - 44, top_g + 52, bz), (tx + 30, top_g + 52, bz), drum, "steel_machined", sides=28)
                sh.rbox((tx + 40, top_g + 50, bz), (20, 46, 48), 4.0, yel, seg=2)                 # gearbox
                finned_motor(sh, (tx + 50, top_g + 50, bz), (tx + 84, top_g + 50, bz), 15.0, yel, "powder_dark")
                sh.rbox((tx - cl / 2 + 20, top_g + 46, bz + ggap / 2 + 10), (36, 40, 12), 2.0, "powder_dark", seg=2)
                hk = max(30.0, min(H - 120.0, p["hook_height"]))
                bot = hk + 62
                rtop = top_g + 34
                for ox, oz in ((-7, -9), (-7, 9), (7, -9), (7, 9)):
                    sr.cylinder((tx + ox, rtop, bz + oz), (tx + ox, bot, bz + oz), 0.9, "galvanised", 8)
                sk.rbox((tx, bot - 18, bz), (32, 36, 30), 4.0, yel, seg=2)                          # hook block
                for oz in (-9, 9):
                    sk.lathe((tx - 17, bot - 12, bz + oz), (tx + 17, bot - 12, bz + oz), [(0, 11), (1, 11)], "powder_dark", sides=24)
                hook(sk, tx, bot - 38, bz, 1.6)
                key = f"hook{bi + 1}" if n > 1 else "hook"
                attach[key] = (tx, bot - 60, bz)
                pendant(sh, tx, top_g, bz, attach, key.replace("hook", "pendant"))
                rig_xform(stage, hp, x_min=-S / 2 + 70, x_max=S / 2 - 70, x_rest=tx, kind="crab", key=key,
                          pendant=key.replace("hook", "pendant"))
                shapes_to_stage(stage, hp, sh, lib)
                rig_xform(stage, f"{hp}/Block", hook_min=30.0, hook_max=max(30.0, min(H - 120.0, top_g - 70)), hook_rest=hk)
                shapes_to_stage(stage, f"{hp}/Block", sk, lib)
                rx = rig_xform(stage, f"{hp}/Ropes", top=rtop, bot_rest=bot)                       # ropes stretch
                rx.AddTranslateOp(opSuffix="pivot").Set(Gf.Vec3d(0, rtop, 0))
                rx.AddScaleOp().Set(Gf.Vec3f(1, 1, 1))
                rx.AddTranslateOp(opSuffix="pivot", isInverseOp=True)
                shapes_to_stage(stage, f"{hp}/Ropes", sr, lib)
                festoon_rig(stage, f"{bp}/Festoon", lib, s, -S / 2 - 15, S / 2 + 15, fy, fz, -S / 2 + 10, tx, "Hoist_1")
            attach[f"bridge{bi + 1}" if n > 1 else "bridge"] = (0, gy, bz)
            shapes_to_stage(stage, bp, s, lib)
        return attach


register(WorkstationCrane("bridge_crane", "crane_bridge", "Freestanding bridge (workstation) crane", {
    "span": Param(800.0, "distance between runways", 300, 3000, "cm"),
    "runway": Param(1200.0, "runway length", 300, 6000, "cm"),
    "height": Param(550.0, "runway beam underside height", 300, 1200, "cm"),
    "column_spacing": Param(600.0, "max distance between runway columns", 300, 1200, "cm"),
    "bridge": Param(0.5, "bridge position along the runway (0-1), one bridge", 0, 1),
    "bridges": Param(1, "bridges on the runways", 1, 8),
    "bridge_positions": Param("", "bridge positions along the runway, fractions 0-1, comma separated"),
    "girders": Param(1, "girders per bridge (1: underhung chain hoists, 2: top-running crab)", 1, 2),
    "hoists": Param(1, "chain hoists per bridge (single girder)", 1, 4),
    "trolley": Param(0.5, "trolley position along the bridge (0-1), one hoist", 0, 1),
    "hook_height": Param(200.0, "hook height above the floor", 30, 1200, "cm"),
    "colour": Param("powder_yellow", "bridge, trucks, hoist frame"),
    "column_colour": Param("powder_yellow", "columns, tie beams and runways"),
    "capacity": Param("", "rated load as on the drawing, e.g. 2000lb (shown on the sign)"),
}))
