"""Detail parts shared by the catalog items: the small things that make equipment read as real - bent tube (not
kinked polylines), weld beads, ribbed grips with end plugs, chamfered hex fittings, crimped hydraulic hoses, cable
glands, an enclosure with a separate lid, gasket seam and recessed screws, shrouded push buttons, an emergency
stop, pilot lamps with lit lenses, a key switch, legend labels, a coil spring.

Face-mounted parts take a frame: p (on the surface), r (right), u (up), n (out of the surface).
"""
import math

from .items_crane import _n, _x, sign_text, warning_label
from .items_flow import poly, round_hole

BOLT = "bolt_zinc"


def at(p, r, u, n, a=0.0, b=0.0, c=0.0):
    """p + a r + b u + c n."""
    return tuple(pi + a * ri + b * ui + c * ni for pi, ri, ui, ni in zip(p, r, u, n))


def bend(points, radius, seg=6):
    """A polyline with every inner corner replaced by an arc-like bend (tube bending radius `radius`)."""
    pts = [tuple(map(float, q)) for q in points]
    if len(pts) < 3:
        return pts
    out = [pts[0]]
    for a, b, c in zip(pts, pts[1:], pts[2:]):
        da = _n(tuple(x - y for x, y in zip(a, b)))
        dc = _n(tuple(x - y for x, y in zip(c, b)))
        cosang = max(-1.0, min(1.0, sum(x * y for x, y in zip(da, dc))))
        half = math.acos(cosang) / 2
        if half < 1e-3 or half > math.pi / 2 - 1e-3:
            out.append(b)
            continue
        t = min(radius / math.tan(half), 0.45 * math.dist(a, b), 0.45 * math.dist(b, c))
        p1 = tuple(bi + di * t for bi, di in zip(b, da))
        p2 = tuple(bi + di * t for bi, di in zip(b, dc))
        for k in range(seg + 1):                                   # quadratic Bezier p1 -> b -> p2
            s_ = k / seg
            out.append(tuple((1 - s_) ** 2 * x + 2 * (1 - s_) * s_ * y + s_ ** 2 * z for x, y, z in zip(p1, b, p2)))
    out.append(pts[-1])
    return out


def weld_bead(s, c, axis, r, mat, bead=0.28):
    """Fillet weld around a tube of radius r where it meets a plate."""
    s.torus(c, axis, r + bead * 0.6, bead, mat, seg=24, sides=6)


def grip(s, a, b, r, mat="grip_rubber", ribs=9):
    """Moulded rubber grip over a tube end a -> b: flange at a, ribbed body, rounded closed end."""
    prof = [(0.0, r * 1.05), (0.03, r * 1.45), (0.09, r * 1.45), (0.12, r * 1.18)]
    for k in range(ribs):
        t0 = 0.16 + 0.74 * k / ribs
        prof += [(t0, r * 1.18), (t0 + 0.74 / ribs * 0.5, r * 1.26)]
    prof += [(0.92, r * 1.2), (0.97, r * 1.0), (1.0, r * 0.45)]
    s.lathe(a, b, prof, mat, sides=20)


def tube_plug(s, end, axis, r, mat="plastic_black"):
    """Push-in plastic end plug on an open tube end."""
    tip = tuple(e + a * 0.6 * r for e, a in zip(end, axis))
    s.lathe(end, tip, [(0, r * 1.02), (0.5, r * 1.02), (1, r * 0.7)], mat, sides=16)


def hex_nut(s, a, axis, af, length, mat=BOLT):
    """Chamfered hex (af = across flats) from a along axis."""
    R = af / math.sqrt(3)
    b = tuple(p + d * length for p, d in zip(a, axis))
    s.lathe(a, b, [(0, R * 0.86), (0.15, R), (0.85, R), (1, R * 0.86)], mat, sides=6)
    return b


def fitting(s, a, axis, d, mat="brass"):
    """Hydraulic adapter screwed into a port at a: hex body, thread step, 37-degree cone nose."""
    b = hex_nut(s, a, axis, d * 1.25, d * 0.55, mat)
    c = tuple(p + q * d * 0.35 for p, q in zip(b, axis))
    s.lathe(b, c, [(0, d * 0.42), (1, d * 0.42)], mat, sides=14)
    e = tuple(p + q * d * 0.3 for p, q in zip(c, axis))
    s.lathe(c, e, [(0, d * 0.4), (1, d * 0.28)], mat, sides=14)
    return e


def hose(s, path, r, mat="hose_rubber", bend_r=None):
    """Hydraulic hose along path (bent smoothly) with crimped ferrules and swivel nuts at both ends."""
    pts = bend(path, bend_r or 5 * r)
    for end, nxt in ((pts[0], pts[1]), (pts[-1], pts[-2])):
        ax = _n(tuple(q - p for p, q in zip(end, nxt)))
        nut_end = hex_nut(s, end, ax, r * 2.6, r * 1.4, BOLT)
        f1 = tuple(p + q * r * 3.2 for p, q in zip(nut_end, ax))
        crimp = [(0, r * 1.3)] + [v for k in range(4) for v in ((0.1 + k * 0.2, r * 1.38), (0.2 + k * 0.2, r * 1.3))] + [(1, r * 1.15)]
        s.lathe(nut_end, f1, crimp, "steel_machined", sides=16)
    s.sweep(pts, r, mat, sides=12)                                  # (its ends sit inside the ferrules)
    return pts


def gland(s, p, axis, d, cable_path=None, mat="plastic_black", cable_r=None):
    """Cable gland: locknut, hex body, domed seal nut, then the cable leaving along cable_path."""
    b = hex_nut(s, p, axis, d * 1.25, d * 0.35, mat)
    c = tuple(q + a * d * 0.55 for q, a in zip(b, axis))
    s.lathe(b, c, [(0, d * 0.62), (0.4, d * 0.62), (1, d * 0.42)], mat, sides=6)
    e = tuple(q + a * d * 0.35 for q, a in zip(c, axis))
    s.lathe(c, e, [(0, d * 0.48), (1, d * 0.36)], mat, sides=16)
    if cable_path:
        s.sweep(bend([e] + list(cable_path), (cable_r or d * 0.3) * 6), cable_r or d * 0.3, "cable_pvc", sides=10)
    return e


def enclosure(s, p, r, u, n, w, h, d, body="abs_lightgrey", screws="stainless"):
    """Moulded enclosure whose back is at p: base box, separate lid with a dark gasket seam, four recessed lid screws
    and mounting lugs. Returns the centre of the lid face."""
    db, dl = d * 0.72, d * 0.26
    s.rbox(at(p, r, u, n, 0, 0, db / 2), (w, h, db), min(1.2, w * 0.08), body, axes=(r, u, n), seg=3)
    s.rbox(at(p, r, u, n, 0, 0, db + 0.12), (w - 0.5, h - 0.5, 0.24), 0.4, "hole", axes=(r, u, n), seg=1)  # seam
    s.rbox(at(p, r, u, n, 0, 0, db + 0.24 + dl / 2), (w + 0.15, h + 0.15, dl), min(1.3, w * 0.09), body,
           axes=(r, u, n), seg=3)                                                                          # lid
    face = at(p, r, u, n, 0, 0, db + 0.24 + dl)
    for su in (-1, 1):
        for sv in (-1, 1):
            q = at(face, r, u, n, su * (w / 2 - 1.3), sv * (h / 2 - 1.3), 0.0)
            round_hole(s, q, r, u, n, 0.75, n=14)
            s.lathe(at(q, r, u, n, 0, 0, 0.0), at(q, r, u, n, 0, 0, 0.14), [(0, 0.5), (0.6, 0.5), (1, 0.4)],
                    screws, sides=12)                                                             # captive screw
            s.obox(at(q, r, u, n, 0, 0, 0.15), (0.6, 0.1, 0.04), (r, u, n), "hole")                       # slot
    for sv in (-1, 1):                                                                                     # wall lugs
        q = at(p, r, u, n, 0, sv * (h / 2 + 1.0), 0.3)
        s.rbox(q, (w * 0.35, 2.4, 0.6), 0.25, body, axes=(r, u, n), seg=1)
        round_hole(s, at(q, r, u, n, 0, sv * 0.2, 0.3), r, u, n, 0.35, n=10)
    return face


def push_button(s, p, r, u, n, cap="lamp_green", arrow=None, d=2.6):
    """22 mm-class push button on a face: black shroud bezel, recessed coloured cap, optional arrow (+1 / "up",
    -1 / "down", "left", "right" in the face's r / u frame)."""
    R = d / 2
    if arrow in ("left", "right"):
        r, u = u, (tuple(-c for c in r) if arrow == "left" else r)
        arrow = 1
    elif arrow in ("up", "down"):
        arrow = 1 if arrow == "up" else -1
    s.lathe(p, at(p, r, u, n, 0, 0, 0.7), [(0, R * 1.45), (0.5, R * 1.45), (0.8, R * 1.35), (1, R * 1.1)],
            "plastic_black", sides=24)                                                                     # shroud
    s.lathe(at(p, r, u, n, 0, 0, 0.6), at(p, r, u, n, 0, 0, 0.95), [(0, R * 0.95), (0.7, R * 0.95), (1, R * 0.88)],
            cap, sides=22)                                                                                  # cap
    if arrow:
        q = at(p, r, u, n, 0, 0, 0.97)
        k = R * 0.45
        poly(s, [at(q, r, u, n, 0, arrow * k), at(q, r, u, n, -k * 0.85, -arrow * k * 0.6),
                 at(q, r, u, n, k * 0.85, -arrow * k * 0.6)], n, "label_vinyl")


def estop(s, p, r, u, n, d=4.0):
    """Emergency stop: yellow legend ring, black collar, red mushroom head."""
    R = d / 2
    s.lathe(p, at(p, r, u, n, 0, 0, 0.15), [(0, R * 1.8), (1, R * 1.8)], "plastic_yellow", sides=32)
    s.lathe(at(p, r, u, n, 0, 0, 0.15), at(p, r, u, n, 0, 0, 0.9), [(0, R * 0.95), (1, R * 0.85)], "plastic_black", sides=24)
    s.lathe(at(p, r, u, n, 0, 0, 0.9), at(p, r, u, n, 0, 0, 2.5),
            [(0, R * 0.55), (0.25, R * 0.6), (0.35, R * 1.25), (0.6, R * 1.32), (0.85, R * 1.15), (1, R * 0.3)],
            "plastic_red", sides=32)


def pilot_lamp(s, p, r, u, n, lens="lamp_orange", d=2.2):
    R = d / 2
    s.lathe(p, at(p, r, u, n, 0, 0, 0.45), [(0, R * 1.4), (0.6, R * 1.4), (1, R * 1.15)], "chrome", sides=24)
    s.lathe(at(p, r, u, n, 0, 0, 0.3), at(p, r, u, n, 0, 0, 1.1), [(0, R * 0.95), (0.5, R * 0.9), (1, R * 0.25)],
            lens, sides=20)


def key_switch(s, p, r, u, n, d=2.2):
    R = d / 2
    s.lathe(p, at(p, r, u, n, 0, 0, 0.5), [(0, R * 1.35), (1, R * 1.2)], "chrome", sides=24)
    s.lathe(at(p, r, u, n, 0, 0, 0.5), at(p, r, u, n, 0, 0, 0.8), [(0, R * 0.8), (1, R * 0.75)], "steel_machined", sides=20)
    s.obox(at(p, r, u, n, 0, 0, 0.81), (0.25, R * 1.0, 0.04), (r, u, n), "hole")
    s.obox(at(p, r, u, n, 0, R * 0.2, 1.4), (0.2, R * 1.1, 1.2), (r, u, n), "steel_machined")             # key


def legend(s, p, r, u, n, text, h=0.9, plate="label_vinyl", ink="plastic_black"):
    """Printed legend plate centred at p with dot-matrix text."""
    w = h * 6 / 7 * len(text) + h * 0.9
    s.rbox(at(p, r, u, n, 0, 0, 0.04), (w, h * 1.7, 0.08), 0.1, plate, axes=(r, u, n), seg=1)
    down = tuple(-c for c in u)
    sign_text(s, text, at(p, r, u, n, -w / 2 + h * 0.45, h * 0.5, -0.2), r, down, h, ink, n)


def warning(s, p, r, n, size=4.0):
    warning_label(s, p, n, r, size)


def spring(s, a, b, R, wire, turns, mat="steel_blued"):
    """Helical extension spring from a to b with hook ends."""
    ax = _n(tuple(q - p for p, q in zip(a, b)))
    side = _n(_x(ax, (0, 1, 0) if abs(ax[1]) < 0.9 else (1, 0, 0)))
    up = _x(ax, side)
    L = math.dist(a, b)
    pts = []
    n_ = int(turns * 16)
    for k in range(n_ + 1):
        t = k / n_
        ang = 2 * math.pi * turns * t
        pts.append(tuple(p + ax_ * (0.15 * L + 0.7 * L * t) + R * (math.cos(ang) * sd + math.sin(ang) * uv)
                         for p, ax_, sd, uv in zip(a, ax, side, up)))
    s.sweep([a] + pts + [b], wire, mat, sides=6)


def warning_triangle(s, p, r, u, n, size, symbol="!", fill="plastic_yellow", ink="plastic_black"):
    """ISO warning decal: yellow triangle with a black border and a symbol ("!" general, "bolt" electrical)."""
    k = size / 2
    q = at(p, r, u, n, 0, 0, 0.04)
    tri = [at(q, r, u, n, 0, k * 0.95), at(q, r, u, n, -k * 1.08, -k * 0.9), at(q, r, u, n, k * 1.08, -k * 0.9)]
    poly(s, tri, n, ink)
    inner = [at(q, r, u, n, 0, k * 0.68, 0.02), at(q, r, u, n, -k * 0.82, -k * 0.74, 0.02),
             at(q, r, u, n, k * 0.82, -k * 0.74, 0.02)]
    poly(s, inner, n, fill)
    c = at(q, r, u, n, 0, -k * 0.1, 0.04)
    if symbol == "bolt":
        pts = [(0.12, 0.5), (-0.2, -0.02), (0.02, -0.02), (-0.12, -0.5), (0.22, 0.08), (0.0, 0.08)]
        poly(s, [at(c, r, u, n, x * k, y * k) for x, y in pts[:3]], n, ink)
        poly(s, [at(c, r, u, n, x * k, y * k) for x, y in pts[2:5]], n, ink)
        poly(s, [at(c, r, u, n, x * k, y * k) for x, y in (pts[0], pts[2], pts[4], pts[5])], n, ink)
    else:
        s.obox(at(c, r, u, n, 0, k * 0.12), (k * 0.14, k * 0.5, 0.03), (r, u, n), ink)
        s.obox(at(c, r, u, n, 0, -k * 0.3), (k * 0.14, k * 0.13, 0.03), (r, u, n), ink)


def caution_label(s, p, r, u, n, w):
    """ANSI-style CAUTION label: yellow field with a black header band and fine print lines."""
    h = w * 0.45
    s.rbox(at(p, r, u, n, 0, 0, 0.04), (w, h, 0.08), 0.15, "plastic_yellow", axes=(r, u, n), seg=1)
    s.obox(at(p, r, u, n, 0, h * 0.28, 0.09), (w * 0.94, h * 0.32, 0.03), (r, u, n), "plastic_black")
    th = h * 0.2
    tw = th * 6 / 7 * 7
    sign_text(s, "CAUTION", at(p, r, u, n, -tw / 2, h * 0.28 + th / 2, -0.12), r, tuple(-c for c in u), th,
              "plastic_yellow", n)
    for k in range(3):
        s.obox(at(p, r, u, n, -w * 0.05, -h * (0.05 + k * 0.14), 0.09), (w * 0.78 - k * w * 0.12, h * 0.05, 0.03),
               (r, u, n), "plastic_black")


def louvres(s, p, r, u, n, w, h, rows, mat):
    """Pressed louvre vent: a dark recess with angled slats (on a face at p, centred)."""
    s.obox(at(p, r, u, n, 0, 0, -0.05), (w, h, 0.1), (r, u, n), "hole")
    pitch = h / rows
    for k in range(rows):
        v = -h / 2 + (k + 0.5) * pitch
        tilt = _n(tuple(ui * 0.8 + ni * 0.6 for ui, ni in zip(u, n)))
        s.rbox(at(p, r, u, n, 0, v, 0.15), (w * 0.96, pitch * 0.55, 0.22), 0.08, mat,
               axes=(r, tilt, _n(_x(r, tilt))), seg=1)


def hinge(s, p, axis, length, mat, r=0.55):
    """Door hinge barrel along axis, centred at p, with a pin cap at each end."""
    a = tuple(pi - ai * length / 2 for pi, ai in zip(p, axis))
    b = tuple(pi + ai * length / 2 for pi, ai in zip(p, axis))
    s.lathe(a, b, [(0, r * 0.85), (0.05, r), (0.95, r), (1, r * 0.85)], mat, sides=14)
    for e, sg in ((a, -1), (b, 1)):
        s.lathe(e, tuple(ei + ai * sg * 0.3 for ei, ai in zip(e, axis)), [(0, r * 0.6), (1, r * 0.45)], "steel_machined",
                sides=12)


def quarter_lock(s, p, r, u, n, d=1.8):
    """Quarter-turn cabinet lock: chrome bezel, insert with a slot (or a T-handle when handle=True)."""
    R = d / 2
    s.lathe(p, at(p, r, u, n, 0, 0, 0.35), [(0, R * 1.3), (0.6, R * 1.3), (1, R * 1.1)], "chrome", sides=20)
    s.lathe(at(p, r, u, n, 0, 0, 0.35), at(p, r, u, n, 0, 0, 0.5), [(0, R * 0.8), (1, R * 0.75)], "plastic_black", sides=18)
    s.obox(at(p, r, u, n, 0, 0, 0.51), (R * 1.1, 0.18, 0.04), (r, u, n), "hole")


def sign(s, p, r, u, n, w, lines, header=None, header_mat="plastic_red", plate="label_vinyl", ink="plastic_black",
         border=None):
    """Safety / notice sign: plate with an optional coloured header (white text) and lines of text, rivets."""
    rows = len(lines) + (1.6 if header else 0)
    th = min(w / (max([len(t) for t in lines] + [len(header or "")]) * 6 / 7 + 1.5), w * 0.16)
    h = th * (rows * 1.7 + 1.2)
    s.rbox(at(p, r, u, n, 0, 0, 0.0), (w, h, 0.3), 0.12, plate, axes=(r, u, n), seg=1)        # thick enough to hide
    if border:
        for a, b, ww, hh in ((0, h / 2 - th * 0.25, w - th * 0.3, th * 0.18), (0, -h / 2 + th * 0.25, w - th * 0.3, th * 0.18),
                             (-w / 2 + th * 0.25, 0, th * 0.18, h - th * 0.3), (w / 2 - th * 0.25, 0, th * 0.18, h - th * 0.3)):
            s.obox(at(p, r, u, n, a, b, 0.17), (ww, hh, 0.03), (r, u, n), border)
    down = tuple(-c for c in u)
    y = h / 2 - th * 0.75
    if header:
        s.rbox(at(p, r, u, n, 0, y - th * 0.55, 0.17), (w * 0.86, th * 1.9, 0.04), th * 0.9, header_mat, axes=(r, u, n), seg=2)
        tw = th * 6 / 7 * len(header)
        sign_text(s, header, at(p, r, u, n, -tw / 2, y + th * 0.4, -0.04), r, down, th, "label_vinyl", n)
        y -= th * 2.4
    for t in lines:
        tw = th * 6 / 7 * len(t)
        sign_text(s, t, at(p, r, u, n, -tw / 2, y, -0.12), r, down, th, ink, n)
        y -= th * 1.7
    for a in (-w / 2 + th * 0.6, w / 2 - th * 0.6):
        for b in (-h / 2 + th * 0.6, h / 2 - th * 0.6):
            s.lathe(at(p, r, u, n, a, b, 0.1), at(p, r, u, n, a, b, 0.3), [(0, th * 0.22), (1, th * 0.16)], "stainless",
                    sides=8)
    return h
