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
"""
import math

from .base import Item, Param, new_shapes, register, shapes_to_stage

SEG = {"0": "abcdef", "1": "bc", "2": "abdeg", "3": "abcdg", "4": "bcfg", "5": "acdfg", "6": "acdefg", "7": "abc",
       "8": "abcdefg", "9": "abcdfg"}
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


def chain_hoist(s, tx, fb, z, gfw, hook_y, body, attach_key, attach):
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
    finned_motor(s, (tx + 14, hb - 22, z), (tx + 42, hb - 22, z), 12.0, body, "powder_dark")
    s.lathe((tx - 14, hb - 22, z), (tx - 24, hb - 22, z), [(0, 12.5), (0.6, 12.5), (0.85, 11.0), (1, 7.0)], body, sides=28)
    for k in range(6):                                        # cover screws
        ang = 2 * math.pi * k / 6
        s.bolt((tx - 24.2, hb - 22 + 9 * math.cos(ang), z + 9 * math.sin(ang)), (-1, 0, 0), 0.7, BOLT)
    s.rbox((tx + 2, hb - 22, z + 17), (24, 22, 7), 2.0, "powder_dark", seg=2)   # electrical box
    for k, gx in enumerate((tx - 6, tx + 2)):                 # cable glands, cables up to the festoon
        s.lathe((gx, hb - 11, z + 17), (gx, hb - 8, z + 17), [(0, 1.6), (0.5, 1.6), (1, 1.1)], "plastic_black", sides=12)
        s.sweep([(gx, hb - 8, z + 17), (gx, hb + 2, z + 19 + k), (gx + 4, fb + 14, z + 21 + 2 * k)], 0.6, "cable_pvc", sides=8)
    # chain outlet, chain, chain bag
    s.lathe((tx + 4, hb - 37, z), (tx + 4, hb - 43, z), [(0, 4.5), (0.6, 3.2), (1, 2.6)], "powder_dark", sides=16)
    cx = tx + 4
    bot_block = hook_y + 34
    chain(s, (cx, hb - 43, z), (cx, bot_block + 8, z))
    s.sweep([(tx - 8, hb - 36, z - 9), (tx - 8, hb - 40, z - 12)], 0.6, BOLT, sides=6)                  # bag hook
    s.lathe((tx - 8, hb - 40, z - 14), (tx - 8, hb - 70, z - 14),
            [(0, 5), (0.15, 7.5), (0.6, 8.5), (0.9, 7.5), (1, 4)], "plastic_black", sides=18)               # chain bag
    # bottom block: cheek plates around a round sheave guard, cross-head, swivel, hook
    yb = bot_block + 6
    s.lathe((cx, yb, z - 5), (cx, yb, z + 5), [(0, 8), (0.1, 9), (0.9, 9), (1, 8)], body, sides=24)
    for uz in (-1, 1):
        s.rbox((cx, yb - 2, z + uz * 5.8), (14, 22, 1.6), 0.6, "powder_dark", seg=1)
        s.bolt((cx, yb, z + uz * 6.6), (0, 0, uz), 1.0, BOLT)
    s.rbox((cx, yb - 12, z), (13, 4, 13), 1.2, "powder_dark", seg=1)
    hook(s, cx, yb - 14, z, 1.0)
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
                bands = [(18 + k * 14, 32 + k * 14, "floor_yellow" if k % 2 == 0 else "powder_dark") for k in range(6)]
                pieces = [(8.0, 18.0, col)] + bands + [(bands[-1][1], splice - 1.4, col), (splice + 1.4, H - 2.4, col)]
                for y0, y1, m_ in pieces:                                                        # column: flush pieces
                    s.rprism((x, y0, z), (x, y1, z), a, a, 2.5, m_, up=(1, 0, 0), seg=3)
                for y in (splice - 1.4, splice + 1.4):                                           # bolted splice
                    s.rbox((x, y, z), (a + 14, 2.6, a + 14), 0.5, col, seg=1)
                for ux, uz in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                    bx, bz = x + ux * (a / 2 + 4), z + uz * (a / 2 + 4)
                    s.bolt((bx, splice + 2.7, bz), (0, 1, 0), 1.4, BOLT)
                    s.bolt((bx, splice - 2.7, bz), (0, -1, 0), 1.4, BOLT)
                s.rbox((x, H - 1.2, z), (a + 8, 2.4, a + 8), 0.5, col, seg=1)                     # cap plate
        # ------------------------------------------------------------------ runways, tie beams, braces
        rail_top = H + rd + 4.0
        for x in xs:
            s.rprism((x, H + rd / 2, -Rw / 2), (x, H + rd / 2, Rw / 2), rw, rd, 2.0, col)          # runway box
            s.rbox((x, H + rd + 2, 0), (5, 4, Rw - 8), 0.6, "steel_dark", seg=1)                 # running rail
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
        for bi, f in enumerate(fr):
            bz = -Rw / 2 + 30 + et / 2 + (Rw - 60 - et) * f
            wy = rail_top + rwh
            t_lo, t_hi = wy - 13, wy + 15
            for x in xs:                                                                          # end trucks
                s.rbox((x, (t_lo + t_hi) / 2, bz), (24, t_hi - t_lo, et), 3.0, yel, seg=2)
                for wz in (bz - et / 2 + 28, bz + et / 2 - 28):
                    for ux in (-1, 1):
                        s.lathe((x + ux * 12, wy, wz), (x + ux * 4, wy, wz),
                                [(0, rwh * 0.4), (0.15, rwh * 0.4), (0.2, rwh), (0.75, rwh), (0.8, rwh + 2), (1, rwh + 2)],
                                "steel_machined", sides=24)
                    s.lathe((x + 12, wy, wz), (x + 14, wy, wz), [(0, 4.5), (1, 3.5)], "steel_dark", sides=16)
                    for k in range(6):
                        ang = 2 * math.pi * k / 6
                        s.bolt((x + 12.2, wy + 3 * math.cos(ang), wz + 3 * math.sin(ang)), (1, 0, 0), 0.7, BOLT)
                for zend, uz in ((bz - et / 2, -1), (bz + et / 2, 1)):
                    s.lathe((x, wy, zend), (x, wy, zend + uz * 9), [(0, 5.5), (0.6, 5.5), (0.8, 5.0), (1, 3.5)],
                            "rubber_black", sides=16)
            dx = xs[1] + 13                                                                       # drive
            finned_motor(s, (dx + 14, wy, bz - et / 2 + 28), (dx + 40, wy, bz - et / 2 + 28), 9.0, yel, "powder_dark")
            s.rbox((dx + 7, wy, bz - et / 2 + 28), (14, 26, 26), 3.0, yel, seg=2)
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
            if cap:                                                                               # capacity sign
                zf = gz[-1] + gw / 2 + 0.2
                wsign = max(44.0, 10.0 * len(cap) + 18)
                s.rbox((0, gy, zf + 0.5), (wsign, 26, 1.0), 0.8, "floor_white", seg=1)
                digits(s, cap, (-wsign / 2 + 9, gy + 9, zf + 1.0), (1, 0, 0), (0, -1, 0), 16.0, 2.6, "plastic_black", (0, 0, 1))
            fz = gz[0] - gw / 2 - 8                                                               # festoon
            fy = gy + gd / 2 - 4
            for bxx in range(int(-S / 2), int(S / 2) + 1, 150):
                s.rbox((bxx, fy + 2, fz + 4), (3, 8, 10), 0.5, "galvanised", seg=1)
            if not dbl:
                hx = [(-S / 2 + 60 + (S - 120) * ((k + 0.5) / nh if nh > 1 else p["trolley"])) for k in range(nh)]
                fb = gy - gd / 2 - 2.6
                last = None
                for hi_, tx in enumerate(hx):
                    key = (f"hook{bi + 1}_{hi_ + 1}" if nh > 1 else f"hook{bi + 1}") if n > 1 else \
                          (f"hook{hi_ + 1}" if nh > 1 else "hook")
                    cx, hb = chain_hoist(s, tx, fb, bz, gw + 14, max(30.0, min(H - 120.0, p["hook_height"])),
                                         "powder_orange", key, attach)
                    pendant(s, cx, hb - 30, bz, attach, key.replace("hook", "pendant"))
                    last = tx
                festoon(s, -S / 2 - 15, S / 2 + 15, fy, fz, -S / 2 + 10, last or 0.0)
            else:                                                                                 # crab + winch
                top_g = gy + gd / 2
                tx = -S / 2 + 70 + (S - 140) * p["trolley"]
                for z in gz:
                    s.rbox((0, top_g + 2, z), (S - 40, 4, 5), 0.6, "steel_dark", seg=1)
                cl = 120.0
                s.rbox((tx, top_g + 20, bz), (cl, 14, ggap + 34), 2.5, yel, seg=2)
                for wx in (tx - cl / 2 + 20, tx + cl / 2 - 20):
                    for z in gz:
                        s.lathe((wx, top_g + 11, z - 5), (wx, top_g + 11, z + 5),
                                [(0, 6), (0.2, 7.5), (0.3, 7.5), (0.35, 6), (1, 6)], "steel_machined", sides=20)
                drum = [(0, 19), (0.04, 19)] + [v for k in range(16) for v in ((0.06 + k * 0.055, 17.8), (0.085 + k * 0.055, 18.4))] + [(0.95, 19), (1, 19)]
                s.lathe((tx - 44, top_g + 52, bz), (tx + 30, top_g + 52, bz), drum, "steel_machined", sides=28)
                s.rbox((tx + 40, top_g + 50, bz), (20, 46, 48), 4.0, yel, seg=2)                 # gearbox
                finned_motor(s, (tx + 50, top_g + 50, bz), (tx + 84, top_g + 50, bz), 15.0, yel, "powder_dark")
                s.rbox((tx - cl / 2 + 20, top_g + 46, bz + ggap / 2 + 10), (36, 40, 12), 2.0, "powder_dark", seg=2)
                hk = max(30.0, min(H - 120.0, p["hook_height"]))
                bot = hk + 62
                for ox, oz in ((-7, -9), (-7, 9), (7, -9), (7, 9)):
                    s.cylinder((tx + ox, top_g + 34, bz + oz), (tx + ox, bot, bz + oz), 0.9, "galvanised", 8)
                s.rbox((tx, bot - 18, bz), (32, 36, 30), 4.0, yel, seg=2)                          # hook block
                for oz in (-9, 9):
                    s.lathe((tx - 17, bot - 12, bz + oz), (tx + 17, bot - 12, bz + oz), [(0, 11), (1, 11)], "powder_dark", sides=24)
                hook(s, tx, bot - 38, bz, 1.6)
                key = f"hook{bi + 1}" if n > 1 else "hook"
                attach[key] = (tx, bot - 60, bz)
                pendant(s, tx, top_g, bz, attach, key.replace("hook", "pendant"))
                festoon(s, -S / 2 - 15, S / 2 + 15, fy, fz, -S / 2 + 10, tx)
            attach[f"bridge{bi + 1}" if n > 1 else "bridge"] = (0, gy, bz)
        shapes_to_stage(stage, path, s, lib)
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
