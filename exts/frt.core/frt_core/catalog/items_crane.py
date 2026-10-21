"""Freestanding workstation bridge crane, built the way they are bought and installed: square-tube columns on base
plates (grout pad, anchor bolts, gussets, hazard bands), header beams tying the column tops, knee braces and sway
rods, I-beam runways with a running rail and end stops, end trucks with wheels and a drive, I-beam bridge girders
(single or double), an electric chain hoist on an underhung trolley (single girder) or a top-running crab with a rope
winch (double girder), pendant control, festoon cable loops, and a capacity sign.

Same parameters as before (span, runway, height, bridges, bridge_positions, girders, trolley, hook_height, capacity),
so scenes read from drawings get this model as they are. Local frame: span along X, runway along Z, floor at Y = 0.
"""
import math

from .base import Item, Param, new_shapes, register, shapes_to_stage

SEG = {"0": "abcdef", "1": "bc", "2": "abdeg", "3": "abcdg", "4": "bcfg", "5": "acdfg", "6": "acdefg", "7": "abc",
       "8": "abcdefg", "9": "abcdfg"}


def _norm(v):
    L = math.sqrt(sum(c * c for c in v)) or 1.0
    return tuple(c / L for c in v)


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def ibeam(s, a, b, depth, fw, tf, tw, mat, up=(0.0, 1.0, 0.0)):
    """I-beam from a to b: two flanges and a web; `up` is the web direction."""
    ax = _norm(tuple(q - p for p, q in zip(a, b)))
    up = _norm(up)
    side = _norm(_cross(up, ax))
    L = math.dist(a, b)
    c = tuple((p + q) / 2 for p, q in zip(a, b))
    for sgn in (-1, 1):
        cc = tuple(ci + sgn * (depth / 2 - tf / 2) * u for ci, u in zip(c, up))
        s.obox(cc, (L, fw, tf), (ax, side, up), mat)
    s.obox(c, (L, tw, depth - 2 * tf), (ax, side, up), mat)


def tube(s, a, b, w, mat):
    """Square tube (as a solid bar) from a to b."""
    ax = _norm(tuple(q - p for p, q in zip(a, b)))
    up = (0.0, 1.0, 0.0) if abs(ax[1]) < 0.9 else (1.0, 0.0, 0.0)
    side = _norm(_cross(up, ax))
    up = _norm(_cross(ax, side))
    c = tuple((p + q) / 2 for p, q in zip(a, b))
    s.obox(c, (math.dist(a, b), w, w), (ax, side, up), mat)


def digits_on_plane(s, text, origin, right, down, h, t, mat, out):
    """Seven-segment digits on a plane: origin top-left, `right` / `down` unit vectors, height h, stroke t, offset `out`."""
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
            c = tuple(o + cu * r + cv * d + 0.3 * n for o, r, d, n in zip(origin, right, down, out))
            s.obox(c, (u1 - u0, v1 - v0, 0.4), (right, down, out), mat)


class WorkstationCrane(Item):
    def footprint(self, p):
        return p["span"] + 50.0, p["runway"]

    def build(self, stage, path, p, lib):
        S, Rw, H = p["span"], p["runway"], p["height"]
        yel, col = p["colour"], p["column_colour"]
        s = new_shapes()
        heavy = S > 900 or Rw > 1800
        a = 25.0 if heavy else 20.0                                   # column tube
        rd, rfw = (40.0, 20.0) if heavy else (34.0, 17.0)             # runway beam depth / flange
        nbay = max(1, math.ceil(Rw / p["column_spacing"]))
        zs = [-Rw / 2 + a / 2 + 2 + k * (Rw - a - 4) / nbay for k in range(nbay + 1)]
        xs = (-S / 2, S / 2)
        # ------------------------------------------------------------------ columns
        for x in xs:
            for z in zs:
                s.box((x - 32, 0, z - 32), (x + 32, 2.5, z + 32), "powder_lightgrey")              # grout pad
                s.box((x - 27, 2.5, z - 27), (x + 27, 5.0, z + 27), "steel_dark")                  # base plate
                for dx in (-20, 20):
                    for dz in (-20, 20):
                        s.cylinder((x + dx, 5.0, z + dz), (x + dx, 11.0, z + dz), 1.2, "galvanised", 8)   # anchor
                        s.box((x + dx - 2.2, 5.0, z + dz - 2.2), (x + dx + 2.2, 7.5, z + dz + 2.2), "galvanised")  # nut
                for ux, uz in ((1, 0), (-1, 0), (0, 1), (0, -1)):                                  # gussets
                    gx, gz = x + ux * (a / 2 + 6), z + uz * (a / 2 + 6)
                    ang = math.radians(45)
                    axis = (ux * math.cos(ang), math.sin(ang), uz * math.cos(ang)) if ux or uz else (0, 1, 0)
                    side = (uz, 0.0, -ux) if (ux or uz) else (1, 0, 0)
                    s.obox((gx - ux * 2, 11.0, gz - uz * 2), (17.0, 1.2, 9.0), (_norm(axis), side, _norm(_cross(_norm(axis), side))), col)
                s.box((x - a / 2, 5.0, z - a / 2), (x + a / 2, H - 2.0, z + a / 2), col)          # column tube
                for k in range(6):                                                                 # hazard bands
                    y0 = 12 + k * 14
                    s.box((x - a / 2 - 0.3, y0, z - a / 2 - 0.3), (x + a / 2 + 0.3, y0 + 14, z + a / 2 + 0.3),
                          "floor_yellow" if k % 2 == 0 else "plastic_black")
                s.box((x - a / 2 - 4, H - 2.0, z - a / 2 - 4), (x + a / 2 + 4, H, z + a / 2 + 4), "steel_dark")  # cap
        # header beams tying each end frame (and every frame when heavy), just under the runway
        hd = 30.0 if heavy else 25.0
        for z in (zs if heavy else (zs[0], zs[-1])):
            ibeam(s, (xs[0] + a / 2, H - 2 - hd / 2, z), (xs[1] - a / 2, H - 2 - hd / 2, z), hd, 15.0, 1.6, 1.0, col)
            for x, ux in ((xs[0], 1), (xs[1], -1)):                                                 # lateral knee braces
                tube(s, (x + ux * a / 2, H - 120, z), (x + ux * 110, H - 2 - hd, z), 8.0, col)
        # ------------------------------------------------------------------ runways
        rail_top = H + rd + 4.0
        for x in xs:
            ibeam(s, (x, H + rd / 2, -Rw / 2), (x, H + rd / 2, Rw / 2), rd, rfw, 2.0, 1.2, col)
            s.box((x - 2.5, H + rd, -Rw / 2 + 3), (x + 2.5, rail_top, Rw / 2 - 3), "steel_dark")   # running rail
            for z in zs:                                                                          # longitudinal knees
                for uz in (-1, 1):
                    if (uz < 0 and z == zs[0]) or (uz > 0 and z == zs[-1]):
                        continue
                    tube(s, (x, H - 110, z + uz * a / 2), (x, H, z + uz * 100), 8.0, col)
            for zend, uz in ((-Rw / 2, 1), (Rw / 2, -1)):                                          # end stops
                s.box((x - 10, H + rd, zend + uz * 1), (x + 10, rail_top + 22, zend + uz * 3), "steel_dark")
                s.cylinder((x, rail_top + 12, zend + uz * 3), (x, rail_top + 12, zend + uz * 13), 6.0, "rubber_black", 12)
            if nbay >= 2:                                                                          # sway rods, end bays
                for i in (0, nbay - 1):
                    za, zb = zs[i] + a / 2, zs[i + 1] - a / 2
                    for (y0, y1) in ((60.0, H - 60), (H - 60, 60.0)):
                        s.cylinder((x, y0, za), (x, y1, zb), 1.1, "galvanised", 6)
                    s.box((x - 2, (H) / 2 - 8, (za + zb) / 2 - 2), (x + 2, H / 2 + 8, (za + zb) / 2 + 2), "galvanised")
        # ------------------------------------------------------------------ bridges
        n = max(1, int(p["bridges"]))
        if p["bridge_positions"].strip():
            fr = [min(1.0, max(0.0, float(v))) for v in p["bridge_positions"].split(",") if v.strip()][:n] or [0.5]
        elif n == 1:
            fr = [p["bridge"]]
        else:
            fr = [(k + 0.5) / n for k in range(n)]
        dbl = int(p["girders"]) >= 2
        gd, gfw = (55.0, 24.0) if (heavy or dbl) else (42.0, 18.0)
        ggap = 90.0 if dbl else 0.0
        et = 150.0 + ggap
        r_w = 10.0
        attach = {}
        for bi, f in enumerate(fr):
            bz = -Rw / 2 + 25 + et / 2 + (Rw - 50 - et) * f
            wheel_y = rail_top + r_w
            t_lo, t_hi = wheel_y - 12, wheel_y + 14
            for x, ux in ((xs[0], -1), (xs[1], 1)):                                                # end trucks
                s.box((x - 9, t_lo, bz - et / 2), (x + 9, t_hi, bz + et / 2), yel)
                for wz in (bz - et / 2 + 24, bz + et / 2 - 24):
                    s.cylinder((x - 4, wheel_y, wz), (x + 4, wheel_y, wz), r_w, "steel_dark", 16)
                    s.cylinder((x - 5.5, wheel_y, wz), (x - 4, wheel_y, wz), r_w + 2, "steel_dark", 16)   # flange
                    s.cylinder((x - 10, wheel_y, wz), (x + 10, wheel_y, wz), 2.0, "chrome", 8)          # axle
                for zend, uz in ((bz - et / 2, -1), (bz + et / 2, 1)):
                    s.cylinder((x, wheel_y, zend), (x, wheel_y, zend + uz * 8), 5.0, "rubber_black", 10)
            dx = xs[1] + 14                                                                         # drive, one side
            s.cylinder((dx, wheel_y, bz - et / 2 + 24), (dx + 30, wheel_y, bz - et / 2 + 24), 9.0, "powder_dark", 16)
            s.box((dx, wheel_y - 11, bz - et / 2 + 10), (dx + 14, wheel_y + 11, bz - et / 2 + 38), "powder_dark")
            s.cylinder((dx + 30, wheel_y, bz - et / 2 + 24), (dx + 34, wheel_y, bz - et / 2 + 24), 6.0, "steel_dark", 12)
            gy = t_hi + gd / 2                                                                     # girders on the trucks
            gz = (bz - ggap / 2, bz + ggap / 2) if dbl else (bz,)
            for z in gz:
                ibeam(s, (xs[0] - 22, gy, z), (xs[1] + 22, gy, z), gd, gfw, 2.4, 1.4, yel)
                for x in xs:                                                                       # connection plates
                    s.box((x - 14, t_hi - 1, z - gfw / 2 - 3), (x + 14, t_hi + 1.5, z + gfw / 2 + 3), "steel_dark")
            # capacity sign on the girder web (digits of the rated load)
            cap = "".join(ch for ch in str(p["capacity"]) if ch.isdigit()) or ""
            sz_ = gz[-1] + (gfw / 2 if not dbl else gfw / 2)
            if cap:
                w_sign = max(40.0, 9.0 * len(cap) + 16)
                s.box((-w_sign / 2, gy - 12, sz_ - 0.2 + 1.2), (w_sign / 2, gy + 12, sz_ + 0.2 + 1.2 + 0.4), "floor_white")
                s.box((-w_sign / 2 - 1, gy - 13, sz_ + 1.1), (w_sign / 2 + 1, gy + 13, sz_ + 1.3), "floor_yellow")
                digits_on_plane(s, cap, (-w_sign / 2 + 8, gy + 9, sz_ + 1.8), (1.0, 0.0, 0.0), (0.0, -1.0, 0.0),
                                14.0, 2.4, "plastic_black", (0.0, 0.0, 1.0))
            # festoon: C-track on the far side of the girder, cable trolleys parked near one end, cable loops
            fz = gz[0] - gfw / 2 - 8
            fy = gy + gd / 2 - 6
            s.box((xs[0] - 20, fy - 2.5, fz - 2.5), (xs[1] + 20, fy + 2.5, fz + 2.5), "galvanised")
            tx = -S / 2 + 60 + (S - 120) * p["trolley"]
            parked = [xs[0] + 10 + k * 22 for k in range(5)] + [tx]
            for k, px in enumerate(parked):
                s.box((px - 4, fy - 8, fz - 3), (px + 4, fy - 2.5, fz + 3), "steel_dark")
                if k:
                    a0 = parked[k - 1]
                    sag = min(60.0, 0.12 * (px - a0) + 18)
                    prev = None
                    for j in range(9):
                        t = j / 8
                        q = (a0 + (px - a0) * t, fy - 8 - sag * math.sin(math.pi * t), fz)
                        if prev is not None:
                            s.cylinder(prev, q, 0.9, "plastic_black", 6)
                        prev = q
            # hoist
            hook_y = max(30.0, min(H - 80.0, p["hook_height"]))
            if not dbl:                                                  # underhung trolley + electric chain hoist
                fb = gy - gd / 2                                         # girder bottom
                for uz in (-1, 1):
                    s.box((tx - 18, fb - 26, bz + uz * (gfw / 2 + 2) - 1), (tx + 18, fb + 6, bz + uz * (gfw / 2 + 2) + 1), yel)
                    for wx in (tx - 11, tx + 11):
                        s.cylinder((wx, fb + 3.5, bz + uz * (gfw / 2 - 1)), (wx, fb + 3.5, bz + uz * (gfw / 2 + 1)), 4.5,
                                   "steel_dark", 12)
                s.box((tx - 18, fb - 30, bz - gfw / 2 - 3), (tx + 18, fb - 26, bz + gfw / 2 + 3), yel)
                hb_top, hb_bot = fb - 30, fb - 72
                s.box((tx - 20, hb_bot, bz - 15), (tx + 20, hb_top, bz + 15), yel)                        # housing
                s.cylinder((tx + 20, hb_bot + 21, bz), (tx + 46, hb_bot + 21, bz), 13.0, yel, 20)       # motor
                s.cylinder((tx + 46, hb_bot + 21, bz), (tx + 50, hb_bot + 21, bz), 10.0, "powder_dark", 16)
                s.cylinder((tx - 20, hb_bot + 21, bz), (tx - 30, hb_bot + 21, bz), 14.0, yel, 20)       # gear cover
                s.box((tx - 18, hb_bot - 2, bz + 15), (tx + 2, hb_bot + 30, bz + 22), "powder_dark")      # control box
                s.box((tx - 3, hb_bot - 6, bz - 20), (tx - 1, hb_bot, bz - 16), "steel_dark")              # bag hanger
                s.cylinder((tx - 3, hb_bot - 8, bz - 19), (tx - 3, hb_bot - 34, bz - 19), 8.0, "plastic_black", 14)  # chain bag
                s.cylinder((tx - 3, hb_bot - 8, bz - 19), (tx - 3, hb_bot - 10, bz - 19), 8.8, "powder_dark", 14)
                cx, cz = tx + 4, bz
                top, bot = hb_bot, hook_y + 46
                k = 0
                y = top
                while y - 4.4 > bot:                                                                      # load chain
                    w1, w2 = (2.6, 0.9) if k % 2 == 0 else (0.9, 2.6)
                    s.box((cx - w1 / 2, y - 4.4, cz - w2 / 2), (cx + w1 / 2, y, cz + w2 / 2), "steel_dark")
                    y -= 3.6
                    k += 1
                yb = (hook_y + 28 + bot) / 2                                                              # bottom block:
                s.cylinder((cx, yb, cz - 5.5), (cx, yb, cz + 5.5), 9.0, yel, 20)                          # sheave guard
                for uz in (-1, 1):
                    s.box((cx - 6, hook_y + 24, cz + uz * 5.5 - (1.2 if uz > 0 else 0)),
                          (cx + 6, bot - 2, cz + uz * 5.5 + (0 if uz > 0 else 1.2)), "steel_dark")      # cheek plates
                s.box((cx - 7, hook_y + 22, cz - 7), (cx + 7, hook_y + 27, cz + 7), "steel_dark")          # cross-head
                s.cylinder((cx, hook_y + 16, cz), (cx, hook_y + 24, cz), 2.6, "chrome", 10)                # shank
                hz = cz
                hook_x = cx
            else:                                                       # top-running crab with a rope winch
                top_g = gy + gd / 2
                for z in gz:
                    s.box((xs[0] + 30, top_g, z - 2.5), (xs[1] - 30, top_g + 4, z + 2.5), "steel_dark")  # crab rails
                cl = 110.0
                s.box((tx - cl / 2, top_g + 12, bz - ggap / 2 - 16), (tx + cl / 2, top_g + 26, bz + ggap / 2 + 16), yel)
                for wx in (tx - cl / 2 + 18, tx + cl / 2 - 18):
                    for z in gz:
                        s.cylinder((wx, top_g + 11, z - 4), (wx, top_g + 11, z + 4), 7.0, "steel_dark", 14)
                s.cylinder((tx - 40, top_g + 46, bz), (tx + 30, top_g + 46, bz), 18.0, "steel_dark", 24)  # drum
                for gx in range(-36, 30, 6):
                    s.cylinder((tx + gx, top_g + 46, bz), (tx + gx + 1.2, top_g + 46, bz), 18.8, "galvanised", 24)
                s.box((tx + 30, top_g + 26, bz - 22), (tx + 52, top_g + 68, bz + 22), yel)                  # gearbox
                s.cylinder((tx + 52, top_g + 46, bz), (tx + 85, top_g + 46, bz), 15.0, yel, 20)            # motor
                s.box((tx - cl / 2, top_g + 26, bz + ggap / 2 + 4), (tx - cl / 2 + 40, top_g + 66, bz + ggap / 2 + 16),
                      "powder_dark")                                                                       # panel
                bot = hook_y + 60
                for ox, oz in ((-6, -8), (-6, 8), (6, -8), (6, 8)):                                        # rope falls
                    s.cylinder((tx + ox, top_g + 30, bz + oz), (tx + ox, bot, bz + oz), 0.9, "galvanised", 6)
                s.box((tx - 16, hook_y + 30, bz - 14), (tx + 16, bot, bz + 14), yel)                        # hook block
                for oz in (-8, 8):
                    s.cylinder((tx - 15, bot - 12, bz + oz), (tx + 15, bot - 12, bz + oz), 11.0, "steel_dark", 18)
                s.cylinder((tx, hook_y + 18, bz), (tx, hook_y + 30, bz), 3.0, "steel_dark", 10)
                hz, hook_x = bz, tx
            # hook with latch
            seg = 10
            pts = []
            for i in range(seg + 1):
                ang = math.radians(90 + i * 230 / seg)
                pts.append((hook_x + 8 * math.cos(ang), hook_y + 8 + 8 * math.sin(ang), hz))
            pts[0] = (hook_x, hook_y + 18, hz)
            for q0, q1 in zip(pts, pts[1:]):
                s.cylinder(q0, q1, 2.4, "steel_dark", 10)
            s.cylinder(pts[-1], (hook_x - 2, hook_y + 16, hz), 0.6, "chrome", 6)                          # latch
            # pendant control on its strain cable
            pz = hz + 26
            s.cylinder((hook_x - 14, hook_y + 60 if not dbl else gy - gd / 2, pz), (hook_x - 14, 128, pz), 0.5, "plastic_black", 5)
            s.box((hook_x - 19, 100, pz - 4), (hook_x - 9, 128, pz + 4), "powder_yellow")
            for j in range(5):
                s.cylinder((hook_x - 14, 123 - j * 4.8, pz + 4), (hook_x - 14, 123 - j * 4.8, pz + 5.4), 1.5,
                           "plastic_black", 8)
            s.cylinder((hook_x - 14, 128, pz), (hook_x - 14, 131, pz), 3.0, "powder_red", 12)               # e-stop
            attach[f"hook{bi + 1}" if n > 1 else "hook"] = (hook_x, hook_y, hz)
            attach[f"bridge{bi + 1}" if n > 1 else "bridge"] = (0, gy, bz)
            attach[f"pendant{bi + 1}" if n > 1 else "pendant"] = (hook_x - 14, 114, pz)
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
    "girders": Param(1, "girders per bridge (1: underhung chain hoist, 2: top-running crab)", 1, 2),
    "trolley": Param(0.5, "trolley position along the bridge (0-1)", 0, 1),
    "hook_height": Param(200.0, "hook height above the floor", 30, 1200, "cm"),
    "colour": Param("powder_yellow", "bridge, trucks, hoist"),
    "column_colour": Param("powder_yellow", "columns, headers and runways"),
    "capacity": Param("", "rated load as on the drawing, e.g. 2000lb (shown on the sign)"),
}))
