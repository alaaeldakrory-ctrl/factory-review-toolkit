"""Bin rack (shelving with plastic picking bins), pillar jib crane and freestanding bridge (workstation) crane.
Same local frame as every catalog item (base.py): origin at the footprint centre on the floor, +X length, +Z front.
"""
import math

from .base import Item, Param, new_shapes, register, shapes_to_stage


# ------------------------------------------------------------------------------------------ bin rack
class BinRack(Item):
    """Steel shelving bay(s) with open-front plastic bins (the small-parts rack next to every assembly station)."""

    def footprint(self, p):
        return p["bays"] * p["width"] + 4.0, p["depth"]

    def build(self, stage, path, p, lib):
        nb, W, D, H, ns = p["bays"], p["width"], p["depth"], p["height"], p["shelves"]
        frame, binc = p["frame"], p["bin"]
        L = nb * W + 4.0
        s = new_shapes()
        x0, z0, z1 = -L / 2, -D / 2, D / 2
        up = 4.0
        for k in range(nb + 1):                                              # angle uprights, front and back
            x = x0 + k * W
            for z in (z0, z1 - up):
                s.box((x, 0, z), (x + up, H, z + up), frame)
            y = 25.0
            while y < H - 20:                                                # side bracing
                s.cylinder((x + up / 2, y, z0 + up), (x + up / 2, min(y + 50, H - 10), z1 - up), 0.8, frame, 6)
                y += 50.0
        levels = [12.0 + k * (H - 30.0) / max(ns - 1, 1) for k in range(ns)]
        attach = {}
        bw = p["bin_width"]
        for b in range(nb):
            bx0, bx1 = x0 + b * W + up, x0 + (b + 1) * W
            for lv, y in enumerate(levels):
                s.box((bx0, y - 2.5, z0), (bx1, y, z1), "galvanised")         # shelf panel
                s.box((bx0, y - 4.5, z1 - 1), (bx1, y, z1), frame)            # front lip
                if lv == len(levels) - 1:
                    continue                                                  # top shelf: no bins
                gap = levels[lv + 1] - y if lv + 1 < len(levels) else 30.0
                bh = min(gap - 6.0, p["bin_height"])
                n = max(1, int((bx1 - bx0 - 2) // (bw + 1.5)))
                pitch = (bx1 - bx0) / n
                for q in range(n):
                    cx = bx0 + (q + 0.5) * pitch
                    a, c = cx - bw / 2, cx + bw / 2
                    bz0, bz1 = z0 + 3, z1 - 3
                    s.box((a, y, bz0), (c, y + 1.2, bz1), binc)               # bin: floor, walls, low open front
                    s.box((a, y, bz0), (a + 1.2, y + bh, bz1), binc)
                    s.box((c - 1.2, y, bz0), (c, y + bh, bz1), binc)
                    s.box((a, y, bz0), (c, y + bh, bz0 + 1.2), binc)
                    s.box((a, y, bz1 - 1.2), (c, y + bh * 0.45, bz1), binc)
                    s.box((a + 2, y + 1.2, bz0 + 2), (c - 2, y + bh * 0.35, bz1 - 4), "steel_dark")   # parts in it
                    s.box((cx - bw * 0.3, y + bh * 0.2, bz1 + 0.01), (cx + bw * 0.3, y + bh * 0.38, bz1 + 0.3),
                          "cardboard")                                                               # label
                    attach[f"B{b + 1}_L{lv + 1}_{q + 1}"] = (cx, y + bh * 0.4, bz1 - 6)
        shapes_to_stage(stage, path, s, lib)
        return attach


register(BinRack("bin_rack", "bin_rack", "Bin rack (shelving with picking bins)", {
    "bays": Param(1, "bays side by side", 1, 10),
    "width": Param(100.0, "bay width", 60, 150, "cm"),
    "depth": Param(50.0, "shelf depth", 30, 80, "cm"),
    "height": Param(200.0, "upright height", 100, 300, "cm"),
    "shelves": Param(6, "shelf levels including the top", 2, 10),
    "bin_width": Param(20.0, "bin width", 10, 45, "cm"),
    "bin_height": Param(18.0, "bin height (capped by the shelf gap)", 8, 40, "cm"),
    "frame": Param("powder_grey", "frame finish"),
    "bin": Param("plastic_blue", "bin colour"),
}))


# ------------------------------------------------------------------------------------------ hook block
def _hook(s, x, y_hook, z, rope_top, scale=1.0):
    """Wire rope from rope_top down to a hook block whose hook eye is at y_hook."""
    k = scale
    s.cylinder((x, y_hook + 38 * k, z), (x, rope_top, z), 0.7 * k, "steel_dark", 6)
    s.box((x - 9 * k, y_hook + 22 * k, z - 6 * k), (x + 9 * k, y_hook + 38 * k, z + 6 * k), "powder_yellow")   # block
    s.cylinder((x, y_hook + 14 * k, z), (x, y_hook + 22 * k, z), 2.2 * k, "steel_dark", 10)               # shank
    seg = 8
    pts = []
    for i in range(seg + 1):                                             # the hook: a 220 deg arc in the XY plane
        a = math.radians(90 + i * 220 / seg)
        pts.append((x + 6 * k * math.cos(a) + 0 * k, y_hook + 6 * k + 6 * k * math.sin(a), z))
    pts[0] = (x, y_hook + 14 * k, z)
    for a, b in zip(pts, pts[1:]):
        s.cylinder(a, b, 1.8 * k, "powder_red" if b is pts[-1] else "steel_dark", 8)


# ------------------------------------------------------------------------------------------ jib crane
class JibCrane(Item):
    """Floor-mounted pillar jib: column on a base plate, I-beam boom with tie-rod, trolley + electric chain hoist."""

    def footprint(self, p):
        r = p["boom"]
        return 2 * r, 2 * r                                                 # the swing circle, as drawings show it

    def build(self, stage, path, p, lib):
        R, Hb, yaw = p["boom"], p["height"], math.radians(p["slew"])
        c = p["colour"]
        s = new_shapes()
        col_r = 14.0 if R <= 500 else 18.0
        s.box((-45, 0, -45), (45, 4, 45), "steel_dark")                              # base plate + anchors
        for dx in (-35, 35):
            for dz in (-35, 35):
                s.cylinder((dx, 4, dz), (dx, 9, dz), 2.0, "galvanised", 8)
        s.cylinder((0, 4, 0), (0, Hb + 40, 0), col_r, c, 20)                         # pillar
        s.cylinder((0, Hb + 40, 0), (0, Hb + 55, 0), col_r + 4, "steel_dark", 20)   # slewing head
        d = (math.cos(yaw), -math.sin(yaw))                                          # boom direction in XZ

        def P(r, y, off=0.0):
            return (d[0] * r - d[1] * off, y, d[1] * r + d[0] * off)
        hb = 30.0 if R <= 500 else 40.0                                              # I-beam: flanges + web
        for y0, y1 in ((Hb, Hb + 2.5), (Hb + hb - 2.5, Hb + hb)):
            self._beam(s, d, 0, R, y0, y1, 9.0, c)
        self._beam(s, d, 0, R, Hb + 2.5, Hb + hb - 2.5, 0.8, c)
        s.cylinder(P(0, Hb + hb + 120), P(R * 0.8, Hb + hb), 2.0, c, 8)               # tie-rod to the column top
        s.cylinder((0, Hb + hb, 0), (0, Hb + hb + 125, 0), col_r * 0.6, c, 16)        # mast for the tie-rod
        s.box((d[0] * R - 2, Hb - 8, d[1] * R - 2), (d[0] * R + 2, Hb + hb, d[1] * R + 2), "rubber_black")   # end stop
        t = max(60.0, min(R - 30.0, R * p["trolley"]))
        tx, tz = d[0] * t, d[1] * t
        s.box((tx - 14, Hb - 10, tz - 14), (tx + 14, Hb + 3, tz + 14), "steel_dark")  # trolley
        s.box((tx - 16, Hb - 48, tz - 13), (tx + 16, Hb - 10, tz + 13), c)            # hoist body
        s.box((tx + 16, Hb - 44, tz - 10), (tx + 26, Hb - 16, tz + 10), "steel_dark") # motor / gearbox
        hook_y = max(20.0, min(Hb - 80.0, p["hook_height"]))
        _hook(s, tx, hook_y, tz, Hb - 48)
        s.cylinder((tx - 12, Hb - 48, tz + 8), (tx - 12, 120, tz + 8), 0.5, "plastic_black", 5)   # pendant cable
        s.box((tx - 16, 100, tz + 4), (tx - 8, 122, tz + 12), "powder_yellow")
        shapes_to_stage(stage, path, s, lib)
        return {"hook": (tx, hook_y, tz), "column_top": (0, Hb + 55, 0), "boom_tip": (d[0] * R, Hb, d[1] * R)}

    @staticmethod
    def _beam(s, d, r0, r1, y0, y1, half_w, mat):
        cx, cz = d[0] * (r0 + r1) / 2, d[1] * (r0 + r1) / 2
        n = (-d[1], d[0])
        s.obox((cx, (y0 + y1) / 2, cz), (r1 - r0, y1 - y0, 2 * half_w),
               ((d[0], 0.0, d[1]), (0.0, 1.0, 0.0), (n[0], 0.0, n[1])), mat)


register(JibCrane("jib_crane", "crane_jib", "Pillar jib crane with chain hoist", {
    "boom": Param(400.0, "boom length (swing radius)", 150, 1000, "cm"),
    "height": Param(400.0, "height under the boom", 200, 800, "cm"),
    "slew": Param(0.0, "boom direction, degrees from +X (toward -Z is +90)", -360, 360, "deg"),
    "trolley": Param(0.7, "trolley position along the boom (0-1)", 0, 1),
    "hook_height": Param(180.0, "hook height above the floor", 20, 800, "cm"),
    "colour": Param("powder_yellow", "crane finish"),
    "capacity": Param("", "rated load as on the drawing, e.g. 2000lb"),
}))


# ------------------------------------------------------------------------------------------ bridge crane
class BridgeCrane(Item):
    """Freestanding workstation bridge crane: its own columns and runways, a bridge girder on end trucks, trolley and
    hoist. Span along X, runway along Z."""

    def footprint(self, p):
        return p["span"] + 50.0, p["runway"]                                   # base plates included

    def build(self, stage, path, p, lib):
        S, Rw, H = p["span"], p["runway"], p["height"]
        c, colc = p["colour"], p["column_colour"]
        s = new_shapes()
        nbay = max(1, math.ceil(Rw / p["column_spacing"]))
        zs = [-Rw / 2 + 25 + k * (Rw - 50) / nbay for k in range(nbay + 1)]       # base plates inside the runway length
        xs = (-S / 2, S / 2)
        rb = 30.0                                                                     # runway beam depth
        for x in xs:
            for z in zs:
                s.box((x - 25, 0, z - 25), (x + 25, 2.5, z + 25), "steel_dark")        # base plate
                s.box((x - 10, 2.5, z - 10), (x + 10, H, z + 10), colc)               # column
                s.box((x - 14, H - 4, z - 14), (x + 14, H, z + 14), colc)              # cap
            s.box((x - 8, H, -Rw / 2), (x + 8, H + 2, Rw / 2), colc)                  # runway: bottom flange
            s.box((x - 1, H + 2, -Rw / 2), (x + 1, H + rb - 2, Rw / 2), colc)
            s.box((x - 8, H + rb - 2, -Rw / 2), (x + 8, H + rb, Rw / 2), colc)
            for z in zs:                                                              # knee braces
                for sg in ((1,) if z == zs[0] else (-1,) if z == zs[-1] else (-1, 1)):
                    s.cylinder((x, H - 90, z), (x, H, z + sg * 90), 2.5, colc, 6)
        for z in (zs[0], zs[-1]):                                                     # end ties between runways
            s.box((xs[0], H - 20, z - 6), (xs[1], H - 8, z + 6), colc)
        bz = -Rw / 2 + 50 + (Rw - 100) * p["bridge"]
        top = H + rb
        for x in xs:                                                                  # end trucks on the runways
            s.box((x - 15, top, bz - 45), (x + 15, top + 22, bz + 45), c)
            for wz in (bz - 32, bz + 32):
                s.cylinder((x - 9, top + 6, wz), (x + 9, top + 6, wz), 6.0, "steel_dark", 12)
        gd = 45.0 if S <= 1200 else 60.0                                               # bridge girder (box)
        s.box((xs[0] - 10, top + 22, bz - 14), (xs[1] + 10, top + 22 + gd, bz + 14), c)
        s.box((xs[0] - 10, top + 22 + gd, bz - 18), (xs[1] + 10, top + 25 + gd, bz + 18), c)
        s.box((xs[0] + 20, top + 30, bz + 14), (xs[0] + 50, top + 60, bz + 30), "powder_grey")   # control panel
        tx = -S / 2 + 60 + (S - 120) * p["trolley"]
        s.box((tx - 30, top + 22 + gd, bz - 32), (tx + 30, top + 40 + gd, bz + 32), "steel_dark")  # top-running trolley
        s.box((tx - 22, top - 25, bz + 18), (tx + 22, top + 35 + gd, bz + 36), c)                  # hoist hanging beside
        s.cylinder((tx, top - 25, bz + 27), (tx, top - 45, bz + 27), 12.0, "steel_dark", 14)      # rope drum
        hook_y = max(30.0, min(H - 80.0, p["hook_height"]))
        _hook(s, tx, hook_y, bz + 27, top - 45, scale=1.3)
        s.cylinder((tx + 18, top - 25, bz + 34), (tx + 18, 125, bz + 34), 0.5, "plastic_black", 5)
        s.box((tx + 13, 100, bz + 30), (tx + 23, 127, bz + 40), "powder_yellow")                   # pendant
        shapes_to_stage(stage, path, s, lib)
        return {"hook": (tx, hook_y, bz + 27), "bridge": (0, top, bz)}


register(BridgeCrane("bridge_crane", "crane_bridge", "Freestanding bridge (workstation) crane", {
    "span": Param(800.0, "distance between runways", 300, 3000, "cm"),
    "runway": Param(1200.0, "runway length", 300, 6000, "cm"),
    "height": Param(550.0, "runway beam underside height", 300, 1200, "cm"),
    "column_spacing": Param(600.0, "max distance between runway columns", 300, 1200, "cm"),
    "bridge": Param(0.5, "bridge position along the runway (0-1)", 0, 1),
    "trolley": Param(0.5, "trolley position along the bridge (0-1)", 0, 1),
    "hook_height": Param(200.0, "hook height above the floor", 30, 1200, "cm"),
    "colour": Param("powder_yellow", "bridge, trucks, hoist"),
    "column_colour": Param("powder_grey", "columns and runways"),
    "capacity": Param("", "rated load as on the drawing, e.g. 4000lb"),
}))
