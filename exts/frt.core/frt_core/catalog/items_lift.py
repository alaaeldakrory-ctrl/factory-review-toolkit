"""Scissor lift table (workstation table that a unit is built on), modelled the way they are built and rigged so
it can be animated:

- base: box-section side rails and cross members, fixed-pivot brackets at one end, roller tracks at the other,
  lifting / tow lugs; `mobile` on castors (swivel + brake at the handle end) with a tow handle, or `fixed` on the
  floor (or in a pit)
- scissor: one scissor pair per side, outer arms pivot on the base and roll under the deck, inner arms pivot on
  the deck and roll on the base; cross tubes, centre pins, rollers; a hydraulic cylinder from a base cross tube to
  the outer arms
- deck: steel deck plate with a black rubber mat and yellow edge strips (or a plain painted / tread deck), apron
  on all sides with black / yellow hazard stripes, capacity plate
- power unit (mobile): oil tank, finned motor, control box with emergency stop, up / down buttons and pilot lamp,
  hydraulic hose to the cylinder

Rig: Deck (rises), Arm_A / Arm_B (rotate about their pivots), Cylinder (swings) / Cylinder/Rod (extends); the
geometry values the rig needs are frt:rig:* attributes on the item. frt_core.rig.lift_table.pose() / animate()
move it and carry what stands on it (frt:on). Local frame: length along X (handle end +X), width along Z, Y up.
"""
import math

from pxr import Gf, Sdf, UsdGeom

from .base import Item, Param, new_shapes, register, shapes_to_stage
from .items_crane import finned_motor, nameplate, sign_text
from . import details as DT
from .items_flow import poly

BOLT = "bolt_zinc"
DROP = 3.5                     # the cylinder tube hangs this far below the outer arms
UNIT = 30.0                    # length of the power-unit bay at the handle end (mobile, onboard power)


def clip_x(pts, x0, x1):
    """Clip a polygon (2-D points) to x0 <= x <= x1 (Sutherland-Hodgman, two planes)."""
    def cut(poly_, keep, xc):
        out = []
        for i, a in enumerate(poly_):
            b = poly_[(i + 1) % len(poly_)]
            ia, ib = keep(a[0]), keep(b[0])
            if ia:
                out.append(a)
            if ia != ib:
                t = (xc - a[0]) / (b[0] - a[0])
                out.append((xc, a[1] + t * (b[1] - a[1])))
        return out
    pts = cut(pts, lambda x: x >= x0, x0)
    return cut(pts, lambda x: x <= x1, x1) if pts else pts


def hazard(s, origin, along, up, out, length, height, w=7.0, mat="powder_black"):
    """45-degree black stripes on a yellow band: origin = lower-left corner of the face."""
    x = -height
    while x < length:
        quad = clip_x([(x, 0.0), (x + w, 0.0), (x + w + height, height), (x + height, height)], 0.0, length)
        if len(quad) >= 3:
            poly(s, [tuple(o + u * a + v * b + n * 0.04 for o, a, b, n in zip(origin, along, up, out))
                     for u, v in quad], out, mat)
        x += 2 * w


class Geo:
    """Scissor geometry for a deck height h (deck top)."""

    def __init__(self, x_f, y_b, top_off, La, s_att, ax, ay, rod0):
        self.x_f, self.y_b, self.top_off, self.La, self.s_att = x_f, y_b, top_off, La, s_att
        self.ax, self.ay, self.rod0 = ax, ay, rod0

    def solve(self, h):
        d = h - self.top_off - self.y_b
        th = math.asin(max(0.0, min(0.99, d / self.La)))
        c, s_ = math.cos(th), math.sin(th)
        bx, by = self.x_f + self.s_att * c + DROP * s_, self.y_b + self.s_att * s_ - DROP * c
        dx, dy = bx - self.ax, by - self.ay
        return {"theta": math.degrees(th), "a": (self.x_f, self.y_b), "b": (self.x_f, h - self.top_off),
                "phi": math.degrees(math.atan2(dy, dx)), "ext": math.hypot(dx, dy) - self.rod0,
                "roll": self.x_f + self.La * math.cos(th)}


def rig_xf(stage, path, t=(0, 0, 0), rz=None):
    x = UsdGeom.Xform.Define(stage, path)
    x.ClearXformOpOrder()
    x.AddTranslateOp().Set(Gf.Vec3d(*t))
    if rz is not None:
        x.AddRotateZOp().Set(float(rz))
    return x


def castor(s, x, z, top, swivel, col):
    """Industrial castor under the base at (x, z): top plate, fork, polyurethane wheel; swivel ones have a
    kingpin, bearing ring and a brake pedal."""
    r, wy = 6.25, 6.25
    s.rbox((x, top - 0.4, z), (10, 0.8, 9), 0.3, col, seg=1)
    for dx in (-3.5, 3.5):
        for dz in (-3.2, 3.2):
            s.bolt((x + dx, top, z + dz), (0, 1, 0), 0.8, BOLT)
    if swivel:
        s.lathe((x, top - 0.8, z), (x, top - 2.4, z), [(0, 3.6), (1, 3.6)], "steel_machined", sides=20)
    off = 2.5 if swivel else 0.0                                   # swivel lead
    for uz in (-1, 1):
        s.rbox((x - off / 2, (top - 2.4 + wy) / 2, z + uz * 2.9), (6.5, top - 2.4 - wy + 4, 0.6), 0.25, col, seg=1)
    s.lathe((x - off, wy, z - 2.2), (x - off, wy, z + 2.2), [(0, r - 0.6), (0.15, r), (0.85, r), (1, r - 0.6)],
            "rubber_black", sides=22)
    s.lathe((x - off, wy, z - 2.4), (x - off, wy, z + 2.4), [(0, 2.6), (1, 2.6)], "plastic_grey", sides=14)   # hub
    for uz in (-1, 1):
        s.bolt((x - off, wy, z + uz * 3.3), (0, 0, uz), 1.0, BOLT)
    if swivel:
        s.rbox((x + 3.5, wy + 3.5, z), (3.0, 0.8, 5.0), 0.3, "powder_red", seg=1)                 # brake pedal


def power_unit(s, L, W, y1, g):
    """Hydraulic power pack on the handle-end bay: tank on rubber mounts with a strap, manifold block with relief
    valve, solenoid valve and breather, flanged motor with nameplate, pressure hose to the lift cylinder."""
    px = L / 2 - UNIT / 2
    yp = y1 + 1.0
    s.rbox((px, y1 + 0.5, 0), (UNIT - 4, 1.0, W - 12), 0.3, "steel_dark", seg=1)                       # mounting plate
    for dx in (-UNIT / 2 + 4, UNIT / 2 - 4):
        for dz in (-W / 2 + 9, W / 2 - 9):
            s.bolt((px + dx, yp, dz), (0, 1, 0), 1.0, BOLT)
    tx, tz = px - 3.0, -W / 4
    for k in range(3):                                                                               # rubber mounts
        a = 2 * math.pi * k / 3 + 0.5
        mx, mz = tx + 4.5 * math.cos(a), tz + 4.5 * math.sin(a)
        s.lathe((mx, yp, mz), (mx, yp + 2.0, mz), [(0, 1.6), (0.5, 1.8), (1, 1.6)], "rubber_black", sides=14)
    y_t0, y_t1 = yp + 2.0, yp + 26.0
    s.lathe((tx, y_t0, tz), (tx, y_t1, tz), [(0, 6.6), (0.02, 7.3), (0.05, 7.4), (0.95, 7.4), (0.98, 7.3), (1, 6.6)],
            "aluminium", sides=36)                                                                   # oil tank
    for y_ in (y_t0 + 0.9, y_t1 - 0.9):
        DT.weld_bead(s, (tx, y_, tz), (0, 1, 0), 7.35, "aluminium", bead=0.22)
    s.lathe((tx, yp + 11.0, tz), (tx, yp + 13.5, tz), [(0, 7.55), (1, 7.55)], "stainless", sides=36)    # strap
    s.rbox((tx + 7.9, yp + 12.2, tz), (1.4, 2.5, 3.0), 0.3, "stainless", seg=1)
    s.bolt((tx + 8.6, yp + 12.2, tz), (1, 0, 0), 0.8, BOLT)
    DT.hex_nut(s, (tx, y_t0 + 2.0, tz + 7.3), (0, 0, 1), 1.6, 0.7, BOLT)                               # drain plug
    s.lathe((tx + 7.25, yp + 7.0, tz), (tx + 7.75, yp + 7.0, tz), [(0, 1.3), (1, 1.2)], "chrome", sides=18)   # sight glass
    s.lathe((tx + 7.6, yp + 7.0, tz), (tx + 7.8, yp + 7.0, tz), [(0, 0.85), (1, 0.85)], "brass", sides=14)
    ym = y_t1 + 3.0                                                                                  # manifold block
    s.rbox((tx, ym, tz), (13.0, 6.0, 13.0), 0.8, "cast_aluminium", seg=2)
    for dx, dz in ((-5, -5), (-5, 5), (5, -5), (5, 5)):
        s.bolt((tx + dx, ym + 3.0, tz + dz), (0, 1, 0), 0.7, BOLT)
    e = DT.fitting(s, (tx - 6.5, ym, tz), (-1, 0, 0), 1.4)                                            # relief valve
    knurl = [(0, 0.9)] + [v for k in range(5) for v in ((0.1 + k * 0.18, 1.0), (0.19 + k * 0.18, 0.9))] + [(1, 0.8)]
    s.lathe(e, (e[0] - 2.5, e[1], e[2]), knurl, "steel_machined", sides=16)
    q = DT.hex_nut(s, (tx, ym, tz + 6.5), (0, 0, 1), 2.0, 0.8, BOLT)                                  # solenoid valve
    s.lathe(q, (q[0], q[1], q[2] + 5.0), [(0, 2.3), (0.05, 2.5), (0.95, 2.5), (1, 2.2)], "plastic_black", sides=24)
    s.rbox((tx, ym + 3.4, q[2] + 2.5), (3.4, 2.2, 3.4), 0.4, "plastic_black", seg=1)                   # DIN plug
    s.bolt((tx, ym + 4.5, q[2] + 2.5), (0, 1, 0), 0.5, "stainless")
    ribs = [(0, 1.6)] + [v for k in range(4) for v in ((0.15 + k * 0.2, 1.75), (0.25 + k * 0.2, 1.6))] + [(1, 1.2)]
    s.lathe((tx + 4.0, ym + 3.0, tz - 4.0), (tx + 4.0, ym + 5.5, tz - 4.0), ribs, "plastic_black", sides=18)   # breather
    s.lathe((tx, ym + 3.0, tz), (tx, ym + 4.2, tz), [(0, 6.4), (1, 6.4)], "cast_aluminium", sides=32)  # motor flange
    for k in range(4):
        a = 2 * math.pi * k / 4 + math.pi / 4
        s.bolt((tx + 5.4 * math.cos(a), ym + 4.2, tz + 5.4 * math.sin(a)), (0, 1, 0), 0.7, BOLT)
    finned_motor(s, (tx, ym + 4.2, tz + 0.01), (tx, ym + 20.0, tz + 0.02), 5.5, "powder_anthracite", "powder_black")
    nameplate(s, (tx - 0.01, ym + 11.5, tz - 6.4), (0, 0, -1), (1, 0, 0), 5.5, 3.4)
    fx = DT.fitting(s, (tx + 6.5, ym - 1.0, tz), (1, 0, 0), 1.4)                                      # pressure port
    DT.hose(s, [fx, (fx[0] + 3, fx[1] - 4, fx[2]), (fx[0] + 1, y1 + 3.0, tz + 10), (g.ax + 22, y1 - 3.0, -2.0),
                (g.ax + 9, g.ay + 2.5, -1.0)], 0.75)
    return (tx, ym, q[2] + 2.5)


def control_box(s, L, W, y1):
    """Control box on a bracket: IP65 enclosure, E-stop, power lamp, key switch, UP / DOWN buttons with legends,
    warning label, cable glands with the motor, valve and supply cables."""
    px = L / 2 - UNIT / 2
    bx, by, bz = px + 3.0, y1 + 24.0, W / 4 - 1.0
    r, u, n = (0, 0, -1), (0, 1, 0), (1, 0, 0)
    w, h, d = 15.0, 25.0, 8.0
    p = (bx - d / 2, by, bz)
    s.rbox((bx - d / 2 - 0.4, y1 + 12.0, bz), (0.8, 24.0, 6.0), 0.2, "steel_dark", seg=1)              # bracket
    s.rbox((bx - d / 2 - 0.4, y1 + 1.4, bz), (6.0, 0.8, 6.0), 0.2, "steel_dark", seg=1)
    for dy in (-6, 6):
        s.bolt((bx - d / 2 - 0.8, by + dy, bz), (-1, 0, 0), 0.6, BOLT)
    face = DT.enclosure(s, p, r, u, n, w, h, d)
    DT.estop(s, DT.at(face, r, u, n, 0, 7.0), r, u, n, 3.6)
    DT.pilot_lamp(s, DT.at(face, r, u, n, -3.8, 1.2), r, u, n, "lamp_white")
    DT.legend(s, DT.at(face, r, u, n, -3.8, -1.3), r, u, n, "POWER", 0.6)
    DT.key_switch(s, DT.at(face, r, u, n, 3.8, 1.2), r, u, n)
    DT.legend(s, DT.at(face, r, u, n, 3.8, -1.3), r, u, n, "ON", 0.6)
    DT.push_button(s, DT.at(face, r, u, n, -3.6, -5.0), r, u, n, "floor_green", arrow=1)
    DT.legend(s, DT.at(face, r, u, n, -3.6, -7.6), r, u, n, "UP", 0.7)
    DT.push_button(s, DT.at(face, r, u, n, 3.6, -5.0), r, u, n, "plastic_black", arrow=-1)
    DT.legend(s, DT.at(face, r, u, n, 3.6, -7.6), r, u, n, "DOWN", 0.7)
    DT.warning(s, DT.at(face, r, u, n, 0, -10.2, 0.05), r, n, 2.6)
    yb = by - h / 2
    tx, ym, tz = L / 2 - UNIT / 2 - 3.0, y1 + 30.0, -W / 4
    DT.gland(s, (bx - 1.0, yb, bz + 3.5), (0, -1, 0), 1.6,
             [(bx - 1.0, y1 + 3.0, bz + 3.5), (tx + 9.5, y1 + 3.0, tz + 9.0), (tx + 9.0, ym + 2.0, tz + 7.5),
              (tx + 8.8, ym + 8.0, tz + 6.8), (tx + 8.7, ym + 12.1, tz + 5.6)],
             cable_r=0.45)                                                                           # to the motor
    DT.gland(s, (bx - 1.0, yb, bz), (0, -1, 0), 1.4,
             [(bx - 1.0, y1 + 3.5, bz), (tx + 3.0, y1 + 3.5, tz + 12.5), (tx + 2.0, ym + 1.0, tz + 12.5),
              (tx + 0.5, ym + 4.4, tz + 11.2)],
             cable_r=0.35)                                                                           # to the valve
    DT.gland(s, (bx - 1.0, yb, bz - 3.5), (0, -1, 0), 1.6,
             [(bx - 1.0, by - 18, bz - 3.5), (bx + 3, y1 + 6, bz - 7), (L / 2 - 1.0, y1 + 5, bz - 9),
              (L / 2 + 2.0, y1 + 14, bz - 10)], cable_r=0.5)                                         # supply cable
    s.rbox((L / 2 + 2.0, y1 + 15.5, bz - 10), (2.6, 3.0, 4.0), 0.6, "plastic_black", seg=2)            # plug on its hook


def tow_handle(s, L, y1, col):
    """Tow / push handle: clevis on the base, pivot bolt with nuts, bent tube yoke welded to a boss, crossbar with
    ribbed grips, return spring."""
    hx = L / 2 - 2.0
    py = y1 + 4.0
    for uz in (-1, 1):                                                                               # clevis plates
        s.rbox((hx, py - 0.5, uz * 4.2), (5.0, 8.0, 0.8), 0.3, col, seg=1)
    s.rbox((hx - 1.5, y1 + 0.4, 0), (6.0, 0.8, 10.0), 0.3, col, seg=1)
    s.cylinder((hx, py, -5.6), (hx, py, 5.6), 0.8, "steel_machined", 12)                              # pivot bolt
    DT.hex_nut(s, (hx, py, 4.6), (0, 0, 1), 1.9, 0.9, BOLT)
    DT.hex_nut(s, (hx, py, -4.6), (0, 0, -1), 1.9, 0.7, BOLT)
    s.lathe((hx, py, -3.6), (hx, py, 3.6), [(0, 2.0), (0.1, 2.3), (0.9, 2.3), (1, 2.0)], "powder_black_tex", sides=20)
    top = (hx + 26.0, y1 + 50.0)
    for uz in (-1, 1):
        start = (hx + 1.6, py + 1.2, uz * 2.6)
        path = DT.bend([start, (hx + 10, y1 + 18, uz * 5.5), (hx + 22, y1 + 40, uz * 4.5), (top[0], top[1], uz * 3.2)], 5.0)
        s.sweep(path, 1.25, "powder_black_tex", sides=16)
        d0 = tuple(b_ - a_ for a_, b_ in zip(path[0], path[1]))
        DT.weld_bead(s, start, _unit(d0), 1.25, "powder_black_tex")
        DT.weld_bead(s, (top[0], top[1] - 1.2, uz * 3.2), (0, 1, 0), 1.25, "powder_black_tex")
    s.cylinder((top[0], top[1], -14.5), (top[0], top[1], 14.5), 1.25, "powder_black_tex", 16)         # crossbar
    for uz in (-1, 1):
        DT.grip(s, (top[0], top[1], uz * 5.0), (top[0], top[1], uz * 15.2), 1.25)
    DT.spring(s, (hx + 5.0, py + 6.0, 0), (hx - 4.0, y1 + 1.0, 0), 0.7, 0.16, 9)                       # return spring


def _unit(v):
    m = math.sqrt(sum(c * c for c in v)) or 1.0
    return tuple(c / m for c in v)


class ScissorTable(Item):
    def footprint(self, p):
        return p["length"], p["width"]

    def geo(self, p):
        """Layout and scissor geometry from the parameters (shared by build and the rig)."""
        L, W = p["length"], p["width"]
        mobile = str(p["style"]).lower().startswith("m")
        unit = mobile and str(p["power"]).lower() == "onboard"
        Ld = L - UNIT if unit else L
        dcx = -UNIT / 2 if unit else 0.0
        y0 = 9.0 if mobile else 1.0                                    # base rails bottom / top
        y1 = y0 + 10.0
        ap = 12.0 if mobile else 16.0                                  # deck apron height
        top_off = 1.8 + 5.0                                            # deck top -> top pivots
        y_b = y1 - 4.0
        x0 = dcx - Ld / 2
        La = min(0.82 * Ld, Ld - 18.0)
        x_f = x0 + 9.0
        ax, ay = x_f + 0.65 * La, y0 + 4.5
        s_att = 0.35 * La
        h_min = max(float(p["lowered"]), y1 + 1.8 + ap + 0.6, y_b + top_off + 9.0)
        h_max = min(max(float(p["raised"]), h_min + 5), y_b + top_off + La * math.sin(math.radians(40)))
        g = Geo(x_f, y_b, top_off, La, s_att, ax, ay, 0.0)
        lo = g.solve(h_min)
        dmin = lo["ext"]                                               # rod0 = 0 here: the pin distance
        g.rod0 = dmin
        return dict(L=L, W=W, Ld=Ld, dcx=dcx, x0=x0, mobile=mobile, unit=unit, y0=y0, y1=y1, ap=ap,
                    h_min=h_min, h_max=h_max, geo=g, barrel=dmin - 5.0)

    def build(self, stage, path, p, lib):
        G = self.geo(p)
        L, W, Ld, dcx, x0 = G["L"], G["W"], G["Ld"], G["dcx"], G["x0"]
        y0, y1, ap, g = G["y0"], G["y1"], G["ap"], G["geo"]
        col = p["color"]
        h = G["h_min"] + (G["h_max"] - G["h_min"]) * float(p["lift"])
        sol = g.solve(h)
        za, zb = W / 2 - 9.0, W / 2 - 14.0                           # outer / inner arm planes
        s = new_shapes()
        # ------------------------------------------------------------------ base frame
        bx0, bx1 = (-L / 2 + 1, L / 2 - 1) if G["mobile"] else (x0 + 2, x0 + Ld - 2)
        yc = (y0 + y1) / 2
        for uz in (-1, 1):
            s.rprism((bx0, yc, uz * (W / 2 - 4.5)), (bx1, yc, uz * (W / 2 - 4.5)), 5.0, 10.0, 0.8, col)    # side rails
            for lx in (bx0 + 0.25 * (bx1 - bx0), bx0 + 0.75 * (bx1 - bx0)):                                # lugs
                s.rbox((lx, y1 - 1.0, uz * (W / 2 + 0.6)), (7.0, 7.0, 1.2), 0.4, col, seg=1)
                s.torus((lx, y1 - 0.5, uz * (W / 2 + 0.6)), (0, 0, 1), 2.0, 0.6, col, seg=12, sides=6)
            s.rbox(((g.x_f + bx1) / 2, g.y_b - 4.2, uz * (zb - 3.6)), (bx1 - g.x_f, 0.6, 4.0), 0.2, "steel_dark",
                   seg=1)                                                                                   # roller track
            for dz in (-1, 1):                                                                              # pivot brackets
                s.rbox((g.x_f, g.y_b, uz * (za + dz * 2.6)), (9.0, 10.0, 0.8), 0.3, col, seg=1)
        for x in (bx0 + 2.5, bx1 - 2.5):
            s.rprism((x, yc, -W / 2 + 2), (x, yc, W / 2 - 2), 5.0, 10.0, 0.8, col)                          # end members
        s.rprism((g.ax, y0 + 1.0, -W / 2 + 2), (g.ax, y0 + 1.0, W / 2 - 2), 6.0, 2.0, 0.4, col, seg=1)    # low cross member
        s.cylinder((g.ax, g.ay, -zb + 4), (g.ax, g.ay, zb - 4), 2.6, col, 16)                               # anchor tube
        for dz in (-4.5, 4.5, -zb + 5, zb - 5):
            s.rbox((g.ax, (g.ay + y0 + 2) / 2, dz), (7.0, g.ay - y0 + 2, 0.8), 0.3, col, seg=1)             # clevis / lugs
        s.cylinder((g.x_f, g.y_b, -W / 2 + 4), (g.x_f, g.y_b, W / 2 - 4), 1.4, "steel_machined", 12)        # pivot axle
        if G["mobile"]:
            for cx in (bx0 + 7, bx1 - 7):
                for uz in (-1, 1):
                    castor(s, cx, uz * (W / 2 - 4.5), y0, cx > 0, col)
        else:
            for fx in (bx0 + 6, (bx0 + bx1) / 2, bx1 - 6):                                                  # anchor plates
                for uz in (-1, 1):
                    s.rbox((fx, 0.5, uz * (W / 2 - 4.5)), (10.0, 1.0, 9.0), 0.2, "steel_dark", seg=1)
                    s.bolt((fx + 3, 1.0, uz * (W / 2 - 1.0)), (0, 1, 0), 1.0, BOLT)
        # ------------------------------------------------------------------ power unit + handle (mobile)
        if G["unit"]:
            power_unit(s, L, W, y1, g)
            control_box(s, L, W, y1)
            tow_handle(s, L, y1, col)
        shapes_to_stage(stage, path, s, lib)
        # ------------------------------------------------------------------ deck (rises)
        hr = h
        sd = new_shapes()
        dtop = "steel_tread" if p["deck"] == "tread" else col
        sd.rbox((dcx, hr - 1.4, 0), (Ld, 0.8, W), 0.3, dtop, seg=1)                                         # deck plate
        if p["deck"] == "mat":
            sd.rbox((dcx, hr - 0.5, 0), (Ld - 2.0, 1.0, W - 11.0), 0.3, "rubber_black", seg=1)
            for uz in (-1, 1):
                sd.rbox((dcx, hr - 0.5, uz * (W / 2 - 3.2)), (Ld - 2.0, 1.0, 5.0), 0.3, "floor_yellow", seg=1)
        ytop_ap = hr - 1.8
        for uz in (-1, 1):                                                                                  # apron
            sd.rbox((dcx, ytop_ap - ap / 2, uz * (W / 2 - 0.2)), (Ld, ap, 0.4), 0.15, "powder_yellow", seg=1)
            if p["stripes"]:
                hazard(sd, (dcx - uz * Ld / 2, ytop_ap - ap, uz * W / 2), (uz, 0, 0), (0, 1, 0), (0, 0, uz), Ld, ap)
        for ux in (-1, 1):
            xe = dcx + ux * (Ld / 2 - 0.2)
            sd.rbox((xe, ytop_ap - ap / 2, 0), (0.4, ap, W - 0.8), 0.15, "powder_yellow", seg=1)
            if p["stripes"]:
                hazard(sd, (dcx + ux * Ld / 2, ytop_ap - ap, ux * (W / 2 - 0.4)), (0, 0, -ux), (0, 1, 0), (ux, 0, 0),
                       W - 0.8, ap)
        for uz in (-1, 1):                                                                                  # deck channels
            sd.rbox((dcx, ytop_ap - 3.0, uz * (za + 6.5)), (Ld - 4, 6.0, 0.6), 0.2, col, seg=1)
            sd.rbox((dcx, ytop_ap - 0.3, uz * (za + 3.6)), (Ld - 6, 0.6, 4.0), 0.2, "steel_dark", seg=1)    # roller track
            for dz in (-1, 1):
                sd.rbox((g.x_f, ytop_ap - 4.0, uz * (zb + dz * 2.6)), (9.0, 8.0, 0.8), 0.3, col, seg=1)    # deck pivots
        sd.cylinder((g.x_f, hr - g.top_off, -W / 2 + 6), (g.x_f, hr - g.top_off, W / 2 - 6), 1.4, "steel_machined", 12)
        cap = "".join(ch for ch in str(p["capacity"]).upper() if ch.isalnum() or ch == " ").strip()
        if cap:                                                                                             # capacity plate
            wsign = 3.0 * 6 / 7 * len(cap) + 3.0
            zf = W / 2 + 0.05
            cx = dcx + Ld / 2 - wsign / 2 - 6
            sd.rbox((cx, ytop_ap - ap / 2, zf), (wsign, 6.5, 0.2), 0.2, "floor_white", seg=1)
            sign_text(sd, cap, (cx - wsign / 2 + 1.5, ytop_ap - ap / 2 + 1.5, zf), (1, 0, 0), (0, -1, 0), 3.0,
                      "plastic_black", (0, 0, 1))
        deck = rig_xf(stage, f"{path}/Deck")
        shapes_to_stage(stage, f"{path}/Deck", sd, lib)
        # ------------------------------------------------------------------ scissor arms (local: along +X)
        La = g.La
        for name, z_arm, zin, pos, ang in (("Arm_A", za, zb, sol["a"], sol["theta"]),
                                            ("Arm_B", zb, za, sol["b"], -sol["theta"])):
            sa = new_shapes()
            for uz in (-1, 1):
                z = uz * z_arm
                sa.rprism((0, 0, z), (La, 0, z), 4.0, 8.0, 1.0, col, seg=2)
                for xe in (0.0, La):
                    sa.lathe((xe, 0, z - 2.4), (xe, 0, z + 2.4), [(0, 4.0), (1, 4.0)], col, sides=18)       # end bosses
                    sa.bolt((xe, 0, z + uz * 2.4), (0, 0, uz), 1.4, BOLT)
                uo = uz if name == "Arm_A" else -uz                                                          # roller: A out, B in
                rz = z + uo * 3.6
                sa.lathe((La, 0, rz - uo * 1.2), (La, 0, rz + uo * 1.4), [(0, 3.6), (0.15, 3.9), (0.85, 3.9), (1, 3.6)],
                         "steel_machined", sides=20)
            for xt in (0.0, La):
                sa.cylinder((xt, 0, -z_arm + 2), (xt, 0, z_arm - 2), 2.6, col, 16)
            if name == "Arm_A":
                for uz in (-1, 1):                                                                          # centre pins
                    sa.lathe((La / 2, 0, uz * (zb - 2.6)), (La / 2, 0, uz * (za + 2.6)), [(0, 1.8), (1, 1.8)],
                             "steel_machined", sides=14)
                    sa.lathe((La / 2, 0, uz * (za + 2.6)), (La / 2, 0, uz * (za + 3.6)), [(0, 3.2), (1, 2.6)], BOLT, sides=6)
                # cylinder tube hung below the outer arms (clear of the inner arms when the table is down)
                sa.cylinder((g.s_att, -DROP, -za + 2), (g.s_att, -DROP, za - 2), 2.2, col, 16)
                for dz in (-za + 0.5, za - 0.5, -4.5, 4.5):
                    sa.rbox((g.s_att, -DROP / 2, dz), (6.0, DROP + 4.0, 0.8), 0.3, col, seg=1)
            rig_xf(stage, f"{path}/{name}", (pos[0], pos[1], 0), ang)
            shapes_to_stage(stage, f"{path}/{name}", sa, lib)
        # ------------------------------------------------------------------ hydraulic cylinder (local: along +X)
        Lb = G["barrel"]
        sc = new_shapes()
        sc.lathe((0, 0, -3.8), (0, 0, 3.8), [(0, 2.6), (1, 2.6)], "steel_dark", sides=16)                  # base eye
        sc.lathe((1.5, 0, 0), (Lb, 0, 0), [(0, 3.0), (0.04, 3.9), (0.08, 3.9), (0.1, 3.5), (0.9, 3.5), (0.93, 3.9),
                                          (0.98, 3.9), (1, 2.4)], "powder_black", sides=24)                 # barrel
        sc.lathe((6.0, -3.4, 0), (6.0, -5.2, 0), [(0, 1.0), (1, 0.8)], "steel_machined", sides=10)         # port (up)
        rig_xf(stage, f"{path}/Cylinder", (g.ax, g.ay, 0), sol["phi"])
        shapes_to_stage(stage, f"{path}/Cylinder", sc, lib)
        stroke = g.solve(G["h_max"])["ext"]
        sr = new_shapes()
        sr.cylinder((max(2.0, Lb - stroke - 3.0), 0, 0), (g.rod0 - 1.5, 0, 0), 1.6, "chrome", 18)
        sr.lathe((g.rod0, 0, -3.8), (g.rod0, 0, 3.8), [(0, 2.4), (1, 2.4)], "steel_dark", sides=16)         # rod eye
        rig_xf(stage, f"{path}/Cylinder/Rod", (sol["ext"], 0, 0))
        shapes_to_stage(stage, f"{path}/Cylinder/Rod", sr, lib)
        # ------------------------------------------------------------------ rig values
        prim = stage.GetPrimAtPath(path)
        vals = dict(x_f=g.x_f, y_b=g.y_b, top_off=g.top_off, La=g.La, s_att=g.s_att, ax=g.ax, ay=g.ay, rod0=g.rod0,
                    h_min=G["h_min"], h_max=G["h_max"], h_rest=h)
        for k, v in vals.items():
            prim.CreateAttribute(f"frt:rig:{k}", Sdf.ValueTypeNames.Double).Set(float(v))
        hx = L / 2 + 24 if G["unit"] else L / 2 + 30
        return {"platform": (dcx, h, 0), "handle": (hx, y1 + 50, 0) if G["unit"] else (L / 2, y1, 0)}


register(ScissorTable("lift_table", "lift_table", "Scissor lift table (mobile or fixed), rigged", {
    "length": Param(200.0, "overall length (with the power unit bay on a mobile table)", 80, 400, "cm"),
    "width": Param(100.0, "platform width", 50, 200, "cm"),
    "lowered": Param(35.0, "lowered deck height (raised to what the frame needs)", 15, 80, "cm"),
    "raised": Param(105.0, "raised deck height (capped by the scissor reach)", 40, 200, "cm"),
    "lift": Param(0.0, "current position 0 (down) .. 1 (up)", 0, 1),
    "style": Param("mobile", "mobile (castors, tow handle) / fixed (on the floor)"),
    "power": Param("onboard", "onboard (power unit at the handle end) / remote"),
    "deck": Param("mat", "mat (rubber, yellow edges) / plain / tread"),
    "stripes": Param(True, "hazard stripes on the deck apron"),
    "capacity": Param("1000 KG", "capacity plate text ('' for none)"),
    "color": Param("powder_yellow", "frame finish"),
}))
