"""Carton flow (roller) rack for parts, modular, modelled the way they are built:

- frames at every bay boundary: two roll-formed uprights (open C section with return lips, teardrop holes at a
  2 in pitch on the face, bolt holes on the sides), base plates with anchors, plastic top caps, bolted lacing
  (horizontals + zig-zag diagonals) between the front and back upright
- per level and bay: a front (pick side, low) and a back (load side, high) step beam with end connectors hooked
  into the teardrops and a locking pin; beam heights snap to the hole pitch, so the slope is what the holes give
- lanes on every level, one of three beds: `wheel` (galvanised channel tracks with nylon wheels, lane guides
  between lanes), `roller` (a full-width roller conveyor bed per lane), `ball` (a full-width ball-transfer mat)
- mid-depth support bars under long lanes, an exit stop and a label holder with location labels (code and
  barcode) on the front beam, a load bar at the back, optional upright guards and a top return level that slopes
  back (empties)
- the load: totes (open bins with rim, handle recess and label) or cartons resting against the exit stop; each
  lane's load is its own Xform (Load/L1_B1_N1) so it can be animated

Modular: bays (or bay_widths "120,90,120"), depth, height, levels (or level_heights of the front beams), lanes per
bay, bed per level (bed "wheel" or "wheel,wheel,roller,ball"), slope. Local frame: bays along X, pick side +Z.
"""
import math

from pxr import Gf, Sdf, UsdGeom, Vt

from .base import Item, Param, new_shapes, register, shapes_to_stage
from .items_crane import member, sign_text

BOLT = "bolt_zinc"
PITCH = 5.08                       # teardrop pitch (2 in)
UW, UD, UT = 8.0, 7.0, 0.25        # upright face width (X), depth (Z), sheet thickness
BEAM_D, BEAM_H, STEP = 5.0, 10.0, 1.8
TOP = {"wheel": 3.35, "roller": 4.1, "ball": 2.25}      # height of the rolling surface above the bed bottom


def _v(*c):
    return tuple(float(x) for x in c)


def _add(*vs):
    return tuple(sum(c) for c in zip(*vs))


def _mul(v, k):
    return tuple(c * k for c in v)


def poly(s, pts, normal, mat):
    """A flat convex polygon (decals: punched holes, barcodes)."""
    s._add_face_list(mat, [tuple(p) for p in pts], [tuple(range(len(pts)))], [tuple(normal)])


def teardrop(s, c, right, up, out, w, mat="hole"):
    """Teardrop hole (round top, pointed bottom) of width w centred at c on a face (decal just off the face)."""
    r = w / 2
    cc = _add(c, _mul(up, 0.3 * w), _mul(out, 0.03))
    pts = [_add(cc, _mul(right, r * math.cos(math.radians(a))), _mul(up, r * math.sin(math.radians(a))))
           for a in range(-25, 206, 23)]
    pts.append(_add(c, _mul(up, -0.9 * w), _mul(out, 0.03)))
    poly(s, pts, out, mat)


def round_hole(s, c, right, up, out, r, mat="hole", n=8):
    poly(s, [_add(c, _mul(right, r * math.cos(2 * math.pi * k / n)), _mul(up, r * math.sin(2 * math.pi * k / n)),
                  _mul(out, 0.03)) for k in range(n)], out, mat)


def upright(s, x, zc, sz, H, col):
    """Roll-formed upright centred (x, zc), punched face toward sz (+1 pick side / -1 load side), open behind."""
    zface = zc + sz * (UD / 2 - UT / 2)
    s.rbox((x, H / 2, zface), (UW, H, UT), 0.1, col, seg=1)                                # face
    for ux in (-1, 1):
        s.rbox((x + ux * (UW / 2 - UT / 2), H / 2, zc), (UT, H, UD), 0.1, col, seg=1)       # sides
        s.rbox((x + ux * (UW / 2 - 0.9), H / 2, zc - sz * (UD / 2 - UT / 2)), (1.8, H, UT), 0.1, col, seg=1)  # lips
        s.cylinder((x + ux * (UW / 2 - 0.3), 0.8, zc + sz * (UD / 2 - 0.3)),               # rolled bend
                   (x + ux * (UW / 2 - 0.3), H, zc + sz * (UD / 2 - 0.3)), 0.32, col, 6, caps=False)
    zo = zc + sz * UD / 2
    y = 6.0
    while y < H - 4:
        for dx in (-2.2, 2.2):
            teardrop(s, (x + dx, y, zo), (1, 0, 0), (0, 1, 0), (0, 0, sz), 1.25)
        y += PITCH
    y = 12.0
    while y < H - 6:                                                                       # side bolt holes
        for ux in (-1, 1):
            round_hole(s, (x + ux * UW / 2, y, zc + sz * 1.2), (0, 0, 1), (0, 1, 0), (ux, 0, 0), 0.45)
        y += 2 * PITCH
    s.rbox((x, 0.4, zc), (UW + 6, 0.8, UD + 6), 0.25, col, seg=1)                          # base plate
    for ux in (-1, 1):
        s.bolt((x + ux * (UW / 2 + 1.6), 0.8, zc), (0, 1, 0), 1.0, BOLT)                   # anchors
    s.rbox((x, H + 0.3, zc), (UW + 0.3, 0.6, UD + 0.3), 0.25, "plastic_black", seg=1)      # top cap


def frame_lacing(s, x, zb, zf, H, col):
    """Bolted lacing between the back (zb) and front (zf) upright centres: horizontals and zig-zag diagonals; the
    member ends sit inside the open upright sections."""
    z0, z1 = zb + UD / 2 - 2.5, zf - UD / 2 + 2.5
    nodes = [12.0]
    npan = max(1, round((H - 24) / max(60.0, (z1 - z0) * 0.9)))
    nodes += [12.0 + (H - 24) * (k + 1) / npan for k in range(npan)]
    bolts = []
    for y in (nodes[0], nodes[-1]):
        member(s, (x, y, z0), (x, y, z1), 3.4, 2.6, 0.4, col, up=(1, 0, 0))
        bolts += [(y, z0 + 1.5), (y, z1 - 1.5)]
    for k in range(npan):
        za, zb_ = (z0, z1) if k % 2 == 0 else (z1, z0)
        a, b = (x, nodes[k] + 2.5, za), (x, nodes[k + 1] - 2.5, zb_)
        member(s, a, b, 3.4, 2.6, 0.4, col, up=(1, 0, 0))
        bolts += [(a[1], za + (1.5 if za == z0 else -1.5)), (b[1], zb_ + (1.5 if zb_ == z0 else -1.5))]
    for y, z in bolts:
        for ux in (-1, 1):
            s.bolt((x + ux * UW / 2, y, z), (ux, 0, 0), 0.9, BOLT)


def step_beam(s, x0, x1, ytop, zface, sz, col):
    """Step beam in front of an upright face (sz: the side it sits on): tall outer part, low step on the inside
    where the lane ends rest. End connectors with hooks in the teardrops and a locking pin."""
    s.rprism((x0, ytop - BEAM_H / 2, zface + sz * (BEAM_D - 1.5)), (x1, ytop - BEAM_H / 2, zface + sz * (BEAM_D - 1.5)),
             3.0, BEAM_H, 0.5, col, seg=2)
    s.rprism((x0, ytop - STEP - (BEAM_H - STEP) / 2, zface + sz * 1.0), (x1, ytop - STEP - (BEAM_H - STEP) / 2, zface + sz * 1.0),
             2.0, BEAM_H - STEP, 0.4, col, seg=2)
    for xe, ux in ((x0, -1), (x1, 1)):
        s.rbox((xe + ux * 1.6, ytop - 4.5, zface + sz * 0.25), (UW / 2 + 0.8, 14, 0.5), 0.2, col, seg=1)   # connector
        s.cylinder((xe, ytop - BEAM_H, zface + sz * 0.5), (xe, ytop, zface + sz * 0.5), 0.3, col, 6)       # weld
        s.cylinder((xe + ux * 2.2, ytop - 2.0, zface + sz * 0.5), (xe + ux * 2.2, ytop - 2.0, zface + sz * 1.6),
                   0.35, "plastic_red", 8)                                                                 # locking pin


class Instances:
    """Repeated small parts (flow wheels, balls, rollers, axle nuts) written as one PointInstancer: a big run of
    racks stays light in the viewport. Prototypes are built once at the origin, axis along X."""

    def __init__(self):
        self.items = {}               # (kind, mat) -> [(pos, quat, scale)]

    def add(self, kind, mat, pos, ln, sx=1.0):
        th = math.atan2(ln.n[2], ln.n[1])                         # lane frame = rotation about X
        q = (math.cos(th / 2), math.sin(th / 2), 0.0, 0.0)
        self.items.setdefault((kind, mat), []).append((pos, q, sx))

    @staticmethod
    def _proto(kind, mat):
        sp = new_shapes()
        if kind == "wheel":
            sp.lathe((-0.8, 0, 0), (0.8, 0, 0), [(0, 1.0), (0.15, 1.25), (0.85, 1.25), (1, 1.0)], mat, sides=10)
            sp.cylinder((-1.75, 0, 0), (1.75, 0, 0), 0.32, "galvanised", 6)                    # axle, riveted
        elif kind == "ball":
            sp.lathe((0, -0.95, 0), (0, 0.95, 0), [(0, 0.3), (0.2, 0.75), (0.5, 0.95), (0.8, 0.75), (1, 0.3)], mat, sides=8)
        elif kind == "roller":
            sp.cylinder((-0.5, 0, 0), (0.5, 0, 0), 1.5, mat, 14)
        else:                                                                                 # hex axle nut
            sp.lathe((0, 0, 0), (0.45, 0, 0), [(0, 0.55), (1, 0.55)], mat, sides=6)
        return sp

    def build(self, stage, path, lib):
        if not self.items:
            return
        pi = UsdGeom.PointInstancer.Define(stage, path)
        UsdGeom.Scope.Define(stage, f"{path}/Prototypes")
        protos, pos, ori, sc, idx = [], [], [], [], []
        for i, ((kind, mat), rows) in enumerate(sorted(self.items.items())):
            pp = f"{path}/Prototypes/{kind}_{i}"
            UsdGeom.Xform.Define(stage, pp)
            shapes_to_stage(stage, pp, self._proto(kind, mat), lib)
            protos.append(Sdf.Path(pp))
            for p_, q, sx in rows:
                pos.append(Gf.Vec3f(*p_))
                ori.append(Gf.Quath(*q))
                sc.append(Gf.Vec3f(sx, 1, 1))
                idx.append(i)
        pi.CreatePrototypesRel().SetTargets(protos)
        pi.CreatePositionsAttr(Vt.Vec3fArray(pos))
        pi.CreateOrientationsAttr(Vt.QuathArray(ori))
        pi.CreateScalesAttr(Vt.Vec3fArray(sc))
        pi.CreateProtoIndicesAttr(Vt.IntArray(idx))


class Lane:
    """Local frame of a sloped lane: u across (X), v up from the bed bottom, w from the load end to the pick end."""

    def __init__(self, x, start, end):
        self.x = x
        self.o = _v(x, *start)
        d = (0.0, end[0] - start[0], end[1] - start[1])
        L = math.hypot(d[1], d[2])
        self.L = L
        self.d = (0.0, d[1] / L, d[2] / L)
        n = (0.0, self.d[2], -self.d[1])
        self.n = n if n[1] > 0 else _mul(n, -1)
        self.e = (1.0, 0.0, 0.0)

    def p(self, u, v, w):
        return _add(self.o, _mul(self.e, u), _mul(self.n, v), _mul(self.d, w))

    def box(self, s, u, v, w, size, mat, r=0.0):
        if r > 0:
            s.rbox(self.p(u, v, w), size, r, mat, axes=(self.e, self.n, self.d), seg=1)
        else:
            s.obox(self.p(u, v, w), size, (self.e, self.n, self.d), mat)


def wheel_track(s, ln, u, wheel, inst):
    L = ln.L
    ln.box(s, u, 0.15, L / 2, (3.2, 0.3, L), "galvanised")
    for su in (-1, 1):
        ln.box(s, u + su * 1.45, 1.3, L / 2, (0.3, 2.6, L), "galvanised")
    for w in (0.6, L - 0.6):                                                   # end hooks over the beam steps
        ln.box(s, u, -0.9, w, (3.2, 1.8, 0.3), "galvanised")
    w = 2.6
    while w < L - 2.0:
        inst.add("wheel", wheel, ln.p(u, 2.1, w), ln)
        w += 5.0


def roller_bed(s, ln, u0, u1, roller, inst):
    L = ln.L
    for uu in (u0 + 0.8, u1 - 0.8):                                            # side frames (channels)
        ln.box(s, uu, 2.0, L / 2, (0.3, 4.0, L), "galvanised", r=0.1)
        ln.box(s, uu + (0.6 if uu < (u0 + u1) / 2 else -0.6), 0.15, L / 2, (1.2, 0.3, L), "galvanised")
    w = 3.0
    while w < L - 2.5:
        inst.add("roller", roller, ln.p((u0 + u1) / 2, 2.6, w), ln, sx=u1 - u0 - 2.0)
        inst.add("nut", BOLT, ln.p(u0 + 0.2, 2.6, w), ln)
        inst.add("nut", BOLT, ln.p(u1 - 0.65, 2.6, w), ln)
        w += 6.0


def ball_mat(s, ln, u0, u1, ball, inst):
    L = ln.L
    ln.box(s, (u0 + u1) / 2, 0.6, L / 2, (u1 - u0 - 0.4, 1.2, L), "plastic_black", r=0.2)
    for uu in (u0 + 0.3, u1 - 0.3):
        ln.box(s, uu, 1.0, L / 2, (0.6, 2.0, L), "galvanised", r=0.1)
    row, w = 0, 3.0
    while w < L - 2.0:
        u = u0 + 3.0 + (2.5 if row % 2 else 0.0)
        while u < u1 - 2.5:
            inst.add("ball", ball, ln.p(u, 1.3, w), ln)
            u += 5.0
        w += 5.0
        row += 1


def tote(s, ln, u, v, wc, tw, tl, th, mat):
    """Open plastic bin: floor, walls, rim, handle recesses, label card on the pick face."""
    ln.box(s, u, v + 0.3, wc, (tw - 1.0, 0.6, tl - 1.0), mat, r=0.2)
    for sw in (-1, 1):
        ln.box(s, u, v + th / 2, wc + sw * (tl / 2 - 0.25), (tw, th, 0.5), mat, r=0.2)
        ln.box(s, u + sw * (tw / 2 - 0.25), v + th / 2, wc, (0.5, th, tl), mat, r=0.2)
        ln.box(s, u, v + th - 0.3, wc + sw * (tl / 2 + 0.2), (tw + 0.8, 0.6, 0.9), mat, r=0.2)   # rim
        ln.box(s, u + sw * (tw / 2 + 0.2), v + th - 0.3, wc, (0.9, 0.6, tl + 0.8), mat, r=0.2)
        ln.box(s, u, v + th - 2.8, wc + sw * (tl / 2 + 0.03), (min(9.0, tw * 0.4), 2.0, 0.05), "hole")  # handle
        for k in (-1, 0, 1):                                                                     # wall ribs
            ln.box(s, u + sw * (tw / 2 + 0.25), v + th / 2 - 0.4, wc + k * tl * 0.3, (0.5, th - 1.6, 0.7), mat, r=0.2)
        for k in (-1, 1):                                                                        # corner posts
            ln.box(s, u + k * (tw / 2 - 0.4), v + th / 2, wc + sw * (tl / 2 - 0.4), (1.2, th - 0.6, 1.2), mat, r=0.4)
    for k in (-1, 1):
        ln.box(s, u + k * tw * 0.3, v + th / 2 - 0.4, wc - tl / 2 - 0.25, (0.7, th - 1.6, 0.5), mat, r=0.2)  # back ribs
    ln.box(s, u, v + th * 0.45, wc + tl / 2 + 0.25, (tw * 0.46, 5.2, 0.35), mat, r=0.2)        # label holder
    ln.box(s, u, v + th * 0.45, wc + tl / 2 + 0.45, (tw * 0.42, 4.5, 0.06), "floor_white")


def carton(s, ln, u, v, wc, tw, tl, th):
    ln.box(s, u, v + th / 2, wc, (tw, th, tl), "cardboard", r=0.4)
    ln.box(s, u, v + th + 0.02, wc, (5.0, 0.06, tl + 0.2), "tape_brown")
    ln.box(s, u, v + th * 0.55, wc + tl / 2 + 0.03, (tw * 0.45, th * 0.35, 0.05), "floor_white")


def label(s, x, y, z, w, code, seed):
    """Location label: white card with a code and a barcode, on the front of the label holder (+Z)."""
    s.rbox((x, y, z + 0.06), (w, 3.8, 0.12), 0.05, "floor_white", seg=1)
    sign_text(s, code, (x - w / 2 + 0.5, y + 1.6, z + 0.05), (1, 0, 0), (0, -1, 0), 1.2, "plastic_black", (0, 0, 1))
    bx, k = x - w / 2 + 0.5, seed * 7919
    while bx < x + w / 2 - 0.6:
        k = (k * 1103515245 + 12345) & 0x7FFFFFFF
        bw = 0.08 + 0.08 * (k % 3)
        if (k >> 4) % 3:
            s.obox((bx + bw / 2, y - 0.8, z + 0.14), (bw, 1.6, 0.04), ((1, 0, 0), (0, 1, 0), (0, 0, 1)), "plastic_black")
        bx += bw + 0.08


def _list(text, conv=float):
    return [conv(t.strip()) for t in str(text).split(",") if t.strip()]


def _snap(y):
    return 6.0 + PITCH * round((y - 6.0) / PITCH)


class RollerRack(Item):
    def widths(self, p):
        return _list(p["bay_widths"]) or [float(p["bay_width"])] * int(p["bays"])

    def footprint(self, p):
        return sum(self.widths(p)) + UW, p["depth"]

    def build(self, stage, path, p, lib):
        bws = self.widths(p)
        D, H = p["depth"], p["height"]
        col, beam = p["frame"], p["beams"]
        L = sum(bws)
        xs = [-L / 2 + sum(bws[:k]) for k in range(len(bws) + 1)]              # frame (bay boundary) positions
        zF, zB = D / 2 - BEAM_D, -D / 2 + BEAM_D                                # upright faces: pick / load side
        zfc, zbc = zF - UD / 2, zB + UD / 2
        s, inst = new_shapes(), Instances()
        for x in xs:
            upright(s, x, zfc, 1, H, col)
            upright(s, x, zbc, -1, H, col)
            frame_lacing(s, x, zbc, zfc, H, col)
        from . import details as DT
        cz, ysg = (zfc + zbc) / 2, min(H - 30.0, 150.0)
        for x, ux in ((xs[0], -1), (xs[-1], 1)):                                                 # load notice
            DT.sign(s, (x + ux * 2.4, ysg, cz), (0, 0, -ux), (0, 1, 0), (ux, 0, 0), min(30.0, (zfc - zbc) * 0.4),
                    ["MAX LOAD", "50 KG LANE", "DO NOT CLIMB"], header="NOTICE", header_mat="powder_blue")
        for k, x in enumerate(xs):                                                               # frame ID tags
            tag = f"F{k + 1}"
            s.rbox((x, 34.0, zfc + UD / 2 + 0.1), (5.0, 2.8, 0.15), 0.3, "aluminium", seg=1)
            sign_text(s, tag, (x - 1.4 * 6 / 7 * len(tag) / 2, 34.8, zfc + UD / 2 + 0.05), (1, 0, 0), (0, -1, 0), 1.4,
                      "plastic_black", (0, 0, 1))
        if p["guards"]:
            from .items_lift import hazard
            for x, ux in ((xs[0], -1), (xs[-1], 1)):                                            # end barriers
                xb = x + ux * (UW / 2 + 5.0)
                for zz in (zbc, zfc):
                    s.rprism((xb, 0.6, zz), (xb, 32.0, zz), 6.0, 6.0, 1.0, "powder_yellow", up=(1, 0, 0))
                    s.rbox((xb, 0.5, zz), (12.0, 1.0, 12.0), 0.3, "powder_yellow", seg=1)
                    for dz in (-4, 4):
                        s.bolt((xb + 4, 1.0, zz + dz), (0, 1, 0), 1.0, BOLT)
                    s.lathe((xb, 32.0, zz), (xb, 33.0, zz), [(0, 3.4), (1, 3.0)], "plastic_black", sides=16)
                s.rprism((xb, 22.0, zbc + 3), (xb, 22.0, zfc - 3), 5.0, 10.0, 1.2, "powder_yellow")
                hazard(s, (xb + ux * 2.55, 17.0, zbc + 3), (0, 0, 1), (0, 1, 0), (ux, 0, 0), zfc - zbc - 6, 10.0, w=6.0)
            for x in (xs[0], xs[-1]):
                s.rprism((x, 0.5, zfc), (x, 40, zfc), UW + 3.0, UD + 3.0, 1.2, "powder_yellow", up=(1, 0, 0))
                for y_ in (10, 20, 30):
                    s.rprism((x, y_, zfc), (x, y_ + 4, zfc), UW + 3.1, UD + 3.1, 1.2, "powder_black", up=(1, 0, 0))
        # ------------------------------------------------------------------ levels
        run = zF - zB + 2 * (BEAM_D - 1.0)
        rise = run * math.tan(math.radians(p["tilt_deg"]))
        nl = int(p["levels"])
        ret = bool(p["return_top"]) and nl >= 2
        fronts = _list(p["level_heights"])[:nl]
        if len(fronts) < nl:
            y_lo, y_hi = 30.0, H - 30.0 - rise - (25.0 if ret else 0.0)
            fronts = [y_lo + k * (y_hi - y_lo) / max(nl - 1, 1) for k in range(nl)]
        beds = _list(p["bed"], str) or ["wheel"]
        attach = {}
        for lv, yf in enumerate(fronts):
            reverse = ret and lv == nl - 1
            yf = _snap(yf)
            yb = _snap(yf + rise)
            if reverse:                                                      # empties roll back to the load side
                yf, yb = yb, yf
            bed = beds[min(lv, len(beds) - 1)].lower()
            bed = bed if bed in TOP else "wheel"
            for b, bw in enumerate(bws):
                x0, x1 = xs[b] + UW / 2, xs[b + 1] - UW / 2
                step_beam(s, x0, x1, yf, zF, 1, beam)
                step_beam(s, x0, x1, yb, zB, -1, beam)
                back, front = (yb - STEP, zB - 2.0), (yf - STEP, zF + 2.0)
                start, end = (front, back) if reverse else (back, front)
                ln = Lane(0.0, start, end)
                ia, ib = x0 + 1.0, x1 - 1.0
                nlanes = int(p["lanes"]) or max(1, round((ib - ia) / 40.0))
                lw = (ib - ia) / nlanes
                if ln.L > 130:                                                 # mid supports under long lanes
                    nm = math.ceil(ln.L / 120.0) - 1
                    for j in range(nm):
                        w = ln.L * (j + 1) / (nm + 1)
                        s.rprism(ln.p(x0, -1.8, w), ln.p(x1, -1.8, w), 3.0, 3.6, 0.4, "galvanised", up=ln.n, seg=2)
                if bed == "ball":
                    ball_mat(s, ln, ia, ib, "steel_machined", inst)
                if bed != "roller":                                             # lane guides
                    for ub in [ia + 1.2] + [ia + k * lw for k in range(1, nlanes)] + [ib - 1.2]:
                        ln.box(s, ub, TOP[bed] + 2.0, ln.L / 2, (0.3, 3.5, ln.L - 6), "galvanised", r=0.1)
                for n in range(nlanes):
                    u0, u1 = ia + n * lw, ia + (n + 1) * lw
                    uc = (u0 + u1) / 2
                    if bed == "wheel":
                        for f in ((-0.3, 0.3) if lw < 55 else (-0.35, 0.0, 0.35)):
                            wheel_track(s, ln, uc + f * lw, p["wheel"], inst)
                    elif bed == "roller":
                        roller_bed(s, ln, u0 + 0.3, u1 - 0.3, "steel_machined", inst)
                    # the load, resting against the exit stop
                    vt = TOP[bed]
                    tw = min(lw - 3.5, 40.0)
                    tl = 40.0 if tw < 36 else 60.0
                    th = 17.0 if tl < 50 else 22.0
                    cnt = int((ln.L - 4.0) // (tl + 0.8))
                    vary = ((lv * 7 + b * 13 + n * 5) % 10) / 10.0                 # lanes run down unevenly
                    fill = round(cnt * min(1.0, max(0.0, p["fill"] - 0.35 + 0.6 * vary)))
                    if reverse and (b + n) % 2:
                        fill = 0
                    kind = str(p["load"]).lower()
                    if not p["totes"] or kind == "none":
                        fill = 0
                    if reverse:
                        fill = min(fill, 2)
                    sl = new_shapes()
                    for k in range(fill):
                        wc = ln.L + 1.2 - tl / 2 - k * (tl + 0.8)
                        if kind == "cartons" or (kind == "mixed" and (b + n + lv) % 2):
                            carton(sl, ln, uc, vt, wc, tw - 1.0, tl - 2.0, th + 4.0)
                        else:
                            tote(sl, ln, uc, vt, wc, tw, tl, th, p["tote"] if (n + lv) % 4 else "plastic_grey")
                    if fill:
                        shapes_to_stage(stage, f"{path}/Load/L{lv + 1}_B{b + 1}_N{n + 1}", sl, lib)
                    pick, load = ln.p(uc, vt, ln.L), ln.p(uc, vt, 0.0)
                    attach[f"lane_L{lv + 1}_B{b + 1}_N{n + 1}_pick"] = pick
                    attach[f"lane_L{lv + 1}_B{b + 1}_N{n + 1}_load"] = load
                    if not reverse:
                        label(s, uc, yf - 5.0, zF + BEAM_D + 0.3, min(lw - 3.0, 9.0),
                              f"{chr(65 + lv)}{b + 1:02d}-{n + 1}", lv * 97 + b * 13 + n)
                # exit stop (pick end), label holder, load bar (load end)
                ye, ze, yl, zl = (yb, zB - 3.8, yf, zF + 4.0) if reverse else (yf, zF + 3.8, yb, zB - 4.0)
                s.rbox(((x0 + x1) / 2, ye + 3.0, ze), (x1 - x0, 7.5, 0.3), 0.1, "galvanised", seg=1)
                s.cylinder((x0, yl + 2.0, zl), (x1, yl + 2.0, zl), 1.1, "galvanised", 12)
                for xe in (x0 + 0.8, x1 - 0.8):
                    s.rbox((xe, yl + 0.5, zl), (0.4, 4.0, 2.4), 0.1, "galvanised", seg=1)
                if not reverse:
                    s.rbox(((x0 + x1) / 2, yf - 5.0, zF + BEAM_D + 0.15), (x1 - x0 - 1.0, 5.6, 0.3), 0.1, "plastic_grey", seg=1)
                attach[f"pick_L{lv + 1}_B{b + 1}"] = ((x0 + x1) / 2, yf + 10, zF + BEAM_D)
        shapes_to_stage(stage, path, s, lib)
        inst.build(stage, f"{path}/Rolling", lib)
        return attach


register(RollerRack("flow_rack", "flow_rack", "Carton flow (roller) rack, modular", {
    "bays": Param(3, "bays side by side", 1, 12),
    "bay_width": Param(120.0, "bay width (frame to frame)", 60, 300, "cm"),
    "bay_widths": Param("", "bay widths, e.g. 120,90,120 (overrides bays / bay_width)"),
    "depth": Param(180.0, "depth (load side to pick side)", 60, 400, "cm"),
    "height": Param(200.0, "frame height", 100, 400, "cm"),
    "levels": Param(4, "flow levels", 1, 8),
    "level_heights": Param("", "front beam heights, e.g. 30,75,120,165 (else evenly spaced)"),
    "tilt_deg": Param(5.0, "lane slope", 2, 12, "deg"),
    "lanes": Param(0, "lanes per bay (0: about 40 cm lanes)", 0, 12),
    "bed": Param("wheel", "wheel / roller / ball, or one per level: wheel,wheel,roller,ball"),
    "return_top": Param(False, "top level slopes back (empty tote return)"),
    "load": Param("totes", "totes / cartons / mixed / none"),
    "totes": Param(True, "fill lanes with the load"),
    "fill": Param(0.75, "how full the lanes are on average (0 - 1)", 0, 1),
    "tote": Param("plastic_blue", "tote colour"),
    "frame": Param("powder_blue", "upright + lacing finish"),
    "beams": Param("powder_orange", "beam finish"),
    "wheel": Param("plastic_grey", "flow wheel finish"),
    "guards": Param(True, "upright guards at the ends of the pick side"),
}))
