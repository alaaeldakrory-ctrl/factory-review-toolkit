"""Small mesh-building kit for generators: boxes, cylinders and prisms collected per material, written as one mesh
per material with proper normals. Coordinates in cm, Y up, in the generator's local frame.
"""
import math

import numpy as np
from pxr import Gf, Sdf, UsdGeom, UsdShade, Vt


class Shapes:
    """Collects faces per material key; build() writes them under a prim."""

    def __init__(self):
        self.groups = {}          # material key -> (points list, counts, indices, normals per face-vertex)

    def _group(self, mat):
        return self.groups.setdefault(mat, ([], [], [], []))

    def _add_face_list(self, mat, pts, faces, normals):
        P, C, I, N = self._group(mat)
        base = len(P)
        P.extend(pts)
        for f, n in zip(faces, normals):
            C.append(len(f))
            I.extend(base + v for v in f)
            if isinstance(n, list):
                N.extend(n)
            else:
                N.extend([n] * len(f))

    # ------------------------------------------------------------------ primitives
    def box(self, lo, hi, mat):
        """Axis-aligned box from corner lo to corner hi."""
        (x0, y0, z0), (x1, y1, z1) = lo, hi
        x0, x1 = min(x0, x1), max(x0, x1)
        y0, y1 = min(y0, y1), max(y0, y1)
        z0, z1 = min(z0, z1), max(z0, z1)
        p = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0), (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
        faces = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (2, 3, 7, 6), (1, 2, 6, 5), (0, 4, 7, 3)]
        normals = [(0, 0, -1), (0, 0, 1), (0, -1, 0), (0, 1, 0), (1, 0, 0), (-1, 0, 0)]
        self._add_face_list(mat, p, faces, normals)

    def obox(self, center, size, axes, mat):
        """Oriented box: center, (sx, sy, sz) along the three unit axes."""
        c = np.asarray(center, float)
        ax = [np.asarray(a, float) / np.linalg.norm(a) for a in axes]
        h = [s / 2.0 for s in size]
        corners = []
        for sz in (-1, 1):
            for sy, sx in ((-1, -1), (-1, 1), (1, 1), (1, -1)):
                corners.append(tuple(c + ax[0] * h[0] * sx + ax[1] * h[1] * sy + ax[2] * h[2] * sz))
        faces = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (2, 3, 7, 6), (1, 2, 6, 5), (0, 4, 7, 3)]
        n = [tuple(-ax[2]), tuple(ax[2]), tuple(-ax[1]), tuple(ax[1]), tuple(ax[0]), tuple(-ax[0])]
        self._add_face_list(mat, corners, faces, n)

    def cylinder(self, a, b, r, mat, sides=12, caps=True):
        """Cylinder from point a to point b, smooth sides."""
        a, b = np.asarray(a, float), np.asarray(b, float)
        ax = b - a
        L = np.linalg.norm(ax)
        if L < 1e-6:
            return
        ax /= L
        t = np.cross(ax, [0, 1, 0] if abs(ax[1]) < 0.9 else [1, 0, 0])
        t /= np.linalg.norm(t)
        u = np.cross(ax, t)
        dirs = [math.cos(2 * math.pi * k / sides) * t + math.sin(2 * math.pi * k / sides) * u for k in range(sides)]
        pts = [tuple(a + r * d) for d in dirs] + [tuple(b + r * d) for d in dirs]
        faces, normals = [], []
        for k in range(sides):
            k2 = (k + 1) % sides
            faces.append((k, k2, sides + k2, sides + k))
            normals.append([tuple(dirs[k]), tuple(dirs[k2]), tuple(dirs[k2]), tuple(dirs[k])])
        if caps:
            faces.append(tuple(range(sides - 1, -1, -1)))
            normals.append(tuple(-ax))
            faces.append(tuple(range(sides, 2 * sides)))
            normals.append(tuple(ax))
        self._add_face_list(mat, pts, faces, normals)

    def quad(self, p0, p1, p2, p3, mat):
        n = np.cross(np.subtract(p1, p0), np.subtract(p3, p0))
        n = n / (np.linalg.norm(n) or 1.0)
        self._add_face_list(mat, [p0, p1, p2, p3], [(0, 1, 2, 3)], [tuple(n)])

    # ------------------------------------------------------------------ smooth primitives (curved, real-looking)
    def rbox(self, center, size, r, mat, axes=((1, 0, 0), (0, 1, 0), (0, 0, 1)), seg=3):
        """Box with rounded edges and corners (radius r), smooth-shaded; size along the three axes."""
        c = np.asarray(center, float)
        ax = [np.asarray(a, float) / np.linalg.norm(a) for a in axes]
        h = np.array(size, float) / 2.0
        r = float(min(r, *(h * 0.999)))
        inner = h - r
        # sample positions along each axis: dense in the rounded bands, two points across the flat middle
        ang = [math.pi / 2 * k / seg for k in range(seg + 1)]
        def coords(hh, ii):
            lo = [-ii - r * math.cos(a) for a in ang]
            hi = [ii + r * math.sin(a) for a in ang]
            return lo + hi
        cs = [coords(h[i], inner[i]) for i in range(3)]
        pts, nrm = [], []
        faces = []
        def vert(p):
            q = np.clip(p, -inner, inner)
            d = p - q
            L = np.linalg.norm(d)
            n = d / L if L > 1e-9 else np.zeros(3)
            loc = q + r * n
            pts.append(tuple(c + ax[0] * loc[0] + ax[1] * loc[1] + ax[2] * loc[2]))
            nrm.append(tuple(ax[0] * n[0] + ax[1] * n[1] + ax[2] * n[2]))
            return len(pts) - 1
        for axis in range(3):                                   # the 6 faces, each a grid over the other two axes
            u, v = [(1, 2), (2, 0), (0, 1)][axis]
            for sgn in (-1, 1):
                grid = []
                for a_ in cs[u]:
                    row = []
                    for b_ in cs[v]:
                        p = np.zeros(3)
                        p[axis] = sgn * h[axis]
                        p[u], p[v] = a_, b_
                        row.append(vert(p))
                    grid.append(row)
                for i in range(len(cs[u]) - 1):
                    for j in range(len(cs[v]) - 1):
                        q = (grid[i][j], grid[i + 1][j], grid[i + 1][j + 1], grid[i][j + 1])
                        faces.append(q if sgn > 0 else q[::-1])
        P, C, I, N = self._group(mat)
        base = len(P)
        P.extend(pts)
        for f in faces:
            C.append(4)
            I.extend(base + v for v in f)
            N.extend(nrm[v] for v in f)

    def rprism(self, a, b, w, h, r, mat, up=(0.0, 1.0, 0.0), seg=3, caps=True):
        """Box section from a to b (w across, h deep) with rounded corners along its length and flat, square-cut
        ends: steel tube, box beams, columns. Sections stacked end to end meet flush (paint bands, splices)."""
        a, b = np.asarray(a, float), np.asarray(b, float)
        ax = b - a
        L = np.linalg.norm(ax)
        if L < 1e-9:
            return
        ax /= L
        upv = np.asarray(up, float)
        if abs(np.dot(upv, ax)) > 0.95:
            upv = np.array([1.0, 0.0, 0.0])
        side = np.cross(upv, ax)
        side /= np.linalg.norm(side)
        upv = np.cross(ax, side)
        r = min(r, w / 2 * 0.999, h / 2 * 0.999)
        prof, nor = [], []                                      # rounded rectangle, counter-clockwise
        for cx, cy, a0 in ((w / 2 - r, h / 2 - r, 0), (-w / 2 + r, h / 2 - r, 90), (-w / 2 + r, -h / 2 + r, 180),
                           (w / 2 - r, -h / 2 + r, 270)):
            for k in range(seg + 1):
                t = math.radians(a0 + 90 * k / seg)
                prof.append((cx + r * math.cos(t), cy + r * math.sin(t)))
                nor.append((math.cos(t), math.sin(t)))
        m = len(prof)
        P, C, I, N = self._group(mat)
        base = len(P)
        for (u, v) in prof:
            P.append(tuple(a + side * u + upv * v))
        for (u, v) in prof:
            P.append(tuple(b + side * u + upv * v))
        for k in range(m):
            k2 = (k + 1) % m
            n1 = side * nor[k][0] + upv * nor[k][1]
            n2 = side * nor[k2][0] + upv * nor[k2][1]
            C.append(4)
            I.extend([base + k, base + m + k, base + m + k2, base + k2])
            N.extend([tuple(n1), tuple(n1), tuple(n2), tuple(n2)])
        if caps:
            C.append(m)
            I.extend(base + k for k in range(m - 1, -1, -1))
            N.extend([tuple(-ax)] * m)
            C.append(m)
            I.extend(base + m + k for k in range(m))
            N.extend([tuple(ax)] * m)

    def lathe(self, a, b, profile, mat, sides=24, caps=True):
        """Surface of revolution about the axis a -> b; profile [(t, radius)] with t from 0 (at a) to 1 (at b).
        Motors, end caps, wheels, hubs, drums, sheaves. Smooth around, creased where the profile turns sharply."""
        a, b = np.asarray(a, float), np.asarray(b, float)
        ax = b - a
        L = np.linalg.norm(ax)
        if L < 1e-9:
            return
        ax /= L
        t1 = np.cross(ax, [0, 1, 0] if abs(ax[1]) < 0.9 else [1, 0, 0])
        t1 /= np.linalg.norm(t1)
        t2 = np.cross(ax, t1)
        ring = [(math.cos(2 * math.pi * k / sides) * t1 + math.sin(2 * math.pi * k / sides) * t2) for k in range(sides)]
        P, C, I, N = self._group(mat)
        for (ta, ra), (tb, rb) in zip(profile, profile[1:]):
            if abs(ra - rb) < 1e-9 and abs(ta - tb) < 1e-9:
                continue
            pa, pb = a + ax * ta * L, a + ax * tb * L
            dz, dr = (tb - ta) * L, rb - ra
            sl = math.hypot(dz, dr) or 1.0
            base = len(P)
            for d in ring:
                n = (d * dz - ax * dr) / sl                          # outward normal of this band
                P.append(tuple(pa + ra * d))
                N.append(tuple(n))
                P.append(tuple(pb + rb * d))
                N.append(tuple(n))
            for k in range(sides):
                k2 = (k + 1) % sides
                C.append(4)
                I.extend([base + 2 * k, base + 2 * k2, base + 2 * k2 + 1, base + 2 * k + 1])
            # normals are per face-vertex: store them in index order
            del N[-2 * sides:]
            for k in range(sides):
                k2 = (k + 1) % sides
                for kk, end in ((k, 0), (k2, 0), (k2, 1), (k, 1)):
                    d = ring[kk]
                    N.append(tuple((d * dz - ax * dr) / sl))
        if caps:
            for t, r, sgn in ((profile[0][0], profile[0][1], -1), (profile[-1][0], profile[-1][1], 1)):
                if r <= 1e-6:
                    continue
                pc = a + ax * t * L
                base = len(P)
                for d in ring:
                    P.append(tuple(pc + r * d))
                C.append(sides)
                idx = list(range(base, base + sides))
                I.extend(idx if sgn > 0 else idx[::-1])
                N.extend([tuple(ax * sgn)] * sides)

    def sweep(self, path, radius, mat, sides=10, radii=None, caps=True, closed=False):
        """Round tube along a polyline (cables, hoses, festoon loops, hooks); radii: one per path point (taper)."""
        pts = [np.asarray(p, float) for p in path]
        n = len(pts)
        if n < 2:
            return
        rr = radii or [radius] * n
        tang = []
        for i in range(n):
            if closed:
                t = pts[(i + 1) % n] - pts[i - 1]
            else:
                t = pts[min(i + 1, n - 1)] - pts[max(i - 1, 0)]
            tang.append(t / (np.linalg.norm(t) or 1.0))
        ref = np.array([0, 1, 0]) if abs(tang[0][1]) < 0.9 else np.array([1, 0, 0])
        u = np.cross(tang[0], ref)
        u /= np.linalg.norm(u)
        frames = []
        for i in range(n):                                       # parallel transport: no twisting
            if i:
                u = u - tang[i] * np.dot(u, tang[i])
                u /= (np.linalg.norm(u) or 1.0)
            v = np.cross(tang[i], u)
            frames.append((u, v))
        rings = []
        for i in range(n):
            u, v = frames[i]
            rings.append([(math.cos(2 * math.pi * k / sides) * u + math.sin(2 * math.pi * k / sides) * v) for k in range(sides)])
        P, C, I, N = self._group(mat)
        segs = n if closed else n - 1
        for i in range(segs):
            j = (i + 1) % n
            base = len(P)
            for k in range(sides):
                P.append(tuple(pts[i] + rr[i] * rings[i][k]))
            for k in range(sides):
                P.append(tuple(pts[j] + rr[j] * rings[j][k]))
            for k in range(sides):
                k2 = (k + 1) % sides
                C.append(4)
                I.extend([base + k, base + k2, base + sides + k2, base + sides + k])
                N.extend([tuple(rings[i][k]), tuple(rings[i][k2]), tuple(rings[j][k2]), tuple(rings[j][k])])
        if caps and not closed:
            for i, sgn in ((0, -1), (n - 1, 1)):
                base = len(P)
                for k in range(sides):
                    P.append(tuple(pts[i] + rr[i] * rings[i][k]))
                C.append(sides)
                idx = list(range(base, base + sides))
                I.extend(idx if sgn > 0 else idx[::-1])
                N.extend([tuple(tang[i] * sgn)] * sides)

    def torus(self, center, axis, R, r, mat, seg=16, sides=8, stretch=0.0, up=None):
        """Ring (chain link when stretch > 0: straight sides of that length). axis: the ring's normal."""
        c = np.asarray(center, float)
        ax = np.asarray(axis, float) / np.linalg.norm(axis)
        e1 = np.asarray(up, float) if up is not None else np.cross(ax, [0, 1, 0] if abs(ax[1]) < 0.9 else [1, 0, 0])
        e1 = e1 - ax * np.dot(e1, ax)
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(ax, e1)
        path = []
        for k in range(seg):
            t = 2 * math.pi * k / seg
            off = (stretch / 2) * (1 if math.sin(t) >= 0 else -1)
            path.append(c + R * math.cos(t) * e2 + (R * math.sin(t) + off) * e1)
        self.sweep(path, r, mat, sides=sides, caps=False, closed=True)

    def bolt(self, pos, axis, d, mat, washer="galvanised", length=None):
        """Hex bolt head with a washer, sitting on a surface at pos, axis pointing out of the surface; d: bolt size."""
        p = np.asarray(pos, float)
        ax = np.asarray(axis, float) / np.linalg.norm(axis)
        self.lathe(p, p + ax * 0.12 * d, [(0, 1.05 * d), (1, 1.05 * d)], washer, sides=14)
        self.cylinder(p + ax * 0.12 * d, p + ax * 0.75 * d, 0.9 * d, mat, sides=6)
        self.lathe(p + ax * 0.75 * d, p + ax * 0.85 * d, [(0, 0.85 * d), (1, 0.55 * d)], mat, sides=6, caps=False)
        if length:
            self.cylinder(p + ax * 0.85 * d, p + ax * (0.85 * d + length), 0.5 * d, mat, sides=8)

    def bolts_grid(self, origin, u, v, nu, nv, du, dv, axis, d, mat):
        """A rectangular bolt pattern on a plate: origin is the pattern centre, u / v the plate directions."""
        o, u, v = np.asarray(origin, float), np.asarray(u, float), np.asarray(v, float)
        for i in range(nu):
            for j in range(nv):
                p = o + u * (i - (nu - 1) / 2) * du + v * (j - (nv - 1) / 2) * dv
                self.bolt(p, axis, d, mat)

    # ------------------------------------------------------------------ output
    def build(self, stage, root, materials):
        """One Mesh per material under root; materials: key -> UsdShade.Material (or None)."""
        out = []
        for k, (P, C, I, N) in self.groups.items():
            if not C:
                continue
            m = UsdGeom.Mesh.Define(stage, f"{root}/{_safe(k)}")
            m.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, p)) for p in P]))
            m.CreateFaceVertexCountsAttr(Vt.IntArray(C))
            m.CreateFaceVertexIndicesAttr(Vt.IntArray(I))
            m.CreateNormalsAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, n)) for n in N]))
            m.SetNormalsInterpolation(UsdGeom.Tokens.faceVarying)
            # box-projected UVs in metres: each face-vertex takes the two coordinates across its normal
            pa, na = np.asarray(P, float)[np.asarray(I, int)], np.asarray(N, float)
            dom = np.argmax(np.abs(na), axis=1)
            uv = np.where(dom[:, None] == 0, pa[:, [2, 1]], np.where(dom[:, None] == 1, pa[:, [0, 2]], pa[:, [0, 1]])) / 100.0
            st_ = UsdGeom.PrimvarsAPI(m.GetPrim()).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray,
                                                                 UsdGeom.Tokens.faceVarying)
            st_.Set(Vt.Vec2fArray.FromNumpy(uv.astype(np.float32)))
            m.CreateSubdivisionSchemeAttr("none")
            m.CreateDoubleSidedAttr(True)
            arr = np.asarray(P, float)
            m.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(*arr.min(0)), Gf.Vec3f(*arr.max(0))]))
            ref = materials.get(k) if materials else None       # materials.MatRef (material, preview colour)
            if ref is not None:
                UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(ref.material)
                m.CreateDisplayColorAttr([Gf.Vec3f(*ref.rgb)])
            out.append(m)
        return out


def _safe(name):
    s = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in str(name))
    return s if s and not s[0].isdigit() else "M_" + s
