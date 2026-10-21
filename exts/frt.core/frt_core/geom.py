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
            m.CreateSubdivisionSchemeAttr("none")
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
