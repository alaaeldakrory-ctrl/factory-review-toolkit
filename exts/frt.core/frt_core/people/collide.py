"""Collisions with the scene's equipment: a distance field built from the static meshes around the workers, and
body capsules tested against it. Pure numpy + pxr (no scipy), so the same file works in Kit, in plain Python and
in Ergo Scan.

    field = SceneField.from_stage(stage, region=((xmin, xmax), (ymin, ymax), (zmin, zmax)))   # Y-up cm
    pen = field.penetration(body_points(P))     # cm each body sample is inside equipment (<= 0: clear)

Static geometry only: anything that moves over time (parts being carried, characters, markers) is left out, so
the part in the worker's hands never counts as a collision. The floor is left out too (feet stand on it).
"""
import math

import numpy as np

H = 3.0                     # voxel size, cm
MAX_D = 36.0                # distances are known up to this far from a surface, cm
TOL = 1.5                   # cm of contact allowed (clothing, skin) before it counts as being inside

# body capsules on the canonical joints: (from, to, radius cm, samples, fraction of the segment used)
CAPSULES = [("pelvis", "spine_02", 13.0, 3, 1.0), ("spine_02", "spine_03", 13.0, 2, 1.0),
            ("spine_03", "neck_01", 10.0, 2, 1.0), ("head", "head", 10.0, 1, 1.0),
            ("upperarm_l", "lowerarm_l", 5.5, 3, 1.0), ("upperarm_r", "lowerarm_r", 5.5, 3, 1.0),
            ("lowerarm_l", "hand_l", 4.5, 2, 0.6), ("lowerarm_r", "hand_r", 4.5, 2, 0.6),
            ("thigh_l", "calf_l", 7.0, 3, 1.0), ("thigh_r", "calf_r", 7.0, 3, 1.0),
            ("calf_l", "foot_l", 5.5, 3, 0.85), ("calf_r", "foot_r", 5.5, 3, 0.85),
            ("foot_l", "ball_l", 4.0, 2, 1.0), ("foot_r", "ball_r", 4.0, 2, 1.0)]          # the foot to the toes
HEAD_UP = 8.0               # the head sphere sits this far above the head joint


def body_points(P, up=(0.0, 1.0, 0.0), capsules=CAPSULES):
    """(points (n, 3), radii (n,), part name per point) for joint positions P (dict, any frame)."""
    pts, rad, part = [], [], []
    for a, b, r, n, frac in capsules:
        if a not in P or b not in P:
            continue
        pa, pb = np.asarray(P[a], float), np.asarray(P[b], float)
        if a == b:
            pts.append(pa + np.asarray(up, float) * HEAD_UP)
            rad.append(r)
            part.append(a)
            continue
        for k in range(n):
            u = frac * (k + 0.5) / n
            pts.append(pa + (pb - pa) * u)
            rad.append(r)
            part.append(a.rsplit("_", 1)[0] if a.endswith(("_l", "_r")) else a)
    return np.array(pts), np.array(rad), part


def _moving(prim, cache):
    """Whether this prim or an ancestor is animated (time-sampled transform / points / visibility)."""
    from pxr import UsdGeom
    p = prim
    while p and p.GetPath().pathString != "/":
        k = p.GetPath()
        if k in cache:
            return cache[k]
        if p.IsA(UsdGeom.Xformable):
            for op in UsdGeom.Xformable(p).GetOrderedXformOps():
                if op.GetNumTimeSamples() > 1:
                    cache[k] = True
                    return True
        p = p.GetParent()
    return False


class SceneField:
    def __init__(self, origin, dims, dist):
        self.origin = np.asarray(origin, float)
        self.dims = np.asarray(dims, int)
        self.d = dist                                # (nx, ny, nz) float32 distance to the nearest surface, cm

    # ------------------------------------------------------------------------------------------- building
    @staticmethod
    def grid(region):
        (x0, x1), (y0, y1), (z0, z1) = region
        origin = np.array([x0, y0, z0], float)
        dims = np.maximum(1, np.ceil((np.array([x1, y1, z1]) - origin) / H).astype(int) + 1)
        return origin, dims

    @classmethod
    def from_triangles(cls, tris, region, base_occ=None):
        origin, dims = cls.grid(region)
        occ = np.zeros(dims, bool) if base_occ is None else base_occ.copy()
        if len(tris):
            T = np.asarray(tris, float)                 # (m, 3, 3)
            a, b, c = T[:, 0], T[:, 1], T[:, 2]
            # sample each triangle densely enough (spacing <= H / 2)
            L = np.maximum(np.linalg.norm(b - a, axis=1), np.linalg.norm(c - a, axis=1))
            steps = np.clip(np.ceil(L / (H * 0.5)).astype(int), 1, 60)
            for s in np.unique(steps):
                sel = steps == s
                g = np.linspace(0.0, 1.0, s + 1)
                uu, vv = np.meshgrid(g, g, indexing="ij")
                keep = uu + vv <= 1.0 + 1e-9
                uu, vv = uu[keep], vv[keep]
                A, B, C = a[sel], b[sel], c[sel]
                Pp = A[:, None, :] + (B - A)[:, None, :] * uu[None, :, None] + (C - A)[:, None, :] * vv[None, :, None]
                idx = np.floor((Pp.reshape(-1, 3) - origin) / H + 0.5).astype(int)
                ok = np.all((idx >= 0) & (idx < dims), axis=1)
                idx = idx[ok]
                occ[idx[:, 0], idx[:, 1], idx[:, 2]] = True
        d = np.full(dims, MAX_D, np.float32)
        d[occ] = 0.0
        cur = occ.copy()
        for k in range(1, int(MAX_D / H) + 1):          # grow layer by layer (26-neighbourhood: a chessboard
            nxt = cur.copy()                            # distance, slightly short on diagonals = conservative)
            for ax in range(3):
                for sh in (1, -1):
                    nxt |= np.roll(cur, sh, axis=ax) & _valid_roll(cur.shape, ax, sh)
            new = nxt & ~cur
            d[new] = k * H
            cur = nxt
            if not new.any():
                break
        return cls(origin, dims, d)

    @classmethod
    def from_stage(cls, stage, region, time=0.0, exclude=(), log=None):
        """Everything standing still at `time` in `region` (frt frame: Y-up cm) as a field (SceneGeometry once)."""
        return SceneGeometry(stage, region, exclude, log=log).field_at(time)

    # ------------------------------------------------------------------------------------------- queries
    def distance(self, pts):
        """Distance (cm) from each point to the nearest equipment surface (trilinear; MAX_D outside the grid)."""
        q = (np.asarray(pts, float) - self.origin) / H
        i0 = np.floor(q).astype(int)
        f = q - i0
        out = np.full(len(q), MAX_D, float)
        ok = np.all((i0 >= 0) & (i0 < self.dims - 1), axis=1)
        if not ok.any():
            return out
        i, fr = i0[ok], f[ok]
        acc = np.zeros(len(i))
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    w = (fr[:, 0] if dx else 1 - fr[:, 0]) * (fr[:, 1] if dy else 1 - fr[:, 1]) * \
                        (fr[:, 2] if dz else 1 - fr[:, 2])
                    acc += w * self.d[i[:, 0] + dx, i[:, 1] + dy, i[:, 2] + dz]
        out[ok] = acc
        return out

    def penetration(self, body):
        """cm each body sample reaches into equipment beyond the contact tolerance (<= 0: clear)."""
        pts, rad, _ = body
        if len(pts) == 0:
            return np.zeros(0)
        return rad - self.distance(pts) - TOL

    def cost(self, P, M=None):
        """Solver penalty for joint positions P (dict) in a frame mapped into the field by 4x4 M."""
        pts, rad, part = body_points(P)
        if M is not None:
            pts = (M[:3, :3] @ pts.T).T + M[:3, 3]
        pen = self.penetration((pts, rad, part))
        pen = pen[pen > 0]
        return float(np.sum((pen / 2.0) ** 2) * 3.0)


class SceneGeometry:
    """The Mesh / Cube prims of a stage overlapping a region (frt frame: Y-up cm), read once. field_at(t) places them
    at time t and keeps what stands still around t (a placed beam is an obstacle, a carried one is not)."""

    def __init__(self, stage, region, exclude=(), floor_cm=3.0, log=None):
        from pxr import Usd, UsdGeom, UsdSkel
        self.stage, self.region, self.floor = stage, region, floor_cm
        k = (UsdGeom.GetStageMetersPerUnit(stage) or 0.01) * 100.0
        zup = UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.z
        self.C = np.array([[k, 0, 0], [0, 0, k], [0, -k, 0]], float) if zup else np.diag([k, k, k])
        self.fps = stage.GetTimeCodesPerSecond() or 24.0
        (x0, x1), (y0, y1), (z0, z1) = region
        t0, t1 = stage.GetStartTimeCode(), stage.GetEndTimeCode()
        self.prims = []                                   # (prim, local vertices, triangles as index triples, animated)
        cache = {}
        n_tri = 0
        it = iter(Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()))
        for p in it:
            path = p.GetPath().pathString
            if any(path.startswith(e) for e in exclude) or p.IsA(UsdSkel.Root):
                it.PruneChildren()
                continue
            if not (p.IsA(UsdGeom.Mesh) or p.IsA(UsdGeom.Cube)):
                continue
            img = UsdGeom.Imageable(p)
            if img.ComputePurpose() in (UsdGeom.Tokens.guide, UsdGeom.Tokens.proxy):
                continue
            # anywhere in the region over the whole timeline (a moving prim: its box at both ends and the middle)
            moving = _moving(p, cache)
            inside = False
            for t in ((t0, (t0 + t1) / 2, t1) if moving else (t0,)):
                rng = UsdGeom.BBoxCache(Usd.TimeCode(t), ["default", "render"]).ComputeWorldBound(p).ComputeAlignedRange()
                if rng.IsEmpty():
                    continue
                lo, hi = self.C @ np.array(rng.GetMin(), float), self.C @ np.array(rng.GetMax(), float)
                lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)
                if not (hi[0] < x0 or lo[0] > x1 or hi[1] < y0 or lo[1] > y1 or hi[2] < z0 or lo[2] > z1) and                         (moving or hi[1] >= floor_cm):
                    inside = True
                    break
            if not inside:
                continue
            if p.IsA(UsdGeom.Cube):
                s_ = float(UsdGeom.Cube(p).GetSizeAttr().Get() or 2.0) / 2.0
                v = np.array([[x, y, z] for x in (-s_, s_) for y in (-s_, s_) for z in (-s_, s_)], float)
                faces = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
            else:
                m = UsdGeom.Mesh(p)
                pts, cnt, ind = m.GetPointsAttr().Get(), m.GetFaceVertexCountsAttr().Get(), m.GetFaceVertexIndicesAttr().Get()
                if not pts or not cnt:
                    continue
                v = np.array(pts, float)
                faces, i0, ind = [], 0, list(ind)
                for c in cnt:
                    faces.append(ind[i0:i0 + c])
                    i0 += c
            tri = np.array([(f[0], f[j], f[j + 1]) for f in faces for j in range(1, len(f) - 1)], int)
            if len(tri):
                self.prims.append((p, v, tri, moving))
                n_tri += len(tri)
        if log:
            log(f"scene geometry: {len(self.prims)} prims ({sum(m for *_, m in self.prims)} moving), {n_tri} triangles")

    def _world(self, p, v, t):
        from pxr import Usd, UsdGeom
        M = np.array(UsdGeom.XformCache(Usd.TimeCode(t)).GetLocalToWorldTransform(p), float).T
        w = (M[:3, :3] @ v.T).T + M[:3, 3]
        return (self.C @ w.T).T, M

    def _static_occ(self):
        if getattr(self, "_occ", None) is None:             # everything that never moves: rasterised once
            tris = []
            for p, v, tri, moving in self.prims:
                if moving:
                    continue
                w, _ = self._world(p, v, 0.0)
                T = w[tri]
                tris.append(T[T[:, :, 1].max(axis=1) >= self.floor])
            T = np.concatenate(tris) if tris else np.zeros((0, 3, 3))
            f = SceneField.from_triangles(T, self.region)
            self._occ = f.d == 0.0
        return self._occ

    def field_at(self, t, still_s=0.4):
        from pxr import Usd, UsdGeom
        base = self._static_occ()
        tris = []
        dt = still_s * self.fps
        for p, v, tri, moving in self.prims:
            if not moving:
                continue
            if UsdGeom.Imageable(p).ComputeVisibility(Usd.TimeCode(t)) == UsdGeom.Tokens.invisible:
                continue
            w, M = self._world(p, v, t)
            if moving:                                       # moving right now (being carried): not an obstacle
                _, Ma = self._world(p, v[:1], t - dt)
                _, Mb = self._world(p, v[:1], t + dt)
                if np.abs(Ma - M).max() > 1e-3 or np.abs(Mb - M).max() > 1e-3:
                    continue
            T = w[tri]
            T = T[T[:, :, 1].max(axis=1) >= self.floor]      # the floor
            tris.append(T)
        T = np.concatenate(tris) if tris else np.zeros((0, 3, 3))
        return SceneField.from_triangles(T, self.region, base)


def _valid_roll(shape, ax, sh):
    """Mask that drops the wrapped-around slice of np.roll."""
    m = np.ones(shape, bool)
    sl = [slice(None)] * 3
    sl[ax] = 0 if sh > 0 else -1
    m[tuple(sl)] = False
    return m


def region_around(points, margin=150.0, height=(0.0, 260.0)):
    P = np.asarray(points, float)
    return ((float(P[:, 0].min() - margin), float(P[:, 0].max() + margin)), height,
            (float(P[:, 2].min() - margin), float(P[:, 2].max() + margin)))


def shape_triangles(shapes, M=None):
    """(m, 3, 3) triangles of a geom.Shapes (fixture, tool), moved by 4x4 M."""
    out = []
    for P, C, I, _ in shapes.groups.values():
        P = np.asarray(P, float)
        if M is not None and len(P):
            P = (M[:3, :3] @ P.T).T + M[:3, 3]
        k = 0
        for c in C:
            f = I[k:k + c]
            out += [(P[f[0]], P[f[j]], P[f[j + 1]]) for j in range(1, c - 1)]
            k += c
    return np.array(out, float).reshape(-1, 3, 3)


def fixture_field(job, M=None):
    """A job's own fixture (moved with the work by M) as a field in the worker frame: the body keeps out of the
    panel / bench / chassis it works at, legs and feet as much as the chest (a knee through a board is as wrong
    as a chest through it). None for a job without fixture geometry."""
    from ..geom import Shapes
    s = Shapes()
    job.fixture(s)
    T = shape_triangles(s, M)
    if not len(T):
        return None
    lo, hi = T.reshape(-1, 3).min(0), T.reshape(-1, 3).max(0)
    region = ((float(lo[0]) - 50.0, float(hi[0]) + 50.0), (0.0, max(220.0, float(hi[1]) + 30.0)),
              (float(lo[2]) - 90.0, float(hi[2]) + 50.0))
    return SceneField.from_triangles(T, region)


_TOOL_PTS = {}
TOOL_BODY = ("pelvis", "spine_02", "spine_03", "head", "thigh_l", "thigh_r", "calf_l", "calf_r", "foot_l", "foot_r")


def tool_points(key, step=2.0):
    """Surface samples of a hand tool (tools.TOOLS key) in its own frame, about `step` cm apart."""
    if key not in _TOOL_PTS:
        from .tools import TOOLS
        T = shape_triangles(TOOLS[key].shapes())
        pts = T.reshape(-1, 3)
        if len(pts):
            q = np.unique(np.floor(pts / step).astype(int), axis=0, return_index=True)[1]
            pts = pts[np.sort(q)]
        _TOOL_PTS[key] = pts
    return _TOOL_PTS[key]


def tool_in_body(P, pts, parts=TOOL_BODY):
    """{body part: cm} a tool's points (in the frame of joint positions P) reach into the trunk, head and legs -
    not the arms, whose hand holds it."""
    out = {}
    if not len(pts):
        return out
    for a, b, r, n, frac in CAPSULES:
        if a not in parts or a not in P or b not in P:
            continue
        pa, pb = np.asarray(P[a], float), np.asarray(P[b], float)
        if a == b:
            pa = pb = pa + np.array([0.0, HEAD_UP, 0.0])
        ab = (pb - pa) * frac
        u = np.clip(((pts - pa) @ ab) / max(float(ab @ ab), 1e-9), 0.0, 1.0)
        pen = r - float(np.min(np.linalg.norm(pts - (pa + np.outer(u, ab)), axis=1)))
        if pen > 0.0:
            q = a.rsplit("_", 1)[0] if a.endswith(("_l", "_r")) else a
            out[q] = max(out.get(q, 0.0), pen)
    return out
