"""Tiny offline preview renderer (numpy + Pillow): z-buffered flat shading of USD meshes (instance proxies too).

    python preview.py stage.usd out.png [eye_x eye_y eye_z  at_x at_y at_z] [--time T] [--hide /path,/path]
"""
import sys

import numpy as np
from PIL import Image, ImageDraw
from pxr import Usd, UsdGeom


def collect(stage, time=Usd.TimeCode.Default(), tint=None, skip=()):
    """[(world points (N,3), triangle index array (M,3), rgb)] for every visible Mesh."""
    out = []
    xc = UsdGeom.XformCache(time)
    for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdGeom.Mesh) or any(str(prim.GetPath()).startswith(s) for s in skip):
            continue
        if UsdGeom.Imageable(prim).ComputeVisibility(time) == "invisible":
            continue
        m = UsdGeom.Mesh(prim)
        pts = m.GetPointsAttr().Get(time)
        cnt = m.GetFaceVertexCountsAttr().Get(time)
        idx = m.GetFaceVertexIndicesAttr().Get(time)
        if not pts or not cnt:
            continue
        P = np.array(pts, dtype=np.float64)
        M = np.array(xc.GetLocalToWorldTransform(prim), dtype=np.float64)
        P = np.c_[P, np.ones(len(P))] @ M
        P = P[:, :3]
        idx = np.array(idx)
        tris = []
        o = 0
        for c in cnt:
            for k in range(1, c - 1):
                tris.append((idx[o], idx[o + k], idx[o + k + 1]))
            o += c
        col = m.GetDisplayColorAttr().Get(time)
        rgb = tuple(col[0]) if col else (0.6, 0.6, 0.6)
        if tint:
            rgb = tint(prim, rgb)
        out.append((P, np.array(tris, dtype=np.int64), rgb))
    return out


def render(meshes, path, eye, at, size=(1280, 800), fov=35.0, bg=(235, 236, 238), ss=2):
    """Z-buffered flat-shaded render (supersampled by ss) of [(points, triangles, rgb)]."""
    eye, at = np.array(eye, float), np.array(at, float)
    f = at - eye
    f /= np.linalg.norm(f)
    r = np.cross(f, [0, 1, 0])
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    W, H = size[0] * ss, size[1] * ss
    foc = (W / 2) / np.tan(np.radians(fov / 2))
    light = np.array([0.35, 0.85, 0.4])
    light /= np.linalg.norm(light)
    zbuf = np.full((H, W), np.inf)
    img = np.empty((H, W, 3))
    img[:] = np.array(bg) / 255.0
    count = 0
    for P, T, rgb in meshes:
        if len(T) == 0:
            continue
        d = P - eye
        z = d @ f
        X = W / 2 + foc * (d @ r) / np.maximum(z, 1e-3)
        Y = H / 2 - foc * (d @ u) / np.maximum(z, 1e-3)
        a, b, c = P[T[:, 0]], P[T[:, 1]], P[T[:, 2]]
        n = np.cross(b - a, c - a)
        ln = np.linalg.norm(n, axis=1)
        ok = ln > 1e-12
        n[ok] /= ln[ok, None]
        view = (a + b + c) / 3 - eye
        n *= -np.sign((n * view).sum(1))[:, None]
        shade = 0.38 + 0.62 * np.clip(n @ light, 0, 1)
        col = np.array(rgb)
        zt = z[T]
        good = ok & (zt.min(1) > 1.0)
        xs, ys = X[T], Y[T]
        x0 = np.clip(np.floor(xs.min(1)), 0, W - 1).astype(int)
        x1 = np.clip(np.ceil(xs.max(1)), 0, W - 1).astype(int)
        y0 = np.clip(np.floor(ys.min(1)), 0, H - 1).astype(int)
        y1 = np.clip(np.ceil(ys.max(1)), 0, H - 1).astype(int)
        good &= (xs.max(1) >= 0) & (xs.min(1) < W) & (ys.max(1) >= 0) & (ys.min(1) < H)
        for i in np.nonzero(good)[0]:
            if x1[i] < x0[i] or y1[i] < y0[i]:
                continue
            gx, gy = np.meshgrid(np.arange(x0[i], x1[i] + 1) + 0.5, np.arange(y0[i], y1[i] + 1) + 0.5)
            (ax, bx, cx), (ay, by, cy) = xs[i], ys[i]
            den = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
            if abs(den) < 1e-12:
                continue
            l0 = ((by - cy) * (gx - cx) + (cx - bx) * (gy - cy)) / den
            l1 = ((cy - ay) * (gx - cx) + (ax - cx) * (gy - cy)) / den
            l2 = 1 - l0 - l1
            inside = (l0 >= 0) & (l1 >= 0) & (l2 >= 0)
            if not inside.any():
                continue
            zi = 1.0 / (l0 / zt[i, 0] + l1 / zt[i, 1] + l2 / zt[i, 2])
            sub = zbuf[y0[i]:y1[i] + 1, x0[i]:x1[i] + 1]
            m = inside & (zi < sub)
            if m.any():
                sub[m] = zi[m]
                img[y0[i]:y1[i] + 1, x0[i]:x1[i] + 1][m] = np.minimum(1, col * shade[i])
                count += 1
    out = (img * 255).astype(np.uint8)
    im = Image.fromarray(out)
    if ss > 1:
        im = im.resize(size, Image.LANCZOS)
    im.save(path)
    return count


if __name__ == "__main__":
    args = [a for a in sys.argv[1:]]
    t = Usd.TimeCode.Default()
    skip = ()
    if "--time" in args:
        k = args.index("--time")
        t = Usd.TimeCode(float(args[k + 1]))
        del args[k:k + 2]
    if "--hide" in args:
        k = args.index("--hide")
        skip = tuple(args[k + 1].split(","))
        del args[k:k + 2]
    st = Usd.Stage.Open(args[0])
    v = [float(x) for x in args[2:8]] if len(args) >= 8 else [400, 300, 450, 0, 120, 0]
    n = render(collect(st, t, skip=skip), args[1], v[:3], v[3:])
    print(n, "triangles")
