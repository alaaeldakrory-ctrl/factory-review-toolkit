"""Bake generated frames: solve frames (warm-started, so motion is continuous), write the worker's SkelAnimation,
root placement (walking) and the tools' transforms as time samples. Returns per-frame angles / roots for the rules.

- static frames (waiting) reuse the previous solution;
- `step` > 1 solves every step-th frame and blends the ones between (rotations slerped, pelvis lerped) - fast
  enough for long processes, invisible at 24 fps for steps of 2.

    report = bake(stage, "/World/People/W50M", worker, frames, lib, start=0, root=(x, z, yaw))
"""
import time

import numpy as np
from pxr import Gf, UsdGeom, UsdSkel

from .kinematics import Pose
from .skeleton import NAMES, Skeleton
from .solver import PostureSolver
from .tools import TOOLS
from .usd_worker import pose_arrays, write_worker


def _q(R):
    from .kinematics import quat_from_matrix
    return np.array(quat_from_matrix(R))


def _R(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _slerp(a, b, u):
    if np.dot(a, b) < 0:
        b = -b
    d = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if d > 0.9995:
        q = a + (b - a) * u
        return q / np.linalg.norm(q)
    th = np.arccos(d)
    return (np.sin((1 - u) * th) * a + np.sin(u * th) * b) / np.sin(th)


def blend(p0, p1, u):
    out = Pose(pelvis=p0.pelvis * (1 - u) + p1.pelvis * u)
    for j in set(p0.rot) | set(p1.rot):
        out.rot[j] = _R(_slerp(_q(p0.local(j)), _q(p1.local(j)), u))
    return out


def solve_frames(S, frames, stance=None, step=1, log=print, label=""):
    t0 = time.time()
    sols = [None] * len(frames)
    prev = None
    key = [i for i, f in enumerate(frames) if not f.static and (i % step == 0 or i == len(frames) - 1
                                                                 or frames[i].root is not None and step > 1 and i % step == 0)]
    keyset = set(key)
    for i, f in enumerate(frames):
        if f.static and prev is not None:
            sols[i] = prev
            continue
        if i not in keyset and prev is not None and i < len(frames) - 1:
            continue
        sols[i] = S.solve(f.targets, look=f.look, stance=f.stance or stance, prev=prev, smooth=1.0, swing=f.swing,
                          base=f.base)
        prev = sols[i]
    # fill the frames between solved ones
    last = None
    for i in range(len(frames)):
        if sols[i] is not None:
            last = i
            continue
        nxt = next((k for k in range(i + 1, len(frames)) if sols[k] is not None), None)
        if last is None or nxt is None:
            sols[i] = sols[last if last is not None else nxt]
            continue
        u = (i - last) / (nxt - last)
        a, b = sols[last], sols[nxt]
        sols[i] = type(a)(blend(a.pose, b.pose, u), a.cost * (1 - u) + b.cost * u, a.terms,
                          {k: a.angles.get(k, 0) * (1 - u) + b.angles.get(k, 0) * u for k in a.angles}, a.v)
    log(f"{label}{len(frames)} frames ({len(keyset)} solved) in {time.time() - t0:.1f} s")
    return sols


PART_BUILDERS = {}          # kind -> builder(Shapes, args) for composite parts (registered by job libraries)


def part_shapes(spec):
    """Geometry of a handled part from its spec: "box:WxHxD:look" (centred), "ring:R:r:look" (cable tie),
    "cyl:R:L:look" (along +X from the origin), or a registered composite kind (PART_BUILDERS)."""
    from ..geom import Shapes
    s = Shapes()
    kind, *rest = spec.split(":")
    if kind in PART_BUILDERS:
        PART_BUILDERS[kind](s, rest)
    elif kind == "cyl":
        s.cylinder((0, 0, 0), (float(rest[1]), 0, 0), float(rest[0]), rest[2] if len(rest) > 2 else "steel_machined",
                   sides=20)
    elif kind == "box":
        w, h, d = (float(v) for v in rest[0].split("x"))
        s.rbox((0, 0, 0), (w, h, d), min(1.2, min(w, h, d) * 0.12), rest[1] if len(rest) > 1 else "cardboard", seg=2)
    elif kind == "ring":
        s.torus((0, 0, 0), (0, 0, 1), float(rest[0]), float(rest[1]), rest[2] if len(rest) > 2 else "plastic_black",
                seg=20, sides=6)
    return s


def _rootT(r):
    import math
    a = math.radians(r[2])
    T = np.eye(4)
    T[:3, :3] = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
    T[:3, 3] = (r[0], 0.0, r[1])
    return T


def bake(stage, path, worker, frames, lib, start=0, stance=None, solver=None, log=print, root=None, step=1,
         parts_root="/World/Asset/Parts", sols=None):
    """Tools (keys in TOOLS) live under the worker; parts (names "part:<id>", spec in the key) live in the world:
    while handled they follow the hands, afterwards they stay where they were put (USD holds the last sample)."""
    """root: the work spot (x, z, yaw) for frames that carry none (frames carry their own while walking)."""
    sk = Skeleton.for_worker(worker)
    S = solver or PostureSolver(sk, worker)
    if sols is None:
        sols = solve_frames(S, frames, stance, step, log, f"{worker.id}: ")
    write_worker(stage, f"{path}/Body", worker, lib, pose=sols[0].pose if sols else None, sk=sk)
    anim = UsdSkel.Animation(stage.GetPrimAtPath(f"{path}/Body/Skeleton/Pose"))
    tr_a, rot_a = anim.GetTranslationsAttr(), anim.GetRotationsAttr()
    for i, sol in enumerate(sols):
        tr, rot = pose_arrays(sk, sol.pose)
        tr_a.Set(tr, start + i)
        rot_a.Set(rot, start + i)
    roots = []
    xf = UsdGeom.Xformable(stage.GetPrimAtPath(path))
    if any(f.root is not None for f in frames) or root is not None:
        xf.ClearXformOpOrder()
        top, roy = xf.AddTranslateOp(), xf.AddRotateYOp()
        cur = root or next(f.root for f in frames if f.root is not None)
        for i, f in enumerate(frames):
            cur = f.root if f.root is not None else (root or cur)
            top.Set(Gf.Vec3d(cur[0], 0.0, cur[1]), start + i)
            roy.Set(float(cur[2]), start + i)
            roots.append(cur)
    names = sorted({k for f in frames for k in f.tools})
    for name in [n for n in names if n.startswith("part:")]:
        pid = name.split(":", 1)[1]
        spec = next(f.tools[name][0] for f in frames if name in f.tools)
        pp = f"{parts_root}/{pid}"
        pxf = UsdGeom.Xform(stage.GetPrimAtPath(pp)) if stage.GetPrimAtPath(pp) else UsdGeom.Xform.Define(stage, pp)
        ops = pxf.GetOrderedXformOps()
        op = ops[0] if ops else pxf.AddTransformOp()
        if not stage.GetPrimAtPath(f"{pp}/Geom"):
            shp = part_shapes(spec)
            shp.build(stage, f"{pp}/Geom", lib.refs(list(shp.groups)))
        vis = pxf.GetVisibilityAttr() if pxf.GetVisibilityAttr().HasAuthoredValue() else pxf.CreateVisibilityAttr()
        appear = spec.startswith("ring")                           # made during the step: hidden before it
        first = True
        for i, f in enumerate(frames):
            if name not in f.tools:
                continue
            if first and appear:
                vis.Set("invisible", start + i - 1 if start + i > 0 else start + i)
            first = False
            r = roots[i] if i < len(roots) else (root or (0.0, 0.0, 0.0))
            W = _rootT(r) @ f.tools[name][1]
            op.Set(Gf.Matrix4d(*W.T.flatten().tolist()), start + i)
            vis.Set("inherited", start + i)
    for name in [n for n in names if not n.startswith("part:")]:
        key = next(f.tools[name][0] for f in frames if name in f.tools)
        tp = f"{path}/Tools/{name}"
        txf = UsdGeom.Xform.Define(stage, tp)
        op = txf.AddTransformOp()
        vis = txf.CreateVisibilityAttr()
        shp = TOOLS[key].shapes()
        shp.build(stage, f"{tp}/Geom", lib.refs(list(shp.groups)))
        for i, f in enumerate(frames):
            if name in f.tools:
                T = f.tools[name][1]
                op.Set(Gf.Matrix4d(*T.T.flatten().tolist()), start + i)
                vis.Set("inherited", start + i)
            else:
                vis.Set("invisible", start + i)
    end = start + len(frames) - 1
    if stage.HasAuthoredTimeCodeRange():
        stage.SetEndTimeCode(max(stage.GetEndTimeCode(), end))
    else:
        stage.SetStartTimeCode(start)
        stage.SetEndTimeCode(end)
    return {"frames": len(frames), "angles": [s.angles for s in sols], "cost": [s.cost for s in sols],
            "reach": [float(s.terms.get("reach_l", 0.0) + s.terms.get("reach_r", 0.0)) for s in sols],
            "terms": [{k: float(v) for k, v in s.terms.items()} for s in sols],
            "roots": roots, "labels": [f.label for f in frames]}
