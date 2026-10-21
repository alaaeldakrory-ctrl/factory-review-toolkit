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
        sols[i] = S.solve(f.targets, look=f.look, stance=f.stance or stance, prev=prev, smooth=1.0, swing=f.swing)
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


def bake(stage, path, worker, frames, lib, start=0, stance=None, solver=None, log=print, root=None, step=1):
    """root: the work spot (x, z, yaw) for frames that carry none (frames carry their own while walking)."""
    sk = Skeleton.for_worker(worker)
    S = solver or PostureSolver(sk, worker)
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
    for name in names:
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
            "roots": roots, "labels": [f.label for f in frames]}
