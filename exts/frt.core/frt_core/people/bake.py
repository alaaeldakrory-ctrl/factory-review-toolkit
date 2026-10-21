"""Bake generated frames: solve every frame (warm-started, so motion is continuous), then write the worker's
SkelAnimation and the tools' transforms as time samples. Also returns per-frame angles for the ergonomics rules.

    report = bake(stage, "/World/People/W50M", worker, frames, lib, start=0)
"""
import time

import numpy as np
from pxr import Gf, Sdf, UsdGeom, UsdSkel

from .skeleton import Skeleton
from .solver import PostureSolver
from .tools import TOOLS
from .usd_worker import pose_arrays, write_worker


def bake(stage, path, worker, frames, lib, start=0, stance=None, solver=None, log=print):
    sk = Skeleton.for_worker(worker)
    S = solver or PostureSolver(sk, worker)
    t0 = time.time()
    sols, prev = [], None
    for f in frames:
        sol = S.solve(f.targets, look=f.look, stance=stance, prev=prev, smooth=1.0)
        sols.append(sol)
        prev = sol
    log(f"{worker.id}: {len(frames)} frames solved in {time.time() - t0:.1f} s")
    root, _ = write_worker(stage, f"{path}/Body", worker, lib, pose=sols[0].pose if sols else None, sk=sk)
    anim = UsdSkel.Animation(stage.GetPrimAtPath(f"{path}/Body/Skeleton/Pose"))
    tr_a, rot_a = anim.GetTranslationsAttr(), anim.GetRotationsAttr()
    for i, sol in enumerate(sols):
        tr, rot = pose_arrays(sk, sol.pose)
        tr_a.Set(tr, start + i)
        rot_a.Set(rot, start + i)
    # tools: one prim per tool instance, its transform keyed every frame (hidden when not in use)
    names = sorted({k for f in frames for k in f.tools})
    for name in names:
        key = next(f.tools[name][0] for f in frames if name in f.tools)
        tp = f"{path}/Tools/{name}"
        xf = UsdGeom.Xform.Define(stage, tp)
        op = xf.AddTransformOp()
        vis = xf.CreateVisibilityAttr()
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
    return {"frames": len(frames), "angles": [s.angles for s in sols], "cost": [s.cost for s in sols]}
