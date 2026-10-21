"""People demo: the motion generators on the crew at real workstations, baked to animation.

- bench A (92 cm): W05F and W95M tighten bolts with a ratchet side by side - same bench, different bodies
- low stand (35 cm): W50M drives a pin with a hammer

    python tools/people_demo.py [out/people/demo.usd]
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "exts", "frt.core"))
from pxr import Gf, UsdGeom   # noqa: E402

from frt_core import catalog as CAT   # noqa: E402
from frt_core import units as U   # noqa: E402
from frt_core.geom import Shapes   # noqa: E402
from frt_core.materials import MaterialLibrary   # noqa: E402
from frt_core.people import motion_gen as G   # noqa: E402
from frt_core.people.bake import bake   # noqa: E402
from frt_core.people.skeleton import Skeleton   # noqa: E402
from frt_core.people.solver import PostureSolver   # noqa: E402
from frt_core.people.workers import WORKERS   # noqa: E402
from frt_core.studio import dress   # noqa: E402
from frt_core.usdio import safe_stage   # noqa: E402


def worker_at(st, path, pos, yaw=0.0):
    xf = UsdGeom.Xform.Define(st, path)
    xf.AddTranslateOp().Set(Gf.Vec3d(*pos))
    xf.AddRotateYOp().Set(float(yaw))


def main(out):
    report = {}
    with safe_stage(out) as st:
        U.setup_stage(st, fps=24.0)
        st.SetDefaultPrim(UsdGeom.Xform.Define(st, "/World").GetPrim())
        lib = MaterialLibrary(st, "/World/Looks")
        UsdGeom.Xform.Define(st, "/World/Asset")
        # bench A: a 200 x 80 table, top at 92 cm, a bracket with bolts on it
        CAT.place(st, "/World/Asset", "industrial_table", "BenchA", (0, 0, 60), 180.0,
                  {"length": 200, "depth": 80, "height": 92}, lib)
        parts = Shapes()
        for bx in (-45.0, 45.0):
            parts.rbox((bx, 94.0, 38.0), (30.0, 4.0, 20.0), 0.8, "powder_blue", seg=2)          # bracket plates
            for dx in (-8.0, 8.0):
                parts.bolt((bx + dx, 96.0, 38.0), (0, 1, 0), 1.6, "bolt_zinc")
        # low stand for the hammer job
        parts.rbox((150.0, 17.5, 40.0), (40.0, 35.0, 40.0), 1.5, "powder_dark", seg=2)
        parts.rbox((150.0, 41.0, 40.0), (16.0, 12.0, 16.0), 1.0, "cast_aluminium", seg=2)       # block with the pin
        parts.lathe((150.0, 47.0, 40.0), (150.0, 52.0, 40.0), [(0, 0.6), (1, 0.6)], "steel_machined", sides=12)
        parts.build(st, "/World/Asset/Parts", lib.refs(list(parts.groups)))
        jobs = [("W05F", (-45.0, 0, -10.0), "ratchet", dict(bolt=(8.0, 99.0, 48.0))),
                ("W95M", (45.0, 0, -10.0), "ratchet", dict(bolt=(8.0, 99.0, 48.0))),
                ("W50M", (150.0, 0, -12.0), "hammer", dict(point=(0.0, 52.0, 52.0)))]
        for wid, pos, kind, kw in jobs:
            w = WORKERS[wid]
            sk = Skeleton.for_worker(w)
            S = PostureSolver(sk, w)
            frames = G.ratchet(S, side="r", strokes=6, **kw) if kind == "ratchet" else G.hammer(S, side="r", strikes=4, **kw)
            path = f"/World/Asset/{wid}"
            worker_at(st, path, pos)
            rep = bake(st, path, w, frames, lib, solver=S)
            a = rep["angles"]
            report[wid] = {"job": kind, "frames": rep["frames"],
                           "max_trunk_flexion": round(max(x["trunk_flexion"] for x in a), 1),
                           "max_shoulder_r": round(max(x["shoulder_r"] for x in a), 1),
                           "max_neck_flexion": round(max(x.get("neck_flexion", 0) for x in a), 1),
                           "min_knee": round(min(min(x["knee_l"], x["knee_r"]) for x in a), 1),
                           "max_knee": round(max(max(x["knee_l"], x["knee_r"]) for x in a), 1)}
        dress(st, lib, st.GetPrimAtPath("/World/Asset"))
    with open(os.path.splitext(out)[0] + ".json", "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps(report, indent=1))
    print("->", out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "out", "people", "demo.usd"))
