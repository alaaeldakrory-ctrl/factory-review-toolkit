"""Walk lab: the crew walking side by side (straight, then a 90 degree turn, then stop facing the camera), for
judging gait by eye. Mocap upper body + planned footsteps (gait.py / mocap.py).

    python tools/walk_lab.py [out/people/walk_lab/walk.usd]
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "exts", "frt.core"))
from pxr import UsdGeom   # noqa: E402

from frt_core import units as U   # noqa: E402
from frt_core.geom import Shapes   # noqa: E402
from frt_core.materials import MaterialLibrary   # noqa: E402
from frt_core.people.bake import bake   # noqa: E402
from frt_core.people.gait import walk   # noqa: E402
from frt_core.people.motion_gen import FPS   # noqa: E402
from frt_core.people.motion_more import idle   # noqa: E402
from frt_core.people.skeleton import Skeleton   # noqa: E402
from frt_core.people.solver import PostureSolver   # noqa: E402
from frt_core.people.workers import WORKERS   # noqa: E402
from frt_core.studio import dress   # noqa: E402
from frt_core.usdio import safe_stage   # noqa: E402


def main(out):
    with safe_stage(out) as st:
        U.setup_stage(st, fps=FPS)
        st.SetDefaultPrim(UsdGeom.Xform.Define(st, "/World").GetPrim())
        lib = MaterialLibrary(st, "/World/Looks")
        UsdGeom.Xform.Define(st, "/World/Asset")
        s = Shapes()                                         # floor marks every 50 cm to read stride length
        for k in range(-2, 13):
            s.box((k * 50.0 - 0.5, 0.0, -160.0), (k * 50.0 + 0.5, 0.15, 160.0), "floor_white")
        s.build(st, "/World/Asset/Marks", lib.refs(list(s.groups)))
        length = 0
        for i, wid in enumerate(("W05F", "W50M", "W95M")):
            w = WORKERS[wid]
            sk = Skeleton.for_worker(w)
            S = PostureSolver(sk, w)
            z = (i - 1) * 110.0
            frames = idle(0.5) + walk(w, sk, [(-100.0, z), (420.0, z), (520.0, z - 20.0)], 90.0, 180.0)
            for f in frames:
                if f.root is None:
                    f.root = (-100.0, z, 90.0)
            frames += idle(1.0)
            for f in frames:
                f.root = f.root or frames[-2].root
            UsdGeom.Xform.Define(st, f"/World/Asset/{wid}")
            bake(st, f"/World/Asset/{wid}", w, frames, lib, solver=S)
            length = max(length, len(frames))
        dress(st, lib, st.GetPrimAtPath("/World/Asset"))
    print("->", out, length, "frames")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "out", "people", "walk_lab", "walk.usd"))
