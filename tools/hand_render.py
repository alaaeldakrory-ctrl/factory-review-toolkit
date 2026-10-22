"""Hand close-ups: one worker per grip, right hand held out in front at chest height (palm in), the matching tool in
the hand where there is one - rendered with the real body to judge fingers and thumb.

    python tools/hand_render.py -> out/people/hand_lab/hands.usd + shots.json (tools/kit_render.py)
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "exts", "frt.core"))
import numpy as np   # noqa: E402
from pxr import Gf, UsdGeom   # noqa: E402

from frt_core import units as U   # noqa: E402
from frt_core.geom import Shapes   # noqa: E402
from frt_core.materials import MaterialLibrary   # noqa: E402
from frt_core.people import handkit as K   # noqa: E402
from frt_core.people.bake import bake   # noqa: E402
from frt_core.people.motion_gen import Frame, rest_targets   # noqa: E402
from frt_core.people.skeleton import Skeleton   # noqa: E402
from frt_core.people.solver import PostureSolver   # noqa: E402
from frt_core.people.workers import WORKERS   # noqa: E402
from frt_core.studio import dress   # noqa: E402
from frt_core.usdio import safe_stage   # noqa: E402

OUT = os.path.join(HERE, "..", "out", "people", "hand_lab")
GRIPS = [("power:1.6", "handle"), ("pinch", "pin"), ("tripod", "pin"), ("lateral", "key"), ("point", None),
         ("relaxed", None), ("trigger:1.7:0.6", "pistol"), ("hook:1.5", "handle")]


def main():
    w = WORKERS["W50M"]
    sk = Skeleton.for_worker(w)
    S = PostureSolver(sk, w)
    shots = []
    os.makedirs(OUT, exist_ok=True)
    out = os.path.join(OUT, "hands.usd")
    with safe_stage(out) as st:
        U.setup_stage(st, fps=24.0)
        st.SetDefaultPrim(UsdGeom.Xform.Define(st, "/World").GetPrim())
        lib = MaterialLibrary(st, "/World/Looks")
        UsdGeom.Xform.Define(st, "/World/Asset")
        R, _ = rest_targets(S)
        for i, (g, prop) in enumerate(GRIPS):
            x = i * 120.0
            sh = sk.world["upperarm_r"]
            pt = np.array([sh[0] + 6.0, sh[1] - 22.0, 38.0])
            if g.startswith(("power", "hook", "trigger")):
                t = K.power(sk, "r", pt, (0.0, 1.0, 0.0), grip=g, lean=(0.0, 0.0, 1.0))
            else:
                t = K.pinch(sk, "r", pt + np.array([0.0, 0.0, 7.0]), approach=(0.0, -0.2, 1.0), palm=(1.0, -0.4, 0.0),
                            grip=g)
            path = f"/World/Asset/H{i}"
            xf = UsdGeom.Xform.Define(st, path)
            xf.AddTranslateOp().Set(Gf.Vec3d(x, 0.0, 0.0))
            bake(st, path, w, [Frame([t, R["l"]], tuple(pt + np.array([0, 0, 20.0])), {})], lib, solver=S,
                 log=lambda *a: None)
            s = Shapes()
            if prop == "handle":                                  # a bar through the grip, along the handle axis
                s.lathe((x + t.point[0], t.point[1] - 9.0, t.point[2]), (x + t.point[0], t.point[1] + 9.0, t.point[2]),
                        [(0, 1.6), (1, 1.6)], "plastic_black", sides=20)
            if s.groups:
                s.build(st, f"{path}/Prop", lib.refs(list(s.groups)))
            c = np.array([x + t.point[0], t.point[1], t.point[2]])
            for k, (dx, dy, dz) in enumerate(((-34.0, 8.0, 26.0), (30.0, 10.0, 30.0), (0.0, 30.0, 4.0))):
                shots.append({"name": f"{i:02d}_{g.replace(':', '_')}_{k}.png", "eye": (c + [dx, dy, dz]).tolist(),
                              "at": c.tolist(), "mm": 50, "frame": 0, "settle": 90})
        dress(st, lib, st.GetPrimAtPath("/World/Asset"))
    with open(os.path.join(OUT, "shots.json"), "w") as f:
        json.dump(shots, f, indent=1)
    print("->", out, len(shots), "shots")


if __name__ == "__main__":
    main()
