"""Pose lab: the posture solver on the crew - each worker reaches a target at floor level, bench height and
overhead (with the same stance), shown in the studio with the targets marked. For judging postures by eye.

    python tools/pose_lab.py [out/people/pose_lab.usd]
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "exts", "frt.core"))
from pxr import Gf, UsdGeom   # noqa: E402

from frt_core import units as U   # noqa: E402
from frt_core.geom import Shapes   # noqa: E402
from frt_core.materials import MaterialLibrary   # noqa: E402
from frt_core.people.skeleton import Skeleton   # noqa: E402
from frt_core.people.solver import HandTarget, PostureSolver   # noqa: E402
from frt_core.people.usd_worker import write_worker   # noqa: E402
from frt_core.people.workers import WORKERS   # noqa: E402
from frt_core.studio import dress   # noqa: E402
from frt_core.usdio import safe_stage   # noqa: E402

TARGETS = {"low": (-8.0, 40.0, 42.0), "bench": (-8.0, 95.0, 42.0), "high": (-8.0, 170.0, 35.0)}


def main(out):
    report = {}
    with safe_stage(out) as st:
        U.setup_stage(st, fps=24.0)
        st.SetDefaultPrim(UsdGeom.Xform.Define(st, "/World").GetPrim())
        lib = MaterialLibrary(st, "/World/Looks")
        UsdGeom.Xform.Define(st, "/World/Asset")
        marks = Shapes()
        for r, wid in enumerate(("W05F", "W50M", "W95M")):
            w = WORKERS[wid]
            sk = Skeleton.for_worker(w)
            S = PostureSolver(sk, w)
            for c, (tname, tp) in enumerate(TARGETS.items()):
                t0 = time.time()
                sol = S.solve([HandTarget("r", tp, fingers=(0, 0, 1), palm=(0, -1, 0), grip="pinch")], look=tp)
                ox, oz = (c - 1) * 110.0, -(r - 1) * 150.0
                xf = UsdGeom.Xform.Define(st, f"/World/Asset/{wid}_{tname}")
                xf.AddTranslateOp().Set(Gf.Vec3d(ox, 0, oz))
                write_worker(st, f"/World/Asset/{wid}_{tname}/Body", w, lib, pose=sol.pose, sk=sk)
                marks.lathe((ox + tp[0], tp[1] - 2.5, oz + tp[2]), (ox + tp[0], tp[1] + 2.5, oz + tp[2]),
                            [(0, 0.3), (0.5, 2.5), (1, 0.3)], "lamp_orange", sides=16)
                marks.box((ox + tp[0] - 0.4, 0, oz + tp[2] - 0.4), (ox + tp[0] + 0.4, tp[1] - 2.5, oz + tp[2] + 0.4),
                          "powder_grey")
                report[f"{wid}_{tname}"] = {k: round(float(v), 1) for k, v in sol.angles.items()}
                report[f"{wid}_{tname}"]["cost"] = round(float(sol.cost), 2)
                print(f"{wid} {tname}: {time.time() - t0:.1f} s, cost {sol.cost:.2f}")
        marks.build(st, "/World/Asset/Targets", lib.refs(list(marks.groups)))
        dress(st, lib, st.GetPrimAtPath("/World/Asset"))
    with open(os.path.splitext(out)[0] + ".json", "w") as f:
        json.dump(report, f, indent=1)
    print("->", out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "out", "people", "pose_lab.usd"))
