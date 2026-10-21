"""People lab: the crew side by side in the studio (relaxed stance) with a height pole - checks proportions,
skinning and materials of the canonical skeleton and the mannequin bodies.

    python tools/people_lab.py [out/people/crew.usd]
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "exts", "frt.core"))
from pxr import Gf, UsdGeom   # noqa: E402

from frt_core import units as U   # noqa: E402
from frt_core.geom import Shapes   # noqa: E402
from frt_core.materials import MaterialLibrary   # noqa: E402
from frt_core.people.skeleton import Skeleton   # noqa: E402
from frt_core.people.usd_worker import relaxed, write_worker   # noqa: E402
from frt_core.people.workers import WORKERS   # noqa: E402
from frt_core.studio import dress   # noqa: E402
from frt_core.usdio import safe_stage   # noqa: E402


def main(out):
    with safe_stage(out) as st:
        U.setup_stage(st, fps=24.0)
        st.SetDefaultPrim(UsdGeom.Xform.Define(st, "/World").GetPrim())
        lib = MaterialLibrary(st, "/World/Looks")
        UsdGeom.Xform.Define(st, "/World/Asset")
        for k, wid in enumerate(("W05F", "W50M", "W95M")):
            w = WORKERS[wid]
            x = (k - 1) * 75.0
            xf = UsdGeom.Xform.Define(st, f"/World/Asset/{wid}")
            xf.AddTranslateOp().Set(Gf.Vec3d(x, 0, 0))
            sk = Skeleton.for_worker(w)
            write_worker(st, f"/World/Asset/{wid}/Body", w, lib, pose=relaxed(sk), sk=sk)
        s = Shapes()                                                    # height pole, 10 cm marks
        s.box((-150, 0, -40), (-147, 200, -37), "powder_lightgrey")
        for y in range(0, 201, 10):
            w_ = 6 if y % 50 == 0 else 3
            s.box((-150 - w_, y - 0.4, -40), (-150, y + 0.4, -37), "plastic_black")
        s.build(st, "/World/Asset/Pole", lib.refs(list(s.groups)))
        dress(st, lib, st.GetPrimAtPath("/World/Asset"))
    print("->", out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "out", "people", "crew.usd"))
