"""Side-by-side scene: each job twice - rule-based (left, as in the library) and Kimodo-based (right: Kimodo's
generated body + our hands / tools) - on identical fixtures, with a sign over each, and a shot per job.

    python tools/kimodo/compare_scene.py ratchet kneel_drive box overhead
    -> out/people/kimodo/compare.usd + shots.json (tools/kit_render.py)
"""
import json
import os
import pickle
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "exts", "frt.core"))
from pxr import Gf, UsdGeom   # noqa: E402

from frt_core import units as U   # noqa: E402
from frt_core.catalog.details import sign   # noqa: E402
from frt_core.geom import Shapes   # noqa: E402
from frt_core.materials import MaterialLibrary   # noqa: E402
from frt_core.people.bake import bake   # noqa: E402
from frt_core.people.jobs import BY_ID   # noqa: E402
from frt_core.people.workers import WORKERS   # noqa: E402
from frt_core.studio import dress   # noqa: E402
from frt_core.usdio import safe_stage   # noqa: E402

OUT = os.path.join(ROOT, "out", "people", "kimodo")
GAP, DZ = 150.0, 420.0


def main():
    jobs = sys.argv[1:] or ["ratchet", "kneel_drive", "box", "overhead"]
    out = os.path.join(OUT, "compare.usd")
    shots = []
    with safe_stage(out) as st:
        U.setup_stage(st, fps=24.0)
        st.SetDefaultPrim(UsdGeom.Xform.Define(st, "/World").GetPrim())
        lib = MaterialLibrary(st, "/World/Looks")
        UsdGeom.Xform.Define(st, "/World/Asset")
        UsdGeom.Xform.Define(st, "/World/Asset/Parts")
        for i, jid in enumerate(jobs):
            job = BY_ID[jid]
            z = i * DZ
            versions = [("RULES", os.path.join(ROOT, "out", "people", "jobs", "cache", jid + ".pkl")),
                        ("ANDY", os.path.join(ROOT, "out", "people", "andy_fit", jid + ".pkl")),
                        ("KIMODO", os.path.join(OUT, jid, "kimodo.pkl"))]
            versions = [v for v in versions if os.path.isfile(v[1])]
            for k, (tag, path) in enumerate(versions):
                c = pickle.load(open(path, "rb"))
                x = (k - (len(versions) - 1) / 2) * GAP * 1.6
                base = f"/World/Asset/J{i}_{jid}_{tag}"
                UsdGeom.Xform.Define(st, base).AddTranslateOp().Set(Gf.Vec3d(x, 0.0, z))
                s = Shapes()
                job.fixture(s)
                sign(s, (0.0, 205.0, 95.0), (1, 0, 0), (0, 1, 0), (0, 0, 1), 40.0,
                     [job.title.upper(), {"RULES": "RULE-BASED", "ANDY": "RECORDED (ANDY) + OUR HANDS",
                                          "KIMODO": "KIMODO + OUR HANDS"}[tag]],
                     header=tag, header_mat={"RULES": "powder_blue", "ANDY": "powder_grey", "KIMODO": "powder_green"}[tag])
                s.build(st, f"{base}/Fixture", lib.refs(list(s.groups)))
                w = WORKERS[c["worker"]]
                UsdGeom.Xform.Define(st, f"{base}/{w.id}")
                bake(st, f"{base}/{w.id}", w, c["frames"], lib, sols=c["sols"], root=(x, z, 0.0),
                     parts_root="/World/Asset/Parts", log=lambda *a: None)
                wx = UsdGeom.Xformable(st.GetPrimAtPath(f"{base}/{w.id}"))
                for op in wx.GetOrderedXformOps():
                    if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                        for tc in op.GetTimeSamples():
                            op.Set(Gf.Vec3d(0.0, 0.0, 0.0), tc)
            n = len(c["frames"])
            fx, fy, fz = job.focus
            shots.append({"seq": f"{i + 1:02d}_{jid}", "eye": [-80.0, fy + 90.0, z + fz - 420.0],
                          "at": [0.0, fy - 5.0, z + fz - 10.0], "mm": 24, "frames": [0, n - 1, 1]})
            print(jid, n, "frames")
        dress(st, lib, st.GetPrimAtPath("/World/Asset"))
    json.dump(shots, open(os.path.join(OUT, "shots.json"), "w"), indent=1)
    print("->", out)


if __name__ == "__main__":
    main()
