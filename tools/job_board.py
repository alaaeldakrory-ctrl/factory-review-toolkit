"""Job board: every hand job in the library (people/jobs.py) at its own station, on the real worker bodies, baked
from the checked solutions (tools/jobs_run.py cache). Writes the scene and a shots file for tools/kit_render.py:
one close-up sequence per job and a path-traced overview.

    python tools/jobs_run.py            # solve + check (cache)
    python tools/job_board.py [--only=a,b] [--cols=6]
    python tools/kit_render.py job_board out/people/jobs/board.usd out/people/jobs/render out/people/jobs/shots.json
"""
import json
import os
import pickle
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "exts", "frt.core"))
from pxr import Gf, UsdGeom   # noqa: E402

from frt_core import units as U   # noqa: E402
from frt_core.catalog.details import sign   # noqa: E402
from frt_core.geom import Shapes   # noqa: E402
from frt_core.materials import MaterialLibrary   # noqa: E402
from frt_core.people.bake import bake   # noqa: E402
from frt_core.people.jobs import JOBS   # noqa: E402
from frt_core.people.motion_gen import FPS   # noqa: E402
from frt_core.people.workers import WORKERS   # noqa: E402
from frt_core.studio import dress   # noqa: E402
from frt_core.usdio import safe_stage   # noqa: E402

OUT = os.path.join(HERE, "..", "out", "people", "jobs")
DX, DZ = 330.0, 380.0                       # station spacing


def station(i, cols):
    return ((i % cols) - (cols - 1) / 2) * DX, (i // cols) * DZ


def close_up(job, x, z):
    """Camera for the hands: panel jobs from the right side past the panel edge, bench jobs from across the bench."""
    fx, fy, fz = job.focus
    at = [x + fx, fy - 4.0, z + fz]
    if job.camera:                                 # the job's own close-up: eye offset from the focus
        return [at[0] + job.camera[0], at[1] + job.camera[1], at[2] + job.camera[2]], at
    if getattr(job.fixture, "kind", "") == "panel":
        eye = [at[0] - 150.0, at[1] + 30.0, at[2] - 30.0]
        at = [at[0] + 8.0, at[1] + 8.0, at[2] - 14.0]
    else:
        eye = [at[0] - 85.0, at[1] + 70.0, at[2] + 145.0]
        at = [at[0], at[1] + 14.0, at[2] - 10.0]
    return eye, at


def main():
    only = next((a.split("=")[1].split(",") for a in sys.argv[1:] if a.startswith("--only=")), None)
    cols = int(next((a.split("=")[1] for a in sys.argv[1:] if a.startswith("--cols=")), 10))
    jobs = [j for j in JOBS if not only or j.id in only]
    out = os.path.join(OUT, "board.usd")
    shots, length = [], 0
    with safe_stage(out) as st:
        U.setup_stage(st, fps=FPS)
        st.SetDefaultPrim(UsdGeom.Xform.Define(st, "/World").GetPrim())
        lib = MaterialLibrary(st, "/World/Looks")
        UsdGeom.Xform.Define(st, "/World/Asset")
        UsdGeom.Xform.Define(st, "/World/Asset/Parts")
        for i, job in enumerate(jobs):
            cache = os.path.join(OUT, "cache", job.id + ".pkl")
            if not os.path.isfile(cache):
                print("no cache for", job.id, "- run tools/jobs_run.py")
                continue
            with open(cache, "rb") as f:
                c = pickle.load(f)
            x, z = station(i, cols)
            base = f"/World/Asset/S{i:02d}_{job.id}"
            xf = UsdGeom.Xform.Define(st, base)
            xf.AddTranslateOp().Set(Gf.Vec3d(x, 0.0, z))
            s = Shapes()
            job.fixture(s)
            sign(s, (68.0, 150.0, 70.0), (1, 0, 0), (0, 1, 0), (0, 0, 1), 34.0, [job.title.upper(), c["worker"]],
                 header=f"{i + 1:02d}", header_mat="powder_blue")
            s.rprism((68.0, 0.0, 69.5), (68.0, 140.0, 69.5), 3.0, 3.0, 0.4, "powder_dark", up=(1, 0, 0))
            s.rbox((68.0, 0.5, 69.5), (24.0, 1.0, 24.0), 0.3, "powder_dark", seg=1)
            s.build(st, f"{base}/Fixture", lib.refs(list(s.groups)))
            w = WORKERS[c["worker"]]
            UsdGeom.Xform.Define(st, f"{base}/{w.id}")
            bake(st, f"{base}/{w.id}", w, c["frames"], lib, sols=c["sols"], root=(x, z, 0.0),
                 parts_root="/World/Asset/Parts", log=lambda *a: None)
            # bake's root sets the worker's own transform in world terms: the worker prim sits under the station,
            # so take the station offset back off
            wx = UsdGeom.Xformable(st.GetPrimAtPath(f"{base}/{w.id}"))
            for op in wx.GetOrderedXformOps():
                if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                    for tc in op.GetTimeSamples():
                        op.Set(Gf.Vec3d(0.0, 0.0, 0.0), tc)
            if c.get("partner"):                          # the second worker, placed in the station frame
                pc = c["partner"]
                w2 = WORKERS[pc["worker"]]
                px, pz, pyaw = pc["place"]
                UsdGeom.Xform.Define(st, f"{base}/{w2.id}_2")
                bake(st, f"{base}/{w2.id}_2", w2, pc["frames"], lib, sols=pc["sols"], root=(px, pz, pyaw),
                     parts_root="/World/Asset/Parts", log=lambda *a: None)
            n = len(c["frames"])
            length = max(length, n)
            eye, at = close_up(job, x, z)
            shots.append({"seq": f"{i + 1:02d}_{job.id}", "eye": eye, "at": at, "mm": 30, "frames": [0, n - 1, 1]})
            print(f"{i + 1:02d} {job.id:12s} {w.id} {n} fr")
        dress(st, lib, st.GetPrimAtPath("/World/Asset"))
    rows = (len(jobs) + cols - 1) // cols
    cx, cz = 0.0, (rows - 1) * DZ / 2
    shots.insert(0, {"name": "overview.png", "eye": [cx - 520.0, 640.0, cz + rows * DZ * 0.5 + 760.0],
                     "at": [cx, 60.0, cz + 40.0], "mm": 24, "frame": 40, "settle": 220})
    with open(os.path.join(OUT, "shots.json"), "w") as f:
        json.dump(shots, f, indent=1)
    print("->", out, length, "frames;", len(shots), "shots")


if __name__ == "__main__":
    main()
