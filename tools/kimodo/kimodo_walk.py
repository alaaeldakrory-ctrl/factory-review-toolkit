"""Walking: Kimodo along a root path (no text) vs our clip-based walk (locomotion.py) on the same route, side by
side, with planted-foot slip measured for both.

    python tools/kimodo/kimodo_walk.py [W50M] -> out/people/kimodo/walk/walk.usd + shots.json
"""
import json
import math
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "exts", "frt.core"))
sys.path.insert(0, HERE)
import numpy as np   # noqa: E402
from pxr import UsdGeom   # noqa: E402

from frt_core import units as U   # noqa: E402
from frt_core.geom import Shapes   # noqa: E402
from frt_core.materials import MaterialLibrary   # noqa: E402
from frt_core.people.bake import bake, solve_frames   # noqa: E402
from frt_core.people.gait import walk   # noqa: E402
from frt_core.people.mocap import retarget   # noqa: E402
from frt_core.people.motion_gen import FPS, Frame   # noqa: E402
from frt_core.people.motion_more import idle   # noqa: E402
from frt_core.people.skeleton import Skeleton   # noqa: E402
from frt_core.people.solver import PostureSolver, Stance   # noqa: E402
from frt_core.people.workers import WORKERS   # noqa: E402
from frt_core.studio import dress   # noqa: E402
from frt_core.usdio import safe_stage   # noqa: E402
from kimodo_job import KIMODO_HIP_M, KIMODO_PY, soma_calibration, soma_to_canonical   # noqa: E402

OUT = os.path.join(ROOT, "out", "people", "kimodo", "walk")
PATH_CM = [(0.0, 0.0), (0.0, 60.0), (20.0, 160.0), (80.0, 260.0), (160.0, 330.0), (200.0, 360.0)]   # x, z (cm)


def main():
    wid = sys.argv[1] if len(sys.argv) > 1 else "W50M"
    w = WORKERS[wid]
    sk = Skeleton.for_worker(w)
    s = KIMODO_HIP_M / float(sk.world["pelvis"][1])
    os.makedirs(OUT, exist_ok=True)
    secs = 6.0
    fr = [0, 30, 60, 90, 120, 150]
    job = {"seconds": secs, "seed": 3, "steps": 100,
           "constraints": [{"type": "root2d", "frame_indices": fr,
                            "smooth_root_2d": [[x * s, z * s] for x, z in PATH_CM]}]}
    jp, gp = os.path.join(OUT, "walk_job.json"), os.path.join(OUT, "kimodo_walk.npz")
    json.dump(job, open(jp, "w"))
    subprocess.run([KIMODO_PY, os.path.join(HERE, "kimodo_run.py"), jp, gp], check=True, capture_output=True,
                   env=dict(os.environ, HF_HOME=r"C:/OmniverseData/kimodo/hf"))
    d = np.load(gp)
    pj, kfps = d["posed_joints"], float(d["fps"])
    from kimodo_names import NAMES77
    frames_k = []
    cal = soma_calibration(sk, s)
    for i in range(int(secs * FPS)):
        x = i / FPS * kfps
        a = min(int(x), len(pj) - 1)
        b = min(a + 1, len(pj) - 1)
        J = soma_to_canonical(pj[a] * (1 - (x - a)) + pj[b] * (x - a), NAMES77, s)
        pose = retarget(sk, J, cal)
        pose.pelvis = J["pelvis"] - sk.world["pelvis"]            # travel rides on the pelvis (root fixed)
        st = Stance()
        st.free = True
        frames_k.append(Frame([], None, {}, stance=st, base=pose, label="kimodo walk"))
    from kimodo_job import floor_fix
    floor_fix(sk, [f.base for f in frames_k])
    S = PostureSolver(sk, w)
    sols_k = solve_frames(S, frames_k, log=lambda *a: None)
    # ours: the clip walk along the same route
    frames_o = idle(0.4) + walk(w, sk, PATH_CM, 0.0, 30.0)
    for f in frames_o:
        if f.root is None:
            f.root = (0.0, 0.0, 0.0)
    last = next(f.root for f in reversed(frames_o) if f.root is not None)
    tail = idle(0.8)
    for f in tail:                                            # standing where the walk ended
        f.root = last
    frames_o += tail
    sols_o = solve_frames(S, frames_o, log=lambda *a: None)
    out = os.path.join(OUT, "walk.usd")
    with safe_stage(out) as st:
        U.setup_stage(st, fps=FPS)
        st.SetDefaultPrim(UsdGeom.Xform.Define(st, "/World").GetPrim())
        lib = MaterialLibrary(st, "/World/Looks")
        UsdGeom.Xform.Define(st, "/World/Asset")
        sh = Shapes()
        for dx in (-90.0, 90.0):                                 # the route, marked on the floor for both
            for (x0, z0), (x1, z1) in zip(PATH_CM, PATH_CM[1:]):
                sh.sweep([(x0 + dx, 0.2, z0), (x1 + dx, 0.2, z1)], 1.0, "floor_yellow", sides=6)
        sh.build(st, "/World/Asset/Route", lib.refs(list(sh.groups)))
        for name, frames, sols, dx in (("Ours", frames_o, sols_o, -90.0), ("Kimodo", frames_k, sols_k, 90.0)):
            p = f"/World/Asset/{name}"
            UsdGeom.Xform.Define(st, p)
            for f in frames:
                if f.root is not None:
                    f.root = (f.root[0] + dx, f.root[1], f.root[2])
            bake(st, p, w, frames, lib, sols=sols, root=(dx, 0.0, 0.0), log=lambda *a: None)
        dress(st, lib, st.GetPrimAtPath("/World/Asset"))
    n = max(len(frames_o), len(frames_k))
    json.dump([{"seq": "walk_compare", "eye": [620.0, 330.0, -60.0], "at": [80.0, 70.0, 200.0], "mm": 22,
                "frames": [0, n - 1, 1]}], open(os.path.join(OUT, "shots.json"), "w"), indent=1)
    print("->", out, "ours", len(frames_o), "kimodo", len(frames_k), "frames")


if __name__ == "__main__":
    main()
