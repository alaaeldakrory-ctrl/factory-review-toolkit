"""Replay an AnDy trial (recorded industrial task motion, CC BY 4.0) on the crew, side by side, for judging how
recorded work motion looks on our bodies.

    python tools/andy_replay.py [trial] [start_s] [seconds]
    -> out/people/andy/replay.usd + shots.json (tools/kit_render.py)
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "exts", "frt.core"))
from pxr import Gf, UsdGeom   # noqa: E402

from frt_core import units as U   # noqa: E402
from frt_core.catalog.details import sign   # noqa: E402
from frt_core.geom import Shapes   # noqa: E402
from frt_core.materials import MaterialLibrary   # noqa: E402
from frt_core.people import andy as A   # noqa: E402
from frt_core.people.bake import bake, solve_frames   # noqa: E402
from frt_core.people.skeleton import Skeleton   # noqa: E402
from frt_core.people.solver import PostureSolver   # noqa: E402
from frt_core.people.workers import WORKERS   # noqa: E402
from frt_core.studio import dress   # noqa: E402
from frt_core.usdio import safe_stage   # noqa: E402

OUT = os.path.join(HERE, "..", "out", "people", "andy")


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "Participant_2274_Setup_B_Seq_1_Trial_5"
    t0 = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
    dur = float(sys.argv[3]) if len(sys.argv) > 3 else 30.0
    tr = A.Trial(name)
    a, b = int(t0 * 24), int((t0 + dur) * 24)
    print(name, tr.n, "frames;", [(A.ACTIONS.get(x, x), round(s / 24, 1)) for x, s, e in tr.segments() if a <= s < b][:30])
    os.makedirs(OUT, exist_ok=True)
    out = os.path.join(OUT, "replay.usd")
    J0 = tr.joints(a)
    with safe_stage(out) as st:
        U.setup_stage(st, fps=24.0)
        st.SetDefaultPrim(UsdGeom.Xform.Define(st, "/World").GetPrim())
        lib = MaterialLibrary(st, "/World/Looks")
        UsdGeom.Xform.Define(st, "/World/Asset")
        s = Shapes()
        for k in range(-6, 7):                                   # floor marks every 50 cm
            s.box((k * 50.0 - 0.4, 0.0, -150.0), (k * 50.0 + 0.4, 0.15, 350.0), "floor_white")
        sign(s, (-330.0, 150.0, 60.0), (1, 0, 0), (0, 1, 0), (0, 0, 1), 60.0,
             ["RECORDED MOTION: ANDY DATASET", "MAURICE ET AL. 2019, CC BY 4.0"], header="ANDY", header_mat="powder_blue")
        s.build(st, "/World/Asset/Marks", lib.refs(list(s.groups)))
        for i, wid in enumerate(("W05F", "W50M", "W95M")):
            w = WORKERS[wid]
            sk = Skeleton.for_worker(w)
            S = PostureSolver(sk, w)
            frames = tr.frames(sk, a, b)
            for f in frames:                                      # side by side, 160 cm apart
                f.root = (f.root[0] + (i - 1) * 160.0, f.root[1], f.root[2])
            sols = solve_frames(S, frames, log=lambda *a_: None)
            UsdGeom.Xform.Define(st, f"/World/Asset/{wid}")
            bake(st, f"/World/Asset/{wid}", w, frames, lib, sols=sols, log=lambda *a_: None)
            print(wid, len(frames), "frames")
        dress(st, lib, st.GetPrimAtPath("/World/Asset"))
    n = b - a
    shots = [{"seq": "replay_front", "eye": [60.0, 150.0, 520.0], "at": [0.0, 90.0, 60.0], "mm": 24, "frames": [0, n - 1, 1]},
             {"seq": "replay_side", "eye": [560.0, 140.0, 140.0], "at": [0.0, 90.0, 60.0], "mm": 26, "frames": [0, n - 1, 1]}]
    with open(os.path.join(OUT, "shots.json"), "w") as f:
        json.dump(shots, f, indent=1)
    print("->", out, n, "frames")


if __name__ == "__main__":
    main()
