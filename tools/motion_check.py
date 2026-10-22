"""Run the motion quality checks on every generator (and walking) for the crew; print a table.

    python tools/motion_check.py [W50M] [--step 1]
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "exts", "frt.core"))
from frt_core.people import motion_carry as C   # noqa: E402
from frt_core.people import motion_gen as G   # noqa: E402
from frt_core.people import motion_more as M   # noqa: E402
from frt_core.people.bake import solve_frames   # noqa: E402
from frt_core.people.check import check_frames   # noqa: E402
from frt_core.people.gait import walk   # noqa: E402
from frt_core.people.skeleton import Skeleton   # noqa: E402
from frt_core.people.solver import PostureSolver   # noqa: E402
from frt_core.people.workers import WORKERS   # noqa: E402


def jobs(S, w, sk):
    return {
        "walk": walk(w, sk, [(0, 0), (300, 0), (300, 200)], 90.0, 0.0),
        "ratchet": G.ratchet(S, (8.0, 99.0, 45.0), strokes=2),
        "hammer": G.hammer(S, (0.0, 80.0, 50.0), strikes=2),
        "torque": M.torque(S, (8.0, 99.0, 45.0)),
        "screwdriver": M.screwdriver(S, (0.0, 105.0, 45.0), turns=2),
        "connector": M.connector(S, (0.0, 110.0, 40.0)),
        "pick_place": M.pick_place(S, (20.0, 95.0, 40.0), (-25.0, 95.0, 45.0)),
        "cable_tie": C.cable_tie(S, (10.0, 110.0, 40.0)),
        "crimp": C.crimp(S, (5.0, 105.0, 30.0)),
        "inspect": M.inspect(S, (0.0, 95.0, 50.0), seconds=0.5),
    }


def main(wid="W50M", step=1, only=None):
    w = WORKERS[wid]
    sk = Skeleton.for_worker(w)
    S = PostureSolver(sk, w)
    rows = []
    for name, fr in jobs(S, w, sk).items():
        if only and name not in only:
            continue
        t0 = time.time()
        sols = solve_frames(S, fr, step=step, log=lambda *a: None)
        r = check_frames(sk, fr, sols, name)
        rows.append(r)
        print(f"{name:12s} {r['frames']:4d} fr {time.time() - t0:5.1f}s  pop {r['worst_pop']:5.1f}  limit+ "
              f"{r['worst_limit_over']:5.1f}  contact {r['worst_contact_cm']:5.1f} cm  foot {r['worst_foot_cm']:4.1f} cm  "
              f"{r['counts']}")
        agg = {}
        for it in r["issues"]:
            k = (it[1], it[2])
            agg.setdefault(k, []).append((it[0], it[3]))
        for (kind, j), v in sorted(agg.items(), key=lambda kv: -len(kv[1]))[:8]:
            print(f"       {kind:8s} {j:14s} x{len(v):3d}  worst {max(x[1] for x in v):6.1f}  frames {v[0][0]}..{v[-1][0]}")
    return rows


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    only = next((x.split("=")[1].split(",") for x in sys.argv[1:] if x.startswith("--only=")), None)
    main(a[0] if a else "W50M", 1, only)
