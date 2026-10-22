"""Generate, solve and check the hand-job library (in parallel), cache the solutions for baking.

    python tools/jobs_run.py [--only=ratchet,screwdriver] [--workers=6]
    -> out/people/jobs/cache/<job>.pkl  and  out/people/jobs/check.json
"""
import json
import multiprocessing as mp
import os
import pickle
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "exts", "frt.core"))
OUT = os.path.join(HERE, "..", "out", "people", "jobs")


def run(jid):
    from frt_core.people.bake import solve_frames
    from frt_core.people.check import check_frames
    from frt_core.people.jobs import BY_ID
    from frt_core.people.skeleton import Skeleton
    from frt_core.people.solver import PostureSolver
    from frt_core.people.workers import WORKERS
    job = BY_ID[jid]
    w = WORKERS[job.worker]
    sk = Skeleton.for_worker(w)
    S = PostureSolver(sk, w)
    t0 = time.time()
    try:
        frames = job.frames(S)
        sols = solve_frames(S, frames, step=1, log=lambda *a: None)
    except Exception as e:
        import traceback
        return {"job": jid, "error": f"{e}\n{traceback.format_exc()[-1500:]}"}
    rep = check_frames(sk, frames, sols, jid)
    with open(os.path.join(OUT, "cache", jid + ".pkl"), "wb") as f:
        pickle.dump({"frames": frames, "sols": sols, "worker": job.worker}, f)
    agg = {}
    for it in rep["issues"]:
        agg.setdefault(f"{it[1]}:{it[2]}", []).append(it[3])
    return {"job": jid, "worker": job.worker, "frames": rep["frames"], "seconds": round(time.time() - t0, 1),
            "worst_pop": rep["worst_pop"], "worst_limit_over": rep["worst_limit_over"],
            "contact_cm": rep["worst_contact_cm"], "counts": rep["counts"],
            "top": {k: (len(v), max(v)) for k, v in sorted(agg.items(), key=lambda kv: -len(kv[1]))[:6]}}


def main():
    sys.path.insert(0, os.path.join(HERE, "..", "exts", "frt.core"))
    from frt_core.people.jobs import JOBS
    only = next((a.split("=")[1].split(",") for a in sys.argv[1:] if a.startswith("--only=")), None)
    nw = int(next((a.split("=")[1] for a in sys.argv[1:] if a.startswith("--workers=")), 6))
    os.makedirs(os.path.join(OUT, "cache"), exist_ok=True)
    ids = [j.id for j in JOBS if not only or j.id in only]
    t0 = time.time()
    with mp.Pool(nw) as pool:
        res = pool.map(run, ids)
    path = os.path.join(OUT, "check.json")
    old = json.load(open(path)) if os.path.isfile(path) else {}
    for r in res:
        old[r["job"]] = r
    json.dump(old, open(path, "w"), indent=1)
    for r in res:
        if "error" in r:
            print(f"{r['job']:12s} ERROR {r['error'][:600]}")
            continue
        print(f"{r['job']:12s} {r['worker']} {r['frames']:4d} fr {r['seconds']:6.1f}s pop {r['worst_pop']:6.1f} "
              f"limit+ {r['worst_limit_over']:6.1f} contact {r['contact_cm']:5.1f}  {r['top']}")
    print(f"total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
