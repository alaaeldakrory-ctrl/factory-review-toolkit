"""Intake of the AnDy dataset into the motion library: every labelled segment is retargeted onto a worker, checked
(joint limits, frame-to-frame jumps), measured and tagged, and listed in an index (the motion stays in the dataset
files; the index points into them).

Tags per clip: action (reach / pick / place / release / carry / screw / fine manipulation), EAWS posture, the
working hand(s), hand height band relative to the person (floor / knee / waist / chest / shoulder / overhead),
reach distance band, walking or not, duration, the person's stature, source + license.

    python tools/andy_library.py [--limit=N] [--workers=4]  -> out/people/library/andy_index.json + summary
"""
import json
import multiprocessing as mp
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "exts", "frt.core"))
OUT = os.path.join(HERE, "..", "out", "people", "library")
KEEP = ("Re", "Pi", "Pl", "Rl", "Ca", "Sc", "Fm")
PAD = 6                                                    # frames of lead-in / out around a segment


def band(h, stature):
    r = h / stature
    for lim, name in ((0.12, "floor"), (0.33, "knee"), (0.52, "waist"), (0.70, "chest"), (0.86, "shoulder")):
        if r < lim:
            return name
    return "overhead"


def do_trial(name):
    import numpy as np
    from frt_core.people import andy as A
    from frt_core.people.bake import solve_frames
    from frt_core.people.check import check_frames
    from frt_core.people.skeleton import Skeleton
    from frt_core.people.solver import PostureSolver
    from frt_core.people.workers import WORKERS
    out = []
    try:
        tr = A.Trial(name)
    except Exception as e:
        return [{"trial": name, "error": str(e)}]
    w = WORKERS["W50M"]
    sk = Skeleton.for_worker(w)
    S = PostureSolver(sk, w)
    stature = max(tr.joints(i)["head_top"][1] for i in range(0, tr.n, 48))
    for act, s, e in tr.segments():
        if act not in KEEP:
            continue
        a, b = max(0, s - PAD), min(tr.n, e + PAD)
        J = [tr.joints(i) for i in range(s, e)]
        wr = {sd: np.array([j[f"hand_{sd}"] for j in J]) for sd in ("l", "r")}
        travel = {sd: float(np.sum(np.linalg.norm(np.diff(wr[sd], axis=0), axis=1))) for sd in ("l", "r")}
        hands = "both" if min(travel.values()) > 0.6 * max(travel.values()) else max(travel, key=travel.get)
        main = "r" if hands == "both" else hands
        hy = float(np.median(wr[main][:, 1]))
        pel = np.array([j["pelvis"] for j in J])
        reach = float(np.median(np.linalg.norm((wr[main] - pel)[:, [0, 2]], axis=1)))
        walk = float(np.linalg.norm(pel[-1, [0, 2]] - pel[0, [0, 2]])) > 25.0
        post = max(set(tr.posture[s:e]), key=tr.posture[s:e].count)
        frames = tr.frames(sk, a, b)
        sols = solve_frames(S, frames, log=lambda *x: None)
        rep = check_frames(sk, frames, sols, name)
        bad = {k: v for k, v in rep["counts"].items() if k in ("limit", "pop")}
        out.append({"trial": name, "action": A.ACTIONS[act], "code": act, "start": a, "end": b,
                    "seconds": round((e - s) / 24.0, 2), "posture": post, "hands": hands,
                    "height": band(hy, stature), "reach_cm": round(reach), "walking": walk,
                    "stature": round(stature), "ok": not bad, "issues": bad,
                    "worst_pop": rep["worst_pop"], "worst_limit_over": rep["worst_limit_over"],
                    "source": "AnDy (Maurice et al. 2019)", "license": "CC BY 4.0"})
    return out


def main():
    from frt_core.people import andy as A
    opt = {a.split("=")[0][2:]: a.split("=")[1] for a in sys.argv[1:] if a.startswith("--")}
    names = A.trials()
    if "limit" in opt:
        names = names[:int(opt["limit"])]
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    clips = []
    with mp.Pool(int(opt.get("workers", 4))) as pool:
        for k, res in enumerate(pool.imap_unordered(do_trial, names)):
            clips += res
            print(f"{k + 1}/{len(names)} trials, {len(clips)} clips", flush=True)
    with open(os.path.join(OUT, "andy_index.json"), "w") as f:
        json.dump({"credit": A.CREDIT, "clips": clips}, f, indent=0)
    ok = [c for c in clips if c.get("ok")]
    print(f"\n{len(ok)}/{len(clips)} clips pass the checks ({time.time() - t0:.0f} s)")
    grid = {}
    for c in ok:
        grid.setdefault(c["action"], {}).setdefault(c["height"], 0)
        grid[c["action"]][c["height"]] += 1
    bands = ("floor", "knee", "waist", "chest", "shoulder", "overhead")
    print(f"{'':18s}" + "".join(f"{b:>9s}" for b in bands))
    for act, row in sorted(grid.items()):
        print(f"{act:18s}" + "".join(f"{row.get(b, 0):9d}" for b in bands))
    rej = {}
    for c in clips:
        for k in (c.get("issues") or {}):
            rej[k] = rej.get(k, 0) + 1
    print("rejected for:", rej)


if __name__ == "__main__":
    main()
