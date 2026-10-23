"""Fit a recorded AnDy clip to one of our jobs: the clip (a real person doing that kind of action) gives the body,
stretched to the job's duration; our solver keeps that body (small corrections for keep-outs / reach / the planted
stance) and puts the hands, tools and fingers exactly on the job's work. The same final pass as the Kimodo
pipeline (S.base_correct).

    python tools/andy_fit.py screwdriver [--action=screw] [--height=chest] [--posture=St_U]
    -> out/people/andy_fit/<job>.pkl + check, and a compare scene (rules | AnDy-based)
"""
import json
import os
import pickle
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "exts", "frt.core"))
import numpy as np   # noqa: E402

OUT = os.path.join(HERE, "..", "out", "people", "andy_fit")


def pick_clips(action, height, posture, min_s, n=5):
    idx = json.load(open(os.path.join(HERE, "..", "out", "people", "library", "andy_index.json")))["clips"]
    cands = [c for c in idx if c.get("ok") and c["action"] == action and c["height"] == height
             and c["posture"] == posture and c["hands"] in ("r", "both") and c["seconds"] >= min_s and not c["walking"]]
    cands.sort(key=lambda c: (c["worst_pop"], -c["seconds"]))
    seen, out = set(), []
    for c in cands:                                        # one clip per person: variety among the tries
        p = c["trial"].split("_Setup")[0]
        if p not in seen:
            seen.add(p)
            out.append(c)
        if len(out) >= n:
            break
    return out


def main():
    from frt_core.people import andy as A
    from frt_core.people.bake import solve_frames
    from frt_core.people.check import check_frames
    from frt_core.people.jobs import BY_ID
    from frt_core.people.skeleton import Skeleton
    from frt_core.people.solver import PostureSolver
    from frt_core.people.workers import WORKERS
    jid = sys.argv[1]
    opt = {a.split("=")[0][2:]: a.split("=")[1] for a in sys.argv[2:] if a.startswith("--")}
    c = pickle.load(open(os.path.join(HERE, "..", "out", "people", "jobs", "cache", jid + ".pkl"), "rb"))
    job = BY_ID[jid]
    w = WORKERS[c["worker"]]
    sk = Skeleton.for_worker(w)
    frames = c["frames"]
    tries = pick_clips(opt.get("action", "screw"), opt.get("height", "chest"), opt.get("posture", "St_U"),
                       float(opt.get("min_s", 2.0)), int(opt.get("tries", 5)))
    if not tries:
        print("no matching clip")
        return
    best = None
    for clip in tries:                                     # the library choice: try the best matches, keep the best fit
        tr = A.Trial(clip["trial"])
        a = max(0, clip["start"] - 24)
        b = min(tr.n, clip["end"] + 24)
        rec = tr.frames(sk, a, b)
        n = len(frames)
        new = []
        for i, f in enumerate(frames):
            g = rec[int(round(i / max(n - 1, 1) * (len(rec) - 1)))]
            new.append(type(f)(f.targets, f.look, dict(f.tools), root=f.root, stance=f.stance, base=g.base.copy(),
                               label=f.label))
        S = PostureSolver(sk, w)
        S.keep_out = job.keep_out_rules()
        S.base_correct = True
        sols = solve_frames(S, new, log=lambda *a_: None)
        rep = check_frames(sk, new, sols, jid + " andy", keep_out=S.keep_out)
        bad = sum(rep["counts"].values())
        print(f"  {clip['trial']} {clip['seconds']}s -> {'ok' if rep['ok'] else rep['counts']}")
        if best is None or bad < best[0]:
            best = (bad, clip, new, sols, rep)
        if rep["ok"]:
            break
    _, clip, new, sols, rep = best
    os.makedirs(OUT, exist_ok=True)
    pickle.dump({"frames": new, "sols": sols, "worker": c["worker"], "clip": clip},
                open(os.path.join(OUT, jid + ".pkl"), "wb"))
    print(json.dumps({"job": jid, "ok": rep["ok"], "counts": rep["counts"], "contact_cm": rep["worst_contact_cm"]}))


if __name__ == "__main__":
    main()
