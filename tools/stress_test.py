"""Stress test: do the job recipes hold up on work they were not authored for?

Each job is re-generated with its work moved relative to the worker - left / right, higher / lower, nearer /
farther, turned (yaw) and tilted - and on a different body, then solved and run through the same motion checks
(joint limits, frame-to-frame jumps, contact, awkward postures, fixture collisions). Nothing is tuned per variant.

How the work is moved: a rigid transform M about the job's focus point applies to the work (tools at work, parts,
fixture keep-out planes, gaze). Hand targets and tools blend from untouched at their start / end positions (hands at
rest, a tool at the hip - body-relative) to fully moved once they have travelled 30 cm from there, so reaches stay
smooth.

    python tools/stress_test.py [--per-job=6] [--workers=12] [--only=a,b]
    -> out/people/stress/report.json + a summary table
"""
import json
import math
import multiprocessing as mp
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "exts", "frt.core"))
OUT = os.path.join(HERE, "..", "out", "people", "stress")

VARIANTS = [("left 25", dict(d=(25.0, 0.0, 0.0))), ("right 25", dict(d=(-25.0, 0.0, 0.0))),
            ("up 15", dict(d=(0.0, 15.0, 0.0))), ("down 15", dict(d=(0.0, -15.0, 0.0))),
            ("far 10", dict(d=(0.0, 0.0, 10.0))), ("near 6", dict(d=(0.0, 0.0, -6.0))),
            ("yaw +25", dict(yaw=25.0)), ("yaw -25", dict(yaw=-25.0)),
            ("tilt +20", dict(tilt=20.0)), ("tilt -15", dict(tilt=-15.0))]
BODIES = ("W05F", "W50M", "W95M")


def _rot(axis, deg):
    import numpy as np
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    th = math.radians(deg)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K


def work_transform(focus, d=(0, 0, 0), yaw=0.0, tilt=0.0):
    import numpy as np
    R = _rot((0, 1, 0), yaw) @ _rot((1, 0, 0), -tilt)            # tilt > 0: the work face tips back, up
    c = np.asarray(focus, float)
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = c + np.asarray(d, float) - R @ c
    return M


def _w(dist):
    u = min(1.0, max(0.0, (dist - 4.0) / 26.0))
    return u * u * (3 - 2 * u)


def apply(frames, M):
    """Move the work of a frame list by M (see module doc). Returns new frames (the input is not changed)."""
    import copy

    import numpy as np
    from frt_core.people.bake import _q, _R, _slerp
    from frt_core.people.solver import HandTarget
    R = M[:3, :3]
    qR = _q(R)
    out = copy.deepcopy(frames)
    ends = {}
    for side in ("l", "r"):
        pts = [np.asarray(t.point, float) for f in (frames[0], frames[-1]) for t in f.targets if t.side == side]
        ends[side] = pts
    tool_ends = {}
    for name in {k for f in frames for k in f.tools if not k.startswith("part:")}:
        tool_ends[name] = [f.tools[name][1][:3, 3] for f in (frames[0], frames[-1]) if name in f.tools]

    def moved_T(T, w):
        """T blended by w between itself and M @ T (rotation slerped)."""
        T2 = M @ T
        if w >= 0.999:
            return T2
        o = T[:3, :3]
        q = _slerp(_q(np.eye(3)), qR, w)
        Rw = _R(q)
        out_ = np.eye(4)
        out_[:3, :3] = Rw @ o
        out_[:3, 3] = T[:3, 3] * (1 - w) + T2[:3, 3] * w
        return out_
    for f, g in zip(frames, out):
        tg = []
        for t in f.targets:
            p = np.asarray(t.point, float)
            dist = min([np.linalg.norm(p - e) for e in ends.get(t.side, [])] or [99.0])
            w = _w(dist)
            Rw = _R(_slerp(_q(np.eye(3)), qR, w))
            p2 = (M[:3, :3] @ p + M[:3, 3]) * w + p * (1 - w)
            fi = None if t.fingers is None else tuple(Rw @ np.asarray(t.fingers, float))
            pa = None if t.palm is None else tuple(Rw @ np.asarray(t.palm, float))
            tg.append(HandTarget(t.side, tuple(p2), fi, pa, t.grip, t.weight))
        g.targets = tg
        for name, (key, T) in f.tools.items():
            if name.startswith("part:"):
                g.tools[name] = (key, M @ T)
            else:
                dist = min([np.linalg.norm(T[:3, 3] - e) for e in tool_ends.get(name, [])] or [99.0])
                g.tools[name] = (key, moved_T(T, _w(dist)))
        if f.look is not None:
            lp = np.asarray(f.look, float)
            g.look = tuple(M[:3, :3] @ lp + M[:3, 3])
    return out


def move_rules(rules, M):
    import numpy as np
    out = []
    for r in rules:
        joints, zmax = r[0], r[1]
        if len(r) == 3:
            p0, n = np.asarray(r[1], float), np.asarray(r[2], float)
        else:
            p0, n = np.array([0.0, 0.0, zmax]), np.array([0.0, 0.0, 1.0])
        out.append((joints, tuple(M[:3, :3] @ p0 + M[:3, 3]), tuple(M[:3, :3] @ n)))
    return out


def task_opts():
    return os.environ.get("FRT_STRESS_OPTS", "")


def run(task):
    jid, vname, vargs, wid = task
    import numpy as np
    from frt_core.people.bake import solve_frames
    from frt_core.people.check import check_frames
    from frt_core.people.jobs import BY_ID
    from frt_core.people.skeleton import Skeleton
    from frt_core.people.solver import PostureSolver
    from frt_core.people.workers import WORKERS
    job = BY_ID[jid]
    w = WORKERS[wid]
    sk = Skeleton.for_worker(w)
    S = PostureSolver(sk, w)
    t0 = time.time()
    try:
        from frt_core.people.placement import solve_placed
        M = work_transform(job.focus, **vargs)
        frames = apply(job.frames(S), M)
        rules = move_rules(job.keep_out_rules(), M)
        if "raw" in task_opts():                          # as authored only
            S.keep_out = rules
            sols = solve_frames(S, frames, step=1, log=lambda *a: None)
            rep = check_frames(sk, frames, sols, jid, keep_out=S.keep_out)
            spot, raw = (0.0, 0.0), rep
        else:                                             # the worker may step to a better spot when it fails
            frames, sols, rep, spot, raw = solve_placed(S, frames, rules)
        counts = dict(rep["counts"])
        from frt_core.people.placement import diagnose
        findings = diagnose(S, frames, rep, sols) if counts else []
        top = {}
        for it in rep["issues"]:
            top.setdefault(f"{it[1]}:{it[2]}", []).append(it[3])
        if job.partner:                                    # the partner moves with the work too
            wid2, place, build2 = job.partner
            w2 = WORKERS[wid2]
            sk2 = Skeleton.for_worker(w2)
            S2 = PostureSolver(sk2, w2)
            off = np.eye(4)
            off[:3, 3] = (place[0], 0.0, place[1])
            M2 = np.linalg.inv(off) @ M @ off
            f2 = apply(build2(S2, sk2, place), M2)
            s2 = solve_frames(S2, f2, step=1, log=lambda *a: None)
            rep2 = check_frames(sk2, f2, s2, jid)
            for k, v in rep2["counts"].items():
                counts[k] = counts.get(k, 0) + v
            for it in rep2["issues"]:
                top.setdefault(f"partner {it[1]}:{it[2]}", []).append(it[3])
        return {"job": jid, "variant": vname, "worker": wid, "ok": not counts, "counts": counts, "spot": spot,
                "ok_authored": bool(raw["ok"]), "counts_authored": dict(raw["counts"]), "findings": findings,
                "top": {k: (len(v), round(max(v), 1)) for k, v in sorted(top.items(), key=lambda kv: -len(kv[1]))[:4]},
                "frames": len(frames), "seconds": round(time.time() - t0, 1)}
    except Exception as e:
        import traceback
        return {"job": jid, "variant": vname, "worker": wid, "ok": False, "error": f"{e}",
                "trace": traceback.format_exc()[-800:]}


def main():
    from frt_core.people.jobs import JOBS
    opt = {a.split("=")[0][2:]: a.split("=")[1] for a in sys.argv[1:] if a.startswith("--")}
    per = int(opt.get("per-job", 6))
    nw = int(opt.get("workers", 12))
    only = opt["only"].split(",") if "only" in opt else None
    tasks = []
    for i, j in enumerate(JOBS):
        if only and j.id not in only:
            continue
        for k in range(per):
            vname, vargs = VARIANTS[(i + k * 3) % len(VARIANTS)]
            wid = BODIES[(i + k) % 3]
            tasks.append((j.id, vname, vargs, wid))
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    mode = "raw" if "raw" in opt else "placed"
    os.environ["FRT_STRESS_OPTS"] = mode
    done_path = os.path.join(OUT, f"results_{mode}.jsonl")   # one line per finished case: the run can be resumed
    res = []
    if os.path.isfile(done_path) and "fresh" not in opt:
        res = [json.loads(ln) for ln in open(done_path) if ln.strip()]
    done = {(r["job"], r["variant"], r["worker"]) for r in res}
    todo = [t for t in tasks if (t[0], t[1], t[3]) not in done]
    print(f"{len(res)} cases already done, {len(todo)} to run", flush=True)
    with mp.Pool(nw) as pool, open(done_path, "a") as log:
        for r in pool.imap_unordered(run, todo):
            res.append(r)
            log.write(json.dumps(r) + "\n")
            log.flush()
            print(f"{len(res):4d}/{len(tasks)} {r['job']:14s} {r['variant']:9s} {r['worker']} "
                  f"{'ok' if r['ok'] else 'FAIL ' + str(r.get('top') or r.get('error'))}", flush=True)
    with open(os.path.join(OUT, "report.json"), "w") as f:
        json.dump(res, f, indent=1)
    n_ok = sum(r["ok"] for r in res)
    n_a = sum(r.get("ok_authored", r["ok"]) for r in res)
    print(f"\nas authored: {n_a}/{len(res)} pass ({100.0 * n_a / max(1, len(res)):.0f} %);  with stepping: "
          f"{n_ok}/{len(res)} ({100.0 * n_ok / max(1, len(res)):.0f} %) in {time.time() - t0:.0f} s")
    for key, label in (("variant", "by variant"), ("worker", "by body")):
        groups = {}
        for r in res:
            groups.setdefault(r[key], []).append(r["ok"])
        print(label + ": " + ", ".join(f"{g} {sum(v)}/{len(v)}" for g, v in sorted(groups.items())))
    kinds = {}
    for r in res:
        for k, v in (r.get("counts") or {}).items():
            kinds[k] = kinds.get(k, 0) + 1
    print("failure kinds (variants affected):", kinds)


if __name__ == "__main__":
    main()
