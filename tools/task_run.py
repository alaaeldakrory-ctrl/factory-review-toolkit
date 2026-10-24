"""Run a task file: compile the steps, solve every frame, check, score (RULA) and bake one scene.

    python tools/task_run.py docs/people/practice/reference_task.toml [--out=out/people/task/<name>] [--workers=11]
        [--worker=W05F]   (the same task on another body: an frt body id or a character .usd)
    -> <out>/task.usda (scene + worker + parts), result.json (per step: frames, checks, RULA), log

The Kit window (Task Builder) runs this in the background and opens the result.
"""
import json
import multiprocessing as mp
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "exts", "frt.core"))
os.environ.setdefault("PXR_USDC_EMIT_DEPRECATION_WARNINGS", "0")

_T = None
_TS = {}                                                             # (--motion=best: both versions)


def _load(path, worker=None):
    from frt_core.people.tasks import Task
    T = Task.load(path)
    if worker:
        T.data["task"]["worker"] = worker
    return T


def _init(path, worker=None, motion=None):
    global _T
    keys = ("solver", "andy") if motion == "best" else (motion or "solver",)
    for k in keys:
        T = _load(path, worker).compile(log=lambda *a: None)
        _motion(T, k if motion == "best" else motion, log=lambda *a: None)   # (best: the solver one stays plain)
        _TS[k] = T
    _T = _TS[keys[0]]


def _motion(T, motion, log=print):
    """--motion=andy (or motion = "andy" in [task]): real recorded bodies under the work (andy_base)."""
    if (motion or T.data["task"].get("motion")) == "andy":
        from frt_core.people.andy_base import apply
        apply(T, log=log)


def _solve(job):
    from frt_core.people.bake import solve_frames
    k, a, b = job
    T = _TS[k]
    return k, a, solve_frames(T.S, T.frames[a:b + 1], log=lambda *a: None)


def _apart(p, q):
    """The biggest joint difference between two solved poses, deg."""
    from frt_core.people.check import rot_angle
    from frt_core.people.skeleton import NAMES
    return max(rot_angle(p.pose.local(j).T @ q.pose.local(j)) for j in NAMES if j != "root")


def _join(job):
    """Carry on from the frame before a piece until the motion meets the piece's own solution (a piece warm-started
    from rest can have the elbow on the other side: handing over after a fixed number of frames was a snap)."""
    from frt_core.people.bake import solve_frames
    k, a, prev, own = job
    T = _TS[k]
    n = len(T.frames)
    out, i = [], a
    while i < n and i - a < len(own):
        hi = min(n, i + 10, a + len(own))
        fix = solve_frames(T.S, T.frames[i:hi], log=lambda *a: None, prev=prev)
        out += fix
        prev, i = fix[-1], hi
        if _apart(fix[-1], own[hi - 1 - a]) < 8.0:
            break
    return a, out


def solve_set(pool, k, rngs, n, say, p0=0.05, p1=0.75):
    """Every frame of one version (k: a key of _TS) solved in pieces, then the joins between them."""
    sols = [None] * n
    done = 0
    for _, a, s in pool.imap_unordered(_solve, [(k,) + r for r in rngs]):
        sols[a:a + len(s)] = s
        done += len(s)
        say(f"solving {done}/{n} frames" + (f" ({k})" if k != "solver" else ""), progress=p0 + (p1 - p0) * done / n)
    # the joins between pieces, each carried on from the frame before it (all at once: they are independent)
    say("joining the pieces", progress=p1)
    jobs = [(k, a, sols[a - 1], sols[a:b + 1]) for a, b in rngs[1:]]
    own = {a: o for _, a, _, o in jobs}
    ran = {}
    for a, s in pool.imap_unordered(_join, jobs):
        sols[a:a + len(s)] = s
        ran[a] = len(s) == len(own[a])
    # a join that ran through its whole piece changed that piece's last frame: the next join started from the
    # old one (a wrong-branch arm carried on) - redo it from the new end, in order
    for (a, _), (a2, _) in zip(rngs[1:], rngs[2:]):
        if ran.get(a):
            _, s = pool.apply(_join, ((k, a2, sols[a2 - 1], own[a2]),))
            sols[a2:a2 + len(s)] = s
            ran[a2] = len(s) == len(own[a2])
    return sols


def _badness(iss):
    """One number per step to compare two versions: every issue counts, the far-over ones more."""
    return sum(1.0 + min(_severity(i), 40.0) / 10.0 for i in iss)


def _fold(T, sols, a, b, over=60.0):
    """How far past `over` the trunk bends (pelvis -> neck from upright), summed over the step's frames / 10: the
    checks don't see a body folded double over the work, a person wouldn't work like that."""
    import numpy as np

    from frt_core.people.kinematics import fk
    tot = 0.0
    for i in range(a, b + 1):
        _, P = fk(T.sk, sols[i].pose, ["neck_01"])
        v = P["neck_01"] - P["pelvis"]
        ang = np.degrees(np.arccos(np.clip(v[1] / max(np.linalg.norm(v), 1e-9), -1.0, 1.0)))
        tot += max(0.0, ang - over) / 10.0
    return tot


def best_of_both(T, TA, pool, rngs, n, say, usd):
    """--motion=best: the task solved twice (solver only / recorded bodies), each step kept from the version with
    fewer and smaller check issues, the hand-over into a switched step re-solved from the kept frame before it."""
    from frt_core.people.bake import _rootT
    from frt_core.people.check import check_frames
    S0 = solve_set(pool, "solver", rngs, n, say, 0.05, 0.40)
    SA = solve_set(pool, "andy", rngs, n, say, 0.40, 0.72)
    bake_scene(T, T.smooth_solutions(S0), usd)                        # (the equipment is the same in both)
    field = T.field or equipment_field(T, usd)
    for f in TA.frames:
        f.world = _rootT(f.root)
    s0, sA = T.smooth_solutions(S0), TA.smooth_solutions(SA)
    r0 = check_frames(T.sk, T.frames, s0, field=field)
    rA = check_frames(TA.sk, TA.frames, sA, field=field)
    pick, sols = {}, list(S0)
    for s in T.steps:
        a, b = s["start"], s["end"]
        b0 = _badness([i for i in r0["issues"] if a <= i[0] <= b]) + _fold(T, s0, a, b)
        bA = _badness([i for i in rA["issues"] if a <= i[0] <= b]) + _fold(TA, sA, a, b)
        any_base = any(TA.frames[i].base is not None and T.frames[i].base is None for i in range(a, b + 1))
        k = "andy" if any_base and bA < 0.8 * b0 else "solver"
        pick[s["id"]] = {"from": k, "solver": round(b0, 1), "andy": round(bA, 1)}
        say(f"  {s['id']}: solver {b0:.0f} / recorded {bA:.0f} -> {k}")
        if k == "andy":
            sols[a:b + 1] = SA[a:b + 1]
            T.frames[a:b + 1] = TA.frames[a:b + 1]
    # where the source switches, carry on from the kept frame before (both ways), in order
    say("joining the switches", progress=0.76)
    keys = [pick[s["id"]]["from"] for s in T.steps]
    for j in range(1, len(T.steps)):
        if keys[j] != keys[j - 1]:
            s = T.steps[j]
            a, b = s["start"], s["end"]
            own = (SA if keys[j] == "andy" else S0)[a:b + 1]
            _, fix = pool.apply(_join, ((keys[j], a, sols[a - 1], own),))
            sols[a:a + len(fix)] = fix
    if "andy" in keys:
        T.S.base_correct = True
    return sols, pick


def chunks(steps, n_frames, size=160):
    """Solve ranges: whole steps, long steps split (each piece warm-starts from rest, joins are re-solved below)."""
    out = []
    bounds = [s["start"] for s in steps] + [n_frames]
    for a, b in zip(bounds, bounds[1:]):
        while b - a > size * 1.5:
            out.append((a, a + size - 1))
            a += size
        out.append((a, b - 1))
    return out


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opt = {a.split("=")[0][2:]: a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--") and "=" in a}
    path = os.path.abspath(args[0] if args else os.path.join(ROOT, "docs", "people", "practice", "reference_task.toml"))
    name = os.path.splitext(os.path.basename(path))[0]
    worker = opt.get("worker")
    if worker:
        name += "_" + os.path.splitext(os.path.basename(worker))[0]
    out = os.path.abspath(opt.get("out", os.path.join(ROOT, "out", "people", "task", name)))
    os.makedirs(out, exist_ok=True)
    status = os.path.join(out, "status.json")

    def say(msg, **kw):
        print(msg, flush=True)
        with open(status + ".tmp", "w") as f:
            json.dump({"msg": msg, "t": time.time(), **kw}, f)
        os.replace(status + ".tmp", status)                         # the window never reads half a file
    t0 = time.time()
    scene_only = "--scene-only" in sys.argv
    try:
        T = _load(path, worker)
        if scene_only:                                               # just the scene and the worker at the start
            T.data["step"] = []
        T.compile(log=print)
        _motion(T, opt.get("motion"))
        if scene_only:
            from frt_core.people.bake import solve_frames
            sols = solve_frames(T.S, T.frames, log=lambda *a: None)
            usd = os.path.join(out, "scene.usda")
            bake_scene(T, sols, usd)
            say("scene ready - add steps, then Solve", progress=1.0, done=True, usd=usd)
            return
    except Exception as e:                                           # authoring errors go back to the window
        import traceback
        say(f"task error: {e}", error=str(e), trace=traceback.format_exc()[-2000:], done=True)
        sys.exit(1)
    n = len(T.frames)
    say(f"compiled {len(T.steps)} steps, {n} frames ({n / 24:.1f} s); solving", progress=0.05)
    nw = int(opt.get("workers", min(11, os.cpu_count() or 4)))
    rngs = chunks(T.steps, n, size=max(24, min(160, -(-n // nw))))    # short tasks: small pieces, every core busy
    pick = None
    with mp.Pool(nw, initializer=_init, initargs=(path, worker, opt.get("motion"))) as pool:
        if opt.get("motion") == "best":
            TA = _load(path, worker).compile(log=lambda *a: None)
            _motion(TA, "andy")
            sols, pick = best_of_both(T, TA, pool, rngs, n, say, os.path.join(out, "task.usda"))
        else:
            sols = solve_set(pool, opt.get("motion") or "solver", rngs, n, say)
    if os.environ.get("FRT_DUMP_SOLS"):                             # (debugging: the solved poses, before smoothing)
        import pickle
        pickle.dump({"sols": sols, "rngs": rngs}, open(os.environ["FRT_DUMP_SOLS"], "wb"))
    sols = T.smooth_solutions(sols)
    say("checking", progress=0.82)
    import numpy as np

    from frt_core.people.bake import _rootT
    from frt_core.people.check import check_frames
    say("baking the scene", progress=0.84)
    usd = os.path.join(out, "task.usda")
    bake_scene(T, sols, usd)
    rep = check_frames(T.sk, T.frames, sols, name, field=T.field or equipment_field(T, usd))
    say("scoring (RULA)", progress=0.93)
    scores = score(usd)
    steps = []
    for s in T.steps:
        a, b = s["start"], s["end"]
        iss = [i for i in rep["issues"] if a <= i[0] <= b]
        cnt = {}
        for i in iss:
            cnt[i[1]] = cnt.get(i[1], 0) + 1
        sc = [scores[i] for i in range(a, b + 1) if i < len(scores) and scores[i] is not None]
        foc = [_rootT(T.frames[i].root)[:3, :3] @ np.asarray(T.frames[i].look, float) + _rootT(T.frames[i].root)[:3, 3]
               for i in range(a, b + 1) if T.frames[i].look is not None and T.frames[i].base is None]
        worst = {}
        for i in iss:                                                 # how bad, not only how often
            if isinstance(i[3], (int, float)):
                over = float(i[3]) - float(i[4] or 0.0) if i[1] in ("limit", "pop", "awkward") else float(i[3])
                worst[i[1]] = round(max(worst.get(i[1], 0.0), over), 1)
        steps.append({**s, "seconds": round((b - a + 1) / 24.0, 1), "checks": cnt, "worst": worst,
                      "focus": [round(float(v), 1) for v in np.median(foc, axis=0)] if foc else None,
                      "rula_max": max(sc) if sc else None, "rula_mean": round(sum(sc) / len(sc), 2) if sc else None,
                      "by_joint": {f"{k[0]} {k[1]}": v for k, v in
                                   sorted(__import__("collections").Counter((i[1], i[2]) for i in iss).items(),
                                          key=lambda kv: -kv[1])},
                      "issues": [list(i) for i in sorted(iss, key=lambda i: -_severity(i))[:12]]})
    res = {"task": T.name, "file": path, "usd": usd, "frames": n, "fps": 24, "steps": steps,
           "counts": rep["counts"], "notes": T.notes, "motion": pick, "solve_s": round(time.time() - t0, 1)}
    json.dump(res, open(os.path.join(out, "result.json"), "w"), indent=1)
    say(f"done in {time.time() - t0:.0f} s: {rep['counts'] or 'clean'}", progress=1.0, done=True, usd=usd)


def equipment_field(T, usd):
    """The scene's still equipment (not the worker, not the parts) as a distance field, so the checks also find
    the body inside a fixture, a wheel or a cart."""
    import numpy as np
    from pxr import Usd

    from frt_core.people.bake import _rootT
    from frt_core.people.collide import SceneField, region_around
    for f in T.frames:
        f.world = _rootT(f.root)
    try:
        st = Usd.Stage.Open(usd)
        pts = np.array([[f.root[0], 0.0, f.root[1]] for f in T.frames])
        return SceneField.from_stage(st, region_around(pts, margin=120.0),
                                     exclude=("/World/Asset/Worker", "/World/Asset/Parts"))
    except Exception as e:                                           # no field: the other checks still run
        print(f"(no equipment check: {e})")
        return None


def _severity(i):
    """How far past its threshold an issue is (to keep the worst ones)."""
    try:
        return float(i[3]) - float(i[4] or 0.0)
    except (TypeError, ValueError, IndexError):
        return 0.0


def bake_scene(T, sols, usd):
    from pxr import Gf, UsdGeom, UsdLux

    from frt_core import units as U
    from frt_core.materials import MaterialLibrary
    from frt_core.people.bake import bake
    from frt_core.studio import dress
    from frt_core.usdio import safe_stage
    with safe_stage(usd) as st:
        U.setup_stage(st, fps=24.0)
        st.SetDefaultPrim(UsdGeom.Xform.Define(st, "/World").GetPrim())
        st.SetStartTimeCode(0)
        st.SetEndTimeCode(len(T.frames) - 1)
        lib = MaterialLibrary(st, "/World/Looks")
        UsdGeom.Xform.Define(st, "/World/Asset")
        T.build_static(st, "/World/Asset/Scene", lib)
        UsdGeom.Xform.Define(st, "/World/Asset/Parts")
        bake(st, "/World/Asset/Worker", T.w, T.frames, lib, sols=sols, parts_root="/World/Asset/Parts",
             log=lambda *a: None, character=T.character)
        # parts nobody touched still need to be in the scene where they lie
        from frt_core.people.bake import part_shapes
        for pid, p in T.parts.items():
            pp = f"/World/Asset/Parts/{pid}"
            if st.GetPrimAtPath(pp):
                continue
            xf = UsdGeom.Xform.Define(st, pp)
            xf.AddTransformOp().Set(Gf.Matrix4d(*p["W"].T.flatten().tolist()))
            shp = part_shapes(p["spec"])
            shp.build(st, f"{pp}/Geom", lib.refs(list(shp.groups)))
        dress(st, lib, st.GetPrimAtPath("/World/Asset"))
        st.GetRootLayer().customLayerData = {"frt_task": T.name}


def score(usd):
    """RULA per frame for the worker, with the independent scan (the same one Ergo Scan uses)."""
    try:
        from pxr import Usd

        from frt_core.people import ergo_scan as E
        rep = E.scan(Usd.Stage.Open(usd), step=1, log=lambda *a: None)
        op = rep["operators"][0]
        rows = op["rows"] if "rows" in op else op.get("frames", [])
        out = {}
        for r in rows:
            out[int(round(r["frame"]))] = r["score"]
        n = max(out) + 1 if out else 0
        return [out.get(i) for i in range(n)]
    except Exception as e:
        print("scoring failed:", e)
        return []


if __name__ == "__main__":
    main()
