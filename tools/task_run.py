"""Run a task file: compile the steps, solve every frame, check, score (RULA) and bake one scene.

    python tools/task_run.py docs/people/practice/reference_task.toml [--out=out/people/task/<name>] [--workers=11]
        [--worker=W05F]   (the same task on another body: an frt body id or a character .usd)
        [--final]         (every frame solved; the default preview solves key poses - every KEY-th frame and every
                           phase's first and last - and blends between them, several times faster)
        [--motion=andy|best]   (recorded bodies under the work / per step the better of with and without)
        [--web=<dir>]     (also the files the website's viewer plays: tools/web_export.py)
        [--clips=<dir>]   (each step as a motion-library clip: GLB + BVH, tools/clip_export.py)
    -> <out>/task.usda (scene + worker + parts), result.json (per step: frames, checks, RULA), log

Steps that did not change since the last run in the same --out (same frames, scene, body, code, quality) are
reused from <out>/step_cache.pkl; only the hand-over into them is re-solved.

The Kit window (Task Builder) runs this in the background and opens the result.
"""
import hashlib
import json
import multiprocessing as mp
import os
import pickle
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "exts", "frt.core"))
os.environ.setdefault("PXR_USDC_EMIT_DEPRECATION_WARNINGS", "0")

_T = None
EFFORT = float(os.environ.get("FRT_EFFORT", 0.5))                # preview: half the search iterations per key pose
KEY = int(os.environ.get("FRT_KEY", 6))                         # preview: a solved pose every KEY frames (0.25 s)
_TS = {}                                                             # (--motion=best: both versions)


def _load(path, worker=None):
    from frt_core.people.tasks import Task
    T = Task.load(path)
    if worker:
        T.data["task"]["worker"] = worker
    return T


def _init(path, worker=None, motion=None, blob=None):
    """A worker's tasks: the main process's compiled ones (blob: pickled {version: Task}), else compiled here."""
    global _T
    if blob is not None:
        _TS.update(pickle.loads(blob))
        _T = list(_TS.values())[0]
        return
    keys = ("solver", "andy") if motion == "best" else (None,)
    for k in keys:
        T = _load(path, worker).compile(log=lambda *a: None)
        _motion(T, k or motion, log=lambda *a: None)                 # (best: the solver one stays plain)
        _TS[k or _key(T, motion)] = T
    _T = list(_TS.values())[0]


def _key(T, motion):
    """Which version a run solves: "andy" (recorded bodies, from --motion or the task file) or "solver"."""
    return "andy" if (motion or T.data["task"].get("motion")) == "andy" else "solver"


def _motion(T, motion, log=print):
    """--motion=andy (or motion = "andy" in [task]): real recorded bodies under the work (andy_base)."""
    if (motion or T.data["task"].get("motion")) == "andy":
        from frt_core.people.andy_base import apply
        apply(T, log=log)


def _solve(job):
    from frt_core.people.bake import solve_frames
    k, a, b, step = job
    T = _TS[k]
    T.S.effort = EFFORT if step > 1 else 1.0
    return k, a, solve_frames(T.S, T.frames[a:b + 1], log=lambda *a: None, step=step)


def _apart(p, q):
    """The biggest joint difference between two solved poses, deg."""
    from frt_core.people.check import rot_angle
    from frt_core.people.skeleton import NAMES
    return max(rot_angle(p.pose.local(j).T @ q.pose.local(j)) for j in NAMES if j != "root")


def _join(job):
    """Carry on from the frame before a piece until the motion meets the piece's own solution (a piece warm-started
    from rest can have the elbow on the other side: handing over after a fixed number of frames was a snap)."""
    from frt_core.people.bake import solve_frames
    k, a, prev, own, step = job
    T = _TS[k]
    n = len(T.frames)
    out, i = [], a
    while i < n and i - a < len(own):
        hi = min(n, i + max(10, 2 * step), a + len(own))
        fix = solve_frames(T.S, T.frames[i:hi], log=lambda *a: None, prev=prev, step=step)
        out += fix
        prev, i = fix[-1], hi
        if _apart(fix[-1], own[hi - 1 - a]) < 8.0:
            break
    return a, out


_CODE = None


def _code_hash():
    """The people code (solver, tasks, kinematics...): a change there makes every cached step stale."""
    global _CODE
    if _CODE is None:
        h = hashlib.sha1()
        d = os.path.join(ROOT, "exts", "frt.core", "frt_core", "people")
        for f in sorted(os.listdir(d)):
            if f.endswith(".py"):
                h.update(open(os.path.join(d, f), "rb").read())
        _CODE = h.hexdigest()
    return _CODE


def step_ranges(T):
    """Each step's frames, up to the next step's first (the first from frame 0, the last to the end)."""
    st = [0] + [s["start"] for s in T.steps[1:]] + [len(T.frames)]
    return [(a, b - 1) for a, b in zip(st, st[1:])]


def step_hash(T, s, k, step):
    """Everything a step's solve depends on: its frames (targets, gaze, stance, walk, base pose), the scene and
    body (the task file without the steps), the code, the version (k) and the key spacing."""
    data = {x: v for x, v in T.data.items() if x != "step"}
    def pose(b):                                                    # (joint order in a pose varies run to run)
        return None if b is None else (b.pelvis.tobytes(), [(j, b.rot[j].tobytes()) for j in sorted(b.rot)])
    fr = [(f.targets, f.look, f.root, f.stance, f.swing, f.static, f.label, pose(f.base))
          for f in T.frames[s[0]:s[1] + 1]]
    h = hashlib.sha1(pickle.dumps(fr, protocol=4))
    h.update(json.dumps(data, sort_keys=True, default=str).encode())
    h.update(f"{_code_hash()} {k} {step} {EFFORT if step > 1 else 1.0}".encode())
    return h.hexdigest()


def _worse(new, sol, k=1.25, add=2.0):
    """A rebuilt pose clearly costlier than the solver's own for that frame (a wound-up forearm, a wrist past its
    limit): the solver's own is kept."""
    return sum(new.terms.values()) > k * sum(sol.terms.values()) + add


def join_fast(T, a, prev, own, blend_n=12):
    """A preview join, no search: the piece's own trunk (eased in from the frame before over blend_n frames), its
    arms rebuilt carrying on from the frame before - the elbow branch flows on across the join. Stops where the
    rebuilt frame is the piece's own again (~5 ms a frame). Returns (frames, settled): not settled = it never met
    the piece's own pose, or carrying on would wind the forearm up (handing over there is a snap)."""
    out = []
    for j, sol in enumerate(own):
        f = T.frames[a + j]
        if sol.v is None or prev.v is None or len(sol.v) != len(prev.v) or f.static:
            out.append(sol)
            prev = sol
            continue
        u = min(1.0, (j + 1) / (blend_n + 1))
        new = T.S.at(prev.v * (1 - u) + sol.v * u, f.targets, look=f.look, stance=f.stance, prev=prev,
                     swing=f.swing, base=f.base, world=getattr(f, "world", None))
        if u >= 1.0 and _apart(new, sol) < 1.0:                      # back on the piece's own branch: keep the rest
            return out + list(own[j:]), True
        if u >= 1.0 and _worse(new, sol):                            # the carried-on arm winds up
            return out + list(own[j:]), False
        out.append(new)
        prev = new
    return out, False


def solve_set(pool, k, T, say, size, step, cache=None, p0=0.05, p1=0.75):
    """Every frame of one version (k: a key of _TS) solved in pieces - steps found in cache reused - then the joins
    between them. cache: {step hash: solutions}, updated in place."""
    n = len(T.frames)
    sols = [None] * n
    segs, hashes = [], []                                            # segs: (a, b, reused)
    for st in step_ranges(T):
        a, b = st
        h = step_hash(T, st, k, step)
        hashes.append((a, b, h))
        if cache is not None and len(cache.get(h, ())) == b - a + 1:
            sols[a:b + 1] = cache[h]
            segs.append((a, b, True))
            continue
        while b - a > size * 1.5:
            segs.append((a, a + size - 1, False))
            a += size
        segs.append((a, b, False))
    todo = [(k, a, b, step) for a, b, r in segs if not r]
    total = sum(b - a + 1 for _, a, b, _ in todo) or 1
    nre = sum(1 for *_, r in segs if r)
    if nre:
        say(f"{nre} of {len(T.steps)} steps unchanged: reused", progress=p0)
    done = 0
    for _, a, s in pool.imap_unordered(_solve, todo):
        sols[a:a + len(s)] = s
        done += len(s)
        say(f"solving {done}/{total} frames" + (f" ({k})" if k != "solver" else ""),
            progress=p0 + (p1 - p0) * done / total)
    # the joins between pieces, each carried on from the frame before it (all at once: they are independent);
    # two reused steps side by side were joined when they were cached
    say("joining the pieces", progress=p1)
    idx = [i for i in range(1, len(segs)) if not (segs[i][2] and segs[i - 1][2])]
    if step > 1:                                                     # preview: cheap joins, in order, here;
        hard = []                                                    # the ones that don't settle: solved (below)
        for i in idx:
            a, b, _ = segs[i]
            fix, ok = join_fast(T, a, sols[a - 1], sols[a:b + 1])
            if ok:
                sols[a:b + 1] = fix
            else:
                hard.append(i)
        idx = hard
    own = {i: sols[segs[i][0]:segs[i][1] + 1] for i in idx}
    at = {segs[i][0]: i for i in idx}
    ran = {}
    cap = None if step == 1 else 48                                 # (a preview's hard joins: up to 2 s each)
    for a, s in pool.imap_unordered(_join, [(k, segs[i][0], sols[segs[i][0] - 1], own[i][:cap], step) for i in idx]):
        sols[a:a + len(s)] = s
        ran[at[a]] = cap is None and len(s) == len(own[at[a]])
    # a join that ran through its whole piece changed that piece's last frame: the next join started from the
    # old one (a wrong-branch arm carried on) - redo it from the new end, in order
    for i in range(2, len(segs)):
        if ran.get(i - 1):
            a2, b2, _ = segs[i]
            o = own.get(i) or sols[a2:b2 + 1]
            _, s = pool.apply(_join, ((k, a2, sols[a2 - 1], o, step),))
            sols[a2:a2 + len(s)] = s
            ran[i] = len(s) == len(o)
    if cache is not None:
        for a, b, h in hashes:
            cache[h] = sols[a:b + 1]
    return sols


def depop(sols, over=60.0, k=6):
    """Last resort for a snap (a body joint turning more than `over` deg in one frame - the elbow jumping to its
    other side where continuity broke): the k frames each side become an eased blend between the poses around it,
    so the arm swings across as a person's would (the hand can leave the work for that half second)."""
    from frt_core.people.bake import blend
    from frt_core.people.check import _family, rot_angle
    from frt_core.people.motion_gen import ease
    n = len(sols)
    joints = [j for j in sols[0].pose.rot if j != "root" and _family(j) not in ("finger", "thumb", "thumb_01")]

    out, cnt, i = list(sols), 0, 1

    def jump(i):
        p, q = out[i].pose, out[i - 1].pose
        return max(rot_angle(q.local(j).T @ p.local(j)) for j in joints)
    while i < n:
        if out[i] is out[i - 1] or jump(i) <= over:
            i += 1
            continue
        a, b = max(0, i - 1 - k), min(n - 1, i + k)
        A, B = out[a], out[b]
        for j in range(a + 1, b):
            u = ease((j - a) / (b - a))
            out[j] = type(A)(blend(A.pose, B.pose, u), A.cost, A.terms, A.angles, A.v)
        cnt += 1
        i = b + 1
    return out, cnt


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


def best_of_both(T, TA, pool, say, usd, size, step, cache):
    """--motion=best: the task solved twice (solver only / recorded bodies), each step kept from the version with
    fewer and smaller check issues, the hand-over into a switched step re-solved from the kept frame before it."""
    from frt_core.people.bake import _rootT
    from frt_core.people.check import check_frames
    S0 = solve_set(pool, "solver", T, say, size, step, cache, 0.05, 0.40)
    SA = solve_set(pool, "andy", TA, say, size, step, cache, 0.40, 0.72)
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
            _, fix = pool.apply(_join, ((keys[j], a, sols[a - 1], own, step),))
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
    t0 = time.time()

    def say(msg, **kw):
        print(f"[{time.time() - t0:5.0f} s] {msg}", flush=True)
        with open(status + ".tmp", "w") as f:
            json.dump({"msg": msg, "t": time.time(), **kw}, f)
        os.replace(status + ".tmp", status)                         # the window never reads half a file
    scene_only = "--scene-only" in sys.argv
    nw = int(opt.get("workers", min(11, os.cpu_count() or 4)))
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
    size = max(24, min(160, -(-n // nw)))                            # short tasks: small pieces, every core busy
    step = 1 if "--final" in sys.argv else KEY
    cpath = os.path.join(out, "step_cache.pkl")
    try:
        cache = pickle.load(open(cpath, "rb"))
    except Exception:                                                # none yet / unreadable: solve everything
        cache = {}
    pick = None
    tasks = {_key(T, opt.get("motion")): T}
    if opt.get("motion") == "best":
        TA = _load(path, worker).compile(log=lambda *a: None)
        _motion(TA, "andy")
        tasks = {"solver": T, "andy": TA}
    # the workers get the compiled task (pickled, ~0.1 s) instead of each compiling it again
    with mp.Pool(nw, initializer=_init, initargs=(path, worker, opt.get("motion"), pickle.dumps(tasks, protocol=4))) as pool:
        if opt.get("motion") == "best":
            sols, pick = best_of_both(T, TA, pool, say, os.path.join(out, "task.usda"), size, step, cache)
        else:
            sols = solve_set(pool, _key(T, opt.get("motion")), T, say, size, step, cache)
    if len(cache) > 300:                                             # (keep the file small: this run's steps only)
        cur = {step_hash(T, st, kk, step) for st in step_ranges(T) for kk in ("solver", "andy")}
        cache = {h: v for h, v in cache.items() if h in cur}
    pickle.dump(cache, open(cpath + ".tmp", "wb"), protocol=4)
    os.replace(cpath + ".tmp", cpath)
    if os.environ.get("FRT_DUMP_SOLS"):                             # (debugging: the solved poses, before smoothing)
        pickle.dump({"sols": sols, "rngs": rngs}, open(os.environ["FRT_DUMP_SOLS"], "wb"))
    sols, npop = depop(sols)
    if npop:
        say(f"{npop} snaps eased", progress=0.81)
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
    if os.environ.get("FRT_DUMP_POST"):                             # (auditing: the task and the final poses)
        pickle.dump({"T": T, "sols": sols, "scores": scores}, open(os.environ["FRT_DUMP_POST"], "wb"), protocol=4)
    res = {"task": T.name, "file": path, "usd": usd, "frames": n, "fps": 24, "steps": steps,
           "counts": rep["counts"], "notes": T.notes, "motion": pick, "quality": "final" if step == 1 else "preview", "solve_s": round(time.time() - t0, 1)}
    json.dump(res, open(os.path.join(out, "result.json"), "w"), indent=1)
    if opt.get("web"):                                               # the files the web viewer plays
        say("writing the web files", progress=0.98)
        from web_export import export
        export(T, sols, res, usd, os.path.abspath(opt["web"]), scores)
    if opt.get("clips"):                                             # motion-library clips (tools/clip_export.py)
        say("writing the library clips", progress=0.99)
        from clip_export import export_clips
        export_clips(T, sols, res, scores, os.path.abspath(opt["clips"]), name,
                     T.name.split(":")[0] if ":" in T.name else "")
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
