"""Apples-to-apples: an existing animation (keyed by script on real characters) against the same animation
re-solved by our solver.

Same scene, same characters and frames. Same hand positions (each working hand goes exactly where the original
put it, palm and fingers the same way), same feet and gaze. Walking and idle standing are kept as they are. Only
the body posture in the working frames is chosen by our solver.

    python tools/resolve_compare.py C:/OmniverseData/Rack_Assembly.usd [--out=out/people/compare] [--step=2]
        [--workers=11] [--range=0,5832]
    -> before_check.json (motion checks on the original), after.usda (the scene with the re-solved workers instead),
       after_check.json, summary.json

Then score both with the same Ergo Scan settings (USD_Ergo / tools/ergo_scan.py).
"""
import json
import math
import multiprocessing as mp
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "exts", "frt.core"))
os.environ.setdefault("PXR_USDC_EMIT_DEPRECATION_WARNINGS", "0")
import numpy as np   # noqa: E402

WALK_CM_S = 35.0             # pelvis faster than this: walking (kept as it is)
WORK_CM = 18.0               # a hand this far from where it hangs at ease is working
SEAM = 6                     # frames blended from the original into the re-solved posture and back


def find_workers(stage):
    """[(name, skeleton prim path, character asset)] for every CC character in the scene."""
    from pxr import Usd, UsdSkel
    out = []
    for p in Usd.PrimRange(stage.GetPseudoRoot()):
        if not p.IsA(UsdSkel.Skeleton):
            continue
        asset = next((s.layer.realPath for s in p.GetPrimStack()[1:] if s.layer.realPath.lower().endswith(".usd")), None)
        top = p
        while top.GetParent() and str(top.GetParent().GetPath()) != "/World":
            top = top.GetParent()
        out.append((top.GetName(), str(p.GetPath()), asset))
    return out


def _yaw(v):
    return math.degrees(math.atan2(v[0], v[2]))


def build_frames(ch, S, recs, fps):
    """Frames for our solver from the read-back original. Returns (frames, kinds, roots, base poses)."""
    from frt_core.people.motion_gen import Frame, rest_targets
    from frt_core.people.solver import HandTarget, Stance
    rest, _ = rest_targets(S)
    pel = np.array([r["P"]["pelvis"] for r in recs])
    n = len(recs)
    roots, kinds, frames, bases = [], [], [], []
    for i, r in enumerate(recs):
        fwd = r["R"]["pelvis"] @ np.array([0.0, 0.0, 1.0])
        root = (float(pel[i][0]), float(pel[i][2]), _yaw(fwd))
        roots.append(root)
        a, b = max(0, i - 4), min(n - 1, i + 4)
        v = np.hypot(*(pel[b] - pel[a])[[0, 2]]) / max(1e-6, (b - a) / fps)
        base = ch.to_pose(r, root)
        bases.append(base)
        # in the root frame
        yr = math.radians(root[2])
        Ry = np.array([[math.cos(yr), 0, math.sin(yr)], [0, 1, 0], [-math.sin(yr), 0, math.cos(yr)]])
        o = np.array([root[0], 0.0, root[1]])
        loc = lambda p: Ry.T @ (np.asarray(p) - o)                       # noqa: E731
        targets = []
        for s, sx in (("l", 1.0), ("r", -1.0)):
            Rh = Ry.T @ r["R"][f"hand_{s}"]
            fingers = Rh @ np.array([sx, 0.0, 0.0])
            palm = Rh @ np.array([0.0, -1.0, 0.0])
            pt = loc(r["P"][f"hand_{s}"]) + fingers * S.hand_len * 0.40 + palm * 1.6
            if np.linalg.norm(pt - np.asarray(rest[s].point)) > WORK_CM:
                # the same contact point; the hand's orientation is left to the solver (the original's is often the
                # impossible part: copying it would copy the twisted arm). KEEP_ORIENT=1 copies it too.
                if os.environ.get("KEEP_ORIENT") == "1":
                    targets.append(HandTarget(s, tuple(pt), tuple(fingers), tuple(palm), "power", 1.0))
                else:
                    targets.append(HandTarget(s, tuple(pt), None, None, "power", 1.0))
        walking = v > WALK_CM_S
        if walking or not targets:
            st = Stance()
            st.free = True
            frames.append(Frame([], None, {}, root=root, stance=st, base=base, label="walk" if walking else "idle"))
            kinds.append("keep")
            continue
        feet = {}
        for s in "lr":
            d = Ry.T @ (r["P"][f"ball_{s}"] - r["P"][f"foot_{s}"])
            feet[s] = (tuple(loc(r["P"][f"foot_{s}"])), _yaw(d))
        st = Stance(left=feet["l"][0], right=feet["r"][0], yaw=0.0, yaw_l=feet["l"][1], yaw_r=feet["r"][1])
        Rhd = Ry.T @ r["R"]["head"]
        look = tuple(loc(r["P"]["head"]) + Rhd @ np.array([0.0, 0.0, 1.0]) * 80.0)
        frames.append(Frame(targets, look, {}, root=root, stance=st, label="work"))
        kinds.append("solve")
    return frames, kinds, roots, bases


def segments(kinds):
    out, i = [], 0
    while i < len(kinds):
        if kinds[i] != "solve":
            i += 1
            continue
        j = i
        while j + 1 < len(kinds) and kinds[j + 1] == "solve":
            j += 1
        out.append((i, j + 1))
        i = j + 1
    return out


_W = {}


def _init(asset, skel_path, scene, times, step, fps):
    from pxr import Usd

    from frt_core.people.character import Character
    from frt_core.people.solver import PostureSolver
    st = Usd.Stage.Open(scene)
    ch = Character.load(asset)
    S = PostureSolver(ch.skeleton, ch.worker)
    recs = ch.read(st, st.GetPrimAtPath(skel_path), times)
    frames, kinds, roots, bases = build_frames(ch, S, recs, fps)
    _W.update(ch=ch, S=S, frames=frames, step=step)


def _solve_seg(seg):
    from frt_core.people.bake import solve_frames
    a, b = seg
    sols = solve_frames(_W["S"], _W["frames"][a:b], step=_W["step"], log=lambda *x: None)
    return seg, [s.pose for s in sols]


def resolve_worker(scene, name, skel_path, asset, times, fps, step, nw, log=print):
    from pxr import Usd

    from frt_core.people.bake import blend
    from frt_core.people.character import Character
    from frt_core.people.motion_gen import ease
    from frt_core.people.solver import PostureSolver
    st = Usd.Stage.Open(scene)
    ch = Character.load(asset)
    S = PostureSolver(ch.skeleton, ch.worker)
    recs = ch.read(st, st.GetPrimAtPath(skel_path), times)
    frames, kinds, roots, bases = build_frames(ch, S, recs, fps)
    segs = segments(kinds)
    log(f"{name}: {len(frames)} frames, {sum(k == 'solve' for k in kinds)} working frames in {len(segs)} segments")
    poses = list(bases)
    t0 = time.time()
    with mp.Pool(nw, initializer=_init, initargs=(asset, skel_path, scene, times, step, fps)) as pool:
        for k, (seg, ps) in enumerate(pool.imap_unordered(_solve_seg, segs)):
            a, b = seg
            for i, p in enumerate(ps):
                poses[a + i] = p
            if k % 10 == 0:
                log(f"  {k + 1}/{len(segs)} segments, {time.time() - t0:.0f} s")
    for a, b in segs:                                   # blend in from the original and back out
        m = min(SEAM, (b - a) // 2)
        for k in range(m):
            u = ease((k + 1) / (m + 1))
            poses[a + k] = blend(bases[a + k], poses[a + k], u)
            poses[b - 1 - k] = blend(bases[b - 1 - k], poses[b - 1 - k], u)
    return ch, frames, kinds, roots, bases, poses


def check(ch, frames, poses):
    from frt_core.people.check import check_frames
    from frt_core.people.solver import Solution
    sols = [Solution(p, 0.0) for p in poses]
    plain = []
    for f in frames:                                    # only body limits / jumps: no targets, no fixtures known
        g = type(f)([], None, {}, root=f.root, stance=f.stance, label=f.label)
        plain.append(g)
    rep = check_frames(ch.skeleton, plain, sols, keep_out=None)
    worst = {}
    for it in rep["issues"]:
        k = f"{it[1]}:{it[2]}"
        worst[k] = max(worst.get(k, 0.0), float(it[3]))
    flagged = sorted({it[0] for it in rep["issues"]})
    return {"counts": dict(rep["counts"]), "frames_flagged": len(flagged), "frames": len(frames),
            "worst": dict(sorted(worst.items(), key=lambda kv: -kv[1])[:12])}


def main():
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdSkel, Vt  # noqa: F401

    from frt_core.people.character import _name
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opt = {a.split("=")[0][2:]: a.split("=")[1] for a in sys.argv[1:] if a.startswith("--") and "=" in a}
    scene = os.path.abspath(args[0])
    out = os.path.abspath(opt.get("out", os.path.join(ROOT, "out", "people", "compare")))
    os.makedirs(out, exist_ok=True)
    st = Usd.Stage.Open(scene)
    fps = st.GetTimeCodesPerSecond()
    t0, t1 = (float(v) for v in opt["range"].split(",")) if "range" in opt else (st.GetStartTimeCode(), st.GetEndTimeCode())
    times = list(np.arange(t0, t1 + 1e-6, 1.0))
    step, nw = int(opt.get("step", 2)), int(opt.get("workers", 11))
    workers = find_workers(st)
    print("workers:", [(w[0], os.path.basename(w[2] or "?")) for w in workers])
    summary = {"scene": scene, "frames": [t0, t1], "fps": fps, "workers": {}}
    results = []
    for name, skel_path, asset in workers:
        ch, frames, kinds, roots, bases, poses = resolve_worker(scene, name, skel_path, asset, times, fps, step, nw)
        before = check(ch, frames, bases)
        after = check(ch, frames, poses)
        summary["workers"][name] = {"working_frames": sum(k == "solve" for k in kinds), "before": before,
                                    "after": after, "working": [float(t) for t, k in zip(times, kinds) if k == "solve"]}
        print(f"{name}: before {before['counts']} ({before['frames_flagged']} frames) -> after {after['counts']} "
              f"({after['frames_flagged']} frames)")
        results.append((name, skel_path, ch, poses, roots))
    # the scene with the re-solved workers: same file underneath, the new motion on the same characters
    path = os.path.join(out, "after.usda")
    lay = Sdf.Layer.CreateNew(path) if not os.path.exists(path) else Sdf.Layer.FindOrOpen(path)
    lay.Clear()
    lay.subLayerPaths.append(scene.replace("\\", "/"))
    ast = Usd.Stage.Open(lay)
    ast.SetEditTarget(lay)
    # the same timing and units as the original (a root layer without them would play the sublayer at 24 fps)
    ast.SetTimeCodesPerSecond(st.GetTimeCodesPerSecond())
    ast.SetFramesPerSecond(st.GetFramesPerSecond())
    ast.SetStartTimeCode(st.GetStartTimeCode())
    ast.SetEndTimeCode(st.GetEndTimeCode())
    UsdGeom.SetStageUpAxis(ast, UsdGeom.GetStageUpAxis(st))
    UsdGeom.SetStageMetersPerUnit(ast, UsdGeom.GetStageMetersPerUnit(st))
    for name, skel_path, ch, poses, roots in results:
        anim_path = f"/World/frt_after/{name}_Motion"
        anim = UsdSkel.Animation.Define(ast, anim_path)
        sk = ast.GetPrimAtPath(skel_path)
        q_joints = [str(j) for j in UsdSkel.Skeleton(sk).GetJointsAttr().Get()]
        if [_name(j) for j in q_joints] != ch.names:
            raise RuntimeError(f"{name}: joint order differs from the asset")
        anim.CreateJointsAttr(Vt.TokenArray(q_joints))
        tr_a, rot_a = anim.CreateTranslationsAttr(), anim.CreateRotationsAttr()
        anim.CreateScalesAttr(Vt.Vec3hArray([Gf.Vec3h(1, 1, 1)] * len(q_joints)))
        for t, p, rt in zip(times, poses, roots):
            tr, rot = ch.local_arrays(p, root=rt)             # poses are root-relative; the root is per frame
            tr_a.Set(tr, t)
            rot_a.Set(rot, t)
        UsdSkel.BindingAPI.Apply(ast.OverridePrim(skel_path)).CreateAnimationSourceRel().SetTargets([anim.GetPath()])
    lay.Save()
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print("->", path)


if __name__ == "__main__":
    main()
