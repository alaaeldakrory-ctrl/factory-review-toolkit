"""Apples-to-apples: an existing animation (keyed by script on real characters) against the same animation
re-solved by our solver.

Kept from the original:
- the scene, characters and frames;
- the hand contact points (each working hand goes where the original put it; the hand's orientation is the
  solver's);
- the gaze;
- walking and idle standing, as they are.

Our solver chooses:
- the body posture in the working frames;
- squatting rather than stooping for low work;
- a small step (up to ~20 cm) when the original stands in a bad spot (the feet ease over to it and back);
- keeping the body out of the scene's equipment (a moment-by-moment collision field: what stands still is an
  obstacle, what is being carried is not).

    python tools/resolve_compare.py C:/OmniverseData/Rack_Assembly.usd [--out=out/people/compare] [--step=2]
        [--workers=11] [--range=0,5832] [--no-step] [--no-collide]
    -> after.usda (the scene with the re-solved workers), summary.json (motion checks before / after, the steps)

Then score both with the same Ergo Scan settings (tools/compare_report.py).
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
SEAM = 12                    # frames blended from the original into the re-solved posture and back
STEP_EASE = 10               # frames the feet take to move over to a better spot (and back)
BUCKET_S = 2.0               # the collision field is rebuilt every this many seconds
SPOTS = [(0.0, -10.0), (0.0, 8.0), (0.0, -18.0), (8.0, 0.0), (-8.0, 0.0), (8.0, -10.0), (-8.0, -10.0), (0.0, 15.0),
         (0.0, 22.0), (8.0, 15.0), (-8.0, 15.0)]       # worker steps dx to its left, dz forward (cm)


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


def rootT(r):
    a = math.radians(r[2])
    T = np.eye(4)
    T[:3, :3] = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
    T[:3, 3] = (r[0], 0.0, r[1])
    return T


def scene_from_character(stage, skel_path, ch):
    """4x4 from the character's frt frame to the scene in frt units (Y-up cm)."""
    from pxr import UsdGeom
    k = (UsdGeom.GetStageMetersPerUnit(stage) or 0.01) * 100.0
    C = np.eye(4)
    C[:3, :3] = np.array([[k, 0, 0], [0, 0, k], [0, -k, 0]]) if UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.z \
        else np.diag([k, k, k])
    L2W = np.array(UsdGeom.Xformable(stage.GetPrimAtPath(skel_path)).ComputeLocalToWorldTransform(0), float).T
    return C @ L2W @ np.linalg.inv(ch.B)


def build_frames(ch, S, recs, fps, M_sc=None, times=None):
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
        yr = math.radians(root[2])
        Ry = np.array([[math.cos(yr), 0, math.sin(yr)], [0, 1, 0], [-math.sin(yr), 0, math.cos(yr)]])
        o = np.array([root[0], 0.0, root[1]])
        loc = lambda p: Ry.T @ (np.asarray(p) - o)                       # noqa: E731
        world = None if M_sc is None else M_sc @ rootT(root)
        targets = []
        for s, sx in (("l", 1.0), ("r", -1.0)):
            Rh = Ry.T @ r["R"][f"hand_{s}"]
            fingers = Rh @ np.array([sx, 0.0, 0.0])
            palm = Rh @ np.array([0.0, -1.0, 0.0])
            pt = loc(r["P"][f"hand_{s}"]) + fingers * S.hand_len * 0.40 + palm * 1.6
            if np.linalg.norm(pt - np.asarray(rest[s].point)) > WORK_CM:
                if os.environ.get("KEEP_ORIENT") == "1":
                    targets.append(HandTarget(s, tuple(pt), tuple(fingers), tuple(palm), "power", 1.0))
                else:
                    targets.append(HandTarget(s, tuple(pt), None, None, "power", 1.0))
        walking = v > WALK_CM_S
        if walking or not targets:
            st = Stance()
            st.free = True
            frames.append(Frame([], None, {}, root=root, stance=st, base=base, label="walk" if walking else "idle",
                                world=world))
            kinds.append("keep")
            continue
        feet = {}
        for s in "lr":
            d = Ry.T @ (r["P"][f"ball_{s}"] - r["P"][f"foot_{s}"])
            feet[s] = (tuple(loc(r["P"][f"foot_{s}"])), _yaw(d))
        st = Stance(left=feet["l"][0], right=feet["r"][0], yaw=0.0, yaw_l=feet["l"][1], yaw_r=feet["r"][1])
        Rhd = Ry.T @ r["R"]["head"]
        look = tuple(loc(r["P"]["head"]) + Rhd @ np.array([0.0, 0.0, 1.0]) * 80.0)
        frames.append(Frame(targets, look, {}, root=root, stance=st, label="work", world=world))
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


class Fields:
    """Collision fields of the scene, one per BUCKET_S seconds, built on demand and a few kept in memory."""

    def __init__(self, stage, region, exclude, fps, keep=3):
        from frt_core.people.collide import SceneGeometry
        self.geo = SceneGeometry(stage, region, exclude)
        self.fps, self.keep, self.cache = fps, keep, {}

    def key(self, t):
        b = BUCKET_S * self.fps
        return int(t // b) * b + b / 2

    def at(self, t):
        k = self.key(t)
        if k not in self.cache:
            if len(self.cache) >= self.keep:
                self.cache.pop(next(iter(self.cache)))
            self.cache[k] = self.geo.field_at(k)
        return self.cache[k]


_W = {}


def _init(asset, skel_path, scene, times, step, fps, region, exclude, collide, stepping):
    from pxr import Usd

    from frt_core.people.character import Character
    from frt_core.people.solver import PostureSolver
    st = Usd.Stage.Open(scene)
    ch = Character.load(asset)
    S = PostureSolver(ch.skeleton, ch.worker)
    recs = ch.read(st, st.GetPrimAtPath(skel_path), times)
    M_sc = scene_from_character(st, skel_path, ch)
    frames, kinds, roots, bases = build_frames(ch, S, recs, fps, M_sc)
    _W.update(ch=ch, S=S, frames=frames, step=step, times=times, stepping=stepping,
              fields=Fields(st, region, exclude, fps) if collide else None)


def _shift(frames, d, a, b):
    """Frames a..b with the feet moved by d (dx left, dz forward in each frame's root), eased in and out."""
    import copy

    from frt_core.people.motion_gen import ease
    out = []
    n = b - a
    for k, f in enumerate(frames[a:b]):
        u = min(1.0, ease((k + 1) / (STEP_EASE + 1)), ease((n - k) / (STEP_EASE + 1)))
        g = copy.copy(f)
        st = copy.copy(f.stance)
        off = np.array([d[0] * u, 0.0, d[1] * u])
        st.left = tuple(np.asarray(f.stance.left) + off)
        st.right = tuple(np.asarray(f.stance.right) + off)
        g.stance = st
        out.append(g)
    return out


def _solve_range(frames, a, b):
    from frt_core.people.bake import solve_frames
    S, fields = _W["S"], _W["fields"]
    sols = []
    # the collision field follows the moment (rebuilt every BUCKET_S): solve in pieces that share a field
    i = 0
    prev_sol = None
    while i < len(frames):
        t = _W["times"][a + i]
        j = i
        if fields is not None:
            k = fields.key(t)
            while j < len(frames) and fields.key(_W["times"][a + j]) == k:
                j += 1
            S.field = fields.at(t)
        else:
            j = len(frames)
        part = frames[i:j]
        sols += solve_frames(S, part, step=_W["step"], log=lambda *x: None, prev=prev_sol)   # continues the stretch
        prev_sol = sols[-1]
        i = j
    S.field = None
    return sols


def _badness(sols):
    """How bad a solved stretch is: trunk, collisions, reach (solver terms) per frame."""
    return float(np.mean([s.terms.get("trunk", 0) + s.terms.get("collide", 0) * 2.0 + s.terms.get("reach_l", 0)
                          + s.terms.get("reach_r", 0) + s.terms.get("knees", 0) for s in sols]))


def _solve_seg(seg):
    a, b = seg
    frames = _W["frames"]
    sols = _solve_range(frames[a:b], a, b)
    chosen, note = (0.0, 0.0), "fine"
    if _W["stepping"] and b - a >= 2 * STEP_EASE:
        import copy
        base_bad = _badness(sols)
        trunk = max(abs(s.angles.get("trunk_flexion", 0.0)) for s in sols)
        hit = max(s.terms.get("collide", 0.0) for s in sols)
        if trunk > 35.0 or hit > 0.5 or base_bad > 1.5:
            note = "tried"
            S, fields = _W["S"], _W["fields"]
            keys = [a + int(u * (b - a - 1)) for u in (0.35, 0.65)]
            # the solver's own cost at the key frames, warm-started from the solved posture (same as the original
            # spot is measured): a better spot has to beat it by 20 %
            def at(k, d):
                f = copy.copy(frames[k])
                st = copy.copy(f.stance)
                st.left = tuple(np.asarray(f.stance.left) + np.array([d[0], 0, d[1]]))
                st.right = tuple(np.asarray(f.stance.right) + np.array([d[0], 0, d[1]]))
                if fields is not None:
                    S.field = fields.at(_W["times"][k])
                sol = S.solve(f.targets, look=f.look, stance=st, prev=sols[k - a], world=f.world)
                return sol.cost + sol.terms.get("collide", 0.0)
            base_c = sum(at(k, (0.0, 0.0)) for k in keys)
            scored = sorted((sum(at(k, d) for k in keys), d) for d in SPOTS)
            S.field = None
            if scored and scored[0][0] < base_c * 0.8:
                d = scored[0][1]
                sols2 = _solve_range(_shift(frames, d, a, b), a, b)
                note = f"candidate {d}: {_badness(sols2):.2f} vs {base_bad:.2f}"
                if _badness(sols2) < base_bad * 0.9:
                    sols, chosen, note = sols2, d, "stepped"
    return seg, [s.pose for s in sols], chosen, note


def resolve_worker(scene, name, skel_path, asset, times, fps, step, nw, region, exclude, collide, stepping,
                   log=print):
    from pxr import Usd

    from frt_core.people.bake import blend
    from frt_core.people.character import Character
    from frt_core.people.motion_gen import ease
    from frt_core.people.solver import PostureSolver
    st = Usd.Stage.Open(scene)
    ch = Character.load(asset)
    S = PostureSolver(ch.skeleton, ch.worker)
    recs = ch.read(st, st.GetPrimAtPath(skel_path), times)
    M_sc = scene_from_character(st, skel_path, ch)
    frames, kinds, roots, bases = build_frames(ch, S, recs, fps, M_sc)
    segs = segments(kinds)
    log(f"{name}: {len(frames)} frames, {sum(k == 'solve' for k in kinds)} working frames in {len(segs)} segments")
    poses = list(bases)
    steps = {}
    t0 = time.time()
    with mp.Pool(nw, initializer=_init, initargs=(asset, skel_path, scene, times, step, fps, region, exclude,
                                                  collide, stepping)) as pool:
        notes = {}
        for k, (seg, ps, d, note) in enumerate(pool.imap_unordered(_solve_seg, segs)):
            a, b = seg
            for i, p in enumerate(ps):
                poses[a + i] = p
            if d != (0.0, 0.0):
                steps[f"{times[a]:g}-{times[b - 1]:g}"] = d
            key_ = note.split(":")[0].split(" (")[0]
            notes[key_] = notes.get(key_, 0) + 1
            if k % 10 == 0:
                log(f"  {k + 1}/{len(segs)} segments, {time.time() - t0:.0f} s")
    log(f"  stepping: {notes}")
    for a, b in segs:                                   # blend in from the original and back out
        m = min(SEAM, (b - a) // 2)
        for k in range(m):
            u = ease((k + 1) / (m + 1))
            poses[a + k] = blend(bases[a + k], poses[a + k], u)
            poses[b - 1 - k] = blend(bases[b - 1 - k], poses[b - 1 - k], u)
    return ch, frames, kinds, roots, bases, poses, steps


def check(ch, frames, poses, fields=None, times=None):
    from frt_core.people.check import check_frames
    from frt_core.people.solver import Solution
    sols = [Solution(p, 0.0) for p in poses]
    plain = []
    for f in frames:                                    # body limits, jumps and the scene's equipment
        plain.append(type(f)([], None, {}, root=f.root, stance=f.stance, label=f.label, world=f.world))
    fld = (lambda i: fields.at(times[i])) if fields is not None else None
    rep = check_frames(ch.skeleton, plain, sols, keep_out=None, field=fld)
    worst = {}
    for it in rep["issues"]:
        k = f"{it[1]}:{it[2]}"
        worst[k] = max(worst.get(k, 0.0), float(it[3]))
    flagged = sorted({it[0] for it in rep["issues"]})
    return {"counts": dict(rep["counts"]), "frames_flagged": len(flagged), "frames": len(frames),
            "worst": dict(sorted(worst.items(), key=lambda kv: -kv[1])[:12])}


def main():
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdSkel, Vt

    from frt_core.people.character import Character, _name
    from frt_core.people.collide import region_around
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opt = {a.split("=")[0][2:]: (a.split("=")[1] if "=" in a else True) for a in sys.argv[1:] if a.startswith("--")}
    scene = os.path.abspath(args[0])
    out = os.path.abspath(opt.get("out", os.path.join(ROOT, "out", "people", "compare")))
    os.makedirs(out, exist_ok=True)
    st = Usd.Stage.Open(scene)
    fps = st.GetTimeCodesPerSecond()
    t0, t1 = (float(v) for v in opt["range"].split(",")) if "range" in opt else (st.GetStartTimeCode(), st.GetEndTimeCode())
    times = list(np.arange(t0, t1 + 1e-6, 1.0))
    step, nw = int(opt.get("step", 2)), int(opt.get("workers", 11))
    collide, stepping = "no-collide" not in opt, "no-step" not in opt
    workers = find_workers(st)
    exclude = tuple(f"/World/{w[0]}" for w in workers)
    print("workers:", [(w[0], os.path.basename(w[2] or "?")) for w in workers], "collide", collide, "step", stepping)
    summary = {"scene": scene, "frames": [t0, t1], "fps": fps, "collide": collide, "stepping": stepping, "workers": {}}
    results = []
    for name, skel_path, asset in workers:
        ch = Character.load(asset)
        M_sc = scene_from_character(st, skel_path, ch)
        recs = ch.read(st, st.GetPrimAtPath(skel_path), times[::10])
        region = region_around([M_sc[:3, :3] @ r["P"]["pelvis"] + M_sc[:3, 3] for r in recs])
        ch, frames, kinds, roots, bases, poses, steps = resolve_worker(scene, name, skel_path, asset, times, fps,
                                                                        step, nw, region, exclude, collide, stepping)
        fields = Fields(st, region, exclude, fps) if collide else None
        before = check(ch, frames, bases, fields, times)
        after = check(ch, frames, poses, fields, times)
        summary["workers"][name] = {"working_frames": sum(k == "solve" for k in kinds), "before": before,
                                    "after": after, "steps": steps,
                                    "working": [float(t) for t, k in zip(times, kinds) if k == "solve"]}
        print(f"{name}: before {before['counts']} ({before['frames_flagged']} frames) -> after {after['counts']} "
              f"({after['frames_flagged']} frames); stepped in {len(steps)} stretches")
        results.append((name, skel_path, ch, poses, roots))
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
        anim = UsdSkel.Animation.Define(ast, f"/World/frt_after/{name}_Motion")
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
