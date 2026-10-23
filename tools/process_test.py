"""A short process in one continuous take, rule-based vs Kimodo-based:
walk to a bin -> bend and pick a part from low -> carry it to the bench -> place it -> ratchet it down.

A (rules): our generators chained along the route (clip walk, carry pick / place, ratchet), solved and checked.
B (Kimodo): Kimodo gets A's route (root waypoints while walking), A's hands (pick, carry, place, ratchet) and
planted feet at the bin and the bench as constraints, generates the whole take with no text; the result goes
through the final pass (Kimodo's body, our hands / tools exactly, keep-outs) and the same checks.

    python tools/process_test.py [W50M] -> out/people/process_test/compare.usd + shots.json + check.json
"""
import json
import math
import os
import pickle
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "exts", "frt.core"))
sys.path.insert(0, os.path.join(HERE, "kimodo"))
import numpy as np   # noqa: E402

OUT = os.path.join(HERE, "..", "out", "people", "process_test")
BIN_SPOT = (0.0, 0.0, 0.0)                    # world (x, z, yaw) where the worker stands to pick
BENCH_SPOT = (230.0, 140.0, 90.0)             # ... to place and fasten (facing +X)
START = (-40.0, -260.0, 0.0)
PART = (24.0, 12.0, 18.0)
SPEC = "box:24x12x18:cast_aluminium"


def rootT(r):
    a = math.radians(r[2])
    T = np.eye(4)
    T[:3, :3] = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
    T[:3, 3] = (r[0], 0.0, r[1])
    return T


def build_rules(w, sk, S):
    from frt_core.people import jobs as J
    from frt_core.people import motion_carry as C
    from frt_core.people.gait import walk
    from frt_core.people.motion_more import idle
    TOP = J.TOP
    pick_at = np.array([0.0, 34.0, 46.0])                  # part centre in the bin (a low bin on a stand)
    place_at = np.array([-4.0, TOP + 6.0, 48.0])
    frames = []
    seg = []

    def add(fr, root, label):
        for f in fr:
            if f.root is None:
                f.root = root
            f.label = f.label or label
        frames.extend(fr)
        seg.append((label, len(frames)))
    add(idle(0.5), START, "stand")
    add(walk(w, sk, [START[:2], (BIN_SPOT[0], BIN_SPOT[1] - 40.0), BIN_SPOT[:2]], START[2], BIN_SPOT[2]), START, "walk to bin")
    f1, hold, Tc = C.carry_pick(S, pick_at, PART, "part", SPEC, True, "r")
    add(f1, BIN_SPOT, "pick")
    wk = walk(w, sk, [BIN_SPOT[:2], (120.0, 60.0), (BENCH_SPOT[0] - 40.0, BENCH_SPOT[1]), BENCH_SPOT[:2]],
              BIN_SPOT[2], BENCH_SPOT[2], carry=hold)
    for f in wk:
        f.tools["part:part"] = (SPEC, Tc)
    add(wk, BIN_SPOT, "carry")
    add(C.carry_place(S, place_at, PART, "part", SPEC, True, "r"), BENCH_SPOT, "place")
    add(J._socket_job(S, sk, "ratchet_38", (place_at[0], place_at[1] + PART[1] / 2, place_at[2]), 3, 45.0, 0.8),
        BENCH_SPOT, "ratchet")
    add(idle(0.6), BENCH_SPOT, "stand")
    for f in frames:
        f.root = f.root or BENCH_SPOT
    smooth_seams(S, frames, [b for _, b in seg[:-1]])
    last = None                                            # standing after work: the eyes stay on the work
    for f in frames:
        walking = f.stance is not None and getattr(f.stance, "free", False)
        if f.look is not None:
            last = f.look
        elif not walking and last is not None:
            f.look = last
        if walking:
            last = None
    return frames, seg


def smooth_seams(S, frames, bounds, n=8):
    """Hand-overs between steps: each hand's target (and grip) blends from where the previous step left it into
    the next step over n frames, instead of switching (a hand with no target counts as resting at the side)."""
    from frt_core.people.motion_gen import ease, lerp_target, rest_targets
    from frt_core.people.kinematics import fk
    from frt_core.people.solver import HandTarget
    rest, _ = rest_targets(S)

    def hand_of(pose, side):
        R_, P_ = fk(S.sk, pose)
        sx = 1.0 if side == "l" else -1.0
        f_ = R_[f"hand_{side}"] @ np.array([sx, 0.0, 0.0])
        pm = R_[f"hand_{side}"] @ np.array([0.0, -1.0, 0.0])
        return HandTarget(side, tuple(P_[f"hand_{side}"] + f_ * S.hand_len * 0.4 + pm * 1.6), tuple(f_), tuple(pm),
                          "relaxed", 0.3)
    for b in bounds:
        if b <= 0 or b >= len(frames):
            continue
        prev = {t.side: t for t in frames[b - 1].targets}
        if frames[b - 1].base is not None:                  # a walking frame: its hands are where the mocap put them
            for side in ("l", "r"):
                if side not in prev:
                    prev[side] = hand_of(frames[b - 1].base, side)
        for k in range(min(n, len(frames) - b)):
            u = ease((k + 1) / (n + 1))
            f = frames[b + k]
            cur = {t.side: t for t in f.targets}
            out = []
            for side in ("l", "r"):
                a_ = prev.get(side)
                c_ = cur.get(side)
                if a_ is None and c_ is None:
                    continue
                a_ = a_ or rest[side]
                c_ = c_ or rest[side]
                g = ("blend", a_.grip, c_.grip, u)
                out.append(lerp_target(a_, c_, u, grip=g))
            f.targets = out


def keep_out_world(frames):
    """Per-frame keep-out rules: the bench front while at the bench (worker frame)."""
    from frt_core.people import jobs as J
    return [(J.HIPS, J.BENCH_FRONT - 10.0)]


def main():
    from frt_core.people.bake import solve_frames
    from frt_core.people.check import check_frames
    from frt_core.people.kinematics import fk
    from frt_core.people.mocap import retarget
    from frt_core.people.motion_gen import FPS
    from frt_core.people.skeleton import NAMES, Skeleton
    from frt_core.people.solver import PostureSolver, Stance
    from frt_core.people.workers import WORKERS
    from kimodo_job import KIMODO_HIP_M, KIMODO_PY, soma_calibration, soma_to_canonical
    from kimodo_names import NAMES77
    wid = sys.argv[1] if len(sys.argv) > 1 else "W50M"
    w = WORKERS[wid]
    sk = Skeleton.for_worker(w)
    S = PostureSolver(sk, w)
    os.makedirs(OUT, exist_ok=True)
    frames, seg = build_rules(w, sk, S)
    sols = solve_frames(S, frames, log=lambda *a: None)
    rep_a = check_frames(sk, frames, sols, "rules")
    print("rules:", len(frames), "frames", seg, rep_a["counts"])
    # ---- Kimodo constraints from A (world space, relative to the start)
    s = KIMODO_HIP_M / float(sk.world["pelvis"][1])
    o = np.array([START[0], 0.0, START[1]])
    walking = [f.stance is not None and getattr(f.stance, "free", False) or (f.stance is not None and f.base is not None)
               for f in frames]
    idx, P, R, ur, ul, feet = [], [], [], [], [], []
    for i in range(0, len(frames), 4):
        f = frames[i]
        T = rootT(f.root)
        Rw, Pw = fk(sk, sols[i].pose)
        P.append([T[:3, :3] @ Pw[n] + T[:3, 3] - o for n in NAMES])
        R.append([T[:3, :3] @ Rw[n] for n in NAMES])
        tg = {t.side: t for t in f.targets}
        ur.append("r" in tg and tg["r"].weight >= 0.5)
        ul.append("l" in tg and tg["l"].weight >= 0.5)
        feet.append(not walking[i])
        idx.append(i)
    r2 = [(i / FPS, (np.array([frames[i].root[0], frames[i].root[1]]) - o[[0, 2]])) for i in range(0, len(frames), 8)]
    src = os.path.join(OUT, "keys.npz")
    np.savez(src, seconds=len(frames) / FPS, fps_in=FPS, frames=np.array(idx), t=np.array(idx) / FPS,
             names=np.array(list(NAMES)), P=np.array(P), R=np.array(R), use_r=np.array(ur), use_l=np.array(ul),
             use_feet=np.array(feet), use_full=np.zeros(len(idx), bool), scale=s, steps=100, seed=0,
             r2_t=np.array([t for t, _ in r2]), r2_xz=np.array([p for _, p in r2]))
    gen = os.path.join(OUT, "kimodo.npz")
    r = subprocess.run([KIMODO_PY, os.path.join(HERE, "kimodo", "kimodo_constrained.py"), src, gen], capture_output=True,
                       text=True, encoding="utf-8", errors="replace", env=dict(os.environ, HF_HOME=r"C:/OmniverseData/kimodo/hf"))
    print(r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr[-1500:])
    g = np.load(gen)
    pj, kfps = g["posed_joints"], float(g["fps"])
    k_ = np.array([1.0, 4.0, 6.0, 4.0, 1.0]) / 16.0          # light temporal smoothing (hands are pinned anyway)
    pad = np.concatenate([pj[:1].repeat(2, 0), pj, pj[-1:].repeat(2, 0)])
    pj = sum(k_[i] * pad[i:i + len(pj)] for i in range(5))
    cal = soma_calibration(sk, s)
    new = []
    for i, f in enumerate(frames):
        x = min(i / FPS * kfps, len(pj) - 1)
        a = int(np.floor(x))
        b = min(a + 1, len(pj) - 1)
        Jw = soma_to_canonical(pj[a] * (1 - (x - a)) + pj[b] * (x - a), NAMES77, s)
        Ti = np.linalg.inv(rootT(f.root))                       # into this frame's worker frame (A's root)
        Jl = {k: Ti[:3, :3] @ (v + o) + Ti[:3, 3] for k, v in Jw.items()}
        pose = retarget(sk, Jl, cal)
        pose.pelvis = Jl["pelvis"] - sk.world["pelvis"]
        st = f.stance
        if st is None or getattr(st, "free", False):
            st2 = Stance()
            st2.free = True
            st = st2 if (st is not None) else None              # walking: Kimodo's legs; at work: planted stance
        new.append(type(f)(f.targets, f.look, dict(f.tools), root=f.root, stance=st, base=pose, label=f.label))
    from kimodo_job import pin_hands
    pin_hands(sk, new, ramp=10)                               # pinned while working, Kimodo's arms in between
    S2 = PostureSolver(sk, w)
    S2.base_correct = True
    sols_b = solve_frames(S2, new, log=lambda *a: None)
    rep_b = check_frames(sk, new, sols_b, "kimodo")
    print("kimodo:", rep_b["counts"])
    pickle.dump({"rules": (frames, sols), "kimodo": (new, sols_b), "worker": wid, "seg": seg},
                open(os.path.join(OUT, "takes.pkl"), "wb"))
    json.dump({"rules": rep_a["counts"], "kimodo": rep_b["counts"], "frames": len(frames), "segments": seg},
              open(os.path.join(OUT, "check.json"), "w"), indent=1)


if __name__ == "__main__" and "--scene" not in sys.argv:
    main()


def scene():
    """Both takes side by side (rules | Kimodo), each with its own bin stand and bench, and a wide shot."""
    from pxr import Gf, UsdGeom
    from frt_core import units as U
    from frt_core.catalog.details import sign
    from frt_core.geom import Shapes
    from frt_core.materials import MaterialLibrary
    from frt_core.people import jobs as J
    from frt_core.people.bake import bake
    from frt_core.people.workers import WORKERS
    from frt_core.studio import dress
    from frt_core.usdio import safe_stage
    d = pickle.load(open(os.path.join(OUT, "takes.pkl"), "rb"))
    w = WORKERS[d["worker"]]
    out = os.path.join(OUT, "compare.usd")
    with safe_stage(out) as st:
        U.setup_stage(st, fps=24.0)
        st.SetDefaultPrim(UsdGeom.Xform.Define(st, "/World").GetPrim())
        lib = MaterialLibrary(st, "/World/Looks")
        UsdGeom.Xform.Define(st, "/World/Asset")
        UsdGeom.Xform.Define(st, "/World/Asset/Parts")
        for k, (tag, dx) in enumerate((("rules", 0.0), ("kimodo", 430.0))):
            frames, sols = d[tag]
            base = f"/World/Asset/{tag}"
            UsdGeom.Xform.Define(st, base)
            s = Shapes()
            for spot, fixture in ((BIN_SPOT, "bin"), (BENCH_SPOT, "bench")):
                T = rootT((spot[0] + dx, spot[1], spot[2]))
                loc = Shapes()
                if fixture == "bin":
                    loc.rbox((0.0, 13.0, 46.0), (44.0, 26.0, 34.0), 0.6, "powder_dark", seg=1)       # stand
                    loc.rbox((0.0, 30.0, 46.0), (40.0, 8.0, 30.0), 0.5, "plastic_blue", seg=1)       # bin
                else:
                    J.bench(loc)
                xf = UsdGeom.Xform.Define(st, f"{base}/{fixture}")      # placed group, geometry inside
                xf.AddTransformOp().Set(Gf.Matrix4d(*T.T.flatten().tolist()))
                loc.build(st, f"{base}/{fixture}/Geom", lib.refs(list(loc.groups)))
            sign(s, (dx + 120.0, 210.0, -60.0), (1, 0, 0), (0, 1, 0), (0, 0, 1), 60.0,
                 ["RULE-BASED" if tag == "rules" else "KIMODO + OUR HANDS"], header=tag.upper(),
                 header_mat="powder_blue" if tag == "rules" else "powder_green")
            s.build(st, f"{base}/Sign", lib.refs(list(s.groups)))
            for f in frames:
                f.root = (f.root[0] + dx, f.root[1], f.root[2])
            UsdGeom.Xform.Define(st, f"{base}/{w.id}")
            bake(st, f"{base}/{w.id}", w, frames, lib, sols=sols, parts_root=f"/World/Asset/Parts/{tag}",
                 log=lambda *a: None)
        dress(st, lib, st.GetPrimAtPath("/World/Asset"))
    n = len(d["rules"][0])
    json.dump([{"seq": "process", "eye": [330.0, 620.0, -760.0], "at": [330.0, 40.0, 10.0], "mm": 18,
                "frames": [0, n - 1, 1]}], open(os.path.join(OUT, "shots.json"), "w"), indent=1)
    print("->", out, n, "frames")


if __name__ == "__main__" and "--scene" in sys.argv:
    scene()
