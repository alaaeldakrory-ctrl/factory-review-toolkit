"""Kimodo in the job pipeline: our solver's work poses become Kimodo end-effector keyframes (hands, feet - scaled to
Kimodo's body), Kimodo generates the whole-body motion around them from its mocap-trained prior (no text), the
result is retargeted onto our worker and our solver pins the hands, grips, tools and fingers exactly ("keep the
body, solve the hands"); then the usual checks. Writes a side-by-side scene: rule-based vs Kimodo-based.

    python tools/kimodo/kimodo_job.py ratchet [--every=4] [--steps=100] [--seed=0]
    -> out/people/kimodo/<job>/compare.usd, shots.json, check.json
"""
import json
import math
import os
import pickle
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "exts", "frt.core"))
import numpy as np   # noqa: E402

KIMODO_PY = os.environ.get("KIMODO_PY", r"C:/OmniverseData/kimodo/env/Scripts/python.exe")
KIMODO_HIP_M = 1.00                                       # Kimodo SOMA body: standing hip height (m)
FROM_SOMA = {"pelvis": "Hips", "spine_01": "Spine1", "spine_02": "Spine2", "spine_03": "Chest", "neck_01": "Neck1",
             "head": "Head", "head_top": "HeadEnd"}
for _s, _S in (("l", "Left"), ("r", "Right")):
    FROM_SOMA.update({f"clavicle_{_s}": f"{_S}Shoulder", f"upperarm_{_s}": f"{_S}Arm",
                      f"lowerarm_{_s}": f"{_S}ForeArm", f"hand_{_s}": f"{_S}Hand", f"middle_{_s}": f"{_S}HandMiddle1",
                      f"index_{_s}": f"{_S}HandIndex1", f"pinky_{_s}": f"{_S}HandPinky1", f"thigh_{_s}": f"{_S}Leg",
                      f"calf_{_s}": f"{_S}Shin", f"foot_{_s}": f"{_S}Foot", f"ball_{_s}": f"{_S}ToeBase",
                      f"toe_{_s}": f"{_S}ToeEnd"})


def keyframes(sk, frames, sols, every):
    from frt_core.people.kinematics import fk
    from frt_core.people.skeleton import NAMES
    idx = list(range(0, len(frames), every))
    if idx[-1] != len(frames) - 1:
        idx.append(len(frames) - 1)
    P, R, ur, ul = [], [], [], []
    for i in idx:
        Rw, Pw = fk(sk, sols[i].pose)
        P.append([Pw[n] for n in NAMES])
        R.append([Rw[n] for n in NAMES])
        tg = {t.side: t for t in frames[i].targets}
        ur.append("r" in tg and tg["r"].weight >= 0.5)
        ul.append("l" in tg and tg["l"].weight >= 0.5)
    return idx, np.array(P), np.array(R), np.array(ur), np.array(ul), list(NAMES)


def soma_calibration(sk, s):
    """Rest-shape offsets between Kimodo's SOMA skeleton and ours (spine curve, collarbones, feet): world rotations
    of those joints when Kimodo's T-pose rest is retargeted onto our skeleton (whose rest is identity)."""
    from frt_core.people.andy import CAL_JOINTS
    from frt_core.people.mocap import retarget
    from kimodo_names import NAMES77, NEUTRAL77
    rest = np.array(NEUTRAL77) + np.array([0.0, KIMODO_HIP_M, 0.0])
    J = soma_to_canonical(rest, NAMES77, s)
    _, Rw = retarget(sk, J, world=True)
    return {j: Rw[j] for j in CAL_JOINTS if j in Rw}


def floor_fix(sk, poses):
    """Put the feet on the floor: Kimodo's leg proportions differ from the worker's, so after retargeting the
    support foot can float or sink a few cm. Lower / raise every pelvis by the gap between the support point (the
    lower of heel / ball, per frame) and the floor, smoothed over ~1/3 s so steps stay natural."""
    from frt_core.people.kinematics import fk
    gap = []
    ball0 = float(sk.world["ball_l"][1])
    for p in poses:
        R, P = fk(sk, p)
        gap.append(min(P["foot_l"][1] - float(sk.world["foot_l"][1]), P["foot_r"][1] - float(sk.world["foot_r"][1]),
                       P["ball_l"][1] - ball0, P["ball_r"][1] - ball0))
    g = np.array(gap)
    k = 9
    pad = np.concatenate([g[:1].repeat(k // 2), g, g[-1:].repeat(k // 2)])
    gs = np.convolve(pad, np.ones(k) / k, mode="valid")
    for p, d in zip(poses, gs):
        p.pelvis = p.pelvis - np.array([0.0, d, 0.0])
    return poses


def soma_to_canonical(pj, names77, s):
    """Kimodo joints (m) -> canonical dict (cm, our worker frame)."""
    ix = {n: i for i, n in enumerate(names77)}
    return {c: pj[ix[so]] / s for c, so in FROM_SOMA.items() if so in ix}


def pin_hands(sk, frames, ramp=6):
    """Hands follow the job's targets only while they work (target weight >= 0.5); in between, the base (Kimodo)
    arms play: each frame's target becomes a blend from the base hand to the job target by a pin weight that
    eases in / out over `ramp` frames, so there is no jump where a hand starts or stops working."""
    from frt_core.people.kinematics import fk
    from frt_core.people.motion_gen import lerp_target
    from frt_core.people.solver import HandTarget
    n = len(frames)
    for side in ("l", "r"):
        work = np.array([any(t.side == side and t.weight >= 0.5 for t in f.targets) for f in frames], float)
        w = work.copy()
        for i in range(n):                                    # ease in / out around the working stretches
            lo, hi = max(0, i - ramp), min(n, i + ramp + 1)
            w[i] = max(work[i], float(np.max(work[lo:hi]) * 0.0))
        dist = np.full(n, 1e9)                                # frames to the nearest working frame
        last = -1e9
        for i in range(n):
            last = i if work[i] else last
            dist[i] = i - last
        last = 1e9
        for i in range(n - 1, -1, -1):
            last = i if work[i] else last
            dist[i] = min(dist[i], last - i)
        w = np.clip(1.0 - dist / float(ramp + 1), 0.0, 1.0)
        w = w * w * (3 - 2 * w)
        for i, f in enumerate(frames):
            t = next((x for x in f.targets if x.side == side), None)
            others = [x for x in f.targets if x.side != side]
            if w[i] <= 1e-3:
                f.targets = others                            # free: the base arm as generated
                continue
            if t is None:
                continue
            if w[i] >= 0.999 or f.base is None:
                continue
            R_, P_ = fk(sk, f.base)
            sx = 1.0 if side == "l" else -1.0
            fi = R_[f"hand_{side}"] @ np.array([sx, 0.0, 0.0])
            pm = R_[f"hand_{side}"] @ np.array([0.0, -1.0, 0.0])
            hl = sk.length(f"hand_{side}", f"middle_01_{side}") / 0.47
            b = HandTarget(side, tuple(P_[f"hand_{side}"] + fi * hl * 0.4 + pm * 1.6), tuple(fi), tuple(pm), "relaxed", 0.3)
            f.targets = others + [lerp_target(b, t, float(w[i]), grip=("blend", "relaxed", t.grip, float(w[i])))]
    return frames


def _source(jid, src=None):
    """(frames, rule-based solutions, skeleton, worker, out dir) of a job: the jobs_run.py cache, or src = a dict
    {"frames", "sols", "sk", "worker", "out"} (a job solved on another body, e.g. tools/kimodo/library_build.py)."""
    if src is not None:
        os.makedirs(src["out"], exist_ok=True)
        return src["frames"], src["sols"], src["sk"], src["worker"], src["out"]
    from frt_core.people.skeleton import Skeleton
    from frt_core.people.workers import WORKERS
    out = os.path.join(ROOT, "out", "people", "kimodo", jid)
    os.makedirs(out, exist_ok=True)
    c = pickle.load(open(os.path.join(ROOT, "out", "people", "jobs", "cache", jid + ".pkl"), "rb"))
    w = WORKERS[c["worker"]]
    return c["frames"], c["sols"], Skeleton.for_worker(w), w, out


PROMPTS = json.load(open(os.path.join(HERE, "cloud", "prompts.json"), encoding="utf-8"))   # (encoded once: cloud/)


def prepare(jid, every=4, steps=100, seed=0, full_every=3, force_full=False, src=None, prompt=None):
    """Keyframes of a solved job for Kimodo -> (keys.npz path, generated npz path)."""
    from frt_core.people.motion_gen import FPS
    frames, sols, sk, w, out = _source(jid, src)
    s = KIMODO_HIP_M / float(sk.world["pelvis"][1])              # cm -> Kimodo metres
    idx, P, R, ur, ul, names = keyframes(sk, frames, sols, every)
    feet = np.ones(len(idx), bool)
    posture = any(getattr(f.stance, "drop", None) is not None for f in frames if f.stance is not None)
    full = np.zeros(len(idx), bool)
    if posture or force_full:                         # kneel / squat: give the posture itself every 3rd keyframe
        full[::full_every] = True
    src = os.path.join(out, "keys.npz")
    np.savez(src, seconds=len(frames) / FPS, fps_in=FPS, frames=np.array(idx), t=np.array(idx) / FPS,
             names=np.array(names), P=P, R=R, use_r=ur, use_l=ul, use_feet=feet, use_full=full, scale=s, steps=steps,
             seed=seed, prompt=PROMPTS.get(jid, "") if prompt is None else prompt)
    return src, os.path.join(out, "kimodo.npz")


def generate(pairs):
    """Run Kimodo on [(keys, out)] in one process (the model loads once)."""
    env = dict(os.environ, HF_HOME=r"C:/OmniverseData/kimodo/hf")
    args = [x for p in pairs for x in p]
    r = subprocess.run([KIMODO_PY, os.path.join(HERE, "kimodo_constrained.py")] + args, capture_output=True, encoding="utf-8", errors="replace",
                       text=True, env=env)
    return r.stdout, r.stderr


TRUNK = ("pelvis", "spine_01", "spine_02", "spine_03", "neck_01", "head")
STOOP_FREE = 6.0        # deg of extra trunk lean Kimodo may add over the solved body before it is pulled back
STOOP_FULL = 8.0        # ... and over this much more, the solved trunk entirely


def _lean(sk, pose):
    from frt_core.people.kinematics import fk
    _, P = fk(sk, pose, ["neck_01"])
    v = P["neck_01"] - P["pelvis"]
    return math.degrees(math.acos(max(-1.0, min(1.0, float(v[1] / (np.linalg.norm(v) + 1e-9))))))


def upright(sk, frames, sols):
    """Kimodo without text prompts stoops: 10-20 deg more trunk lean than the task needs, even standing at rest, and
    a stooped chest leaves the hands behind it (arms back past the hips). Where its trunk leans more than the solved
    body's at the same moment, the trunk, neck and head are eased back to the solved ones (smoothed over 0.4 s)."""
    from frt_core.people.bake import blend
    u = np.array([min(1.0, max(0.0, (_lean(sk, f.base) - _lean(sk, s_.pose) - STOOP_FREE) / STOOP_FULL))
                  for f, s_ in zip(frames, sols)])
    k = 5
    u = np.array([u[max(0, i - k):i + k + 1].max() for i in range(len(u))])            # (no flicker at the edges)
    u = np.convolve(np.pad(u, k, mode="edge"), np.ones(2 * k + 1) / (2 * k + 1), mode="valid")
    for f, s_, w in zip(frames, sols, u):
        if w <= 1e-3:
            continue
        b = blend(f.base, s_.pose, float(w))
        for j in TRUNK:
            f.base.rot[j] = b.rot[j]
    return u


def finalize(jid, src=None, keep_out=None, M=None):
    """Kimodo result -> our worker, final pass (Kimodo's body, our hands), checks -> summary dict. keep_out: the
    keep-out rules when the work was moved (api.move_rules; default: the job's own); M: that move (the fixture
    the body keeps out of moves with it)."""
    from frt_core.people.collide import fixture_field
    from frt_core.people.bake import solve_frames
    from frt_core.people.check import check_frames
    from frt_core.people.jobs import BY_ID
    from frt_core.people.mocap import retarget
    from frt_core.people.motion_gen import FPS
    from frt_core.people.solver import PostureSolver
    job = BY_ID[jid]
    frames, sols, sk, w, out = _source(jid, src)
    s = KIMODO_HIP_M / float(sk.world["pelvis"][1])
    gen = os.path.join(out, "kimodo.npz")
    if not os.path.isfile(gen):
        return {"job": jid, "error": "no Kimodo output"}
    g = np.load(gen)
    pj, names77, kfps = g["posed_joints"], [str(x) for x in g["names77"]], float(g["fps"])
    k_ = np.array([1.0, 4.0, 6.0, 4.0, 1.0]) / 16.0          # light temporal smoothing (hands are pinned anyway)
    pad = np.concatenate([pj[:1].repeat(2, 0), pj, pj[-1:].repeat(2, 0)])
    pj = sum(k_[i] * pad[i:i + len(pj)] for i in range(5))
    # resample Kimodo (30 fps) to our frames (24 fps), retarget, keep our hand targets / tools / gaze
    new = []
    cal = soma_calibration(sk, s)
    for i, f in enumerate(frames):
        x = i / FPS * kfps
        a = min(int(np.floor(x)), len(pj) - 1)
        b = min(a + 1, len(pj) - 1)
        u = x - a
        J = soma_to_canonical(pj[a] * (1 - u) + pj[b] * u, names77, s)
        pose = retarget(sk, J, cal)
        pose.pelvis = J["pelvis"] - sk.world["pelvis"]
        # the job's own stance (feet exactly where authored; legs solved to them), Kimodo's body on top
        new.append(type(f)(f.targets, f.look, dict(f.tools), root=f.root, stance=f.stance, base=pose, label=f.label))
        new[-1].world = getattr(f, "world", None)
    upright(sk, new, sols)
    if os.environ.get("KIMODO_PIN", "all") == "work":          # measured: pinning on every frame passes 26/50, work-only 7/50
        pin_hands(sk, new)                    # pinned while working, Kimodo's arms in between
    # KIMODO_PIN=all: the job's hand targets on every frame (Kimodo gives only the body)
    S = PostureSolver(sk, w)
    S.keep_out = job.keep_out_rules() if keep_out is None else keep_out
    S.field = fixture_field(job, M)                 # legs and feet out of the fixture too
    S.base_correct = True                     # Kimodo's body, corrected for keep-outs / reach / the planted feet
    ks = solve_frames(S, new, log=lambda *a: None)
    rep_k = check_frames(sk, new, ks, jid + " kimodo", keep_out=S.keep_out, field=S.field)
    rep_r = check_frames(sk, frames, sols, jid + " rules", keep_out=S.keep_out, field=S.field)
    with open(os.path.join(out, "kimodo.pkl"), "wb") as f:
        pickle.dump({"frames": new, "sols": ks, "worker": w.id}, f)
    summary = {"job": jid, "kimodo": {"ok": rep_k["ok"], "counts": rep_k["counts"],
                                      "worst_contact_cm": rep_k["worst_contact_cm"]},
               "rules": {"ok": rep_r["ok"], "counts": rep_r["counts"]}}
    json.dump(summary, open(os.path.join(out, "check.json"), "w"), indent=1)
    return summary


def main():
    jid = sys.argv[1]
    opt = {a.split("=")[0][2:]: a.split("=")[1] for a in sys.argv[2:] if a.startswith("--")}
    src, gen = prepare(jid, int(opt.get("every", 4)), int(opt.get("steps", 100)), int(opt.get("seed", 0)),
                       int(opt.get("full_every", 3)), "full" in opt)
    so, se = generate([(src, gen)])
    print(so.strip().splitlines()[-1] if so.strip() else se[-2000:])
    print(json.dumps(finalize(jid)))


if __name__ == "__main__":
    main()
