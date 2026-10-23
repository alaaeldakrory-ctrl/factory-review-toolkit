"""Kimodo in the job pipeline: our solver's work poses become Kimodo end-effector keyframes (hands, feet - scaled to
Kimodo's body), Kimodo generates the whole-body motion around them from its mocap-trained prior (no text), the
result is retargeted onto our worker and our solver pins the hands, grips, tools and fingers exactly ("keep the
body, solve the hands"); then the usual checks. Writes a side-by-side scene: rule-based vs Kimodo-based.

    python tools/kimodo/kimodo_job.py ratchet [--every=4] [--steps=100] [--seed=0]
    -> out/people/kimodo/<job>/compare.usd, shots.json, check.json
"""
import json
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


def soma_to_canonical(pj, names77, s):
    """Kimodo joints (m) -> canonical dict (cm, our worker frame)."""
    ix = {n: i for i, n in enumerate(names77)}
    return {c: pj[ix[so]] / s for c, so in FROM_SOMA.items() if so in ix}


def main():
    from frt_core.people.bake import solve_frames
    from frt_core.people.check import check_frames
    from frt_core.people.jobs import BY_ID
    from frt_core.people.mocap import retarget
    from frt_core.people.motion_gen import FPS
    from frt_core.people.skeleton import Skeleton
    from frt_core.people.solver import PostureSolver, Stance
    from frt_core.people.workers import WORKERS
    jid = sys.argv[1]
    opt = {a.split("=")[0][2:]: a.split("=")[1] for a in sys.argv[2:] if a.startswith("--")}
    every, steps, seed = int(opt.get("every", 4)), int(opt.get("steps", 100)), int(opt.get("seed", 0))
    out = os.path.join(ROOT, "out", "people", "kimodo", jid)
    os.makedirs(out, exist_ok=True)
    c = pickle.load(open(os.path.join(ROOT, "out", "people", "jobs", "cache", jid + ".pkl"), "rb"))
    job = BY_ID[jid]
    w = WORKERS[c["worker"]]
    sk = Skeleton.for_worker(w)
    frames, sols = c["frames"], c["sols"]
    s = KIMODO_HIP_M / float(sk.world["pelvis"][1])              # cm -> Kimodo metres
    idx, P, R, ur, ul, names = keyframes(sk, frames, sols, every)
    feet = np.ones(len(idx), bool)
    posture = any(getattr(f.stance, "drop", None) is not None for f in frames if f.stance is not None)
    full = np.zeros(len(idx), bool)
    if posture or "full" in opt:                      # kneel / squat: give the posture itself every 3rd keyframe
        full[::int(opt.get("full_every", 3))] = True
    src = os.path.join(out, "keys.npz")
    np.savez(src, seconds=len(frames) / FPS, fps_in=FPS, frames=np.array(idx), t=np.array(idx) / FPS,
             names=np.array(names), P=P, R=R, use_r=ur, use_l=ul, use_feet=feet, use_full=full, scale=s, steps=steps,
             seed=seed)
    gen = os.path.join(out, "kimodo.npz")
    env = dict(os.environ, HF_HOME=r"C:/OmniverseData/kimodo/hf")
    r = subprocess.run([KIMODO_PY, os.path.join(HERE, "kimodo_constrained.py"), src, gen], capture_output=True,
                       text=True, env=env)
    print(r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr[-2000:])
    g = np.load(gen)
    pj, names77, kfps = g["posed_joints"], [str(x) for x in g["names77"]], float(g["fps"])
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
        g_ = type(f)(f.targets, f.look, dict(f.tools), root=f.root, stance=f.stance, base=pose, label=f.label)
        new.append(g_)
    S = PostureSolver(sk, w)
    S.keep_out = job.keep_out_rules()
    S.base_correct = True                     # Kimodo's body, corrected for keep-outs / reach / the planted feet
    ks = solve_frames(S, new, log=lambda *a: None)
    rep_k = check_frames(sk, new, ks, jid + " kimodo", keep_out=S.keep_out)
    rep_r = check_frames(sk, frames, sols, jid + " rules", keep_out=S.keep_out)
    with open(os.path.join(out, "kimodo.pkl"), "wb") as f:
        pickle.dump({"frames": new, "sols": ks, "worker": c["worker"]}, f)
    summary = {"job": jid, "kimodo": {"ok": rep_k["ok"], "counts": rep_k["counts"], "worst_contact_cm": rep_k["worst_contact_cm"]},
               "rules": {"ok": rep_r["ok"], "counts": rep_r["counts"]}, "keyframes": len(idx), "every": every}
    json.dump(summary, open(os.path.join(out, "check.json"), "w"), indent=1)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
