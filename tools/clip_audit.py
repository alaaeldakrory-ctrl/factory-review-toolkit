"""Checks on what is shown: a solved motion measured on the character it plays on (the retargeted Mixamo rig), for
the things the solver's own checks never looked at - whether the body is one a person would hold:

    shrug        a shoulder joint raised above its rest height along the chest's own up axis (> 3 cm)
    arms cross   the forearms closer than 7 cm to each other (the arms tangle or cross the body midline)
    seat out     the hips behind the heels (> 14 cm) while standing: bottom sticking out
    bent double  the trunk bent past 50 deg while both working hands are above the knees (a squat / step would do)
    wrist        the hand bent against the forearm past 55 deg (flexion / extension) or 30 deg sideways
    foot slide   a planted foot moving more than 2.5 cm between frames
    floor        a foot more than 2 cm through the floor, or both feet off it (> 4 cm) while standing still
    retarget     the character's wrist more than 1 cm from the solved one (the body shown is not the one solved)
    behind back  an elbow drawn back past 19 cm behind the shoulder, or a wrist behind it (4 cm working, 10 cm free),
                 along the chest's own forward axis: no one works with the arms behind the back
    arms held up a hand that is not working, while standing, with its elbow bent past 45 deg: at ease the arms
                 hang (elbow ~25 deg); held up in front of the belly they look short and duck-like

    python tools/clip_audit.py <kimodo.pkl | rules.pkl> [...]      (frames + sols, on the website's character)
    -> per file: counts, worst values, frames; exit 1 if any issue
"""
import math
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "exts", "frt.core")]

LIMITS = {"shrug": 3.0, "arms cross": 7.0, "seat out": 14.0, "bent double": 50.0, "wrist": 55.0, "wrist side": 30.0,
          "foot slide": 2.5, "floor": 2.0, "retarget": 1.0, "behind elbow": 19.0, "behind wrist": 4.0,
          "behind free wrist": 10.0, "arms held up": 45.0}


def _ang(a, b):
    return math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))))))


def _seg_dist(p1, q1, p2, q2):
    """Closest distance between segments p1q1 and p2q2."""
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a, e, f = float(d1 @ d1), float(d2 @ d2), float(d2 @ r)
    c, b = float(d1 @ r), float(d1 @ d2)
    den = a * e - b * b
    s = min(1.0, max(0.0, (b * f - c * e) / den)) if den > 1e-9 else 0.0
    t = (b * s + f) / e if e > 1e-9 else 0.0
    if t < 0.0:
        t, s = 0.0, min(1.0, max(0.0, -c / a))
    elif t > 1.0:
        t, s = 1.0, min(1.0, max(0.0, (b - c) / a))
    return float(np.linalg.norm((p1 + d1 * s) - (p2 + d2 * t)))


def audit(frames, sols, sk, rt=None):
    """{"counts": {kind: frames}, "worst": {kind: value}, "frames": {kind: [first few frames]}}."""
    from frt_core.people.kinematics import fk
    from web_export import PRE as W_PRE, Retarget
    rt = rt or Retarget(sk)
    back = lambda v: (np.asarray(v) - np.array([0.0, rt.floor, 0.0]) - rt.shift) * rt.scale   # rig -> solved body cm
    sp2 = rt.rig.idx[W_PRE + "Spine2"]
    rest_sh = {S: (rt.rig.pos(f"{S}Arm")[1] - rt.rig.pos("Spine2")[1]) * rt.scale for S in ("Left", "Right")}
    hits, worst = {}, {}

    def hit(k, i, v):
        hits.setdefault(k, []).append(i)
        worst[k] = round(max(worst.get(k, 0.0), float(v)), 1)
    prev_feet = None
    for i, (f, s) in enumerate(zip(frames, sols)):
        R, P = fk(sk, s.pose)
        rt.pose(s.pose)
        W, hw = rt._last
        M = {b: back(rt.positions(W, hw, b)) for b in
             ("Neck", "LeftArm", "RightArm", "LeftForeArm", "RightForeArm", "LeftHand", "RightHand", "Hips",
              "LeftFoot", "RightFoot", "LeftToeBase", "RightToeBase", "LeftHandMiddle1", "RightHandMiddle1",
              "LeftHandIndex1", "LeftHandPinky1", "RightHandIndex1", "RightHandPinky1", "Spine2")}
        for S, s_ in (("Left", "l"), ("Right", "r")):
            up = float(np.dot(M[f"{S}Arm"] - M["Spine2"], W[sp2] @ np.array([0.0, 1.0, 0.0]))) - rest_sh[S]
            raised = float(M[f"{S}Hand"][1]) > float(M[f"{S}Arm"][1]) - 5.0   # (reaching up: the shoulder rises)
            if up > LIMITS["shrug"] and not raised:
                hit("shrug", i, up)
            fw = R["spine_03"] @ np.array([0.0, 0.0, 1.0])            # the chest's forward axis (solved body; the
            be = -float(np.dot(M[f"{S}ForeArm"] - M[f"{S}Arm"], fw))  # rig's own bone axes are not anatomical)
            bw = -float(np.dot(M[f"{S}Hand"] - M[f"{S}Arm"], fw))
            works = any(t.side == s_ and t.weight >= 0.5 for t in f.targets)
            lw = LIMITS["behind wrist"] if works else LIMITS["behind free wrist"]
            idle = not any(t.side == s_ and t.weight >= 0.35 for t in f.targets)
            if idle and getattr(f.stance, "drop", None) is None and f.root is None:
                el = _ang(M[f"{S}ForeArm"] - M[f"{S}Arm"], M[f"{S}Hand"] - M[f"{S}ForeArm"])
                if el > LIMITS["arms held up"]:
                    hit("arms held up", i, el)
            if be > LIMITS["behind elbow"] or bw > lw:
                hit("behind back", i, max(be - LIMITS["behind elbow"], bw - lw))
            e = float(np.linalg.norm(M[f"{S}Hand"] - P[f"hand_{s_}"]))
            if e > LIMITS["retarget"]:
                hit("retarget", i, e)
            fa = M[f"{S}Hand"] - M[f"{S}ForeArm"]                     # wrist: palm direction against the forearm
            hd = M[f"{S}HandMiddle1"] - M[f"{S}Hand"]
            across = M[f"{S}HandIndex1"] - M[f"{S}HandPinky1"]
            nrm = np.cross(hd, across) * (1.0 if S == "Left" else -1.0)
            flex = 90.0 - _ang(fa, nrm)                                # + towards the palm
            side = math.degrees(math.asin(max(-1.0, min(1.0, float(np.dot(fa, across) / (np.linalg.norm(fa) * np.linalg.norm(across) + 1e-12))))))
            bend = _ang(fa, hd)
            if bend > LIMITS["wrist"] and abs(flex) > LIMITS["wrist"] * 0.8:
                hit("wrist", i, bend)
            elif abs(side) > LIMITS["wrist side"] + 0 and bend > LIMITS["wrist side"]:
                hit("wrist", i, abs(side))
        dcross = _seg_dist(M["LeftForeArm"], M["LeftHand"], M["RightForeArm"], M["RightHand"])
        two = sum(1 for t in f.targets if t.weight >= 0.5) == 2
        if dcross < LIMITS["arms cross"] and not (two and float(np.linalg.norm(M["LeftHand"] - M["RightHand"])) < 14.0):
            hit("arms cross", i, LIMITS["arms cross"] - dcross)
        fwd = R["pelvis"] @ np.array([0.0, 0.0, 1.0])
        fwd = np.array([fwd[0], 0.0, fwd[2]])
        fwd /= np.linalg.norm(fwd) + 1e-12
        heels = (M["LeftFoot"] + M["RightFoot"]) / 2
        standing = getattr(f.stance, "drop", None) is None
        seat = float(np.dot(heels - M["Hips"], fwd))
        if standing and seat > LIMITS["seat out"]:
            hit("seat out", i, seat)
        tr = M["Spine2"] - M["Hips"]
        flex = _ang(tr, np.array([0.0, 1.0, 0.0]))
        knee_y = 0.5 * (P["calf_l"][1] + P["calf_r"][1])
        hands_up = [t for t in f.targets if t.weight >= 0.5 and float(t.point[1]) > knee_y + 10.0]
        if standing and flex > LIMITS["bent double"] and hands_up:
            hit("bent double", i, flex)
        feet = {s_: M[f"{S}Foot"] for S, s_ in (("Left", "l"), ("Right", "r"))}
        for s_ in "lr":
            low = min(feet[s_][1], M[("Left" if s_ == "l" else "Right") + "ToeBase"][1])
            if low < -LIMITS["floor"]:
                hit("floor", i, -low)
        st = f.stance
        if prev_feet is not None and st is not None and not getattr(st, "free", False) and f.root is None:
            for s_ in "lr":
                d = float(np.linalg.norm((feet[s_] - prev_feet[s_])[[0, 2]]))
                if d > LIMITS["foot slide"]:
                    hit("foot slide", i, d)
        prev_feet = feet
    return {"counts": {k: len(v) for k, v in hits.items()}, "worst": worst,
            "frames": {k: v[:8] for k, v in hits.items()}}


def main():
    from frt_core.people.mixamo import load
    site = os.environ.get("FRT_SITE", os.path.join(HERE, "..", "..", "frt-site"))
    ch = load(os.path.join(site, "assets", "worker.glb"))
    bad = 0
    for p in sys.argv[1:]:
        d = pickle.load(open(p, "rb"))
        fr = d["frames"] if "frames" in d else d["T"].frames                # (a job / library clip, or a task run)
        r = audit(fr, d["sols"], ch.skeleton)
        bad += bool(r["counts"])
        print(f"{p}: {r['counts'] or 'clean'} {r['worst'] if r['counts'] else ''}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
