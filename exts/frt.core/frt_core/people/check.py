"""Motion quality checks on solved frames - things that make animation look wrong:

- joint limits: every joint's rotation from its rest pose stays inside a human range (no elbow bent backwards,
  no wrist folded flat, no finger over-curled, no spine segment bent like a hinge);
- pops: no joint rotates more than a frame-to-frame limit (a sudden 180 / 360 degree flip, a hand shape snapping);
- contact: the hand reaches its target (the palm where the tool grip is) and planted feet stay planted;
- hyperextension: elbows and knees never bend the wrong way.

    rep = check_frames(sk, frames, sols); rep["ok"], rep["issues"][:10]
"""
import math

import numpy as np

from .kinematics import fk
from .skeleton import NAMES

# max rotation from rest (deg) per joint family
LIMITS = {"pelvis": 75, "spine_01": 35, "spine_02": 35, "spine_03": 35, "neck_01": 55, "head": 65,
          "clavicle": 35, "upperarm": 175, "lowerarm": 165, "hand": 95, "thumb_01": 80, "thumb": 95,
          "finger": 110, "thigh": 135, "calf": 155, "foot": 65, "ball": 55}
POP = {"default": 32.0, "finger": 55.0, "hand": 40.0}      # max change per frame at 24 fps (deg)
# postures that are possible but look (and are) wrong for hand work: arm folded up tight, trunk bent over, head
# turned or dropped far - the work is too close, too low or behind the worker (solver angles, deg)
AWKWARD = {"elbow_l": 150.0, "elbow_r": 150.0, "trunk_flexion": 32.0, "trunk_side": 20.0, "trunk_twist": 30.0,
           "neck_flexion": 50.0, "neck_twist": 60.0}
AWKWARD_LOW_TRUNK = 50.0


def _family(j):
    base = j.rsplit("_", 1)[0] if j.endswith(("_l", "_r")) else j
    if base.startswith(("index", "middle", "ring", "pinky")):
        return "finger"
    if base.startswith("thumb"):
        return "thumb_01" if base == "thumb_01" else "thumb"
    return base


def rot_angle(R):
    c = (np.trace(R) - 1.0) / 2.0
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def check_frames(sk, frames, sols, name=""):
    issues = []
    prev = None
    worst_pop, worst_lim, worst_contact, worst_foot = 0.0, 0.0, 0.0, 0.0
    for i, (f, sol) in enumerate(zip(frames, sols)):
        pose = sol.pose
        for j in NAMES:
            if j == "root":
                continue
            fam = _family(j)
            a = rot_angle(pose.local(j))
            lim = LIMITS.get(fam, 120)
            if a > lim:
                worst_lim = max(worst_lim, a - lim)
                issues.append((i, "limit", j, round(a, 1), lim))
            if prev is not None:
                d = rot_angle(prev.local(j).T @ pose.local(j))
                cap = POP["finger"] if fam in ("finger", "thumb", "thumb_01") else POP["hand"] if fam == "hand" else POP["default"]
                if d > cap:
                    worst_pop = max(worst_pop, d)
                    issues.append((i, "pop", j, round(d, 1), cap))
        ang = getattr(sol, "angles", None) or {}
        R, P = fk(sk, pose)
        if f.root is None or getattr(f, "static", True):             # (not while walking)
            low = any(t.weight >= 0.5 and t.point[1] < P["pelvis"][1] + 5.0 for t in f.targets)
            for k, lim in AWKWARD.items():
                if k == "trunk_flexion" and low:
                    lim = AWKWARD_LOW_TRUNK                          # hands below the hips: bending is the task
                if abs(float(ang.get(k, 0.0))) > lim:
                    issues.append((i, "awkward", k, round(abs(float(ang[k])), 1), lim))
        for s in ("l", "r"):                                 # elbows / knees never the wrong way
            ua, fa = P[f"lowerarm_{s}"] - P[f"upperarm_{s}"], P[f"hand_{s}"] - P[f"lowerarm_{s}"]
            th, ca = P[f"calf_{s}"] - P[f"thigh_{s}"], P[f"foot_{s}"] - P[f"calf_{s}"]
            knee_fwd = np.dot(np.cross(th, ca), R["pelvis"] @ np.array([1.0, 0.0, 0.0]))
            if knee_fwd < -1e-3 * np.linalg.norm(th) * np.linalg.norm(ca) and _ang(th, ca) > 3:   # forward bend: +X
                issues.append((i, "knee backwards", s, round(_ang(th, ca), 1), 0))
        for t in f.targets:                                  # contact: grip point reached
            Rh = R[f"hand_{t.side}"]
            sx = 1.0 if t.side == "l" else -1.0
            fingers = Rh @ np.array([sx, 0.0, 0.0])
            palm = Rh @ np.array([0.0, -1.0, 0.0])
            grip = P[f"hand_{t.side}"] + fingers * _hand_len(sk) * 0.40 + palm * 1.6
            e = float(np.linalg.norm(grip - np.asarray(t.point, float)))
            if t.weight >= 0.5:
                worst_contact = max(worst_contact, e)
                if e > 4.0:
                    issues.append((i, "contact", t.side, round(e, 1), 4.0))
        st = f.stance
        if st is not None and not getattr(st, "free", False):
            for s in ("l", "r"):
                e = float(np.linalg.norm(P[f"foot_{s}"] - np.asarray(getattr(st, "left" if s == "l" else "right"))))
                worst_foot = max(worst_foot, e)
        prev = pose
    kinds = {}
    for it in issues:
        kinds[it[1]] = kinds.get(it[1], 0) + 1
    return {"name": name, "frames": len(frames), "ok": not issues, "counts": kinds, "issues": issues,
            "worst_pop": round(worst_pop, 1), "worst_limit_over": round(worst_lim, 1),
            "worst_contact_cm": round(worst_contact, 1), "worst_foot_cm": round(worst_foot, 1)}


def _ang(a, b):
    return math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))))))


def _hand_len(sk):
    return sk.length("hand_l", "middle_01_l") / 0.47
