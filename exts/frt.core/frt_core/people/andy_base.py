"""Recorded bodies under task work: each step's work frames get a real person's motion from the AnDy library (an
action, hand height and posture like the step's) as the base pose; the solver keeps that body (small corrections
for the equipment, reach and the planted feet) and puts the hands exactly on the work.

    apply(T)          # after Task.compile: frames get .base, T.S.base_correct is set

The clip plays at its own speed, back and forth when the step is longer than it (a person working in place sways
and shifts slowly, never in slow motion); neighbouring clips cross-fade over BLEND frames.
"""
import json
import math
import os

import numpy as np

from .bake import blend

INDEX = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "out", "people", "library", "andy_index.json")
BLEND = 12
VERB_ACTION = {"fasten": "screw", "hand_start": "fine manipulation", "connect": "fine manipulation",
               "cable_tie": "fine manipulation", "inspect": "fine manipulation", "latch": "fine manipulation"}
SKIP_PHASES = ("walk", "kneel down", "stand up")
_trials = {}


def band(h, stature):
    r = h / stature
    for lim, name in ((0.12, "floor"), (0.33, "knee"), (0.52, "waist"), (0.70, "chest"), (0.86, "shoulder")):
        if r < lim:
            return name
    return "overhead"


def posture_for(height, kneeling):
    if kneeling:
        return ("Kn_BF", "Cr_BF")
    return {"floor": ("Cr_BF",), "knee": ("Cr_BF", "St_BF"), "waist": ("St_BF", "St_BS", "St_U"),
            "chest": ("St_U", "St_BS"), "shoulder": ("St_OS", "St_U", "St_OH"), "overhead": ("St_OH", "St_OS")}[height]


BANDS = ("floor", "knee", "waist", "chest", "shoulder", "overhead")


def pick(idx, action, height, reach, stature, postures, used):
    """Best library clip for the step: where the recorded hand really was (height band, reach from the pelvis, scaled
    to the body) closest to the step's work - a clip recorded for other work bends the body wrong; then the posture
    and action asked, the cleanest, a person not used yet."""
    hb = BANDS.index(height)
    best = None
    for x in idx:
        if not (x.get("ok") and x["hands"] in ("r", "both") and not x["walking"] and x["seconds"] >= 2.0
                and x["action"] in (action, "fine manipulation") and not x["posture"].startswith(("Kn", "Wa"))):
            continue
        r = x["reach_cm"] * stature / max(x["stature"], 1.0)
        sc = (2.0 * abs(BANDS.index(x["height"]) - hb) + abs(r - reach) / 10.0 + (x["posture"] not in postures)
              + 0.5 * (x["action"] != action) + 0.5 * (x["trial"] in used) + 0.02 * x["worst_pop"])
        if best is None or sc < best[0]:
            best = (sc, x)
    return None if best is None else best[1]


def trial(name):
    from .andy import Trial
    if name not in _trials:
        _trials[name] = Trial(name)
    return _trials[name]


def _log(R):
    c = max(-1.0, min(1.0, (np.trace(R) - 1.0) / 2.0))
    th = math.acos(c)
    if th < 1e-6:
        return np.zeros(3)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2.0 * math.sin(th))
    return w * th


def spread_spine(pose):
    """The recorded bend shared evenly by the three spine joints (same chest orientation): the retarget can put a
    whole lumbar bend into one joint, past what one joint takes."""
    from .kinematics import axis_angle
    J = ("spine_01", "spine_02", "spine_03")
    Rt = pose.local(J[0]) @ pose.local(J[1]) @ pose.local(J[2])
    w = _log(Rt)
    th = float(np.linalg.norm(w))
    part = axis_angle(w / th, th / 3.0) if th > 1e-6 else np.eye(3)
    for j in J:
        pose.rot[j] = part.copy()
    return pose


def clip_poses(sk, clip):
    tr = trial(clip["trial"])
    rec = tr.frames(sk, clip["start"], min(tr.n, clip["end"] + 1))
    return [spread_spine(f.base) for f in rec]


def apply(T, log=print):
    idx = json.load(open(INDEX))["clips"]
    sk = T.sk
    stature = float(sk.tips["head_top"][1]) if "head_top" in getattr(sk, "tips", {}) else 175.0
    used, runs = set(), []
    for s in T.steps:
        act = VERB_ACTION.get(s["do"])
        if act is None:
            continue
        fr = [i for i in range(s["start"], s["end"] + 1) if T.frames[i].base is None and T.frames[i].root is not None
              and not any(p in T.frames[i].label.split(":")[-1] for p in SKIP_PHASES)]
        if not fr:
            continue
        ys = [float(t.point[1]) for i in fr for t in T.frames[i].targets if t.side == "r" and t.weight >= 0.5]
        if not ys:
            continue
        kneel = any(getattr(T.frames[i].stance, "drop", None) is not None for i in fr)
        if kneel:                                                      # (our kneel: the recorded ones fight it)
            continue
        h = band(float(np.median(ys)), stature)
        xz = [math.hypot(float(t.point[0]), float(t.point[2])) for i in fr for t in T.frames[i].targets
              if t.side == "r" and t.weight >= 0.5]
        reach = float(np.median(xz))
        clip = pick(idx, act, h, reach, stature, posture_for(h, False), used)
        if clip is None:
            log(f"  {s['id']}: no recorded clip for {act} / {h}")
            continue
        used.add(clip["trial"])
        poses = clip_poses(sk, clip)
        m = len(poses)
        for k, i in enumerate(fr):                                    # real speed, back and forth
            q = k % (2 * m - 2) if m > 1 else 0
            T.frames[i].base = poses[q if q < m else 2 * m - 2 - q].copy()
        runs.append(fr)
        log(f"  {s['id']}: work {h} / reach {reach:.0f} -> {clip['trial']} {clip['action']} / {clip['posture']} / {clip['height']} reach {clip['reach_cm']} ({clip['seconds']} s)")
    # cross-fade where a recorded stretch meets another (or a walk) right next to it
    for fr in runs:
        a = fr[0]
        if a > 0 and T.frames[a - 1].base is not None:
            p0 = T.frames[a - 1].base
            for k in range(min(BLEND, len(fr))):
                i = fr[k]
                T.frames[i].base = blend(p0, T.frames[i].base, (k + 1) / (BLEND + 1))
    T.S.base_correct = True
    return T
