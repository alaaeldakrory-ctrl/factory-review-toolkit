"""The AnDy dataset (Maurice et al. 2019, "Human Movement and Ergonomics: an Industry-Oriented Dataset for
Collaborative Robotics", IJRR; Zenodo 10.5281/zenodo.3254403, CC BY 4.0): Xsens motion of 13 people doing
industry-style tasks (screwing at several heights, picking / placing / carrying loads), with per-frame action and
posture labels from three annotators (postures on the EAWS grid).

    trial = Trial("Participant_541_Setup_A_Seq_3_Trial_1")
    trial.segments()          # [(action, start, end)] at 24 fps, majority of the annotators
    frames = trial.frames(sk, a, b)   # Frames replaying the motion on a worker (root moving with the hips)

Data folder: FRT_ANDY or C:/OmniverseData/andy (xsens_bvh/, labels_csv/, participant_specifications.csv).
"""
import collections
import csv
import math
import os

import numpy as np

from .bvh import BvhClip, xsens_joints
from .mocap import retarget

ROOT = os.environ.get("FRT_ANDY", r"C:/OmniverseData/andy")
ACTIONS = {"Id": "idle", "Re": "reach", "Pi": "pick", "Pl": "place", "Rl": "release", "Ca": "carry", "Fm": "fine manipulation",
           "Sc": "screw"}
POSTURES = {"St": "standing", "Wa": "walking", "Cr": "crouching", "Kn": "kneeling"}
CAL_JOINTS = ("spine_01", "spine_02", "spine_03", "neck_01", "head", "clavicle_l", "clavicle_r", "foot_l", "foot_r")
CREDIT = ("Motion: AnDy dataset, Maurice P. et al. (2019), Human Movement and Ergonomics: an Industry-Oriented Dataset "
          "for Collaborative Robotics, IJRR - CC BY 4.0")


def trials():
    d = os.path.join(ROOT, "xsens_bvh")
    out = []
    for p in sorted(os.listdir(d)) if os.path.isdir(d) else []:
        for f in sorted(os.listdir(os.path.join(d, p))):
            if f.endswith(".xsens.bvh"):
                out.append(f[:-len(".xsens.bvh")])
    return out


def _vote(row, key):
    v = [row[f"{key}_Annotator{k}"] for k in (1, 2, 3)]
    return collections.Counter(v).most_common(1)[0][0]


class Trial:
    def __init__(self, name):
        self.name = name
        part = name.split("_Setup")[0]
        self.bvh_path = os.path.join(ROOT, "xsens_bvh", part, name + ".xsens.bvh")
        self.lab_path = os.path.join(ROOT, "labels_csv", part, name + ".labels.csv")
        self.clip = BvhClip(self.bvh_path, fps=24)
        rows = list(csv.DictReader(open(self.lab_path)))
        n = min(len(rows), self.clip.frames)
        self.action = [_vote(r, "current_action") for r in rows[:n]]
        self.posture = [_vote(r, "detailed_posture") for r in rows[:n]]
        self.n = n
        self._J = {}

    def segments(self, min_len=6):
        out, a, s = [], None, 0
        for i, x in enumerate(self.action + [None]):
            if x != a:
                if a is not None and i - s >= min_len:
                    out.append((a, s, i))
                a, s = x, i
        return out

    def joints(self, i):
        if i not in self._J:
            self._J[i] = xsens_joints(self.clip, i)
        return self._J[i]

    def heading(self, i):
        """Facing yaw (deg; 0 = +Z, positive towards +X) from the hip line."""
        J = self.joints(i)
        hl = J["thigh_l"] - J["thigh_r"]
        fwd = np.cross(hl, np.array([0.0, 1.0, 0.0]))
        return math.degrees(math.atan2(fwd[0], fwd[2]))

    def facing_frame(self, i, yaw):
        """Joints of frame i turned by -yaw about the pelvis' vertical (the heading goes to the root)."""
        a = math.radians(-yaw)
        Ry = np.array([[math.cos(a), 0.0, math.sin(a)], [0.0, 1.0, 0.0], [-math.sin(a), 0.0, math.cos(a)]])
        J = self.joints(i)
        c = np.array([J["pelvis"][0], 0.0, J["pelvis"][2]])
        return {k: Ry @ (v - c) + c for k, v in J.items()}

    def neutral(self):
        """Index of a quiet upright standing frame (idle, St_U) - the calibration stance."""
        for i in range(self.n):
            if self.action[i] == "Id" and self.posture[i] == "St_U":
                return i
        return 0

    def calibration(self, sk):
        if getattr(self, "_cal", None) is None:
            i = self.neutral()
            _, Rw = retarget(sk, self.facing_frame(i, self.heading(i)), world=True)
            self._cal = {j: Rw[j] for j in CAL_JOINTS if j in Rw}
        return self._cal

    def frames(self, sk, a=0, b=None, origin=None, drop=None):
        """Frames replaying frames a..b on skeleton sk: legs, trunk, arms as recorded (directions retargeted), the
        root following the hips over the floor (scaled to the body), the pelvis height scaled the same way."""
        from .motion_gen import Frame
        from .solver import Stance
        b = self.n if b is None else min(b, self.n)
        cal = self.calibration(sk)
        yaw = np.unwrap(np.radians([self.heading(i) for i in range(a, b)]))
        yaw = np.degrees(np.convolve(np.pad(yaw, 5, mode="edge"), np.ones(11) / 11.0, mode="valid"))
        J0 = self.joints(a)
        hip0 = float(np.mean([self.joints(i)["pelvis"][1] for i in range(a, b, max(1, (b - a) // 20))]))
        stand = max(self.joints(i)["pelvis"][1] for i in range(a, b, max(1, (b - a) // 40)))
        k = float(sk.world["pelvis"][1]) / max(stand, 1.0)
        o = np.asarray(origin if origin is not None else (J0["pelvis"][0], 0.0, J0["pelvis"][2]), float)
        if drop is None:
            from .kinematics import fk
            low = []
            for i in range(a, b, max(1, (b - a) // 40)):
                p = retarget(sk, self.facing_frame(i, yaw[i - a]), cal)
                p.pelvis = np.array([0.0, (self.joints(i)["pelvis"][1] - stand) * k, 0.0])
                R, P = fk(sk, p)
                low.append(min(P["foot_l"][1], P["foot_r"][1], P["ball_l"][1], P["ball_r"][1]))
            drop = float(np.percentile(low, 10)) - float(sk.world["ball_l"][1])
        out = []
        for i in range(a, b):
            J = self.joints(i)
            pose = retarget(sk, self.facing_frame(i, yaw[i - a]), cal)
            pose.pelvis = np.array([0.0, (J["pelvis"][1] - stand) * k - drop, 0.0])
            st = Stance()
            st.free = True
            x, z = (J["pelvis"][0] - o[0]) * k, (J["pelvis"][2] - o[2]) * k
            out.append(Frame([], None, {}, root=(float(x), float(z), float(yaw[i - a])), stance=st, base=pose,
                             label=f"{ACTIONS.get(self.action[i], self.action[i])} / {self.posture[i]}"))
        return out


def catalog(limit=None):
    """Every labelled segment in the dataset: [(trial, action, start, end, posture)] (slow-ish: reads the labels)."""
    out = []
    for name in trials()[:limit]:
        part = name.split("_Setup")[0]
        rows = list(csv.DictReader(open(os.path.join(ROOT, "labels_csv", part, name + ".labels.csv"))))
        act = [_vote(r, "current_action") for r in rows]
        pos = [_vote(r, "detailed_posture") for r in rows]
        a, s = None, 0
        for i, x in enumerate(act + [None]):
            if x != a:
                if a is not None and i - s >= 6:
                    out.append((name, a, s, i, collections.Counter(pos[s:i]).most_common(1)[0][0]))
                a, s = x, i
    return out
