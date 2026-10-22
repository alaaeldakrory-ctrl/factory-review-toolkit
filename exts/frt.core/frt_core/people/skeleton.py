"""The canonical worker skeleton: game-engine biped layout (the MPFB "game engine" rig and the common UE mannequin
naming), so MakeHuman bodies, Mixamo clips and our solver all speak the same joints. Every motion is authored once
against it and plays on every worker.

Frame: Y up, cm, the worker faces +Z, their left is +X. Rest pose = T-pose with identity joint rotations (world
positions only), which keeps the IK simple; MPFB / mocap rigs map onto it by joint name (retarget.py).

    sk = Skeleton.for_worker(WORKERS["W05F"]); sk.world["hand_l"]
"""
import numpy as np

FINGERS = ("thumb", "index", "middle", "ring", "pinky")


def _joint_list():
    j = [("root", None), ("pelvis", "root"), ("spine_01", "pelvis"), ("spine_02", "spine_01"),
         ("spine_03", "spine_02"), ("neck_01", "spine_03"), ("head", "neck_01")]
    for s in ("l", "r"):
        j += [(f"clavicle_{s}", "spine_03"), (f"upperarm_{s}", f"clavicle_{s}"), (f"lowerarm_{s}", f"upperarm_{s}"),
              (f"hand_{s}", f"lowerarm_{s}")]
        for f in FINGERS:
            j += [(f"{f}_01_{s}", f"hand_{s}"), (f"{f}_02_{s}", f"{f}_01_{s}"), (f"{f}_03_{s}", f"{f}_02_{s}")]
    for s in ("l", "r"):
        j += [(f"thigh_{s}", "pelvis"), (f"calf_{s}", f"thigh_{s}"), (f"foot_{s}", f"calf_{s}"), (f"ball_{s}", f"foot_{s}")]
    return j


JOINTS = _joint_list()
NAMES = [n for n, _ in JOINTS]
PARENT = dict(JOINTS)
INDEX = {n: i for i, n in enumerate(NAMES)}


def path(name):
    """UsdSkel joint path ('root/pelvis/spine_01/...')."""
    parts = []
    while name:
        parts.append(name)
        name = PARENT[name]
    return "/".join(reversed(parts))


class Skeleton:
    def __init__(self, world, tips):
        self.world = world          # joint -> rest world position (np.array 3)
        self.tips = tips            # end points without joints: head top, finger tips, toe tips

    @property
    def local(self):
        return {n: self.world[n] - (self.world[PARENT[n]] if PARENT[n] else 0.0) for n in NAMES}

    def length(self, a, b):
        return float(np.linalg.norm(self.world[b] - self.world[a]))

    @classmethod
    def for_worker(cls, worker, prefer_body=True):
        """The worker's skeleton: from their realistic body (MPFB export) when there is one, else built from the
        anthropometric dimensions."""
        if prefer_body:
            from . import body as B
            d = B.load(worker.id)
            if d is not None:
                return B.skeleton(d)
        return cls.anthropometric(worker)

    @classmethod
    def anthropometric(cls, worker):
        d = worker.dims
        P = {}
        hip_y = d["hip_height"] - 1.5                      # hip joint centres ~ at the greater trochanter
        knee_y = d["knee_height"] - 4.0
        ank_y = d["ankle_height"]
        sh_y = d["shoulder_height"] - 4.0                  # glenohumeral centre below the acromion
        hip_x = d["hip_breadth"] * 0.27
        P["root"] = np.zeros(3)
        P["pelvis"] = np.array([0.0, hip_y + 6.0, -1.0])
        chest = sh_y - 6.0 - P["pelvis"][1]
        P["spine_01"] = np.array([0.0, P["pelvis"][1] + chest * 0.18, -2.0])
        P["spine_02"] = np.array([0.0, P["pelvis"][1] + chest * 0.48, -2.5])
        P["spine_03"] = np.array([0.0, P["pelvis"][1] + chest * 0.80, -2.0])
        P["neck_01"] = np.array([0.0, sh_y + 4.0, -2.0])
        P["head"] = np.array([0.0, d["eye_height"] - 8.0, -0.5])          # atlanto-occipital pivot
        tips = {"head_top": np.array([0.0, d["stature"], 0.0])}
        hl = d["hand_length"]
        for s, sx in (("l", 1.0), ("r", -1.0)):
            P[f"clavicle_{s}"] = np.array([sx * 2.5, sh_y - 1.5, -0.5])
            P[f"upperarm_{s}"] = np.array([sx * (d["shoulder_breadth"] / 2 - 3.0), sh_y, -1.0])
            P[f"lowerarm_{s}"] = P[f"upperarm_{s}"] + np.array([sx * d["upper_arm"], 0.0, 0.0])
            P[f"hand_{s}"] = P[f"lowerarm_{s}"] + np.array([sx * d["forearm"], 0.0, 0.0])
            k = hl / 19.0
            base = {"thumb": (0.12, -1.0, 2.6), "index": (0.47, 0.0, 2.3), "middle": (0.47, 0.0, 0.7),
                    "ring": (0.45, 0.0, -0.9), "pinky": (0.42, -0.3, -2.4)}
            segs = {"thumb": (0.20, 0.17, 0.14), "index": (0.24, 0.14, 0.11), "middle": (0.26, 0.16, 0.12),
                    "ring": (0.24, 0.15, 0.11), "pinky": (0.19, 0.11, 0.10)}
            for f in FINGERS:
                bx, by, bz = base[f]
                p = P[f"hand_{s}"] + np.array([sx * bx * hl, by * k, bz * k])
                dirn = np.array([sx * 0.6, 0.0, 0.8]) if f == "thumb" else np.array([sx, 0.0, 0.0])
                dirn = dirn / np.linalg.norm(dirn)
                for i, L in enumerate(segs[f]):
                    P[f"{f}_0{i + 1}_{s}"] = p
                    p = p + dirn * L * hl
                tips[f"{f}_tip_{s}"] = p
            P[f"thigh_{s}"] = np.array([sx * hip_x, hip_y, 0.0])
            P[f"calf_{s}"] = np.array([sx * hip_x, knee_y, 1.0])
            P[f"foot_{s}"] = np.array([sx * hip_x, ank_y, 0.0])
            P[f"ball_{s}"] = np.array([sx * hip_x, 2.0, d["foot_length"] * 0.55])
            tips[f"toe_{s}"] = np.array([sx * hip_x, 1.5, d["foot_length"] * 0.78])
            tips[f"heel_{s}"] = np.array([sx * hip_x, 2.5, -d["foot_length"] * 0.22])
        return cls(P, tips)
