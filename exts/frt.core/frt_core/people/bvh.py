"""BVH motion files (Xsens, CMU, ...): hierarchy, channels, and world joint positions / rotations per frame.

    clip = BvhClip(path, fps=24)            # resampled to 24 fps
    clip.positions(i) -> {joint: xyz}      # world positions (file units, here cm), including end sites as
                                            # "<joint>_end"
    clip.rotations(i) -> {joint: 3x3}      # world rotations
"""
import math

import numpy as np


def _rot(axis, deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    if axis == "X":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "Y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


class BvhClip:
    def __init__(self, path, fps=None):
        with open(path) as f:
            lines = f.read().split("\n")
        self.joints, self.parent, self.offset, self.channels, self.end = [], {}, {}, {}, {}
        stack, i = [], 0
        while i < len(lines):
            t = lines[i].strip().split()
            i += 1
            if not t:
                continue
            if t[0] in ("ROOT", "JOINT"):
                name = t[1]
                self.parent[name] = stack[-1] if stack else None
                self.joints.append(name)
                cur = name
            elif t[0] == "End":
                cur = ("end", stack[-1])
            elif t[0] == "{":
                stack.append(cur)
            elif t[0] == "}":
                stack.pop()
            elif t[0] == "OFFSET":
                v = np.array([float(x) for x in t[1:4]])
                if isinstance(cur, tuple):
                    self.end[cur[1]] = v
                else:
                    self.offset[cur] = v
            elif t[0] == "CHANNELS":
                self.channels[cur] = t[2:2 + int(t[1])]
            elif t[0] == "MOTION":
                break
        n = int(lines[i].split(":")[1])
        dt = float(lines[i + 1].split(":")[1])
        data = np.array([[float(x) for x in ln.split()] for ln in lines[i + 2:i + 2 + n] if ln.strip()])
        self.src_fps = 1.0 / dt
        if fps:                                              # resample by nearest frame (240 Hz -> 24 fps)
            idx = np.round(np.arange(0, len(data) - 1, self.src_fps / fps)).astype(int)
            data = data[idx]
            self.fps = float(fps)
        else:
            self.fps = self.src_fps
        self.data = data
        self.frames = len(data)
        self.col = {}
        k = 0
        for j in self.joints:
            self.col[j] = (k, self.channels[j])
            k += len(self.channels[j])

    def _local(self, j, row):
        k, ch = self.col[j]
        R = np.eye(3)
        p = self.offset[j].copy()
        for c, v in zip(ch, row[k:k + len(ch)]):
            if c.endswith("rotation"):
                R = R @ _rot(c[0], v)
            elif c.endswith("position"):
                p["XYZ".index(c[0])] = v
        return R, p

    def world(self, i):
        """({joint: R}, {joint: p}) for frame i; end sites as "<joint>_end" in the positions."""
        row = self.data[i]
        Rw, Pw = {}, {}
        for j in self.joints:
            R, p = self._local(j, row)
            par = self.parent[j]
            if par is None:
                Rw[j], Pw[j] = R, p
            else:
                Rw[j] = Rw[par] @ R
                Pw[j] = Pw[par] + Rw[par] @ p
            if j in self.end:
                Pw[j + "_end"] = Pw[j] + Rw[j] @ self.end[j]
        return Rw, Pw


# Xsens MVN skeleton -> canonical joint names (mocap.retarget input)
XSENS = {"pelvis": "Hips", "spine_01": "Chest", "spine_02": "Chest3", "spine_03": "Chest4", "neck_01": "Neck",
         "head": "Head", "head_top": "Head_end"}
for _s, _S in (("l", "Left"), ("r", "Right")):
    XSENS.update({f"clavicle_{_s}": f"{_S}Collar", f"upperarm_{_s}": f"{_S}Shoulder", f"lowerarm_{_s}": f"{_S}Elbow",
                  f"hand_{_s}": f"{_S}Wrist", f"middle_{_s}": f"{_S}Wrist_end", f"thigh_{_s}": f"{_S}Hip",
                  f"calf_{_s}": f"{_S}Knee", f"foot_{_s}": f"{_S}Ankle", f"ball_{_s}": f"{_S}Toe",
                  f"toe_{_s}": f"{_S}Toe_end"})


def xsens_joints(clip, i):
    """Canonical-named joint positions of frame i (cm), with index / pinky points made from the wrist's rotation
    (Xsens has no fingers): the thumb side is +Z of the hand in the T-pose (palms down)."""
    Rw, Pw = clip.world(i)
    M = {k: Pw[v] for k, v in XSENS.items() if v in Pw}
    for s, S in (("l", "Left"), ("r", "Right")):
        side = Rw[f"{S}Wrist"] @ np.array([0.0, 0.0, 3.0])
        M[f"index_{s}"] = M[f"middle_{s}"] + side
        M[f"pinky_{s}"] = M[f"middle_{s}"] - side
    return M
