"""Motion capture on the canonical skeleton: Mixamo FBX clips (read by fbx_anim.py) retargeted by bone direction.

Each canonical bone is turned so it points where the matching Mixamo bone points, with a reference direction for
twist (hip line / shoulder line for the trunk, the elbow / knee bend plane for the limbs, index-pinky line for the
hand). This does not depend on either rig's rest pose, and it works for any body size because only directions are
copied: lengths stay the worker's own. The pelvis offset (bob, sway, surge) is copied scaled by leg length.

A walking clip becomes a cycle sampled by gait phase (0..1 over two steps, phase 0 = right toe-off), so the upper
body, pelvis rotation, arm swing and head carriage come from a real person while the feet stay exactly on the
planned footsteps (the solver's leg IK).

    cyc = cycle("Walking"); pose = cyc.pose(sk, phase)     # kinematics.Pose with the pelvis offset
"""
import math
import os

import numpy as np

from .kinematics import I3, Pose, frame_rot, unit

MIXAMO_DIR = os.environ.get("FRT_MIXAMO", r"C:/OmniverseData/mixamo/fbx")
UP = np.array([0.0, 1.0, 0.0])
FWD = np.array([0.0, 0.0, 1.0])
LEFT = np.array([1.0, 0.0, 0.0])

MAP = {"pelvis": "Hips", "spine_01": "Spine", "spine_02": "Spine1", "spine_03": "Spine2", "neck_01": "Neck",
       "head": "Head", "head_top": "HeadTop_End"}
for _s, _S in (("l", "Left"), ("r", "Right")):
    MAP.update({f"clavicle_{_s}": f"{_S}Shoulder", f"upperarm_{_s}": f"{_S}Arm", f"lowerarm_{_s}": f"{_S}ForeArm",
                f"hand_{_s}": f"{_S}Hand", f"index_{_s}": f"{_S}HandIndex1", f"middle_{_s}": f"{_S}HandMiddle1",
                f"pinky_{_s}": f"{_S}HandPinky1", f"thigh_{_s}": f"{_S}UpLeg", f"calf_{_s}": f"{_S}Leg",
                f"foot_{_s}": f"{_S}Foot", f"ball_{_s}": f"{_S}ToeBase", f"toe_{_s}": f"{_S}Toe_End"})


def _hinge(R_parent, rest_dir, new_dir):
    """World rotation for a bone that only bends relative to its parent (minimal rotation, no added twist)."""
    a = unit(R_parent @ rest_dir)
    b = unit(new_dir)
    v = np.cross(a, b)
    s, c = np.linalg.norm(v), float(np.dot(a, b))
    if s < 1e-9:
        return R_parent.copy()
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]]) / s
    ang = math.atan2(s, c)
    Rm = I3 + math.sin(ang) * K + (1 - math.cos(ang)) * K @ K
    return Rm @ R_parent


def retarget(sk, M):
    """Canonical local rotations from Mixamo world joint positions M (dict canonical name -> xyz, cm)."""
    W = sk.world
    R = {}
    hip_left = unit(M["thigh_l"] - M["thigh_r"])
    sh_left = unit(M["upperarm_l"] - M["upperarm_r"])

    def rest(a, b):
        return unit((W[b] if b in W else sk.tips[b]) - W[a])
    # trunk: axis along the chain, twist from the hip line (low) blending to the shoulder line (high)
    R["pelvis"] = frame_rot(rest("pelvis", "spine_01"), LEFT, M["spine_01"] - M["pelvis"], hip_left)
    for j, nxt, k in (("spine_01", "spine_02", 0.25), ("spine_02", "spine_03", 0.6), ("spine_03", "neck_01", 1.0)):
        ref = unit(hip_left * (1 - k) + sh_left * k)
        R[j] = frame_rot(rest(j, nxt), LEFT, M[nxt] - M[j], ref)
    R["neck_01"] = frame_rot(rest("neck_01", "head"), LEFT, M["head"] - M["neck_01"], sh_left)
    R["head"] = frame_rot(unit(sk.tips["head_top"] - W["head"]), LEFT, M["head_top"] - M["head"], sh_left)
    fwd = unit(np.cross(hip_left, UP))
    for s, sx in (("l", 1.0), ("r", -1.0)):
        R[f"clavicle_{s}"] = frame_rot(rest(f"clavicle_{s}", f"upperarm_{s}"), UP,
                                       M[f"upperarm_{s}"] - M[f"clavicle_{s}"], UP)
        ua = M[f"lowerarm_{s}"] - M[f"upperarm_{s}"]
        fa = M[f"hand_{s}"] - M[f"lowerarm_{s}"]
        bend = fa - unit(ua) * np.dot(fa, unit(ua))
        ref = bend if np.linalg.norm(bend) > 0.08 * np.linalg.norm(fa) else R["spine_03"] @ FWD
        R[f"upperarm_{s}"] = frame_rot(rest(f"upperarm_{s}", f"lowerarm_{s}"), FWD, ua, ref)
        R[f"lowerarm_{s}"] = _hinge(R[f"upperarm_{s}"], rest(f"lowerarm_{s}", f"hand_{s}"), fa)
        R[f"hand_{s}"] = frame_rot(rest(f"hand_{s}", f"middle_01_{s}"), FWD, M[f"middle_{s}"] - M[f"hand_{s}"],
                                   M[f"index_{s}"] - M[f"pinky_{s}"])
        th = M[f"calf_{s}"] - M[f"thigh_{s}"]
        R[f"thigh_{s}"] = frame_rot(rest(f"thigh_{s}", f"calf_{s}"), FWD, th, fwd)
        R[f"calf_{s}"] = _hinge(R[f"thigh_{s}"], rest(f"calf_{s}", f"foot_{s}"), M[f"foot_{s}"] - M[f"calf_{s}"])
        R[f"foot_{s}"] = frame_rot(rest(f"foot_{s}", f"ball_{s}"), LEFT, M[f"ball_{s}"] - M[f"foot_{s}"], hip_left)
    from .skeleton import PARENT
    out = Pose()
    for j, Rw in R.items():
        p = PARENT[j]
        while p is not None and p not in R:
            p = PARENT[p]
        Rp = R[p] if p is not None else I3
        out.rot[j] = Rp.T @ Rw
    return out


class Cycle:
    """A looping clip (walk) sampled by phase; positions made in place (the clip's own travel removed)."""

    def __init__(self, name, path=None):
        from .fbx_anim import FbxClip
        clip = FbxClip(path or os.path.join(MIXAMO_DIR, name + ".fbx"))
        pre = next(n.split(":")[0] + ":" for n in clip.names if n.endswith("Hips"))
        self.n = clip.frames
        frames = []
        for f in range(self.n):
            g = clip.global_mats(f)
            frames.append({k: np.array(g[pre + v].ExtractTranslation(), float) for k, v in MAP.items()
                           if pre + v in g})
        hips = np.array([fr["pelvis"] for fr in frames])
        t = np.arange(self.n)
        # in-place: remove the straight-line travel of the hips (least squares) - what is left is sway / bob / surge
        A = np.column_stack([t, np.ones_like(t)])
        self.travel = {}
        for ax in (0, 2):
            k, c = np.linalg.lstsq(A, hips[:, ax], rcond=None)[0]
            self.travel[ax] = k
            for fr in frames:
                pass
            for i, fr in enumerate(frames):
                for key in fr:
                    fr[key] = fr[key].copy()
                    fr[key][ax] -= k * i + c
        self.frames = frames
        self.hip_height = float(np.mean(hips[:, 1]))
        self.speed = math.hypot(self.travel[0], self.travel[2]) * clip.fps
        # phase 0 = right toe-off: the frame the right foot starts to move forward fastest after standing still
        rz = np.array([fr["foot_r"][2] for fr in frames])
        v = np.gradient(rz)
        stand = v < 0.25 * v.max() * 0.0 + np.percentile(v, 35)
        start = next((i for i in range(1, self.n) if stand[i - 1] and not stand[i]), 0)
        self.offset = start
        self._cache = {}

    def sample(self, phase):
        """Mixamo joint positions at gait phase (linear between frames, looping)."""
        x = (self.offset + (phase % 1.0) * self.n) % self.n
        i0 = int(math.floor(x))
        i1 = (i0 + 1) % self.n
        u = x - i0
        a, b = self.frames[i0], self.frames[i1]
        return {k: a[k] * (1 - u) + b[k] * u for k in a}

    def pose(self, sk, phase, sway=1.0):
        """Canonical Pose at gait phase: retargeted rotations + pelvis offset scaled to this body."""
        key = (id(sk), round(phase % 1.0, 3))
        if key in self._cache:
            return self._cache[key]
        M = self.sample(phase)
        p = retarget(sk, M)
        scale = (sk.world["pelvis"][1]) / max(self.hip_height, 1.0)
        off = (M["pelvis"] - np.array([0.0, self.hip_height, 0.0])) * scale
        off[0] *= sway
        p.pelvis = np.array([off[0], off[1], off[2]])
        self._cache[key] = p
        return p


_CYCLES = {}


def cycle(name):
    if name not in _CYCLES:
        _CYCLES[name] = Cycle(name)
    return _CYCLES[name]


def available(name="Walking"):
    return os.path.isfile(os.path.join(MIXAMO_DIR, name + ".fbx"))
