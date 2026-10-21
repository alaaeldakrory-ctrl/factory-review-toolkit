"""Forward kinematics and rotation helpers for the canonical skeleton (numpy, 3x3 matrices).

Rest pose has identity joint rotations, so a joint's world position is its parent's world position plus the parent's
world rotation applied to the rest offset. A Pose holds local rotations (and the pelvis offset from its rest spot).
"""
import math

import numpy as np

from .skeleton import NAMES, PARENT

I3 = np.eye(3)


def rx(deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def ry(deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rz(deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def axis_angle(axis, rad):
    a = np.asarray(axis, float)
    n = np.linalg.norm(a)
    if n < 1e-12 or abs(rad) < 1e-12:
        return I3.copy()
    k = a / n
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return I3 + math.sin(rad) * K + (1 - math.cos(rad)) * K @ K


def unit(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def frame_rot(rest_axis, rest_ref, new_axis, new_ref):
    """Rotation R with R @ rest_axis = new_axis and R @ rest_ref in the plane of (new_axis, new_ref)."""
    def basis(a, r):
        a = unit(a)
        r = np.asarray(r, float) - a * np.dot(r, a)
        if np.linalg.norm(r) < 1e-9:                                        # reference parallel: any perpendicular
            r = np.cross(a, [0, 1, 0] if abs(a[1]) < 0.9 else [1, 0, 0])
        r = unit(r)
        return np.column_stack([a, r, np.cross(a, r)])
    return basis(new_axis, new_ref) @ basis(rest_axis, rest_ref).T


def angle(a, b):
    return math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(unit(a), unit(b)))))))


def quat_from_matrix(R):
    """(w, x, y, z) from a rotation matrix."""
    t = R[0, 0] + R[1, 1] + R[2, 2]
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        return (0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s)
    i = int(np.argmax([R[0, 0], R[1, 1], R[2, 2]]))
    if i == 0:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        return ((R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s)
    if i == 1:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        return ((R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s)
    s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
    return ((R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s)


class Pose:
    """Local joint rotations (3x3) + pelvis offset (cm, from its rest spot) + root placement in the scene."""

    def __init__(self, rot=None, pelvis=None):
        self.rot = dict(rot or {})
        self.pelvis = np.zeros(3) if pelvis is None else np.asarray(pelvis, float)

    def copy(self):
        return Pose({k: v.copy() for k, v in self.rot.items()}, self.pelvis.copy())

    def local(self, j):
        return self.rot.get(j, I3)


def fk(sk, pose, joints=None):
    """World rotations and positions of every joint (or of `joints` and their ancestors only)."""
    need = NAMES if joints is None else _ancestors(joints)
    R, P = {}, {}
    for j in NAMES:
        if j not in need:
            continue
        p = PARENT[j]
        if p is None:
            R[j], P[j] = pose.local(j), sk.world[j].copy()
            continue
        off = sk.world[j] - sk.world[p]
        if j == "pelvis":
            off = off + pose.pelvis
        P[j] = P[p] + R[p] @ off
        R[j] = R[p] @ pose.local(j)
    return R, P


_ANC = {}


def _ancestors(joints):
    key = tuple(sorted(joints))
    if key not in _ANC:
        out = set()
        for j in joints:
            while j:
                out.add(j)
                j = PARENT[j]
        _ANC[key] = out
    return _ANC[key]


# segment mass fractions (Dempster / Winter) for the centre of mass, attached to the joint that carries them
MASS = {"pelvis": 0.142, "spine_01": 0.139, "spine_02": 0.106, "spine_03": 0.10, "head": 0.081,
        "upperarm_l": 0.028, "lowerarm_l": 0.016, "hand_l": 0.006, "upperarm_r": 0.028, "lowerarm_r": 0.016,
        "hand_r": 0.006, "thigh_l": 0.10, "calf_l": 0.0465, "foot_l": 0.0145, "thigh_r": 0.10, "calf_r": 0.0465,
        "foot_r": 0.0145}
CHILD = {"pelvis": "spine_01", "spine_01": "spine_02", "spine_02": "spine_03", "spine_03": "neck_01",
         "head": None, "upperarm_l": "lowerarm_l", "lowerarm_l": "hand_l", "hand_l": None,
         "upperarm_r": "lowerarm_r", "lowerarm_r": "hand_r", "hand_r": None, "thigh_l": "calf_l",
         "calf_l": "foot_l", "foot_l": "ball_l", "thigh_r": "calf_r", "calf_r": "foot_r", "foot_r": "ball_r"}


def com(P):
    """Whole-body centre of mass from joint world positions (segment midpoints, Dempster fractions)."""
    tot, acc = 0.0, np.zeros(3)
    for j, m in MASS.items():
        c = CHILD[j]
        mid = (P[j] + P[c]) / 2 if c and c in P else P[j]
        acc += m * mid
        tot += m
    return acc / tot
