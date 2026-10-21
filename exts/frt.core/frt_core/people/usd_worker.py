"""Write a worker into USD as a UsdSkel character: SkelRoot -> Skeleton (canonical joints, T-pose rest) + body
meshes bound to it + a SkelAnimation. Until the MakeHuman / MPFB bodies are exported, the body is a generated
mannequin on the same skeleton (one rigid segment per joint: capsule limbs, vest torso, gloves, boots), so
everything built on top - solver, motions, rules - already runs on the real joints.

    write_worker(stage, "/World/People/W50M", WORKERS["W50M"], lib, pose=relaxed(sk))
"""
import math

import numpy as np
from pxr import Gf, Sdf, UsdGeom, UsdSkel, Vt

from ..geom import Shapes
from .skeleton import INDEX, NAMES, PARENT, Skeleton, path as jpath


def _capsule(s, a, b, r0, r1, mat, sides=16):
    """Limb segment from a to b, radius r0 -> r1, rounded ends."""
    s.lathe(tuple(a), tuple(b), [(0.0, r0 * 0.35), (0.04, r0 * 0.8), (0.12, r0), (0.88, r1), (0.96, r1 * 0.8),
                                 (1.0, r1 * 0.35)], mat, sides=sides)


def _ellipsoid(s, c, rx, ry, rz, mat, rings=10, sides=18):
    """Ellipsoid as a lathe about Y, then squashed in Z (rz / rx) about its centre."""
    prof = [(k / rings, math.sin(math.pi * k / rings) * rx + 1e-3) for k in range(rings + 1)]
    P = s._group(mat)[0]
    n0 = len(P)
    s.lathe((c[0], c[1] - ry, c[2]), (c[0], c[1] + ry, c[2]), prof, mat, sides=sides)
    if abs(rz - rx) > 1e-6:
        for i in range(n0, len(P)):
            x, y, z = P[i]
            P[i] = (x, y, c[2] + (z - c[2]) * rz / rx)


def body_parts(sk, worker):
    """{joint: Shapes} - the mannequin segment carried by each joint (rest pose, world coordinates)."""
    W, d = sk.world, worker.dims
    gloves = "gloves" in worker.ppe
    vest = "vest" in worker.ppe
    parts = {}

    def S(j):
        return parts.setdefault(j, Shapes())
    pel = W["pelvis"]
    S("pelvis").rbox(tuple(pel + [0, -3.0, 1.0]), (d["hip_breadth"] * 0.95, 16.0, 21.0), 6.0, "fabric_dark", seg=3)
    shirt = "hivis_yellow" if vest else "fabric_navy"
    S("spine_01").rbox(tuple((W["spine_01"] + W["spine_02"]) / 2 + [0, 0, 1.5]), (d["hip_breadth"] * 0.9,
                       W["spine_02"][1] - W["spine_01"][1] + 6, 20.0), 7.0, shirt, seg=3)
    S("spine_02").rbox(tuple((W["spine_02"] + W["spine_03"]) / 2 + [0, 0, 2.0]), (d["shoulder_breadth"] * 0.82,
                       W["spine_03"][1] - W["spine_02"][1] + 6, 22.0), 8.0, shirt, seg=3)
    S("spine_03").rbox(tuple(W["spine_03"] + [0, 4.0, 1.0]), (d["shoulder_breadth"] * 0.88, 14.0, 20.0), 7.0, shirt, seg=3)
    if vest:                                                  # reflective bands on the vest
        for y in (W["spine_02"][1] - 2.0, W["spine_02"][1] + 5.0):
            S("spine_02").rbox((0.0, y, 2.0), (d["shoulder_breadth"] * 0.83, 1.8, 22.3), 7.5, "reflective", seg=3)
    _capsule(S("neck_01"), W["neck_01"], W["head"], 5.5, 5.0, "skin")
    hl = d["head_length"]
    _ellipsoid(S("head"), W["head"] + [0, hl * 0.42, 1.0], hl * 0.39, hl * 0.55, hl * 0.5, "skin")
    _ellipsoid(S("head"), W["head"] + [0, hl * 0.62, -0.5], hl * 0.41, hl * 0.4, hl * 0.52, "hair_dark")   # hair cap
    S("head").rbox(tuple(W["head"] + [0, hl * 0.5, hl * 0.47]), (hl * 0.55, 2.2, 1.2), 0.5, "plastic_black", seg=1)  # glasses
    for s in ("l", "r"):
        _capsule(S(f"clavicle_{s}"), W[f"clavicle_{s}"], W[f"upperarm_{s}"], 4.5, 5.5, shirt)
        _capsule(S(f"upperarm_{s}"), W[f"upperarm_{s}"], W[f"lowerarm_{s}"], 5.6, 4.6, "fabric_navy")
        _capsule(S(f"lowerarm_{s}"), W[f"lowerarm_{s}"], W[f"hand_{s}"], 4.4, 3.4, "fabric_navy")
        glove = "glove" if gloves else "skin"
        sx = 1.0 if s == "l" else -1.0
        hd = W[f"hand_{s}"]
        S(f"hand_{s}").rbox(tuple(hd + [sx * d["hand_length"] * 0.24, -0.3, 0.0]), (d["hand_length"] * 0.48, 2.8, 8.2),
                            1.2, glove, seg=2)
        for f in ("thumb", "index", "middle", "ring", "pinky"):
            for i in (1, 2, 3):
                j = f"{f}_0{i}_{s}"
                end = W[f"{f}_0{i + 1}_{s}"] if i < 3 else sk.tips[f"{f}_tip_{s}"]
                r = 1.05 if f != "pinky" else 0.85
                _capsule(S(j), W[j], end, r, r * 0.9, glove, sides=10)
        _capsule(S(f"thigh_{s}"), W[f"thigh_{s}"], W[f"calf_{s}"], 8.0, 5.8, "fabric_dark")
        _capsule(S(f"calf_{s}"), W[f"calf_{s}"], W[f"foot_{s}"] + [0, 3.0, 0], 5.6, 4.0, "fabric_dark")
        heel, ball, toe = sk.tips[f"heel_{s}"], W[f"ball_{s}"], sk.tips[f"toe_{s}"]
        S(f"foot_{s}").rbox(tuple((heel + ball) / 2 + [0, 3.5, 0]), (9.5, 9.0, float(ball[2] - heel[2]) + 3), 3.0,
                            "boot", seg=2)
        S(f"ball_{s}").rbox(tuple((ball + toe) / 2 + [0, 2.5, 0]), (9.5, 5.5, float(toe[2] - ball[2]) + 3), 2.4,
                            "boot", seg=2)
    return parts


def _quat_axis_angle(axis, deg):
    a = np.radians(deg) / 2
    ax = np.asarray(axis, float) / np.linalg.norm(axis)
    return Gf.Quatf(float(np.cos(a)), *(float(c) for c in ax * np.sin(a)))


def relaxed(sk):
    """Standing at ease: arms down by the sides, elbows and fingers slightly bent (local joint rotations)."""
    q = {}
    for s, sx in (("l", 1.0), ("r", -1.0)):
        q[f"upperarm_{s}"] = _quat_axis_angle((0, 0, 1), -sx * 78) * _quat_axis_angle((1, 0, 0), 0)
        q[f"lowerarm_{s}"] = _quat_axis_angle((0, 1, 0), -sx * 12)
        q[f"hand_{s}"] = _quat_axis_angle((1, 0, 0), sx * 0)
        for f in ("index", "middle", "ring", "pinky"):
            for i in (1, 2, 3):
                q[f"{f}_0{i}_{s}"] = _quat_axis_angle((0, 0, 1), -sx * 14)
        q[f"thumb_01_{s}"] = _quat_axis_angle((0, 1, 0), sx * 10)
    return q


def pose_arrays(sk, pose):
    """(translations, rotations) UsdSkel arrays for a pose: a kinematics.Pose (3x3 local rotations + pelvis offset)
    or a dict joint -> Gf.Quatf; joints left out keep their rest orientation."""
    from .kinematics import Pose, quat_from_matrix
    loc = sk.local
    tr = [Gf.Vec3f(*map(float, loc[n])) for n in NAMES]
    if isinstance(pose, Pose):
        i = NAMES.index("pelvis")
        tr[i] = Gf.Vec3f(*map(float, loc["pelvis"] + pose.pelvis))
        rot = []
        for n in NAMES:
            w, x, y, z = quat_from_matrix(pose.local(n))
            rot.append(Gf.Quatf(float(w), float(x), float(y), float(z)))
    else:
        pose = pose or {}
        rot = [pose.get(n, Gf.Quatf(1, 0, 0, 0)) for n in NAMES]
    return Vt.Vec3fArray(tr), Vt.QuatfArray(rot)


def write_worker(stage, path, worker, lib, pose=None, sk=None):
    """SkelRoot at path with Skeleton, mannequin body (rigidly bound) and a SkelAnimation holding `pose`
    (joint -> Gf.Quatf local rotation; joints left out keep their rest orientation)."""
    sk = sk or Skeleton.for_worker(worker)
    root = UsdSkel.Root.Define(stage, path)
    skel = UsdSkel.Skeleton.Define(stage, f"{path}/Skeleton")
    skel.CreateJointsAttr(Vt.TokenArray([jpath(n) for n in NAMES]))
    bind = [Gf.Matrix4d().SetTranslate(Gf.Vec3d(*map(float, sk.world[n]))) for n in NAMES]
    loc = sk.local
    rest = [Gf.Matrix4d().SetTranslate(Gf.Vec3d(*map(float, loc[n]))) for n in NAMES]
    skel.CreateBindTransformsAttr(Vt.Matrix4dArray(bind))
    skel.CreateRestTransformsAttr(Vt.Matrix4dArray(rest))
    anim = UsdSkel.Animation.Define(stage, f"{path}/Skeleton/Pose")
    anim.CreateJointsAttr(Vt.TokenArray([jpath(n) for n in NAMES]))
    tr, rot = pose_arrays(sk, pose)
    anim.CreateTranslationsAttr(tr)
    anim.CreateRotationsAttr(rot)
    anim.CreateScalesAttr(Vt.Vec3hArray([Gf.Vec3h(1, 1, 1)] * len(NAMES)))
    UsdSkel.BindingAPI.Apply(skel.GetPrim()).CreateAnimationSourceRel().SetTargets([anim.GetPath()])
    UsdSkel.BindingAPI.Apply(root.GetPrim()).CreateSkeletonRel().SetTargets([skel.GetPath()])
    UsdGeom.Scope.Define(stage, f"{path}/Body")
    for j, shp in body_parts(sk, worker).items():
        meshes = shp.build(stage, f"{path}/Body/{j}", lib.refs(list(shp.groups)))
        for m in meshes:
            b = UsdSkel.BindingAPI.Apply(m.GetPrim())
            b.CreateJointIndicesPrimvar(True, 1).Set(Vt.IntArray([INDEX[j]]))
            b.CreateJointWeightsPrimvar(True, 1).Set(Vt.FloatArray([1.0]))
            b.CreateGeomBindTransformAttr(Gf.Matrix4d(1.0))
    prim = root.GetPrim()
    prim.CreateAttribute("frt:worker", Sdf.ValueTypeNames.String).Set(worker.id)
    prim.CreateAttribute("frt:stature", Sdf.ValueTypeNames.Float).Set(float(worker.stature))
    return root, sk
