"""Realistic worker bodies (MakeHuman / MPFB, CC0) on the canonical skeleton.

tools/blender/mpfb_workers.py exports each worker's body as <id>_body.npz: the skinned mesh (Y-up cm, facing +Z,
left +X), UVs, skin weights per joint name and the joints' rest positions - already in the frt rest pose (T-pose,
palms down). Here the canonical Skeleton is built from those joint positions (so the solver works on the body's
true proportions) and the mesh is written as a UsdSkel skinned mesh. Work clothes are painted by body region from
the skin weights: hi-vis vest with reflective bands, shirt sleeves, work trousers, gloves, safety boots, skin, hair.

Bodies are looked up in FRT_BODIES, ~/.frt/bodies, then C:/OmniverseData/mpfb_bodies.
"""
import os

import numpy as np
from pxr import Gf, Sdf, UsdGeom, UsdShade, UsdSkel, Vt

from .skeleton import INDEX, NAMES, PARENT, Skeleton

DIRS = [os.environ.get("FRT_BODIES", ""), os.path.join(os.path.expanduser("~"), ".frt", "bodies"),
        r"C:/OmniverseData/mpfb_bodies"]
_CACHE = {}


def path_of(wid):
    if os.environ.get("FRT_NO_BODIES"):                          # tests: anthropometric skeletons only
        return None
    for d in DIRS:
        if d and os.path.isfile(os.path.join(d, f"{wid}_body.npz")):
            return os.path.join(d, f"{wid}_body.npz")
    return None


def load(wid):
    if wid in _CACHE:
        return _CACHE[wid]
    p = path_of(wid)
    if p is None:
        _CACHE[wid] = None
        return None
    d = dict(np.load(p, allow_pickle=False))
    d["joints"] = [("root" if str(j) == "Root" else str(j)) for j in d["joints"]]
    d["parents"] = [("root" if str(j) == "Root" else str(j)) for j in d["parents"]]
    _CACHE[wid] = d
    return d


def skeleton(d):
    """Canonical Skeleton from the body's joints (rest = T-pose, identity rotations, positions only)."""
    J = d["joints"]
    H, T = d["heads"].astype(float), d["tails"].astype(float)
    world = {n: H[J.index(n)].copy() for n in NAMES}
    world["root"] = np.zeros(3)
    tips = {"head_top": T[J.index("head")].copy()}
    pts = d["points"]
    tips["head_top"][1] = float(pts[:, 1].max())
    for s in ("l", "r"):
        for f in ("thumb", "index", "middle", "ring", "pinky"):
            tips[f"{f}_tip_{s}"] = T[J.index(f"{f}_03_{s}")].copy()
        tips[f"toe_{s}"] = T[J.index(f"ball_{s}")].copy()
        foot = world[f"foot_{s}"]
        near = pts[(np.abs(pts[:, 0] - foot[0]) < 6) & (pts[:, 1] < foot[1] + 2)]
        heel_z = float(near[:, 2].min()) if len(near) else foot[2] - 6.0
        tips[f"heel_{s}"] = np.array([foot[0], 2.5, heel_z])
    return Skeleton(world, tips)


# clothing regions by dominant joint
def _region(j):
    if j in ("head", "neck_01"):
        return "skin"
    if j.startswith(("spine", "clavicle")):
        return "hivis_yellow"
    if j.startswith(("upperarm", "lowerarm")):
        return "fabric_navy"
    if j.startswith(("hand", "thumb", "index", "middle", "ring", "pinky")):
        return "glove"
    if j.startswith(("foot", "ball")):
        return "boot"
    return "fabric_dark"                          # pelvis, thighs, calves: work trousers


def face_regions(d, sk, ppe=("vest", "gloves")):
    """Clothing by body geometry in the rest pose (T-pose): clean straight edges at the collar, waist, wrists and
    boot tops. Arms run along +-X, so 'beyond the wrist / shoulder' is a test on |x|."""
    pts = d["points"]
    counts, idx = d["counts"], d["indices"]
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    W = sk.world
    wrist_x = abs(W["hand_l"][0]) - 1.0
    shoulder_x = abs(W["upperarm_l"][0]) - 3.0
    neck_y = W["neck_01"][1] - 2.0
    waist_y = W["pelvis"][1] + 9.0
    boot_y = W["foot_l"][1] + 7.0
    head_c = W["head"]
    top = float(pts[:, 1].max())
    hl = top - head_c[1]
    sp = W["spine_02"][1]
    reg = []
    for a, n in zip(starts, counts):
        c = pts[idx[a:a + n]].mean(0)
        ax, y = abs(c[0]), c[1]
        if ax > wrist_x and y > W["pelvis"][1]:
            r = "glove" if "gloves" in ppe else "skin"
        elif ax > shoulder_x and y > W["pelvis"][1] + 10:
            r = "fabric_navy"                                          # sleeves
        elif y > neck_y:
            rel = (y - head_c[1]) / max(hl, 1e-6)
            r = "hair_dark" if (rel > 0.62 or (rel > 0.12 and c[2] < head_c[2] - 3.0)) else "skin"
        elif y > waist_y:
            r = "hivis_yellow" if "vest" in ppe else "fabric_navy"
            if r == "hivis_yellow" and (abs(y - (sp - 4.0)) < 1.6 or abs(y - (sp + 4.0)) < 1.6):
                r = "reflective"
        elif y < boot_y:
            r = "boot"
        else:
            r = "fabric_dark"                                          # work trousers
        reg.append(r)
    return reg


def vertex_normals(pts, counts, idx):
    n = np.zeros_like(pts, dtype=np.float64)
    a = 0
    for c in counts:
        f = idx[a:a + c]
        p = pts[f].astype(np.float64)
        fn = np.zeros(3)
        for k in range(c):                                       # Newell normal
            u, v = p[k], p[(k + 1) % c]
            fn += np.array([(u[1] - v[1]) * (u[2] + v[2]), (u[2] - v[2]) * (u[0] + v[0]), (u[0] - v[0]) * (u[1] + v[1])])
        n[f] += fn
        a += c
    L = np.linalg.norm(n, axis=1, keepdims=True)
    return n / np.maximum(L, 1e-9)


def dress_points(d, sk):
    """Clothes thickness on the body: safety boots (feet thickened, toes merged into a toe cap), gloves and
    trousers a few mm proud of the skin."""
    pts = d["points"].astype(np.float64).copy()
    nrm = vertex_normals(pts, d["counts"], d["indices"])
    W = sk.world
    boot_y = W["foot_l"][1] + 7.0
    wrist_x = abs(W["hand_l"][0]) - 1.0
    feet = pts[:, 1] < boot_y
    t = np.clip((boot_y - pts[:, 1]) / 4.0, 0.0, 1.0)              # boot top blends into the trouser leg
    pts[feet] += nrm[feet] * (0.9 * t[feet])[:, None]
    for s in ("l", "r"):                                         # toe cap: toes pulled into one rounded front
        ball = W[f"ball_{s}"]
        m = feet & (pts[:, 2] > ball[2]) & (np.sign(pts[:, 0]) == np.sign(ball[0]))
        pts[m, 0] = ball[0] + (pts[m, 0] - ball[0]) * 0.92
        pts[m, 1] = np.maximum(pts[m, 1], 0.4)
    hands = (np.abs(pts[:, 0]) > wrist_x) & (pts[:, 1] > W["pelvis"][1])
    pts[hands] += nrm[hands] * 0.12
    legs = (pts[:, 1] < W["pelvis"][1] + 9.0) & ~feet
    pts[legs] += nrm[legs] * 0.25
    return pts.astype(np.float32)


def write_body_mesh(stage, path, d, sk, lib, ppe=("vest", "gloves")):
    """The skinned body mesh under `path` (sibling of the Skeleton), one GeomSubset per clothing region."""
    m = UsdGeom.Mesh.Define(stage, path)
    pts = dress_points(d, sk)
    m.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(pts.astype(np.float32)))
    m.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(d["counts"].astype(np.int32)))
    m.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(d["indices"].astype(np.int32)))
    m.CreateSubdivisionSchemeAttr("catmullClark")
    m.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(*pts.min(0).tolist()), Gf.Vec3f(*pts.max(0).tolist())]))
    if len(d["uv"]):
        st = UsdGeom.PrimvarsAPI(m.GetPrim()).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray,
                                                              UsdGeom.Tokens.faceVarying)
        st.Set(Vt.Vec2fArray.FromNumpy((d["uv"] * 2.0).astype(np.float32)))    # fabric / skin grain scale
    # skin: map the body's joint names onto the canonical joint order
    J = d["joints"]
    remap = np.array([INDEX.get(n, 0) for n in J], np.int32)
    ji = remap[d["joint_indices"]]
    b = UsdSkel.BindingAPI.Apply(m.GetPrim())
    K = ji.shape[1]
    b.CreateJointIndicesPrimvar(False, K).Set(Vt.IntArray.FromNumpy(ji.reshape(-1).astype(np.int32)))
    b.CreateJointWeightsPrimvar(False, K).Set(Vt.FloatArray.FromNumpy(d["joint_weights"].reshape(-1).astype(np.float32)))
    b.CreateGeomBindTransformAttr(Gf.Matrix4d(1.0))
    regions = face_regions(d, sk, ppe)
    looks = sorted(set(regions))
    refs = lib.refs(looks)
    for look in looks:
        faces = [i for i, r in enumerate(regions) if r == look]
        sub = UsdGeom.Subset.Define(stage, f"{path}/{look}")
        sub.CreateElementTypeAttr("face")
        sub.CreateIndicesAttr(Vt.IntArray(faces))
        sub.CreateFamilyNameAttr("materialBind")
        UsdShade.MaterialBindingAPI.Apply(sub.GetPrim()).Bind(refs[look].material)
    UsdGeom.Subset.SetFamilyType(m, "materialBind", UsdGeom.Tokens.partition)
    m.CreateDisplayColorAttr([Gf.Vec3f(0.6, 0.55, 0.5)])
    return m
