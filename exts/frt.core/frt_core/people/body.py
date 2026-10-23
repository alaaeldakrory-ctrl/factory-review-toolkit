"""Realistic worker bodies (MakeHuman / MPFB, CC0) on the canonical skeleton.

tools/blender/mpfb_workers.py exports each worker's body as <id>_body.npz: the skinned mesh (Y-up cm, facing +Z,
left +X), UVs, skin weights per joint name and the joints' rest positions - already in the frt rest pose (T-pose,
palms down). Here the canonical Skeleton is built from those joint positions (so the solver works on the body's
true proportions) and the mesh is written as a UsdSkel skinned mesh. Work clothes are painted by body region from
the skin weights: hi-vis vest with reflective bands, shirt sleeves, work trousers, gloves, safety boots, skin, hair.

Bodies are looked up in FRT_BODIES, <repo>/data/bodies (the offline bundle), ~/.frt/bodies, then
C:/OmniverseData/mpfb_bodies.
"""
import os

import numpy as np
from pxr import Gf, Sdf, UsdGeom, UsdShade, UsdSkel, Vt

from .skeleton import INDEX, NAMES, PARENT, Skeleton

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
DIRS = [os.environ.get("FRT_BODIES", ""), os.path.join(_REPO, "data", "bodies"),
        os.path.join(os.path.expanduser("~"), ".frt", "bodies"),
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
    d["wid"] = wid
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


def _wid(d):
    return str(d.get("wid", "body")).lower()


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


# ---------------------------------------------------------------------------------------------- outfit texture
OUTFIT_VERSION = 1
REGIONS = ("skin", "hair_dark", "hivis_yellow", "reflective", "fabric_navy", "fabric_dark", "glove", "boot")


def classify(P, sk, ppe=("vest", "gloves")):
    """Garment at rest-pose (T-pose) body points P (n x 3) -> index into REGIONS. Smooth geometric rules, so the
    garment edges are clean lines whatever the mesh topology: cuffs at the wrists, a sleeveless vest with an armhole
    through the shoulder joint, two reflective hoops and two bands over the shoulders, V neck, waist, boot tops."""
    W = sk.world
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    ax = np.abs(x)
    pel = W["pelvis"][1]
    wrist_x = abs(W["hand_l"][0]) - 1.0
    sh_x = abs(W["upperarm_l"][0])
    neck_y = W["neck_01"][1] - 1.5
    waist_y = pel + 9.0
    boot_y = W["foot_l"][1] + 7.0
    head = W["head"]
    hl = float(P[:, 1].max()) - head[1]
    sp = W["spine_02"][1]
    out = np.full(len(P), REGIONS.index("fabric_dark"))                       # work trousers
    out[y < boot_y] = REGIONS.index("boot")
    upper = y > waist_y
    torso = upper & (ax < sh_x - 1.0 - np.clip((W["upperarm_l"][1] - y) * 0.15, 0, 3))
    out[upper] = REGIONS.index("fabric_navy")                                  # shirt (sleeves; torso without a vest)
    if "vest" in ppe:
        out[torso] = REGIONS.index("hivis_yellow")
        band = torso & ((np.abs(y - (sp - 5.0)) < 1.6) | (np.abs(y - (sp + 4.0)) < 1.6))
        brace = torso & (y > sp + 4.0) & (np.abs(ax - 8.5) < 1.5) & (y < neck_y - 1.0)
        out[band | brace] = REGIONS.index("reflective")
        vneck = torso & (z > W["spine_03"][2]) & (y > neck_y - 13.0 + ax * 1.1)   # the shirt collar shows
        out[vneck] = REGIONS.index("fabric_navy")
    out[(ax > wrist_x) & (y > pel)] = REGIONS.index("glove" if "gloves" in ppe else "skin")
    above = y > neck_y
    rel = (y - head[1]) / max(hl, 1e-6)
    hair = above & ((rel > 0.62) | ((rel > 0.12) & (z < head[2] - 3.0)))
    out[above] = REGIONS.index("skin")
    out[hair] = REGIONS.index("hair_dark")
    return out


def _raster(uv, idx, counts, pts, res):
    """For every texel: the rest-pose body point it shows (barycentric over the UV triangles), and a coverage mask."""
    Pm = np.zeros((res, res, 3), np.float32)
    M = np.zeros((res, res), bool)
    UV = np.column_stack([uv[:, 0] * res - 0.5, (1.0 - uv[:, 1]) * res - 0.5])
    a = 0
    for c in counts:
        for t in range(1, c - 1):
            k = [a, a + t, a + t + 1]
            q = UV[k]
            x0, y0 = np.floor(q.min(0)).astype(int)
            x1, y1 = np.ceil(q.max(0)).astype(int)
            x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, res - 1), min(y1, res - 1)
            if x1 < x0 or y1 < y0:
                continue
            v0, v1 = q[1] - q[0], q[2] - q[0]
            den = v0[0] * v1[1] - v1[0] * v0[1]
            if abs(den) < 1e-12:
                continue
            gx, gy = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1))
            px, py = gx - q[0][0], gy - q[0][1]
            b1 = (px * v1[1] - v1[0] * py) / den
            b2 = (v0[0] * py - px * v0[1]) / den
            b0 = 1.0 - b1 - b2
            m = (b0 >= -0.02) & (b1 >= -0.02) & (b2 >= -0.02)
            if not m.any():
                continue
            P3 = pts[idx[k]]
            Pm[gy[m], gx[m]] = b0[m, None] * P3[0] + b1[m, None] * P3[1] + b2[m, None] * P3[2]
            M[gy[m], gx[m]] = True
        a += c
    return Pm, M


def outfit_maps(wid, d, sk, ppe=("vest", "gloves"), res=2048):
    """Albedo + roughness maps of the worker's clothes in the body's UV layout (cached; ./textures/outfit/...)."""
    from .. import textures as GT
    from ..materials import LOOKS
    key = f"{wid.lower()}_{'_'.join(sorted(ppe)) or 'none'}_v{OUTFIT_VERSION}"
    dd = os.path.join(GT.ROOT, "outfit")
    out = {"albedo": os.path.join(dd, key + "_albedo.png"), "roughness": os.path.join(dd, key + "_rough.png")}
    out["rel"] = {k: GT.rel(v) for k, v in out.items()}
    if os.path.isfile(out["albedo"]) and os.path.isfile(out["roughness"]):
        return out
    os.makedirs(dd, exist_ok=True)
    Pm, M = _raster(d["uv"].astype(np.float64), d["indices"], d["counts"], d["points"].astype(np.float64), res)
    lab = np.zeros((res, res), int)
    lab[M] = classify(Pm[M].astype(np.float64), sk, ppe)
    for _ in range(8):                                        # dilate past the UV island edges (no seams)
        for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            sh = np.roll(np.roll(lab, dy, 0), dx, 1)
            shm = np.roll(np.roll(M, dy, 0), dx, 1)
            fill = ~M & shm
            lab[fill] = sh[fill]
            M = M | fill
    col = np.array([LOOKS[r][0] for r in REGIONS], np.float32)              # sRGB
    rgh = np.array([LOOKS[r][1] for r in REGIONS], np.float32)
    rng = np.random.default_rng(7)
    weave = 1.0 + 0.05 * (rng.random((res, res)) - 0.5)                    # fabric / skin grain
    GT.write_png(out["albedo"], np.clip(col[lab] * weave[..., None], 0.0, 1.0))
    GT.write_png(out["roughness"], np.clip(rgh[lab] * (2.0 - weave), 0.02, 1.0))
    return out


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
        st.Set(Vt.Vec2fArray.FromNumpy(d["uv"].astype(np.float32)))            # the outfit texture's layout
    # skin: map the body's joint names onto the canonical joint order
    J = d["joints"]
    remap = np.array([INDEX.get(n, 0) for n in J], np.int32)
    ji = remap[d["joint_indices"]]
    b = UsdSkel.BindingAPI.Apply(m.GetPrim())
    K = ji.shape[1]
    b.CreateJointIndicesPrimvar(False, K).Set(Vt.IntArray.FromNumpy(ji.reshape(-1).astype(np.int32)))
    b.CreateJointWeightsPrimvar(False, K).Set(Vt.FloatArray.FromNumpy(d["joint_weights"].reshape(-1).astype(np.float32)))
    b.CreateGeomBindTransformAttr(Gf.Matrix4d(1.0))
    maps = outfit_maps(_wid(d), d, sk, ppe)
    name = f"outfit_{_wid(d)}_{'_'.join(sorted(ppe)) or 'none'}"
    UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(lib.textured(name, maps["rel"]["albedo"], maps["rel"]["roughness"]))
    m.CreateDisplayColorAttr([Gf.Vec3f(0.6, 0.55, 0.5)])
    return m
