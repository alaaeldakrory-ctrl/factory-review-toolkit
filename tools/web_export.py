"""A solved task for the web viewer: the scene as a GLB, the worker's motion retargeted onto a Mixamo character, the
parts' and tools' motion, and per-frame RULA - small files a browser plays (three.js, see site/).

    python tools/task_run.py <task.toml> --web=<site>/jobs/<name>        (writes the files after the solve)

    <dir>/scene.glb     equipment (merged by material) + one node per part / tool, metres, Y up
    <dir>/motion.bin    int16 frames: per bone xyzw (x32767) | hips xyz (mm) | root x z (mm), yaw (0.01 deg)
                        | per moving part xyz (mm) + xyzw (x32767)
    <dir>/job.json      bones, parts, fps, frame count, steps (frames, checks, RULA), per-frame RULA and phase

The character's rest pose (worker.glb, Mixamo T-pose) differs from the frt rest pose (arms down): each bone gets the
frt joint's rotation applied on top of a fixed rest alignment (the rotation taking the Mixamo bone direction to the
frt one about the body's forward axis), so the motion carries over whatever the rest pose.
"""
import json
import math
import os
import struct

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
SITE = os.environ.get("FRT_SITE", os.path.join(ROOT, "..", "frt-site"))   # the website (its own folder, not this repo)
CHARACTER = os.path.join(SITE, "assets", "worker.glb")
PRE = "mixamorig1"

# frt joint -> (Mixamo bone, frt joint the bone points to, Mixamo bone it points to)
_SIDE = {"l": "Left", "r": "Right"}


def bone_map():
    m = {"pelvis": ("Hips", "spine_01", "Spine"), "spine_01": ("Spine", "spine_02", "Spine1"),
         "spine_02": ("Spine1", "spine_03", "Spine2"), "spine_03": ("Spine2", "neck_01", "Neck"),
         "neck_01": ("Neck", "head", "Head"), "head": ("Head", "head_top", "HeadTop_End")}
    for s, S in _SIDE.items():
        m[f"clavicle_{s}"] = (f"{S}Shoulder", f"upperarm_{s}", f"{S}Arm")
        m[f"upperarm_{s}"] = (f"{S}Arm", f"lowerarm_{s}", f"{S}ForeArm")
        m[f"lowerarm_{s}"] = (f"{S}ForeArm", f"hand_{s}", f"{S}Hand")
        m[f"hand_{s}"] = (f"{S}Hand", f"middle_01_{s}", f"{S}HandMiddle1")
        for f, F in (("thumb", "Thumb"), ("index", "Index"), ("middle", "Middle"), ("ring", "Ring"), ("pinky", "Pinky")):
            for k in (1, 2, 3):
                m[f"{f}_0{k}_{s}"] = (f"{S}Hand{F}{k}", f"{f}_0{k + 1}_{s}" if k < 3 else f"{f}_tip_{s}",
                                      f"{S}Hand{F}{k + 1}")
        m[f"thigh_{s}"] = (f"{S}UpLeg", f"calf_{s}", f"{S}Leg")
        m[f"calf_{s}"] = (f"{S}Leg", f"foot_{s}", f"{S}Foot")
        m[f"foot_{s}"] = (f"{S}Foot", f"ball_{s}", f"{S}ToeBase")
        m[f"ball_{s}"] = (f"{S}ToeBase", f"toe_{s}", f"{S}Toe_End")
    return m


# bones standing the same way in both rest poses: no direction matching (only the arms differ, T-pose vs arms down)
UPRIGHT = ("pelvis", "spine_01", "spine_02", "spine_03", "neck_01", "head")
UPRIGHT_PRE = ("thigh_", "calf_", "foot_", "ball_")


# ---------------------------------------------------------------------------------------------- small maths
def _q2m(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _m2q(R):
    """3x3 -> (x, y, z, w), w >= 0."""
    from frt_core.people.character import _quats
    w, x, y, z = _quats(np.asarray(R)[None])[0]
    return np.array([x, y, z, w])


def _orth(M):
    u, _, vt = np.linalg.svd(M)
    return u @ vt


def _swing(a, b):
    """The smallest rotation taking unit vector a to unit vector b."""
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if c < -0.9999:
        ax = np.cross(a, [1.0, 0.0, 0.0])
        if np.linalg.norm(ax) < 1e-6:
            ax = np.cross(a, [0.0, 1.0, 0.0])
        ax = ax / np.linalg.norm(ax)
        return 2 * np.outer(ax, ax) - np.eye(3)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K / (1.0 + c)


def _frame(d, ref):
    d = d / np.linalg.norm(d)
    r = ref - d * np.dot(ref, d)
    if np.linalg.norm(r) < 1e-6:
        r = np.cross(d, [1.0, 0.0, 0.0])
    r = r / np.linalg.norm(r)
    return np.column_stack([d, r, np.cross(d, r)])


# ---------------------------------------------------------------------------------------------- the character
def read_glb(path):
    b = open(path, "rb").read()
    n = struct.unpack("<I", b[12:16])[0]
    return json.loads(b[20:20 + n])


class Rig:
    """The Mixamo character's node tree at rest: world rotations and positions (its own units) per node name."""

    def __init__(self, path=CHARACTER):
        J = read_glb(path)
        nodes = J["nodes"]
        self.parent = {}
        for i, n in enumerate(nodes):
            for c in n.get("children", []):
                self.parent[c] = i
        self.name = [n.get("name", f"n{i}") for i, n in enumerate(nodes)]
        self.idx = {n: i for i, n in enumerate(self.name)}
        self.local = []
        for n in nodes:
            M = np.eye(4)
            if "matrix" in n:
                M = np.array(n["matrix"], float).reshape(4, 4).T
            else:
                S = np.diag(list(n.get("scale", [1, 1, 1])) + [1.0])
                R = np.eye(4)
                R[:3, :3] = _q2m(n.get("rotation", [0, 0, 0, 1]))
                T = np.eye(4)
                T[:3, 3] = n.get("translation", [0, 0, 0])
                M = T @ R @ S
            self.local.append(M)
        self.world = [None] * len(nodes)
        for i in range(len(nodes)):
            self._w(i)

    def _w(self, i):
        if self.world[i] is None:
            p = self.parent.get(i)
            self.world[i] = self.local[i] if p is None else self._w(p) @ self.local[i]
        return self.world[i]

    def pos(self, b):
        return self.world[self.idx[PRE + b]][:3, 3]

    def rot(self, b):
        return _orth(self.world[self.idx[PRE + b]][:3, :3])


class Retarget:
    """frt solved poses -> Mixamo bone local rotations (+ hips position), for one body and one character."""

    def __init__(self, sk, rig=None):
        self.sk, self.rig = sk, rig or Rig()
        self.map = {j: v for j, v in bone_map().items() if j in sk.world and PRE + v[0] in self.rig.idx}
        self.bones = [v[0] for v in self.map.values()]
        rest = dict(sk.world)
        rest.update(getattr(sk, "tips", {}))
        # the frt rest has no finger tips / toes / head top as joints: extend each chain straight on
        for j, (b, cj, cb) in self.map.items():
            if cj not in rest:
                par = self._parent(j)
                rest[cj] = rest[j] + (rest[j] - rest[par]) * 0.8 if par else rest[j] + np.array([0, 10.0, 0])
        self.align = {}
        for j, (b, cj, cb) in self.map.items():
            dm = self.rig.pos(cb) - self.rig.pos(b) if PRE + cb in self.rig.idx else self.rig.rot(b) @ [0, 1.0, 0]
            df = rest[cj] - rest[j]
            ref = np.array([0.0, 1.0, 0.0]) if abs(df[2]) > 0.7 * np.linalg.norm(df) else np.array([0.0, 0.0, 1.0])
            Rm, Rf = _frame(np.asarray(dm, float), ref), _frame(np.asarray(df, float), ref)
            self.align[j] = Rf @ Rm.T                                   # Mixamo bone direction -> frt direction
            if j in UPRIGHT or j.startswith(UPRIGHT_PRE):                # upright in both rests (the head bone
                self.align[j] = np.eye(3)                                # points at the face): taken as it is
        # whole-character scale: the character's head-top height to the frt body's
        top = rest.get("head_top", rest["head"] + np.array([0, 18.0, 0]))
        floor = min(self.rig.pos(f"{S}ToeBase")[1] for S in _SIDE.values())
        self.scale = float(top[1]) / float(self.rig.pos("HeadTop_End")[1] - floor)
        self.floor = floor
        self.hips_parent = self.rig.parent.get(self.rig.idx[PRE + "Hips"])
        self.offset = {}                                              # (the character keeps its own bones)
        self.node_scale = {i: float(np.linalg.norm(self.rig.world[i][:3, 0])) for i in range(len(self.rig.world))}
        self.seg = {}
        for b in ("ForeArm", "Hand", "Leg", "Foot"):
            for S in _SIDE.values():
                n = f"{S}{b}"
                self.seg[n] = float(np.linalg.norm(self.rig.pos(n) - self.rig.pos(self._up(n))))
        # the character's legs are not the frt body's: lift the hips by the difference so the feet reach the floor
        self.hip_lift = float(self.rig.pos("Hips")[1] - self.floor) - float(rest["pelvis"][1]) / self.scale

    @staticmethod
    def _up(n):
        return n.replace("ForeArm", "Arm").replace("Hand", "ForeArm") if "Arm" in n or "Hand" in n else             n.replace("Leg", "UpLeg").replace("Foot", "Leg") if n.endswith(("Leg", "Foot")) and "Up" not in n else n

    def _parent(self, j):
        from frt_core.people.skeleton import PARENT
        return PARENT.get(j)

    def pose(self, pose):
        """One solved pose -> ({bone: local quaternion xyzw}, hips local position in the character's units).

        The character keeps its own proportions (moving its joints would stretch the skinned mesh): the trunk, neck,
        head, hands and fingers take the solved rotations, the hips the solved pelvis position (lifted by the
        difference in leg length), and the arms and legs are re-solved for the character's own segment lengths so
        that the wrists and ankles land where the solved ones are, the elbows and knees pointing the same way."""
        from frt_core.people.kinematics import fk
        R, P = fk(self.sk, pose)
        rig = self.rig
        W = {}                                                         # posed world rotation per Mixamo node index
        for j, (b, _, _) in self.map.items():
            W[rig.idx[PRE + b]] = R[j] @ self.align[j] @ rig.rot(b)
        to_rig = lambda v: np.asarray(v, float) / self.scale + np.array([0.0, self.floor, 0.0])   # frt cm -> rig
        hips_w = to_rig(P["pelvis"]) + np.array([0.0, self.hip_lift, 0.0])
        drop = 0.0                                                     # a foot the shorter leg can't reach: hips down
        for S in _SIDE.values():
            s_ = "l" if S == "Left" else "r"
            A = self.positions(W, hips_w, f"{S}UpLeg")
            over = float(np.linalg.norm(to_rig(P[f"foot_{s_}"]) - A)) - (self.seg[f"{S}Leg"] + self.seg[f"{S}Foot"]) * 0.995
            drop = max(drop, over)
        hips_w[1] -= drop
        for S in _SIDE.values():
            s_ = "l" if S == "Left" else "r"
            self._limb(W, hips_w, f"{S}Arm", f"{S}ForeArm", f"{S}Hand", to_rig(P[f"hand_{s_}"]), to_rig(P[f"lowerarm_{s_}"]),
                       to_rig(P[f"upperarm_{s_}"]))
            self._limb(W, hips_w, f"{S}UpLeg", f"{S}Leg", f"{S}Foot", to_rig(P[f"foot_{s_}"]), to_rig(P[f"calf_{s_}"]),
                       to_rig(P[f"thigh_{s_}"]))
        out = {}
        for j, (b, _, _) in self.map.items():
            i = rig.idx[PRE + b]
            p = rig.parent.get(i)
            Pw = W[p] if p in W else (_orth(rig.world[p][:3, :3]) if p is not None else np.eye(3))
            out[b] = _m2q(Pw.T @ W[i])
        M = rig.world[self.hips_parent] if self.hips_parent is not None else np.eye(4)
        hip = (np.linalg.inv(M) @ np.append(hips_w, 1.0))[:3]
        return out, hip

    def positions(self, W, hips_w, upto):
        """World position of `upto` (a Mixamo bone) for posed world rotations W, the hips at hips_w (rig units)."""
        rig = self.rig
        chain, i = [], rig.idx[PRE + upto]
        while rig.name[i] != PRE + "Hips":
            chain.append(i)
            i = rig.parent[i]
        p, cur = hips_w.copy(), i
        for c in reversed(chain):
            Rp = W.get(cur, _orth(rig.world[cur][:3, :3]))
            p = p + Rp @ (rig.local[c][:3, 3] * self.node_scale[cur])
            cur = c
        return p

    def _limb(self, W, hips_w, b0, b1, b2, target, mid, root):
        """Two-bone IK on the character's own lengths: b2 to `target`, the b1 joint on the side the solved one is."""
        rig = self.rig
        i0, i1, i2 = (rig.idx[PRE + b] for b in (b0, b1, b2))
        A = self.positions(W, hips_w, b0)
        L1, L2 = self.seg[b1], self.seg[b2]
        d = target - A
        D = float(np.linalg.norm(d))
        if D < 1e-6:
            return
        u = d / D
        D = min(D, (L1 + L2) * 0.9995)
        pole = (mid - root) - u * np.dot(mid - root, u)               # the solved knee / elbow side
        if np.linalg.norm(pole) < 1e-6:
            pole = np.cross(u, [1.0, 0.0, 0.0])
        pole = pole / np.linalg.norm(pole)
        a = (L1 * L1 - L2 * L2 + D * D) / (2 * D)
        h = math.sqrt(max(L1 * L1 - a * a, 0.0))
        E = A + u * a + pole * h
        Tg = A + u * D
        for i, j, frm, to in ((i0, i1, A, E), (i1, i2, E, Tg)):
            cur = W[i] @ (rig.local[j][:3, 3] / (np.linalg.norm(rig.local[j][:3, 3]) + 1e-12))
            want = (to - frm) / (np.linalg.norm(to - frm) + 1e-12)
            W[i] = _swing(cur, want) @ W[i]

# ---------------------------------------------------------------------------------------------- the scene
ROUGH = (("chrome", 0.12), ("polished", 0.15), ("steel", 0.38), ("galv", 0.42), ("bolt", 0.35), ("powder", 0.5),
         ("plastic", 0.45), ("rubber", 0.82), ("grip", 0.8), ("cable", 0.6), ("studio", 0.85), ("alum", 0.32))


def _material(prim):
    from pxr import Usd, UsdShade
    m = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
    if not m:
        return "default", {"baseColorFactor": [0.5, 0.5, 0.5, 1], "metallicFactor": 0.0, "roughnessFactor": 0.6}
    name = m.GetPrim().GetName()
    col, met, coat = (0.5, 0.5, 0.5), 0.0, 0.0
    for c in Usd.PrimRange(m.GetPrim()):
        if c.GetTypeName() == "Shader":
            sh = UsdShade.Shader(c)
            for i in sh.GetInputs():
                v, k = i.Get(), i.GetBaseName()
                if k in ("paint_tint", "diffuse_color_constant") and v is not None:
                    col = tuple(float(x) for x in v)
                elif k in ("metallic", "metallic_constant") and v is not None:
                    met = float(v)
                elif k == "coat" and v is not None:
                    coat = float(v)
    rough = next((r for key, r in ROUGH if key in name), 0.5)
    return name, {"baseColorFactor": [*col, 1.0], "metallicFactor": met, "roughnessFactor": rough,
                  "extras": {"coat": coat}}


def _tris(mesh, M):
    """World-space (or M-space) triangles of a UsdGeom.Mesh: positions (n, 3) and flat normals, unindexed."""
    pts = np.asarray(mesh.GetPointsAttr().Get(0), float)
    fc = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), int)
    fi = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), int)
    if len(pts) == 0 or len(fc) == 0:
        return None
    P = (np.c_[pts, np.ones(len(pts))] @ M)[:, :3] * 0.01          # (USD row vectors) cm -> m
    tri, k = [], 0
    for c in fc:
        for t in range(1, c - 1):
            tri.append((fi[k], fi[k + t], fi[k + t + 1]))
        k += c
    tri = np.asarray(tri, int)
    if mesh.GetOrientationAttr().Get() == "leftHanded":
        tri = tri[:, ::-1]
    V = P[tri].reshape(-1, 3)
    n = np.cross(V[1::3] - V[0::3], V[2::3] - V[0::3])
    n = n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-12)
    return V.astype(np.float32), np.repeat(n, 3, axis=0).astype(np.float32)


class Glb:
    def __init__(self):
        self.bin = bytearray()
        self.j = {"asset": {"version": "2.0", "generator": "frt web_export"}, "buffers": [], "bufferViews": [],
                  "accessors": [], "meshes": [], "materials": [], "nodes": [], "scenes": [{"nodes": []}], "scene": 0}
        self.mat = {}

    def _acc(self, a, kind, target=34962):
        a = np.ascontiguousarray(a)
        while len(self.bin) % 4:
            self.bin.append(0)
        self.j["bufferViews"].append({"buffer": 0, "byteOffset": len(self.bin), "byteLength": a.nbytes,
                                      "target": target})
        self.bin += a.tobytes()
        acc = {"bufferView": len(self.j["bufferViews"]) - 1, "componentType": 5126, "count": len(a), "type": kind}
        if kind == "VEC3":
            acc["min"], acc["max"] = a.min(0).tolist(), a.max(0).tolist()
        self.j["accessors"].append(acc)
        return len(self.j["accessors"]) - 1

    def material(self, name, pbr):
        if name not in self.mat:
            ex = pbr.pop("extras", {})
            self.j["materials"].append({"name": name, "pbrMetallicRoughness": pbr, "extras": ex})
            self.mat[name] = len(self.j["materials"]) - 1
        return self.mat[name]

    def mesh(self, prims):
        """prims: [(material index, positions, normals)] -> mesh index."""
        out = []
        for m, V, N in prims:
            out.append({"attributes": {"POSITION": self._acc(V, "VEC3"), "NORMAL": self._acc(N, "VEC3")}, "material": m})
        self.j["meshes"].append({"primitives": out})
        return len(self.j["meshes"]) - 1

    def node(self, name, mesh=None, T=None, root=True):
        n = {"name": name}
        if mesh is not None:
            n["mesh"] = mesh
        if T is not None:
            n["translation"] = [float(v) for v in T[0]]
            n["rotation"] = [float(v) for v in T[1]]
        self.j["nodes"].append(n)
        if root:
            self.j["scenes"][0]["nodes"].append(len(self.j["nodes"]) - 1)
        return len(self.j["nodes"]) - 1

    def save(self, path):
        while len(self.bin) % 4:
            self.bin.append(0)
        self.j["buffers"] = [{"byteLength": len(self.bin)}]
        js = json.dumps(self.j, separators=(",", ":")).encode()
        js += b" " * ((4 - len(js) % 4) % 4)
        with open(path, "wb") as f:
            f.write(struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(js) + 8 + len(self.bin)))
            f.write(struct.pack("<II", len(js), 0x4E4F534A) + js)
            f.write(struct.pack("<II", len(self.bin), 0x004E4942) + bytes(self.bin))


def _pq(M):
    """USD row-vector 4x4 -> (position m, quaternion xyzw)."""
    A = np.asarray(M, float).T
    return A[:3, 3] * 0.01, _m2q(_orth(A[:3, :3]))


def export_scene(usd, path, n_frames, origins=None):
    """scene.glb (one node per asset, at its floor position: the demo page drags them) + the moving parts'
    per-frame (position, quaternion). origins: asset id -> (x, z) cm."""
    from pxr import Usd, UsdGeom
    st = Usd.Stage.Open(usd)
    g = Glb()
    xc = UsdGeom.XformCache(0)
    scene = st.GetPrimAtPath("/World/Asset/Scene")
    for a in scene.GetChildren():
        o = (origins or {}).get(a.GetName(), (0.0, 0.0))
        M0 = np.eye(4)
        M0[3, 0], M0[3, 2] = -o[0], -o[1]                            # (row vectors: world -> asset floor origin)
        by_mat = {}
        for p in Usd.PrimRange(a):
            if p.GetTypeName() == "Mesh":
                t = _tris(UsdGeom.Mesh(p), np.asarray(xc.GetLocalToWorldTransform(p), float) @ M0)
                if t is not None:
                    name, pbr = _material(p)
                    by_mat.setdefault(name, [pbr, [], []])
                    by_mat[name][1].append(t[0])
                    by_mat[name][2].append(t[1])
        if by_mat:
            g.node(a.GetName(), g.mesh([(g.material(k, v[0]), np.concatenate(v[1]), np.concatenate(v[2]))
                                        for k, v in by_mat.items()]), ((o[0] * 0.01, 0.0, o[1] * 0.01), (0, 0, 0, 1)))
    parts, moving = [], []
    root = st.GetPrimAtPath("/World/Asset/Parts")
    for p in (root.GetChildren() if root else []):
        W0 = np.asarray(xc.GetLocalToWorldTransform(p), float)
        inv = np.linalg.inv(W0)
        pr = []
        for c in Usd.PrimRange(p):
            if c.GetTypeName() == "Mesh":
                t = _tris(UsdGeom.Mesh(c), np.asarray(xc.GetLocalToWorldTransform(c), float) @ inv)
                if t is not None:
                    name, pbr = _material(c)
                    pr.append((g.material(name, pbr), t[0], t[1]))
        if not pr:
            continue
        g.node(p.GetName(), g.mesh(pr), _pq(W0))
        parts.append(p.GetName())
        ops = UsdGeom.Xformable(p).GetOrderedXformOps()
        if any(o.GetNumTimeSamples() > 1 for o in ops):
            moving.append(p)
    track = {}
    for i in range(n_frames):
        xi = UsdGeom.XformCache(i)
        for p in moving:
            track.setdefault(p.GetName(), []).append(_pq(xi.GetLocalToWorldTransform(p)))
    g.save(path)
    return parts, track


# ---------------------------------------------------------------------------------------------- the job
def export(T, sols, res, usd, out, scores=None, title=None):
    """Everything the viewer needs for one solved task into `out`."""
    from frt_core.people.bake import _rootT
    os.makedirs(out, exist_ok=True)
    n = len(sols)
    rt = Retarget(T.sk)
    origins = {k: (float(a["at"][0]), float(a["at"][2])) for k, a in T.assets.items() if "at" in a}
    parts, track = export_scene(usd, os.path.join(out, "scene.glb"), n, origins)
    moving = sorted(track)
    rows = []
    for i, s in enumerate(sols):
        q, hip = rt.pose(s.pose)
        r = T.frames[i].root or (0.0, 0.0, 0.0)
        row = [int(round(v * 32767)) for b in rt.bones for v in q[b]]
        row += [int(round(v * 10)) for v in hip]                      # (character units x10: 0.1 unit)
        row += [int(round(r[0] * 10)), int(round(r[1] * 10)), int(round(((r[2] + 180) % 360 - 180) * 100))]
        for m in moving:
            p, qq = track[m][i]
            row += [int(round(v * 1000)) for v in p] + [int(round(v * 32767)) for v in qq]
        rows.append(row)
    A = np.clip(np.array(rows, np.int64), -32768, 32767).astype("<i2")
    A.tofile(os.path.join(out, "motion.bin"))
    rula = [None if v is None else int(v) for v in (scores or [None] * n)][:n]
    steps = [{k: s[k] for k in ("id", "do", "start", "end", "seconds", "checks", "rula_max", "rula_mean", "worst")
              if k in s} | {"label": _label(s)} for s in res["steps"]]
    phase = [f.label.split(":")[-1] for f in T.frames]
    job = {"title": title or T.name, "fps": 24, "frames": n, "row": int(A.shape[1]), "bones": rt.bones,
           "offsets": {k: [round(float(x), 5) for x in v] for k, v in rt.offset.items()},
           "character_scale": rt.scale * 0.01, "hips_units": 0.1, "parts": parts, "moving": moving,
           "steps": steps, "rula": rula, "phase": phase, "counts": res.get("counts", {}),
           "solve_s": res.get("solve_s"), "quality": res.get("quality"), "notes": res.get("notes", [])[-6:],
           "layout": layout(T)}
    json.dump(job, open(os.path.join(out, "job.json"), "w"), separators=(",", ":"))
    return job


def layout(T):
    """What the demo page edits: the spots, the assets (floor position, size, kind), which bin each part is in."""
    assets = []
    for k, a in T.assets.items():
        if "at" not in a:
            continue
        assets.append({"id": k, "kind": a.get("kind"), "at": [float(v) for v in a["at"]],
                       "size": [float(v) for v in a.get("size", [20, 20, 20])], "spot": a.get("spot"),
                       "on": _on(T, k)})
    return {"spots": {k: [float(v) for v in s_] for k, s_ in T.spots.items()}, "assets": assets,
            "parts": {q: p.get("home") for q, p in T.parts.items()}, "steps": T.data.get("step", [])}


def _on(T, k):
    """The asset this one stands on (a bin or a tool on a cart): it moves with it in the editor."""
    a = T.assets[k]
    x, y, z = (float(v) for v in a["at"])
    if y < 5:
        return None
    for j, b in T.assets.items():
        if j == k or "at" not in b or b.get("kind") not in ("cart", "table", "bench", "shelf", "fixture", "rack"):
            continue
        bx, _, bz = (float(v) for v in b["at"])
        w, _, d = (float(v) for v in b.get("size", [0, 0, 0]))
        if abs(x - bx) <= w / 2 + 2 and abs(z - bz) <= d / 2 + 2:
            return j
    return None


VERB = {"hand_start": "Start fasteners by hand", "fasten": "Tighten", "connect": "Plug connectors",
        "carry": "Carry", "cable_tie": "Cable tie", "inspect": "Inspect", "transfer": "Transfer parts",
        "latch": "Latch", "get_tool": "Get tool", "put_back": "Put tool back"}


def _label(s):
    v = VERB.get(s.get("do"), str(s.get("do", "")).replace("_", " ").capitalize())
    on = str(s.get("on", "") or s.get("what", "")).split(".")[0].replace("_", " ")
    return f"{v} - {on}" if on else v
