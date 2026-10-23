"""Character adapter: frt motion on real characters (NVIDIA / Reallusion CC rigs such as Uniform_M_0001, Worker, Debra).

The solver keeps working on the canonical skeleton, but that skeleton is built from the character's own joints. The
character's bind pose is a T-pose like ours, so arm, leg and finger lengths are exact and hands land on the tools.
A solved pose then transfers joint for joint:

    world rotation of a character joint = (world rotation of its canonical joint, identity at rest) x its bind rotation

Character joints with no canonical match (twist and share bones, toes, face, eyes, props) ride rigidly on their
nearest mapped parent. The forearm twist is spread over the forearm's twist bones, so the wrist does not candy-wrap.

The character asset is only referenced, never changed or converted. The adapter adds a new SkelAnimation and points
the character's skeleton at it (an override in the scene).

    ch = Character.load("C:/Nvidia_Assets/.../uniform_m_0001.usd")         # once per character asset
    res = api.solve_job("ratchet", worker=ch)                               # solved on the character's proportions
    api.write_job_usd(res, "C:/out/ratchet_uniform_m.usd")                  # the character does the job
"""
import math
import os

import numpy as np

from .skeleton import FINGERS, NAMES, PARENT, Skeleton

# canonical joint -> character joint (Reallusion CC / ActorCore naming)
CC_MAP = {"pelvis": "Hip", "spine_01": "Waist", "spine_02": "Spine01", "spine_03": "Spine02", "neck_01": "NeckTwist01",
          "head": "Head"}
_CC_FINGER = {"thumb": "Thumb", "index": "Index", "middle": "Mid", "ring": "Ring", "pinky": "Pinky"}
for _s, _S in (("l", "L"), ("r", "R")):
    CC_MAP.update({f"clavicle_{_s}": f"{_S}_Clavicle", f"upperarm_{_s}": f"{_S}_Upperarm",
                   f"lowerarm_{_s}": f"{_S}_Forearm", f"hand_{_s}": f"{_S}_Hand", f"thigh_{_s}": f"{_S}_Thigh",
                   f"calf_{_s}": f"{_S}_Calf", f"foot_{_s}": f"{_S}_Foot", f"ball_{_s}": f"{_S}_ToeBase"})
    for _f, _F in _CC_FINGER.items():
        for _k in (1, 2, 3):
            CC_MAP[f"{_f}_0{_k}_{_s}"] = f"{_S}_{_F}{_k}"
# the pelvis joint the legs hang from rides with the hips; the second neck joint takes part of the neck bend
FOLLOW = {"Pelvis": ("pelvis", 1.0), "NeckTwist02": ("neck_01", 1.0)}
TWIST = {"L_ForearmTwist01": ("l", 0.33), "L_ForearmTwist02": ("l", 0.66), "R_ForearmTwist01": ("r", 0.33),
         "R_ForearmTwist02": ("r", 0.66)}


def _name(p):
    return str(p).split("/")[-1]


def _m3(M):
    return np.array([[M[i][j] for j in range(3)] for i in range(3)], float).T      # Gf row-vector -> column form


def _quat(R):
    """3x3 (column form) -> Gf.Quatf."""
    from pxr import Gf
    m = Gf.Matrix3d(*R.T.flatten().tolist())
    q = m.ExtractRotation().GetQuat()
    return Gf.Quatf(float(q.GetReal()), *map(float, q.GetImaginary()))


def _axis_angle(axis, ang):
    a = np.asarray(axis, float)
    a = a / (np.linalg.norm(a) + 1e-12)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * K @ K


def _twist_angle(R, axis):
    """Twist angle (rad) of rotation R about `axis` (swing-twist decomposition, axis in R's own frame)."""
    a = np.asarray(axis, float)
    a = a / (np.linalg.norm(a) + 1e-12)
    w = math.sqrt(max(0.0, 1.0 + R[0, 0] + R[1, 1] + R[2, 2])) / 2.0
    if w < 1e-6:
        return 0.0
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (4.0 * w)
    p = float(np.dot(v, a))
    return 2.0 * math.atan2(p, w)


class Character:
    """A character asset (UsdSkel) ready for frt motion: canonical skeleton from its joints, worker profile, and
    the per-frame write-out onto its own rig."""

    def __init__(self, asset, skel_path, joints, bind, to_frt, tips_height, worker):
        self.asset = asset
        self.skel_path = skel_path                     # skeleton prim inside the asset (default-prim relative)
        self.joints = joints                           # character joint paths (UsdSkel order)
        self.names = [_name(j) for j in joints]
        self.parent = {}
        for j in joints:
            p = j.rsplit("/", 1)[0] if "/" in j else None
            self.parent[_name(j)] = _name(p) if p else None
        self.bind = bind                               # name -> 4x4 (column form) bind world, skeleton space
        self.B = to_frt                                # 4x4: skeleton space -> frt worker frame (Y up, +Z fwd, cm)
        self.Binv = np.linalg.inv(to_frt)
        self.worker = worker
        self.skeleton = self._canonical(tips_height)
        self.id = worker.id

    # -------------------------------------------------------------------------------------------- loading
    @classmethod
    def load(cls, asset, wid=None, sex=None):
        from pxr import Usd, UsdGeom, UsdSkel

        from .workers import custom
        st = Usd.Stage.Open(asset)
        skel = next((p for p in Usd.PrimRange(st.GetPseudoRoot()) if p.IsA(UsdSkel.Skeleton)), None)
        if skel is None:
            raise ValueError(f"{asset}: no UsdSkel skeleton")
        S = UsdSkel.Skeleton(skel)
        joints = [str(j) for j in S.GetJointsAttr().Get()]
        names = [_name(j) for j in joints]
        missing = [v for v in CC_MAP.values() if v not in names and not v.endswith(("Thumb3", "Pinky3"))]
        if [m for m in missing if not m.split("_")[-1].startswith(("Thumb", "Index", "Mid", "Ring", "Pinky"))]:
            raise ValueError(f"{asset}: not a Reallusion CC rig (missing {missing[:6]})")
        bindm = S.GetBindTransformsAttr().Get()
        bind = {}
        for n, M in zip(names, bindm):
            T = np.eye(4)
            T[:3, :3] = _m3(M)
            T[:3, 3] = np.array(M.ExtractTranslation(), float)
            bind[n] = T
        P = {n: bind[n][:3, 3] for n in names}
        up = P["Head"] - P["Hip"]
        left = P["L_Upperarm"] - P["R_Upperarm"]
        up /= np.linalg.norm(up)
        left = left - up * np.dot(left, up)
        left /= np.linalg.norm(left)
        fwd = np.cross(left, up)
        R = np.vstack([left, up, fwd])                                   # rows: frt x (left), y (up), z (forward)
        B = np.eye(4)
        B[:3, :3] = R
        hip = R @ P["Hip"]
        B[:3, 3] = (-hip[0], 0.0, -hip[2])                              # feet stay on the asset's own floor
        # head top: the character's mesh height above its feet (in frt frame)
        bc = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
        rng = bc.ComputeWorldBound(skel.GetParent()).ComputeAlignedRange()
        L2W = UsdGeom.Xformable(skel).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        corners = [np.array(L2W.GetInverse().Transform(rng.GetCorner(i)), float) for i in range(8)]
        top = max(float((B[:3, :3] @ c + B[:3, 3])[1]) for c in corners)
        base = os.path.splitext(os.path.basename(asset))[0]
        wid = wid or base
        if sex is None:
            low = base.lower()
            sex = "F" if ("_f_" in low or low.endswith("_f") or "-f-" in low or low in ("debra",)) else "M"
        w = custom(wid, top, sex)
        w.label = f"{base} ({top:.0f} cm)"
        st_skel = str(skel.GetPath())
        dp = str(st.GetDefaultPrim().GetPath()) if st.GetDefaultPrim() else ""
        rel = st_skel[len(dp):].lstrip("/") if dp and st_skel.startswith(dp + "/") else st_skel.lstrip("/")
        ch = cls(asset, rel, joints, bind, B, top, w)
        ch.skel_to_asset = np.array(L2W, float).T                        # skeleton space -> asset root (column form)
        if dp:
            Ldp = np.array(UsdGeom.Xformable(st.GetDefaultPrim()).ComputeLocalToWorldTransform(0), float).T
            ch.skel_to_asset = np.linalg.inv(Ldp) @ ch.skel_to_asset    # relative to the default prim (referenced)
        return ch

    def _p(self, n):
        return self.B[:3, :3] @ self.bind[n][:3, 3] + self.B[:3, 3]

    def _canonical(self, top):
        world = {"root": np.zeros(3)}
        for c, n in CC_MAP.items():
            if n in self.bind:
                world[c] = self._p(n)
        for s, S in (("l", "L"), ("r", "R")):                        # fingers the rig lacks: from the knuckle row
            for f, F in _CC_FINGER.items():
                for k in (1, 2, 3):
                    c = f"{f}_0{k}_{s}"
                    if c not in world:
                        prev = world.get(f"{f}_0{k - 1}_{s}", world[f"hand_{s}"])
                        d = prev - world.get(f"{f}_0{k - 2}_{s}", world[f"hand_{s}"]) if k > 1 else \
                            np.array([1.0 if s == "l" else -1.0, 0, 0]) * 4.0
                        world[c] = prev + d * 0.8
        tips = {"head_top": np.array([world["head"][0], top, world["head"][2]])}
        for s in "lr":
            for f in FINGERS:
                a, b = world[f"{f}_02_{s}"], world[f"{f}_03_{s}"]
                tips[f"{f}_tip_{s}"] = b + (b - a) * 0.85
            ball, foot = world[f"ball_{s}"], world[f"foot_{s}"]
            d = ball - foot
            d[1] = 0.0
            tips[f"toe_{s}"] = np.array([ball[0], 1.5, ball[2] + max(4.0, 0.45 * np.linalg.norm(d))])
            tips[f"heel_{s}"] = np.array([foot[0], 2.5, foot[2] - 5.0])
        for n in NAMES:
            if n not in world:
                world[n] = world[PARENT[n]].copy()
        return Skeleton(world, tips)

    # -------------------------------------------------------------------------------------------- write-out
    def _mapped(self, n):
        """(canonical joint driving this character joint, how) - nearest mapped ancestor for the rest."""
        inv = {v: k for k, v in CC_MAP.items()}
        if n in inv:
            return inv[n]
        if n in FOLLOW:
            return FOLLOW[n][0]
        p = self.parent.get(n)
        while p is not None:
            if p in inv:
                return inv[p]
            if p in FOLLOW:
                return FOLLOW[p][0]
            p = self.parent.get(p)
        return "root"

    def _twist(self, Rc, s):
        """(forearm axis, twist of the forearm joint, twist elbow -> hand) for side s; axis in the forearm's frame."""
        sk = self.skeleton
        a = sk.world[f"hand_{s}"] - sk.world[f"lowerarm_{s}"]
        a = a / np.linalg.norm(a)
        tw_fa = _twist_angle(Rc[f"upperarm_{s}"].T @ Rc[f"lowerarm_{s}"], a)
        tw_h = _twist_angle(Rc[f"lowerarm_{s}"].T @ Rc[f"hand_{s}"], a)
        return a, tw_fa, tw_fa + tw_h

    def pose_world(self, pose, root=None):
        """Character joint name -> 4x4 world (skeleton space) for one solved canonical pose. root: (x, z, yaw) of the
        pose's frame inside the character's frt frame (a re-rooted read-back), None = at the origin."""
        from .kinematics import fk
        Rc, Pc = fk(self.skeleton, pose)
        if root is not None:
            a = math.radians(root[2])
            Ry = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
            o = np.array([root[0], 0.0, root[1]])
            Rc = {k: Ry @ v for k, v in Rc.items()}
            Pc = {k: Ry @ v + o for k, v in Pc.items()}
        Bi, B3 = self.Binv, self.B[:3, :3]
        drive = getattr(self, "_drive", None)
        if drive is None:
            drive = self._drive = {n: self._mapped(n) for n in self.names}
        inv = {v: k for k, v in CC_MAP.items()}
        W = {}
        for j in self.joints:                                         # parents come first in UsdSkel order
            n = _name(j)
            c = drive[n]
            Rf = Rc.get(c, np.eye(3))
            if n in TWIST or (n in inv and inv[n].startswith("lowerarm")):
                # the forearm's twist (pronation) goes to the twist bones, a third and two thirds of the way from
                # the elbow, and the hand. The forearm joint itself only bends.
                s_, u = TWIST.get(n, (inv.get(n, "_l")[-1], 0.0))
                a, tw_fa, tw_all = self._twist(Rc, s_)
                Rf = Rc[f"lowerarm_{s_}"] @ _axis_angle(a, -tw_fa + u * tw_all)
            delta = B3.T @ Rf @ B3                                    # frt-frame delta -> skeleton space
            T = np.eye(4)
            T[:3, :3] = delta @ self.bind[n][:3, :3]
            p = self.parent.get(n)
            if n in inv:                                              # mapped: the solved position
                T[:3, 3] = Bi[:3, :3] @ (Pc[inv[n]] - self.B[:3, 3])
            elif p is None:
                T[:3, 3] = self.bind[n][:3, 3]
            else:                                                     # rigid on its parent
                Wp = W[p]
                Dp = Wp[:3, :3] @ self.bind[p][:3, :3].T
                T[:3, 3] = Wp[:3, 3] + Dp @ (self.bind[n][:3, 3] - self.bind[p][:3, 3])
            W[n] = T
        return W

    def local_arrays(self, pose, root=None):
        """UsdSkel (translations, rotations) for one pose, in the skeleton's joint order."""
        from pxr import Gf, Vt
        W = self.pose_world(pose, root)
        tr, rot = [], []
        for j in self.joints:
            n = _name(j)
            p = self.parent.get(n)
            L = W[n] if p is None else np.linalg.inv(W[p]) @ W[n]
            R = L[:3, :3]
            u, _, vt = np.linalg.svd(R)                               # clean numerical drift
            R = u @ vt
            tr.append(Gf.Vec3f(*map(float, L[:3, 3])))
            rot.append(_quat(R))
        return Vt.Vec3fArray(tr), Vt.QuatfArray(rot)

    def write(self, stage, path, sols, start=0):
        """The character at `path` (an Xform in a Y-up cm stage, in the worker frame), playing `sols`."""
        from pxr import Gf, Sdf, UsdGeom, UsdSkel, Vt
        body = UsdGeom.Xform.Define(stage, path)
        body.ClearXformOpOrder()
        M = self.B @ np.linalg.inv(self.skel_to_asset)               # asset root -> frt worker frame
        body.AddTransformOp().Set(Gf.Matrix4d(*M.T.flatten().tolist()))
        body.GetPrim().GetReferences().AddReference(self.asset.replace("\\", "/"))
        skel_prim = stage.GetPrimAtPath(f"{path}/{self.skel_path}")
        anim = UsdSkel.Animation.Define(stage, f"{path}_Motion")
        anim.CreateJointsAttr(Vt.TokenArray(self.joints))
        tr_a, rot_a = anim.CreateTranslationsAttr(), anim.CreateRotationsAttr()
        anim.CreateScalesAttr(Vt.Vec3hArray([Gf.Vec3h(1, 1, 1)] * len(self.joints)))
        for i, s in enumerate(sols):
            tr, rot = self.local_arrays(s.pose)
            tr_a.Set(tr, start + i)
            rot_a.Set(rot, start + i)
        over = stage.OverridePrim(skel_prim.GetPath() if skel_prim else Sdf.Path(f"{path}/{self.skel_path}"))
        UsdSkel.BindingAPI.Apply(over).CreateAnimationSourceRel().SetTargets([anim.GetPath()])
        prim = body.GetPrim()
        prim.CreateAttribute("frt:worker", Sdf.ValueTypeNames.String).Set(self.worker.id)
        prim.CreateAttribute("frt:character", Sdf.ValueTypeNames.String).Set(self.asset)
        prim.CreateAttribute("frt:stature", Sdf.ValueTypeNames.Float).Set(float(self.worker.stature))
        return anim


    # -------------------------------------------------------------------------------------------- read-back
    def read(self, stage, skel_prim, times):
        """The character's existing animation, read back in the canonical joints: per time code a dict
        {"R": canonical joint -> world rotation delta (identity = T-pose), "P": canonical joint -> position} in the
        character's frt frame (B applied to skeleton space). The exact inverse of write()."""
        from pxr import Usd, UsdSkel
        root = skel_prim
        while root and not root.IsA(UsdSkel.Root):
            root = root.GetParent()
        cache = UsdSkel.Cache()
        cache.Populate(UsdSkel.Root(root), Usd.TraverseInstanceProxies())
        q = cache.GetSkelQuery(UsdSkel.Skeleton(skel_prim))
        order = [_name(j) for j in q.GetJointOrder()]
        idx = {c: order.index(n) for c, n in CC_MAP.items() if n in order}
        B3 = self.B[:3, :3]
        out = []
        for t in times:
            S = q.ComputeJointSkelTransforms(Usd.TimeCode(float(t)))
            R, P = {}, {}
            for c, i in idx.items():
                n = CC_MAP[c]
                M = S[i]
                Rs = _m3(M)
                R[c] = B3 @ Rs @ self.bind[n][:3, :3].T @ B3.T
                P[c] = B3 @ np.array(M.ExtractTranslation(), float) + self.B[:3, 3]
            out.append({"R": R, "P": P})
        return out

    def to_pose(self, rec, root):
        """A read-back record as a canonical Pose relative to a root (x, z, yaw) in the character's frt frame."""
        from .kinematics import Pose
        a = math.radians(root[2])
        Ry = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
        o = np.array([root[0], 0.0, root[1]])
        Rl = {c: Ry.T @ r for c, r in rec["R"].items()}
        pose = Pose()
        for c in NAMES:
            if c == "root" or c not in Rl:
                continue
            par = PARENT[c]
            Rp = Rl.get(par, np.eye(3)) if par != "root" else np.eye(3)
            pose.rot[c] = Rp.T @ Rl[c]
        pose.pelvis = Ry.T @ (rec["P"]["pelvis"] - o) - self.skeleton.world["pelvis"]
        return pose


_CACHE = {}


def load(asset, **kw):
    """Character.load, cached per asset path."""
    key = (os.path.abspath(asset), tuple(sorted(kw.items())))
    if key not in _CACHE:
        _CACHE[key] = Character.load(asset, **kw)
    return _CACHE[key]
