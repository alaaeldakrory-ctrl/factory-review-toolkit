"""Minimal binary-FBX reader for skeleton animation (e.g. Mixamo "without skin" downloads).

Reads the node tree, LimbNode models (Lcl T/R/S, Pre/Post rotation, rotation order), their animation curves and
the parent links, and evaluates global joint matrices per frame (FBX cm, Y up).

    clip = FbxClip("Walking.fbx");  clip.names, clip.frames, clip.fps;  clip.global_mats(frame) -> {name: Gf.Matrix4d}
"""
import math
import struct
import zlib

from pxr import Gf

TICKS = 46186158000.0          # FBX time units per second


# --------------------------------------------------------------------------------------------- binary parse
def _props(buf, pos, n):
    out = []
    for _ in range(n):
        t = chr(buf[pos])
        pos += 1
        if t == "Y":
            out.append(struct.unpack_from("<h", buf, pos)[0]); pos += 2
        elif t == "C":
            out.append(bool(buf[pos])); pos += 1
        elif t == "I":
            out.append(struct.unpack_from("<i", buf, pos)[0]); pos += 4
        elif t == "F":
            out.append(struct.unpack_from("<f", buf, pos)[0]); pos += 4
        elif t == "D":
            out.append(struct.unpack_from("<d", buf, pos)[0]); pos += 8
        elif t == "L":
            out.append(struct.unpack_from("<q", buf, pos)[0]); pos += 8
        elif t in "fdlib":
            n_el, enc, clen = struct.unpack_from("<III", buf, pos)
            pos += 12
            raw = buf[pos:pos + clen]
            pos += clen
            if enc == 1:
                raw = zlib.decompress(raw)
            fmt = {"f": "f", "d": "d", "l": "q", "i": "i", "b": "?"}[t]
            out.append(list(struct.unpack("<%d%s" % (n_el, fmt), raw)))
        elif t in "SR":
            ln = struct.unpack_from("<I", buf, pos)[0]
            pos += 4
            v = buf[pos:pos + ln]
            pos += ln
            out.append(v.decode("utf-8", "replace") if t == "S" else v)
        else:
            raise ValueError("unknown FBX property type %r" % t)
    return out, pos


class Node:
    __slots__ = ("name", "props", "children")

    def __init__(self, name, props):
        self.name, self.props, self.children = name, props, []

    def find(self, name):
        return next((c for c in self.children if c.name == name), None)

    def findall(self, name):
        return [c for c in self.children if c.name == name]


def _node(buf, pos, wide):
    if wide:
        end, nprops, plen = struct.unpack_from("<QQQ", buf, pos)
        pos += 24
    else:
        end, nprops, plen = struct.unpack_from("<III", buf, pos)
        pos += 12
    nlen = buf[pos]
    pos += 1
    if end == 0:
        return None, pos
    name = buf[pos:pos + nlen].decode("ascii", "replace")
    pos += nlen
    props, pos = _props(buf, pos, nprops)
    node = Node(name, props)
    null = 25 if wide else 13
    while pos < end - null + 0 and pos < end:
        child, pos = _node(buf, pos, wide)
        if child is None:
            break
        node.children.append(child)
    return node, end


def parse(path):
    buf = open(path, "rb").read()
    if not buf.startswith(b"Kaydara FBX Binary"):
        raise ValueError("not a binary FBX file")
    version = struct.unpack_from("<I", buf, 23)[0]
    wide = version >= 7500
    pos = 27
    root = Node("root", [])
    while pos < len(buf) - 160:
        n, pos = _node(buf, pos, wide)
        if n is None:
            break
        root.children.append(n)
    return root


# --------------------------------------------------------------------------------------------- scene
def _p70(node):
    out = {}
    p = node.find("Properties70")
    if p:
        for c in p.findall("P"):
            out[c.props[0]] = c.props[4:]
    return out


def _euler(v, order=0):
    """FBX euler (deg) -> Gf.Rotation; order 0 = XYZ (X applied first)."""
    axes = {0: "XYZ", 1: "XZY", 2: "YZX", 3: "YXZ", 4: "ZXY", 5: "ZYX"}[order]
    unit = {"X": Gf.Vec3d(1, 0, 0), "Y": Gf.Vec3d(0, 1, 0), "Z": Gf.Vec3d(0, 0, 1)}
    idx = {"X": 0, "Y": 1, "Z": 2}
    r = Gf.Rotation(Gf.Vec3d(1, 0, 0), 0)
    for a in axes:                               # row-vector convention: first rotation on the left
        r = r * Gf.Rotation(unit[a], v[idx[a]])
    return r


class Curve:
    def __init__(self, times, values, default):
        self.t, self.v, self.d = times, values, default

    def at(self, t):
        if not self.t:
            return self.d
        if t <= self.t[0]:
            return self.v[0]
        if t >= self.t[-1]:
            return self.v[-1]
        lo, hi = 0, len(self.t) - 1
        while hi - lo > 1:
            m = (lo + hi) // 2
            if self.t[m] <= t:
                lo = m
            else:
                hi = m
        u = (t - self.t[lo]) / (self.t[hi] - self.t[lo])
        return self.v[lo] + (self.v[hi] - self.v[lo]) * u


class FbxClip:
    def __init__(self, path, fps=30.0):
        root = parse(path)
        objs = root.find("Objects")
        conns = root.find("Connections")
        self.fps = fps
        models, cnodes, curves = {}, {}, {}
        for o in objs.children:
            uid = o.props[0]
            if o.name == "Model":
                models[uid] = o
            elif o.name == "AnimationCurveNode":
                cnodes[uid] = o
            elif o.name == "AnimationCurve":
                kt = o.find("KeyTime").props[0]
                kv = o.find("KeyValueFloat").props[0]
                curves[uid] = ([k / TICKS for k in kt], kv)
        parent, node_target, curve_of = {}, {}, {}
        for c in conns.findall("C"):
            kind, child, par = c.props[0], c.props[1], c.props[2]
            if child in models and (par in models or par == 0):
                parent[child] = par
            elif child in cnodes and par in models and kind == "OP":
                node_target[child] = (par, c.props[3])
            elif child in curves and par in cnodes and kind == "OP":
                curve_of.setdefault(par, {})[c.props[3]] = curves[child]
        self.joints = {}
        for uid, m in models.items():
            name = m.props[1].split("\x00")[0].split("::")[-1]
            p = _p70(m)
            self.joints[uid] = {
                "name": name, "parent": parent.get(uid, 0),
                "T": list(p.get("Lcl Translation", [0, 0, 0])), "R": list(p.get("Lcl Rotation", [0, 0, 0])),
                "S": list(p.get("Lcl Scaling", [1, 1, 1])), "pre": list(p.get("PreRotation", [0, 0, 0])),
                "post": list(p.get("PostRotation", [0, 0, 0])), "order": int(p.get("RotationOrder", [0])[0]),
                "curves": {}}
        t_end = 0.0
        for cn, (muid, prop) in node_target.items():
            key = {"Lcl Translation": "T", "Lcl Rotation": "R", "Lcl Scaling": "S"}.get(prop)
            if not key or muid not in self.joints:
                continue
            j = self.joints[muid]
            chans = curve_of.get(cn, {})
            cv = []
            for k, axis in enumerate(("d|X", "d|Y", "d|Z")):
                if axis in chans:
                    times, vals = chans[axis]
                    cv.append(Curve(times, vals, j[key][k]))
                    t_end = max(t_end, times[-1] if times else 0)
                else:
                    cv.append(Curve([], [], j[key][k]))
            j["curves"][key] = cv
        self.duration = t_end
        self.frames = int(round(t_end * fps)) + 1
        self.by_name = {j["name"]: uid for uid, j in self.joints.items()}
        self.names = [j["name"] for j in self.joints.values()]

    def local(self, uid, t):
        j = self.joints[uid]
        val = {}
        for key in ("T", "R", "S"):
            if key in j["curves"]:
                val[key] = [c.at(t) for c in j["curves"][key]]
            else:
                val[key] = j[key]
        r = _euler(val["R"], j["order"])
        pre = _euler(j["pre"]) if any(j["pre"]) else None
        post = _euler(j["post"]) if any(j["post"]) else None
        if pre is not None:
            r = r * pre                           # world = R then PreRotation (row vectors: Pre applied after)
        if post is not None:
            r = post.GetInverse() * r
        m = Gf.Matrix4d().SetScale(Gf.Vec3d(*val["S"])) * Gf.Matrix4d().SetRotate(r)
        m.SetTranslateOnly(Gf.Vec3d(*val["T"]))
        return m

    def global_mats(self, frame=None, t=None):
        t = frame / self.fps if t is None else t
        out = {}

        def g(uid):
            if uid in out:
                return out[uid]
            j = self.joints[uid]
            m = self.local(uid, t)
            if j["parent"] in self.joints:
                m = m * g(j["parent"])
            out[uid] = m
            return m
        for uid in self.joints:
            g(uid)
        return {self.joints[u]["name"]: m for u, m in out.items()}
