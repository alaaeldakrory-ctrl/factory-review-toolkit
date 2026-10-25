"""Motion-library clips from a solved job: each step (and each walk) as a clip that starts at the origin facing +Z,
retargeted onto the Mixamo character, tagged (action, posture, height, tool, industry) and scored (RULA), written as

    <dir>/<clip>.glb    the character with the clip (opens in Blender, three.js, Unreal, Windows 3D Viewer)
    <dir>/<clip>.bvh    the skeleton motion only (Mixamo bone names, cm, Y up, 24 fps)
    <dir>/<clip>.json   tags + the per-frame bone rotations the site's preview plays (from motion.bin of the job)

    python tools/task_run.py <task.toml> --clips=<site>/library     (after the solve; index.json is rebuilt)
"""
import json
import math
import os
import struct

import numpy as np

import web_export as W

FPS = 24
TOOL_NAME = {"ratchet_38": "ratchet", "torque_wrench": "torque wrench", "screwdriver": "screwdriver",
             "impact_driver": "impact driver"}
VERB = {"hand_start": "Start by hand", "fasten": "Tighten", "connect": "Plug connector", "carry": "Carry",
        "cable_tie": "Cable tie", "inspect": "Inspect", "transfer": "Pick and place", "latch": "Latch",
        "push_door": "Push door shut", "walk": "Walk"}


def _yaw(r):
    return math.radians(r[2])


def _ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def clip_frames(T, sols, rt, a, b):
    """Frames a..b retargeted, relative to the clip's first frame: (bone -> (n, 4) local xyzw), hips (n, 3) local,
    world rotations per bone (n, 3, 3) and hips world (n, 3) in cm, all in the clip's own frame."""
    rig = rt.rig
    r0 = T.frames[a].root or (0.0, 0.0, 0.0)
    y0 = _yaw(r0)
    q_all = {k: [] for k in rt.bones}
    hips_l, hips_w, Wall = [], [], {k: [] for k in rt.bones}
    hp = rt.hips_parent
    Mp = rig.world[hp] if hp is not None else np.eye(4)
    Rp = W._orth(Mp[:3, :3])
    sp = float(np.linalg.norm(Mp[:3, 0]))
    for i in range(a, b + 1):
        q, hip = rt.pose(sols[i].pose)
        r = T.frames[i].root or r0
        Ry = _ry(_yaw(r) - y0)
        off = _ry(-y0) @ np.array([r[0] - r0[0], 0.0, r[1] - r0[1]]) / rt.scale      # cm -> rig units
        # world rotations (character space), then the clip-relative turn on the hips
        Wd = {}
        for k in rt.bones:
            j = rig.idx[W.PRE + k]
            p = rig.parent.get(j)
            Pw = Wd.get(p, W._orth(rig.world[p][:3, :3]) if p is not None else np.eye(3))
            Wd[j] = Pw @ W._q2m(q[k])
        hj = rig.idx[W.PRE + "Hips"]
        Wd[hj] = Ry @ Wd[hj]
        for k in rt.bones:                                            # (only the hips turned: children follow)
            j = rig.idx[W.PRE + k]
            if k != "Hips":
                p = rig.parent[j]
                Wd[j] = Wd[p] @ W._q2m(q[k])
        hw = (Mp @ np.append(hip, 1.0))[:3]
        hw = Ry @ hw + off
        hl = (np.linalg.inv(Mp) @ np.append(hw, 1.0))[:3]
        q_all["Hips"].append(W._m2q(Rp.T @ Wd[hj]))
        for k in rt.bones:
            if k != "Hips":
                q_all[k].append(np.asarray(q[k], float))
            Wall[k].append(Wd[rig.idx[W.PRE + k]])
        hips_l.append(hl)
        hips_w.append((hw - np.array([0.0, rt.floor, 0.0])) * rt.scale)
    return ({k: np.array(v) for k, v in q_all.items()}, np.array(hips_l),
            {k: np.array(v) for k, v in Wall.items()}, np.array(hips_w), sp)


# ---------------------------------------------------------------------------------------------- BVH
def write_bvh(path, rt, Wall, hips_w):
    """Mixamo bone names; rest = the character's T-pose with identity rotations (offsets = rest bone vectors, cm);
    each frame's rotations are the world turn from rest, expressed per joint (ZXY Euler)."""
    rig = rt.rig
    bones = rt.bones
    kids = {k: [] for k in bones}
    for k in bones:
        p = rig.parent.get(rig.idx[W.PRE + k])
        pn = rig.name[p][len(W.PRE):] if p is not None and rig.name[p].startswith(W.PRE) else None
        if pn in kids:
            kids[pn].append(k)
    Wr = {k: W._orth(rig.world[rig.idx[W.PRE + k]][:3, :3]) for k in bones}
    pos = {k: (rig.pos(k) - np.array([0.0, rt.floor, 0.0])) * rt.scale for k in bones}
    lines, order = ["HIERARCHY"], []

    def emit(k, depth, parent):
        ind = "  " * depth
        off = pos[k] - (pos[parent] if parent else np.zeros(3))
        lines.append(f"{ind}{'ROOT' if parent is None else 'JOINT'} mixamorig:{k}")
        lines.append(f"{ind}{{")
        lines.append(f"{ind}  OFFSET {off[0]:.4f} {off[1]:.4f} {off[2]:.4f}")
        lines.append(f"{ind}  CHANNELS {'6 Xposition Yposition Zposition' if parent is None else '3'} Zrotation Xrotation Yrotation")
        order.append(k)
        for c in kids[k]:
            emit(c, depth + 1, k)
        if not kids[k]:
            lines.append(f"{ind}  End Site")
            lines.append(f"{ind}  {{")
            lines.append(f"{ind}    OFFSET 0.0 {max(2.0, np.linalg.norm(off) * 0.5):.4f} 0.0")
            lines.append(f"{ind}  }}")
        lines.append(f"{ind}}}")
    emit("Hips", 0, None)
    n = len(hips_w)
    lines += ["MOTION", f"Frames: {n}", f"Frame Time: {1.0 / FPS:.6f}"]
    par = {}
    for k in bones:
        for c in kids[k]:
            par[c] = k
    for f in range(n):
        D = {k: Wall[k][f] @ Wr[k].T for k in bones}                 # world turn from rest
        row = [*hips_w[f]]
        for k in order:
            L = (D[par[k]].T @ D[k]) if k in par else D[k]
            row += _zxy(L)
        lines.append(" ".join(f"{v:.4f}" for v in row))
    open(path, "w").write("\n".join(lines) + "\n")


def _zxy(R):
    """R = Rz(z) Rx(x) Ry(y) -> degrees (z, x, y)."""
    x = math.asin(max(-1.0, min(1.0, R[2, 1])))
    if abs(math.cos(x)) > 1e-6:
        z = math.atan2(-R[0, 1], R[1, 1])
        y = math.atan2(-R[2, 0], R[2, 2])
    else:
        z, y = math.atan2(R[1, 0], R[0, 0]), 0.0
    return [math.degrees(z), math.degrees(x), math.degrees(y)]


# ---------------------------------------------------------------------------------------------- GLB
def write_glb(path, rt, q_all, hips_l):
    """The character (worker.glb as it is) with one animation added: a rotation track per bone, the hips' translation."""
    b = open(W.CHARACTER, "rb").read()
    jl = struct.unpack("<I", b[12:16])[0]
    J = json.loads(b[20:20 + jl])
    o = 20 + jl
    bl = struct.unpack("<I", b[o:o + 4])[0]
    binc = bytearray(b[o + 8:o + 8 + bl])
    n = len(hips_l)

    def acc(a, kind, mm=False):
        a = np.ascontiguousarray(a, dtype=np.float32)
        while len(binc) % 4:
            binc.append(0)
        J["bufferViews"].append({"buffer": 0, "byteOffset": len(binc), "byteLength": a.nbytes})
        binc.extend(a.tobytes())
        x = {"bufferView": len(J["bufferViews"]) - 1, "componentType": 5126, "count": int(a.shape[0]), "type": kind}
        if mm:
            x["min"], x["max"] = [float(a.min())], [float(a.max())]
        J["accessors"].append(x)
        return len(J["accessors"]) - 1
    t = acc(np.arange(n, dtype=np.float32)[:, None] / FPS, "SCALAR", True)
    idx = {nd.get("name"): i for i, nd in enumerate(J["nodes"])}
    samplers, channels = [], []
    for k, Q in q_all.items():
        node = idx.get(W.PRE + k)
        if node is None:
            continue
        samplers.append({"input": t, "output": acc(Q, "VEC4"), "interpolation": "LINEAR"})
        channels.append({"sampler": len(samplers) - 1, "target": {"node": node, "path": "rotation"}})
    samplers.append({"input": t, "output": acc(hips_l, "VEC3"), "interpolation": "LINEAR"})
    channels.append({"sampler": len(samplers) - 1, "target": {"node": idx[W.PRE + "Hips"], "path": "translation"}})
    J["animations"] = [{"name": os.path.splitext(os.path.basename(path))[0], "samplers": samplers, "channels": channels}]
    J["buffers"][0]["byteLength"] = len(binc)
    js = json.dumps(J, separators=(",", ":")).encode()
    js += b" " * ((4 - len(js) % 4) % 4)
    while len(binc) % 4:
        binc.append(0)
    with open(path, "wb") as f:
        f.write(struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(js) + 8 + len(binc)))
        f.write(struct.pack("<II", len(js), 0x4E4F534A) + js)
        f.write(struct.pack("<II", len(binc), 0x004E4942) + bytes(binc))


# ---------------------------------------------------------------------------------------------- tags
def tags(T, sols, s, a, b, rt):
    from frt_core.people.kinematics import fk
    st = T.sk.tips.get("head_top", np.array([0, 175.0, 0]))[1]
    ys, flex, kneel = [], [], False
    for i in range(a, b + 1, 3):
        f = T.frames[i]
        ys += [float(t.point[1]) for t in f.targets if t.weight >= 0.5]
        if getattr(f.stance, "drop", None) is not None:
            kneel = True
        R, P = fk(T.sk, sols[i].pose, ["neck_01"])
        v = P["neck_01"] - P["pelvis"]
        flex.append(math.degrees(math.atan2(math.hypot(v[0], v[2]), v[1])))
    h = float(np.median(ys)) / st if ys else 0.5
    height = "overhead" if h > 0.86 else "shoulder" if h > 0.70 else "chest" if h > 0.52 else "waist" if h > 0.33 else "low"
    posture = "kneeling" if kneel or s.get("posture") == "kneel" else "bent over" if np.median(flex) > 30 else "standing"
    out = {"action": VERB.get(s["do"], s["do"]), "height": height, "posture": posture}
    if s.get("tool"):
        tid = s["tool"]
        out["tool"] = TOOL_NAME.get(T.parts.get(tid, {}).get("tool", tid), str(tid).replace("_", " "))
    if s.get("what") and s["do"] in ("carry", "hand_start", "transfer"):
        out["part"] = str(s["what"]).replace("_", " ")
    return out


def export_clips(T, sols, res, scores, out, job_id, industry=""):
    """Every step of the job (and its first walk) as a library clip into `out`; returns the clips' index entries."""
    os.makedirs(out, exist_ok=True)
    rt = W.Retarget(T.sk)
    entries = []
    cuts = []
    for k, s in enumerate(res["steps"]):
        a, b = s["start"], s["end"]
        fr = [i for i in range(a, b + 1) if not T.frames[i].label.endswith(":walk")]
        if len(fr) > 12:
            cuts.append((s, fr[0], fr[-1]))
    walk = [i for i, f in enumerate(T.frames) if f.label.endswith(":walk")]
    if len(walk) > 30:                                                # the first walk, as its own clip
        a = walk[0]
        b = a
        while b + 1 < len(T.frames) and T.frames[b + 1].label.endswith(":walk"):
            b += 1
        cuts.append(({"id": "walk", "do": "walk", "seconds": (b - a + 1) / FPS}, a, b))
    for s, a, b in cuts:
        cid = f"{job_id}__{s['id']}"
        q_all, hips_l, Wall, hips_w, _ = clip_frames(T, sols, rt, a, b)
        write_glb(os.path.join(out, cid + ".glb"), rt, q_all, hips_l)
        write_bvh(os.path.join(out, cid + ".bvh"), rt, Wall, hips_w)
        tg = tags(T, sols, s, a, b, rt) if s["do"] != "walk" else {"action": "Walk", "height": "-", "posture": "walking"}
        rula = [v for v in (scores or [])[a:b + 1] if v]
        name = tg["action"]
        if tg.get("tool"):
            name += f" · {tg['tool']}"
        elif tg.get("part") and s["do"] != "hand_start":
            name += f" · {tg['part']}"
        if s["do"] == "hand_start":
            name += f" · {tg.get('part', 'parts')}s"
        ctx = tg["height"] if tg["posture"] == "standing" else tg["posture"]
        if s["do"] != "walk":
            name += f", {ctx}"
        # the preview: the clip's frames as int16 (same layout as motion.bin's bone part + hips)
        rows = np.concatenate([np.concatenate([q_all[k] for k in rt.bones], axis=1) * 32767.0, hips_l * 10.0], axis=1)
        np.clip(np.round(rows), -32768, 32767).astype("<i2").tofile(os.path.join(out, cid + ".bin"))
        e = {"id": cid, "name": name, "industry": industry, "job": job_id, "frames": b - a + 1,
             "seconds": round((b - a + 1) / FPS, 1), "tags": tg,
             "rula_max": max(rula) if rula else None, "rula_mean": round(sum(rula) / len(rula), 1) if rula else None,
             "files": {"glb": cid + ".glb", "bvh": cid + ".bvh"}}
        json.dump(e, open(os.path.join(out, cid + ".json"), "w"))
        entries.append(e)
    rebuild_index(out, rt)
    return entries


def rebuild_index(out, rt=None):
    """library/index.json: every clip's entry (and the bone order the preview files use)."""
    clips = []
    for f in sorted(os.listdir(out)):
        if f.endswith(".json") and f != "index.json":
            clips.append(json.load(open(os.path.join(out, f))))
    idx = {"bones": rt.bones if rt else [], "fps": FPS, "character_scale": rt.scale * 0.01 if rt else 1.0, "clips": clips}
    if rt is None and os.path.isfile(os.path.join(out, "index.json")):
        idx["bones"] = json.load(open(os.path.join(out, "index.json"))).get("bones", [])
    json.dump(idx, open(os.path.join(out, "index.json"), "w"), separators=(",", ":"))
