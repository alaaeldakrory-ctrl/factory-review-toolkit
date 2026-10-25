"""A Mixamo character (a .glb with the standard mixamorig skeleton) as the worker's body.

The website shows a Mixamo character. Solving on another body (an frt body or an ActorCore character) and bending
the result onto the Mixamo one afterwards cannot work well: its shoulders are wider, its forearms and legs shorter,
so the wrists, the shoulders and the hips all have to give somewhere - hands miss what they hold, shoulders shrug,
hips drop. Here the canonical skeleton is built from the character's own joints instead (its bind pose is a T-pose,
palms down, like ours), so the solver works on the body that is shown and the web retarget is joint for joint.

    [task]
    worker = "C:/.../worker.glb"            (any path ending .glb)

The .glb is only read (its node tree); nothing of it is copied anywhere.
"""
import os

import numpy as np

from .skeleton import FINGERS, NAMES, PARENT, Skeleton

PRE = "mixamorig1"
_SIDE = {"l": "Left", "r": "Right"}
_F = {"thumb": "Thumb", "index": "Index", "middle": "Middle", "ring": "Ring", "pinky": "Pinky"}


def joint_map():
    """canonical joint -> Mixamo bone."""
    m = {"pelvis": "Hips", "spine_01": "Spine", "spine_02": "Spine1", "spine_03": "Spine2", "neck_01": "Neck",
         "head": "Head"}
    for s, S in _SIDE.items():
        m.update({f"clavicle_{s}": f"{S}Shoulder", f"upperarm_{s}": f"{S}Arm", f"lowerarm_{s}": f"{S}ForeArm",
                  f"hand_{s}": f"{S}Hand", f"thigh_{s}": f"{S}UpLeg", f"calf_{s}": f"{S}Leg",
                  f"foot_{s}": f"{S}Foot", f"ball_{s}": f"{S}ToeBase"})
        for f, F in _F.items():
            for k in (1, 2, 3):
                m[f"{f}_0{k}_{s}"] = f"{S}Hand{F}{k}"
    return m


class MixamoCharacter:
    """Worker profile and canonical skeleton of a Mixamo character. Duck-types character.Character where the
    solver needs it (worker, skeleton, id); the USD bake draws the frt body on this skeleton (write = None)."""

    write = None

    def __init__(self, path, wid=None, sex="M"):
        import sys
        tools = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "tools"))
        if tools not in sys.path:
            sys.path.insert(0, tools)
        from web_export import Rig

        from .workers import custom
        self.asset = path
        rig = Rig(path)
        pos = lambda b: np.asarray(rig.pos(b), float) * 100.0          # (the rig is in metres) -> cm
        floor = min(pos(f"{S}ToeBase")[1] for S in _SIDE.values()) - 1.6
        world = {"root": np.zeros(3)}
        for c, b in joint_map().items():
            if PRE + b in rig.idx:
                world[c] = pos(b) - np.array([0.0, floor, 0.0])
        off = np.array([world["pelvis"][0], 0.0, world["pelvis"][2]])  # pelvis over the origin
        world = {k: v - off for k, v in world.items()}
        world["root"] = np.zeros(3)
        top = float(pos("HeadTop_End")[1] - floor)
        tips = {"head_top": np.array([world["head"][0], top, world["head"][2]])}
        for s, S in _SIDE.items():
            for f, F in _F.items():
                end = PRE + f"{S}Hand{F}4"
                if end in rig.idx:
                    tips[f"{f}_tip_{s}"] = pos(f"{S}Hand{F}4") - np.array([0.0, floor, 0.0]) - off
                else:
                    a, b = world[f"{f}_02_{s}"], world[f"{f}_03_{s}"]
                    tips[f"{f}_tip_{s}"] = b + (b - a) * 0.85
            toe = pos(f"{S}Toe_End") - np.array([0.0, floor, 0.0]) - off
            tips[f"toe_{s}"] = np.array([toe[0], 1.5, toe[2]])
            foot = world[f"foot_{s}"]
            tips[f"heel_{s}"] = np.array([foot[0], 2.5, foot[2] - 5.5])
        for n in NAMES:
            if n not in world:
                world[n] = world[PARENT[n]].copy()
        for f in FINGERS:                                               # (every finger tip the solver asks for)
            for s in "lr":
                tips.setdefault(f"{f}_tip_{s}", world[f"{f}_03_{s}"])
        self.skeleton = Skeleton(world, tips)
        self.skeleton.rig = (os.path.abspath(path), floor / 100.0, off / 100.0)   # (web_export: joint for joint)
        self.skeleton.torso = torso_profile(path, self.skeleton, floor, off)   # (None: the solver's generic trunk)
        base = os.path.splitext(os.path.basename(path))[0]
        self.worker = custom(wid or f"mixamo_{base}", top, sex)
        self.worker.label = f"{base} ({top:.0f} cm)"
        self.id = self.worker.id


BLENDER = os.environ.get("FRT_BLENDER", r"C:/Program Files/Blender Foundation/Blender 5.2/blender.exe")
TRUNK = ("Hips", "Spine", "Spine1", "Spine2")


def skin_points(path):
    """The character's skinned mesh at rest (tools/blender/glb_skin.py, once per file, cached): points (cm) and the
    dominant bone name per point; None without Blender."""
    import hashlib
    import subprocess
    h = hashlib.sha1(open(path, "rb").read()).hexdigest()[:16]
    repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
    out = os.path.join(repo, "out", "people", "cache", f"skin_{h}.npz")
    if not os.path.isfile(out):
        if not os.path.isfile(BLENDER):
            return None
        os.makedirs(os.path.dirname(out), exist_ok=True)
        subprocess.run([BLENDER, "-b", "--python", os.path.join(repo, "tools", "blender", "glb_skin.py"), "--", path,
                        out], capture_output=True, timeout=600)
        if not os.path.isfile(out):
            return None
    d = np.load(out)
    names = [str(n)[len(PRE):] if str(n).startswith(PRE) else str(n) for n in d["names"]]
    return d["points"].astype(float), [names[i] for i in d["bone"]]


def torso_profile(path, sk, floor, off, step=4.0):
    """The trunk as the mesh has it (skin and clothes on the hips and spine bones), like character.Character's:
    [(t, half width, front, back)] along pelvis -> chest, cm from the spine. The arms keep out of this."""
    got = skin_points(path)
    if got is None:
        return None
    P, bone = got
    keep = np.array([b in TRUNK for b in bone])
    V = P[keep] - np.array([0.0, floor, 0.0]) - off
    a = sk.world["pelvis"]
    b = sk.world["spine_03"] + np.array([0.0, 6.0, 0.0])
    top = min(b[1], sk.world["upperarm_l"][1] - 10.0)                    # up to the armpits (higher, the spine's
    #                                                                      skin is the shoulder: it would read as chest)
    out = []
    for y in np.arange(a[1], top + 1e-6, step):
        band = V[np.abs(V[:, 1] - y) < step * 0.75]
        if len(band) < 12:
            continue
        t_ = float((y - a[1]) / (b[1] - a[1]))
        c = a + (b - a) * t_
        dx, dz = band[:, 0] - c[0], band[:, 2] - c[2]
        out.append((round(t_, 3), float(np.percentile(np.abs(dx), 96)), float(max(1.0, np.percentile(dz, 97))),
                    float(max(1.0, np.percentile(-dz, 97)))))
    return out or None


def load(path, **kw):
    return MixamoCharacter(path, **kw)
