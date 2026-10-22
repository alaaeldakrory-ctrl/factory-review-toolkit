"""Blender (headless) script: make the crew's bodies with MPFB and export them for frt_core.people.

    blender -b --factory-startup -noaudio --python tools/blender/mpfb_workers.py -- <out_dir> [W05F,W50M,W95M]

Per worker: MPFB human with macros (gender, muscle, weight, age) and the height macro bisected to the target
stature; helper geometry removed; game-engine rig (same joint names as the frt canonical skeleton); the body posed
into the frt rest pose (T-pose: arms along +-X, palms down, fingers straight) and that pose applied as the rest
pose. Exported as <id>_body.npz: points (Y-up cm, facing +Z, left +X), face counts / indices, UVs, skin weights
(joint names, up to 8 influences per point) and the joints' rest head / tail positions.
"""
import json
import os
import sys

import bpy
import addon_utils
from mathutils import Matrix, Vector

addon_utils.enable("bl_ext.blender_org.mpfb", default_set=True)
from bl_ext.blender_org.mpfb.services.humanservice import HumanService            # noqa: E402
from bl_ext.blender_org.mpfb.services.targetservice import TargetService          # noqa: E402

CREW = {"W05F": dict(gender=0.0, stature=151.0, muscle=0.45, weight=0.45, age=0.45),
        "W50M": dict(gender=1.0, stature=174.0, muscle=0.55, weight=0.5, age=0.5),
        "W95M": dict(gender=1.0, stature=185.5, muscle=0.6, weight=0.55, age=0.5)}
FINGERS = ("index", "middle", "ring", "pinky")


def conv(v):
    """Blender world (Z-up m, the human faces -Y, left +X) -> frt (Y-up cm, faces +Z, left +X). det = +1."""
    return (v.x * 100.0, v.z * 100.0, -v.y * 100.0)


def height_of(obj):
    dg = bpy.context.evaluated_depsgraph_get()
    ev = obj.evaluated_get(dg)
    zs = [(obj.matrix_world @ v.co).z for v in ev.data.vertices]
    return (max(zs) - min(zs)) * 100.0


def make(spec, h):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    mac = TargetService.get_default_macro_info_dict()
    mac.update({k: spec[k] for k in ("gender", "muscle", "weight", "age")})
    mac["height"] = h
    return HumanService.create_human(macro_detail_dict=mac, feet_on_ground=True)


def apply_modifiers(obj, keep=("ARMATURE",)):
    bpy.context.view_layer.objects.active = obj
    for m in list(obj.modifiers):
        if m.type not in keep:
            try:
                bpy.ops.object.modifier_apply(modifier=m.name)
            except Exception as e:                                      # shape-keyed meshes: bake keys first
                print("modifier", m.name, e)


def bake_shape_keys(obj):
    if obj.data.shape_keys:
        bpy.context.view_layer.objects.active = obj
        obj.shape_key_add(name="__mix", from_mix=True)
        keys = obj.data.shape_keys.key_blocks
        mix = keys["__mix"].data
        for i, v in enumerate(obj.data.vertices):
            v.co = mix[i].co
        for k in list(keys):
            obj.shape_key_remove(k)


def align_bone(rig, name, direction, up=None):
    """Pose bone `name` so its head->tail points along world `direction` (and its Z axis towards `up`)."""
    pb = rig.pose.bones[name]
    bpy.context.view_layer.update()
    M = rig.matrix_world @ pb.matrix
    head = M.translation.copy()
    y = Vector(direction).normalized()
    if up is None:
        z = (M.to_3x3() @ Vector((0, 0, 1)))
    else:
        z = Vector(up)
    z = (z - y * z.dot(y)).normalized()
    x = y.cross(z).normalized()
    R = Matrix((x, y, z)).transposed()
    W = R.to_4x4()
    W.translation = head
    pb.matrix = rig.matrix_world.inverted() @ W
    bpy.context.view_layer.update()


def roll_hand(rig, s):
    import math
    pb = rig.pose.bones[f"hand_{s}"]
    for _ in range(3):
        bpy.context.view_layer.update()
        w = rig.matrix_world
        idx = w @ rig.pose.bones[f"index_01_{s}"].head
        pky = w @ rig.pose.bones[f"pinky_01_{s}"].head
        M = w @ pb.matrix
        axis = (M.to_3x3() @ Vector((0, 1, 0))).normalized()
        d = idx - pky
        d = (d - axis * d.dot(axis)).normalized()
        want = Vector((0, -1, 0))
        want = (want - axis * want.dot(axis)).normalized()
        ang = math.atan2(axis.dot(d.cross(want)), d.dot(want))
        if abs(ang) < 1e-3:
            break
        R = Matrix.Rotation(ang, 4, axis)
        head = M.translation.copy()
        W = Matrix.Translation(head) @ R @ Matrix.Translation(-head) @ M
        pb.matrix = w.inverted() @ W


def t_pose(rig):
    """frt rest pose: arms out along +-X (Blender X), palms down (-Z), fingers straight; legs untouched."""
    bpy.context.view_layer.objects.active = rig
    bpy.ops.object.mode_set(mode="POSE")
    for s, sx in (("l", 1.0), ("r", -1.0)):
        for b in (f"upperarm_{s}", f"lowerarm_{s}", f"hand_{s}"):
            align_bone(rig, b, (sx, 0, 0), up=(0, 1 if s == "l" else -1, 0) if b != f"hand_{s}" else None)
        # hand: fingers along +-X, palm facing down - roll the hand until the index knuckle is in front of the
        # pinky knuckle (front = Blender -Y) and level with it
        align_bone(rig, f"hand_{s}", (sx, 0, 0))
        roll_hand(rig, s)
        for f in FINGERS:
            for i in (1, 2, 3):
                align_bone(rig, f"{f}_0{i}_{s}", (sx, 0, 0))
    bpy.ops.object.mode_set(mode="OBJECT")


def palm_down_check(rig, s):
    """Signed palm normal (Blender Z) from the index / pinky knuckles and the hand direction."""
    b = rig.data.bones
    pb = rig.pose.bones
    w = rig.matrix_world
    idx = w @ pb[f"index_01_{s}"].head
    pky = w @ pb[f"pinky_01_{s}"].head
    hand = (w @ pb[f"hand_{s}"].tail) - (w @ pb[f"hand_{s}"].head)
    n = hand.cross(idx - pky) if s == "l" else (idx - pky).cross(hand)
    return n.normalized()


def apply_pose_as_rest(rig, body):
    bpy.context.view_layer.objects.active = body
    arm = next(m for m in body.modifiers if m.type == "ARMATURE")
    bpy.ops.object.modifier_copy(modifier=arm.name)
    bpy.ops.object.modifier_apply(modifier=arm.name)              # bake the posed shape into the mesh
    bpy.context.view_layer.objects.active = rig
    bpy.ops.object.mode_set(mode="POSE")
    bpy.ops.pose.armature_apply(selected=False)
    bpy.ops.object.mode_set(mode="OBJECT")


def export(body, rig, path, extra):
    import numpy as np
    me = body.data
    W = body.matrix_world
    pts = np.array([conv(W @ v.co) for v in me.vertices], np.float32)
    counts = np.array([len(p.vertices) for p in me.polygons], np.int32)
    idx = np.array([i for p in me.polygons for i in p.vertices], np.int32)
    uv = None
    if me.uv_layers.active:
        uv = np.array([tuple(me.uv_layers.active.data[li].uv) for p in me.polygons for li in p.loop_indices], np.float32)
    bones = [b.name for b in rig.data.bones]
    gname = {g.index: g.name for g in body.vertex_groups}
    K = 8
    jidx = np.zeros((len(me.vertices), K), np.int32)
    jw = np.zeros((len(me.vertices), K), np.float32)
    for v in me.vertices:
        inf = sorted(((g.weight, gname[g.group]) for g in v.groups if gname.get(g.group) in bones and g.weight > 1e-4),
                     reverse=True)[:K]
        tot = sum(w for w, _ in inf) or 1.0
        for k, (w, n) in enumerate(inf):
            jidx[v.index, k] = bones.index(n)
            jw[v.index, k] = w / tot
    rw = rig.matrix_world
    heads = np.array([conv(rw @ b.head_local) for b in rig.data.bones], np.float32)
    tails = np.array([conv(rw @ b.tail_local) for b in rig.data.bones], np.float32)
    parents = [b.parent.name if b.parent else "" for b in rig.data.bones]
    np.savez_compressed(path, points=pts, counts=counts, indices=idx, uv=uv if uv is not None else np.zeros((0, 2)),
                        joint_indices=jidx, joint_weights=jw, joints=np.array(bones), heads=heads, tails=tails,
                        parents=np.array(parents), meta=np.array(json.dumps(extra)))


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    out = argv[0] if argv else os.path.join(os.path.expanduser("~"), ".frt", "bodies")
    which = argv[1].split(",") if len(argv) > 1 else list(CREW)
    os.makedirs(out, exist_ok=True)
    for wid in which:
        spec = CREW[wid]
        lo, hi = 0.0, 1.0
        for _ in range(9):
            h = (lo + hi) / 2
            st = height_of(make(spec, h))
            lo, hi = (h, hi) if st < spec["stature"] else (lo, h)
        body = make(spec, (lo + hi) / 2)
        st = height_of(body)
        rig = HumanService.add_builtin_rig(body, "game_engine")
        bake_shape_keys(body)
        apply_modifiers(body)                                  # the helper mask: body geometry only
        t_pose(rig)
        palm = {s: tuple(round(c, 2) for c in palm_down_check(rig, s)) for s in ("l", "r")}
        apply_pose_as_rest(rig, body)
        export(body, rig, os.path.join(out, f"{wid}_body.npz"),
               {"id": wid, "stature": st, "macros": spec, "height_macro": (lo + hi) / 2, "palm_normal": palm})
        print("BODY", wid, round(st, 1), len(body.data.vertices), "verts", palm)


main()
