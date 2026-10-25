# blender -b --python tools/blender/glb_skin.py -- <character.glb> <out.npz>
# The character's skinned mesh at rest: points (cm, Y up, +Z forward) and the bone that moves each point most.
import sys
import bpy
import numpy as np
a = sys.argv[sys.argv.index("--") + 1:]
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=a[0])
pts, dom, names = [], [], []
dg = bpy.context.evaluated_depsgraph_get()
for ob in bpy.data.objects:
    if ob.type != "MESH" or not ob.vertex_groups:
        continue
    ev = ob.evaluated_get(dg)                                       # skinned by the armature as it stands (node rest)
    me = ev.to_mesh()
    M = ev.matrix_world
    groups = [g.name for g in ob.vertex_groups]
    for v, v0 in zip(me.vertices, ob.data.vertices):
        if not v0.groups:
            continue
        g = max(v0.groups, key=lambda e: e.weight)
        w = M @ v.co
        pts.append((w.x * 100.0, w.z * 100.0, -w.y * 100.0))           # Blender Z up -> Y up, cm
        n = groups[g.group]
        if n not in names:
            names.append(n)
        dom.append(names.index(n))
np.savez(a[1], points=np.array(pts, np.float32), bone=np.array(dom, np.int32), names=np.array(names))
print("SKIN OK", len(pts), "points", len(names), "bones")
