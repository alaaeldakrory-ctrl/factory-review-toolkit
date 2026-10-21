"""Quick offline picture of a built scene through one of its cameras (/World/Cameras/<name>), or a custom view.

    python tools/preview_scene.py scene.usd out.png [Overview|Plan|"ex,ey,ez;ax,ay,az"] [--hide /World/Building]
"""
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pxr import Gf, Usd, UsdGeom   # noqa: E402
import preview as PV   # noqa: E402


def main(scene, out, view="Overview", hide=(), size=(1280, 720), cut=None):
    st = Usd.Stage.Open(scene)
    if ";" in view:
        e, a = [Gf.Vec3d(*[float(v) for v in q.split(",")]) for q in view.split(";")]
        eye, at, fov = e, a, 60.0
    else:
        cam = st.GetPrimAtPath(f"/World/Cameras/{view.capitalize() if view.islower() else view}")
        m = UsdGeom.Xformable(cam).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        eye = m.ExtractTranslation()
        fwd = -Gf.Vec3d(m[2][0], m[2][1], m[2][2]).GetNormalized()
        at = eye + fwd * 1000.0
        f = UsdGeom.Camera(cam).GetFocalLengthAttr().Get() or 24.0
        fov = math.degrees(2 * math.atan(20.955 / 2 / f))
    geo = PV.collect(st, skip=tuple(hide))
    if cut is None:
        cut = 850.0 if eye[1] > 1100 else None          # looking from above the roof: cut the roof away
    if cut:
        geo = [g for g in geo if g[0][:, 1].min() < cut]
    PV.render(geo, out, tuple(eye), tuple(at), size=size, fov=fov)
    print("->", out, f"({len(geo)} meshes)")


if __name__ == "__main__":
    a = sys.argv[1:]
    hide = a[a.index("--hide") + 1].split(",") if "--hide" in a else ()
    cut = float(a[a.index("--cut") + 1]) if "--cut" in a else None
    pos = [x for i, x in enumerate(a) if not x.startswith("--") and (i == 0 or a[i - 1] not in ("--hide", "--cut"))]
    main(pos[0], pos[1], pos[2] if len(pos) > 2 else "Overview", hide, cut=cut)
