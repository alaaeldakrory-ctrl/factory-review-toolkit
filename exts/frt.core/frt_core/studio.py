"""Product-shot studio for catalog items: a seamless dark cyclorama, a generated softbox environment (textures
.studio_hdri) with key / fill / rim area lights, a hero camera framed on the item, and an optional turntable.
Library previews, thumbnails and asset sheets come from here; the scene folder is portable like any other.

    build_studio("out/studio/lift_table.usd", "lift_table", {"cabinet": True}, turntable=240)
"""
import math

from pxr import Gf, Sdf, UsdGeom, UsdLux

from . import catalog as CAT
from . import units as U
from .geom import Shapes
from .materials import MaterialLibrary
from .usdio import safe_stage


def _cyclorama(stage, path, lib, half_w, front, back, radius, height):
    """Floor sweeping up into a back wall (profile in Y-Z, extruded along X)."""
    prof = [(front, 0.0), (-back, 0.0)]
    for k in range(1, 13):
        a = math.pi / 2 * k / 12
        prof.append((-back - radius * math.sin(a), radius - radius * math.cos(a)))
    prof.append((-back - radius, height))
    s = Shapes()
    for (z0, y0), (z1, y1) in zip(prof, prof[1:]):
        s.quad((-half_w, y0, z0), (half_w, y0, z0), (half_w, y1, z1), (-half_w, y1, z1), "studio_grey")
    s.build(stage, path, lib.refs(["studio_grey"]))


def _aim(xf, eye, at):
    m = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*at), Gf.Vec3d(0, 1, 0)).GetInverse()
    xf.ClearXformOpOrder()
    xf.AddTransformOp().Set(m)


def _rect(stage, path, eye, at, w, h, intensity, color=(1.0, 1.0, 1.0)):
    lt = UsdLux.RectLight.Define(stage, path)
    lt.CreateWidthAttr(float(w))
    lt.CreateHeightAttr(float(h))
    lt.CreateIntensityAttr(float(intensity))
    lt.CreateColorAttr(Gf.Vec3f(*color))
    _aim(UsdGeom.Xformable(lt), eye, at)              # rect lights emit along -Z, like a camera looks
    return lt


def build_studio(out, key, params=None, age=1.0, turntable=0, rig=True):
    """Write a studio scene for catalog item `key`. turntable > 0: the item turns 360 degrees over that many frames
    (24 fps); rig: rigged items also move (a lift table lifts, a crane runs its hook)."""
    from . import textures as GT
    with safe_stage(out) as st:
        U.setup_stage(st, fps=24.0)
        st.SetDefaultPrim(UsdGeom.Xform.Define(st, "/World").GetPrim())
        lib = MaterialLibrary(st, "/World/Looks", age=age)
        UsdGeom.Xform.Define(st, "/World/Asset")
        prim, p = CAT.place(st, "/World/Asset", key, "Item", (0, 0, 0), 0.0, params or {}, lib)
        dress(st, lib, prim, key=key, turntable=turntable, rig=rig)
    return out


def dress(st, lib, prim, key=None, turntable=0, rig=False):
    """Studio around what is under /World/Asset (prim = the thing to frame): cyclorama, softbox environment,
    key / fill / rim lights, square hero camera, optional turntable (and rig animation for known items)."""
    from . import textures as GT
    if True:
        bc = UsdGeom.BBoxCache(0, ["default", "render"])
        box = bc.ComputeWorldBound(prim).ComputeAlignedRange()
        mn, mx = box.GetMin(), box.GetMax()
        size = mx - mn
        R = max(size[0], size[2], size[1] * 0.8)
        c = (mn + mx) / 2
        _cyclorama(st, "/World/Studio/Cyc", lib, R * 6.0, R * 3.0, R * 1.2, R * 0.9, R * 3.0)
        hdr = GT.studio_hdri()
        dome = UsdLux.DomeLight.Define(st, "/World/Studio/Environment")
        dome.CreateTextureFileAttr().Set(Sdf.AssetPath(hdr["rel"]))
        dome.CreateTextureFormatAttr("latlong")
        dome.CreateIntensityAttr(350.0)
        top = mx[1]
        _rect(st, "/World/Studio/Key", (c[0] + R * 1.1, top + R * 0.9, c[2] + R * 1.2), (c[0], c[1], c[2]), R, R * 0.7, 9000.0,
              (1.0, 0.97, 0.93))
        _rect(st, "/World/Studio/Fill", (c[0] - R * 1.4, top * 0.6 + R * 0.3, c[2] + R * 0.8), (c[0], c[1], c[2]), R * 1.2,
              R * 0.8, 2500.0, (0.92, 0.96, 1.0))
        _rect(st, "/World/Studio/Rim", (c[0] - R * 0.4, top + R * 0.7, c[2] - R * 1.3), (c[0], top * 0.7, c[2]), R * 1.4,
              R * 0.25, 7000.0)
        if turntable and rig and key == "lift_table":                  # frame the whole travel, not the rest pose
            from .rig import lift_table as RL
            RL.pose(st, "/World/Asset/Item", lift=1.0)
            hi = UsdGeom.BBoxCache(0, ["default", "render"]).ComputeWorldBound(prim).ComputeAlignedRange()
            mx = Gf.Vec3d(mx[0], max(mx[1], hi.GetMax()[1]), mx[2])
            RL.pose(st, "/World/Asset/Item", lift=0.0)
            c = (mn + mx) / 2
            R = max(R, (mx[1] - mn[1]) * 0.8)
        cam = UsdGeom.Camera.Define(st, "/World/Studio/Hero")
        cam.CreateFocalLengthAttr(35.0)
        cam.CreateHorizontalApertureAttr(24.0)                           # square frame: same fit both ways
        cam.CreateVerticalApertureAttr(24.0)
        cam.CreateClippingRangeAttr(Gf.Vec2f(5.0, 1.0e5))
        # fit the item as it turns: a cylinder of its plan half-diagonal, its full height, with margin
        half = math.tan(math.atan(12.0 / 35.0))
        rxz = 0.5 * math.hypot(mx[0] - mn[0], mx[2] - mn[2])
        hh = 0.5 * (mx[1] - mn[1])
        el = math.radians(18.0)
        d = 1.18 * max(rxz / half, (hh * math.cos(el) + rxz * math.sin(el)) / half) + rxz
        dirn = (0.62 * math.cos(el), math.sin(el), 0.78 * math.cos(el))
        _aim(UsdGeom.Xformable(cam), (c[0] + d * dirn[0], c[1] + d * dirn[1], c[2] + d * dirn[2]), (c[0], c[1], c[2]))
        if turntable:
            st.SetStartTimeCode(0)
            st.SetEndTimeCode(turntable)
            xf = UsdGeom.Xformable(st.GetPrimAtPath("/World/Asset"))
            op = xf.AddRotateYOp()
            op.Set(0.0, 0)
            op.Set(360.0, turntable)
            if rig and key == "bridge_crane":                           # every bridge travels, hooks run
                from .rig import crane as RC
                lims = RC.limits(st, "/World/Asset/Item")
                nb = len(lims)
                for bi, (bname, lim) in enumerate(sorted(lims.items())):
                    z0, z1, zr = lim["z"]
                    span_z = (z1 - z0) / nb                                     # each bridge keeps to its share
                    lo, hi = z0 + span_z * bi + 0.1 * span_z, z0 + span_z * (bi + 1) - 0.1 * span_z
                    a_, b_ = (lo, hi) if bi % 2 == 0 else (hi, lo)
                    x0, x1, _ = lim["Hoist_1"]["x"]
                    h0, h1, _ = lim["Hoist_1"]["hook"]
                    RC.animate(st, "/World/Asset/Item", [
                        (0, {"z": a_, "x": x0 + (x1 - x0) * (0.3 if bi % 2 == 0 else 0.7), "hook": h1}),
                        (turntable * 0.4, {"z": b_, "hook": h0 + 20}),
                        (turntable * 0.7, {"x": x0 + (x1 - x0) * (0.6 if bi % 2 == 0 else 0.35)}),
                        (turntable, {"z": a_, "hook": h1})], bridge=bi + 1, hoist=1)
            if rig and key == "lift_table":
                from .rig import lift_table as RL
                RL.animate(st, "/World/Asset/Item", [(0, {"lift": 0}), (turntable * 0.35, {"lift": 1}),
                                                     (turntable * 0.65, {"lift": 1}), (turntable, {"lift": 0})])
