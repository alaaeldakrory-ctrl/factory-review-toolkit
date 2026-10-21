"""Up-axis and unit normalisation. Scenes are Y up in centimetres; assets in any other convention (NVIDIA packs are
Z up, some CAD is in metres or millimetres) are referenced under a wrapper prim whose transform corrects them, so the
asset file itself is never edited.

    info = asset_info(path)                    # up axis, metersPerUnit, bounds in the asset's own units
    place_asset(stage, "/World/Items/C1", path, pos=(x, 0, z), yaw=90, pivot="floor")
"""
import math

from pxr import Gf, Usd, UsdGeom

from . import METERS_PER_UNIT, UP_AXIS

_INFO_CACHE = {}


def stage_frame(stage):
    return UsdGeom.GetStageUpAxis(stage), UsdGeom.GetStageMetersPerUnit(stage)


def setup_stage(stage, fps=30.0):
    UsdGeom.SetStageUpAxis(stage, UP_AXIS)
    UsdGeom.SetStageMetersPerUnit(stage, METERS_PER_UNIT)
    stage.SetTimeCodesPerSecond(fps)
    stage.SetFramesPerSecond(fps)


def asset_info(path):
    """{'up', 'mpu', 'min', 'max'} of an asset file (bounds in its own units and axes). Cached per path."""
    if path in _INFO_CACHE:
        return _INFO_CACHE[path]
    st = Usd.Stage.Open(path, Usd.Stage.LoadAll)
    bc = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
    root = st.GetDefaultPrim() or st.GetPseudoRoot()
    r = bc.ComputeWorldBound(root).ComputeAlignedRange()
    info = {"up": UsdGeom.GetStageUpAxis(st), "mpu": UsdGeom.GetStageMetersPerUnit(st) or 0.01,
            "min": tuple(r.GetMin()) if not r.IsEmpty() else (0, 0, 0),
            "max": tuple(r.GetMax()) if not r.IsEmpty() else (0, 0, 0)}
    _INFO_CACHE[path] = info
    return info


def correction(info, dst_mpu=METERS_PER_UNIT):
    """(scale, z_up) that bring an asset into Y-up, cm."""
    return info["mpu"] / dst_mpu, info["up"] == "Z"


def bounds_in_scene(info, dst_mpu=METERS_PER_UNIT):
    """Asset bounds after the correction: (min, max) as Y-up, cm tuples."""
    s, zup = correction(info, dst_mpu)
    lo, hi = info["min"], info["max"]
    if zup:     # rotateX(-90): (x, y, z) -> (x, z, -y)
        lo, hi = (lo[0], lo[2], -hi[1]), (hi[0], hi[2], -lo[1])
    return tuple(v * s for v in lo), tuple(v * s for v in hi)


def place_asset(stage, path, asset, pos=(0, 0, 0), yaw=0.0, pivot="floor", scale=(1, 1, 1)):
    """Reference `asset` at `path` corrected to Y-up cm. pivot: 'floor' puts the asset's lowest point on pos and
    centres it on pos in plan, 'origin' keeps the asset's own origin at pos. scale: extra non-uniform (x, y, z) scale
    in the scene frame (stretching a module to a bay). Returns (wrapper prim, scene-frame bounds of the asset)."""
    info = asset_info(asset)
    s, zup = correction(info)
    lo, hi = bounds_in_scene(info)
    prim = stage.DefinePrim(path, "Xform")
    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(*pos))
    xf.AddRotateYOp().Set(float(yaw))
    if tuple(scale) != (1, 1, 1):
        xf.AddScaleOp(opSuffix="fit").Set(Gf.Vec3f(*scale))
    if pivot == "floor":
        cx, cz = (lo[0] + hi[0]) / 2, (lo[2] + hi[2]) / 2
        xf.AddTranslateOp(opSuffix="pivot").Set(Gf.Vec3d(-cx, -lo[1], -cz))
    if s != 1.0:
        xf.AddScaleOp(opSuffix="units").Set(Gf.Vec3f(s, s, s))
    if zup:
        xf.AddRotateXOp(opSuffix="zup").Set(-90.0)
    inner = stage.DefinePrim(path + "/Asset", "Xform")
    inner.GetReferences().AddReference(asset)
    return prim, (lo, hi)


def yaw_dirs(yaw_deg):
    a = math.radians(yaw_deg)
    return (math.cos(a), -math.sin(a)), (math.sin(a), math.cos(a))     # local +X and +Z in world (x, z)
