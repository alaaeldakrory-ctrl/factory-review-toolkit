"""Pose and animate a scissor lift table built by the catalog (items_lift): raise the deck to a height (or a 0..1
lift fraction); the arms turn about their pivots, the cylinder swings and extends, and what stands on the table
(frt:on = its id, beside it under the same parent) and its attach points ride up with the deck.

    pose(stage, "/World/Equipment/LT1", height=95, time=48)
    animate(stage, "/World/Equipment/LT1", [(0, {"lift": 0}), (72, {"lift": 1}), (200, {"lift": 1}), (260, {"height": 60})])
"""
from pxr import Gf, Sdf, Usd, UsdGeom

from ..catalog.items_lift import Geo
from .crane import _channel, _rig, _t, _translate, _op


def limits(stage, path):
    p = stage.GetPrimAtPath(path)
    return _rig(p, "h_min"), _rig(p, "h_max"), _rig(p, "h_rest")


def _geo(p):
    return Geo(*(_rig(p, k) for k in ("x_f", "y_b", "top_off", "La", "s_att", "ax", "ay", "rod0")))


def _follow(stage, prim, d, t):
    """Move a riding prim by d (cm, up) from where it was built."""
    op = _translate(prim)
    rest = prim.GetAttribute("frt:rig:rest")
    if not rest:
        rest = prim.CreateAttribute("frt:rig:rest", Sdf.ValueTypeNames.Double3)
        rest.Set(op.Get())
    r = rest.Get()
    op.Set(Gf.Vec3d(r[0], r[1] + d, r[2]), t)


def pose(stage, path, height=None, lift=None, time=None, carry=True):
    """Deck top to `height` cm (or `lift` 0..1 of its travel), clamped to the table's travel. Returns the height."""
    t = _t(time)
    p = stage.GetPrimAtPath(path)
    if not p or _rig(p, "h_min") is None:
        raise KeyError(f"{path} is not a rigged lift_table")
    lo, hi, rest = limits(stage, path)
    h = lo + (hi - lo) * float(lift) if height is None and lift is not None else (rest if height is None else height)
    h = max(lo, min(hi, float(h)))
    sol = _geo(p).solve(h)
    _translate(p.GetChild("Deck")).Set(Gf.Vec3d(0, h - rest, 0), t)
    for name, pos, ang in (("Arm_A", sol["a"], sol["theta"]), ("Arm_B", sol["b"], -sol["theta"])):
        a = p.GetChild(name)
        _translate(a).Set(Gf.Vec3d(pos[0], pos[1], 0), t)
        _op(a, UsdGeom.XformOp.TypeRotateZ).Set(float(ang), t)
    cyl = p.GetChild("Cylinder")
    _op(cyl, UsdGeom.XformOp.TypeRotateZ).Set(float(sol["phi"]), t)
    _translate(cyl.GetChild("Rod")).Set(Gf.Vec3d(sol["ext"], 0, 0), t)
    att = p.GetChild("Attach")
    if att and att.GetChild("platform"):
        _follow(stage, att.GetChild("platform"), h - rest, t)
    if carry:
        me = p.GetName()
        for sib in p.GetParent().GetChildren():
            a = sib.GetAttribute("frt:on")
            if a and a.Get() == me:
                _follow(stage, sib, h - rest, t)
    return h


def animate(stage, path, keys, ease=True, step=1.0, carry=True):
    """Keyframes [(time, {"height": cm} or {"lift": 0..1}), ...], eased between keys and baked every `step`."""
    p = stage.GetPrimAtPath(path)
    lo, hi, _ = limits(stage, path)
    ks = sorted(((tm, float(v["height"]) if "height" in v else lo + (hi - lo) * float(v["lift"])) for tm, v in keys),
                key=lambda k: k[0])
    if not ks or p is None:
        return
    f, t1 = ks[0][0], ks[-1][0]
    while f <= t1 + 1e-9:
        pose(stage, path, height=_channel(ks, f, ease), time=f, carry=carry)
        f += step
    t0 = ks[0][0]
    if stage.HasAuthoredTimeCodeRange():
        t0, t1 = min(stage.GetStartTimeCode(), t0), max(stage.GetEndTimeCode(), t1)
    stage.SetStartTimeCode(t0)
    stage.SetEndTimeCode(t1)
