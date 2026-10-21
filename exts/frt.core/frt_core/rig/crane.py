"""Pose and animate a bridge crane built by the catalog (items_crane): move a bridge along the runway, a hoist
along its bridge and its hook up and down. Positions are crane-local (cm): bridge z = centre of the bridge along the
runway, hoist x = trolley position across the span, hook = hook height above the floor (as the hook_height param).
Values are clamped to the travel limits stored on the rig; the chain is re-laid link by link, the festoon trolleys
follow the hoist and the hook attachment points move with the hook.

    pose(stage, "/World/Equipment/BC1", bridge=1, z=150, hoist=2, x=-120, hook=60, time=48)
    animate(stage, "/World/Equipment/BC1", [(0, {"z": -200}), (96, {"z": 200, "hook": 60})], hoist=1)
"""
from pxr import Gf, Sdf, Usd, UsdGeom

from ..catalog.items_crane import festoon_xs, lay_chain


def _t(time):
    return Usd.TimeCode.Default() if time is None else Usd.TimeCode(time)


def _rig(prim, name, default=None):
    a = prim.GetAttribute(f"frt:rig:{name}")
    return a.Get() if a and a.HasValue() else default


def _translate(prim):
    return next(op for op in UsdGeom.Xformable(prim).GetOrderedXformOps()
                if op.GetOpType() == UsdGeom.XformOp.TypeTranslate and op.GetOpName() == "xformOp:translate")


def _op(prim, kind, suffix=None):
    for op in UsdGeom.Xformable(prim).GetOrderedXformOps():
        if op.GetOpType() == kind and (suffix is None or op.GetOpName().endswith(":" + suffix)) and not op.IsInverseOp():
            return op
    return None


def _clamp(v, lo, hi):
    return max(lo, min(hi, float(v)))


def limits(stage, path):
    """Travel limits of every bridge / hoist / hook: {"Bridge_1": {"z": (lo, hi, rest), "Hoist_1": {...}}}."""
    out = {}
    for b in Usd.PrimRange(stage.GetPrimAtPath(path)):
        if b.GetName().startswith("Bridge_") and _rig(b, "z_min") is not None:
            d = out[b.GetName()] = {"z": (_rig(b, "z_min"), _rig(b, "z_max"), _rig(b, "z_rest"))}
            for h in b.GetChildren():
                if h.GetName().startswith("Hoist_"):
                    blk = h.GetChild("Block")
                    d[h.GetName()] = {"x": (_rig(h, "x_min"), _rig(h, "x_max"), _rig(h, "x_rest")),
                                      "hook": (_rig(blk, "hook_min"), _rig(blk, "hook_max"), _rig(blk, "hook_rest"))}
    return out


def pose(stage, path, bridge=1, z=None, hoist=1, x=None, hook=None, time=None):
    """Move one bridge (z), one of its hoists (x) and that hoist's hook (hook); None leaves a value as it is.
    Returns the applied (clamped) crane-local values."""
    t = _t(time)
    b = stage.GetPrimAtPath(f"{path}/Bridge_{bridge}")
    if not b:
        raise KeyError(f"{path} has no Bridge_{bridge} (is it a rigged bridge_crane?)")
    h = b.GetChild(f"Hoist_{hoist}")
    blk = h.GetChild("Block")
    bt, ht, kt = _translate(b), _translate(h), _translate(blk)
    dz = bt.Get(t)[2] if z is None else _clamp(z, _rig(b, "z_min"), _rig(b, "z_max")) - _rig(b, "z_rest")
    dx = ht.Get(t)[0] if x is None else _clamp(x, _rig(h, "x_min"), _rig(h, "x_max")) - _rig(h, "x_rest")
    dy = kt.Get(t)[1] if hook is None else _clamp(hook, _rig(blk, "hook_min"), _rig(blk, "hook_max")) - _rig(blk, "hook_rest")
    bt.Set(Gf.Vec3d(0, 0, dz), t)
    ht.Set(Gf.Vec3d(dx, 0, 0), t)
    kt.Set(Gf.Vec3d(0, dy, 0), t)
    hk = _rig(blk, "hook_rest") + dy
    chain = h.GetChild("Chain")
    if chain:                                                   # chain hoist: re-lay the links to the block
        lay_chain(UsdGeom.PointInstancer(chain), hk + 42, t)
    ropes = h.GetChild("Ropes")
    if ropes:                                                   # crab: wire ropes stretch from the drum
        top, rest = _rig(ropes, "top"), _rig(ropes, "bot_rest")
        _op(ropes, UsdGeom.XformOp.TypeScale).Set(Gf.Vec3f(1, (top - (rest + dy)) / (top - rest), 1), t)
    fest = b.GetChild("Festoon")
    if fest and _rig(fest, "hoist") == h.GetName():
        _festoon(fest, _rig(h, "x_rest") + dx, t)
    att = stage.GetPrimAtPath(f"{path}/Attach")
    if att:                                                     # attachment points follow the parts
        for name, d in ((_rig(h, "key"), (dx, dy, dz)), (_rig(h, "pendant"), (dx, 0, dz)), (_rig(b, "key"), (0, 0, dz))):
            a = att.GetChild(name or "")
            if a:
                if not a.GetAttribute("frt:rig:rest"):
                    a.CreateAttribute("frt:rig:rest", Sdf.ValueTypeNames.Double3).Set(_translate(a).Get())
                r = a.GetAttribute("frt:rig:rest").Get()
                _translate(a).Set(Gf.Vec3d(r[0] + d[0], r[1] + d[1], r[2] + d[2]), t)
    return {"z": _rig(b, "z_rest") + dz, "x": _rig(h, "x_rest") + dx, "hook": hk}


def _festoon(fest, run, t):
    park, rest = _rig(fest, "park"), list(_rig(fest, "rest"))
    xs = festoon_xs(park, run, len(rest))
    for k, x in enumerate(xs):
        p = fest.GetChild(f"Trolley_{k}")
        _translate(p).Set(Gf.Vec3d(x - rest[k], 0, 0), t)
    for k in range(len(xs) - 1):
        p = fest.GetChild(f"Loop_{k}")
        _translate(p).Set(Gf.Vec3d(xs[k] - rest[k], 0, 0), t)
        _op(p, UsdGeom.XformOp.TypeScale).Set(Gf.Vec3f((xs[k + 1] - xs[k]) / (rest[k + 1] - rest[k]), 1, 1), t)


def _channel(keys, f, ease):
    """Value of one channel at frame f from its [(time, value)] keys: eased between keys, held outside them."""
    if f <= keys[0][0]:
        return keys[0][1]
    for (t0, v0), (t1, v1) in zip(keys, keys[1:]):
        if f <= t1:
            u = (f - t0) / (t1 - t0)
            u = u * u * (3 - 2 * u) if ease else u                    # smoothstep: start and stop gently
            return v0 + (v1 - v0) * u
    return keys[-1][1]


def animate(stage, path, keys, bridge=1, hoist=1, ease=True, step=1.0):
    """Keyframes [(time, {"z": .., "x": .., "hook": ..}), ...] for one bridge and hoist. Each channel moves between
    the keys that set it (smoothstep eased, so drives start and stop gently) and holds outside them; the crane is
    baked every `step` time codes so the chain, festoon and attach points follow exactly. Extends the stage's time
    range to cover the keys."""
    keys = sorted(keys, key=lambda k: k[0])
    if not keys:
        return
    ch = {c: [(t, float(v[c])) for t, v in keys if c in v] for c in ("z", "x", "hook")}
    t0, t1 = keys[0][0], keys[-1][0]
    f = t0
    while f <= t1 + 1e-9:
        pose(stage, path, bridge=bridge, hoist=hoist, time=f,
             **{c: (_channel(k, f, ease) if k else None) for c, k in ch.items()})
        f += step
    if stage.HasAuthoredTimeCodeRange():
        t0, t1 = min(stage.GetStartTimeCode(), t0), max(stage.GetEndTimeCode(), t1)
    stage.SetStartTimeCode(t0)
    stage.SetEndTimeCode(t1)
