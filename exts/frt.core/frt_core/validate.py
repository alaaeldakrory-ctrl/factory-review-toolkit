"""Scene validation: the problems we kept hitting by hand, found automatically.

    findings = validate(stage)            # FindingList
    checks: units (stage up axis / metersPerUnit), floating and sunk objects, overlapping equipment, implausible sizes
    (usually wrong units), missing and absolute asset paths.

Works on tagged objects (frt:role, see tags.py); run tags.auto_tag() first on scenes built by hand.
"""
import os

from pxr import Gf, Sdf, Usd, UsdGeom, UsdUtils

from . import METERS_PER_UNIT, UP_AXIS
from . import rules as R
from . import tags as T
from .findings import Finding, FindingList

STRUCTURE = {"floor", "building", "aisle", "walkway", "zone", "keepout", "camera", "column", "wall", "door", "dock", "exit"}


def _boxes(stage, roles=None, time=Usd.TimeCode.Default()):
    bc = UsdGeom.BBoxCache(time, ["default", "render"], useExtentsHint=True)
    out = []
    for p in T.tagged(stage):
        role = T.role_of(p)
        if roles and role not in roles:
            continue
        r = bc.ComputeWorldBound(p).ComputeAlignedRange()
        if r.IsEmpty():
            continue
        out.append((p, role, r.GetMin(), r.GetMax()))
    return out


def check_units(stage, fl):
    up, mpu = UsdGeom.GetStageUpAxis(stage), UsdGeom.GetStageMetersPerUnit(stage)
    if up != UP_AXIS or abs(mpu - METERS_PER_UNIT) > 1e-9:
        fl.add(Finding("validate.units", "warning",
                       f"Stage is {up}-up in {mpu:g} m units; toolkit scenes are {UP_AXIS}-up in centimetres. "
                       f"Wrap it with units.place_asset instead of editing it.", rule="", value=mpu))


def check_support(stage, fl, rules):
    tol_up = rules.get("validate.floating_tolerance_cm", 1.0)
    tol_dn = rules.get("validate.sunk_tolerance_cm", 1.0)
    boxes = [b for b in _boxes(stage) if b[1] not in STRUCTURE]
    tops = [(b[0], b[2], b[3]) for b in boxes]
    for p, role, lo, hi in boxes:
        base = lo[1]
        if abs(base) <= tol_up and base >= -tol_dn:
            continue
        on = p.GetAttribute("frt:on")
        if on and on.Get():                                   # placed on a named support (a unit on a ramp)
            continue
        # resting on something? a box whose top is at our base and overlaps our footprint
        supported = False
        for q, qlo, qhi in tops:
            if q == p:
                continue
            if qlo[0] < hi[0] and qhi[0] > lo[0] and qlo[2] < hi[2] and qhi[2] > lo[2] and abs(qhi[1] - base) <= tol_up * 3:
                supported = True
                break
        if supported:
            continue
        c = ((lo[0] + hi[0]) / 2, base, (lo[2] + hi[2]) / 2)
        if base > tol_up:
            fl.add(Finding("validate.floating", "warning", f"{p.GetName()} ({role}) floats {base:.1f} cm above the floor "
                           f"with nothing under it", [str(p.GetPath())], c, round(base, 2), tol_up, "cm",
                           "validate.floating_tolerance_cm"))
        elif base < -tol_dn:
            fl.add(Finding("validate.sunk", "warning", f"{p.GetName()} ({role}) sits {-base:.1f} cm below the floor",
                           [str(p.GetPath())], c, round(base, 2), -tol_dn, "cm", "validate.sunk_tolerance_cm"))


def check_overlaps(stage, fl, rules):
    ignore = set(rules.get("validate.overlap_ignore_roles", [])) | STRUCTURE
    min_v = rules.get("validate.overlap_min_cm3", 2000.0)
    overhead = set(rules.get("validate.overhead_roles", []))
    boxes = [b for b in _boxes(stage) if b[1] not in ignore]
    for i in range(len(boxes)):
        p, rp, lo, hi = boxes[i]
        for j in range(i + 1, len(boxes)):
            q, rq, qlo, qhi = boxes[j]
            onp, onq = p.GetAttribute("frt:on"), q.GetAttribute("frt:on")
            if (onp and onp.Get() == T.id_of(q)) or (onq and onq.Get() == T.id_of(p)):
                continue                                      # one stands on the other
            if (rp in overhead) != (rq in overhead):          # a table under a crane is the point of the crane
                continue
            d = [min(hi[k], qhi[k]) - max(lo[k], qlo[k]) for k in range(3)]
            if min(d) <= 0:
                continue
            v = d[0] * d[1] * d[2]
            if v < min_v:
                continue
            # one sitting on top of the other (pallet on a rack level) is support, not clash: tiny vertical overlap
            if d[1] < 2.0:
                continue
            c = tuple((max(lo[k], qlo[k]) + min(hi[k], qhi[k])) / 2 for k in range(3))
            fl.add(Finding("validate.overlap", "warning",
                           f"{p.GetName()} ({rp}) and {q.GetName()} ({rq}) overlap by {d[0]:.0f} x {d[2]:.0f} cm in plan",
                           [str(p.GetPath()), str(q.GetPath())], c, round(v / 1e6, 3), min_v / 1e6, "m3",
                           "validate.overlap_min_cm3"))


def check_sizes(stage, fl, rules):
    table = rules.get("validate.size", {})
    for p, role, lo, hi in _boxes(stage):
        lim = table.get(role)
        if not lim:
            continue
        plan = max(hi[0] - lo[0], hi[2] - lo[2])
        height = hi[1] - lo[1]
        for name, v, (a, b) in (("plan size", plan, lim["plan"]), ("height", height, lim["height"])):
            if v < a or v > b:
                hint = " (wrong units?)" if v < a / 5 or v > b * 5 else ""
                c = ((lo[0] + hi[0]) / 2, hi[1], (lo[2] + hi[2]) / 2)
                fl.add(Finding("validate.size", "warning", f"{p.GetName()} ({role}) {name} {v:.0f} cm is outside "
                               f"{a}-{b} cm{hint}", [str(p.GetPath())], c, round(v, 1), b if v > b else a, "cm",
                               f"validate.size.{role}"))


BUILTIN_MDL = ("OmniPBR", "OmniGlass", "OmniSurface", "OmniHair", "OmniEmissive", "OmniUe4", "gltf/pbr", "nvidia/", "omni_")


def _builtin(path):
    """Material definitions every RTX renderer ships (OmniPBR.mdl, ...): not files a scene has to carry."""
    p = path.replace("\\", "/")
    return p.lower().endswith(".mdl") and "/" not in p.strip("./") or any(p.startswith(b) for b in BUILTIN_MDL)


def check_assets(stage, fl, rules, package_root=None):
    layer = stage.GetRootLayer()
    try:
        layers, assets, unresolved = UsdUtils.ComputeAllDependencies(layer.identifier)
    except Exception as e:   # noqa: BLE001
        fl.add(Finding("validate.assets", "warning", f"Could not walk dependencies: {e}"))
        return
    for u in unresolved:
        if _builtin(u):
            continue
        fl.add(Finding("validate.missing_asset", "critical", f"Missing file: {u}", value=None))
    sev = rules.get("validate.absolute_paths", "warning")
    root = os.path.normcase(os.path.abspath(package_root or os.path.dirname(layer.realPath or ".")))
    outside = []
    for p in [l.realPath for l in layers] + list(assets):
        if not p:
            continue
        ap = os.path.normcase(os.path.abspath(p))
        if not ap.startswith(root):
            outside.append(p)
    if outside:
        sample = ", ".join(os.path.basename(x) for x in outside[:4])
        fl.add(Finding("validate.outside_files", sev, f"{len(outside)} files live outside the scene folder "
                       f"(e.g. {sample}); the scene will not open on another PC until it is packaged",
                       value=len(outside), rule="validate.absolute_paths"))


def validate(stage, rules=None, package_root=None, checks=("units", "support", "overlaps", "sizes", "assets")):
    rules = rules or R.load()
    fl = FindingList()
    if "units" in checks:
        check_units(stage, fl)
    if "support" in checks:
        check_support(stage, fl, rules)
    if "overlaps" in checks:
        check_overlaps(stage, fl, rules)
    if "sizes" in checks:
        check_sizes(stage, fl, rules)
    if "assets" in checks:
        check_assets(stage, fl, rules, package_root)
    return fl
