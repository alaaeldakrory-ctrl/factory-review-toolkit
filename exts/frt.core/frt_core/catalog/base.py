"""Equipment catalog framework.

Every catalog item builds itself from a few parameters at real size, with proper materials, a role tag and named
attachment points (where a pallet sits, where a hand grips, where a hook attaches). Item local frame: origin at the
centre of the footprint on the floor, +X along the length, +Z toward the front (operator / pick side), Y up, cm.

    place(stage, "/World/Items", "industrial_table", "T1", pos=(x, 0, z), yaw=90, params={"length": 200})
"""
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from pxr import Gf, Sdf, UsdGeom

from .. import tags as T
from ..geom import Shapes
from ..materials import MaterialLibrary

CATALOG = {}


@dataclass
class Param:
    default: Any
    doc: str = ""
    lo: Optional[float] = None
    hi: Optional[float] = None
    unit: str = ""
    choices: Tuple = ()


@dataclass
class Item:
    key: str
    role: str
    title: str
    params: Dict[str, Param] = field(default_factory=dict)
    source: str = "generator"           # generator / pack / mixed
    licence: str = "toolkit"            # what may be redistributed (pack assets stay local)

    def resolve(self, given=None):
        out = {k: p.default for k, p in self.params.items()}
        for k, v in (given or {}).items():
            if k not in self.params:
                raise KeyError(f"{self.key}: unknown parameter {k!r}; known: {', '.join(self.params)}")
            p = self.params[k]
            if p.choices and v not in p.choices:
                raise ValueError(f"{self.key}.{k} must be one of {p.choices}")
            if p.lo is not None and isinstance(v, (int, float)) and not (p.lo <= v <= p.hi):
                raise ValueError(f"{self.key}.{k}={v} outside {p.lo}..{p.hi} {p.unit}")
            out[k] = v
        return out

    def clamp(self, given):
        """Numbers brought into their ranges (drawings are not always realistic): (params, [notes])."""
        out, notes = dict(given or {}), []
        for k, v in list(out.items()):
            p = self.params.get(k)
            if p is None:
                del out[k]
                notes.append(f"{self.key}: no parameter {k!r}, ignored")
            elif p.lo is not None and isinstance(v, (int, float)) and not isinstance(v, bool) and not p.lo <= v <= p.hi:
                out[k] = min(max(v, p.lo), p.hi)
                notes.append(f"{self.key}.{k}={v:g} {p.unit} outside {p.lo:g}..{p.hi:g}, used {out[k]:g}")
        return out, notes

    def footprint(self, p):
        """(length along X, depth along Z) in cm."""
        raise NotImplementedError

    def build(self, stage, path, p, lib):
        """Write geometry under `path` (an Xform at the item origin). Returns {attach name: (x, y, z) local}."""
        raise NotImplementedError


def register(item):
    CATALOG[item.key] = item
    return item


def place(stage, parent, key, item_id, pos=(0, 0, 0), yaw=0.0, params=None, lib=None, source="recipe"):
    item = CATALOG[key]
    p = item.resolve(params)
    lib = lib or MaterialLibrary(stage)
    path = f"{parent}/{item_id}"
    prim = stage.DefinePrim(path, "Xform")
    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(*pos))
    xf.AddRotateYOp().Set(float(yaw))
    attach = item.build(stage, path, p, lib) or {}
    if attach:
        UsdGeom.Scope.Define(stage, f"{path}/Attach")
        for name, (x, y, z) in attach.items():
            a = UsdGeom.Xform.Define(stage, f"{path}/Attach/{name}")
            a.AddTranslateOp().Set(Gf.Vec3d(x, y, z))
    T.tag(prim, item.role, id=item_id, source=source, catalog=key, params=p)
    return prim, p


def shapes_to_stage(stage, path, shapes, lib, names=None):
    mats = lib.refs(names or list(shapes.groups))
    return shapes.build(stage, f"{path}/Geom", mats)


def new_shapes():
    return Shapes()
