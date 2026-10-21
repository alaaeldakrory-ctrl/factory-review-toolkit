"""Revision sync: what changed between two drawing revisions (or between a drawing and the last build).

    changes = diff(old_plant, new_plant)        # Plant objects or <name>.plant.json paths / dicts
    for c in changes: print(c)                  # Change(kind="moved", what="equipment", id="T3", ...)
    fl = to_findings(changes)                   # same list as review findings (severity info), for the findings panel

Equipment is matched by its tag (T3, LT2, ...), stations by name. Small differences below the tolerances are
drawing noise, not changes.
"""
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Optional

from .findings import Finding, FindingList

MOVE_TOL_CM = 5.0
SIZE_TOL_CM = 5.0
ROT_TOL_DEG = 1.0


@dataclass
class Change:
    kind: str                   # added / removed / moved / resized / rotated / changed
    what: str                   # equipment / station / door / building
    id: str
    message: str
    old: Optional[dict] = None
    new: Optional[dict] = None
    location: Optional[tuple] = None    # plant cm, for "go to"
    fields: list = field(default_factory=list)

    def __str__(self):
        return f"{self.kind:8s} {self.what:9s} {self.id:10s} {self.message}"


def _as_dict(p):
    if isinstance(p, str):
        with open(p, encoding="utf-8") as fh:
            return json.load(fh)
    if isinstance(p, dict):
        return p
    return asdict(p)


def _m(v):
    return f"{v / 100:.2f} m"


def _equipment(a, b, out):
    old = {e["id"]: e for e in a.get("equipment", [])}
    new = {e["id"]: e for e in b.get("equipment", [])}
    for k in sorted(old.keys() - new.keys()):
        e = old[k]
        out.append(Change("removed", "equipment", k, f"{e['catalog']} removed", old=e, location=(e["x"], e["y"])))
    for k in sorted(new.keys() - old.keys()):
        e = new[k]
        out.append(Change("added", "equipment", k, f"{e['catalog']} added at ({_m(e['x'])}, {_m(e['y'])})", new=e,
                          location=(e["x"], e["y"])))
    for k in sorted(old.keys() & new.keys()):
        o, n = old[k], new[k]
        msgs, flds = [], []
        if o["catalog"] != n["catalog"]:
            msgs.append(f"type {o['catalog']} -> {n['catalog']}")
            flds.append("catalog")
        d = math.hypot(n["x"] - o["x"], n["y"] - o["y"])
        if d > MOVE_TOL_CM:
            msgs.append(f"moved {d:.0f} cm ({n['x'] - o['x']:+.0f}, {n['y'] - o['y']:+.0f})")
            flds.append("position")
        if abs(n["length"] - o["length"]) > SIZE_TOL_CM or abs(n["depth"] - o["depth"]) > SIZE_TOL_CM:
            msgs.append(f"size {o['length']:.0f} x {o['depth']:.0f} -> {n['length']:.0f} x {n['depth']:.0f} cm")
            flds.append("size")
        dr = (n["rot"] - o["rot"] + 180) % 360 - 180
        if abs(dr) > ROT_TOL_DEG:
            msgs.append(f"rotated {dr:+.0f} deg")
            flds.append("rot")
        if msgs:
            kind = "moved" if flds == ["position"] else "resized" if flds == ["size"] else \
                "rotated" if flds == ["rot"] else "changed"
            out.append(Change(kind, "equipment", k, f"{n['catalog']}: " + ", ".join(msgs), old=o, new=n,
                              location=(n["x"], n["y"]), fields=flds))


def _stations(a, b, out):
    old = {s["name"]: s for s in a.get("stations", [])}
    new = {s["name"]: s for s in b.get("stations", [])}
    ctr = lambda s: ((s["x0"] + s["x1"]) / 2, (s["y0"] + s["y1"]) / 2)
    for k in sorted(old.keys() - new.keys()):
        out.append(Change("removed", "station", k, "station removed", old=old[k], location=ctr(old[k])))
    for k in sorted(new.keys() - old.keys()):
        s = new[k]
        out.append(Change("added", "station", k, f"station added, {_m(s['x1'] - s['x0'])} x {_m(s['y1'] - s['y0'])}",
                          new=s, location=ctr(s)))
    for k in sorted(old.keys() & new.keys()):
        o, n = old[k], new[k]
        ow, oh, nw, nh = o["x1"] - o["x0"], o["y1"] - o["y0"], n["x1"] - n["x0"], n["y1"] - n["y0"]
        (ox, oy), (nx, ny) = ctr(o), ctr(n)
        msgs, flds = [], []
        if abs(nw - ow) > SIZE_TOL_CM or abs(nh - oh) > SIZE_TOL_CM:
            msgs.append(f"size {_m(ow)} x {_m(oh)} -> {_m(nw)} x {_m(nh)}")
            flds.append("size")
        elif math.hypot(nx - ox, ny - oy) > MOVE_TOL_CM:        # a resize moves the centre too; report it once
            msgs.append(f"moved {math.hypot(nx - ox, ny - oy):.0f} cm ({nx - ox:+.0f}, {ny - oy:+.0f})")
            flds.append("position")
        if msgs:
            out.append(Change("resized" if flds == ["size"] else "moved", "station", k, ", ".join(msgs), old=o, new=n,
                              location=(nx, ny), fields=flds))


def _doors(a, b, out):
    key = lambda d: (d["kind"], d["wall"], round(d["pos"] / 50.0))
    old = {key(d): d for d in a.get("doors", [])}
    new = {key(d): d for d in b.get("doors", [])}
    for k in sorted(old.keys() - new.keys()):
        d = old[k]
        out.append(Change("removed", "door", f"{d['kind']}@{d['wall']}", f"{d['kind']} on the {d['wall']} wall at "
                          f"{_m(d['pos'])} removed", old=d))
    for k in sorted(new.keys() - old.keys()):
        d = new[k]
        out.append(Change("added", "door", f"{d['kind']}@{d['wall']}", f"{d['kind']} on the {d['wall']} wall at "
                          f"{_m(d['pos'])} added", new=d))


def diff(old, new):
    a, b = _as_dict(old), _as_dict(new)
    out = []
    if abs(a["width"] - b["width"]) > SIZE_TOL_CM or abs(a["depth"] - b["depth"]) > SIZE_TOL_CM:
        out.append(Change("resized", "building", "BUILDING", f"{_m(a['width'])} x {_m(a['depth'])} -> "
                          f"{_m(b['width'])} x {_m(b['depth'])}", fields=["size"]))
    _doors(a, b, out)
    _stations(a, b, out)
    _equipment(a, b, out)
    return out


def to_findings(changes, plant, revision=""):
    """plant: the new revision (Plant, dict or path), to turn plant cm into scene points for "go to"."""
    p = _as_dict(plant)
    fl = FindingList()
    for c in changes:
        loc = (c.location[0] - p["width"] / 2, 0.0, p["depth"] / 2 - c.location[1]) if c.location else None
        fl.add(Finding(f"revision.{c.kind}", "info", f"{c.what} {c.id}: {c.message}", location=loc,
                       reference=revision))
    return fl
