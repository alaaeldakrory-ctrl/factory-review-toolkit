"""Interpret a cleaned drawing into a Plant: building outline, column grid, doors and docks, aisles, walkways and
recognised equipment, all in plant cm with the building's south-west corner at (0, 0), x east, y north.

DXF: layers and block names drive recognition (legend). PDF: no layers or blocks, so line weight / colour and the
labels inside rectangles do. Anything that looks like equipment but matches nothing goes to `unknown` for the user to
confirm instead of being guessed.

    dr = dxf_import.load("plant.dxf"); plant = interpret(dr)
"""
import fnmatch
import math
import os
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

try:
    import tomllib
except ImportError:
    import tomli as tomllib

from . import align

DEFAULT_LEGEND = os.path.join(os.path.dirname(os.path.abspath(__file__)), "legend_default.toml")


@dataclass
class Door:
    kind: str                 # dock / rollup / door
    wall: str                 # S / N / W / E
    pos: float                # centre, cm from the wall's west / south end
    width: float
    ref: str = ""


@dataclass
class Equip:
    id: str
    catalog: str
    x: float
    y: float
    length: float
    depth: float
    rot: float
    ref: str = ""
    label: str = ""


@dataclass
class Station:
    name: str                 # "Station 3"
    x0: float
    y0: float
    x1: float
    y1: float
    ref: str = ""
    label_inside: bool = True


@dataclass
class Plant:
    source: str
    width: float = 0.0        # x extent, cm
    depth: float = 0.0        # y extent, cm
    grid_x: List[float] = field(default_factory=list)
    grid_y: List[float] = field(default_factory=list)
    columns: List[Tuple[float, float]] = field(default_factory=list)
    doors: List[Door] = field(default_factory=list)
    aisles: List[Tuple[float, float, float, float]] = field(default_factory=list)
    walkways: List[Tuple[float, float, float, float]] = field(default_factory=list)
    stations: List[Station] = field(default_factory=list)
    equipment: List[Equip] = field(default_factory=list)
    unknown: List[dict] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


def load_legend(path=None):
    with open(path or DEFAULT_LEGEND, "rb") as fh:
        return tomllib.load(fh)


def _match(name, patterns):
    n = (name or "").upper()
    return any(fnmatch.fnmatch(n, p.upper()) for p in patterns)


def _cluster(values, tol):
    out = []
    for v in sorted(values):
        if out and abs(v - out[-1][-1]) <= tol:
            out[-1].append(v)
        else:
            out.append([v])
    return [sum(c) / len(c) for c in out]


def _rgb_kind(c):
    if not c:
        return None
    r, g, b = c[:3]
    if r > 0.95 and g > 0.95 and b > 0.95:
        return "white"
    if r > 0.7 and g > 0.5 and b < 0.3:
        return "yellow"
    if g > 0.45 and r < 0.4 and b < 0.45:
        return "green"
    if b > 0.45 and r < 0.3:
        return "blue"
    if max(r, g, b) < 0.15:
        return "black"
    return "other"


# ------------------------------------------------------------------------------------------ outline + grid
def _outline_dxf(dr, lg):
    walls = [s for s in dr.segs if _match(s.layer, lg["layers"]["walls"])]
    walls += [type("S", (), {"a": p.pts[i], "b": p.pts[(i + 1) % len(p.pts)]})
              for p in dr.polys if _match(p.layer, lg["layers"]["walls"]) for i in range(len(p.pts))]
    return walls


def _outline_pdf(dr):
    heavy = [s for s in dr.segs if s.width >= 1.0 and _rgb_kind(s.color) in ("black", None)]
    rects = [p for p in dr.polys if p.width >= 1.0 and not p.fill and _rgb_kind(p.color) in ("black", None)]
    out = list(heavy)
    for p in rects:
        for i in range(len(p.pts)):
            out.append(type("S", (), {"a": p.pts[i], "b": p.pts[(i + 1) % len(p.pts)]}))
    return out


def _columns(dr, lg):
    cols = []
    if dr.kind == "dxf":
        for i in dr.inserts:
            if _match(i.layer, lg["layers"]["columns"]) or _match(i.name, lg["blocks"]["columns"]):
                cols.append(i.pos)
    else:   # small filled squares
        for p in dr.polys:
            x0, y0, x1, y1 = p.bbox()
            if p.fill and 15 <= x1 - x0 <= 120 and 15 <= y1 - y0 <= 120 and abs((x1 - x0) - (y1 - y0)) < 10:
                cols.append(((x0 + x1) / 2, (y0 + y1) / 2))
    return cols


def interpret(dr, legend=None):
    lg = legend if isinstance(legend, dict) else load_legend(legend)
    plant = Plant(dr.source, notes=list(dr.notes))
    walls = _outline_dxf(dr, lg) if dr.kind == "dxf" else _outline_pdf(dr)
    cols = _columns(dr, lg)
    if not walls and not cols:
        raise ValueError("no walls or columns found; check the legend's layer names")
    # building extents: the column grid when the perimeter columns sit on the walls, else the walls' inner box
    if cols:
        gx = _cluster([c[0] for c in cols], 30.0)
        gy = _cluster([c[1] for c in cols], 30.0)
    else:
        gx, gy = [], []
    if walls:
        xs = [v for s in walls for v in (s.a[0], s.b[0])]
        ys = [v for s in walls for v in (s.a[1], s.b[1])]
        wx0, wx1, wy0, wy1 = min(xs), max(xs), min(ys), max(ys)
    else:
        wx0, wx1, wy0, wy1 = gx[0], gx[-1], gy[0], gy[-1]
    if gx and abs(gx[0] - wx0) < 100 and abs(gx[-1] - wx1) < 100:
        x0, x1, y0, y1 = gx[0], gx[-1], gy[0], gy[-1]
        plant.notes.append("building outline taken from the perimeter column grid")
    else:
        x0, x1, y0, y1 = wx0, wx1, wy0, wy1
        plant.notes.append("building outline taken from the wall lines")
    align.to_origin(dr, (x0, y0))
    plant.notes.append(dr.notes[-1])
    plant.width, plant.depth = round(x1 - x0, 1), round(y1 - y0, 1)
    plant.grid_x = [round(v - x0, 1) for v in gx]
    plant.grid_y = [round(v - y0, 1) for v in gy]
    plant.columns = [(c[0] - x0, c[1] - y0) for c in cols]
    _doors(dr, lg, plant)
    _floor(dr, lg, plant)
    _stations(dr, lg, plant)
    _equipment(dr, lg, plant)
    return plant


# ------------------------------------------------------------------------------------------ text phrases
def phrases(texts):
    """Join words that sit on one line next to each other ("STATION" "3" -> "STATION 3"); DXF texts are usually
    whole already, PDF words never are. Returns [(phrase, centre)]."""
    words = sorted(texts, key=lambda t: (-t.pos[1], t.pos[0]))
    out, used = [], set()
    for i, w in enumerate(words):
        if i in used:
            continue
        group, last = [w], w
        used.add(i)
        h = max(w.height, 1.0)
        for j in range(len(words)):
            if j in used:
                continue
            v = words[j]
            if abs(v.pos[1] - last.pos[1]) < 0.6 * h and 0 < v.pos[0] - last.pos[0] < (len(last.text) + 2.5) * h:
                group.append(v)
                used.add(j)
                last = v
        txt = " ".join(g.text.strip() for g in group)
        out.append((txt, (sum(g.pos[0] for g in group) / len(group), sum(g.pos[1] for g in group) / len(group))))
    return out


# ------------------------------------------------------------------------------------------ stations
def _stations(dr, lg, plant):
    cfg = lg.get("stations", {})
    pat = re.compile(cfg.get("label", r"^(STATION|STN|ST|CELL)\s*[-_#.]?\s*(\d{1,3}[A-Z]?)$"), re.I)
    min_size = cfg.get("min_size_cm", 250)
    reach = cfg.get("label_distance_cm", 250)
    # PDF phrases can run into a neighbouring label ("STATION 1 TABLE"): look for the name anywhere in the phrase
    word = re.compile(r"\b" + pat.pattern.lstrip("^").rstrip("$") + r"\b", re.I)
    labels = []
    for t, p in phrases(dr.texts):
        m = word.search(t.strip())
        if m:
            labels.append((m.group(0), p))
    used = set()
    rects = []
    for p in dr.polys:
        if not p.closed or p.fill or len(p.pts) != 4:
            continue
        x0, y0, x1, y1 = p.bbox()
        if x1 - x0 < min_size or y1 - y0 < min_size:
            continue
        if dr.kind == "dxf":
            if not _match(p.layer, lg["layers"].get("stations", [])):
                continue
        elif _rgb_kind(p.color) in ("yellow", "green", "black", "white", "blue") or p.width >= 1.0:
            continue
        rects.append((p, x0, y0, x1, y1))
    for p, x0, y0, x1, y1 in sorted(rects, key=lambda r: (r[1], r[2])):
        best, bd, inside = None, 1e9, False
        for k, (t, q) in enumerate(labels):
            if k in used:
                continue
            if x0 <= q[0] <= x1 and y0 <= q[1] <= y1:
                d, ins = 0.0, True
            else:
                dx = max(x0 - q[0], 0, q[0] - x1)
                dy = max(y0 - q[1], 0, q[1] - y1)
                d, ins = math.hypot(dx, dy), False
            if d < bd and d <= reach:
                best, bd, inside = k, d, ins
        if best is None:
            plant.unknown.append({"what": "rectangle without a station name", "pos": ((x0 + x1) / 2, (y0 + y1) / 2),
                                  "ref": p.ref})
            continue
        used.add(best)
        m = pat.match(labels[best][0].strip())
        plant.stations.append(Station(f"Station {m.group(2).upper()}", round(x0, 1), round(y0, 1), round(x1, 1),
                                      round(y1, 1), p.ref, inside))
    plant.stations.sort(key=lambda st: (len(st.name), st.name))


# ------------------------------------------------------------------------------------------ doors
def _wall_of(p, plant, tol=150.0):
    x, y = p
    d = {"S": abs(y), "N": abs(y - plant.depth), "W": abs(x), "E": abs(x - plant.width)}
    w = min(d, key=d.get)
    if d[w] > tol:
        return None, None
    return w, (x if w in "SN" else y)


def _door_kind(name, lg):
    for d in lg.get("doors", []):
        if _match(name, d["match"]):
            return d["kind"]
    return None


def _doors(dr, lg, plant):
    if dr.kind == "dxf":
        for i in dr.inserts:
            kind = _door_kind(i.name, lg)
            if not kind and not _match(i.layer, lg["layers"]["doors"]):
                continue
            kind = kind or "door"
            wall, pos = _wall_of(i.pos, plant)
            if not wall:
                continue
            w, _ = i.size()
            plant.doors.append(Door(kind, wall, round(pos, 1), round(w, 1), i.ref))
    else:   # white gap lines on the outline + the nearest door label
        gaps = [s for s in dr.segs if _rgb_kind(s.color) == "white" and s.length() > 50]
        for g in gaps:
            mid = ((g.a[0] + g.b[0]) / 2, (g.a[1] + g.b[1]) / 2)
            wall, pos = _wall_of(mid, plant)
            if not wall:
                continue
            best, bd = None, 1e9
            for t in dr.texts:
                k = _door_kind(t.text, lg)
                if k and math.dist(t.pos, mid) < bd:
                    best, bd = k, math.dist(t.pos, mid)
            plant.doors.append(Door(best if bd < 600 else "door", wall, round(pos, 1), round(g.length(), 1), g.ref))
    plant.doors.sort(key=lambda d: (d.wall, d.pos))


# ------------------------------------------------------------------------------------------ floor marking
def _floor(dr, lg, plant):
    for p in dr.polys:
        if not p.closed or p.fill:
            continue
        bb = p.bbox()
        if dr.kind == "dxf":
            if _match(p.layer, lg["layers"]["aisles"]):
                plant.aisles.append(tuple(round(v, 1) for v in bb))
            elif _match(p.layer, lg["layers"]["walkways"]):
                plant.walkways.append(tuple(round(v, 1) for v in bb))
        else:
            k = _rgb_kind(p.color)
            if k == "yellow":
                plant.aisles.append(tuple(round(v, 1) for v in bb))
            elif k == "green":
                plant.walkways.append(tuple(round(v, 1) for v in bb))


# ------------------------------------------------------------------------------------------ equipment
def _symbol(name, lg, labels=False):
    for s in lg.get("symbols", []):
        if labels:
            if any(name.upper() == l.upper() for l in s.get("labels", [])):
                return s["catalog"]
        elif _match(name, s["match"]):
            return s["catalog"]
    return None


def _nearest_id(dr, lg, p, used, max_d=None):
    pat = re.compile(lg.get("ids", {}).get("pattern", r"^[A-Z]{1,3}-?\d{1,3}$"))
    max_d = max_d or lg.get("ids", {}).get("max_distance_cm", 300)
    best, bd = None, max_d
    for t in dr.texts:
        s = t.text.strip()
        if not pat.match(s) or s in used:
            continue
        d = math.dist(t.pos, p)
        if d < bd:
            best, bd = s, d
    return best


def _equipment(dr, lg, plant):
    used = set()
    counter = {}

    def new_id(cat):
        counter[cat] = counter.get(cat, 0) + 1
        return f"{cat.upper()[:3]}_{counter[cat]}"
    if dr.kind == "dxf":
        for i in dr.inserts:
            cat = _symbol(i.name, lg)
            is_equip_layer = _match(i.layer, lg["layers"]["equipment"])
            if not cat:
                if is_equip_layer:
                    plant.unknown.append({"what": "block", "name": i.name, "pos": i.pos, "ref": i.ref})
                continue
            L, D = i.size()
            x0, y0, x1, y1 = i.bbox
            # the block's local centre, rotated, gives the footprint centre
            cxl, cyl = (x0 + x1) / 2 * i.scale[0], (y0 + y1) / 2 * i.scale[1]
            a = math.radians(i.rot)
            cx = i.pos[0] + cxl * math.cos(a) - cyl * math.sin(a)
            cy = i.pos[1] + cxl * math.sin(a) + cyl * math.cos(a)
            eid = _nearest_id(dr, lg, (cx - L / 2, cy + D / 2), used, max(L, D) + 300) or new_id(cat)
            used.add(eid)
            plant.equipment.append(Equip(eid, cat, round(cx, 1), round(cy, 1), round(L, 1), round(D, 1),
                                         round(i.rot % 360, 2), i.ref, i.name))
    else:
        for p in dr.polys:
            if p.fill or not p.closed:
                continue
            k = _rgb_kind(p.color)
            if k not in ("blue", None):
                continue
            x0, y0, x1, y1 = p.bbox()
            w, h = x1 - x0, y1 - y0
            if w < 30 or h < 30 or w > 5000 or h > 5000:
                continue
            inside = [t for t in dr.texts if x0 <= t.pos[0] <= x1 and y0 <= t.pos[1] <= y1]
            words = " ".join(t.text for t in sorted(inside, key=lambda t: t.pos[0]))
            cat = _symbol(words, lg, labels=True) or next((_symbol(t.text, lg, labels=True) for t in inside if _symbol(t.text, lg, labels=True)), None)
            if not cat:
                if inside:
                    plant.unknown.append({"what": "labelled rectangle", "label": words, "pos": ((x0 + x1) / 2, (y0 + y1) / 2), "ref": p.ref})
                continue
            L, D, rot = (w, h, 0.0) if w >= h else (h, w, 90.0)
            if cat == "forklift":           # its long side is its length, drawn as is
                L, D, rot = (w, h, 90.0) if w >= h else (h, w, 0.0)
                L, D = min(L, D), max(L, D)
            eid = _nearest_id(dr, lg, (x0, y1), used, max(w, h) + 300) or new_id(cat)
            used.add(eid)
            plant.equipment.append(Equip(eid, cat, round((x0 + x1) / 2, 1), round((y0 + y1) / 2, 1), round(L, 1),
                                         round(D, 1), rot, p.ref, words))
    plant.equipment.sort(key=lambda e: e.id)
