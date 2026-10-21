"""Split a drawing into objects: the things a person sees (a table, a scissor unit, a crane, a kit spot, a station
box), each made of many lines.

    objs = find_objects(drawing)          # [DrawObject], plant cm

Rules (no block information in a PDF):
- a DXF block insert is one object;
- a closed rectangle (or circle) bigger than `standalone_cm` stands alone, so rows of touching kit spots or racks
  stay separate objects;
- everything else on the same layer that touches (gap <= `gap_cm`) is grouped: the hundreds of strokes of a
  forklift or a scissor unit become one object;
- a standalone shape lying inside a group of the same layer (a unit's body) joins that group.
Texts are then attached: to the smallest object they sit in, else to the nearest one within `text_reach_cm`.
"""
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class DrawObject:
    id: int
    layer: str
    kind: str                                     # shape / group / insert / text
    bbox: Tuple[float, float, float, float]
    n: int = 1                                    # primitives in it
    closed: bool = False                          # a single closed outline
    circle: bool = False
    rect: bool = False
    color: Optional[Tuple[float, float, float]] = None
    name: str = ""                                # DXF block name
    texts: List[str] = field(default_factory=list)          # its own labels (it is the smallest object around them)
    inner_texts: List[str] = field(default_factory=list)    # labels inside it that belong to a smaller object
    near_texts: List[str] = field(default_factory=list)     # text just outside
    refs: List[str] = field(default_factory=list)
    seg_lengths: List[float] = field(default_factory=list)  # for symbol fingerprints

    @property
    def size(self):
        x0, y0, x1, y1 = self.bbox
        return x1 - x0, y1 - y0

    @property
    def center(self):
        x0, y0, x1, y1 = self.bbox
        return (x0 + x1) / 2, (y0 + y1) / 2

    def contains(self, p, pad=0.0):
        x0, y0, x1, y1 = self.bbox
        return x0 - pad <= p[0] <= x1 + pad and y0 - pad <= p[1] <= y1 + pad


def _is_rect(pts, tol=0.02):
    if len(pts) != 4:
        return False
    for i in range(4):
        a, b, c = pts[i - 1], pts[i], pts[(i + 1) % 4]
        u, v = (a[0] - b[0], a[1] - b[1]), (c[0] - b[0], c[1] - b[1])
        nu, nv = math.hypot(*u), math.hypot(*v)
        if nu < 1e-6 or nv < 1e-6 or abs((u[0] * v[0] + u[1] * v[1]) / (nu * nv)) > tol:
            return False
    return True


def _is_circle(pts, tol=0.04):
    if len(pts) < 12:
        return False
    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)
    r = [math.hypot(p[0] - cx, p[1] - cy) for p in pts]
    m = sum(r) / len(r)
    return m > 0 and max(abs(x - m) for x in r) / m < tol


class _DSU:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.p[b] = a


def _bb_gap(a, b):
    dx = max(b[0] - a[2], a[0] - b[2], 0.0)
    dy = max(b[1] - a[3], a[1] - b[3], 0.0)
    return max(dx, dy)


def find_objects(dr, gap_cm=3.0, standalone_cm=40.0, text_reach_cm=120.0, ignore_layers=()):
    from .interpret import _match
    prims = []          # (layer, bbox, kind, payload)
    for s in dr.segs:
        if ignore_layers and _match(s.layer, ignore_layers):
            continue
        bb = (min(s.a[0], s.b[0]), min(s.a[1], s.b[1]), max(s.a[0], s.b[0]), max(s.a[1], s.b[1]))
        prims.append((s.layer, bb, "seg", s))
    for p in dr.polys:
        if ignore_layers and _match(p.layer, ignore_layers):
            continue
        prims.append((p.layer, p.bbox(), "poly", p))
    objs = []
    # standalone shapes
    loose = []
    for i, (layer, bb, kind, pl) in enumerate(prims):
        if kind == "poly":
            w, h = bb[2] - bb[0], bb[3] - bb[1]
            rect, circ = _is_rect(pl.pts), _is_circle(pl.pts)
            if (rect or circ) and min(w, h) >= standalone_cm:
                objs.append(DrawObject(0, layer, "shape", bb, 1, True, circ, rect, pl.color, refs=[pl.ref],
                                       seg_lengths=[w, h]))
                continue
        loose.append(i)
    # group loose primitives per layer by contact, with a spatial hash
    cell = 15.0
    by_layer = {}
    for i in loose:
        by_layer.setdefault(prims[i][0], []).append(i)
    for layer, idx in by_layer.items():
        dsu = _DSU(len(idx))
        grid = {}
        for k, i in enumerate(idx):
            x0, y0, x1, y1 = prims[i][1]
            for gx in range(int(math.floor((x0 - gap_cm) / cell)), int(math.floor((x1 + gap_cm) / cell)) + 1):
                for gy in range(int(math.floor((y0 - gap_cm) / cell)), int(math.floor((y1 + gap_cm) / cell)) + 1):
                    grid.setdefault((gx, gy), []).append(k)
        for members in grid.values():
            if len(members) < 2:
                continue
            # compare against one representative per already-joined set: dense cells (a unit's 4000 strokes) stay linear
            reps = {}
            for m in members:
                bm = prims[idx[m]][1]
                joined = False
                for r, br in list(reps.items()):
                    if _bb_gap(bm, br) <= gap_cm:
                        dsu.union(r, m)
                        reps[r] = (min(br[0], bm[0]), min(br[1], bm[1]), max(br[2], bm[2]), max(br[3], bm[3]))
                        joined = True
                        break
                if not joined:
                    reps[m] = bm
        groups = {}
        for k in range(len(idx)):
            groups.setdefault(dsu.find(k), []).append(idx[k])
        for g in groups.values():
            bbs = [prims[i][1] for i in g]
            bb = (min(b[0] for b in bbs), min(b[1] for b in bbs), max(b[2] for b in bbs), max(b[3] for b in bbs))
            first = prims[g[0]][3]
            single_poly = len(g) == 1 and prims[g[0]][2] == "poly"
            lens = []
            for i in g:
                pl = prims[i][3]
                lens.append(pl.length() if prims[i][2] == "seg" else (bbs[0][2] - bbs[0][0]) + (bbs[0][3] - bbs[0][1]))
            objs.append(DrawObject(0, layer, "shape" if single_poly else "group", bb, len(g), single_poly,
                                   single_poly and _is_circle(first.pts), single_poly and _is_rect(first.pts),
                                   first.color, refs=sorted({prims[i][3].ref for i in g})[:20], seg_lengths=lens))
    for ins in dr.inserts:
        x0, y0, x1, y1 = ins.bbox
        L, D = ins.size()
        objs.append(DrawObject(0, ins.layer, "insert", (ins.pos[0] - L / 2, ins.pos[1] - D / 2, ins.pos[0] + L / 2,
                                                        ins.pos[1] + D / 2), 1, name=ins.name, refs=[ins.ref]))
    # a standalone shape inside a bigger group of the same layer belongs to it (a unit's body panel)
    groups = [o for o in objs if o.kind == "group" and o.n >= 8]
    keep = []
    for o in objs:
        if o.kind == "shape":
            host = next((g for g in groups if g.layer == o.layer and g.contains((o.bbox[0], o.bbox[1]), 1)
                         and g.contains((o.bbox[2], o.bbox[3]), 1)), None)
            if host is not None and o.size[0] * o.size[1] < 0.8 * host.size[0] * host.size[1]:
                host.n += 1
                continue
        keep.append(o)
    objs = keep
    for k, o in enumerate(objs):
        o.id = k + 1
    _attach_texts(dr, objs, text_reach_cm)
    return objs


def _attach_texts(dr, objs, reach):
    from .interpret import phrases
    order = sorted(objs, key=lambda o: o.size[0] * o.size[1])
    for txt, p in phrases(dr.texts):
        hosts = [o for o in order if o.contains(p) and max(o.size) < 3000]
        if hosts:
            hosts[0].texts.append(txt)
            for o in hosts[1:]:
                o.inner_texts.append(txt)
            continue
        best, bd = None, reach
        for o in objs:
            d = math.hypot(max(o.bbox[0] - p[0], 0, p[0] - o.bbox[2]), max(o.bbox[1] - p[1], 0, p[1] - o.bbox[3]))
            if d < bd:
                best, bd = o, d
        if best is not None:
            best.near_texts.append(txt)
