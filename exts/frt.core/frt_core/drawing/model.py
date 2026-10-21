"""Common 2D drawing model that both importers fill (DXF and PDF), in centimetres, x east, y north.

Everything a drawing contains that the interpreter needs: segments, polylines, rectangles, block inserts and texts,
each with its layer (DXF) or stroke colour / width (PDF) and a source reference for revision sync.
"""
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

Pt = Tuple[float, float]


@dataclass
class Seg:
    a: Pt
    b: Pt
    layer: str = ""
    color: Optional[Tuple[float, float, float]] = None
    width: float = 0.0
    ref: str = ""

    def length(self):
        return math.dist(self.a, self.b)


@dataclass
class Poly:
    pts: List[Pt]
    closed: bool = False
    layer: str = ""
    color: Optional[Tuple[float, float, float]] = None
    width: float = 0.0
    fill: bool = False
    ref: str = ""

    def bbox(self):
        xs = [p[0] for p in self.pts]
        ys = [p[1] for p in self.pts]
        return min(xs), min(ys), max(xs), max(ys)


@dataclass
class Insert:
    name: str
    pos: Pt
    rot: float = 0.0                     # degrees, counter-clockwise
    scale: Tuple[float, float] = (1.0, 1.0)
    bbox: Tuple[float, float, float, float] = (0, 0, 0, 0)   # block-local extents (min x, min y, max x, max y), cm
    layer: str = ""
    ref: str = ""

    def size(self):
        x0, y0, x1, y1 = self.bbox
        return abs(x1 - x0) * abs(self.scale[0]), abs(y1 - y0) * abs(self.scale[1])


@dataclass
class Text:
    text: str
    pos: Pt
    height: float = 0.0
    layer: str = ""
    ref: str = ""


@dataclass
class Drawing:
    source: str
    kind: str                            # "dxf" / "pdf"
    segs: List[Seg] = field(default_factory=list)
    polys: List[Poly] = field(default_factory=list)
    inserts: List[Insert] = field(default_factory=list)
    texts: List[Text] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)       # what the importer cleaned or guessed
    layers: List[str] = field(default_factory=list)      # CAD layers present (DXF, or a PDF plotted with layers)

    def transform(self, fn, rot_deg=0.0, scale=1.0):
        """Apply a 2D point map (and the rotation / uniform scale it implies to inserts)."""
        for s in self.segs:
            s.a, s.b = fn(s.a), fn(s.b)
        for p in self.polys:
            p.pts = [fn(q) for q in p.pts]
        for i in self.inserts:
            i.pos = fn(i.pos)
            i.rot = (i.rot + rot_deg) % 360.0
            i.scale = (i.scale[0] * scale, i.scale[1] * scale)
        for t in self.texts:
            t.pos = fn(t.pos)
            t.height *= scale

    def extents(self):
        xs, ys = [], []
        for s in self.segs:
            xs += [s.a[0], s.b[0]]
            ys += [s.a[1], s.b[1]]
        for p in self.polys:
            xs += [q[0] for q in p.pts]
            ys += [q[1] for q in p.pts]
        for i in self.inserts:
            xs.append(i.pos[0])
            ys.append(i.pos[1])
        return (min(xs), min(ys), max(xs), max(ys)) if xs else (0, 0, 0, 0)
