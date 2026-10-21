"""Read a drawing with a drawing-standard profile: import -> objects -> classified objects -> stations.

    r = read("layout.pdf", profile="lift_team", calibration=((x0, y0), (x1, y1), "40'"))
    r.items          # [(DrawObject, Classified)]
    r.stations       # [ReadStation] with name, tasks, box, anchor object

Every decision carries its reason, and anything unsure stays "unknown" for review instead of being guessed.
"""
import math
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from . import load as load_drawing
from .classify import classify_all, load_profile
from .interpret import phrases
from .objects import find_objects


@dataclass
class ReadStation:
    name: str
    tasks: List[str]
    label_pos: Tuple[float, float]
    box: Tuple[float, float, float, float]
    box_from: str                       # "station box" / "unit footprint" / "label only"
    anchor: Optional[int] = None        # object id of the unit / box it belongs to
    model: str = ""


@dataclass
class Reading:
    drawing: object
    objects: list
    classes: list
    stations: List[ReadStation] = field(default_factory=list)
    profile: dict = None

    @property
    def items(self):
        return list(zip(self.objects, self.classes))

    def counts(self):
        out = {}
        for c in self.classes:
            out[c.type] = out.get(c.type, 0) + 1
        return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def text_lines(texts):
    """Words -> lines: [(text, left x, y, height)], left to right, gaps under 2.5 text heights."""
    words = sorted(texts, key=lambda t: (-t.pos[1], t.pos[0]))
    lines = []
    for w in words:
        h = max(w.height, 1.0)
        half = w.width / 2 if getattr(w, "width", 0) else len(w.text) * 0.3 * h     # real extent when known
        for ln in lines:
            if abs(ln["y"] - w.pos[1]) < 0.35 * max(h, ln["h"]) and 0 <= w.pos[0] - half - ln["right"] < 2.5 * h:
                ln["words"].append(w.text)
                ln["right"] = w.pos[0] + half
                break
        else:
            lines.append({"words": [w.text], "left": w.pos[0] - half, "right": w.pos[0] + half, "y": w.pos[1], "h": h})
    return [(" ".join(ln["words"]), ln["left"], ln["y"], ln["h"]) for ln in lines]


def _clean_task(t, bullets):
    words = [w for w in t.split() if not (len(w) == 1 and not w.isalnum()) and w not in bullets]
    return " ".join(words).strip()


def _stations(dr, objs, classes, prof):
    cfg = prof.get("stations", {})
    pat = re.compile(cfg.get("label", r"^(ST|STATION|TEST)\s*(\d{1,3})"), re.I)
    reach = cfg.get("search_cm", 700)
    bullets = tuple(cfg.get("bullets", ["•", "-", "*"]))
    lines = text_lines(dr.texts)
    labels = []
    for t, left, y, h in lines:
        tt = _clean_task(t, bullets)
        m = pat.match(tt)
        if m and not any(l[0] == tt[:m.end()].strip() and math.dist(l[2], (left, y)) < 3 * h for l in labels):
            labels.append((tt[:m.end()].strip(), tt[m.end():].strip(" -:"), (left, y), h, m))
    taken = set()
    tasks_of = {}
    for name, rest, (lx, ly), h, m in labels:
        tasks, y = ([rest] if rest else []), ly
        while True:                           # walk down: the next line just below, in the label's column
            nxt = [(t, x, yy, hh) for t, x, yy, hh in lines if (t, x, yy) not in taken
                   and 0.4 * h < y - yy < 2.2 * h and lx - 3.5 * h <= x <= lx + 6 * h]
            if not nxt:
                break
            t, x, yy, hh = max(nxt, key=lambda q: q[2])
            taken.add((t, x, yy))
            c = _clean_task(t, bullets)
            if pat.match(c):
                break
            if c:
                tasks.append(c)
            y = yy
        tasks_of[(lx, ly)] = tasks
    labels = [(n, tasks_of[p], p, m) for n, _, p, _, m in labels]
    units = [(o, c) for o, c in zip(objs, classes) if c.type in ("product_unit",)]
    boxes = [(o, c) for o, c in zip(objs, classes) if c.type == "station_box"]
    used = set()
    out = []
    for name, tasks, p, m in labels:
        # the unit the label sits under: above it (y up), nearest, not already taken
        best, bd = None, reach
        for o, c in units:
            if o.id in used:
                continue
            cx, cy = o.center
            if cy < p[1] - 50:
                continue
            d = math.hypot(cx - p[0], (o.bbox[1] - p[1]) * 1.5)
            if d < bd:
                best, bd = o, d
        box, src, anchor = None, "label only", None
        if best is not None:
            used.add(best.id)
            anchor = best.id
            host = next((b for b, _ in sorted(boxes, key=lambda bc: bc[0].size[0] * bc[0].size[1])
                         if b.contains(best.center)), None)
            if host is not None:
                box, src = host.bbox, "station box"
            else:
                x0, y0, x1, y1 = best.bbox
                pad = cfg.get("box_margin_cm", 120)
                box, src = (x0 - pad, y0 - pad, x1 + pad, y1 + pad), "unit footprint"
        else:
            box = (p[0] - 150, p[1], p[0] + 150, p[1] + 300)
        num = m.group(2).upper()
        kind = m.group(1).upper()
        title = (f"Test {num}" if kind in ("TEST", "PRUEBA", "COLLAUDO") else f"Station {num}")
        out.append(ReadStation(title, tasks, p, tuple(round(v, 1) for v in box), src, anchor))
    out.sort(key=lambda s: (s.name.split()[0], int(re.sub(r"\D", "", s.name.split()[-1]) or 0)))
    return out


def _absorb_nested(objs, classes, skip=("unknown", "ignore", "area", "station_box", "column", "wall")):
    """Two objects of one type, one inside the other (a jib's swing circle and its boom symbol): keep the outer."""
    from .classify import Classified
    idx = [i for i, c in enumerate(classes) if c.type not in skip]
    for i in idx:
        oi = objs[i]
        for j in idx:
            if i == j or classes[j].type != classes[i].type:
                continue
            oj = objs[j]
            if oj.contains((oi.bbox[0], oi.bbox[1]), 5) and oj.contains((oi.bbox[2], oi.bbox[3]), 5)                     and oj.size[0] * oj.size[1] > oi.size[0] * oi.size[1]:
                params = {**classes[i].params, **classes[j].params}
                classes[j].params = params
                classes[i] = Classified("ignore", classes[i].confidence, f"part of #{oj.id} ({classes[j].type})")
                break


def read(path, profile="base", calibration=None, scale=None, min_confidence=0.3):
    prof = load_profile(profile) if isinstance(profile, str) else profile
    imp = prof.get("import", {})
    kw = {}
    if path.lower().endswith(".pdf"):
        kw = {"calibration": calibration, "scale": scale, "skip_layers": imp.get("skip_layers", [])}
    dr = load_drawing(path, **kw)
    objs = find_objects(dr, gap_cm=imp.get("gap_cm", 8.0), ignore_layers=imp.get("ignore_layers", []))
    classes = classify_all(objs, prof, min_confidence)
    _absorb_nested(objs, classes)
    r = Reading(dr, objs, classes, profile=prof)
    r.stations = _stations(dr, objs, classes, prof)
    return r
