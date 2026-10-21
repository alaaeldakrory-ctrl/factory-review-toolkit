"""Read a drawing with a drawing-standard profile: import -> objects -> classified objects -> stations.

    r = read("layout.pdf", profile="lift_team", calibration=((x0, y0), (x1, y1), "40'"))
    r.items          # [(DrawObject, Classified)]
    r.stations       # [ReadStation] with name, tasks, box, anchor object

Every decision carries its reason, and anything unsure stays "unknown" for review instead of being guessed.
"""
import math
import os
import re

try:
    import tomllib
except ImportError:
    import tomli as tomllib
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

    def reclassify(self, profile=None, min_confidence=0.3):
        """After a correction: reload the profile and the drawing's settings, classify again (no re-import)."""
        from .classify import load_profile as _lp
        st = load_settings(self.drawing.source)
        self.profile = _lp(profile or st.get("profile") or self.profile["path"])
        self.classes = classify_all(self.objects, self.profile, min_confidence)
        _absorb_nested(self.objects, self.classes)
        _apply_overrides(self.objects, self.classes, st.get("overrides", []))
        self.stations = _stations(self.drawing, self.objects, self.classes, self.profile)
        return self

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


def settings_path(path):
    return path + ".frt.toml"


def load_settings(path):
    """Per-drawing settings beside the drawing (<drawing>.frt.toml): profile, calibration."""
    p = settings_path(path)
    if not os.path.isfile(p):
        return {}
    with open(p, "rb") as fh:
        return tomllib.load(fh).get("read", {})


def save_settings(path, **values):
    from .classify import _toml_value
    cur = load_settings(path)
    cur.update({k: v for k, v in values.items() if v is not None})
    with open(settings_path(path), "w", encoding="utf-8") as fh:
        fh.write("# how to read this drawing (written by the Factory Review Toolkit)" + chr(10) + "[read]" + chr(10))
        for k, v in cur.items():
            fh.write(f"{k} = {_toml_value(v)}" + chr(10))
    return cur


def pick(objs, p, size=None):
    """The object a reviewer means by a point: the smallest one containing it (or, given its size, the one of that
    size), else the nearest within 1 m."""
    hits = [o for o in objs if o.contains(p)]
    if hits and size:
        best = min(hits, key=lambda o: abs(max(o.size) - size[0]) + abs(min(o.size) - size[1]))
        if abs(max(best.size) - size[0]) < 0.1 * size[0] + 10:
            return best
    if hits:
        return min(hits, key=lambda o: o.size[0] * o.size[1])
    near = min(objs, key=lambda o: math.dist(o.center, p), default=None)
    return near if near is not None and math.dist(near.center, p) < 100 else None


def _apply_overrides(objs, classes, overrides):
    """Per-drawing corrections from review: {at = [x, y] (drawing cm), type = "room", params = {...}}."""
    from .classify import Classified
    pos = {id(o): i for i, o in enumerate(objs)}
    for ov in overrides:
        o = pick(objs, tuple(ov["at"]), ov.get("size"))
        if o is None:
            continue
        t = ov["type"] if ov["type"] != "none" else "ignore"
        classes[pos[id(o)]] = Classified(t, 1.0, "set by reviewer", params=dict(ov.get("params", {})))


def add_override(path, at, type_, params=None, size=None):
    cur = load_settings(path).get("overrides", [])
    cur = [o for o in cur if not (math.dist(o["at"], at) <= 1.0 and o.get("size") == size)]
    entry = {"at": [round(at[0], 1), round(at[1], 1)], "type": type_}
    if size:
        entry["size"] = [round(size[0], 1), round(size[1], 1)]
    if params:
        entry["params"] = params
    return save_settings(path, overrides=cur + [entry])


def read(path, profile=None, calibration=None, scale=None, min_confidence=0.3):
    st = load_settings(path)
    if calibration is None and "calibration" in st:
        c = st["calibration"]
        calibration = (tuple(c["a"]), tuple(c["b"]), c["distance"])
    scale = scale or st.get("scale")
    profile = profile or st.get("profile", "base")
    prof = load_profile(profile) if isinstance(profile, str) else profile
    imp = prof.get("import", {})
    kw = {}
    if path.lower().endswith(".pdf"):
        kw = {"calibration": calibration, "scale": scale, "skip_layers": imp.get("skip_layers", [])}
    dr = load_drawing(path, **kw)
    objs = find_objects(dr, gap_cm=imp.get("gap_cm", 8.0), ignore_layers=imp.get("ignore_layers", []))
    classes = classify_all(objs, prof, min_confidence)
    _absorb_nested(objs, classes)
    _apply_overrides(objs, classes, st.get("overrides", []))
    r = Reading(dr, objs, classes, profile=prof)
    r.stations = _stations(dr, objs, classes, prof)
    return r
