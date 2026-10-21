"""Classify drawing objects with a drawing-standard profile: ordered rules, each naming the evidence it needs.

    prof = load_profile("base")                         # or a path / a team profile that extends base
    for o in objs: c = classify(o, prof)                # Classified(type, confidence, reason, params)

A rule (TOML [[rules]]) fires when every condition it lists holds:
    type      what it is (catalog item or role: jib_crane, station_box, staged_kit, product_unit, trash, ...)
    layer     layer-name patterns, any of them (shell wildcards, case-insensitive)
    text      label patterns found inside the object (or just beside it when near = true)
    name      DXF block-name patterns
    shape     "rect" / "circle" / "group" (many strokes)
    size_cm   [min, max] of the object's longer side;  short_cm  [min, max] of its shorter side
    min_parts / max_parts   number of strokes (a detailed symbol has thousands)
    param     where a value comes from, e.g. { model = "layer" } or { capacity = "text:(\\d+)\\s*(lb|kg)" }
Confidence grows with the kinds of evidence a rule uses (layer, text, shape, size...); the best rule wins, ties go to
the earlier rule. A team profile `extends` the base one; its rules are tried first. Corrections made in review go into
the team profile as new rules, so the next drawing of that standard starts from them.
"""
import fnmatch
import os
import re
from dataclasses import dataclass, field

try:
    import tomllib
except ImportError:
    import tomli as tomllib

PROFILES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "profiles")
USER_PROFILES = os.path.join(os.path.expanduser("~"), ".frt", "profiles")
EVIDENCE = ("layer", "text", "name", "shape", "size_cm", "short_cm", "min_parts", "max_parts")
WEIGHT = {"layer": 0.25, "text": 0.7, "text_near": 0.4, "name": 0.45, "shape": 0.15, "size_cm": 0.15, "short_cm": 0.1,
          "min_parts": 0.1, "max_parts": 0.05}


@dataclass
class Classified:
    type: str
    confidence: float
    reason: str
    rule: int = -1
    params: dict = field(default_factory=dict)


def _find(name):
    if os.path.isfile(name):
        return name
    for d in (USER_PROFILES, PROFILES):
        p = os.path.join(d, name if name.endswith(".toml") else name + ".toml")
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(f"no drawing profile {name!r} (looked in {USER_PROFILES} and {PROFILES})")


def load_profile(name="base"):
    """Rules of a profile and everything it extends, own rules first. Also merges [import] and [stations]."""
    path = _find(name)
    with open(path, "rb") as fh:
        p = tomllib.load(fh)
    rules = list(p.get("rules", []))
    imp, st = dict(p.get("import", {})), dict(p.get("stations", {}))
    names = [p.get("profile", {}).get("name", os.path.basename(path))]
    for base in p.get("profile", {}).get("extends", []):
        b = load_profile(base)
        rules += b["rules"]
        for k, v in b["import"].items():
            imp[k] = sorted(set(imp.get(k, [])) | set(v)) if isinstance(v, list) else imp.get(k, v)
        for k, v in b["stations"].items():
            st.setdefault(k, v)
        names += b["names"]
    return {"rules": rules, "import": imp, "stations": st, "names": names, "path": path}


def _any(value, patterns):
    v = (value or "").upper()
    return any(fnmatch.fnmatch(v, x.upper()) for x in patterns)


def _text_hit(o, patterns, near):
    """-> (text, inside?) for the first label matching a pattern; labels inside the object count first."""
    extra = (o.near_texts + getattr(o, "inner_texts", [])) if near else []
    for t, inside in [(t, True) for t in o.texts] + [(t, False) for t in extra]:
        u = t.upper()
        for x in patterns:
            if fnmatch.fnmatch(u, x.upper()):
                return t, inside
    return None


def _check(o, r):
    """-> list of evidence descriptions if the rule holds, else None."""
    why = []
    if "layer" in r:
        if not _any(o.layer, r["layer"]):
            return None
        why.append(("layer", f"layer {o.layer}"))
    if "name" in r:
        if not _any(o.name, r["name"]):
            return None
        why.append(("name", f"block {o.name}"))
    if "text" in r:
        hit = _text_hit(o, r["text"], r.get("near", False))
        if hit is None:
            return None
        why.append(("text" if hit[1] else "text_near", f"{'text' if hit[1] else 'label beside'} '{hit[0]}'"))
    if "shape" in r:
        s = r["shape"]
        ok = (s == "rect" and o.rect) or (s == "circle" and o.circle) or (s == "group" and o.kind == "group") \
            or (s == "closed" and o.closed)
        if not ok:
            return None
        why.append(("shape", s))
    L, S = max(o.size), min(o.size)
    if "size_cm" in r:
        lo, hi = r["size_cm"]
        if not lo <= L <= hi:
            return None
        why.append(("size_cm", f"{L / 100:.1f} m long"))
    if "short_cm" in r:
        lo, hi = r["short_cm"]
        if not lo <= S <= hi:
            return None
        why.append(("short_cm", f"{S / 100:.2f} m wide"))
    if "min_parts" in r:
        if o.n < r["min_parts"]:
            return None
        why.append(("min_parts", f"{o.n} strokes"))
    if "max_parts" in r and o.n > r["max_parts"]:
        return None
    return why


def _params(o, r):
    out = {}
    for k, src in r.get("param", {}).items():
        if src == "layer":
            out[k] = o.layer
        elif src.startswith("text:"):
            pat = re.compile(src[5:], re.I)
            for t in o.texts + o.near_texts + getattr(o, "inner_texts", []):
                m = pat.search(t)
                if m:
                    out[k] = m.group(0)
                    break
        else:
            out[k] = src
    return out


def classify(o, prof):
    best = None
    for i, r in enumerate(prof["rules"]):
        why = _check(o, r)
        if why is None:
            continue
        conf = min(0.99, sum(WEIGHT.get(k, 0.1) for k, _ in why) + r.get("bonus", 0.0))
        if best is None or conf > best.confidence + 1e-9:
            best = Classified(r["type"], round(conf, 2), " + ".join(d for _, d in why), i, _params(o, r))
    return best or Classified("unknown", 0.0, "no rule matched")


def classify_all(objs, prof, min_confidence=0.3):
    out = []
    for o in objs:
        c = classify(o, prof)
        if c.type != "ignore" and c.confidence < min_confidence:
            c = Classified("unknown", c.confidence, f"weak: {c.reason}" if c.reason else "no rule matched")
        out.append(c)
    return out
