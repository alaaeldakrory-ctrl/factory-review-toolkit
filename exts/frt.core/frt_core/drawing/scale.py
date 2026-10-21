"""Find a drawing's scale from its own size labels when it states none (fit-to-paper plots).

Labels such as "5.00x6.00m", "3.80 X 6.99", "1200 x 1000 mm" sit next to (or inside) the shape they describe. Each
label next to a rectangle with the same proportions votes for a scale; the scale most labels agree on wins.

    f, votes, why = suggest(drawing)     # f: multiply the drawing's lengths by f to get real cm (None if no agreement)
"""
import math
import re

SIZE = re.compile(r"(\d+(?:[.,]\d+)?)\s*[xX×]\s*(\d+(?:[.,]\d+)?)\s*(mm|cm|m)?\b")


def _real_cm(a, b, unit):
    a, b = float(a.replace(",", ".")), float(b.replace(",", "."))
    if unit is None:                                # bare numbers: metres if small, millimetres if large
        unit = "m" if max(a, b) < 100 else "mm"
    f = {"mm": 0.1, "cm": 1.0, "m": 100.0}[unit]
    return sorted((a * f, b * f))


def suggest(dr, tol=0.04, min_votes=2):
    from .objects import _is_rect
    rects = [p for p in dr.polys if p.closed and _is_rect(p.pts)]
    cands = []
    for t in dr.texts:
        for m in SIZE.finditer(t.text):
            A, B = _real_cm(*m.groups())
            if A <= 0 or B <= 0:
                continue
            reach = 8 * max(t.height, 1e-6)
            for p in rects:
                x0, y0, x1, y1 = p.bbox()
                dx = max(x0 - t.pos[0], 0, t.pos[0] - x1)
                dy = max(y0 - t.pos[1], 0, t.pos[1] - y1)
                if math.hypot(dx, dy) > reach:
                    continue
                a, b = sorted((x1 - x0, y1 - y0))
                if a <= 0:
                    continue
                fa, fb = A / a, B / b
                if abs(fa - fb) <= tol * max(fa, fb):
                    cands.append(((fa + fb) / 2, t.text))
    if not cands:
        return None, 0, "no size labels next to matching rectangles"
    best, members = None, []
    for f, _ in cands:                                # the largest group of agreeing votes
        grp = [(g, s) for g, s in cands if abs(math.log(g / f)) <= tol]
        if len(grp) > len(members):
            best, members = f, grp
    labels = sorted({s for _, s in members})
    if len(labels) < min_votes:
        return None, len(labels), f"only {len(labels)} label(s) agree ({', '.join(labels)})"
    f = sorted(g for g, _ in members)[len(members) // 2]
    return f, len(labels), f"{len(labels)} size labels agree ({', '.join(labels[:4])}{', ...' if len(labels) > 4 else ''})"


def rescale(dr, f):
    """Multiply every length in the drawing by f (in place)."""
    dr.transform(lambda p: (p[0] * f, p[1] * f), 0.0, f)
    for t in dr.texts:                               # transform already scaled the heights
        if getattr(t, "width", 0):
            t.width *= f
    if getattr(dr, "cm_per_pt", None):
        dr.cm_per_pt *= f
    return dr


# ------------------------------------------------------------------------------------------ known block sizes
def learn_sizes(reading, types=("staged_kit",), min_count=2):
    """Typical sizes of a team's standard blocks from a calibrated drawing: [{type, layer, size, count}]."""
    groups = {}
    for o, c in reading.items:
        if c.type in types:
            groups.setdefault((c.type, o.layer), []).append((max(o.size), min(o.size)))
    out = []
    for (t, layer), v in sorted(groups.items()):
        if len(v) < min_count:
            continue
        Ls, Ss = sorted(a for a, _ in v), sorted(b for _, b in v)
        out.append({"type": t, "layer": layer, "size": [round(Ls[len(Ls) // 2], 1), round(Ss[len(Ss) // 2], 1)],
                    "count": len(v)})
    return out


def from_known_sizes(reading, sizes, tol=0.06, min_votes=3):
    """Scale factor from blocks whose real size the team profile knows: (f, votes, why) or (None, ...)."""
    ref = {(s["type"], s["layer"]): s["size"] for s in sizes}
    ratios = []
    for o, c in reading.items:
        r = ref.get((c.type, o.layer))
        if r is None or min(o.size) <= 0:
            continue
        fa, fb = r[0] / max(o.size), r[1] / min(o.size)
        if abs(fa - fb) <= 2 * tol * max(fa, fb):
            ratios.append((fa + fb) / 2)
    if len(ratios) < min_votes:
        return None, len(ratios), f"only {len(ratios)} known blocks"
    ratios.sort()
    f = ratios[len(ratios) // 2]
    agree = [x for x in ratios if abs(x / f - 1) <= tol]
    if len(agree) < min_votes:
        return None, len(agree), "known blocks disagree about the scale"
    return f, len(agree), f"{len(agree)} known blocks agree"
