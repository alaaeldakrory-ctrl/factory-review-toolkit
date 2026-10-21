"""Vector PDF import into the common drawing model (cm). Most plant layout PDFs are vector drawings, so the lines are
read directly rather than traced from a picture.

Scale: from a calibration (two page points and their real distance) or, if not given, from a "SCALE 1:N" note on the
page. Page points become drawing cm with y pointing up (PDF pages count y downwards). Stroke colour and width are
kept: the interpreter uses them where a PDF has no layers (thick black = walls, yellow = aisles, green = walkways).

Needs PyMuPDF (pip install pymupdf).
"""
import math
import re

from .model import Drawing, Poly, Seg, Text

PT_MM = 25.4 / 72.0


def find_scale(words):
    text = " ".join(w[4] for w in words)
    m = re.search(r"(?:SCALE|ESCALA|MASSSTAB)?\s*1\s*:\s*(\d{1,5})", text, re.I)
    return float(m.group(1)) if m else None


def load(path, page=0, scale=None, calibration=None, skip_layers=()):
    """calibration: ((x0, y0), (x1, y1), real distance) in page points (PyMuPDF frame); the distance in cm or as
    written on the drawing ("40'", "16'-10\"", "12 m"). skip_layers: CAD layer patterns to leave out; "LAYER:tiny"
    leaves out only that layer's hatch-sized strokes (under 1 pt), keeping its outlines."""
    import fnmatch
    import fitz
    from .lengths import parse_length
    doc = fitz.open(path)
    pg = doc[page]
    words = pg.get_text("words")
    dr = Drawing(path, "pdf")
    if calibration:
        (a, b, real) = calibration
        k = parse_length(real) / math.dist(a, b)
        dr.notes.append(f"scale from calibration: {k:.4g} cm per point (1:{k * 10 / PT_MM:.0f})")
    else:
        n = scale or find_scale(words)
        if not n:
            n = 100.0
            dr.notes.append("no scale on the page and none given; assumed 1:100 (calibrate with two points)")
        else:
            dr.notes.append(f"scale 1:{n:g} {'given' if scale else 'read from the page'}")
        k = PT_MM * n / 10.0                    # cm per point
    dr.cm_per_pt = k
    hpage = pg.rect.height
    dr.page_size = (pg.rect.width, hpage)
    P = lambda p: (float(p[0]) * k, (hpage - float(p[1])) * k)
    layers, skipped = set(), 0
    whole = [x.upper() for x in skip_layers if not x.lower().endswith(":tiny")]
    tiny = [x[:-5].upper() for x in skip_layers if x.lower().endswith(":tiny")]
    for i, d in enumerate(pg.get_drawings()):
        layer = d.get("layer") or ""
        lu = layer.upper()
        if whole and any(fnmatch.fnmatch(lu, x) for x in whole):
            skipped += 1
            continue
        if tiny and any(fnmatch.fnmatch(lu, x) for x in tiny):
            r = d["rect"]
            if r.width < 1.0 and r.height < 1.0:
                skipped += 1
                continue
        layers.add(layer)
        col = d.get("color")
        width = float(d.get("width") or 0.0) * PT_MM            # mm on paper
        fill = d.get("fill") is not None
        ref = f"pdf:{page}:{i}"
        c_col = tuple(col) if col else None
        f_col = c_col if col else (tuple(d["fill"]) if fill else None)
        items = d["items"]
        ops = {it[0] for it in items}
        if ops <= {"l", "c"} and items:
            # one path of joined pieces (how AutoCAD plots polylines, circles, arcs): a closed chain is one polygon
            chain = [P(items[0][1])]
            joined = True
            for it in items:
                q0 = P(it[1])
                if math.dist(q0, chain[-1]) > 0.5:
                    joined = False
                    break
                if it[0] == "l":
                    chain.append(P(it[2]))
                else:
                    p0, p1, p2, p3 = (P(q) for q in it[1:5])
                    for k8 in range(1, 9):
                        t = k8 / 8.0
                        u = 1 - t
                        chain.append(tuple(u ** 3 * a + 3 * u * u * t * b + 3 * u * t * t * c + t ** 3 * e
                                           for a, b, c, e in zip(p0, p1, p2, p3)))
            if joined and len(chain) > 3:
                ends_meet = math.dist(chain[0], chain[-1]) < 0.5
                if bool(d.get("closePath")) or ends_meet:
                    dr.polys.append(Poly(chain[:-1] if ends_meet else chain, True, layer, f_col, width, fill, ref))
                    continue
            if not joined and ops == {"l"} and len(items) >= 8:
                # a wide polyline plotted as triangle strips: all its points hug its own bounding box -> a rectangle
                pts = [q for it in items for q in it[1:3]]
                x0, y0 = min(q.x for q in pts), min(q.y for q in pts)
                x1, y1 = max(q.x for q in pts), max(q.y for q in pts)
                tol = 1.2
                if x1 - x0 > 3 * tol and y1 - y0 > 3 * tol:
                    near = [min(abs(q.x - x0), abs(q.x - x1), abs(q.y - y0), abs(q.y - y1)) <= tol for q in pts]
                    sides = (any(abs(q.x - x0) <= tol for q in pts), any(abs(q.x - x1) <= tol for q in pts),
                             any(abs(q.y - y0) <= tol for q in pts), any(abs(q.y - y1) <= tol for q in pts))
                    def along(q0, q1):              # a strip piece: runs along one edge, or is a short cross-piece
                        if math.dist((q0.x, q0.y), (q1.x, q1.y)) <= 3 * tol:
                            return True
                        return any(abs(u - e) <= tol and abs(v - e) <= tol
                                   for u, v, e in ((q0.x, q1.x, x0), (q0.x, q1.x, x1), (q0.y, q1.y, y0), (q0.y, q1.y, y1)))
                    if all(near) and all(sides) and all(along(it[1], it[2]) for it in items):
                        q = [P((x0, y0)), P((x1, y0)), P((x1, y1)), P((x0, y1))]
                        dr.polys.append(Poly(q, True, layer, f_col, width or 0.4 * PT_MM, fill, ref))
                        continue
            if joined and "c" in ops:
                for a, b in zip(chain, chain[1:]):
                    dr.segs.append(Seg(a, b, layer, c_col, width, ref))
                continue
        for it in items:
            op = it[0]
            if op == "l":
                dr.segs.append(Seg(P(it[1]), P(it[2]), layer, c_col, width, ref))
            elif op == "re":
                r = it[1]
                q = [P((r.x0, r.y0)), P((r.x1, r.y0)), P((r.x1, r.y1)), P((r.x0, r.y1))]
                dr.polys.append(Poly(q, True, layer, f_col, width, fill, ref))
            elif op == "qu":
                qd = it[1]
                q = [P(qd.ul), P(qd.ur), P(qd.lr), P(qd.ll)]
                dr.polys.append(Poly(q, True, layer, f_col, width, fill, ref))
            elif op == "c":
                dr.segs.append(Seg(P(it[1]), P(it[4]), layer, c_col, width, ref))
    dr.layers = sorted(layers - {""})
    # one text per PDF text line, in reading order, with its real extent and direction (CAD text is often rotated)
    for bi, blk in enumerate(pg.get_text("dict")["blocks"]):
        for li, ln in enumerate(blk.get("lines", [])):
            txt = " ".join(" ".join(sp["text"].split()) for sp in ln["spans"] if sp["text"].strip()).strip()
            if not txt:
                continue
            x0, y0, x1, y1 = ln["bbox"]
            dx, dy = ln.get("dir", (1.0, 0.0))
            vertical = abs(dy) > abs(dx)
            thick = (x1 - x0) if vertical else (y1 - y0)
            long_ = (y1 - y0) if vertical else (x1 - x0)
            dr.texts.append(Text(txt, P(((x0 + x1) / 2, (y0 + y1) / 2)), thick * k, "", f"pdf:{page}:t{bi}.{li}",
                                 long_ * k, vertical))
    dr.notes.append(f"{len(dr.segs)} lines, {len(dr.polys)} closed shapes, {len(dr.texts)} words"
                    + (f", CAD layers: {len(dr.layers)}" if dr.layers else "") + (f", {skipped} paths skipped" if skipped else ""))
    return dr
