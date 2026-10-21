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


def load(path, page=0, scale=None, calibration=None):
    """calibration: ((x0, y0), (x1, y1), real_distance_cm) in page points (PyMuPDF frame)."""
    import fitz
    doc = fitz.open(path)
    pg = doc[page]
    words = pg.get_text("words")
    dr = Drawing(path, "pdf")
    if calibration:
        (a, b, real_cm) = calibration
        k = real_cm / math.dist(a, b)
        dr.notes.append(f"scale from calibration: {k:.4g} cm per point")
    else:
        n = scale or find_scale(words)
        if not n:
            n = 100.0
            dr.notes.append("no scale on the page and none given; assumed 1:100 (calibrate with two points)")
        else:
            dr.notes.append(f"scale 1:{n:g} {'given' if scale else 'read from the page'}")
        k = PT_MM * n / 10.0                    # cm per point
    hpage = pg.rect.height
    P = lambda p: (float(p[0]) * k, (hpage - float(p[1])) * k)
    for i, d in enumerate(pg.get_drawings()):
        col = d.get("color")
        width = float(d.get("width") or 0.0) * PT_MM            # mm on paper
        fill = d.get("fill") is not None
        ref = f"pdf:{page}:{i}"
        c_col = tuple(col) if col else None
        ops = {it[0] for it in d["items"]}
        if "c" in ops and ops <= {"l", "c"}:
            # a path with curves (circles, arcs, rounded shapes): flatten the Beziers; closed -> one polygon
            chain = [P(d["items"][0][1])]
            for it in d["items"]:
                if it[0] == "l":
                    chain.append(P(it[2]))
                else:
                    p0, p1, p2, p3 = (P(q) for q in it[1:5])
                    for k8 in range(1, 9):
                        t = k8 / 8.0
                        u = 1 - t
                        chain.append(tuple(u ** 3 * a + 3 * u * u * t * b + 3 * u * t * t * c + t ** 3 * e
                                           for a, b, c, e in zip(p0, p1, p2, p3)))
            ends_meet = abs(chain[0][0] - chain[-1][0]) + abs(chain[0][1] - chain[-1][1]) < 1.0
            closed = bool(d.get("closePath")) or ends_meet
            if closed and len(chain) > 3:
                dr.polys.append(Poly(chain[:-1] if ends_meet else chain, True, "",
                                     c_col if col else (tuple(d["fill"]) if fill else None), width, fill, ref))
            else:
                for a, b in zip(chain, chain[1:]):
                    dr.segs.append(Seg(a, b, "", c_col, width, ref))
            continue
        for it in d["items"]:
            op = it[0]
            if op == "l":
                dr.segs.append(Seg(P(it[1]), P(it[2]), "", tuple(col) if col else None, width, ref))
            elif op == "re":
                r = it[1]
                q = [P((r.x0, r.y0)), P((r.x1, r.y0)), P((r.x1, r.y1)), P((r.x0, r.y1))]
                dr.polys.append(Poly(q, True, "", tuple(col) if col else (tuple(d["fill"]) if fill else None), width, fill, ref))
            elif op == "qu":
                qd = it[1]
                q = [P(qd.ul), P(qd.ur), P(qd.lr), P(qd.ll)]
                dr.polys.append(Poly(q, True, "", tuple(col) if col else None, width, fill, ref))
            elif op == "c":
                dr.segs.append(Seg(P(it[1]), P(it[4]), "", tuple(col) if col else None, width, ref))
    for w in words:
        x0, y0, x1, y1, txt = w[:5]
        dr.texts.append(Text(txt, P(((x0 + x1) / 2, (y0 + y1) / 2)), (y1 - y0) * k, "", f"pdf:{page}:w"))
    dr.notes.append(f"{len(dr.segs)} lines, {len(dr.polys)} rectangles, {len(dr.texts)} words")
    return dr
