"""Read an overhead crane system from its drawing: runway pairs and the bridges on them.

    systems = analyse(drawing, crane_object)
    # [{"axis": "y", "runways": (x0, x1), "along": (y0, y1), "bridges": [y, ...], "girders": 2}, ...]

How cranes are drawn: each runway is a beam (a band of parallel long lines a few cm wide) along the system; a bridge
is a beam across a runway pair, often two beams close together (double girder). A frame may hold several runway
pairs side by side (two systems sharing a frame). Thin single lines (end stops, centre lines) are not beams.
"""


def _bands(vals, gap=20.0):
    out = []
    for v in sorted(vals):
        if out and v - out[-1][-1] <= gap:
            out[-1].append(v)
        else:
            out.append([v])
    return [(min(b), max(b)) for b in out]


def _beams(bands, min_w=4.0):
    return [b for b in bands if b[1] - b[0] >= min_w]


def analyse(dr, o, double_gap_cm=120.0):
    x0, y0, x1, y1 = o.bbox
    segs = [s for s in dr.segs if s.layer == o.layer
            and x0 - 1 <= min(s.a[0], s.b[0]) and max(s.a[0], s.b[0]) <= x1 + 1
            and y0 - 1 <= min(s.a[1], s.b[1]) and max(s.a[1], s.b[1]) <= y1 + 1]
    horiz = [s for s in segs if abs(s.a[1] - s.b[1]) < 2]
    vert = [s for s in segs if abs(s.a[0] - s.b[0]) < 2]
    W, H = x1 - x0, y1 - y0
    # runway beams: long lines along the system, for each orientation
    rx = _beams(_bands([(s.a[0] + s.b[0]) / 2 for s in vert if s.length() > 0.6 * H]))     # runways along y
    ry = _beams(_bands([(s.a[1] + s.b[1]) / 2 for s in horiz if s.length() > 0.6 * W]))    # runways along x
    options = []
    for axis, beams, cross, lo, hi in (("y", rx, horiz, y0, y1), ("x", ry, vert, x0, x1)):
        if len(beams) < 2:
            continue
        beams = beams[: len(beams) // 2 * 2]
        systems = []
        for k in range(0, len(beams), 2):
            p = (beams[k][0] + beams[k][1]) / 2
            q = (beams[k + 1][0] + beams[k + 1][1]) / 2
            span = q - p
            if span < 150:
                continue
            # bridge beams: across this runway pair, about as long as the span, between the runways
            pos = []
            for s in cross:
                if s.length() < 0.8 * span:
                    continue
                if axis == "y":
                    a, b = sorted((s.a[0], s.b[0]))
                    if a > p + 0.2 * span or b < q - 0.2 * span:
                        continue
                    pos.append((s.a[1] + s.b[1]) / 2)
                else:
                    a, b = sorted((s.a[1], s.b[1]))
                    if a > p + 0.2 * span or b < q - 0.2 * span:
                        continue
                    pos.append((s.a[0] + s.b[0]) / 2)
            girders = _beams(_bands(pos))
            # double girders: two beams close together are one bridge
            bridges, double = [], False
            i = 0
            while i < len(girders):
                c = (girders[i][0] + girders[i][1]) / 2
                if i + 1 < len(girders) and (girders[i + 1][0] + girders[i + 1][1]) / 2 - c <= double_gap_cm:
                    c2 = (girders[i + 1][0] + girders[i + 1][1]) / 2
                    bridges.append(round((c + c2) / 2, 1))
                    double = True
                    i += 2
                else:
                    bridges.append(round(c, 1))
                    i += 1
            systems.append({"axis": axis, "runways": (round(p, 1), round(q, 1)), "along": (round(lo, 1), round(hi, 1)),
                            "bridges": bridges, "girders": 2 if double else 1})
        if systems:
            options.append(systems)
    if not options:
        return []
    # the reading with more bridges found is the right orientation (the other one finds only end ties)
    return max(options, key=lambda ss: (sum(len(s["bridges"]) for s in ss), -len(ss)))
