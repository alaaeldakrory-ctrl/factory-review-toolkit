"""DXF import into the common drawing model (cm), with the cleaning real drawings need:

- units from the file ($INSUNITS), or given explicitly when the file does not say
- arcs and circles approximated by segments, polylines kept as polylines (their bulges flattened)
- duplicate segments removed, zero-length and tiny stray lines dropped, endpoints closer than a tolerance snapped
  together (closes small gaps in outlines)
- block inserts kept as inserts with their block-local extents, so equipment symbols can be recognised by name

Needs ezdxf (pip install ezdxf).
"""
import math

from .model import Drawing, Insert, Poly, Seg, Text

# $INSUNITS code -> centimetres per drawing unit
INSUNITS_CM = {0: None, 1: 2.54, 2: 30.48, 3: 160934.4, 4: 0.1, 5: 1.0, 6: 100.0, 7: 100000.0, 8: 2.54e-6,
               9: 2.54e-3, 10: 91.44, 13: 1e-4, 14: 10.0}


def _block_extents(doc, name, cache):
    if name in cache:
        return cache[name]
    from ezdxf import bbox
    blk = doc.blocks.get(name)
    ext = (0.0, 0.0, 0.0, 0.0)
    if blk is not None:
        try:      # geometry only: label texts inside a symbol must not change its size
            geo = [g for g in blk if g.dxftype() not in ("TEXT", "MTEXT", "ATTDEF", "ATTRIB", "POINT")]
            e = bbox.extents(geo, fast=True)
            if e.has_data:
                ext = (e.extmin.x, e.extmin.y, e.extmax.x, e.extmax.y)
        except Exception:   # noqa: BLE001
            pass
    cache[name] = ext
    return ext


def load(path, units_cm=None, snap_cm=1.0, min_seg_cm=2.0, arc_step_deg=10.0):
    import ezdxf
    doc = ezdxf.readfile(path)
    code = doc.header.get("$INSUNITS", 0)
    k = units_cm or INSUNITS_CM.get(code)
    dr = Drawing(path, "dxf")
    if k is None:
        k = 0.1
        dr.notes.append("file has no units ($INSUNITS=0); assumed millimetres")
    else:
        dr.notes.append(f"units: $INSUNITS={code} -> {k:g} cm per drawing unit")
    P = lambda v: (float(v[0]) * k, float(v[1]) * k)
    cache = {}
    msp = doc.modelspace()
    for e in msp:
        t = e.dxftype()
        layer = e.dxf.layer if e.dxf.hasattr("layer") else "0"
        ref = f"dxf:{e.dxf.handle}"
        if t == "LINE":
            dr.segs.append(Seg(P(e.dxf.start), P(e.dxf.end), layer, ref=ref))
        elif t in ("LWPOLYLINE", "POLYLINE"):
            try:
                pts = [P(p) for p in e.flattening(0.01 / k if k else 1)]
            except Exception:   # noqa: BLE001
                pts = [P(p) for p in (e.get_points("xy") if t == "LWPOLYLINE" else [v.dxf.location for v in e.vertices])]
            closed = bool(e.closed if t == "LWPOLYLINE" else e.is_closed)
            if closed and len(pts) > 2 and math.dist(pts[0], pts[-1]) < 1e-6:
                pts = pts[:-1]
            dr.polys.append(Poly(pts, closed, layer, ref=ref))
        elif t in ("ARC", "CIRCLE"):
            c, r = e.dxf.center, e.dxf.radius
            a0, a1 = (e.dxf.start_angle, e.dxf.end_angle) if t == "ARC" else (0.0, 360.0)
            if a1 < a0:
                a1 += 360.0
            n = max(2, int((a1 - a0) / arc_step_deg))
            pts = [P((c[0] + r * math.cos(math.radians(a0 + (a1 - a0) * i / n)),
                      c[1] + r * math.sin(math.radians(a0 + (a1 - a0) * i / n)))) for i in range(n + 1)]
            dr.polys.append(Poly(pts[:-1] if t == "CIRCLE" else pts, t == "CIRCLE", layer, ref=ref))
        elif t == "INSERT":
            ext = _block_extents(doc, e.dxf.name, cache)
            dr.inserts.append(Insert(e.dxf.name, P(e.dxf.insert), float(e.dxf.get("rotation", 0.0)),
                                     (float(e.dxf.get("xscale", 1.0)), float(e.dxf.get("yscale", 1.0))),
                                     tuple(v * k for v in ext), layer, ref))
        elif t == "TEXT":
            dr.texts.append(Text(e.dxf.text, P(e.dxf.insert), float(e.dxf.height) * k, layer, ref))
        elif t == "MTEXT":
            dr.texts.append(Text(e.plain_text(), P(e.dxf.insert), float(e.dxf.char_height) * k, layer, ref))
    clean(dr, snap_cm, min_seg_cm)
    return dr


def clean(dr, snap_cm=1.0, min_seg_cm=2.0):
    """Drop tiny strays and duplicates, snap near-coincident endpoints."""
    before = len(dr.segs)
    segs = [s for s in dr.segs if s.length() >= min_seg_cm]
    dropped = before - len(segs)
    seen, uniq = set(), []
    for s in segs:
        key = tuple(sorted((tuple(round(v / snap_cm) for v in s.a), tuple(round(v / snap_cm) for v in s.b)))) + (s.layer,)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(s)
    dups = len(segs) - len(uniq)
    # snap endpoints: cluster points within snap * 5 per layer
    pts = []
    for s in uniq:
        pts += [s.a, s.b]
    reps = []
    tol = snap_cm * 5.0
    def rep(p):
        for r in reps:
            if math.dist(r, p) <= tol:
                return r
        reps.append(p)
        return p
    snapped = 0
    for s in uniq:
        a, b = rep(s.a), rep(s.b)
        if a != s.a or b != s.b:
            snapped += 1
        s.a, s.b = a, b
    dr.segs = uniq
    if dropped:
        dr.notes.append(f"dropped {dropped} stray segments shorter than {min_seg_cm:g} cm")
    if dups:
        dr.notes.append(f"removed {dups} duplicate segments")
    if snapped:
        dr.notes.append(f"snapped {snapped} segment ends to close gaps up to {tol:g} cm")
    return dr
