"""A reviewed Reading -> Plant (the same plant data the scene builder takes from the sample pipeline).

    plant = to_plant(reading)

- building: the largest closed rectangle that holds most of the drawing (or the extents of wall objects)
- columns: column objects; grid lines where at least three columns line up
- doors / roll-up doors / docks: snapped to the nearest outside wall
- stations: from the reader (name, tasks, box); rooms get walls; walkways and aisles become floor marking
- equipment: typed objects mapped to catalog items (TYPE_CATALOG); operators are kept as proposed positions only
Coordinates move so the building's south-west corner is (0, 0).
"""
import math

from .interpret import Door, Equip, Plant, Station, _cluster

TYPE_CATALOG = {
    "flow_rack": "flow_rack", "industrial_table": "industrial_table", "lift_table": "lift_table",
    "conveyor": "conveyor", "pallet_rack": "pallet_rack", "parts_rack": "bin_rack", "bin_rack": "bin_rack",
    "jib_crane": "jib_crane", "bridge_crane": "bridge_crane", "crane": "bridge_crane", "forklift": "forklift",
    "tool_cart": "tool_cart", "trash": "trash_bin", "computer_station": "computer_station", "ramp": "ramp",
    "scale": "platform_scale", "staged_kit": "staged_kit", "product_unit": "scissor_unit", "crate": "crate",
    "pallet": "pallet_boxes",
}
PREFIX = {"flow_rack": "FR", "industrial_table": "T", "lift_table": "LT", "conveyor": "C", "pallet_rack": "R",
          "bin_rack": "BR", "jib_crane": "JC", "bridge_crane": "BC", "forklift": "FL", "tool_cart": "TC",
          "trash_bin": "TR", "computer_station": "PC", "ramp": "RP", "platform_scale": "SC", "staged_kit": "K",
          "scissor_unit": "U", "crate": "CR", "pallet_boxes": "P"}


def _building(reading):
    objs = [o for o, c in reading.items if c.type not in ("ignore",)]
    xs = [v for o in objs for v in (o.bbox[0], o.bbox[2])]
    ys = [v for o in objs for v in (o.bbox[1], o.bbox[3])]
    ext = (max(xs) - min(xs)) * (max(ys) - min(ys))
    walls = [o for o, c in reading.items if c.type == "wall"]
    rects = [o for o, c in reading.items if o.rect and c.type in ("area", "wall", "unknown", "building")]
    big = max(rects, key=lambda o: o.size[0] * o.size[1], default=None)
    if big is not None and big.size[0] * big.size[1] > 0.4 * ext:
        return big.bbox, f"building outline: the {big.size[0] / 100:.1f} x {big.size[1] / 100:.1f} m rectangle #{big.id}"
    if walls:
        return (min(o.bbox[0] for o in walls), min(o.bbox[1] for o in walls), max(o.bbox[2] for o in walls),
                max(o.bbox[3] for o in walls)), "building outline: extents of the wall objects"
    return (min(xs), min(ys), max(xs), max(ys)), "building outline: drawing extents (no wall or outline found)"


def _grid(values, extent, tol=60.0, min_count=2):
    lines = []
    for v in sorted(values):
        if lines and abs(v - lines[-1][-1]) <= tol:
            lines[-1].append(v)
        else:
            lines.append([v])
    g = [round(sum(c) / len(c), 1) for c in lines if len(c) >= min_count]
    return [v for v in g if -tol <= v <= extent + tol]


def _wall(cx, cy, W, D):
    d = {"S": cy, "N": D - cy, "W": cx, "E": W - cx}
    w = min(d, key=d.get)
    return w, d[w], (cx if w in "SN" else cy)


def _crane_systems(reading, o, c, loc, pl, counter):
    """A crane frame with readable runways and bridges -> one bridge crane per runway pair, with its bridges where
    they are drawn. False if the drawing does not show them (then the frame is placed as one default crane)."""
    from .cranes import analyse
    systems = analyse(reading.drawing, o)
    if not systems:
        return False
    for sy in systems:
        p, q = sy["runways"]
        lo, hi = sy["along"]
        length, span = hi - lo, q - p
        if sy["axis"] == "y":                      # runways run north-south: local Z = plant y, f = 0 at the north end
            cx, cy, rot = (p + q) / 2, (lo + hi) / 2, 0.0
            fr = [(hi - b) / length for b in sy["bridges"]]
        else:                                      # runways run east-west: turned 90, f = 0 at the west end
            cx, cy, rot = (lo + hi) / 2, (p + q) / 2, 90.0
            fr = [(b - lo) / length for b in sy["bridges"]]
        fr = sorted(min(1.0, max(0.0, f)) for f in fr)
        counter["bridge_crane"] = counter.get("bridge_crane", 0) + 1
        eid = f"BC{counter['bridge_crane']}"
        label = "; ".join([f"bridges={max(1, len(fr))}", f"girders={sy['girders']}",
                           "bridge_positions=" + ",".join(f"{f:.3f}" for f in fr)]
                          + [f"{k}={v}" for k, v in c.params.items()])
        x, y = loc((cx, cy))
        pl.equipment.append(Equip(eid, "bridge_crane", x, y, round(span + 50, 1), round(length, 1), rot,
                                  f"obj#{o.id}", label))
    return True


FINISHED = ("FG", "FINISHED", "TERMINADO", "FINITO", "SHIPPING")


def _in_finished_area(reading, o):
    """A model spot inside an area labelled for finished goods (1 DAY FG INVENTORY) holds a finished machine."""
    import re
    for a, c in reading.items:
        if a is o or max(a.size) < 600 or not a.contains(o.center):
            continue
        words = " ".join(a.texts + a.inner_texts).upper()
        if any(re.search(r"(?<![A-Z])" + re.escape(w) + r"(?![A-Z])", words) for w in FINISHED):
            return True
    return False


def to_plant(reading):
    (bx0, by0, bx1, by1), note = _building(reading)
    W, D = round(bx1 - bx0, 1), round(by1 - by0, 1)
    pl = Plant(reading.drawing.source, W, D, notes=list(reading.drawing.notes) + [note])
    loc = lambda p: (round(p[0] - bx0, 1), round(p[1] - by0, 1))
    inside = lambda o: bx0 - 200 <= o.center[0] <= bx1 + 200 and by0 - 200 <= o.center[1] <= by1 + 200
    counter = {}
    for o, c in sorted(reading.items, key=lambda oc: (round(-oc[0].center[1] / 300), oc[0].center[0])):
        if c.type in ("unknown", "ignore") or not inside(o):
            continue
        cx, cy = loc(o.center)
        L, S = max(o.size), min(o.size)
        rot = 0.0 if o.size[0] >= o.size[1] else 90.0
        if c.type == "column":
            pl.columns.append((cx, cy))
        elif c.type in ("door", "rollup_door", "dock"):
            wall, dist, pos = _wall(cx, cy, W, D)
            if dist < 300:
                kind = {"door": "door", "rollup_door": "rollup", "dock": "dock"}[c.type]
                pl.doors.append(Door(kind, wall, round(pos, 1), round(o.size[0] if wall in "SN" else o.size[1], 1),
                                     f"obj#{o.id}"))
        elif c.type == "room":
            x0, y0 = loc((o.bbox[0], o.bbox[1]))
            pl.rooms.append({"name": c.params.get("name") or (o.texts or [f"Room {len(pl.rooms) + 1}"])[0], "box": [x0, y0, x0 + o.size[0], y0 + o.size[1]],
                             "ref": f"obj#{o.id}"})
        elif c.type in ("walkway", "aisle"):
            x0, y0 = loc((o.bbox[0], o.bbox[1]))
            (pl.walkways if c.type == "walkway" else pl.aisles).append((x0, y0, round(x0 + o.size[0], 1), round(y0 + o.size[1], 1)))
        elif c.type == "operator":
            pl.operators.append({"pos": [cx, cy], "ref": f"obj#{o.id}", "confirmed": False})
        elif c.type in ("crane", "bridge_crane") and _crane_systems(reading, o, c, loc, pl, counter):
            pass
        elif c.type in TYPE_CATALOG:
            cat = TYPE_CATALOG[c.type]
            if c.type == "staged_kit" and "kind" not in c.params and _in_finished_area(reading, o):
                c.params["kind"] = "finished"
            counter[cat] = counter.get(cat, 0) + 1
            eid = f"{PREFIX.get(cat, cat[:2].upper())}{counter[cat]}"
            label = "; ".join(f"{k}={v}" for k, v in c.params.items())
            pl.equipment.append(Equip(eid, cat, cx, cy, round(L, 1), round(S, 1), rot, f"obj#{o.id}", label))
        elif c.type == "area" and o.rect:
            x0, y0 = loc((o.bbox[0], o.bbox[1]))
            pl.zones.append({"box": [x0, y0, round(x0 + o.size[0], 1), round(y0 + o.size[1], 1)], "ref": f"obj#{o.id}"})
    for o, c in reading.items:
        if c.type == "unknown" and inside(o) and max(o.size) >= 40:
            pl.unknown.append({"what": f"{o.layer} object", "pos": loc(o.center), "ref": f"obj#{o.id}"})
    for st in reading.stations:
        x0, y0 = loc((st.box[0], st.box[1]))
        x1, y1 = loc((st.box[2], st.box[3]))
        pl.stations.append(Station(st.name, x0, y0, x1, y1, f"label at {st.label_pos[0]:.0f},{st.label_pos[1]:.0f}",
                                   True, tasks=list(st.tasks), box_from=st.box_from))
    pl.grid_x = _grid([c[0] for c in pl.columns], W)
    pl.grid_y = _grid([c[1] for c in pl.columns], D)
    pl.equipment.sort(key=lambda e: (e.catalog, e.id))
    return pl
