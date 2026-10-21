"""Layout checks on the plant plan (what the drawing says), before anything is animated:

    findings = check(plant, rules)        # Plant, plant dict or <scene>.plant.json

- aisle and walkway widths against the rule pack (the widest-longest aisle is the main forklift aisle)
- equipment standing in an aisle or a walkway (forklifts and overhead cranes excepted)
- walkways running too close alongside a forklift aisle
- exit travel distance: rectilinear (along aisles) from every 2.5 m of floor to the nearest personnel door; docks and
  roll-up doors are not exits
- equipment sticking out of its station's floor marking

Limits come from rules [layout] / [safety]. Locations are scene points (cm), so "Go" works.
"""
import math

from . import rules as R
from .findings import Finding, FindingList
from .revision import _as_dict

MOVING = {"forklift"}                       # drives in aisles by design
OVERHEAD = {"jib_crane", "bridge_crane"}    # over the floor, not on it


def _rect(e):
    """Axis-aligned plan box of an equipment item (rotation applied)."""
    a = math.radians(e["rot"])
    hl, hd = e["length"] / 2, e["depth"] / 2
    ex = abs(hl * math.cos(a)) + abs(hd * math.sin(a))
    ey = abs(hl * math.sin(a)) + abs(hd * math.cos(a))
    return e["x"] - ex, e["y"] - ey, e["x"] + ex, e["y"] + ey


def _inter(a, b):
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return (w, h) if w > 0 and h > 0 else None


def _gap(a, b):
    dx = max(b[0] - a[2], a[0] - b[2], 0.0)
    dy = max(b[1] - a[3], a[1] - b[3], 0.0)
    return math.hypot(dx, dy)


def _door_xy(d, W, D):
    return {"S": (d["pos"], 0.0), "N": (d["pos"], D), "W": (0.0, d["pos"]), "E": (W, d["pos"])}[d["wall"]]


def check(plant, rules=None, exit_kinds=("door",)):
    p = _as_dict(plant)
    rules = rules or R.load()
    W, D = p["width"], p["depth"]
    scene = lambda x, y: (x - W / 2, 0.0, D / 2 - y)
    fl = FindingList()
    aisles = [tuple(a) for a in p.get("aisles", [])]
    walks = [tuple(w) for w in p.get("walkways", [])]
    equip = p.get("equipment", [])

    # widths
    if aisles:
        main = max(aisles, key=lambda a: (min(a[2] - a[0], a[3] - a[1]), max(a[2] - a[0], a[3] - a[1])))
        for k, a in enumerate(aisles):
            w = min(a[2] - a[0], a[3] - a[1])
            key = "layout.main_aisle_min_cm" if a == main else "layout.cross_aisle_min_cm"
            lim = rules.get(key, 360 if a == main else 240)
            if w < lim:
                fl.add(Finding("layout.aisle_width", "critical" if w < 0.8 * lim else "warning",
                               f"{'Main' if a == main else 'Cross'} aisle {k + 1} is {w / 100:.2f} m wide; needs "
                               f"{lim / 100:.2f} m", location=scene((a[0] + a[2]) / 2, (a[1] + a[3]) / 2),
                               value=w, limit=lim, unit="cm", rule=key))
    # building columns standing in an aisle: what is left beside them
    col = rules.get("layout.column_size_cm", 40.0)
    for k, a in enumerate(aisles):
        horiz = (a[2] - a[0]) >= (a[3] - a[1])
        inside = [c for c in p.get("columns", []) if 50 < c[0] < W - 50 and 50 < c[1] < D - 50    # not wall columns
                  and a[0] < c[0] < a[2] and a[1] < c[1] < a[3]]
        if not inside:
            continue
        lo, hi = (a[1], a[3]) if horiz else (a[0], a[2])
        clear = min(max(c[1 if horiz else 0] - col / 2 - lo, hi - c[1 if horiz else 0] - col / 2) for c in inside)
        lim = rules.get("layout.main_aisle_min_cm", 360) if a == main else rules.get("layout.cross_aisle_min_cm", 240)
        where = ", ".join(f"{(c[0] if horiz else c[1]) / 100:.0f}" for c in inside)
        fl.add(Finding("layout.column_in_aisle", "warning" if clear >= rules.get("layout.cross_aisle_min_cm", 240)
                       else "critical",
                       f"Aisle {k + 1} has {len(inside)} building column(s) in it (at {'x' if horiz else 'y'} = "
                       f"{where} m), leaving {clear / 100:.2f} m beside them; guard them; this aisle needs "
                       f"{lim / 100:.2f} m", location=scene(*inside[len(inside) // 2]), value=clear,
                       limit=lim, unit="cm", rule="layout.column_in_aisle"))
    for k, wk in enumerate(walks):
        w = min(wk[2] - wk[0], wk[3] - wk[1])
        lim = rules.get("layout.walkway_min_cm", 90)
        if w < lim:
            fl.add(Finding("layout.walkway_width", "warning", f"Walkway {k + 1} is {w / 100:.2f} m wide; needs "
                           f"{lim / 100:.2f} m", location=scene((wk[0] + wk[2]) / 2, (wk[1] + wk[3]) / 2),
                           value=w, limit=lim, unit="cm", rule="layout.walkway_min_cm"))

    # equipment in aisles / walkways
    for e in equip:
        if e["catalog"] in MOVING | OVERHEAD:
            continue
        r = _rect(e)
        for kind, zones, sev in (("aisle", aisles, "critical"), ("walkway", walks, "warning")):
            for k, z in enumerate(zones):
                hit = _inter(r, z)
                if hit and min(hit) > 5:
                    ix0, iy0 = max(r[0], z[0]), max(r[1], z[1])
                    fl.add(Finding(f"layout.in_{kind}", sev,
                                   f"{e['id']} ({e['catalog']}) stands in {kind} {k + 1} "
                                   f"({hit[0] / 100:.2f} x {hit[1] / 100:.2f} m)",
                                   location=scene(ix0 + hit[0] / 2, iy0 + hit[1] / 2), value=round(hit[0] * hit[1] / 1e4, 2),
                                   unit="m2", rule=f"layout.{kind}"))

    # walkway alongside a forklift aisle
    sep = rules.get("safety.forklift_pedestrian_separation_cm", 100)
    for i, wk in enumerate(walks):
        for k, a in enumerate(aisles):
            g = _gap(wk, a)
            if _inter(wk, a):
                continue                                  # a crossing: marked as such on the floor, not a defect
            if g < sep:
                fl.add(Finding("safety.walkway_separation", "warning",
                               f"Walkway {i + 1} runs {g:.0f} cm from aisle {k + 1}; keep {sep:.0f} cm or a barrier",
                               location=scene((wk[0] + wk[2]) / 2, (wk[1] + wk[3]) / 2), value=g, limit=sep,
                               unit="cm", rule="safety.forklift_pedestrian_separation_cm"))

    # exit travel distance (rectilinear: travel follows aisles, not diagonals through racks)
    lim = rules.get("layout.exit_travel_max_cm", 6100)
    exits = [_door_xy(d, W, D) for d in p.get("doors", []) if d["kind"] in exit_kinds]
    if not exits:
        fl.add(Finding("layout.no_exit", "critical", "No personnel exit door on the drawing", rule="layout.exits"))
    else:
        step = 250.0
        worst, far = None, 0
        for i in range(int(W // step) + 1):
            for j in range(int(D // step) + 1):
                x, y = min(i * step, W - 50), min(j * step, D - 50)
                d = min(abs(x - ex) + abs(y - ey) for ex, ey in exits)
                if d > lim:
                    far += 1
                    if worst is None or d > worst[0]:
                        worst = (d, x, y)
        if worst:
            area = far * step * step / 1e4
            fl.add(Finding("layout.exit_travel", "warning",
                           f"About {area:.0f} m2 of floor is more than {lim / 100:.0f} m of aisle travel from a "
                           f"personnel exit (worst {worst[0] / 100:.0f} m here); add an exit door nearby",
                           location=scene(worst[1], worst[2]), value=round(worst[0]), limit=lim, unit="cm",
                           rule="layout.exit_travel_max_cm"))

    # equipment crossing its station's floor marking
    for st in p.get("stations", []):
        sr = (st["x0"], st["y0"], st["x1"], st["y1"])
        for e in equip:
            if e["catalog"] in OVERHEAD:
                continue
            r = _rect(e)
            inside_c = sr[0] <= e["x"] <= sr[2] and sr[1] <= e["y"] <= sr[3]
            if inside_c and not (r[0] >= sr[0] - 1 and r[1] >= sr[1] - 1 and r[2] <= sr[2] + 1 and r[3] <= sr[3] + 1):
                out = max(sr[0] - r[0], sr[1] - r[1], r[2] - sr[2], r[3] - sr[3])
                fl.add(Finding("layout.station_boundary", "info",
                               f"{e['id']} ({e['catalog']}) sticks {out:.0f} cm out of {st['name']}'s floor marking",
                               location=scene(e["x"], e["y"]), value=round(out), unit="cm",
                               rule="layout.station_boundary"))
    return fl
