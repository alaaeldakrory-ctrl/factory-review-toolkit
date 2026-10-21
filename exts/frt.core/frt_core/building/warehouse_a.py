"""Building generator on NVIDIA's modular warehouse (Warehouse_NVD Modular/Warehouse_A): walls, ceilings, columns,
lights and dock doors come from the pack's modules; this fits them to the plant's real column grid and doors.

Each grid bay gets one module, stretched in plan to the bay's real size (modules are 10 x 10 m; a 12 m bay is
stretched 1.2x). Perimeter bays take a wall module facing out; doors and docks pick the matching module in their bay.
Personnel doors (the pack has none) are added as our own door sets on the wall.

Module facts (measured): Z-up, cm, tile x 0..1000, y -1000..0; "Straight" carries its wall just outside the x = 0
edge, "Straight90" just outside the y = 0 edge, "CornerIn" both. Dock modules extend outside the wall.
Scene frame: plant x -> scene X (east), plant y -> scene -Z (north is -Z), building centred on the origin.
"""
import os

from pxr import Gf, UsdGeom

from .. import tags as T
from ..geom import Shapes

NV = os.environ.get("FRT_NVIDIA_ASSETS", r"C:/Nvidia_Assets")
MOD = NV + "/Warehouse_NVD/Assets/DigitalTwin/Assets/Warehouse/Modular/Warehouse_A/"
TILE = {"H10m": 1000.0, "H450cm": 500.0}

# wall module variants by what the wall bay carries (see the module catalogue in docs/warehouse_a_modules.md)
WALL = {
    "plain": ["A04", "A01", "A04", "A12"],      # braced panel, panel with electrics, braced, small window
    "windows": ["A05"],
    "rollup": ["A09", "A02", "A10", "A08"],      # medium roll-up, large, double sectional, high-speed
    "dock": ["A03", "A07", "A11", "A06"],        # single dock, with leveler, double dock, with trailer
}
# turn (deg about +Y) that puts a Straight / Straight90 wall on each side, and CornerIn on each corner pair
SIDE = {"W": ("Straight", 0.0), "E": ("Straight", 180.0), "N": ("Straight90", 0.0), "S": ("Straight90", 180.0)}
CORNER = {frozenset("WN"): 0.0, frozenset("NE"): -90.0, frozenset("ES"): 180.0, frozenset("SW"): 90.0}


def module_path(variant, kind, height="H10m", mirror=False):
    return MOD + f"Warehouse_{variant}_{height}_{kind}{'Mirror' if mirror else ''}_01.usd"


def to_scene(plant, x, y):
    return x - plant.width / 2.0, plant.depth / 2.0 - y


def _bays(lines, extent, tile):
    """Bay boundaries: the grid lines if there are at least two, else whole tiles over the extent."""
    if len(lines) >= 2 and lines[0] <= 1 and abs(lines[-1] - extent) <= 1:
        return list(lines)
    n = max(1, round(extent / tile))
    return [extent * i / n for i in range(n + 1)]


def build(stage, root, plant, height="H10m", wall_style="mixed", lib=None):
    """Write the building under root (an Xform). Returns notes (list of str)."""
    tile = TILE[height]
    notes = []
    bx = _bays(plant.grid_x, plant.width, tile)
    by = _bays(plant.grid_y, plant.depth, tile)
    nx, ny = len(bx) - 1, len(by) - 1
    UsdGeom.Xform.Define(stage, root)
    T.tag(stage.GetPrimAtPath(root), "building", source="warehouse_a")
    stretch = sorted({round((bx[i + 1] - bx[i]) / tile, 3) for i in range(nx)} | {round((by[j + 1] - by[j]) / tile, 3) for j in range(ny)})
    if any(abs(s - 1) > 0.2 for s in stretch):
        notes.append(f"bays stretched by {stretch} (more than 20 % looks distorted; consider the {('H450cm' if height == 'H10m' else 'H10m')} set)")
    # which bay edge each door falls on
    door_at = {}
    for d in plant.doors:
        if d.wall in "SN":
            i = max(0, min(nx - 1, next((k for k in range(nx) if bx[k] <= d.pos < bx[k + 1]), nx - 1)))
            j = 0 if d.wall == "S" else ny - 1
            centre = (bx[i] + bx[i + 1]) / 2
        else:
            j = max(0, min(ny - 1, next((k for k in range(ny) if by[k] <= d.pos < by[k + 1]), ny - 1)))
            i = 0 if d.wall == "W" else nx - 1
            centre = (by[j] + by[j + 1]) / 2
        door_at.setdefault((i, j, d.wall), []).append((d, centre))
    plain_k = 0
    used = {"dock": 0, "rollup": 0}
    personnel = []
    for i in range(nx):
        for j in range(ny):
            edges = set()
            if i == 0:
                edges.add("W")
            if i == nx - 1:
                edges.add("E")
            if j == 0:
                edges.add("S")
            if j == ny - 1:
                edges.add("N")
            x0, x1, y0, y1 = bx[i], bx[i + 1], by[j], by[j + 1]
            sx, sz = (x1 - x0) / tile, (y1 - y0) / tile
            cx, cz = to_scene(plant, (x0 + x1) / 2, (y0 + y1) / 2)
            here = [(d, c, e) for e in edges for d, c in door_at.get((i, j, e), [])]
            personnel += [d for d, c, e in here if d.kind == "door"]
            big = [(d, c, e) for d, c, e in here if d.kind in ("dock", "rollup")]
            if not edges:
                f, turn = module_path("A01", "Center", height, mirror=(i + j) % 2 == 1), 0.0
            elif len(edges) == 1:
                side = next(iter(edges))
                kind, turn = SIDE[side]
                if big:
                    d, c, _ = big[0]
                    variant = WALL[d.kind][used[d.kind] % len(WALL[d.kind])]
                    used[d.kind] += 1
                    if abs(d.pos - c) > 100:
                        notes.append(f"{d.kind} on the {d.wall} wall at {d.pos / 100:.1f} m placed at its bay centre "
                                     f"({abs(d.pos - c) / 100:.1f} m away)")
                    if len(big) > 1:
                        notes.append(f"{len(big)} doors share one bay on the {side} wall; only one is modelled")
                elif wall_style == "windows" or (wall_style == "mixed" and side in "WE" and j % 2 == 1):
                    variant = WALL["windows"][0]
                else:
                    variant = WALL["plain"][plain_k % len(WALL["plain"])]
                    plain_k += 1
                f = module_path(variant, kind, height)
            else:
                pair = frozenset(e for e in edges if e in "WE") | frozenset(e for e in edges if e in "SN")
                f, turn = module_path("A01", "CornerIn", height), CORNER.get(pair, 0.0)
                for d, c, e in big:
                    notes.append(f"{d.kind} in a corner bay ({d.wall} wall at {d.pos / 100:.1f} m) is not modelled by a module")
            if not os.path.exists(f):
                notes.append(f"missing module {os.path.basename(f)}")
                continue
            prim = stage.DefinePrim(f"{root}/Bay_{i:02d}_{j:02d}", "Xform")
            xf = UsdGeom.Xformable(prim)
            xf.AddTranslateOp().Set(Gf.Vec3d(cx, 0, cz))
            if abs(sx - 1) > 1e-3 or abs(sz - 1) > 1e-3:
                xf.AddScaleOp(opSuffix="bay").Set(Gf.Vec3f(sx, 1.0, sz))
            xf.AddRotateYOp().Set(turn)
            xf.AddTranslateOp(opSuffix="centre").Set(Gf.Vec3d(-tile / 2, 0, -tile / 2))
            xf.AddRotateXOp(opSuffix="zup").Set(-90.0)
            stage.DefinePrim(f"{root}/Bay_{i:02d}_{j:02d}/Module", "Xform").GetReferences().AddReference(f)
    # personnel doors (the pack has none): a steel door set on the inside face of the wall, tagged as an exit
    s = Shapes()
    for k, d in enumerate(personnel):
        _door_set(s, plant, d)
        p = stage.DefinePrim(f"{root}/Doors/PersonnelDoor_{k + 1:02d}", "Xform")
        T.tag(p, "exit", id=f"EXIT_{d.wall}_{int(d.pos)}", source=d.ref)
    if lib and s.groups:
        s.build(stage, f"{root}/Doors/Geom", lib.refs(list(s.groups)))
    notes.append(f"building: {nx} x {ny} bays of {tile / 100:.0f} m modules ({height}), {len(plant.doors)} doors")
    return notes


def _door_set(s, plant, d, w=100.0, h=210.0):
    """Frame, leaf, push bar and exit sign, 2 cm inside the wall line."""
    inset = 2.0
    if d.wall in "SN":
        x, y = d.pos, (0 + inset if d.wall == "S" else plant.depth - inset)
        cx, cz = to_scene(plant, x, y)
        s.box((cx - w / 2 - 6, 0, cz - 1.5), (cx + w / 2 + 6, h + 6, cz + 1.5), "powder_grey")
        s.box((cx - w / 2, 0.5, cz - 2.5), (cx + w / 2, h, cz + 2.5), "powder_lightgrey")
        s.box((cx - w / 2 + 10, 100, cz - 4), (cx + w / 2 - 10, 104, cz + 4), "steel_dark")
        s.box((cx - 20, h + 15, cz - 3), (cx + 20, h + 32, cz + 3), "floor_green")
    else:
        y, x = d.pos, (0 + inset if d.wall == "W" else plant.width - inset)
        cx, cz = to_scene(plant, x, y)
        s.box((cx - 1.5, 0, cz - w / 2 - 6), (cx + 1.5, h + 6, cz + w / 2 + 6), "powder_grey")
        s.box((cx - 2.5, 0.5, cz - w / 2), (cx + 2.5, h, cz + w / 2), "powder_lightgrey")
        s.box((cx - 4, 100, cz - w / 2 + 10), (cx + 4, 104, cz + w / 2 - 10), "steel_dark")
        s.box((cx - 3, h + 15, cz - 20), (cx + 3, h + 32, cz + 20), "floor_green")
