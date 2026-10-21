"""Scene recipes: one small TOML file describes a whole scene; build() turns it into USD. Change a line, rebuild.

    [scene]
    name = "Demo Plant"
    output = "../../out/demo_plant.usd"          # relative paths are relative to the recipe file

    [drawing]
    file = "../drawings/demo_plant_revA.dxf"     # DXF or PDF
    legend = ""                                  # optional plant CAD standard (frt_core/drawing/legend_default.toml)
    pdf_scale = 0                                # 0 = read "SCALE 1:N" from the page

    [building]
    generator = "warehouse_a"
    height = "H10m"                              # H10m (10 m clear) or H450cm
    wall_style = "mixed"                         # plain / windows / mixed

    [equipment]
    from_drawing = true
    [equipment.defaults]                         # parameter defaults per catalog item
    conveyor = { type = "roller" }

    [[items]]                                    # extra items not on the drawing (plant cm, x east, y north)
    catalog = "industrial_table"
    id = "T9"
    at = [3000, 2500]
    rot = 0
    params = { length = 240 }

    [look]
    sky = true

Output next to the USD: <name>.plant.json (what the drawing said, for revision sync) and <name>.findings.json.
"""
import json
import os
from dataclasses import asdict

try:
    import tomllib
except ImportError:
    import tomli as tomllib

from pxr import Gf, Sdf, UsdGeom, UsdLux

from . import catalog as CAT
from . import drawing as DRW
from . import tags as T
from . import units as U
from . import validate as V
from .building import warehouse_a as WA
from .geom import Shapes
from .materials import MaterialLibrary
from .usdio import safe_stage

NV = os.environ.get("FRT_NVIDIA_ASSETS", r"C:/Nvidia_Assets")
HDRI = NV + "/Industrial_NVD/Assets/ArchVis/Industrial/Stages/HDRI/CloudyDay_04_04.hdr"


def load(path):
    with open(path, "rb") as fh:
        r = tomllib.load(fh)
    base = os.path.dirname(os.path.abspath(path))
    rel = lambda p: os.path.normpath(os.path.join(base, p)) if p and not os.path.isabs(p) else p
    r.setdefault("scene", {})
    r["scene"]["output"] = rel(r["scene"].get("output", os.path.splitext(os.path.basename(path))[0] + ".usd"))
    if "drawing" in r:
        r["drawing"]["file"] = rel(r["drawing"]["file"])
        if r["drawing"].get("legend"):
            r["drawing"]["legend"] = rel(r["drawing"]["legend"])
    r["_path"] = os.path.abspath(path)
    return r


# drawing size -> catalog parameters
def params_from_drawing(e, defaults):
    p = dict(defaults.get(e.catalog, {}))
    L, D = e.length, e.depth
    if e.catalog == "industrial_table":
        p.update(length=round(L), depth=round(D))
    elif e.catalog == "lift_table":
        p.update(length=round(L), width=round(D))
    elif e.catalog == "flow_rack":
        bays = max(1, round((L - 8) / 120.0))
        p.update(bays=bays, bay_width=round((L - 8) / bays, 1), depth=round(D))
    elif e.catalog == "conveyor":
        p.update(length=round(L), width=max(30, round(D - 12)))
    elif e.catalog == "pallet_rack":
        bw = p.get("bay_width", 270.0)
        p["bays"] = max(1, round((L - 8) / (bw + 8)))
        p["bay_width"] = round((L - 8) / p["bays"] - 8, 1)
        if D > 180:
            p.update(double=True, frame_depth=round((D - p.get("flue", 30.0)) / 2, 1))
        else:
            p.update(double=False, frame_depth=round(D, 1))
    return p


#           a
#         f   b        seven-segment strokes per character, painted on the floor
#           g
#         e   c
#           d
SEGMENTS = {"0": "abcdef", "1": "bc", "2": "abdeg", "3": "abcdg", "4": "bcfg", "5": "acdfg", "6": "acdefg", "7": "abc",
            "8": "abcdefg", "9": "abcdfg", "A": "abcefg", "B": "cdefg", "C": "adef", "D": "bcdeg", "E": "adefg",
            "F": "aefg", "-": "g"}


def _number_plate(s, plant, x, y, text, gw=40.0, gh=70.0, t=8.0, gap=15.0, pad=20.0):
    """Yellow plate with the station number in black, plant cm, (x, y) = plate's south-west corner, read from the south."""
    text = "".join(ch for ch in text.upper() if ch in SEGMENTS)[:4] or "-"
    w = 2 * pad + len(text) * gw + (len(text) - 1) * gap
    h = 2 * pad + gh

    def rect(ax, ay, bx, by, y0, y1, mat):
        (sx0, sz0), (sx1, sz1) = WA.to_scene(plant, ax, ay), WA.to_scene(plant, bx, by)
        s.box((min(sx0, sx1), y0, min(sz0, sz1)), (max(sx0, sx1), y1, max(sz0, sz1)), mat)

    rect(x, y, x + w, y + h, 0.06, 0.27, "floor_yellow")
    for i, ch in enumerate(text):
        gx, gy = x + pad + i * (gw + gap), y + pad
        seg = {"a": (gx, gy + gh - t, gx + gw, gy + gh), "d": (gx, gy, gx + gw, gy + t),
               "g": (gx, gy + gh / 2 - t / 2, gx + gw, gy + gh / 2 + t / 2),
               "f": (gx, gy + gh / 2, gx + t, gy + gh), "b": (gx + gw - t, gy + gh / 2, gx + gw, gy + gh),
               "e": (gx, gy, gx + t, gy + gh / 2), "c": (gx + gw - t, gy, gx + gw, gy + gh / 2)}
        for k in SEGMENTS[ch]:
            rect(*seg[k], 0.27, 0.34, "rubber_black")


def _floor_marking(stage, root, plant, lib):
    s = Shapes()
    y = 0.25
    for k, (x0, y0, x1, y1) in enumerate(plant.aisles):
        for (a, b) in (((x0, y0), (x1, y0 + 10)), ((x0, y1 - 10), (x1, y1)), ((x0, y0), (x0 + 10, y1)), ((x1 - 10, y0), (x1, y1))):
            ax, az = WA.to_scene(plant, *a)
            bx, bz = WA.to_scene(plant, *b)
            s.box((ax, 0.05, az), (bx, y, bz), "floor_yellow")
    for (x0, y0, x1, y1) in plant.walkways:
        ax, az = WA.to_scene(plant, x0, y0)
        bx, bz = WA.to_scene(plant, x1, y1)
        s.box((ax, 0.04, az), (bx, 0.2, bz), "floor_green")
        for xx in range(int(x0) + 150, int(x1) - 50, 300):                      # white chevrons along the walkway
            cx, cz = WA.to_scene(plant, xx, (y0 + y1) / 2)
            s.box((cx - 40, 0.2, cz - (y1 - y0) * 0.3), (cx + 40, 0.3, cz + (y1 - y0) * 0.3), "floor_white")
    for st in plant.stations:                                                   # station borders: white 8 cm lines
        x0, y0, x1, y1 = st.x0, st.y0, st.x1, st.y1
        for (a, b) in (((x0, y0), (x1, y0 + 8)), ((x0, y1 - 8), (x1, y1)), ((x0, y0), (x0 + 8, y1)), ((x1 - 8, y0), (x1, y1))):
            ax, az = WA.to_scene(plant, *a)
            bx, bz = WA.to_scene(plant, *b)
            s.box((ax, 0.06, az), (bx, 0.26, bz), "floor_white")
        _number_plate(s, plant, x0 + 30, y0 + 30, st.name.split()[-1])         # front-left (south-west) corner
    if s.groups:
        s.build(stage, f"{root}/Geom", lib.refs(list(s.groups)))
    for st in plant.stations:
        sp = stage.DefinePrim(f"{root}/{st.name.replace(' ', '_')}", "Scope")
        T.tag(sp, "station", id=st.name, source=st.ref)
        sp.CreateAttribute("frt:name", Sdf.ValueTypeNames.String, custom=True).Set(st.name)
        sp.CreateAttribute("frt:bounds", Sdf.ValueTypeNames.Float4, custom=True).Set(Gf.Vec4f(st.x0, st.y0, st.x1, st.y1))
        cx, cz = WA.to_scene(plant, (st.x0 + st.x1) / 2, (st.y0 + st.y1) / 2)   # scene point + size, for "go to"
        sp.CreateAttribute("frt:center", Sdf.ValueTypeNames.Float3, custom=True).Set(Gf.Vec3f(cx, 0, cz))
        sp.CreateAttribute("frt:size", Sdf.ValueTypeNames.Float2, custom=True).Set(Gf.Vec2f(st.x1 - st.x0, st.y1 - st.y0))
    for k, a in enumerate(plant.aisles):
        T.tag(stage.DefinePrim(f"{root}/Aisle_{k + 1:02d}", "Scope"), "aisle", id=f"AISLE_{k + 1}")
    for k, a in enumerate(plant.walkways):
        T.tag(stage.DefinePrim(f"{root}/Walkway_{k + 1:02d}", "Scope"), "walkway", id=f"WALK_{k + 1}")


def _look(stage, r):
    if not r.get("look", {}).get("sky", True):
        return
    sky = UsdLux.DomeLight.Define(stage, "/World/Lights/Sky")
    if os.path.exists(HDRI):
        sky.CreateTextureFileAttr().Set(Sdf.AssetPath(HDRI))
    sky.CreateIntensityAttr(900.0)
    UsdGeom.Xformable(sky).AddRotateXOp().Set(-90.0)
    sun = UsdLux.DistantLight.Define(stage, "/World/Lights/Sun")
    sun.CreateIntensityAttr(2500.0)
    sun.CreateAngleAttr(0.6)
    sun.CreateColorAttr(Gf.Vec3f(1.0, 0.95, 0.88))
    UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(-38.0, -60.0, 0.0))


def _cameras(stage, plant):
    W, D = plant.width, plant.depth
    # inside the hall, under the roof (modules are ~10 m clear): a high corner view, plus a top-down plan view
    shots = {"Overview": ((W / 2 - 250, 820, D / 2 - 250), (-W * 0.12, 60, -D * 0.12)),
             "Plan": ((0, max(W, D) * 1.25, 1.0), (0, 0, 0))}
    for name, (eye, at) in shots.items():
        cam = UsdGeom.Camera.Define(stage, f"/World/Cameras/{name}")
        cam.CreateFocalLengthAttr(18.0 if name == "Overview" else 24.0)
        cam.CreateClippingRangeAttr(Gf.Vec2f(10.0, 1.0e6))
        UsdGeom.Xformable(cam).AddTransformOp().Set(
            Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*at), Gf.Vec3d(0, 1, 0) if name != "Plan" else Gf.Vec3d(0, 0, -1)).GetInverse())


def build(recipe, validate=True, log=print):
    r = load(recipe) if isinstance(recipe, str) else recipe
    out = r["scene"]["output"]
    notes = []
    plant = None
    if "drawing" in r:
        d = r["drawing"]
        kw = {}
        if d["file"].lower().endswith(".pdf") and d.get("pdf_scale"):
            kw["scale"] = float(d["pdf_scale"])
        plant = DRW.read_plant(d["file"], d.get("legend") or None, **kw)
        notes += plant.notes
        log(f"drawing: {os.path.basename(d['file'])}: {plant.width / 100:.1f} x {plant.depth / 100:.1f} m, "
            f"{len(plant.doors)} doors, {len(plant.equipment)} equipment items, {len(plant.unknown)} unknown")
    with safe_stage(out) as stage:
        U.setup_stage(stage)
        world = UsdGeom.Xform.Define(stage, "/World").GetPrim()
        stage.SetDefaultPrim(world)
        world.SetCustomDataByKey("frt:recipe", r["_path"] if "_path" in r else "")
        lib = MaterialLibrary(stage, "/World/Looks")
        if plant and r.get("building", {}).get("generator", "warehouse_a") == "warehouse_a":
            b = r.get("building", {})
            notes += WA.build(stage, "/World/Building", plant, b.get("height", "H10m"), b.get("wall_style", "mixed"), lib)
            UsdGeom.Xform.Define(stage, "/World/FloorMarking")
            _floor_marking(stage, "/World/FloorMarking", plant, lib)
        UsdGeom.Xform.Define(stage, "/World/Equipment")
        defaults = r.get("equipment", {}).get("defaults", {})
        placed = 0
        if plant and r.get("equipment", {}).get("from_drawing", True):
            for e in plant.equipment:
                if e.catalog not in CAT.CATALOG:
                    notes.append(f"{e.id}: no catalog item {e.catalog!r}")
                    continue
                x, z = WA.to_scene(plant, e.x, e.y)
                CAT.place(stage, "/World/Equipment", e.catalog, e.id, (x, 0, z), e.rot, params_from_drawing(e, defaults),
                          lib, source=e.ref)
                placed += 1
        for it in r.get("items", []):
            x, z = WA.to_scene(plant, *it["at"]) if plant else (it["at"][0], -it["at"][1])
            p = dict(defaults.get(it["catalog"], {}))
            p.update(it.get("params", {}))
            CAT.place(stage, "/World/Equipment", it["catalog"], it["id"], (x, 0, z), it.get("rot", 0.0), p, lib, source="recipe")
            placed += 1
        _look(stage, r)
        if plant:
            _cameras(stage, plant)
        notes.append(f"equipment: {placed} items placed")
    base = os.path.splitext(out)[0]
    result = {"usd": out, "notes": notes}
    if plant:
        with open(base + ".plant.json", "w", encoding="utf-8") as fh:
            json.dump(asdict(plant), fh, indent=1)
        result["plant"] = base + ".plant.json"
    if validate:
        from pxr import Usd
        st = Usd.Stage.Open(out)
        fl = V.validate(st)
        fl.save(base + ".findings.json")
        result["findings"] = base + ".findings.json"
        result["summary"] = fl.summary()
        log(fl.summary())
    for n in notes:
        log("  - " + n)
    return result
