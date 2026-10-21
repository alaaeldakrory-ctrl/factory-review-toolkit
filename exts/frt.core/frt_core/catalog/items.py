"""The first catalog items: parametric industrial table, lift table, flow rack and belt / roller conveyor, plus pack
assets (pallet with boxes, crate, forklift) placed from the NVIDIA packs. See base.py for the local frame.
"""
import math
import os

from .. import units as U
from .base import Item, Param, new_shapes, register, shapes_to_stage

NV = os.environ.get("FRT_NVIDIA_ASSETS", r"C:/Nvidia_Assets")
PACK = {
    "pallet_block": NV + "/Warehouse_NVD/Assets/DigitalTwin/Assets/Warehouse/Shipping/Pallets/Wood/Block_A/BlockPallet_A01_PR_NVD_01.usd",
    "box_a": NV + "/Industrial_NVD/Assets/ArchVis/Industrial/Containers/Cardboard/Cardbox_A1.usd",
    "box_b": NV + "/Industrial_NVD/Assets/ArchVis/Industrial/Containers/Cardboard/Cardbox_B2.usd",
    "crate": NV + "/Warehouse_NVD/Assets/DigitalTwin/Assets/Warehouse/Shipping/Wood_Crate_on_Pallet/HeavyDuty_A/HeavyDutyWoodCrateAssembly_A02_PR_NVD_01.usd",
    "forklift": NV + "/Warehouse_NVD/Assets/DigitalTwin/Assets/Warehouse/Equipment/Forklifts/Forklift_C/Forklift_C01_PR_V_NVD_01.usd",
}


# ------------------------------------------------------------------------------------------ industrial table
class IndustrialTable(Item):
    def footprint(self, p):
        return p["length"], p["depth"]

    def build(self, stage, path, p, lib):
        L, D, H = p["length"], p["depth"], p["height"]
        frame, top = p["frame"], {"steel": "galvanised", "wood": "wood_top", "esd": "esd_mat"}[p["top"]]
        s = new_shapes()
        t_top, leg = 4.0, 5.0
        x0, x1, z0, z1 = -L / 2, L / 2, -D / 2, D / 2
        s.box((x0, H - t_top, z0), (x1, H, z1), top)                                     # worktop
        s.box((x0 + 1, H - t_top - 6, z0 + 1), (x1 - 1, H - t_top, z0 + 4), frame)        # apron rails
        s.box((x0 + 1, H - t_top - 6, z1 - 4), (x1 - 1, H - t_top, z1 - 1), frame)
        legs = [(x0 + 3, z0 + 3), (x1 - 3 - leg, z0 + 3), (x0 + 3, z1 - 3 - leg), (x1 - 3 - leg, z1 - 3 - leg)]
        for lx, lz in legs:
            s.box((lx, 3.5, lz), (lx + leg, H - t_top, lz + leg), frame)
            s.cylinder((lx + leg / 2, 0, lz + leg / 2), (lx + leg / 2, 3.5, lz + leg / 2), 3.0, "rubber_black", 10)
        for z in (z0 + 3, z1 - 3 - leg):                                                   # side stretchers
            s.box((x0 + 3, 16, z), (x1 - 3, 20, z + leg), frame)
        if p["shelf"]:
            s.box((x0 + 4, 20, z0 + 4), (x1 - 4, 22, z1 - 4), "galvanised")
        if p["drawers"]:
            w = min(55.0, L * 0.35)
            dx0, dx1 = x1 - 6 - w, x1 - 6
            dy1 = H - t_top - 6
            dy0 = dy1 - 14 * p["drawers"]
            s.box((dx0, dy0, z0 + 6), (dx1, dy1, z1 - 2), frame)
            for k in range(p["drawers"]):
                y = dy1 - 14 * (k + 1) + 1
                s.box((dx0 + 1, y, z1 - 2), (dx1 - 1, y + 12, z1 - 0.8), "powder_lightgrey")
                s.box((dx0 + w / 2 - 8, y + 5, z1 - 0.8), (dx0 + w / 2 + 8, y + 7, z1 + 0.6), "steel_dark")
        attach = {"top": (0, H, 0), "work_front": (0, H, z1 - 15)}
        if p["light_frame"]:
            hh = H + 110
            for x in (x0 + 2, x1 - 7):
                s.box((x, H, z0 + 1), (x + 5, hh, z0 + 6), frame)
            s.box((x0 + 2, hh - 5, z0 + 1), (x1 - 2, hh, z0 + 6), frame)
            s.box((x0 + 7, H + 10, z0 + 3), (x1 - 7, H + 70, z0 + 4), "powder_lightgrey")      # pegboard / back panel
            s.box((x0 + 10, hh - 9, z0 + 6), (x1 - 10, hh - 5, z0 + 30), "aluminium")          # light fitting
            s.box((x0 + 12, hh - 9.4, z0 + 8), (x1 - 12, hh - 9, z0 + 28), "light_emit")
            attach["light"] = (0, hh - 9, z0 + 18)
        shapes_to_stage(stage, path, s, lib)
        return attach


register(IndustrialTable("industrial_table", "table", "Industrial work table", {
    "length": Param(180.0, "worktop length", 60, 400, "cm"),
    "depth": Param(80.0, "worktop depth", 50, 150, "cm"),
    "height": Param(90.0, "worktop height", 60, 120, "cm"),
    "top": Param("wood", "worktop", choices=("steel", "wood", "esd")),
    "frame": Param("powder_blue", "frame finish (material library look)"),
    "shelf": Param(True, "lower shelf"),
    "drawers": Param(2, "drawer unit under the top (0-3)", 0, 3),
    "light_frame": Param(True, "uprights with back panel and a light"),
}))


# ------------------------------------------------------------------------------------------ lift table
class LiftTable(Item):
    def footprint(self, p):
        return p["length"], p["width"]

    def build(self, stage, path, p, lib):
        L, W = p["length"], p["width"]
        low, high = p["lowered"], p["raised"]
        h = low + (high - low) * p["lift"]
        c = p["color"]
        s = new_shapes()
        base_h, plat_t = 8.0, 9.0
        # base frame
        for z in (-W / 2 + 4, W / 2 - 14):
            s.box((-L / 2 + 4, 0, z), (L / 2 - 4, base_h, z + 10), c)
        for x in (-L / 2 + 4, L / 2 - 14):
            s.box((x, 0, -W / 2 + 4), (x + 10, base_h, W / 2 - 4), c)
        # platform: deck plate on a channel frame, yellow / black toe guard strip
        s.box((-L / 2, h - plat_t, -W / 2), (L / 2, h - 1.5, W / 2), c)
        s.box((-L / 2 + 0.5, h - 1.5, -W / 2 + 0.5), (L / 2 - 0.5, h, W / 2 - 0.5), "steel_tread")
        s.box((-L / 2 - 0.4, h - plat_t, W / 2), (L / 2 + 0.4, h - plat_t + 3, W / 2 + 0.4), "powder_yellow")
        # scissor legs: two crossed bars per side, length fixed so the span shrinks as the table rises
        v = max(h - plat_t - base_h, 2.0)
        bar = 0.88 * L
        span = math.sqrt(max(bar * bar - v * v, 1.0))
        y0, y1 = base_h, h - plat_t
        for zs in (-1, 1):
            z = zs * (W / 2 - 16)
            for sx in (-1, 1):
                a = (sx * span / 2, y0, z)
                b = (-sx * span / 2, y1, z)
                mid = tuple((a[i] + b[i]) / 2 for i in range(3))
                d = [b[i] - a[i] for i in range(3)]
                n = math.sqrt(sum(x * x for x in d))
                ax = tuple(x / n for x in d)
                s.obox(mid, (n, 9.0, 3.0), (ax, (-ax[1], ax[0], 0.0), (0, 0, 1)), c)
                s.cylinder((a[0], a[1], z - 3), (a[0], a[1], z + 3), 2.5, "steel_dark", 10)
                s.cylinder((b[0], b[1], z - 3), (b[0], b[1], z + 3), 2.5, "steel_dark", 10)
            s.cylinder((0, (y0 + y1) / 2, z - 4), (0, (y0 + y1) / 2, z + 4), 2.2, "chrome", 10)       # centre pivot
        # hydraulic cylinder between the two scissor pairs
        s.cylinder((-span * 0.3, base_h + 6, 0), (span * 0.15, (y0 + y1) / 2 + v * 0.2, 0), 5.0, "steel_dark", 12)
        s.cylinder((span * 0.05, (y0 + y1) / 2, 0), (span * 0.15, (y0 + y1) / 2 + v * 0.2, 0), 2.6, "chrome", 12)
        # power unit + foot pedal beside the table
        px = L / 2 + 6
        s.box((px, 0, -18), (px + 28, 34, 18), "powder_dark")
        s.box((px + 2, 34, -16), (px + 26, 38, 16), "steel_dark")
        s.box((px + 34, 0, -12), (px + 60, 8, 12), "powder_dark")
        s.box((px + 38, 8, -9), (px + 56, 10, 9), "rubber_black")
        shapes_to_stage(stage, path, s, lib)
        return {"platform": (0, h, 0), "pedal": (px + 47, 10, 0)}


register(LiftTable("lift_table", "lift_table", "Scissor lift table", {
    "length": Param(130.0, "platform length", 60, 300, "cm"),
    "width": Param(80.0, "platform width", 50, 200, "cm"),
    "lowered": Param(22.0, "lowered platform height", 15, 60, "cm"),
    "raised": Param(105.0, "raised platform height", 50, 200, "cm"),
    "lift": Param(0.55, "current position 0 (down) .. 1 (up)", 0, 1),
    "color": Param("powder_blue", "frame finish"),
}))


# ------------------------------------------------------------------------------------------ flow rack
class FlowRack(Item):
    def footprint(self, p):
        return p["bays"] * p["bay_width"] + 8, p["depth"]

    def build(self, stage, path, p, lib):
        nb, bw, D, H, nl = p["bays"], p["bay_width"], p["depth"], p["height"], p["levels"]
        tilt = math.radians(p["tilt_deg"])
        frame = p["frame"]
        s = new_shapes()
        L = nb * bw
        x0 = -L / 2
        post = 7.0
        zf, zb = D / 2 - post, -D / 2                     # front (pick side, +Z) and back posts
        for k in range(nb + 1):
            x = x0 + k * bw - post / 2
            for z in (zf, zb):
                s.box((x, 0, z), (x + post, H, z + post), frame)
                s.box((x - 3, 0, z - 3), (x + post + 3, 1.0, z + post + 3), "steel_dark")     # base plate
            for y in (15.0, H * 0.5, H - 5):                                                    # frame braces
                s.box((x + 2, y, zb + post), (x + post - 2, y + 4, zf), frame)
        rise = (D - 2 * post) * math.tan(tilt)
        y_lo, y_hi = 35.0, H - 35.0
        spacing = (y_hi - y_lo) / max(nl - 1, 1)
        attach = {}
        for lv in range(nl):
            yf = y_lo + lv * spacing                       # front beam (low); back beam higher: cartons roll to the front
            yb = yf + rise
            s.box((x0, yf - 9, zf), (x0 + L, yf, zf + post), frame)
            s.box((x0, yb - 9, zb), (x0 + L, yb, zb + post), frame)
            s.box((x0, yf, zf + post - 1.5), (x0 + L, yf + 4.5, zf + post), "powder_yellow")      # stop lip / label rail
            s.box((x0 + 1, yf - 8, zf + post), (x0 + L - 1, yf - 3, zf + post + 0.4), "floor_white")   # labels
            nt = max(3, int(bw // 22))
            for b in range(nb):
                bx0 = x0 + b * bw + post
                bx1 = x0 + (b + 1) * bw - post
                for t in range(nt):
                    tx = bx0 + (t + 0.5) * (bx1 - bx0) / nt
                    a = (tx, yb, zb + post)
                    c = (tx, yf, zf)
                    n = math.dist(a, c)
                    ax = ((c[0] - a[0]) / n, (c[1] - a[1]) / n, (c[2] - a[2]) / n)
                    mid = ((a[0] + c[0]) / 2, (a[1] + c[1]) / 2 - 1.0, (a[2] + c[2]) / 2)
                    s.obox(mid, (2.0, 2.4, n), ((1, 0, 0), (0, ax[2], -ax[1]), ax), "galvanised")
                    for r in range(int(n // 7.5)):
                        u = (r + 0.5) * 7.5 / n
                        rp = (a[0], a[1] + (c[1] - a[1]) * u + 0.6, a[2] + (c[2] - a[2]) * u)
                        s.cylinder((rp[0] - 1.6, rp[1], rp[2]), (rp[0] + 1.6, rp[1], rp[2]), 1.4, "plastic_grey", 6, caps=False)
                if p["totes"] and (b + lv) % 3 != 2:
                    for k in range(3):                    # a lane of blue totes resting on the rollers
                        tz = zf - 24 - k * 34
                        ty = yf + (zf - tz) / max(zf - (zb + post), 1) * rise + 2.0
                        s.box((bx0 + 6, ty, tz - 15), (bx1 - 6, ty + 18, tz + 15), "plastic_blue")
                attach[f"pick_L{lv + 1}_B{b + 1}"] = (x0 + (b + 0.5) * bw, yf + 10, zf)
        shapes_to_stage(stage, path, s, lib)
        return attach


register(FlowRack("flow_rack", "flow_rack", "Carton flow rack", {
    "bays": Param(3, "bays side by side", 1, 8),
    "bay_width": Param(120.0, "bay width", 60, 250, "cm"),
    "depth": Param(180.0, "depth (load side to pick side)", 60, 400, "cm"),
    "height": Param(200.0, "frame height", 120, 350, "cm"),
    "levels": Param(4, "roller levels", 2, 7),
    "tilt_deg": Param(5.0, "roller lane slope", 2, 12, "deg"),
    "frame": Param("powder_blue", "frame finish"),
    "totes": Param(True, "fill lanes with totes"),
}))


# ------------------------------------------------------------------------------------------ conveyor
class Conveyor(Item):
    def footprint(self, p):
        return p["length"], p["width"] + 12

    def build(self, stage, path, p, lib):
        L, W, H = p["length"], p["width"], p["height"]
        frame = p["frame"]
        s = new_shapes()
        x0, x1 = -L / 2, L / 2
        side_h = 16.0
        for zs in (-1, 1):
            z = zs * (W / 2 + 3)
            s.box((x0, H - side_h + 2, z - 3), (x1, H + 2, z + 3), frame)                # side channels
            if p["guide_rails"]:
                s.box((x0, H + 12, z - 1), (x1, H + 15, z + 1), "galvanised")
                for k in range(int(L // 150) + 1):
                    gx = min(x0 + 20 + k * 150, x1 - 5)
                    s.box((gx - 1, H + 2, z - 1), (gx + 1, H + 12, z + 1), "galvanised")
        if p["type"] == "belt":
            s.box((x0 + 4, H - 1.0, -W / 2), (x1 - 4, H, W / 2), "belt_black")
            s.box((x0 + 4, H - 6, -W / 2 + 2), (x1 - 4, H - 1.0, W / 2 - 2), "steel_dark")       # slider bed
            for x in (x0 + 6, x1 - 6):
                s.cylinder((x, H - 5, -W / 2), (x, H - 5, W / 2), 5.0, "steel_dark", 14)          # end pulleys
        else:
            pitch = 10.0
            for k in range(int((L - 8) // pitch)):
                x = x0 + 6 + k * pitch
                s.cylinder((x, H - 2.5, -W / 2 + 1), (x, H - 2.5, W / 2 - 1), 2.5, "galvanised", 10)
        n_leg = max(2, int(math.ceil(L / 250.0)) + 1)
        for k in range(n_leg):
            x = x0 + 15 + k * (L - 30) / (n_leg - 1)
            for zs in (-1, 1):
                z = zs * (W / 2 - 2)
                s.box((x - 2.5, 4, z - 2.5), (x + 2.5, H - side_h + 2, z + 2.5), frame)
                s.cylinder((x, 0, z), (x, 4, z), 3.0, "rubber_black", 10)
            s.box((x - 2, 25, -W / 2), (x + 2, 29, W / 2), frame)
        if p["motor"]:
            mx = x1 - 30
            s.box((mx - 15, H - side_h - 26, W / 2 + 6), (mx + 15, H - side_h - 2, W / 2 + 26), "powder_dark")
            s.cylinder((mx, H - side_h - 14, W / 2 + 26), (mx, H - side_h - 14, W / 2 + 44), 9.0, "powder_grey", 16)
        shapes_to_stage(stage, path, s, lib)
        return {"start": (x0 + 20, H, 0), "end": (x1 - 20, H, 0), "middle": (0, H, 0)}


register(Conveyor("conveyor", "conveyor", "Belt / roller conveyor", {
    "length": Param(600.0, "length", 100, 5000, "cm"),
    "width": Param(60.0, "belt / roller width", 30, 150, "cm"),
    "height": Param(85.0, "transport height", 30, 200, "cm"),
    "type": Param("roller", "transport", choices=("belt", "roller")),
    "frame": Param("powder_lightgrey", "frame finish"),
    "guide_rails": Param(True, "side guide rails"),
    "motor": Param(True, "drive motor at the +X end"),
}))


# ------------------------------------------------------------------------------------------ pack assets
class PackAsset(Item):
    """A NVIDIA pack asset placed (and wrapped to Y-up cm) as is."""

    def __init__(self, key, role, title, asset, size, params=None):
        super().__init__(key, role, title, params or {}, source="pack", licence="local")
        self.asset, self.size = asset, size

    def footprint(self, p):
        return self.size

    def build(self, stage, path, p, lib):
        _, (lo, hi) = U.place_asset(stage, f"{path}/Pack", self.asset)
        return {"top": (0, hi[1] - lo[1], 0)}


class PalletWithBoxes(Item):
    def footprint(self, p):
        return 120.0, 100.0

    def build(self, stage, path, p, lib):
        _, (lo, hi) = U.place_asset(stage, f"{path}/Pallet", PACK["pallet_block"], yaw=90)
        top = hi[1] - lo[1]
        box = PACK["box_b"] if p["box"] == "cube" else PACK["box_a"]
        _, (blo, bhi) = U.place_asset(stage, f"{path}/Probe", box)
        stage.RemovePrim(f"{path}/Probe")
        bx, by, bz = bhi[0] - blo[0], bhi[1] - blo[1], bhi[2] - blo[2]
        nx, nz = max(1, int(118 // bx)), max(1, int(98 // bz))
        k = 0
        for layer in range(p["layers"]):
            for i in range(nx):
                for j in range(nz):
                    x = -nx * bx / 2 + (i + 0.5) * bx
                    z = -nz * bz / 2 + (j + 0.5) * bz
                    U.place_asset(stage, f"{path}/Boxes/B{k:03d}", box, pos=(x, top + layer * by, z))
                    k += 1
        if p["wrap"]:
            s = new_shapes()
            h = p["layers"] * by
            s.box((-nx * bx / 2 - 0.4, top, -nz * bz / 2 - 0.4), (nx * bx / 2 + 0.4, top + h * 0.95, nz * bz / 2 + 0.4), "stretch_wrap")
            shapes_to_stage(stage, path, s, lib)
        return {"top": (0, top + p["layers"] * by, 0), "fork_entry": (0, 6, 50)}


register(PalletWithBoxes("pallet_boxes", "pallet", "Pallet with boxes", {
    "layers": Param(3, "box layers", 1, 6),
    "box": Param("cube", "box size", choices=("cube", "long")),
    "wrap": Param(False, "stretch wrap"),
}, source="mixed", licence="local"))
register(PackAsset("crate", "crate", "Wooden crate on pallet", PACK["crate"], (120.0, 100.0)))
register(PackAsset("forklift", "forklift", "Counterbalance forklift", PACK["forklift"], (137.0, 373.0)))


# ------------------------------------------------------------------------------------------ pallet rack
class PalletRack(Item):
    """Selective pallet racking: rows of bays, beam levels, single or back-to-back (double) rows, optional loads."""

    def _dims(self, p):
        up = 8.0
        L = p["bays"] * p["bay_width"] + (p["bays"] + 1) * up
        D = p["frame_depth"] * (2 if p["double"] else 1) + (p["flue"] if p["double"] else 0.0)
        return L, D, up

    def footprint(self, p):
        L, D, _ = self._dims(p)
        return L, D

    def build(self, stage, path, p, lib):
        L, D, up = self._dims(p)
        nb, bw, fd, H = p["bays"], p["bay_width"], p["frame_depth"], p["height"]
        frame, beam = p["frame"], p["beam"]
        s = new_shapes()
        x0 = -L / 2
        rows = [(-D / 2, -D / 2 + fd)] + ([(D / 2 - fd, D / 2)] if p["double"] else [])
        nl = p["levels"]
        levels = [150.0 + k * (H - 220.0) / max(nl - 1, 1) for k in range(nl)] if nl > 1 else [150.0]
        attach = {}
        load_k = 0
        for r, (z0, z1) in enumerate(rows):
            for k in range(nb + 1):
                x = x0 + k * (bw + up)
                for z in (z0, z1 - 7.0):
                    s.box((x, 0, z), (x + up, H, z + 7.0), frame)
                    s.box((x - 3, 0, z - 3), (x + up + 3, 1.0, z + 10), "galvanised")
                # frame bracing: horizontals + zig-zag diagonals
                y = 20.0
                step = 120.0
                while y + step < H - 20:
                    a = (x + up / 2, y, z0 + 7)
                    b = (x + up / 2, y + step, z1 - 7)
                    s.cylinder(a, b, 1.6, frame, 6)
                    s.cylinder((x + up / 2, y, z0 + 7), (x + up / 2, y, z1 - 7), 1.6, frame, 6)
                    y += step
            for b in range(nb):
                bx0 = x0 + b * (bw + up) + up
                bx1 = bx0 + bw
                for lv, yb in enumerate(levels):
                    for z in (z0, z1 - 5.0):
                        s.box((bx0, yb - 12, z), (bx1, yb, z + 5.0), beam)
                    if p["wire_decks"]:
                        s.box((bx0, yb - 3, z0 + 5), (bx1, yb - 2, z1 - 5), "galvanised")
                    attach[f"R{r + 1}_B{b + 1}_L{lv + 1}"] = ((bx0 + bx1) / 2, yb, (z0 + z1) / 2)
                # loads: pallets with wrapped cartons on the floor and the beam levels
                n_pal = max(1, int(bw // 125))
                for lv, yb in enumerate([0.0] + levels):
                    clear = (levels[lv] - 12 - yb) if lv < len(levels) else (H - yb)
                    for q in range(n_pal):
                        load_k += 1
                        if (load_k * 7 + r * 3 + b) % 10 >= p["fill"] * 10:
                            continue
                        px = bx0 + (q + 0.5) * bw / n_pal
                        pz = (z0 + z1) / 2
                        hgt = min(clear - 18, 95 + (load_k % 3) * 15)
                        if hgt < 30:
                            continue
                        s.box((px - 50, yb, pz - 60), (px + 50, yb + 14, pz + 60), "wood_top")
                        s.box((px - 47, yb + 14, pz - 56), (px + 47, yb + 14 + hgt, pz + 56),
                              "stretch_wrap" if load_k % 4 == 0 else "cardboard")
        if p["double"]:
            for k in range(nb + 1):
                x = x0 + k * (bw + up)
                for y in (H * 0.35, H * 0.75):
                    s.box((x + 1, y, -D / 2 + fd), (x + up - 1, y + 5, D / 2 - fd), "galvanised")   # row spacers
        shapes_to_stage(stage, path, s, lib)
        return attach


register(PalletRack("pallet_rack", "pallet_rack", "Selective pallet racking", {
    "bays": Param(3, "bays in the row", 1, 30),
    "bay_width": Param(270.0, "clear beam length per bay", 100, 400, "cm"),
    "frame_depth": Param(110.0, "upright frame depth", 60, 150, "cm"),
    "height": Param(600.0, "upright height", 200, 1500, "cm"),
    "levels": Param(3, "beam levels above the floor position", 1, 8),
    "double": Param(False, "back-to-back double row"),
    "flue": Param(30.0, "gap between back-to-back rows", 10, 60, "cm"),
    "frame": Param("powder_blue", "upright finish"),
    "beam": Param("powder_grey", "beam finish (not red / orange: red marks quality holds)"),
    "wire_decks": Param(True, "wire decks on the beams"),
    "fill": Param(0.7, "share of positions holding a pallet load", 0, 1),
}))
