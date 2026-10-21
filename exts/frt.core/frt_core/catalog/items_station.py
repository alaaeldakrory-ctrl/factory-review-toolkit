"""Assembly-line items: a generic scissor lift (the product at a station, until a model library replaces it), a staged
incoming kit, and the small things every station has: trash bin, computer station, tool cart, drive-on ramp, platform
scale. Same local frame as every catalog item (base.py): origin at the footprint centre on the floor, +X length,
+Z front, cm.
"""
import math

from .base import Item, Param, new_shapes, register, shapes_to_stage


# ------------------------------------------------------------------------------------------ scissor lift (generic)
class ScissorUnit(Item):
    """Electric slab scissor lift, generic proportions from its footprint. lift_height raises the platform (0 = stowed)."""

    def footprint(self, p):
        return p["length"], p["width"]

    def build(self, stage, path, p, lib):
        L, Wd, lift = p["length"], p["width"], p["lift_height"]
        ch, arms, deck, rail = p["chassis"], p["arms"], p["deck"], p["rails"]
        s = new_shapes()
        hl, hw = L / 2, Wd / 2
        r = 15.0 if Wd < 100 else 19.0
        y_ch0, y_ch1 = 9.0, 40.0
        s.box((-hl + 6, y_ch0, -hw + 3), (hl - 6, y_ch1, hw - 3), ch)                    # chassis
        s.box((-hl + 2, y_ch0 + 6, -hw + 8), (-hl + 6, y_ch1 - 4, hw - 8), "steel_dark")    # bumpers
        s.box((hl - 6, y_ch0 + 6, -hw + 8), (hl - 2, y_ch1 - 4, hw - 8), "steel_dark")
        for sx in (-1, 1):                                                                   # wheels, axle along Z
            for sz in (-1, 1):
                x, z = sx * (hl - 32), sz * (hw - 7)
                s.cylinder((x, r, z - 6), (x, r, z + 6), r, "rubber_black", 16)
                s.cylinder((x, r, z - 6.5), (x, r, z + 6.5), r * 0.55, "steel_dark", 12)
        for sz in (-1, 1):                                                                   # pothole guards, trays
            s.box((-hl * 0.45, y_ch0 - 4, sz * hw - (3 if sz > 0 else -3) - 1.5), (hl * 0.45, y_ch0 + 18,
                  sz * hw - (3 if sz > 0 else -3) + 1.5), "steel_dark")
        s.box((-hl + 14, y_ch1, -hw + 10), (-hl + 70, y_ch1 + 14, hw - 10), "plastic_black")   # battery / hydraulic tray
        # scissor stack: n pairs of X arms per side, from the chassis to the platform
        n = p["pairs"]
        stow = 16.0                                                                          # stack height stowed
        h = stow + lift
        a = L * 0.78                                                                         # arm length
        step = h / n
        ang = math.asin(min(0.98, step / a))
        run = a * math.cos(ang)
        y0 = y_ch1 + 6
        for sz in (-1, 1):
            z = sz * (hw - 13)
            for k in range(n):
                yb, yt = y0 + k * step, y0 + (k + 1) * step
                for d in (1, -1):
                    c = (0.0, (yb + yt) / 2, z + d * 2.6 * sz)
                    ax = (math.cos(ang), d * math.sin(ang), 0.0)
                    s.obox(c, (a, 6.0, 3.5), (ax, (-ax[1], ax[0], 0.0), (0.0, 0.0, 1.0)), arms)
                s.cylinder((0, (yb + yt) / 2, z - 5), (0, (yb + yt) / 2, z + 5), 2.0, "chrome", 8)
                for x in (-run / 2, run / 2):
                    s.cylinder((x, yt, z - 5), (x, yt, z + 5), 1.8, "steel_dark", 8)
        for k in range(n):                                                                   # cross tubes
            y = y0 + (k + 0.5) * step
            s.cylinder((0, y, -hw + 13), (0, y, hw - 13), 2.2, arms, 8)
        # platform with roll-out extension, guardrails, kick plates, control box
        yd = y0 + h
        ext = p["deck_extension"]
        s.box((-hl + 4, yd, -hw + 1), (hl - 4 + ext, yd + 9, hw - 1), deck)
        s.box((-hl + 4.5, yd + 9, -hw + 1.5), (hl - 4.5 + ext, yd + 9.6, hw - 1.5), "steel_tread")
        xr0, xr1 = -hl + 6, hl - 6 + ext
        rh = 110.0
        posts = [xr0, xr1] + [xr0 + (xr1 - xr0) * f for f in (0.33, 0.66)]
        for x in posts:
            for sz in (-1, 1):
                s.box((x - 2, yd + 9, sz * (hw - 3) - 2), (x + 2, yd + 9 + rh, sz * (hw - 3) + 2), rail)
        for zz in (-hw + 3, hw - 3):
            for y in (yd + 9 + rh, yd + 9 + rh * 0.55):
                s.box((xr0 - 2, y - 2, zz - 2), (xr1 + 2, y + 2, zz + 2), rail)
            s.box((xr0, yd + 9, zz - 0.8), (xr1, yd + 24, zz + 0.8), deck)
        for x in (xr0, xr1):
            for y in (yd + 9 + rh, yd + 9 + rh * 0.55):
                s.box((x - 2, y - 2, -hw + 3), (x + 2, y + 2, hw - 3), rail)
            s.box((x - 0.8, yd + 9, -hw + 3), (x + 0.8, yd + 24, hw - 3), deck)
        s.box((xr0 + 3, yd + 80, -14), (xr0 + 13, yd + 102, 14), "plastic_black")             # platform controller
        s.box((-hl + 20, y_ch1 - 2, hw - 3), (-hl + 48, y_ch1 + 14, hw + 1), "plastic_black")  # ground controls
        shapes_to_stage(stage, path, s, lib)
        return {"platform": (0, yd + 9, 0), "controller": (xr0 + 8, yd + 92, 0), "ground_controls": (-hl + 34, y_ch1 + 6, hw + 1)}


register(ScissorUnit("scissor_unit", "product", "Scissor lift (generic, until the model library)", {
    "length": Param(245.0, "overall length", 120, 600, "cm"),
    "width": Param(81.0, "overall width", 60, 260, "cm"),
    "lift_height": Param(0.0, "platform raised above stowed", 0, 1600, "cm"),
    "pairs": Param(4, "scissor pairs", 2, 8),
    "deck_extension": Param(0.0, "roll-out deck extended", 0, 150, "cm"),
    "model": Param("", "model name (from the drawing or the line)"),
    "chassis": Param("powder_dark", "chassis finish"),
    "arms": Param("powder_orange", "scissor arms finish"),
    "deck": Param("powder_grey", "platform finish"),
    "rails": Param("powder_orange", "guardrails finish"),
}, source="generator"))


# ------------------------------------------------------------------------------------------ staged kit
class StagedKit(Item):
    """Incoming kit for one unit, staged on its floor spot: chassis on a dolly, scissor-arm bundle, boxed parts."""

    def footprint(self, p):
        return p["length"], p["width"]

    def build(self, stage, path, p, lib):
        L, Wd = p["length"], p["width"]
        s = new_shapes()
        hl, hw = L / 2 - 4, Wd / 2 - 4
        s.box((-hl, 6, -hw), (hl, 12, hw), "steel_dark")                                      # dolly frame
        for sx in (-1, 1):
            for sz in (-1, 1):
                s.cylinder((sx * (hl - 10), 3, sz * (hw - 8) - 2), (sx * (hl - 10), 3, sz * (hw - 8) + 2), 3.5,
                           "rubber_black", 10)
        cl = hl * 0.92
        s.box((-cl, 12, -hw + 6), (cl, 38, hw - 6), p["chassis"])                             # bare chassis
        s.box((-cl + 15, 38, -hw + 14), (-cl + 55, 50, hw - 14), "plastic_black")
        for k in range(3):                                                                    # arm bundle on top
            y = 40 + k * 7
            for zz in (-hw * 0.5, hw * 0.1):
                s.box((-cl * 0.85, y, zz), (cl * 0.85, y + 6, zz + 6), p["arms"])
        s.box((cl * 0.2, 62, -hw + 8), (cl * 0.95, 100, hw * 0.2), "cardboard")               # boxed parts
        s.box((cl * 0.2, 62, hw * 0.25), (cl * 0.7, 88, hw - 8), "cardboard")
        s.box((cl * 0.2 - 0.3, 79, -hw + 7.7), (cl * 0.95 + 0.3, 82, hw * 0.2 + 0.3), "tape_brown")
        shapes_to_stage(stage, path, s, lib)
        return {"pick": (0, 60, hw)}


register(StagedKit("staged_kit", "product", "Staged incoming kit (one unit)", {
    "length": Param(240.0, "spot length", 80, 600, "cm"),
    "width": Param(85.0, "spot width", 40, 300, "cm"),
    "model": Param("", "model (from the drawing layer)"),
    "chassis": Param("powder_dark", "chassis finish"),
    "arms": Param("powder_orange", "arms finish"),
}))


# ------------------------------------------------------------------------------------------ small station items
class TrashBin(Item):
    def footprint(self, p):
        return p["diameter"], p["diameter"]

    def build(self, stage, path, p, lib):
        r, H = p["diameter"] / 2, p["height"]
        s = new_shapes()
        s.cylinder((0, 0, 0), (0, H - 4, 0), r * 0.92, p["colour"], 24)
        s.cylinder((0, H - 6, 0), (0, H, 0), r, "plastic_black", 24)
        s.cylinder((0, H, 0), (0, H + 0.6, 0), r * 0.85, "stretch_wrap", 24)                  # liner showing
        shapes_to_stage(stage, path, s, lib)
        return {"opening": (0, H, 0)}


register(TrashBin("trash_bin", "prop", "Trash bin", {
    "diameter": Param(60.0, "diameter", 30, 150, "cm"),
    "height": Param(90.0, "height", 40, 130, "cm"),
    "colour": Param("plastic_grey", "body colour"),
}))


class ComputerStation(Item):
    def footprint(self, p):
        return p["length"], p["depth"]

    def build(self, stage, path, p, lib):
        L, D = p["length"], p["depth"]
        s = new_shapes()
        s.box((-L / 2, 0, -D / 2), (L / 2, 4, D / 2), "steel_dark")                           # base plate
        s.box((-4, 4, -4), (4, 100, 4), p["frame"])                                           # column
        s.box((-L / 2 + 4, 100, -D / 2 + 6), (L / 2 - 4, 104, D / 2), p["frame"])             # work shelf
        s.box((-L / 2 + 10, 104, D / 2 - 24), (L / 2 - 10, 106, D / 2 - 6), "plastic_black")  # keyboard
        s.box((-2, 104, -D / 2 + 8), (2, 140, -D / 2 + 12), "steel_dark")                     # monitor arm
        s.box((-27, 128, -D / 2 + 12), (27, 162, -D / 2 + 15), "plastic_black")               # monitor
        s.box((-25, 130, -D / 2 + 15), (25, 160, -D / 2 + 15.4), "light_emit")
        s.box((L / 2 - 30, 4, -D / 2 + 6), (L / 2 - 6, 50, D / 2 - 6), "plastic_black")       # PC
        shapes_to_stage(stage, path, s, lib)
        return {"screen": (0, 145, -D / 2 + 15)}


register(ComputerStation("computer_station", "prop", "Computer station (standing)", {
    "length": Param(70.0, "footprint length", 40, 200, "cm"),
    "depth": Param(60.0, "footprint depth", 40, 150, "cm"),
    "frame": Param("powder_grey", "frame finish"),
}))


class ToolCart(Item):
    def footprint(self, p):
        return p["length"], p["width"]

    def build(self, stage, path, p, lib):
        L, Wd, H, c = p["length"], p["width"], p["height"], p["colour"]
        s = new_shapes()
        hl, hw = L / 2 - 8, Wd / 2
        for sx in (-1, 1):
            for sz in (-1, 1):
                x, z = sx * (hl - 4), sz * (hw - 4)
                s.box((x - 2, 12, z - 2), (x + 2, H, z + 2), c)
                s.cylinder((x, 5, z - 2), (x, 5, z + 2), 5, "rubber_black", 10)
                s.box((x - 2, 10, z - 2), (x + 2, 12, z + 2), "steel_dark")
        for y in (18, (18 + H) / 2, H - 3):
            s.box((-hl, y, -hw), (hl, y + 3, hw), c)
            s.box((-hl, y + 3, -hw), (hl, y + 6, -hw + 1), c)
            s.box((-hl, y + 3, hw - 1), (hl, y + 6, hw), c)
        s.cylinder((hl + 6, H - 10, -hw + 8), (hl + 6, H - 10, hw - 8), 1.5, "chrome", 8)       # handle
        for z in (-hw + 8, hw - 8):
            s.box((hl, H - 12, z - 1.5), (hl + 7, H - 8, z + 1.5), "chrome")
        s.box((-hl + 10, H + 3, -hw + 8), (-hl + 40, H + 15, hw * 0.2), "powder_red")          # tool box on top
        shapes_to_stage(stage, path, s, lib)
        return {"top": (0, H + 3, 0), "handle": (hl + 6, H - 10, 0)}


register(ToolCart("tool_cart", "prop", "Tool cart", {
    "length": Param(90.0, "length incl. handle", 50, 200, "cm"),
    "width": Param(50.0, "width", 30, 100, "cm"),
    "height": Param(95.0, "top shelf height", 60, 130, "cm"),
    "colour": Param("powder_blue", "finish"),
}))


class Ramp(Item):
    """Drive-on ramp: a tread-plate slope rising along +X to `height`, then flat for the rest."""

    def footprint(self, p):
        return p["length"], p["width"]

    def build(self, stage, path, p, lib):
        L, Wd, Hh = p["length"], p["width"], p["height"]
        s = new_shapes()
        sl = min(L * 0.6, Hh / math.tan(math.radians(p["slope_deg"])))
        ang = math.atan2(Hh, sl)
        plen = math.hypot(sl, Hh)
        c = (-L / 2 + sl / 2, Hh / 2 + 0.8, 0.0)          # lifted by half the plate: its lower edge meets the floor
        ax = (math.cos(ang), math.sin(ang), 0.0)
        s.obox(c, (plen, 1.5, Wd - 4), (ax, (-ax[1], ax[0], 0.0), (0.0, 0.0, 1.0)), "steel_tread")
        s.box((-L / 2 + sl, Hh - 1.5, -Wd / 2 + 2), (L / 2, Hh, Wd / 2 - 2), "steel_tread")
        for z in (-Wd / 2, Wd / 2 - 3):                                                       # side rails / supports
            s.box((-L / 2 + sl, 0, z), (L / 2, Hh - 1.5, z + 3), p["frame"])
            s.obox((c[0], Hh / 2 + 2.5, z + 1.5), (plen, 5, 3), (ax, (-ax[1], ax[0], 0.0), (0.0, 0.0, 1.0)), p["frame"])
        for z in (-Wd / 2 + 1, Wd / 2 - 4):                                                   # safety edge
            s.box((-L / 2 + sl, Hh, z), (L / 2, Hh + 0.4, z + 3), "floor_yellow")
        shapes_to_stage(stage, path, s, lib)
        return {"top": (L / 2 - (L - sl) / 2, Hh, 0)}


register(Ramp("ramp", "prop", "Drive-on ramp", {
    "length": Param(400.0, "length", 100, 1500, "cm"),
    "width": Param(250.0, "width", 60, 600, "cm"),
    "height": Param(25.0, "top height", 5, 150, "cm"),
    "slope_deg": Param(8.0, "slope", 2, 25, "deg"),
    "frame": Param("powder_yellow", "side finish"),
}))


class PlatformScale(Item):
    def footprint(self, p):
        return p["length"], p["width"]

    def build(self, stage, path, p, lib):
        L, Wd = p["length"], p["width"]
        s = new_shapes()
        s.box((-L / 2 + 15, 0, -Wd / 2), (L / 2 - 15, 8, Wd / 2), "steel_tread")               # weighing platform
        for sx in (-1, 1):                                                                    # entry lips
            c = (sx * (L / 2 - 7.5), 4, 0)
            ang = math.atan2(8, 15) * -sx
            ax = (math.cos(ang), math.sin(ang), 0.0)
            s.obox(c, (17, 1.2, Wd), (ax, (-ax[1], ax[0], 0.0), (0, 0, 1)), "steel_dark")
        s.box((-L / 2 + 15, 8, -Wd / 2), (L / 2 - 15, 8.3, -Wd / 2 + 4), "floor_yellow")
        s.box((-L / 2 + 15, 8, Wd / 2 - 4), (L / 2 - 15, 8.3, Wd / 2), "floor_yellow")
        x, z = -L / 2 + 25, Wd / 2 + 25                                                        # indicator post
        s.box((x - 3, 0, z - 3), (x + 3, 130, z + 3), "powder_grey")
        s.box((x - 18, 120, z - 6), (x + 18, 145, z + 4), "plastic_black")
        s.box((x - 14, 124, z + 4), (x + 14, 141, z + 4.3), "light_emit")
        shapes_to_stage(stage, path, s, lib)
        return {"platform": (0, 8, 0), "display": (x, 132, z + 4)}


register(PlatformScale("platform_scale", "prop", "Platform scale (drive-on)", {
    "length": Param(300.0, "length", 80, 1500, "cm"),
    "width": Param(150.0, "width", 60, 500, "cm"),
}))
