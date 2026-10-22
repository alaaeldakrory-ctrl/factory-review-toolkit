"""Hand tools for the workers: geometry plus how a hand holds them.

Each tool is modelled in its own frame (cm). It defines:
- `grip`: where the dominant hand's palm centre sits, which way the fingers point, the palm normal and the
  finger preset (solver.GRIPS);
- optionally `grip2` for the support hand;
- `work`: the working point / axis (socket on the bolt, hammer face, screwdriver tip).

Animation moves the tool and derives the hand target from the grip, so hand and tool never drift apart.
"""
from dataclasses import dataclass, field

import numpy as np

from ..geom import Shapes
from .solver import HandTarget


@dataclass
class Grip:
    point: tuple
    fingers: tuple
    palm: tuple
    kind: str = "power"
    radius: float = 0.0              # handle radius (cm) the fingers wrap; 0 = use the preset hand shape


@dataclass
class Tool:
    key: str
    title: str
    build: callable
    grip: Grip
    grip2: Grip = None
    work: dict = field(default_factory=dict)

    def shapes(self):
        s = Shapes()
        self.build(s)
        return s

    def hand_target(self, T, side, which="grip", weight=1.0):
        """HandTarget for the hand holding this tool when the tool frame is placed by T (4x4, tool -> worker)."""
        g = self.grip if which == "grip" else self.grip2
        R, t = T[:3, :3], T[:3, 3]
        m = np.array([1.0, 1.0, -1.0]) if side == "l" else np.ones(3)       # grips are written for the right hand
        return HandTarget(side, tuple(R @ (np.asarray(g.point, float) * m) + t),
                          fingers=tuple(R @ (np.asarray(g.fingers, float) * m)),
                          palm=tuple(R @ (np.asarray(g.palm, float) * m)), grip=f"{g.kind}:{g.radius}" if g.radius else g.kind,
                          weight=weight)


def _ratchet(s, length=25.0):
    # head at the origin, socket down (-Y) onto the bolt, handle along +X
    s.lathe((0, 0, 0), (0, 1.6, 0), [(0, 2.1), (0.15, 2.4), (0.85, 2.4), (1, 2.1)], "steel_machined", sides=24)
    s.rbox((0.6, 1.9, 0.0), (1.2, 0.5, 0.8), 0.2, "steel_machined", seg=1)                 # selector lever
    s.lathe((0, 0, 0), (0, -4.0, 0), [(0, 1.15), (0.1, 1.25), (0.9, 1.25), (1, 1.15)], "chrome", sides=20)   # socket
    s.rprism((2.0, 0.8, 0), (length * 0.45, 0.8, 0), 1.6, 1.0, 0.45, "chrome")             # beam
    s.lathe((length * 0.42, 0.8, 0), (length, 0.8, 0), [(0, 0.8), (0.06, 1.25)] +
            [v for k in range(8) for v in ((0.1 + k * 0.11, 1.25), (0.15 + k * 0.11, 1.15))] + [(0.96, 1.2), (1, 0.7)],
            "grip_rubber", sides=16)


def _torque_wrench(s, length=45.0):
    _ratchet(s, 12.0)
    s.lathe((10.0, 0.8, 0), (length * 0.62, 0.8, 0), [(0, 1.0), (1, 1.1)], "chrome", sides=18)       # tube
    s.lathe((length * 0.62, 0.8, 0), (length, 0.8, 0), [(0, 1.4)] +
            [v for k in range(10) for v in ((0.05 + k * 0.09, 1.55), (0.09 + k * 0.09, 1.42))] + [(1, 0.9)],
            "plastic_black", sides=18)                                                        # knurled handle
    s.lathe((length * 0.55, 0.8, 0), (length * 0.62, 0.8, 0), [(0, 1.15), (1, 1.15)], "steel_machined", sides=18)  # scale


def _hammer(s, length=32.0):
    # handle along +X from the butt (x = 0) to the head; head axis along Y, striking face down (-Y)
    s.lathe((0, 0, 0), (length - 2.0, 0, 0), [(0, 1.5), (0.08, 1.7), (0.5, 1.45), (0.8, 1.25), (1, 1.4)],
            "plastic_yellow", sides=16)                                                       # fibreglass handle
    s.lathe((0, 0, 0), (13.0, 0, 0), [(0, 1.6)] + [v for k in range(6) for v in ((0.1 + k * 0.14, 1.85), (0.16 + k * 0.14, 1.72))]
            + [(1, 1.6)], "grip_rubber", sides=16)                                            # grip
    s.lathe((length, -5.5, 0), (length, 5.0, 0), [(0, 1.4), (0.05, 1.6), (0.35, 1.5), (0.65, 1.4), (0.8, 1.0), (1, 0.6)],
            "steel_blued", sides=20)                                                          # head + peen
    s.lathe((length, -5.5, 0), (length, -6.0, 0), [(0, 1.6), (1, 1.45)], "steel_machined", sides=20)   # polished face


def _screwdriver(s, length=24.0):
    # tip at the origin, shaft and handle up +Y
    s.lathe((0, 0, 0), (0, 12.0, 0), [(0, 0.15), (0.03, 0.35), (0.08, 0.35), (1, 0.35)], "chrome", sides=10)
    s.lathe((0, 12.0, 0), (0, length, 0), [(0, 1.0), (0.1, 1.6)] + [v for k in range(6) for v in
            ((0.18 + k * 0.12, 1.7), (0.24 + k * 0.12, 1.55))] + [(0.95, 1.6), (1, 1.2)], "plastic_blue", sides=8)


def _connector(s):
    # a sealed automotive connector half: body along +Z (mating direction), latch on top
    s.rbox((0, 0, 0), (3.2, 2.0, 3.6), 0.4, "plastic_black", seg=2)
    s.rbox((0, 1.2, -0.4), (1.2, 0.5, 2.2), 0.2, "plastic_grey", seg=1)                    # latch
    s.lathe((0, 0, -1.8), (0, 0, -4.0), [(0, 1.1), (1, 0.9)], "rubber_black", sides=12)     # seal boot
    s.sweep([(0, 0, -4.0), (0, -0.5, -9.0), (0.5, -2.0, -14.0)], 0.55, "cable_pvc", sides=8)  # wire bundle


TOOLS = {
    "ratchet_38": Tool("ratchet_38", "3/8 in ratchet", _ratchet,
                       Grip((19.0, 2.6, 0.0), (0.0, 0.0, -1.0), (0.0, -1.0, 0.0), "power", 1.25),
                       Grip((0.0, 3.6, 0.0), (1.0, 0.0, 0.0), (0.0, -1.0, 0.0), "flat"),
                       {"socket": (0.0, -4.0, 0.0), "axis": (0.0, -1.0, 0.0)}),
    "torque_wrench": Tool("torque_wrench", "Click torque wrench", _torque_wrench,
                          Grip((38.0, 2.8, 0.0), (0.0, 0.0, -1.0), (0.0, -1.0, 0.0), "power", 1.5),
                          Grip((0.0, 3.6, 0.0), (1.0, 0.0, 0.0), (0.0, -1.0, 0.0), "flat"),
                          {"socket": (0.0, -4.0, 0.0), "axis": (0.0, -1.0, 0.0)}),
    "hammer": Tool("hammer", "Hammer", _hammer,
                   Grip((6.5, 0.0, 2.6), (0.0, -1.0, 0.0), (0.0, 0.0, -1.0), "power", 1.8),
                   None, {"face": (32.0, -6.0, 0.0), "normal": (0.0, -1.0, 0.0)}),
    "screwdriver": Tool("screwdriver", "Screwdriver", _screwdriver,
                        Grip((0.0, 18.0, 2.4), (0.0, 1.0, 0.0), (0.0, 0.0, -1.0), "power", 1.6),
                        None, {"tip": (0.0, 0.0, 0.0), "axis": (0.0, -1.0, 0.0)}),
    "connector": Tool("connector", "Connector half", _connector,
                      Grip((0.0, 0.2, -2.6), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0), "pinch"),
                      None, {"mate": (0.0, 0.0, 1.8), "axis": (0.0, 0.0, 1.0)}),
}


def place(origin, x_axis, y_axis):
    """4x4 tool -> worker transform from the tool origin and where its X and Y axes should point."""
    x = np.asarray(x_axis, float)
    x = x / np.linalg.norm(x)
    y = np.asarray(y_axis, float)
    y = y - x * np.dot(y, x)
    y = y / np.linalg.norm(y)
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = x, y, np.cross(x, y), origin
    return T


# ---------------------------------------------------------------------------------------------- more hand tools
# Frames: the tool's working point at the origin. Grips in TOOLS are kept for the older generators; the hand-job
# library (jobs.py) places hands with the rules in handkit.py.
def _wrench(s, length=18.0):
    # open end at the origin (jaws open towards -X), shaft along +X, ring end at +length
    for sy in (-1, 1):
        s.rbox((-0.6, 0.0, sy * 1.15), (2.6, 0.6, 0.7), 0.2, "chrome", seg=1)                    # jaws
    s.rbox((1.0, 0.0, 0.0), (2.2, 0.6, 3.0), 0.5, "chrome", seg=1)
    s.rprism((1.5, 0, 0), (length - 1.8, 0, 0), 1.3, 0.55, 0.25, "chrome", up=(0, 0, 1))
    s.lathe((length, -0.35, 0), (length, 0.35, 0), [(0, 1.6), (1, 1.6)], "chrome", sides=20)
    s.lathe((length, -0.36, 0), (length, 0.36, 0), [(0, 0.85), (1, 0.85)], "hole", sides=12)


def _hex_key(s):
    # short arm in the socket along -Y from the origin, long arm along +X at the top
    s.sweep([(0, -2.8, 0), (0, 0.0, 0), (0.6, 0.6, 0), (9.0, 0.6, 0)], 0.32, "steel_blued", sides=6)


def _impact(s):
    # bit tip at the origin, body behind it along -X, pistol handle down from x = -11
    s.lathe((0, 0, 0), (-3.5, 0, 0), [(0, 0.15), (0.2, 0.35), (1, 0.35)], "chrome", sides=6)               # bit
    s.lathe((-3.5, 0, 0), (-7.5, 0, 0), [(0, 1.1), (0.4, 1.6), (1, 1.9)], "steel_machined", sides=16)       # chuck
    s.lathe((-7.5, 0, 0), (-17.0, 0, 0), [(0, 2.6), (0.5, 2.9), (0.9, 2.8), (1, 2.2)], "plastic_yellow", sides=20)
    s.lathe((-14.0, 0, 0), (-17.2, 0, 0), [(0, 2.9), (1, 2.5)], "plastic_black", sides=20)                  # motor cap
    s.rprism((-12.0, -1.0, 0), (-13.5, -13.0, 0), 2.6, 3.4, 1.2, "grip_rubber", up=(1, 0, 0))             # handle
    s.rbox((-10.6, -3.2, 0), (1.2, 2.2, 1.0), 0.4, "plastic_black", seg=1)                                # trigger
    s.rbox((-13.8, -15.0, 0), (8.5, 4.0, 6.0), 1.0, "plastic_black", seg=2)                               # battery
    s.rbox((-13.8, -12.8, 0), (8.7, 0.6, 6.2), 0.2, "plastic_yellow", seg=1)


def _pliers(s, kind="cutter"):
    # jaws at the origin pointing +X, pivot at x = -3, handles along -X (to x = -18), opening a little
    mat = {"cutter": "plastic_blue", "stripper": "plastic_yellow", "crimper": "powder_blue"}[kind]
    s.rbox((-0.8, 0, 0), (3.2, 0.8, 1.8), 0.4, "steel_dark", seg=1)                                       # jaws
    s.lathe((-3.0, -0.6, 0), (-3.0, 0.6, 0), [(0, 0.9), (1, 0.9)], "steel_machined", sides=14)            # pivot
    for sz in (-1, 1):
        s.sweep([(-3.0, 0, sz * 0.4), (-8.0, 0, sz * 1.1), (-17.5, 0, sz * 1.6)], 0.55, "steel_dark", sides=8)
        s.sweep([(-7.5, 0, sz * 1.05), (-17.8, 0, sz * 1.65)], 0.85, mat, sides=10)                     # grips
    if kind == "crimper":
        s.rbox((-1.2, 0, 0), (2.4, 1.4, 2.4), 0.3, "steel_dark", seg=1)


def _plug(s):
    # mains plug: pins along +Z from the face at z = 0, body behind, cable out the back
    for sx in (-0.95, 0.95):
        s.lathe((sx, 0, 0), (sx, 0, 1.9), [(0, 0.24), (1, 0.24)], "steel_machined", sides=10)
    s.rbox((0, 0, -1.8), (4.2, 3.8, 3.6), 1.0, "plastic_black", seg=2)
    s.lathe((0, 0, -3.6), (0, 0, -6.0), [(0, 0.9), (1, 0.55)], "rubber_black", sides=12)
    s.sweep([(0, 0, -6.0), (0, -1.5, -11.0), (0, -6.0, -16.0)], 0.4, "cable_pvc", sides=8)


def _scanner(s):
    # window at the origin facing +X, handle down
    s.rbox((-4.0, 0, 0), (9.0, 4.0, 4.6), 1.6, "plastic_yellow", seg=2)
    s.rbox((0.45, 0, 0), (0.2, 3.0, 3.6), 0.3, "plastic_black", seg=1)
    s.rprism((-6.0, -1.5, 0), (-7.5, -11.0, 0), 2.4, 3.2, 1.1, "plastic_black", up=(1, 0, 0))
    s.rbox((-4.6, -3.4, 0), (1.1, 2.0, 1.0), 0.4, "plastic_black", seg=1)


def _caulk(s):
    # nozzle tip at the origin pointing +X, tube along -X, handle under the rear
    s.lathe((0, 0, 0), (-6.0, 0, 0), [(0, 0.15), (0.6, 0.6), (1, 1.2)], "plastic_grey", sides=12)
    s.lathe((-6.0, 0, 0), (-28.0, 0, 0), [(0, 2.4), (1, 2.4)], "plastic_white" if False else "nylon_white", sides=20)
    s.rbox((-17.0, -2.8, 0), (24.0, 0.6, 3.6), 0.2, "steel_dark", seg=1)                                  # cradle
    s.rprism((-30.0, -1.0, 0), (-31.5, -12.0, 0), 2.2, 3.0, 1.0, "powder_dark", up=(1, 0, 0))             # handle
    s.rprism((-28.0, -2.0, 0), (-27.0, -11.0, 0), 1.2, 2.4, 0.5, "steel_dark", up=(1, 0, 0))             # trigger
    s.cylinder((-29.0, 0, 0), (-40.0, 0, 0), 0.35, "steel_machined", 8)                                  # plunger rod


def _mallet(s, length=30.0):
    _hammer_like = TOOLS.get("hammer")
    s.lathe((0, 0, 0), (length - 2.0, 0, 0), [(0, 1.5), (0.5, 1.35), (1, 1.4)], "wood_top", sides=14)
    s.lathe((length, -5.5, 0), (length, 5.5, 0), [(0, 3.2), (0.08, 3.6), (0.92, 3.6), (1, 3.2)], "rubber_black", sides=22)


def _rag(s):
    s.rbox((0, 0, 0), (12.0, 1.4, 10.0), 0.7, "fabric_navy", seg=2)


def _pen(s):
    # tip at the origin, body along +Y
    s.lathe((0, 0, 0), (0, 13.5, 0), [(0, 0.05), (0.05, 0.4), (0.9, 0.45), (1, 0.35)], "plastic_blue", sides=10)


TOOLS.update({
    "wrench_comb": Tool("wrench_comb", "Combination wrench", _wrench, Grip((14.0, 0.0, 0.0), (0, 0, -1), (0, -1, 0), "power", 0.9)),
    "hex_key": Tool("hex_key", "Hex key", _hex_key, Grip((6.0, 0.6, 0.0), (0, 0, -1), (0, -1, 0), "power", 0.6)),
    "impact_driver": Tool("impact_driver", "Impact driver", _impact, Grip((-12.6, -7.0, 0.0), (1, 0, 0), (0, 0, 1), "power", 1.7)),
    "flush_cutter": Tool("flush_cutter", "Side cutters", lambda s: _pliers(s, "cutter"), Grip((-13.0, 0, 0), (-1, 0, 0), (0, -1, 0), "power", 1.6)),
    "wire_stripper": Tool("wire_stripper", "Wire stripper", lambda s: _pliers(s, "stripper"), Grip((-13.0, 0, 0), (-1, 0, 0), (0, -1, 0), "power", 1.6)),
    "crimper": Tool("crimper", "Crimping pliers", lambda s: _pliers(s, "crimper"), Grip((-13.0, 0, 0), (-1, 0, 0), (0, -1, 0), "power", 1.6)),
    "plug": Tool("plug", "Mains plug", _plug, Grip((0.0, 2.6, -2.0), (0, 0, 1), (0, -1, 0), "pinch")),
    "scanner": Tool("scanner", "Barcode scanner", _scanner, Grip((-6.8, -6.0, 0.0), (1, 0, 0), (0, 0, 1), "power", 1.5)),
    "caulk_gun": Tool("caulk_gun", "Caulking gun", _caulk, Grip((-30.8, -6.0, 0.0), (1, 0, 0), (0, 0, 1), "power", 1.5)),
    "mallet": Tool("mallet", "Rubber mallet", _mallet, Grip((6.5, 0.0, 2.6), (0, -1, 0), (0, 0, -1), "power", 1.5)),
    "rag": Tool("rag", "Rag", _rag, Grip((0, 1.8, 0), (0, 0, 1), (0, -1, 0), "flat")),
    "pen": Tool("pen", "Pen", _pen, Grip((0, 3.0, 0), (0, 1, 0), (1, 0, 0), "tripod")),
})
