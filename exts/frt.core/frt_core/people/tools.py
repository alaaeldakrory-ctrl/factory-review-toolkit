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
                          palm=tuple(R @ (np.asarray(g.palm, float) * m)), grip=g.kind, weight=weight)


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
            ((0.18 + k * 0.12, 1.7), (0.24 + k * 0.12, 1.55))] + [(0.95, 1.6), (1, 1.2)], "plastic_red", sides=8)


def _connector(s):
    # a sealed automotive connector half: body along +Z (mating direction), latch on top
    s.rbox((0, 0, 0), (3.2, 2.0, 3.6), 0.4, "plastic_black", seg=2)
    s.rbox((0, 1.2, -0.4), (1.2, 0.5, 2.2), 0.2, "plastic_grey", seg=1)                    # latch
    s.lathe((0, 0, -1.8), (0, 0, -4.0), [(0, 1.1), (1, 0.9)], "rubber_black", sides=12)     # seal boot
    s.sweep([(0, 0, -4.0), (0, -0.5, -9.0), (0.5, -2.0, -14.0)], 0.55, "cable_pvc", sides=8)  # wire bundle


TOOLS = {
    "ratchet_38": Tool("ratchet_38", "3/8 in ratchet", _ratchet,
                       Grip((19.0, 2.6, 0.0), (0.0, 0.0, -1.0), (0.0, -1.0, 0.0), "power"),
                       Grip((0.0, 3.6, 0.0), (1.0, 0.0, 0.0), (0.0, -1.0, 0.0), "flat"),
                       {"socket": (0.0, -4.0, 0.0), "axis": (0.0, -1.0, 0.0)}),
    "torque_wrench": Tool("torque_wrench", "Click torque wrench", _torque_wrench,
                          Grip((38.0, 2.8, 0.0), (0.0, 0.0, -1.0), (0.0, -1.0, 0.0), "power"),
                          Grip((0.0, 3.6, 0.0), (1.0, 0.0, 0.0), (0.0, -1.0, 0.0), "flat"),
                          {"socket": (0.0, -4.0, 0.0), "axis": (0.0, -1.0, 0.0)}),
    "hammer": Tool("hammer", "Hammer", _hammer,
                   Grip((6.5, 0.0, 2.6), (0.0, -1.0, 0.0), (0.0, 0.0, -1.0), "power"),
                   None, {"face": (32.0, -6.0, 0.0), "normal": (0.0, -1.0, 0.0)}),
    "screwdriver": Tool("screwdriver", "Screwdriver", _screwdriver,
                        Grip((0.0, 18.0, 2.4), (0.0, 1.0, 0.0), (0.0, 0.0, -1.0), "power"),
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
