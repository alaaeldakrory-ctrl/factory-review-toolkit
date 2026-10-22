"""Hand jobs for aerial work platform assembly (scissor lifts, boom lifts, telehandlers): hydraulics, overhead /
kneeling / under-frame work, heavy two-handed and two-person handling, electrical, finishing and QC.

Same conventions as jobs.py (worker at the origin facing +Z, left = +X, Y up, cm; hands placed by the grip rules in
handkit.py; tools carried with a rigid grip). New here: kneeling and squatting stances, a partner worker (two-person
handling), composite parts (hydraulic cylinder, wheel, hose end, joystick).
"""
import math

import numpy as np

from . import handkit as K
from . import jobs as J
from . import motion_carry as C
from . import motion_gen as G
from .bake import PART_BUILDERS
from .motion_gen import Frame, rest_targets
from .solver import HandTarget

X, Y, Z = J.X, J.Y, J.Z
TOP, PZ = J.TOP, J.PZ
wpt, wdir = J.wpt, J.wdir


_POSTURE_REST = {}


def posture_rest(S, st):
    """Hand targets of the arms hanging relaxed in a posture (kneel / squat) - the rest pose to reach from and
    return to (the standing rest targets would sit just below the shoulder when kneeling)."""
    from .kinematics import fk
    key = (id(S.sk), st.drop, st.left, st.right)
    if key in _POSTURE_REST:
        return _POSTURE_REST[key]
    sol = S.solve([], look=None, stance=st)
    R, P = fk(S.sk, sol.pose)
    out = {}
    hl = S.hand_len
    for side in ("l", "r"):
        sx = 1.0 if side == "l" else -1.0
        Rh = R[f"hand_{side}"]
        f = Rh @ np.array([sx, 0.0, 0.0])
        pm = Rh @ np.array([0.0, -1.0, 0.0])
        pt = P[f"hand_{side}"] + f * hl * 0.40 + pm * 1.6
        out[side] = HandTarget(side, tuple(pt), tuple(f), tuple(pm), "relaxed", 0.0)
    _POSTURE_REST[key] = out
    return out


def knee_rest(sk, side="l", st=None, fwd=30.0, y=None):
    """Off hand resting on the knee when kneeling / crouching (palm down, fingers forward over the kneecap)."""
    sx = 1.0 if side == "l" else -1.0
    hw = abs(sk.world["thigh_l"][0] - sk.world["thigh_r"][0]) / 2 + 4.0
    hy = y if y is not None else sk.length("calf_l", "foot_l") + float(sk.world["foot_l"][1]) + 6.0
    return K.brace(sk, side, (sx * hw - sx * 1.0, hy, fwd), normal=(0.0, 1.0, 0.0), toward=(-sx * 0.6, 0.0, 0.8),
                   weight=0.5, grip="relaxed")                     # fingers in over the kneecap


def keyed(duration, drivers=None, tools=None, **kw):
    """handkit.keyed with `drivers`: {name: [(t, 4x4)]} transform tracks that hand callables can read (Td[name]) -
    a part being carried, a wheel, a cylinder - without being drawn as tools."""
    drivers = drivers or {}
    tl = dict(tools or {})
    for k, v in drivers.items():
        tl[k] = ("_driver", v)
    frames = K.keyed(duration, tools=tl, **kw)
    for f in frames:
        for k in drivers:
            f.tools.pop(k, None)
    return frames


# ---------------------------------------------------------------------------------------------- composite parts
def _hydcyl(s, a):
    """Hydraulic cylinder along +X: rod eye at the origin, rod, gland, barrel, base clevis."""
    rod, bar = float(a[0]), float(a[1])
    s.torus((0, 0, 0), (0, 0, 1), 2.6, 1.3, "steel_machined", seg=20, sides=8)
    s.cylinder((2.4, 0, 0), (rod, 0, 0), 1.8, "chrome", sides=20)
    s.cylinder((rod, 0, 0), (rod + 5.0, 0, 0), 4.0, "steel_machined", sides=24)
    s.cylinder((rod + 5.0, 0, 0), (rod + bar, 0, 0), 4.6, "powder_dark", sides=24)
    s.cylinder((rod + 12.0, 4.0, 0), (rod + 12.0, 6.5, 0), 1.1, "steel_machined", sides=12)          # port
    s.rbox((rod + bar + 2.5, 0, 0), (5.0, 7.0, 6.0), 0.6, "powder_dark", seg=2)


def _wheel(s, a):
    """Tyre and rim, axis along Z, centre at the origin."""
    R = float(a[0]) if a else 28.0
    s.torus((0, 0, 0), (0, 0, 1), R - 8.0, 8.0, "rubber_black", seg=36, sides=12, stretch=0.0)
    s.cylinder((0, 0, -5.0), (0, 0, 5.0), R - 14.0, "powder_grey", sides=32)
    s.cylinder((0, 0, 5.0), (0, 0, 6.0), 7.0, "steel_machined", sides=24)


def _hose_end(s, a):
    """Hose end: swivel nut at the origin (axis Z, thread towards +Z), crimped ferrule and hose back along -Z."""
    L = float(a[0]) if a else 35.0
    s.lathe((0, 0, 0), (0, 0, -2.2), [(0, 1.3), (1, 1.3)], "bolt_zinc", sides=6)
    s.lathe((0, 0, -2.2), (0, 0, -7.0), [(0, 1.2), (0.2, 1.5), (1, 1.5)], "steel_machined", sides=16)
    s.sweep([(0, 0, -7.0), (0, -1.5, -7.0 - L * 0.5), (1.0, -6.0, -7.0 - L)], 1.15, "hose_rubber", sides=12)


def _coupler(s, a):
    """Male quick-disconnect half: nipple at the origin pointing +Z, body and hose back along -Z."""
    s.lathe((0, 0, 0), (0, 0, -3.0), [(0, 0.8), (0.3, 1.0), (1, 1.0)], "steel_machined", sides=16)
    s.lathe((0, 0, -3.0), (0, 0, -8.0), [(0, 1.6), (1, 1.6)], "steel_machined", sides=6)
    s.sweep([(0, 0, -8.0), (0, -1.0, -16.0), (0.5, -5.0, -26.0)], 1.0, "hose_rubber", sides=12)


def _joystick(s, a):
    """Joystick: base flange at the origin (down = -Y), boot and stick up +Y, knob on top."""
    s.rbox((0, -3.0, 0), (6.0, 6.0, 6.0), 0.6, "plastic_black", seg=2)
    s.rbox((0, 0.3, 0), (9.0, 0.6, 9.0), 0.3, "plastic_black", seg=1)
    s.lathe((0, 0.6, 0), (0, 4.0, 0), [(0, 2.6), (0.5, 2.0), (1, 1.0)], "rubber_black", sides=16)
    s.cylinder((0, 4.0, 0), (0, 12.0, 0), 0.6, "steel_machined", sides=10)
    s.lathe((0, 12.0, 0), (0, 19.0, 0), [(0, 1.6), (0.5, 2.1), (1, 1.8)], "plastic_black", sides=16)


def _sheet(s, a):
    """Thin sheet hanging down from its top edge (origin): "sheet:W:H:look"."""
    w, h = float(a[0]), float(a[1])
    s.rbox((0, -h / 2, 0), (w, h, 0.15), 0.05, a[2] if len(a) > 2 else "label_vinyl", seg=1)


for _k, _f in (("sheet", _sheet), ("hydcyl", _hydcyl), ("wheel", _wheel), ("hose_end", _hose_end), ("coupler", _coupler),
               ("joystick", _joystick)):
    PART_BUILDERS[_k] = _f


# ---------------------------------------------------------------------------------------------- fixtures
def fx(*parts, kind="bench"):
    def f(s):
        for p in parts:
            p(s)
    f.kind = kind
    return f


def legs(s, x0, x1, z0, z1, top):
    for x in (x0, x1):
        for z in (z0, z1):
            s.rprism((x, 0.0, z), (x, top, z), 4.0, 4.0, 0.6, "powder_dark", up=(1, 0, 0))


def rail(s, y, z, x0=-60.0, x1=60.0, h=10.0, d=5.0, look="powder_grey"):
    """A frame rail (rectangular tube) along X, centred at height y, front face at z - d/2."""
    s.rbox(((x0 + x1) / 2, y, z), (x1 - x0, h, d), 0.5, look, seg=2)


def manifold(s, c, size=(14.0, 10.0, 8.0)):
    J.block(s, c, size, "cast_aluminium")


def b_manifold_front(s):
    J.bench(s)
    manifold(s, (-4.0, TOP + 6.0, 40.0))
    s.lathe((-4.0, TOP + 6.0, 36.0), (-4.0, TOP + 6.0, 34.6), [(0, 1.0), (1, 1.0)], "steel_machined", sides=6)


def b_two_wrench(s):
    J.bench(s)
    J.block(s, (-4.0, TOP + 6.0, 46.0), (16.0, 12.0, 14.0), "powder_dark")                         # riser
    manifold(s, (-4.0, TOP + 18.0, 44.0), (16.0, 12.0, 10.0))
    s.lathe((-4.0, TOP + 18.0, 39.0), (-4.0, TOP + 18.0, 36.5), [(0, 1.3), (1, 1.3)], "steel_machined", sides=6)
    s.lathe((-4.0, TOP + 18.0, 36.5), (-4.0, TOP + 18.0, 34.0), [(0, 1.25), (1, 1.25)], "bolt_zinc", sides=6)
    s.sweep([(-4.0, TOP + 18.0, 34.0), (-4.0, TOP + 16.0, 26.0), (-12.0, TOP + 4.0, 16.0)], 1.15, "hose_rubber", sides=12)


def b_hose_rail(s):
    legs(s, -55.0, 55.0, 46.0, 54.0, 92.0)
    rail(s, 97.0, 50.0)
    s.sweep([(55.0, 99.0, 46.6), (-55.0, 99.0, 46.6)], 1.1, "hose_rubber", sides=12)
    s.rbox((-34.0, 102.5, 50.0), (10.0, 1.0, 6.0), 0.3, "plastic_blue", seg=1)                  # clip tray


def b_coupler(s):
    J.panel(s)
    s.rbox((-4.0, 105.0, PZ - 2.0), (8.0, 8.0, 4.0), 0.6, "cast_aluminium", seg=2)
    s.lathe((-4.0, 105.0, PZ - 4.0), (-4.0, 105.0, PZ - 9.0), [(0, 1.9), (0.4, 2.1), (1, 2.0)], "steel_machined",
            sides=20)                                                                                  # female half


def b_cap_ports(s):
    J.bench(s)
    manifold(s, (-6.0, TOP + 4.0, 46.0), (30.0, 8.0, 10.0))
    for x in (-18.0, -10.0, -2.0, 6.0):
        s.lathe((x, TOP + 8.0, 46.0), (x, TOP + 9.4, 46.0), [(0, 0.9), (1, 0.9)], "steel_machined", sides=6)


BEAM_Y = 196.0


def b_overhead(s):
    for x in (-70.0, 70.0):
        s.rprism((x, 0.0, 40.0), (x, BEAM_Y + 7.0, 40.0), 8.0, 8.0, 0.8, "powder_dark", up=(1, 0, 0))
    s.rbox((0.0, BEAM_Y, 40.0), (150.0, 14.0, 16.0), 1.0, "powder_yellow", seg=2)                  # boom section
    s.lathe((-4.0, BEAM_Y - 7.0, 40.0), (-4.0, BEAM_Y - 9.0, 40.0), [(0, 1.2), (1, 1.2)], "bolt_zinc", sides=6)


def b_kneel_plate(s):
    s.rbox((0.0, 40.0, 44.0), (120.0, 80.0, 2.0), 0.6, "powder_grey", seg=2)                       # chassis side
    s.rbox((0.0, 2.0, 50.0), (120.0, 4.0, 14.0), 0.4, "powder_dark", seg=1)
    s.rbox((-6.0, 46.0, 42.6), (10.0, 10.0, 1.0), 0.4, "powder_dark", seg=1)


def b_under(s):
    for x in (-60.0, 60.0):
        for z in (20.0, 80.0):
            s.rprism((x, 0.0, z), (x, 72.0, z), 6.0, 6.0, 0.6, "powder_dark", up=(1, 0, 0))
    s.rbox((0.0, 78.0, 50.0), (130.0, 12.0, 66.0), 0.8, "powder_grey", seg=2)                     # frame deck
    s.rbox((-4.0, 70.5, 31.0), (5.0, 3.0, 3.5), 0.5, "plastic_black", seg=1)                       # receptacle
    s.sweep([(-4.0, 72.0, 31.0), (-12.0, 71.0, 40.0), (-40.0, 71.0, 60.0)], 0.6, "cable_pvc", sides=8)


def b_lean_in(s):
    J.bench(s, depth=60.0)
    J.block(s, (-4.0, TOP + 2.5, 72.0), (16.0, 5.0, 12.0))
    J.bolt_up(s, (-4.0, TOP + 5.0, 72.0))
    s.rbox((0.0, TOP + 40.0, 66.0), (110.0, 3.0, 38.0), 0.5, "powder_grey", seg=2)                 # cover over the back
    for x in (-54.0, 54.0):
        s.rbox((x, TOP + 20.0, 66.0), (3.0, 40.0, 38.0), 0.5, "powder_grey", seg=2)


CLEVIS = np.array([-46.0, 112.0, 50.0])


def b_clevis(s):
    s.rbox((-70.0, 56.0, 50.0), (16.0, 112.0, 16.0), 0.8, "powder_dark", seg=2)
    for dz in (-4.0, 4.0):
        s.rbox((CLEVIS[0] - 8.0, CLEVIS[1], CLEVIS[2] + dz), (16.0, 10.0, 2.0), 0.4, "powder_dark", seg=1)
    s.rbox((20.0, 89.5, 50.0), (110.0, 4.0, 30.0), 0.5, "powder_grey", seg=1)                    # cradle stand top
    legs(s, -30.0, 70.0, 38.0, 62.0, 87.5)


def b_pivot(s):
    s.rbox((0.0, 112.0, 56.0), (30.0, 30.0, 12.0), 1.0, "powder_yellow", seg=2)                    # boom pivot boss
    s.rprism((0.0, 0.0, 60.0), (0.0, 97.0, 60.0), 12.0, 12.0, 1.0, "powder_dark", up=(1, 0, 0))
    s.lathe((0.0, 112.0, 46.0), (0.0, 112.0, 50.0), [(0, 2.2), (1, 2.2)], "steel_machined", sides=20)  # pin head


HUB = np.array([0.0, 62.0, 70.0])
LUG_HUB = HUB + np.array([56.0, 8.0, -4.0])


def b_hub(s, c=None):
    c = HUB if c is None else c
    _hub_at(s, c)


def b_lug_hub(s):
    _hub_at(s, LUG_HUB, cart=False)


def _hub_at(s, c, cart=True):
    x, y, z = c
    s.rbox((x, y / 2 + 8.0, z + 16.0), (40.0, y + 16.0, 20.0), 1.0, "powder_dark", seg=2)          # axle housing
    s.cylinder((x, y, z + 6.0), (x, y, z), 9.0, "steel_machined", sides=28)
    for k in range(6):
        a = 2 * math.pi * k / 6
        s.cylinder((x + 10.0 * math.cos(a), y + 10.0 * math.sin(a), z),
                   (x + 10.0 * math.cos(a), y + 10.0 * math.sin(a), z - 6.0), 0.7, "bolt_zinc", sides=8)
    if cart:
        s.rbox((-10.0, 14.0, 44.0), (50.0, 28.0, 30.0), 0.6, "powder_blue", seg=1)                 # wheel cart


def b_hub_mounted(s):
    b_lug_hub(s)


def b_hoist(s):
    legs(s, -40.0, 40.0, 42.0, 66.0, 86.0)
    s.rbox((0.0, 88.0, 54.0), (90.0, 4.0, 34.0), 0.5, "powder_grey", seg=2)


def b_circlip(s):
    J.bench(s)
    J.block(s, (-4.0, 104.0 - 10.0, 46.0), (14.0, 4.0, 14.0))
    s.cylinder((-4.0, 104.0, 46.0), (-4.0, 104.0, 38.0), 1.6, "steel_machined", sides=20)        # pin end
    s.rbox((-4.0, 99.0, 46.0), (6.0, 10.0, 6.0), 0.4, "powder_dark", seg=1)


def b_ctrl(s):
    J.panel(s)
    s.rbox((-4.0, 96.0, PZ - 4.0), (34.0, 2.0, 8.0), 0.3, "powder_dark", seg=1)                    # ledge


def b_deutsch(s):
    J.panel(s)
    s.rbox((-4.0, 96.0, PZ - 1.5), (5.0, 3.5, 3.0), 0.5, "plastic_grey", seg=1)
    s.sweep([(-4.0, 96.0, PZ - 3.0), (-12.0, 96.0, PZ - 4.0), (-30.0, 104.0, PZ - 1.0)], 0.6, "cable_pvc", sides=8)


def b_console(s):
    legs(s, -30.0, 30.0, 28.0, 68.0, 110.0)
    s.rbox((0.0, 112.0, 48.0), (64.0, 4.0, 44.0), 0.5, "powder_grey", seg=2)
    s.obox((-4.0, 114.05, 45.0), (5.0, 0.1, 5.0), ((1, 0, 0), (0, 1, 0), (0, 0, 1)), "hole")


def b_battery(s):
    J.bench(s)
    s.rbox((-4.0, TOP + 1.0, 46.0), (38.0, 2.0, 26.0), 0.4, "powder_dark", seg=1)                  # battery tray
    s.rbox((26.0, 33.5, 44.0), (40.0, 67.0, 30.0), 0.6, "powder_blue", seg=2)                       # cart


def b_decal(s):
    s.rbox((0.0, 110.0, 49.0), (110.0, 80.0, 2.0), 0.6, "powder_yellow", seg=2)                    # panel
    s.rprism((0.0, 0.0, 52.0), (0.0, 80.0, 52.0), 10.0, 10.0, 0.6, "powder_dark", up=(1, 0, 0))
    s.rbox((-2.0, 88.0, 42.0), (50.0, 1.0, 12.0), 0.3, "powder_dark", seg=1)                      # shelf
    for x in (-24.0, 20.0):
        s.rbox((x, 88.0, 47.0), (1.5, 1.0, 2.0), 0.2, "powder_dark", seg=1)


def b_stripe(s):
    J.bench(s)
    J.block(s, (-6.0, TOP + 1.5, 44.0), (30.0, 3.0, 14.0), "powder_dark")
    for x in (-16.0, -6.0, 4.0):
        J.bolt_up(s, (x, TOP + 3.0, 44.0))


def b_grease(s):
    J.bench(s)
    J.block(s, (-4.0, TOP + 4.5, 44.0), (14.0, 9.0, 14.0), "powder_dark")
    s.lathe((-4.0, TOP + 9.0, 44.0), (-4.0, TOP + 10.4, 44.0), [(0, 0.5), (0.6, 0.5), (0.7, 0.7), (1, 0.5)],
            "brass", sides=10)                                                                          # grease nipple


def b_inspect(s):
    for x in (-60.0, 60.0):
        s.rprism((x, 0.0, 55.0), (x, 74.0, 55.0), 6.0, 6.0, 0.6, "powder_dark", up=(1, 0, 0))
    s.rbox((0.0, 80.0, 55.0), (130.0, 12.0, 14.0), 0.8, "powder_grey", seg=2)
    s.sweep([(-30.0, 73.8, 49.0), (30.0, 73.8, 49.0)], 0.5, "steel_dark", sides=8)                  # weld seam


def b_guardrail(s):
    for x in (-50.0, 50.0):
        s.cylinder((x, 0.0, 46.0), (x, 110.0, 46.0), 2.1, "galvanised", sides=16)
    s.cylinder((-56.0, 110.0, 46.0), (56.0, 110.0, 46.0), 2.1, "galvanised", sides=16)
    s.cylinder((-50.0, 60.0, 46.0), (50.0, 60.0, 46.0), 2.1, "galvanised", sides=16)
    s.rbox((0.0, 2.0, 46.0), (120.0, 4.0, 40.0), 0.4, "steel_tread", seg=1)
    s.rbox((-50.0, 96.0, 43.0), (6.0, 6.0, 4.0), 0.5, "galvanised", seg=1)                         # pin socket


# ---------------------------------------------------------------------------------------------- hydraulics
def j_fitting_start(S, sk):
    """Hand-start a hose fitting onto a manifold port: the off hand holds the hose and presents the end, the
    fingers spin the swivel nut on."""
    R, _ = rest_targets(S)
    P = np.array([-4.0, TOP + 6.0, 34.6])                          # port face (thread along +Z)
    spec = "hose_end:30"

    def T_nut(back, turn):
        T = K.T_at(P - Z * back, X, Y)
        return K.rot_about(T, Z, turn, P)
    lh = lambda Td: K.power(sk, "l", wpt(Td["p"], (0.5, -1.6, -13.0)), wdir(Td["p"], (0.0, 0.2, 1.0)),
                            grip="power:1.2", weight=0.8)
    pk = [(0.0, T_nut(14.0, 0.0)), (1.2, T_nut(14.0, 0.0)), (2.0, T_nut(1.0, 0.0))]
    t, turn = 2.0, 0.0
    rk = [(0.0, R["r"])]
    grab = lambda Td: K.pinch(sk, "r", wpt(Td["p"], (-1.9, 0.0, -1.0)), approach=(-0.1, 0.0, 1.0), palm=(1.0, 0.0, 0.0),
                              grip="tripod")
    rk += [(t - 0.3, grab)]
    for k in range(4):                                             # spin: twist, let go, regrip
        turn -= 55.0
        pk += [(t + 0.45, T_nut(1.0 - 0.25 * (k + 1), turn)), (t + 0.7, T_nut(1.0 - 0.25 * (k + 1), turn))]
        rk += [(t + 0.45, grab), (t + 0.7, grab)]
        t += 0.7
    rk += [(t + 0.9, R["r"])]
    return keyed(t + 1.0, drivers={"p": pk}, hands={"r": rk, "l": [(0.0, R["l"]), (1.1, lh), (t, lh), (t + 0.9, R["l"])]},
                 parts={"part:hose": (spec, pk)}, look=tuple(P), label="start fitting")


def j_two_wrench(S, sk):
    """Tighten a hydraulic fitting with two wrenches: a backup wrench holds the adapter, the other turns the nut."""
    R, _ = rest_targets(S)
    Pn = np.array([-4.0, TOP + 18.0, 35.2])                        # swivel nut (manifold on a riser)
    Pb = np.array([-4.0, TOP + 18.0, 37.8])                        # adapter hex (backup)

    def T_w(P, d):
        T = K.T_at(np.zeros(3), K.unit(d), -Z)
        T[:3, 3] = P - wdir(T, (18.0, 0.0, 0.0))
        return T

    def d_r(th):
        a = math.radians(-165.0 + th)                                 # out to the right, swung down
        return (math.cos(a), math.sin(a), 0.0)
    rh = lambda Td: K.power(sk, "r", wpt(Td["t"], (5.0, 0.0, 0.0)), wdir(Td["t"], X), grip="power:0.9")
    lh = lambda Td: K.power(sk, "l", wpt(Td["t2"], (5.0, 0.0, 0.0)), wdir(Td["t2"], X), grip="power:0.9")
    tk = [(0.0, None), (1.1, T_w(Pn, d_r(-15.0)))]
    t = 1.1
    for _ in range(3):
        tk += [(t + 0.55, T_w(Pn, d_r(20.0))), (t + 0.9, T_w(Pn, d_r(-15.0)))]
        t += 0.9
    tk, hk = J.tool_track(sk, rh, tk + [(t + 1.0, None)])
    Tb = T_w(Pb, (0.55, 0.85, 0.0))                                # backup handle up to the left
    tk2, hk2 = J.tool_track(sk, lh, [(0.0, None), (1.0, Tb), (t, Tb), (t + 1.0, None)], key="t2")
    return K.keyed(t + 1.1, hands={"r": hk, "l": hk2}, tools={"t": ("wrench_comb", tk), "t2": ("wrench_comb", tk2)},
                   look=tuple(Pn), label="two wrenches")


def j_hose_clip(S, sk):
    """Route a hose along a frame rail: the off hand holds the hose in place, the other takes a P-clip from the
    tray, wraps it over the hose and presses it onto the rail."""
    R, _ = rest_targets(S)
    tray = np.array([-34.0, 103.4, 50.0])
    spots = [np.array([4.0, 99.0, 46.0]), np.array([-16.0, 99.0, 46.0])]
    spec = "box:3x4x1.6:plastic_black"
    hold = K.power(sk, "l", np.array([20.0, 99.0, 45.4]), -X, grip="power:1.2", weight=0.7)
    rk, lk = [(0.0, R["r"])], [(0.0, R["l"]), (0.8, hold)]
    parts = {}
    t = 0.6
    for k, sp in enumerate(spots):
        pick = K.pinch(sk, "r", tray + Y * 1.0 - X * k * 3.0, approach=(0.2, -0.8, 0.6), grip="pinch")
        over = K.pinch(sk, "r", sp - Z * 3.0 + Y * 2.0, approach=(0.0, -0.3, 1.0), palm=(0, -1, 0), grip="pinch")
        press = K.pinch(sk, "r", sp - Z * 0.6 + Y * 2.0, approach=(0.0, -0.3, 1.0), palm=(0, -1, 0), grip="point")
        rk += [(t + 0.6, pick), (t + 0.75, pick), (t + 1.45, over), (t + 1.7, press), (t + 1.85, press), (t + 2.1, over)]
        T_tray = K.T_at(tray - X * k * 3.0 + Y * 0.6, X, Y)
        T_sp = K.T_at(sp + Y * 1.4 - Z * 1.2, X, Y)
        parts[f"part:clip{k}"] = (spec, [(0.0, T_tray), (t + 0.75, T_tray), (t + 1.45, K.T_at(sp - Z * 3.2 + Y * 1.4, X, Y)),
                                         (t + 1.7, T_sp)])
        t += 2.1
    rk += [(t + 0.8, R["r"])]
    lk += [(t, hold), (t + 0.8, R["l"])]
    return K.keyed(t + 0.9, hands={"r": rk, "l": lk}, parts=parts, look=[(0.0, tray), (1.2, spots[0]), (t - 1.5, spots[-1])],
                   label="clip hose")


def j_coupler(S, sk):
    """Connect a quick-disconnect coupler: push the male half in, turn the locking sleeve; the off hand steadies
    the hose."""
    R, _ = rest_targets(S)
    P = np.array([-4.0, 105.0, PZ - 9.0])                          # female mouth (axis -Z towards the worker)

    def T_at(back, turn=0.0):
        T = K.T_at(P - Z * back, X, Y)
        return K.rot_about(T, Z, turn, P)
    rh = lambda Td: K.power(sk, "r", wpt(Td["t"], (0.0, 0.0, -5.5)), wdir(Td["t"], (0.0, 0.0, 1.0)), grip="power:1.6",
                            lean=(0.0, -0.6, 1.0))
    lh = lambda Td: K.power(sk, "l", wpt(Td["t"], (0.3, -1.4, -15.0)), wdir(Td["t"], (0.0, 0.15, 1.0)),
                            grip="power:1.0", weight=0.7)
    T0 = K.T_at(np.array([-26.0, 90.0, 16.0]), X, Y)
    tk = [(0.0, T0), (1.1, T_at(5.0)), (1.6, T_at(0.6)), (1.8, T_at(-0.3)), (1.95, T_at(0.0)), (2.5, T_at(0.0, -25.0)),
          (3.6, T_at(0.0, -25.0))]
    hand = [(0.0, rh), (2.6, rh), (3.1, R["r"]), (3.6, R["r"])]
    return keyed(3.7, drivers={"t": tk}, hands={"r": hand, "l": [(0.0, R["l"]), (0.9, lh), (2.6, lh), (3.4, R["l"])]},
                 parts={"part:coupler": ("coupler", tk)}, look=tuple(P), label="coupler")


def j_cap_ports(S, sk):
    """Cap open ports: the off hand holds the caps, the other takes one at a time and presses it on."""
    R, _ = rest_targets(S)
    ports = [np.array([x, TOP + 9.4, 46.0]) for x in (6.0, -2.0, -10.0, -18.0)]
    palm_c = np.array([18.0, TOP + 12.0, 34.0])                     # caps cupped in the left palm
    cup = K.brace(sk, "l", palm_c, normal=(0.0, -1.0, 0.0), toward=(-0.3, 0.0, 1.0), weight=0.6, grip="relaxed")
    spec = "box:2.2x1.6x2.2:plastic_yellow"
    rk, parts = [(0.0, R["r"])], {}
    t = 0.9
    for k, pt in enumerate(ports):
        cap0 = palm_c + np.array([-1.5 + k * 1.0, 3.0, 1.0 - k * 1.2])
        take = K.pinch(sk, "r", cap0 + Y * 1.0, approach=(0.6, -0.6, 0.5), grip="pinch")
        over = K.pinch(sk, "r", pt + Y * 3.5, approach=(0.1, -0.85, 0.5), grip="pinch")
        on = K.pinch(sk, "r", pt + Y * 1.6, approach=(0.1, -0.85, 0.5), grip="pinch")
        rk += [(t + 0.45, take), (t + 0.55, take), (t + 1.05, over), (t + 1.25, on), (t + 1.35, on), (t + 1.6, over)]
        parts[f"part:cap{k}"] = (spec, [(0.0, K.T_at(cap0, X, Y)), (t + 0.55, K.T_at(cap0, X, Y)),
                                        (t + 1.05, K.T_at(pt + Y * 2.4, X, Y)), (t + 1.25, K.T_at(pt + Y * 0.5, X, Y))])
        t += 1.6
    rk += [(t + 0.8, R["r"])]
    return K.keyed(t + 0.9, hands={"r": rk, "l": [(0.0, R["l"]), (0.8, cup), (t, cup), (t + 0.8, R["l"])]},
                   parts=parts, look=[(0.0, palm_c), (1.2, ports[0]), (t - 1.0, ports[-1])], label="cap ports")


# ---------------------------------------------------------------------------------------------- overhead / low
def j_overhead(S, sk):
    """Ratchet a bolt under a boom section above the head: socket up, the off hand on the boom."""
    R, _ = rest_targets(S)
    B = np.array([-4.0, BEAM_Y - 9.0, 40.0])                        # bolt head (underside)
    d0 = K.unit((-1.0, 0.0, -0.4))

    def T_th(th):
        c, s_ = math.cos(math.radians(th)), math.sin(math.radians(th))
        d = d0 * c + np.cross(-Y, d0) * s_
        return K.T_at(B - Y * 4.0, d, -Y)                            # socket (tool -Y) up onto the bolt
    rh = lambda Td: K.power(sk, "r", wpt(Td["t"], (19.0, 1.0, 0.0)), -wdir(Td["t"], X), grip="power:1.25")
    br = K.power(sk, "l", np.array([22.0, BEAM_Y - 6.0, 32.6]), X, grip="power:1.6", lean=(0.0, 1.0, 0.2),
                 weight=0.5)                                       # holding the boom's lower front edge
    tk = [(0.0, None), (1.5, T_th(-20.0))]
    t = 1.5
    for _ in range(4):
        tk += [(t + 0.5, T_th(20.0)), (t + 0.85, T_th(-20.0))]
        t += 0.85
    tk, hk = J.tool_track(sk, rh, tk + [(t + 1.1, None)])
    mid = K.power(sk, "l", np.array([24.0, 150.0, 30.0]), X, grip="power:2.2", lean=(0.0, 1.0, 0.4), weight=0.3)
    return K.keyed(t + 1.2, hands={"r": hk, "l": [(0.0, R["l"]), (1.0, mid), (1.7, br), (t, br), (t + 0.5, mid),
                                                  (t + 1.1, R["l"])]},
                   tools={"t": ("ratchet_38", tk)}, look=tuple(B), label="overhead ratchet")


def j_kneel_drive(S, sk):
    """Kneel on one knee and drive a bolt into the chassis side low down with an impact driver."""
    st = K.kneel_stance(sk, knee_down="r")
    R = posture_rest(S, st)
    P = np.array([-6.0, 46.0, 52.0])

    def T_at(push=0.0):
        return K.T_at(P - Z * (0.0 - push), Z, Y)

    def rh(pull):
        return lambda Td: K.power(sk, "r", wpt(Td["t"], (-12.8, -7.0, 0.0)), wdir(Td["t"], Y),
                                  grip=f"trigger:1.7:{pull}", lean=wdir(Td["t"], X))
    br = knee_rest(sk, "l", st, fwd=st.left[2] - 4.0, y=sk.length("calf_l", "foot_l") + 9.0)
    T0 = K.T_at(np.array([-26.0, 40.0, 14.0]), (0.0, -0.2, 1.0), Y)
    tk = [(0.0, T0), (1.1, T_at(-3.0)), (1.5, T_at(0.0))]
    t = 1.5
    for k in range(8):
        tk += [(t + 0.18, T_at(0.15 * (k + 1) / 8))]
        t += 0.18
    tk += [(t + 0.4, T_at(-3.0)), (t + 1.4, T0)]
    hand = [(0.0, rh(0.0)), (1.5, rh(0.0)), (1.65, rh(1.0)), (t, rh(1.0)), (t + 0.2, rh(0.0))]
    return K.keyed(t + 1.5, hands={"r": hand, "l": [(0.0, R["l"]), (1.0, br), (t + 0.3, br), (t + 1.4, R["l"])]},
                   tools={"t": ("impact_driver", tk)}, look=tuple(P), label="kneel drive", stance=st)


def j_squat_connect(S, sk):
    """Squat and reach under the frame to mate a connector upwards into its receptacle."""
    st = K.kneel_stance(sk, knee_down="r", forward=30.0)
    st.hinge = (-2.0, 16.0)                                        # kneeling upright: the hips stay over the knee
    R = posture_rest(S, st)
    P = np.array([-4.0, 69.0, 31.0])                                # receptacle mouth (mating +Y)

    def T_at(back):
        T = K.T_at(np.zeros(3), X, -Z)                               # tool Z (mating) -> world +Y
        T[:3, 3] = P - Y * (back + 1.8)
        return T
    rh = lambda Td: K.pinch(sk, "r", wpt(Td["t"], (0.0, 1.2, -1.2)), approach=(0.0, 0.7, 0.7), palm=(1.0, 0.0, 0.0),
                            grip="pinch")
    lh = lambda Td: K.power(sk, "l", wpt(Td["t"], (0.3, -1.4, -11.0)), wdir(Td["t"], (0.0, 0.25, 0.97)),
                            grip="power:0.7", weight=0.8)
    T0 = T_at(14.0)
    tk = [(0.0, T0), (1.0, T0), (1.5, T_at(3.0)), (1.9, T_at(-0.2)), (2.0, T_at(0.0)), (2.5, T_at(0.0)),
          (2.65, T_at(0.6)), (2.8, T_at(0.0)), (3.8, T_at(0.0))]
    hand = [(0.0, R["r"]), (0.9, rh), (2.8, rh), (3.3, R["r"])]
    return K.keyed(3.9, hands={"r": hand, "l": [(0.0, R["l"]), (0.9, lh), (2.9, lh), (3.5, R["l"])]},
                   tools={"t": ("connector", tk)}, look=tuple(P), label="under-frame connector", stance=st)


def j_lean_in(S, sk):
    """Lean in under a cover to ratchet a bolt at the back of the chassis (a long forward reach)."""
    return J._socket_job(S, sk, "ratchet_38", (-4.0, TOP + 5.0, 72.0), 3, 40.0, 0.85)


# ---------------------------------------------------------------------------------------------- heavy
CYL_SPEC = "hydcyl:34:62"


def _cyl_track():
    """Cylinder transform keys (station frame): lifted off the cradle, aligned in front of the clevis, slid in."""
    def T_c(dx, dy, dz, wig=0.0):
        T = K.T_at(CLEVIS + np.array([dx, dy, dz]), X, Y)
        return K.rot_about(T, Y, wig, CLEVIS + np.array([dx, dy, dz]))
    return [(0.0, T_c(12.0, -16.0, 0.0)), (1.2, T_c(12.0, -16.0, 0.0)), (2.8, T_c(12.0, 0.0, -2.0)),
            (3.6, T_c(4.0, 0.0, 0.0, 2.0)), (4.0, T_c(1.0, 0.0, 0.0, -1.5)), (4.4, T_c(0.0, 0.0, 0.0)),
            (5.6, T_c(0.0, 0.0, 0.0))]


def _cyl_hands(sk, Td, off, xs, sides=("r", "l")):
    """Hands under the cylinder at local x positions (palms up, fingers wrapping), in a worker frame offset by
    `off` (the worker's place in the station)."""
    T = Td["c"]
    out = []
    for side, x in zip(sides, xs):
        p = wpt(T, (x, -3.0, 0.0)) - off
        out.append(K.power(sk, side, p, wdir(T, X) if side == "l" else -wdir(T, X), grip="power:4.0",
                           lean=(0.0, -0.3, 1.0), weight=0.9))
    return out


def j_cylinder(S, sk):
    """Two-person: lift a hydraulic cylinder off the cradle and align its rod eye into the clevis (this worker on
    the rod end, the partner on the barrel)."""
    R, _ = rest_targets(S)
    keys = _cyl_track()
    off = np.zeros(3)
    hands = {"r": [(0.0, R["r"])], "l": [(0.0, R["l"])]}
    for side, x in (("r", 10.0), ("l", 30.0)):
        f = (lambda side_, x_: (lambda Td: _cyl_hands(sk, Td, off, [x_], (side_,))[0]))(side, x)
        hands[side] += [(1.1, f), (4.8, f), (5.5, R[side])]
    return keyed(5.7, drivers={"c": keys}, hands=hands, parts={"part:cyl": (CYL_SPEC, keys)},
                 look=[(0.0, CLEVIS + np.array([20.0, -16.0, 0.0])), (2.6, CLEVIS)], label="align cylinder")


def p_cylinder(S, sk, place):
    """The partner: on the barrel end."""
    R, _ = rest_targets(S)
    keys = _cyl_track()
    off = np.array([place[0], 0.0, place[1]])
    hands = {"r": [(0.0, R["r"])], "l": [(0.0, R["l"])]}
    for side, x in (("r", 70.0), ("l", 88.0)):
        f = (lambda side_, x_: (lambda Td: _cyl_hands(sk, Td, off, [x_], (side_,))[0]))(side, x)
        hands[side] += [(1.1, f), (4.8, f), (5.5, R[side])]
    return keyed(5.7, drivers={"c": keys}, hands=hands,
                 look=[(0.0, CLEVIS + np.array([80.0, -16.0, 0.0]) - off), (2.6, CLEVIS + np.array([80.0, 0, 0]) - off)],
                 label="align cylinder (partner)")


def j_pivot_pin(S, sk):
    """Drive a boom pivot pin home with a hammer: horizontal strikes (handle across in front, the head swung back
    about the wrist and driven forward onto the pin), the off hand steadying the boss."""
    R, _ = rest_targets(S)
    Pin = np.array([0.0, 112.0, 46.0])                              # pin head face (struck towards +Z)
    d_h = K.unit(np.array([1.0, 0.0, 0.25]))                        # handle: from the hand across to the head

    def T_s(lift, seat=0.0):
        T = K.T_at(np.zeros(3), d_h, -Z)                            # head axis (tool Y) along -Z: face (tool -Y) to +Z
        T[:3, 3] = Pin + Z * seat - wdir(T, (32.0, -6.0, 0.0))
        g = wpt(T, (6.5, 0.0, 0.0))
        return K.rot_about(T, Y, lift, g)                           # lift > 0 swings the head back towards the worker
    rh = lambda Td: K.power(sk, "r", wpt(Td["t"], (6.5, 0.0, 0.0)), wdir(Td["t"], X), grip="power:1.6",
                            lean=(0.0, -1.0, 0.0))
    tk = [(0.0, None), (1.5, T_s(40.0))]
    t = 1.5
    for k in range(4):
        sd = 0.4 * (k + 1)
        tk += [(t + 0.28, T_s(0.0, sd)), (t + 0.36, T_s(5.0, sd)), (t + 0.75, T_s(40.0, sd))]
        t += 0.75
    tk, hk = J.tool_track(sk, rh, tk + [(t + 1.1, None)])
    br = K.brace(sk, "l", (12.0, 120.0, 50.0), normal=(0.0, 0.0, -1.0), toward=(0.6, 0.8, 0.0), weight=0.4)
    return K.keyed(t + 1.2, hands={"r": hk, "l": [(0.0, R["l"]), (1.0, br), (t, br), (t + 1.0, R["l"])]},
                   tools={"t": ("hammer", tk)}, look=tuple(Pin), label="pivot pin")


def j_wheel(S, sk):
    """Lift a wheel off the cart onto the hub studs (two hands on the tyre, legs bent), then push it home."""
    st = K.squat_stance(sk, depth=0.4, width=12.0, heels_up=10.0)
    st.hinge = (-2.0, 25.0)
    R = posture_rest(S, st)
    c0 = np.array([-10.0, 56.0, 44.0])

    def T_w(c):
        return K.T_at(np.asarray(c, float), X, Y)
    keys = [(0.0, T_w(c0)), (1.2, T_w(c0)), (2.4, T_w(HUB + np.array([0.0, 0.5, -10.0]))),
            (3.0, T_w(HUB + np.array([0.0, 0.0, -8.0]))), (3.8, T_w(HUB + np.array([0.0, 0.0, -2.0]))),
            (4.8, T_w(HUB + np.array([0.0, 0.0, -2.0])))]

    def hand(side):
        sx = 1.0 if side == "l" else -1.0

        def f(Td):
            c = Td["w"][:3, 3]
            p = c + np.array([sx * 27.0, 2.0, -4.0])
            return HandTarget(side, tuple(p), (0.0, 0.2, 1.0), (-sx, 0.0, 0.0), "hook:5", 0.9)
        return f
    hands = {s_: [(0.0, R[s_]), (1.1, hand(s_)), (4.0, hand(s_)), (4.8, R[s_])] for s_ in ("r", "l")}
    return keyed(4.9, drivers={"w": keys}, hands=hands, parts={"part:wheel": ("wheel:28", keys)},
                 look=[(0.0, c0), (2.2, HUB)], label="fit wheel", stance=st)


def j_lug_torque(S, sk):
    """Torque a lug nut with a long torque wrench: both hands on the handle, pull down to the click."""
    st = K.kneel_stance(sk, knee_down="l", forward=30.0)
    R = posture_rest(S, st)
    N = LUG_HUB + np.array([0.0, 10.0, -6.0])                       # top lug nut

    def T_th(th):
        a = math.radians(146.0 + th)                                 # handle back across in front, pulled down
        d = np.array([math.cos(a), math.sin(a), 0.0])
        return K.T_at(N - Z * 4.0, d, -Z)                            # socket (tool -Y) onto the nut (+Z)
    rh = lambda Td: K.power(sk, "r", wpt(Td["t"], (66.0, 0.8, 0.0)), -wdir(Td["t"], X), grip="power:1.45",
                            lean=(0.0, -0.2, 1.0))
    lh = lambda Td: K.power(sk, "l", wpt(Td["t"], (50.0, 0.8, 0.0)), -wdir(Td["t"], X), grip="power:1.1",
                            lean=(0.0, -0.2, 1.0), weight=0.8)
    tk = [(0.0, None), (1.5, T_th(0.0)), (3.2, T_th(30.0)), (3.28, T_th(32.0)), (3.9, T_th(32.0))]
    tk, hk = J.tool_track(sk, rh, tk + [(5.1, None)], at=R["r"].point)
    return K.keyed(5.2, hands={"r": hk, "l": [(0.0, R["l"]), (1.6, lh), (3.9, lh), (4.6, R["l"])]},
                   tools={"t": ("torque_long", tk)}, look=tuple(N), label="lug torque", stance=st)


def j_guide_hoist(S, sk):
    """Guide a counterweight lowered by a hoist onto its mount: palms on its sides, steering it down."""
    R, _ = rest_targets(S)
    spec, size = "box:60x34x40:powder_dark", (60.0, 34.0, 40.0)

    def T_c(y, sway=0.0):
        return K.T_at(np.array([sway, y, 56.0]), X, Y)
    keys = [(0.0, T_c(160.0, 3.0)), (1.0, T_c(146.0, -2.5)), (2.0, T_c(132.0, 2.0)), (3.0, T_c(120.0, -1.0)),
            (4.0, T_c(110.0, 0.4)), (4.6, T_c(107.0)), (5.6, T_c(107.0))]
    sling = [(t, K.T_at(T[:3, 3] + Y * 77.0, X, Y)) for t, T in keys]

    def hand(side):
        sx = 1.0 if side == "l" else -1.0

        def f(Td):
            c = Td["b"][:3, 3]
            return K.brace(sk, side, c + np.array([sx * 30.0, -2.0, -6.0]), normal=(sx, 0.0, 0.0), toward=(0.0, 0.6, 1.0),
                           weight=0.7)
        return f
    hands = {s_: [(0.0, R[s_]), (1.0, hand(s_)), (4.8, hand(s_)), (5.5, R[s_])] for s_ in ("r", "l")}
    return keyed(5.7, drivers={"b": keys}, hands=hands,
                 parts={"part:cwt": (spec, keys), "part:sling": ("box:1.2x120x1.2:steel_dark", sling)},
                 look=[(0.0, (0.0, 150.0, 56.0)), (4.0, (0.0, 105.0, 56.0))], label="guide hoist")


def j_circlip(S, sk):
    """Fit a circlip into the groove on a pin end with circlip pliers; the off hand steadies the part."""
    P = np.array([-4.0, 104.0, 38.6])
    frames = J._plier_job(S, sk, "circlip_pliers", P, (-0.15, 0.0, 1.0), 1, "circlip",
                          hold_left=K.brace(sk, "l", (12.0, 96.0, 40.0), normal=(0.0, 1.0, 0.0), toward=(0.0, 0.0, 1.0)))
    n = len(frames)
    T = K.T_at(P + Z * 0.9, X, Y)
    for i in range(int(n * 0.5), n):                               # the clip sits in the groove once squeezed on
        frames[i].tools["part:circlip"] = ("ring:1.7:0.18:steel_blued", T)
    return frames


# ---------------------------------------------------------------------------------------------- electrical
def j_control_box(S, sk):
    """Mount the platform control box: the off hand lifts it from the ledge and holds it on the plate, the other
    hand starts the four bolts."""
    R, _ = rest_targets(S)
    c_led = np.array([-4.0, 106.0, PZ - 4.5])
    c_on = np.array([-4.0, 115.0, PZ - 4.5])
    spec = "box:30x18x8:abs_lightgrey"
    bk = [(0.0, K.T_at(c_led, X, Y)), (1.0, K.T_at(c_led, X, Y)), (1.7, K.T_at(c_on, X, Y))]
    lh = lambda Td: K.brace(sk, "l", Td["b"][:3, 3] + np.array([15.0, 0.0, 0.0]), normal=(1.0, 0.0, 0.0),
                            toward=(0.0, 0.25, 1.0), weight=0.8)
    corners = [c_on + np.array([dx, dy, -4.4]) for dx, dy in ((-12.0, 7.0), (-12.0, -7.0), (12.0, -7.0), (12.0, 7.0))]
    rk = [(0.0, R["r"])]
    t = 1.8
    for p in corners:
        g = K.pinch(sk, "r", p, approach=(0.0, -0.15, 1.0), palm=(0.0, -1.0, 0.2), grip="tripod")
        rk += [(t + 0.5, g)]
        for k in range(2):                                         # two finger twists per bolt
            rk += [(t + 0.5 + 0.3 * (k + 1), J.rot_target(g, Z, -45.0, p)), (t + 0.65 + 0.3 * (k + 1), g)]
        t += 1.3
    rk += [(t + 0.8, R["r"])]
    return keyed(t + 0.9, drivers={"b": bk}, hands={"r": rk, "l": [(0.0, R["l"]), (0.9, lh), (t, lh), (t + 0.8, R["l"])]},
                 parts={"part:ctrl": (spec, bk)}, look=tuple(c_on), label="control box")


def j_deutsch(S, sk):
    """Mate a sealed connector, then push its wedge lock home with a fingertip."""
    R, _ = rest_targets(S)
    P = np.array([-4.0, 96.0, PZ])
    frames = J._connector_frames(S, sk, P)
    last = frames[-1].tools["t"][1]
    wedge = wpt(last, (0.0, 0.0, -1.2)) + np.array([0.0, 1.2, 0.0])
    over = K.pinch(sk, "r", wedge - Z * 2.5, approach=(0.0, -0.1, 1.0), palm=(0, -1, 0), grip="point")
    press = K.pinch(sk, "r", wedge - Z * 0.3, approach=(0.0, -0.1, 1.0), palm=(0, -1, 0), grip="point")
    extra = K.keyed(1.8, hands={"r": [(0.0, R["r"]), (0.6, over), (0.85, press), (1.05, press), (1.25, over), (1.8, R["r"])],
                                "l": [(0.0, R["l"])]},
                    tools={"t": ("connector", [(0.0, last)])}, look=tuple(P), label="wedge lock")
    return frames + extra


def j_joystick(S, sk):
    """Drop a joystick into the console cutout from above, then reach under the console and spin on its panel nut."""
    R, _ = rest_targets(S)
    H = np.array([-4.0, 114.6, 45.0])                               # cutout (top face)
    spec = "joystick"
    bk = [(0.0, K.T_at(H + np.array([-18.0, 6.0, -16.0]), X, Y)), (1.2, K.T_at(H + np.array([-18.0, 6.0, -16.0]), X, Y)),
          (2.0, K.T_at(H + Y * 8.0, X, Y)), (2.6, K.T_at(H, X, Y)), (6.0, K.T_at(H, X, Y))]
    rh = lambda Td: K.power(sk, "r", wpt(Td["j"], (0.0, 15.5, 0.0)), wdir(Td["j"], (0.0, 0.0, -1.0)), grip="power:2.0",
                            lean=(0.0, -1.0, 0.4))
    nut = H - Y * 8.0
    g = K.pinch(sk, "l", nut, approach=(-0.35, 0.75, 0.55), palm=(-0.6, 0.0, 0.8), grip="tripod")
    lk = [(0.0, R["l"]), (3.2, g)]
    t = 3.2
    for k in range(3):
        lk += [(t + 0.35, J.rot_target(g, Y, 50.0, nut)), (t + 0.6, g)]
        t += 0.6
    lk += [(t + 0.9, R["l"])]
    return keyed(t + 1.0, drivers={"j": bk}, hands={"r": [(0.0, R["r"]), (1.1, rh), (2.8, rh), (3.4, R["r"])], "l": lk},
                 parts={"part:joy": (spec, bk)}, look=[(0.0, bk[0][1][:3, 3]), (1.6, H), (3.0, nut)], label="joystick")


def j_battery(S, sk):
    """Lift a battery off the low cart and set it in its tray at bench height (two hands)."""
    size = (33.0, 22.0, 18.0)
    spec = "box:33x22x18:plastic_black"
    pick = np.array([24.0, 78.0, 44.0])
    place_at = np.array([-4.0, TOP + 13.0, 46.0])
    f1, _, _ = C.carry_pick(S, pick, size, "battery", spec, True, "r")
    f2 = C.carry_place(S, place_at, size, "battery", spec, True, "r")
    return f1 + f2


# ---------------------------------------------------------------------------------------------- finishing / QC
def j_decal(S, sk):
    """Apply a large decal: lift it off the shelf by its top corners, set it on the panel, squeegee it down."""
    R, _ = rest_targets(S)
    spec = "sheet:40:24:label_vinyl"
    shelf = np.array([-2.0, 88.6, 43.0])
    on = np.array([-2.0, 102.0, 47.9])                              # decal centre on the panel face (z = 48)
    E0 = shelf - Z * 12.0                                           # near edge of the sheet lying on the shelf
    T_shelf = K.T_at(E0, X, Z)                                      # flat: hangs from its edge away from the worker
    T_up = K.T_at(on + Y * 12.0 - Z * 6.0, X, Y)                    # upright, top edge 6 cm off the panel
    T_on = K.T_at(on + Y * 12.0, X, Y)
    dk = [(0.0, T_shelf), (1.0, T_shelf), (2.0, T_up), (2.5, T_on)]

    def corner(side):
        sx = 1.0 if side == "l" else -1.0

        def f(Td):
            T = Td["d"]
            return K.pinch(sk, side, wpt(T, (sx * 19.0, -1.0, -0.3)), approach=(0.0, -0.4, 1.0), palm=(0, -1, 0.3),
                           grip="pinch")
        return f
    sq = lambda Td: K.power(sk, "r", wpt(Td["t"], (0.0, 3.6, 0.0)), wdir(Td["t"], (0.0, 0.0, 1.0)), grip="power:0.9",
                            lean=(0.0, 0.0, 1.0))

    def T_sq(x, y):
        return K.T_at(np.array([x, y, 47.6]), (1.0, 0.0, 0.0), (0.0, 0.25, -1.0))
    Tq0 = K.T_at(np.array([-26.0, 89.6, 40.0]), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0))  # squeegee on the shelf, handle up
    tk = [(0.0, Tq0), (3.0, Tq0), (3.7, T_sq(14.0, 110.0))]
    t = 3.7
    for k, y in enumerate((110.0, 102.0, 94.0)):
        tk += [(t + 0.6, T_sq(-18.0, y)), (t + 0.85, T_sq(14.0, y - 8.0) if k < 2 else T_sq(-18.0, y))]
        t += 0.85
    last = tk[-1][1]
    tk += [(t + 0.9, Tq0)]
    h_pick = J.held(sq, Tq0)
    h_end = J.held(sq, last)
    rk = [(0.0, R["r"]), (0.9, corner("r")), (2.5, corner("r")), (3.0, h_pick), (3.0 + 1e-3, h_pick),
          (3.7, sq), (t, sq), (t + 1e-3, h_end), (t + 0.9, J.held(sq, Tq0)), (t + 1.0, J.held(sq, Tq0))]
    rk[3] = (3.0, h_pick)
    lk = [(0.0, R["l"]), (0.9, corner("l")), (2.5, corner("l")),
          (3.2, K.brace(sk, "l", on + np.array([24.0, 2.0, -0.3]), normal=(0.0, 0.0, -1.0), toward=(0.8, 0.6, 0.0),
                        weight=0.5)), (t, K.brace(sk, "l", on + np.array([24.0, 2.0, -0.3]), normal=(0.0, 0.0, -1.0),
                                                  toward=(0.8, 0.6, 0.0), weight=0.5)), (t + 0.8, R["l"]), (t + 1.6, R["l"])]
    rk += [(t + 1.6, R["r"])]
    return keyed(t + 1.7, drivers={"d": dk}, hands={"r": rk, "l": lk}, tools={"t": ("squeegee", tk)},
                 parts={"part:decal": (spec, dk)}, look=[(0.0, shelf), (1.6, on)], label="decal")


def j_torque_stripe(S, sk):
    """Mark torqued bolts with a paint pen: a stripe across each nut onto the flange."""
    R, _ = rest_targets(S)
    bolts = [np.array([x, TOP + 4.6, 44.0]) for x in (4.0, -6.0, -16.0)]
    d = K.unit(np.array([-0.35, 0.85, -0.45]))

    def pen_at(p, lift=0.0):
        return K.T_at(np.asarray(p, float) + Y * (0.2 + lift), np.cross(d, X), d)
    rh = lambda Td: K.pinch(sk, "r", wpt(Td["t"], (0.0, 2.6, 0.0)), approach=(0.15, -0.55, 1.0), palm=(0.2, -1, 0.0),
                            grip="tripod")
    T0 = K.T_at(np.array([-24.0, 92.0, 16.0]), X, Y)
    tk = [(0.0, T0)]
    t = 0.8
    for b in bolts:
        a, e = b + np.array([1.0, 0.0, -1.6]), b + np.array([-1.0, -1.4, 3.2])
        tk += [(t + 0.3, pen_at(a, 1.2)), (t + 0.45, pen_at(a)), (t + 0.9, pen_at(e)), (t + 1.05, pen_at(e, 1.5))]
        t += 1.05
    tk += [(t + 1.0, T0)]
    br = K.brace(sk, "l", (24.0, TOP, 40.0), toward=(0.0, 0.0, 1.0))
    return K.keyed(t + 1.1, hands={"r": [(0.0, rh)], "l": [(0.0, R["l"]), (0.8, br), (t, br), (t + 1.0, R["l"])]},
                   tools={"t": ("pen", tk)}, look=[(0.0, bolts[0]), (t - 1.0, bolts[-1])], label="torque stripe")


def j_grease(S, sk):
    """Grease a pivot: the off hand holds the gun barrel with the coupler on the nipple, the other pumps the lever."""
    R, _ = rest_targets(S)
    Nip = np.array([-4.0, TOP + 10.4, 44.0])
    T_on = K.T_at(Nip, X, Y)
    T_off = K.T_at(Nip + Y * 4.0, X, Y)
    pivot_l = np.array([-2.8, 11.0, 0.0])

    def T_lever(T, a):
        Tl = K.T_at(wpt(T, pivot_l), wdir(T, X), wdir(T, Y))
        return K.rot_about(Tl, wdir(T, Z), -a, wpt(T, pivot_l))
    lh = lambda Td: K.power(sk, "l", wpt(Td["t"], (0.0, 26.0, 0.0)), wdir(Td["t"], Y), grip="power:2.7", weight=0.9)
    tk = [(0.0, None), (1.2, T_off), (1.6, T_on)]
    t = 1.6
    for _ in range(5):
        tk += [(t + 0.45, T_on), (t + 0.8, T_on)]
        t += 0.8
    tk += [(t + 0.3, T_off)]
    tk, hk = J.tool_track(sk, lh, tk + [(t + 1.3, None)], key="t")
    ang = [(0.0, 0.0), (1.6, 0.0)]
    tt = 1.6
    for _ in range(5):
        ang += [(tt + 0.45, 32.0), (tt + 0.8, 0.0)]
        tt += 0.8

    def a_at(tq):
        for (t0, a0), (t1, a1) in zip(ang, ang[1:]):
            if tq <= t1:
                u = 0.0 if t1 <= t0 else (tq - t0) / (t1 - t0)
                return a0 + (a1 - a0) * G.ease(u)
        return ang[-1][1]
    frames = K.keyed(t + 1.4, hands={"l": hk}, tools={"t": ("grease_gun", tk)}, look=tuple(Nip), label="grease")
    rest = rest_targets(S)[0]["r"]
    for i, f in enumerate(frames):                                  # the lever (a part) and the pumping hand on it
        tq = i / G.FPS
        Tg = f.tools["t"][1]
        Tl = T_lever(Tg, a_at(tq))
        f.tools["part:lever"] = ("box:1.6x19x1.2:steel_blued", Tl @ _offset((-0.6, 9.5, 0.0)))
        hand = K.power(sk, "r", wpt(Tl, (-0.6, 17.0, 0.0)), wdir(Tl, Y), grip="power:0.9",
                       lean=K.unit(wdir(Tl, (-1.0, 0.3, 0.0))))
        u = min(1.0, max(0.0, (tq - 0.4) / 1.0)) if tq < 1.6 else min(1.0, max(0.0, (t + 1.2 - tq) / 0.9))
        f.targets = [x for x in f.targets if x.side != "r"] + [G.lerp_target(rest, hand, G.ease(u))]
    return frames


def _offset(v):
    T = np.eye(4)
    T[:3, 3] = v
    return T


def j_inspect(S, sk):
    """Squat and inspect a weld under the frame with a flashlight, the light following the seam."""
    st = K.squat_stance(sk, depth=0.42, heels_up=26.0)
    st.yaw_l, st.yaw_r = -6.0, 6.0
    R = posture_rest(S, st)
    seam = [np.array([x, 73.8, 49.0]) for x in (22.0, 8.0, -8.0, -22.0)]
    hand_at = np.array([-14.0, 58.0, 26.0])

    def T_light(target):
        d = K.unit(target - hand_at)
        return K.T_at(hand_at + d * 10.0, d, np.cross(np.cross(d, Y), d))
    rh = lambda Td: K.power(sk, "r", wpt(Td["t"], (-11.0, 0.0, 0.0)), wdir(Td["t"], X), grip="power:1.45")
    tk = [(0.0, None), (1.3, T_light(seam[0]))]
    t = 1.3
    for p in seam[1:]:
        tk += [(t + 1.0, T_light(p))]
        t += 1.0
    tk, hk = J.tool_track(sk, rh, tk + [(t + 1.1, None)], at=R["r"].point)
    br = knee_rest(sk, "l", st, fwd=26.0, y=sk.length("calf_l", "foot_l") * 0.9 + 10.0)
    return K.keyed(t + 1.2, hands={"r": hk, "l": [(0.0, R["l"]), (1.0, br), (t, br), (t + 1.0, R["l"])]},
                   tools={"t": ("flashlight", tk)}, look=[(0.0, seam[0])] + [(1.3 + k * 1.0, p) for k, p in enumerate(seam)],
                   label="inspect weld", stance=st)


def j_guardrail(S, sk):
    """Check a platform guardrail: both hands on the top rail, push and pull; then pull the detent pin and refit it."""
    R, _ = rest_targets(S)
    rail_l = K.power(sk, "l", np.array([22.0, 110.0, 46.0]), X, grip="power:2.1", lean=(0.0, -0.6, 1.0))
    rail_r = K.power(sk, "r", np.array([-22.0, 110.0, 46.0]), -X, grip="power:2.1", lean=(0.0, -0.6, 1.0))

    def shift(t_, dz):
        return HandTarget(t_.side, tuple(np.asarray(t_.point) + Z * dz), t_.fingers, t_.palm, t_.grip, t_.weight)
    lk = [(0.0, R["l"]), (1.0, rail_l), (1.5, shift(rail_l, 1.2)), (1.9, shift(rail_l, -1.2)), (2.3, shift(rail_l, 1.0)),
          (2.7, rail_l), (3.3, R["l"])]
    pin0 = np.array([-50.0, 96.0, 40.4])
    spec = "box:1.2x1.2x7:steel_machined"
    g = K.pinch(sk, "r", pin0 - Z * 3.6, approach=(0.05, -0.1, 1.0), palm=(1.0, -0.3, 0.0), grip="pinch")
    rk = [(0.0, R["r"]), (1.0, rail_r), (1.5, shift(rail_r, 1.2)), (1.9, shift(rail_r, -1.2)), (2.3, shift(rail_r, 1.0)),
          (2.7, rail_r), (3.5, g), (4.0, shift(g, -6.0)), (4.6, shift(g, -6.0)), (5.1, g), (5.8, R["r"])]
    T_in = K.T_at(pin0, X, Y)
    pk = [(0.0, T_in), (3.5, T_in), (4.0, K.T_at(pin0 - Z * 6.0, X, Y)), (4.6, K.T_at(pin0 - Z * 6.0, X, Y)), (5.1, T_in)]
    return K.keyed(5.9, hands={"r": rk, "l": lk}, parts={"part:pin": (spec, pk)},
                   look=[(0.0, (0.0, 110.0, 46.0)), (3.1, pin0)], label="guardrail check")


# ---------------------------------------------------------------------------------------------- the library
Job = J.Job
JOBS_AWP = [
    Job("fitting_start", "Start a hose fitting", "hydraulics", "W50M", j_fitting_start, fx(b_manifold_front),
        (-4, TOP + 6, 34)),
    Job("two_wrench", "Two-wrench fitting", "hydraulics", "W95M", j_two_wrench, fx(b_two_wrench), (-4, TOP + 18, 36)),
    Job("hose_clip", "Clip a hose to the rail", "hydraulics", "W05F", j_hose_clip, fx(b_hose_rail), (-6, 99, 44)),
    Job("coupler", "Quick coupler", "hydraulics", "W50M", j_coupler, fx(b_coupler, kind="panel"), (-4, 105, 32)),
    Job("cap_ports", "Cap open ports", "hydraulics", "W95M", j_cap_ports, fx(b_cap_ports), (-4, TOP + 9, 42)),
    Job("overhead", "Overhead ratchet", "overhead / low", "W50M", j_overhead, fx(b_overhead), (-4, 165, 36),
        camera=(-150.0, -40.0, -120.0)),
    Job("kneel_drive", "Kneel and drive", "overhead / low", "W95M", j_kneel_drive, fx(b_kneel_plate, kind="panel"),
        (-6, 50, 36), camera=(-150.0, 30.0, -70.0)),
    Job("squat_connect", "Under-frame connector", "overhead / low", "W05F", j_squat_connect, fx(b_under),
        (-4, 66, 30), camera=(-130.0, 10.0, -90.0)),
    Job("lean_in", "Lean in to ratchet", "overhead / low", "W50M", j_lean_in, fx(b_lean_in), (-4, 100, 60),
        camera=(-150.0, 40.0, -10.0)),
    Job("cylinder", "Align a cylinder (2 people)", "heavy", "W95M", j_cylinder, fx(b_clevis), (10, 104, 40),
        partner=("W50M", (78.0, 0.0, 0.0), p_cylinder), camera=(-70.0, 60.0, 230.0)),
    Job("pivot_pin", "Drive a pivot pin", "heavy", "W50M", j_pivot_pin, fx(b_pivot), (0, 112, 42),
        camera=(-150.0, 20.0, -40.0)),
    Job("wheel", "Fit a wheel", "heavy", "W95M", j_wheel, fx(b_hub), (0, 60, 56), camera=(-150.0, 50.0, -30.0)),
    Job("lug_torque", "Torque lug nuts", "heavy", "W50M", j_lug_torque, fx(b_hub_mounted), (20, 80, 56),
        camera=(-140.0, 50.0, -60.0)),
    Job("guide_hoist", "Guide a hoisted load", "heavy", "W05F", j_guide_hoist, fx(b_hoist), (0, 120, 46),
        camera=(-150.0, 30.0, -60.0)),
    Job("circlip", "Fit a circlip", "heavy", "W05F", j_circlip, fx(b_circlip), (-4, 104, 36)),
    Job("control_box", "Mount a control box", "electrical", "W50M", j_control_box, fx(b_ctrl, kind="panel"),
        (-4, 116, 30)),
    Job("deutsch", "Sealed connector + wedge", "electrical", "W95M", j_deutsch, fx(b_deutsch, kind="panel"),
        (-4, 96, 36)),
    Job("joystick", "Fit a joystick", "electrical", "W50M", j_joystick, fx(b_console), (-4, 98, 42),
        camera=(-140.0, 20.0, -60.0)),
    Job("battery", "Fit a battery", "electrical", "W95M", j_battery, fx(b_battery), (0, 90, 48)),
    Job("decal", "Apply a decal", "finishing", "W05F", j_decal, fx(b_decal), (-2, 102, 44), camera=(-110.0, 20.0, -150.0)),
    Job("torque_stripe", "Torque-stripe bolts", "finishing", "W50M", j_torque_stripe, fx(b_stripe), (-6, 98, 42)),
    Job("grease", "Grease a pivot", "finishing", "W95M", j_grease, fx(b_grease), (-4, 108, 42)),
    Job("inspect", "Inspect a weld (crouch)", "finishing", "W05F", j_inspect, fx(b_inspect), (-4, 62, 40),
        camera=(-140.0, 10.0, -80.0)),
    Job("guardrail", "Check the guardrail", "finishing", "W50M", j_guardrail, fx(b_guardrail), (-14, 104, 40),
        camera=(-120.0, 40.0, -130.0)),
]
