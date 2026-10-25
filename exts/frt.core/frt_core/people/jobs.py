"""Hand-job library: 26 manufacturing hand tasks as keyframed motions (handkit rules + the posture solver).

Every job is in the worker frame (worker at the origin facing +Z): a fixture (bench top 92 cm with its front edge
25 cm ahead, or a wall panel at z = 42), the tools / parts, and hand keyframes:

  fastening    hand-start nut, ratchet, torque wrench, screwdriver, impact driver, hex key, combination wrench
  electrical   mate connector, unplug connector, mains plug, cable tie, cut tie tail, strip wire, crimp,
               route cable into clips
  assembly     hammer, mallet, insert pin, two-hand box move, caulking bead
  operate      push buttons, turn handwheel, scan barcode, apply label, wipe down, sign checklist

    for job in JOBS: frames = job.frames(solver)      # -> bake
"""
import math
from dataclasses import dataclass

import numpy as np

from ..geom import Shapes
from . import handkit as K
from . import motion_carry as C
from . import motion_gen as G
from .motion_gen import rest_targets
from .solver import HandTarget

Y = np.array([0.0, 1.0, 0.0])
Z = np.array([0.0, 0.0, 1.0])
X = np.array([1.0, 0.0, 0.0])
TOP = 92.0                      # bench top
PZ = 42.0                       # panel face (worker side)
UPPER = ("head", "neck_01", "spine_03", "spine_02", "spine_01", "pelvis")   # head / chest / belly centres
HIPS = ("pelvis", "thigh_l", "thigh_r")
BENCH_FRONT = 25.0
SZ = PZ + 16.0                  # panels worked at arm's length (screwdriver, push buttons): the worker stands back


@dataclass
class Job:
    id: str
    title: str
    group: str
    worker: str
    build: callable             # (S, sk) -> frames
    fixture: callable           # (Shapes) -> None
    focus: tuple                # camera target (worker frame)
    partner: tuple = None       # (worker id, (x, z, yaw) in the station, build(S, sk, place) -> frames)
    camera: tuple = None        # optional close-up (eye offset from the focus)
    keep_out: tuple = None      # [(joint names, max z)] - default from the fixture (panel face / bench front)

    def frames(self, S):
        from . import tools
        tools.HAND = S                             # (fingertip grips placed by this body's own hand)
        return self.build(S, S.sk)

    def keep_out_rules(self):
        if self.keep_out is not None:
            return list(self.keep_out)
        kind = getattr(self.fixture, "kind", "")
        if kind == "panel":                        # chest / belly 13 cm, the head (face forward of it) 20 cm
            z = getattr(self.fixture, "z", PZ)
            return [(UPPER, z - 13.0), (("head",), z - 20.0)]
        if kind == "bench":
            return [(HIPS, BENCH_FRONT - 10.0)]
        return []


# ---------------------------------------------------------------------------------------------- fixtures
def bench(s, width=110.0, z0=25.0, depth=60.0, top=TOP):
    s.rbox((0, top - 2.0, z0 + depth / 2), (width, 4.0, depth), 0.6, "wood_top", seg=2)
    for x in (-width / 2 + 4, width / 2 - 4):
        for z in (z0 + 4, z0 + depth - 4):
            s.rprism((x, 0.0, z), (x, top - 4.0, z), 4.0, 4.0, 0.6, "powder_blue", up=(1, 0, 0))
    s.rprism((-width / 2 + 4, 14.0, z0 + depth / 2), (width / 2 - 4, 14.0, z0 + depth / 2), 3.0, 3.0, 0.5,
             "powder_blue")


def panel(s, width=110.0, z=PZ, height=(70.0, 165.0)):
    y0, y1 = height
    s.rbox((0, (y0 + y1) / 2, z + 1.0), (width, y1 - y0, 2.0), 0.6, "powder_lightgrey", seg=2)
    for x in (-width / 2 + 3, width / 2 - 3):
        s.rprism((x, 0.0, z + 4.0), (x, y1, z + 4.0), 4.0, 4.0, 0.6, "powder_dark", up=(1, 0, 0))
        s.rbox((x, 0.5, z + 4.0), (12.0, 1.0, 30.0), 0.3, "powder_dark", seg=1)


def bolt_up(s, p, d=1.6):
    s.bolt(tuple(p), (0, 1, 0), d, "bolt_zinc")


def block(s, c, size, look="cast_aluminium"):
    s.rbox(tuple(c), tuple(size), 0.8, look, seg=2)


# ---------------------------------------------------------------------------------------------- helpers
def rot_target(t, axis, deg, pivot):
    a = K.unit(axis)
    th = math.radians(deg)
    Kx = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    R = np.eye(3) + math.sin(th) * Kx + (1 - math.cos(th)) * Kx @ Kx
    pv = np.asarray(pivot, float)
    return HandTarget(t.side, tuple(R @ (np.asarray(t.point) - pv) + pv), tuple(R @ np.asarray(t.fingers)),
                      tuple(R @ np.asarray(t.palm)), t.grip, t.weight)


def hip_tool(sk, side, key_x, key_y):
    """A tool hanging in the hand at the hip: its X / Y axes."""
    sx = 1.0 if side == "l" else -1.0
    p = np.array([sx * (abs(sk.world["upperarm_l"][0]) + 4.0), sk.world["pelvis"][1] - 2.0, 10.0])
    return K.T_at(p, key_x, key_y)


def wpt(T, p):
    return T[:3, :3] @ np.asarray(p, float) + T[:3, 3]


def wdir(T, d):
    return T[:3, :3] @ np.asarray(d, float)


def held(hand_fn, T_ref, key="t"):
    """The hand that holds a tool, rigidly attached to it: the hand target hand_fn gives at tool pose T_ref, carried
    along with the tool (a real grip does not slide while the tool is brought to the work)."""
    h = hand_fn({key: T_ref})
    Ri, o = T_ref[:3, :3].T, T_ref[:3, 3]
    lp, lf, ln = Ri @ (np.asarray(h.point) - o), Ri @ np.asarray(h.fingers), Ri @ np.asarray(h.palm)

    def f(Td):
        T = Td[key]
        return HandTarget(h.side, tuple(wpt(T, lp)), tuple(wdir(T, lf)), tuple(wdir(T, ln)), h.grip, h.weight)
    return f


def carried(sk, hand_fn, T_ref, key="t", at=None):
    """Tool pose at the hip that puts the holding hand (rigid grip as at T_ref) hanging naturally: fingers down, palm
    towards the thigh, by the hip. The hand then turns the shortest way between hanging and working."""
    h = hand_fn({key: T_ref})
    side = h.side
    sx = 1.0 if side == "l" else -1.0
    f_w = K.unit(h.fingers)
    p_w = K.unit(np.asarray(h.palm) - f_w * np.dot(h.palm, f_w))
    Fw = np.column_stack([f_w, p_w, np.cross(f_w, p_w)])
    f_h = K.unit((0.0, -1.0, 0.12))
    p_h = K.unit(np.array([-sx, 0.0, 0.0]) - f_h * np.dot((-sx, 0.0, 0.0), f_h))
    Fh = np.column_stack([f_h, p_h, np.cross(f_h, p_h)])
    Rm = Fh @ Fw.T
    q_h = np.array([sx * (abs(sk.world["upperarm_l"][0]) + 3.0), sk.world["pelvis"][1] - 12.0, 8.0]) if at is None \
        else np.asarray(at, float)
    T0 = np.eye(4)
    T0[:3, :3] = Rm @ T_ref[:3, :3]
    T0[:3, 3] = q_h - Rm @ (np.asarray(h.point) - T_ref[:3, 3])
    return T0


def tool_track(sk, hand_fn, tk, key="t", reach=1.0, at=None):
    """Tool keys [(0, None), (t1, T_first), ..., (tn, T_last), (tn + reach, None)]: the None ends become carry
    poses at the hip; returns (tool keys, hand keys) with the hand rigid on the tool while carrying it."""
    t1, Tf = tk[1]
    tn, Tl = tk[-2]
    tk = [(tk[0][0], carried(sk, hand_fn, Tf, key, at))] + tk[1:-1] + [(tk[-1][0], carried(sk, hand_fn, Tl, key, at))]
    e = 1e-3
    hk = [(0.0, held(hand_fn, Tf, key)), (t1 - e, held(hand_fn, Tf, key)), (t1, hand_fn),
          (tn, hand_fn), (tn + e, held(hand_fn, Tl, key))]
    return tk, hk


def cycle_keys(t0, n, period, a, b, frac=0.6):
    """Alternating keys a -> b (in frac of the period) -> a, n times from t0."""
    out = []
    t = t0
    for _ in range(n):
        out += [(t + period * frac, b), (t + period, a)]
        t += period
    return out, t


# ---------------------------------------------------------------------------------------------- fastening
def j_hand_nut(S, sk):
    R, _ = rest_targets(S)
    P = np.array([-6.0, TOP + 4.0, 42.0])                         # nut on a stud
    grab = K.pinch(sk, "r", P + Y * 0.6, approach=(0.15, -0.55, 1.0), palm=(0, -1, 0.2))
    br = K.brace(sk, "l", (20.0, TOP, 40.0))
    keys = [(0.0, R["r"]), (0.9, grab)]
    t = 0.9
    for _ in range(4):                                          # finger-thread: twist, let go, regrip
        keys += [(t + 0.45, rot_target(grab, Y, -55.0, P)), (t + 0.7, grab)]
        t += 0.7
    keys += [(t + 0.9, R["r"])]
    nut = [(0.0, K.T_at(P, X, Y))]
    tt = 0.9
    for k in range(4):
        nut += [(tt + 0.45, K.rot_about(K.T_at(P - Y * 0.25 * (k + 1), X, Y), Y, -55.0 * (k + 1), P))]
        tt += 0.7
    return K.keyed(t + 1.0, hands={"r": keys, "l": [(0.0, R["l"]), (0.8, br), (t, br), (t + 0.9, R["l"])]},
                   parts={"part:nut": ("box:1.7x1.0x1.7:bolt_zinc", nut)}, look=tuple(P), label="hand start")


def _socket_job(S, sk, tool, bolt, strokes, swing, period, hold=0.0, click=False):
    R, _ = rest_targets(S)
    B = np.array(bolt, float)
    d0 = K.unit((-1.0, 0.0, -0.35 if tool == "ratchet_38" else -0.45))   # handle out to the right and back

    def T_th(th):
        c, s_ = math.cos(math.radians(th)), math.sin(math.radians(th))
        d = d0 * c + np.cross(Y, d0) * s_
        return K.T_at(B + Y * 4.0, d, Y)
    L = 19.0 if tool == "ratchet_38" else 38.0

    def rh(Td):
        T = Td["t"]
        return K.power(sk, "r", wpt(T, (L, 1.0, 0.0)), -wdir(T, X), grip=f"power:{1.25 if L < 30 else 1.5}")

    def lh(Td):
        T = Td["t"]
        return K.brace(sk, "l", wpt(T, (0.0, 1.6, 0.0)), Y, weight=0.6)
    tk = [(0.0, None), (1.0, T_th(-swing / 2))]
    t = 1.0
    if click:                                                     # one slow pull to the click, small jerk, hold
        tk += [(t + period, T_th(swing / 2)), (t + period + 0.08, T_th(swing / 2 + 3)), (t + period + 0.6 + hold,
               T_th(swing / 2 + 3))]
        t += period + 0.6 + hold
    else:
        for _ in range(strokes):                                  # power stroke, quicker return
            tk += [(t + period * 0.6, T_th(swing / 2)), (t + period, T_th(-swing / 2))]
            t += period
    tk, hk = tool_track(sk, rh, tk + [(t + 1.0, None)])
    lk = [(0.0, R["l"]), (1.0, lh), (t, lh), (t + 1.0, R["l"])]
    return K.keyed(t + 1.1, hands={"r": hk, "l": lk}, tools={"t": (tool, tk)}, look=tuple(B),
                   label="ratchet" if not click else "torque")


def j_ratchet(S, sk):
    return _socket_job(S, sk, "ratchet_38", (-4.0, TOP + 5.0, 44.0), 4, 50.0, 0.8)


def j_torque(S, sk):
    return _socket_job(S, sk, "torque_wrench", (-4.0, TOP + 5.0, 44.0), 1, 40.0, 1.8, hold=0.4, click=True)


def j_screwdriver(S, sk):
    R, _ = rest_targets(S)
    P = np.array([0.0, 100.0, SZ])                               # screw in the panel, driven towards +Z

    def T_rot(rot):
        x = K.unit(np.array([math.cos(math.radians(rot)), math.sin(math.radians(rot)), 0.0]))
        return K.T_at(P, x, -Z)                                   # tool Y (shaft -> handle) towards the worker

    def rh(Td):
        T = Td["t"]
        g = K.power(sk, "r", wpt(T, (0.0, 18.0, 0.0)), wdir(T, (0, -1, 0)), grip="power:1.6", lean=(0.0, -1.0, 0.0))
        return K.tilt(g, wdir(T, (0, -1, 0)), 60.0)                # handshake grip, handle diagonal in the palm
    # drive strokes: the hand rolls the tool 60 deg (forearm supination), loosens, rolls back over the handle and
    # grips again - the screw only ever turns forward
    tk = [(0.0, None), (1.2, T_rot(0.0))]
    t, ang = 1.2, 0.0
    for k in range(5):
        ang += 60.0
        tk += [(t + 0.5, T_rot(ang)), (t + 0.75, T_rot(ang))]
        t += 0.75
    tk, hk = tool_track(sk, rh, tk + [(t + 1.0, None)])
    h0 = rh({"t": T_rot(0.0)})

    def roll(a, grip):
        r_ = rot_target(h0, Z, a, P)
        return HandTarget(r_.side, r_.point, r_.fingers, r_.palm, grip, r_.weight)
    mid, tt = [(1.2, h0)], 1.2
    for k in range(5):
        last = k == 4
        mid += [(tt + 0.08, roll(0.0, "power:1.6")), (tt + 0.5, roll(60.0, "power:1.6")),
                (tt + 0.58, roll(60.0, "power:2.3" if not last else "power:1.6")),
                (tt + 0.75, roll(0.0, "power:2.3" if not last else "power:1.6"))]
        tt += 0.75
    hk = hk[:2] + mid + hk[-1:]
    guide = K.pinch(sk, "l", P - Z * 3.5 + X * 0.6, approach=(-0.8, -0.25, 0.55), palm=(-0.3, -1.0, 0.0),
                    grip="tripod", weight=0.7)              # the off hand guides the shaft near the tip
    return K.keyed(t + 1.1, hands={"r": hk, "l": [(0.0, R["l"]), (1.3, guide), (t, guide), (t + 1.0, R["l"])]},
                   tools={"t": ("screwdriver", tk)}, look=tuple(P), label="screwdriver")


def j_impact(S, sk):
    R, _ = rest_targets(S)
    P = np.array([-6.0, 106.0, PZ])

    def T_at(push=0.0, buzz=0.0):
        return K.T_at(P - Z * (0.0 - push) + Y * buzz, Z, Y)       # barrel (tool X) into the panel

    def rh_pull(pull):
        def f(Td):
            T = Td["t"]
            return K.power(sk, "r", wpt(T, (-12.8, -7.0, 0.0)), wdir(T, Y), grip=f"trigger:1.7:{pull}",
                           lean=wdir(T, X))
        return f
    T0 = K.T_at(np.array([-26.0, 88.0, 14.0]), (0, -0.3, 1), Y)
    tk = [(0.0, T0), (1.1, T_at(-3.0)), (1.5, T_at(0.0))]
    t = 1.5
    for k in range(10):                                           # driving: vibration while the trigger is held
        tk += [(t + 0.09, T_at(0.15 * k / 10)), (t + 0.18, T_at(0.15 * (k + 0.5) / 10))]
        t += 0.18
    tk += [(t + 0.4, T_at(-3.0)), (t + 1.4, T0)]
    hand = [(0.0, rh_pull(0.0)), (1.5, rh_pull(0.0)), (1.65, rh_pull(1.0)), (t, rh_pull(1.0)), (t + 0.2, rh_pull(0.0))]
    def br(Td):                                    # the off hand on top of the drill body, pushing it into the work
        T = Td["t"]
        return K.brace(sk, "l", wpt(T, (-11.5, 2.9, 0.0)), normal=wdir(T, Y), toward=wdir(T, X), weight=0.7,
                       grip="relaxed")
    return K.keyed(t + 1.5, hands={"r": hand, "l": [(0.0, R["l"]), (1.0, br), (t + 0.3, br), (t + 1.4, R["l"])]},
                   tools={"t": ("impact_driver", tk)}, look=tuple(P), label="impact driver")


def j_hex_key(S, sk):
    R, _ = rest_targets(S)
    P = np.array([-4.0, TOP + 1.0, 42.0])

    def T_th(th):
        d = np.array([-math.cos(math.radians(th)), 0.0, -math.sin(math.radians(th))])
        return K.T_at(P + Y * 2.8, d, Y)

    def rh(Td):
        T = Td["t"]
        return K.power(sk, "r", wpt(T, (6.5, 0.6, 0.0)), -wdir(T, X), grip="power:0.6")
    tk = [(0.0, None), (1.0, T_th(-20.0))]
    t = 1.0
    for _ in range(4):
        tk += [(t + 0.55, T_th(25.0)), (t + 0.95, T_th(-20.0))]
        t += 0.95
    tk, hk = tool_track(sk, rh, tk + [(t + 1.0, None)])
    br = K.brace(sk, "l", (20.0, TOP, 40.0))
    return K.keyed(t + 1.1, hands={"r": hk, "l": [(0.0, R["l"]), (0.9, br), (t, br), (t + 1.0, R["l"])]},
                   tools={"t": ("hex_key", tk)}, look=tuple(P), label="hex key")


def j_wrench(S, sk):
    R, _ = rest_targets(S)
    P = np.array([-4.0, TOP + 3.5, 42.0])

    def T_th(th):
        d = np.array([-math.cos(math.radians(th)), 0.0, -0.3 - math.sin(math.radians(th))])
        d = K.unit(d)
        T = K.T_at(np.zeros(3), d, Y)
        T[:3, 3] = P - wdir(T, (18.0, 0.0, 0.0))                  # ring end on the nut
        return T

    def rh(Td):
        T = Td["t"]
        return K.power(sk, "r", wpt(T, (5.0, 0.0, 0.0)), wdir(T, X), grip="power:0.9")
    tk = [(0.0, None), (1.0, T_th(-20.0))]
    t = 1.0
    for _ in range(4):
        tk += [(t + 0.55, T_th(25.0)), (t + 0.9, T_th(-20.0))]
        t += 0.9
    tk, hk = tool_track(sk, rh, tk + [(t + 1.0, None)])
    br = K.brace(sk, "l", (20.0, TOP, 40.0))
    return K.keyed(t + 1.1, hands={"r": hk, "l": [(0.0, R["l"]), (0.9, br), (t, br), (t + 1.0, R["l"])]},
                   tools={"t": ("wrench_comb", tk)}, look=tuple(P), label="wrench")


# ---------------------------------------------------------------------------------------------- electrical
def _connector_frames(S, sk, P, unplug=False):
    R, _ = rest_targets(S)

    def T_at(back):
        T = K.T_at(np.zeros(3), X, Y)
        T[:3, 3] = P - Z * back - Z * 1.8
        return T

    def rh(press=0.0):
        def f(Td):
            T = Td["t"]
            g = ("blend", "pinch", "lateral", press)
            return K.pinch(sk, "r", wpt(T, (0.0, 1.2, -1.2)), approach=(0.0, -0.25, 1.0), palm=(0, -1, 0), grip=g)
        return f
    def br(Td):                                    # the off hand holds the wire bundle behind the connector
        T = Td["t"]
        return K.power(sk, "l", wpt(T, (0.3, -1.4, -11.0)), wdir(T, (0.0, 0.25, 0.97)), grip="power:0.7", weight=0.8)
    T0 = K.T_at(np.array([-24.0, 90.0, 16.0]), X, Y)
    if not unplug:
        tk = [(0.0, T0), (1.0, T_at(5.0)), (1.4, T_at(1.5)), (1.8, T_at(-0.2)), (1.9, T_at(0.0)), (2.3, T_at(0.0)),
              (2.45, T_at(0.6)), (2.6, T_at(0.0))]
        hand = [(0.0, rh()), (2.6, rh()), (3.0, R["r"]), (3.6, R["r"])]
        tk += [(3.6, T_at(0.0))]
        end = 3.6
    else:
        tk = [(0.0, T_at(0.0)), (1.0, T_at(0.0)), (1.25, T_at(0.15)), (1.9, T_at(6.0)), (2.4, T_at(7.0)),
              (3.4, T0)]
        hand = [(0.0, R["r"]), (1.0, rh(0.0)), (1.2, rh(1.0)), (1.6, rh(1.0)), (1.9, rh(0.0)), (3.4, rh(0.0))]
        end = 3.4
    return K.keyed(end + 0.2, hands={"r": hand, "l": [(0.0, R["l"]), (0.9, br), (end - 0.6, br), (end + 0.2, R["l"])]},
                   tools={"t": ("connector", tk)}, look=tuple(P), label="connector" if not unplug else "unplug")


def j_connector(S, sk):
    return _connector_frames(S, sk, np.array([-4.0, 112.0, PZ]))


def j_unplug(S, sk):
    return _connector_frames(S, sk, np.array([-4.0, 112.0, PZ]), unplug=True)


def j_plug(S, sk):
    R, _ = rest_targets(S)
    P = np.array([-4.0, 98.0, PZ])                               # wall socket face

    def T_at(back, wig=0.0):
        T = K.T_at(np.zeros(3), K.unit((1.0, wig, 0.0)), Y)
        T[:3, 3] = P - Z * back
        return T

    def rh(Td):
        T = Td["t"]
        return K.pinch(sk, "r", wpt(T, (0.0, 0.0, -2.2)) + np.array([-2.4, 0.0, 0.0]), approach=(0.35, 0.0, 1.0),
                       palm=(1, 0, 0), grip="lateral")
    T0 = K.T_at(np.array([-24.0, 88.0, 16.0]), X, Y)
    tk = [(0.0, T0), (1.0, T_at(6.0)), (1.5, T_at(1.9)), (1.8, T_at(0.6, 0.06)), (2.0, T_at(0.2, -0.06)),
          (2.2, T_at(0.0)), (3.4, T_at(0.0))]
    hand = [(0.0, rh), (2.3, rh), (2.8, R["r"]), (3.4, R["r"])]
    def br(Td):                                    # the off hand feeds the cable behind the plug
        T = Td["t"]
        return K.power(sk, "l", wpt(T, (0.0, -2.0, -12.0)), wdir(T, (0.0, 0.3, 0.95)), grip="power:0.6", weight=0.8)
    return K.keyed(3.5, hands={"r": hand, "l": [(0.0, R["l"]), (0.9, br), (2.3, br), (3.2, R["l"])]},
                   tools={"t": ("plug", tk)}, look=tuple(P), label="plug")


def _harness_hold(sk, P):
    return K.power(sk, "l", P + X * 9.0, -X, grip="power:1.0", weight=0.7)


def j_cable_tie(S, sk):
    R, _ = rest_targets(S)
    P = np.array([0.0, 108.0, PZ - 4.0])                         # harness along X in front of the panel
    r = 2.6
    hold = _harness_hold(sk, P)

    def at(th):                                                   # fingertips push the tie round; the hand keeps
        c = P + (Y * math.cos(th) - Z * math.sin(th)) * r         # its orientation (fingers forward, palm down)
        return K.pinch(sk, "r", c, approach=(0.15, -0.35, 1.0), palm=(0, -1, 0))
    keys = [(0.0, R["r"]), (0.9, at(0.0))]
    t = 0.9
    for k in range(1, 9):
        keys.append((t + 0.1 * k, at(-2 * math.pi * k / 8 * 0.95)))
    t += 0.9
    tail = P + Y * r
    up = (0.15, 0.2, 1.0)
    keys += [(t + 0.15, K.pinch(sk, "r", tail + Y * 1.0, approach=up)), (t + 0.5, K.pinch(sk, "r", tail + Y * 4.5, approach=up)),
             (t + 0.6, K.pinch(sk, "r", tail + Y * 6.0, approach=up)), (t + 1.5, R["r"])]
    tie = (t + 0.15, K.T_at(P, Y, Z))
    return K.keyed(t + 1.6, hands={"r": keys, "l": [(0.0, R["l"]), (0.8, hold), (t + 0.8, hold), (t + 1.5, R["l"])]},
                   parts={"part:tie": ("ring:2.6:0.2:plastic_black", [tie])}, look=tuple(P), label="cable tie")


def _plier_job(S, sk, tool, P, approach, squeezes, label, pull=None, hold_left=None):
    R, _ = rest_targets(S)
    a = K.unit(approach)

    def T_at(off=0.0):
        T = K.T_at(np.zeros(3), a, Y)
        T[:3, 3] = P + a * off
        return T

    def rh(r):
        def f(Td):
            T = Td["t"]
            return K.power(sk, "r", wpt(T, (-13.0, 0.0, 0.0)), wdir(T, (0, 0, -1)), grip=f"power:{r:.2f}",
                           lean=wdir(T, X))
        return f
    T0 = hip_tool(sk, "r", (0, -1, 0.3), (0, 0, 1))
    tk = [(0.0, T0), (1.0, T_at(-1.5)), (1.3, T_at(0.0))]
    hand = [(0.0, rh(2.6)), (1.3, rh(2.6))]
    t = 1.3
    for _ in range(squeezes):
        hand += [(t + 0.35, rh(1.5)), (t + 0.55, rh(1.5)), (t + 0.8, rh(2.6))]
        t += 0.8
    if pull is not None:                                          # strip: pull the jaws along the wire
        tk += [(t, T_at(0.0)), (t + 0.5, T_at(pull))]
        hand[-1] = (t - 0.25, rh(1.5))
        hand += [(t + 0.5, rh(1.5)), (t + 0.7, rh(2.6))]
        t += 0.7
    tk += [(t, tk[-1][1]), (t + 1.0, T0)]
    left = hold_left or K.pinch(sk, "l", P - a * 7.0 + np.array([6.0, 0.0, 0.0]), approach=(-0.3, 0, 1.0))
    return K.keyed(t + 1.1, hands={"r": hand, "l": [(0.0, R["l"]), (0.9, left), (t, left), (t + 1.0, R["l"])]},
                   tools={"t": (tool, tk)}, look=tuple(P), label=label)


def j_cut_tie(S, sk):
    P = np.array([0.0, 108.0 + 2.6 + 1.2, PZ - 4.0])
    return _plier_job(S, sk, "flush_cutter", P, (-0.3, -0.2, 1.0), 1, "cut tie",
                      hold_left=_harness_hold(sk, P - Y * 3.8))


def j_strip(S, sk):
    P = np.array([-4.0, 104.0, 38.0])
    return _plier_job(S, sk, "wire_stripper", P, (-0.2, 0.0, 1.0), 1, "strip wire", pull=-2.5)


def j_crimp(S, sk):
    P = np.array([-4.0, 104.0, 36.0])
    return _plier_job(S, sk, "crimper", P, (-0.2, 0.0, 1.0), 2, "crimp")


def j_route(S, sk):
    R, _ = rest_targets(S)
    clips = [np.array([x, 116.0, PZ - 1.0]) for x in (24.0, 4.0, -16.0)]
    keys = [(0.0, R["r"])]
    lkeys = [(0.0, R["l"])]
    t = 0.6
    for k, c in enumerate(clips):                                 # thumb / fingers press the cable into each clip
        over = K.pinch(sk, "r", c - Z * 2.5, approach=(0, -0.2, 1.0), palm=(0, -1, 0), grip="point")
        press = K.pinch(sk, "r", c - Z * 0.4, approach=(0, -0.2, 1.0), palm=(0, -1, 0), grip="point")
        keys += [(t + 0.6, over), (t + 0.85, press), (t + 1.05, over)]
        hold = K.power(sk, "l", c + X * 14.0 - Z * 4.0, -X, grip="power:0.8", weight=0.6)
        lkeys += [(t + 0.6, hold)]
        t += 1.1
    keys += [(t + 0.9, R["r"])]
    lkeys += [(t + 0.9, R["l"])]
    return K.keyed(t + 1.0, hands={"r": keys, "l": lkeys}, look=[(0.0, clips[0]), (t, clips[-1])], label="route cable")


# ---------------------------------------------------------------------------------------------- assembly
def j_hammer(S, sk):
    return G.hammer(S, (0.0, TOP + 9.0, 60.0), (0, 1, 0), "r", strikes=4)     # at arm's length: hand ~35 cm out


def j_mallet(S, sk):
    R, _ = rest_targets(S)
    P = np.array([-6.0, TOP + 2.0, 44.0])                          # cover plate to seat

    def T_lift(lift):
        u = K.unit(P - np.array([-18.0, 140.0, 0.0]) * np.array([1, 0, 1]) * 0 + np.array([0.0, 0.0, 0.0]))
        d = K.unit(np.array([-0.55, 0.0, -0.85]))                  # handle back towards the right hand
        T = K.T_at(np.zeros(3), -d, Y)
        T[:3, 3] = P - wdir(T, (30.0, -5.5, 0.0))                  # head face on the plate
        return K.rot_about(T, np.cross(Y, -d), lift, wpt(T, (6.5, 0.0, 0.0)))

    def rh(Td):
        T = Td["t"]
        return K.power(sk, "r", wpt(T, (6.5, 0.0, 0.0)), wdir(T, X), grip="power:1.5", lean=(0.0, -0.5, 1.0))
    tk = [(0.0, None), (1.0, T_lift(35.0))]
    t = 1.0
    for _ in range(3):
        tk += [(t + 0.3, T_lift(0.0)), (t + 0.38, T_lift(4.0)), (t + 0.7, T_lift(35.0))]
        t += 0.7
    tk, hk = tool_track(sk, rh, tk + [(t + 1.0, None)])
    br = K.brace(sk, "l", (20.0, TOP + 1.0, 44.0))
    return K.keyed(t + 1.1, hands={"r": hk, "l": [(0.0, R["l"]), (0.9, br), (t, br), (t + 1.0, R["l"])]},
                   tools={"t": ("mallet", tk)}, look=tuple(P), label="mallet")


def j_insert_pin(S, sk):
    R, _ = rest_targets(S)
    A = np.array([16.0, TOP + 1.0, 36.0])                          # pins in a tray (left), hole in a block (centre)
    B = np.array([-6.0, TOP + 8.0, 44.0])
    pin = "box:0.8x4.0x0.8:steel_machined"
    gA = K.pinch(sk, "r", A + Y * 2.0, approach=(0.3, -0.6, 1.0))
    over = K.pinch(sk, "r", B + Y * 7.0, approach=(0.1, -0.7, 1.0))
    seat = K.pinch(sk, "r", B + Y * 3.0, approach=(0.1, -0.7, 1.0))
    keys = [(0.0, R["r"]), (1.0, gA), (1.25, gA), (2.1, over), (2.6, seat), (2.8, seat), (3.6, R["r"])]
    pk = [(0.0, K.T_at(A, X, Y)), (1.25, K.T_at(A, X, Y)), (2.1, K.T_at(B + Y * 4.0, X, Y)), (2.6, K.T_at(B, X, Y))]
    br = K.brace(sk, "l", (24.0, TOP + 9.0, 44.0), toward=(0.1, 0, 1))
    return K.keyed(3.7, hands={"r": keys, "l": [(0.0, R["l"]), (1.5, br), (3.0, br), (3.7, R["l"])]},
                   parts={"part:pin": (pin, pk)}, look=[(0.0, A), (1.6, B)], label="insert pin")


def j_box(S, sk):
    size = (36.0, 22.0, 26.0)
    spec = "box:36x22x26:cardboard"
    pick = np.array([26.0, TOP + 11.0, 48.0])
    place_at = np.array([-28.0, TOP + 11.0, 48.0])
    f1, _, _ = C.carry_pick(S, pick, size, "box", spec, True, "r")
    f2 = C.carry_place(S, place_at, size, "box", spec, True, "r")
    return f1 + f2


def j_caulk(S, sk):
    R, _ = rest_targets(S)
    y = TOP + 6.0
    xs = (22.0, -22.0)

    def T_at(x, pull=0.0):
        b = K.unit(np.array([0.25, -0.75, 0.6]))                  # nozzle angled into the seam, dragged to -X
        T = K.T_at(np.zeros(3), b, np.cross(np.cross(b, Y), b))
        T[:3, 3] = np.array([x, y, 44.0])
        return T

    def rh(pull):
        def f(Td):
            T = Td["t"]
            return K.power(sk, "r", wpt(T, (-30.8, -6.0, 0.0)), wdir(T, Y), grip=f"trigger:1.5:{pull}",
                           lean=wdir(T, X))
        return f

    def lh(Td):
        T = Td["t"]
        return K.power(sk, "l", wpt(T, (-14.0, -2.6, 0.0)), -wdir(T, X), grip="power:2.4", weight=0.7)
    T0 = K.T_at(np.array([-22.0, 95.0, 22.0]), K.unit((0.2, -0.4, 1.0)), Y)
    tk = [(0.0, T0), (1.0, T_at(xs[0])), (4.2, T_at(xs[1])), (5.2, T0)]
    hand = [(0.0, rh(0.0)), (1.0, rh(0.0))]
    t = 1.0
    while t < 4.0:                                                # squeeze the trigger in strokes
        hand += [(t + 0.25, rh(1.0)), (t + 0.5, rh(0.2))]
        t += 0.5
    hand += [(5.2, rh(0.0))]
    return K.keyed(5.3, hands={"r": hand, "l": [(0.0, R["l"]), (0.9, lh), (4.3, lh), (5.2, R["l"])]},
                   tools={"t": ("caulk_gun", tk)}, look=[(0.0, (xs[0], y, 44)), (4.2, (xs[1], y, 44))], label="sealant")


# ---------------------------------------------------------------------------------------------- operate / inspect
def j_buttons(S, sk):
    R, _ = rest_targets(S)
    bs = [np.array([-8.0, 112.0, SZ - 1.2]), np.array([4.0, 112.0, SZ - 1.2]), np.array([16.0, 118.0, SZ - 2.5])]
    keys = [(0.0, R["r"])]
    t = 0.4
    for b in bs:
        over = K.pinch(sk, "r", b - Z * 3.0, approach=(0, -0.15, 1.0), palm=(0.7, -0.7, 0.0), grip="point")
        push = K.pinch(sk, "r", b - Z * 0.2 + Z * 0.8, approach=(0, -0.15, 1.0), palm=(0.7, -0.7, 0.0), grip="point")
        keys += [(t + 0.7, over), (t + 0.95, push), (t + 1.15, push), (t + 1.35, over)]
        t += 1.35
    keys += [(t + 0.9, R["r"])]
    return K.keyed(t + 1.0, hands={"r": keys, "l": [(0.0, R["l"])]}, look=[(0.0, bs[0]), (t - 1.0, bs[-1])],
                   label="buttons")


def j_handwheel(S, sk):
    R, _ = rest_targets(S)
    C0 = np.array([0.0, 112.0, PZ - 8.0])                         # wheel centre, axis -Z towards the worker
    rad = 13.0

    def on_rim(side, ang):
        a = math.radians(ang)
        p = C0 + (X * math.cos(a) + Y * math.sin(a)) * rad
        tang = (-X * math.sin(a) + Y * math.cos(a))
        return K.power(sk, side, p, tang if side == "r" else -tang, grip="power:1.3", lean=(0.0, 0.0, 1.0))
    keys_r, keys_l = [(0.0, R["r"])], [(0.0, R["l"])]
    t = 1.0
    r0, l0 = 160.0, 20.0
    keys_r.append((t, on_rim("r", r0)))
    keys_l.append((t, on_rim("l", l0)))
    for _ in range(3):                                            # turn 50 deg together, both regrip back
        keys_r += [(t + 0.8, on_rim("r", r0 - 50)), (t + 1.2, on_rim("r", r0))]
        keys_l += [(t + 0.8, on_rim("l", l0 - 50)), (t + 1.2, on_rim("l", l0))]
        t += 1.2
    keys_r.append((t + 0.9, R["r"]))
    keys_l.append((t + 0.9, R["l"]))
    return K.keyed(t + 1.0, hands={"r": keys_r, "l": keys_l}, look=tuple(C0), label="handwheel")


def j_scan(S, sk):
    R, _ = rest_targets(S)
    L = np.array([6.0, TOP + 14.0, 52.0])                         # label on a box on the bench
    aim = K.unit(L - np.array([-14.0, 118.0, 26.0]))

    def T_at(d):
        T = K.T_at(np.zeros(3), aim, np.cross(np.cross(aim, Y), aim))
        T[:3, 3] = L - aim * d
        return T

    def rh(pull):
        def f(Td):
            T = Td["t"]
            return K.power(sk, "r", wpt(T, (-6.8, -6.0, 0.0)), wdir(T, Y), grip=f"trigger:1.5:{pull}",
                           lean=wdir(T, X))
        return f
    T0 = K.T_at(np.array([-24.0, 90.0, 16.0]), K.unit((0.2, -0.3, 1.0)), Y)
    tk = [(0.0, T0), (1.0, T_at(22.0)), (2.6, T_at(22.0)), (3.6, T0)]
    hand = [(0.0, rh(0.0)), (1.2, rh(0.0)), (1.35, rh(1.0)), (2.0, rh(1.0)), (2.15, rh(0.0)), (3.6, rh(0.0))]
    br = K.brace(sk, "l", (24.0, TOP + 28.0, 52.0), normal=(0, 1, 0))
    return K.keyed(3.7, hands={"r": hand, "l": [(0.0, R["l"]), (1.0, br), (2.6, br), (3.6, R["l"])]},
                   tools={"t": ("scanner", tk)}, look=tuple(L), label="scan")


def j_label(S, sk):
    R, _ = rest_targets(S)
    A = np.array([20.0, TOP + 0.4, 36.0])                          # label sheet on the bench (right)
    F = np.array([-6.0, TOP + 14.0, 38.0])                         # box front face (z = 38, facing the worker)
    spec = "box:6x4x0.12:floor_white"
    peel = K.pinch(sk, "r", A + Y * 0.5, approach=(0.4, -0.7, 1.0))
    over = K.pinch(sk, "r", F - Z * 4.0, approach=(0, 0, 1.0), palm=(0, -1, 0))
    stick = K.pinch(sk, "r", F - Z * 0.5, approach=(0, 0, 1.0), palm=(0, -1, 0))
    wipe0 = K.brace(sk, "r", F + X * 3.0 - Z * 0.3, normal=(0, 0, -1), toward=(0, 1, 0), weight=0.8)
    wipe1 = K.brace(sk, "r", F - X * 3.0 - Z * 0.3, normal=(0, 0, -1), toward=(0, 1, 0), weight=0.8)
    keys = [(0.0, R["r"]), (1.0, peel), (1.3, peel), (2.1, over), (2.5, stick), (2.7, stick), (3.0, wipe0), (3.4, wipe1),
            (3.7, wipe0), (4.6, R["r"])]
    T_sheet = K.T_at(A, X, -Z)
    T_face = K.T_at(F - Z * 0.07, X, Y)
    pk = [(0.0, T_sheet), (1.3, T_sheet), (2.1, K.T_at(F - Z * 3.5, X, Y)), (2.5, T_face)]
    br = K.brace(sk, "l", (12.0, TOP + 28.0, 50.0))
    return K.keyed(4.7, hands={"r": keys, "l": [(0.0, R["l"]), (1.8, br), (3.9, br), (4.6, R["l"])]},
                   parts={"part:label": (spec, pk)}, look=[(0.0, A), (1.6, F)], label="label")


def j_wipe(S, sk):
    R, _ = rest_targets(S)
    c = np.array([-4.0, TOP, 50.0])

    def at(x, z):
        return K.brace(sk, "r", c + np.array([x, 1.2, z]), toward=(0.0, 0.0, 1.0), weight=0.9)

    def rag(Td):
        return None
    keys = [(0.0, R["r"]), (0.9, at(16, -6))]
    pts = [(-20, -6), (-20, 4), (16, 4), (16, 12), (-20, 12), (-20, -6), (16, -6)]
    t = 0.9
    for x, z in pts:
        t += 0.45
        keys.append((t, at(x, z)))
    keys.append((t + 0.9, R["r"]))
    rk = [(tk, K.T_at(np.asarray(h.point) - np.asarray(h.palm) * 0.0 - Y * 1.1, K.unit(h.fingers), Y)) if h is not None else (tk, None)
          for tk, h in keys[1:-1]]
    br = K.brace(sk, "l", (24.0, TOP, 36.0))
    return K.keyed(t + 1.0, hands={"r": keys, "l": [(0.0, R["l"]), (0.9, br), (t, br), (t + 0.9, R["l"])]},
                   parts={"part:rag": ("box:12x1.2x10:fabric_navy", [(keys[1][0] - 0.3, rk[0][1])] + rk)},
                   look=tuple(c), label="wipe")


def j_checklist(S, sk):
    R, _ = rest_targets(S)
    B = np.array([-4.0, TOP + 0.8, 42.0])                         # clipboard on the bench
    d = K.unit(np.array([-0.35, 0.85, -0.45]))                     # pen leaning back over the right hand

    def pen_at(u, v, lift=0.0):
        tip = B + X * u + Z * v + Y * (0.3 + lift)
        return K.T_at(tip, np.cross(d, X), d)

    def rh(Td):
        T = Td["t"]
        return K.pinch(sk, "r", wpt(T, (0.0, 2.6, 0.0)), approach=(0.15, -0.55, 1.0), palm=(0.2, -1, 0.0), grip="tripod")
    T0 = K.T_at(np.array([-24.0, 92.0, 16.0]), X, Y)
    tk = [(0.0, T0), (1.0, pen_at(6, 6, 1.0)), (1.2, pen_at(6, 6))]
    t = 1.2
    for row in range(3):                                            # initials on three lines
        for k in range(5):
            tk.append((t + 0.11, pen_at(6 - k * 1.3, 6 - row * 4 + (0.8 if k % 2 else 0.0))))
            t += 0.11
        tk += [(t + 0.2, pen_at(6, 2 - row * 4, 0.8)), (t + 0.35, pen_at(6, 2 - row * 4))]
        t += 0.35
    tk += [(t + 1.0, T0)]
    br = K.brace(sk, "l", B + X * 14.0 + Z * 2.0, toward=(0.0, 0.0, 1.0))
    return K.keyed(t + 1.1, hands={"r": [(0.0, rh)], "l": [(0.0, R["l"]), (0.9, br), (t, br), (t + 1.0, R["l"])]},
                   tools={"t": ("pen", tk)}, look=tuple(B), label="checklist")


# ---------------------------------------------------------------------------------------------- the library
def _f_bench(extra=None):
    def f(s):
        bench(s)
        if extra:
            extra(s)
    f.kind = "bench"
    return f


def _f_panel(extra=None, z=PZ):
    def f(s):
        panel(s, z=z)
        if extra:
            extra(s)
    f.kind = "panel"
    f.z = z
    return f


def _bolts(s):
    block(s, (-4.0, TOP + 2.5, 44.0), (16.0, 5.0, 12.0))
    bolt_up(s, (-4.0, TOP + 5.0, 44.0))


def _stud(s):
    block(s, (-6.0, TOP + 1.5, 42.0), (14.0, 3.0, 12.0))
    s.cylinder((-6.0, TOP + 3.0, 42.0), (-6.0, TOP + 7.0, 42.0), 0.5, "steel_machined", 10)


def _hexsock(s):
    block(s, (-4.0, TOP + 0.5, 42.0), (18.0, 1.0, 12.0), "powder_dark")


def _wrench_nut(s):
    block(s, (-4.0, TOP + 1.5, 42.0), (14.0, 3.0, 12.0))
    s.lathe((-4.0, TOP + 3.0, 42.0), (-4.0, TOP + 4.6, 42.0), [(0, 1.0), (1, 1.0)], "bolt_zinc", sides=6)


def _screw(s):
    s.rbox((0.0, 100.0, SZ - 0.4), (14.0, 10.0, 0.8), 0.4, "powder_dark", seg=1)


def _impact_hole(s):
    s.rbox((-6.0, 106.0, PZ - 0.5), (10.0, 10.0, 1.0), 0.4, "powder_dark", seg=1)


def _receptacle(s):
    s.rbox((-4.0, 112.0, PZ - 1.5), (5.0, 3.5, 3.0), 0.5, "plastic_black", seg=1)
    s.sweep([(-4.0, 112.0, PZ - 3.0), (-12.0, 112.0, PZ - 4.0), (-30.0, 118.0, PZ - 1.0)], 0.6, "cable_pvc", sides=8)


def _socket(s):
    s.rbox((-4.0, 98.0, PZ - 0.6), (8.0, 8.0, 1.2), 1.0, "nylon_white", seg=2)
    for x in (-0.95, 0.95):
        s.obox((-4.0 + x, 98.0, PZ - 1.25), (0.5, 0.9, 0.1), ((1, 0, 0), (0, 1, 0), (0, 0, 1)), "hole")


def _harness(s):
    s.sweep([(-40.0, 108.0, PZ - 4.0), (40.0, 108.0, PZ - 4.0)], 1.0, "cable_pvc", sides=10)
    for x in (-30.0, 30.0):
        s.rbox((x, 108.0, PZ - 2.0), (2.0, 3.0, 4.0), 0.3, "plastic_black", seg=1)


def _clips(s):
    for x in (24.0, 4.0, -16.0):
        s.rbox((x, 116.0, PZ - 0.8), (2.4, 2.4, 1.6), 0.4, "plastic_black", seg=1)
    s.sweep([(40.0, 112.0, PZ - 1.5), (24.0, 116.0, PZ - 1.2), (-16.0, 116.0, PZ - 1.2), (-34.0, 110.0, PZ - 2.0)], 0.45,
            "cable_pvc", sides=8)


def _wire(s):
    s.sweep([(10.0, 104.0, 38.0), (-6.0, 104.0, 38.0)], 0.3, "cable_pvc", sides=8)


def _pin_block(s):
    block(s, (-6.0, TOP + 4.0, 44.0), (12.0, 8.0, 12.0))
    s.rbox((16.0, TOP + 0.6, 36.0), (10.0, 1.2, 7.0), 0.3, "plastic_blue", seg=1)


def _anvil(s):
    block(s, (0.0, TOP + 4.0, 60.0), (12.0, 8.0, 12.0))
    s.lathe((0.0, TOP + 8.0, 60.0), (0.0, TOP + 9.0, 60.0), [(0, 0.6), (1, 0.6)], "steel_machined", sides=12)


def _plate(s):
    s.rbox((-6.0, TOP + 1.0, 44.0), (24.0, 2.0, 18.0), 0.4, "powder_grey", seg=1)


def _seam(s):
    s.rbox((0.0, TOP + 3.0, 50.0), (60.0, 6.0, 10.0), 0.4, "powder_grey", seg=1)


def _controls(s):
    s.rbox((4.0, 114.0, SZ - 2.0), (40.0, 20.0, 4.0), 1.0, "abs_lightgrey", seg=2)


def _wheel(s):
    s.torus((0.0, 112.0, PZ - 8.0), (0, 0, 1), 13.0, 1.2, "powder_dark", seg=32, sides=10)
    s.cylinder((0.0, 112.0, PZ - 8.0), (0.0, 112.0, PZ), 1.5, "steel_machined", 12)
    for k in range(3):
        a = 2 * math.pi * k / 3
        s.cylinder((0.0, 112.0, PZ - 8.0), (13.0 * math.cos(a), 112.0 + 13.0 * math.sin(a), PZ - 8.0), 0.6,
                   "stainless", 8)


def _scan_box(s):
    s.rbox((6.0, TOP + 12.0, 56.0), (30.0, 24.0, 22.0), 0.6, "cardboard", seg=2)
    s.rbox((6.0, TOP + 14.0, 44.9), (8.0, 4.0, 0.1), 0.1, "floor_white", seg=1)


def _clipboard(s):
    s.rbox((-4.0, TOP + 0.4, 44.0), (23.0, 0.8, 31.0), 0.3, "wood_top", seg=1)
    s.rbox((-4.0, TOP + 0.85, 43.0), (21.0, 0.1, 28.0), 0.05, "label_vinyl", seg=1)
    s.rbox((-4.0, TOP + 1.3, 58.0), (8.0, 1.0, 2.5), 0.3, "steel_machined", seg=1)


def _label_box(s):
    s.rbox((-6.0, TOP + 12.0, 50.0), (30.0, 24.0, 24.0), 0.6, "cardboard", seg=2)
    s.rbox((20.0, TOP + 0.2, 36.0), (9.0, 0.4, 7.0), 0.1, "label_vinyl", seg=1)


JOBS = [
    Job("hand_nut", "Hand-start nut", "fastening", "W05F", j_hand_nut, _f_bench(_stud), (-6, 97, 42)),
    Job("ratchet", "Ratchet a bolt", "fastening", "W50M", j_ratchet, _f_bench(_bolts), (-4, 98, 42)),
    Job("torque", "Torque to click", "fastening", "W95M", j_torque, _f_bench(_bolts), (-4, 98, 42)),
    Job("screwdriver", "Drive a screw", "fastening", "W05F", j_screwdriver, _f_panel(_screw, z=SZ), (0, 100, 52)),
    Job("impact", "Impact driver", "fastening", "W50M", j_impact, _f_panel(_impact_hole), (-6, 104, 36)),
    Job("hex_key", "Hex key", "fastening", "W95M", j_hex_key, _f_bench(_hexsock), (-4, 95, 42)),
    Job("wrench", "Combination wrench", "fastening", "W05F", j_wrench, _f_bench(_wrench_nut), (-4, 96, 40)),
    Job("connector", "Mate a connector", "electrical", "W50M", j_connector, _f_panel(_receptacle), (-4, 112, 38)),
    Job("unplug", "Unplug a connector", "electrical", "W95M", j_unplug, _f_panel(_receptacle), (-4, 112, 38)),
    Job("plug", "Mains plug in", "electrical", "W05F", j_plug, _f_panel(_socket), (-4, 98, 38)),
    Job("cable_tie", "Cable tie", "electrical", "W50M", j_cable_tie, _f_panel(_harness), (0, 108, 36)),
    Job("cut_tie", "Cut tie tail", "electrical", "W95M", j_cut_tie, _f_panel(_harness), (0, 110, 36)),
    Job("strip", "Strip a wire", "electrical", "W05F", j_strip, _f_bench(_wire), (-4, 104, 36)),
    Job("crimp", "Crimp a terminal", "electrical", "W50M", j_crimp, _f_bench(_wire), (-4, 104, 34)),
    Job("route", "Route cable in clips", "electrical", "W95M", j_route, _f_panel(_clips), (4, 116, 38)),
    Job("hammer", "Hammer a pin", "assembly", "W50M", j_hammer, _f_bench(_anvil), (0, 102, 56)),
    Job("mallet", "Seat a cover", "assembly", "W05F", j_mallet, _f_bench(_plate), (-6, 96, 44)),
    Job("insert_pin", "Insert a pin", "assembly", "W95M", j_insert_pin, _f_bench(_pin_block), (2, 98, 40)),
    Job("box", "Move a box", "assembly", "W50M", j_box, _f_bench(), (0, 104, 48)),
    Job("caulk", "Sealant bead", "assembly", "W95M", j_caulk, _f_bench(_seam), (0, 98, 46)),
    Job("buttons", "Push buttons", "operate", "W05F", j_buttons, _f_panel(_controls, z=SZ), (4, 114, 52)),
    Job("handwheel", "Turn a handwheel", "operate", "W50M", j_handwheel, _f_panel(_wheel), (0, 112, 32)),
    Job("scan", "Scan a barcode", "operate", "W95M", j_scan, _f_bench(_scan_box), (2, 104, 40)),
    Job("label", "Apply a label", "operate", "W05F", j_label, _f_bench(_label_box), (4, 102, 38)),
    Job("wipe", "Wipe down", "operate", "W50M", j_wipe, _f_bench(), (-4, 94, 50)),
    Job("checklist", "Sign the checklist", "operate", "W95M", j_checklist, _f_bench(_clipboard), (0, 96, 44)),
]
from .jobs_awp import JOBS_AWP  # noqa: E402  (aerial work platform assembly: hydraulics, overhead, heavy, QC)

JOBS = JOBS + JOBS_AWP
BY_ID = {j.id: j for j in JOBS}
