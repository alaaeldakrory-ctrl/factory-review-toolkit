"""More motion generators: pick & place, connector push-to-click, torque click, screwdriver, inspect, idle.
Same contract as motion_gen: worker frame in, list of Frame out."""
import numpy as np

from .motion_gen import FPS, Frame, ease, lerp_T, lerp_target, ratchet, rest_targets, tool_from_hand
from .solver import HandTarget
from .tools import TOOLS, place


def _hold(targets, look, n, tools=None, label=""):
    return [Frame(list(targets), look, dict(tools or {}), static=(k > 0), label=label) for k in range(n)]


def _palm_down_at(side, p, fwd=(0, 0, 1), grip="pinch"):
    return HandTarget(side, tuple(p), fingers=tuple(fwd), palm=(0.0, -1.0, 0.0), grip=grip)


def pick_place(solver, pick, place_at, side="r", two_hands=False, width=30.0, grip="pinch"):
    """Get a part at `pick` and place it at `place_at` (MTM-UAS get & place): reach, grasp, move (lift, carry over,
    lower), position, release, return. two_hands: grasp a box-like part from both sides (`width` apart)."""
    rest, _ = rest_targets(solver)
    pick, place_at = np.asarray(pick, float), np.asarray(place_at, float)

    def hands_at(p):
        if not two_hands:
            return [_palm_down_at(side, p + np.array([0, 3.0, 0]), grip=grip)]
        return [HandTarget("l", tuple(p + np.array([width / 2 + 1.5, 4.0, 0])), fingers=(0, -0.3, 1), palm=(-1, 0, 0),
                           grip="flat"),
                HandTarget("r", tuple(p + np.array([-width / 2 - 1.5, 4.0, 0])), fingers=(0, -0.3, 1), palm=(1, 0, 0),
                           grip="flat")]
    frames = []
    nr = int(0.8 * FPS)
    a = hands_at(pick)
    for k in range(nr):
        u = ease((k + 1) / nr)
        frames.append(Frame([lerp_target(rest[t.side], t, u) for t in a], tuple(pick), label="reach"))
    frames += _hold(a, tuple(pick), int(0.25 * FPS), label="grasp")
    mid = (pick + place_at) / 2 + np.array([0, 12.0, 0])
    nm = int(max(0.6, np.linalg.norm(place_at - pick) / 80.0) * FPS)
    for k in range(nm):
        u = ease((k + 1) / nm)
        p = (1 - u) ** 2 * pick + 2 * (1 - u) * u * mid + u * u * place_at
        frames.append(Frame(hands_at(p), tuple(place_at), label="move"))
    frames += _hold(hands_at(place_at), tuple(place_at), int(0.3 * FPS), label="position")
    b = hands_at(place_at)
    for k in range(nr):
        u = ease((k + 1) / nr)
        frames.append(Frame([lerp_target(t, rest[t.side], u) for t in b], tuple(place_at), label="return"))
    return frames


def connector(solver, at, axis=(0, 0, 1), side="r"):
    """Mate a connector: bring the half up short of the mating face, push home with a small overshoot at the
    click, tug test, let go (the connector stays mated). The other hand steadies the mating half."""
    tl = TOOLS["connector"]
    rest, _ = rest_targets(solver)
    P = np.asarray(at, float)
    n = np.asarray(axis, float)
    n = n / np.linalg.norm(n)
    up = np.array([0.0, 1.0, 0.0]) if abs(n[1]) < 0.9 else np.array([0.0, 0.0, 1.0])

    def T_at(back):
        x = np.cross(up, n)
        T = place(np.zeros(3), x, up)
        T[:3, 3] = P - n * back - T[:3, :3] @ np.asarray(tl.work["mate"], float)
        return T
    other = "l" if side == "r" else "r"
    sx = 1.0 if other == "l" else -1.0
    steady = HandTarget(other, tuple(P + n * 3.0 + np.array([sx * 3.0, 0.5, 0])), fingers=tuple(-n),
                        palm=(-sx, 0, 0), grip="pinch", weight=0.5)
    T_rest = tool_from_hand(tl, rest[side])
    frames = []
    nr = int(0.8 * FPS)
    for k in range(nr):
        u = ease((k + 1) / nr)
        T = lerp_T(T_rest, T_at(4.0), u)
        frames.append(Frame([tl.hand_target(T, side), lerp_target(rest[other], steady, u)], tuple(P),
                            {"connector": ("connector", T)}, label="align"))
    cur = 4.0
    for back, dur in [(4.0, 0.25), (1.2, 0.35), (-0.25, 0.10), (0.0, 0.12), (0.0, 0.2), (0.8, 0.12), (0.0, 0.12)]:
        nn = max(2, int(dur * FPS))
        for k in range(nn):
            T = T_at(cur + (back - cur) * ease((k + 1) / nn))
            frames.append(Frame([tl.hand_target(T, side), steady], tuple(P), {"connector": ("connector", T)},
                                label="push" if back < 2 else "tug"))
        cur = back
    T_home = T_at(0.0)
    for k in range(nr):
        u = ease((k + 1) / nr)
        frames.append(Frame([lerp_target(tl.hand_target(T_home, side), rest[side], u), lerp_target(steady, rest[other], u)],
                            tuple(P), {"connector": ("connector", T_home)}, label="release"))
    return frames


def torque(solver, bolt, axis=(0, 1, 0), side="r", torque_nm=70.0):
    """Final torque with a click wrench: one slow steady pull to the click, hold, release."""
    frames = ratchet(solver, bolt, axis, side, strokes=1, swing=35.0, period=2.2, tool="torque_wrench")
    for f in frames:
        f.label = "torque"
    return frames


def screwdriver(solver, screw, axis=(0, 0, 1), side="r", turns=6):
    """Drive a screw: tip on the screw, ~110 deg twists with a regrip between."""
    tl = TOOLS["screwdriver"]
    rest, _ = rest_targets(solver)
    P = np.asarray(screw, float)
    n = np.asarray(axis, float)
    n = n / np.linalg.norm(n)
    ref = np.array([0.0, 1.0, 0.0]) if abs(n[1]) < 0.9 else np.array([0.0, 0.0, 1.0])

    def T_at(rot):
        x = np.cross(ref, -n)
        x = x / np.linalg.norm(x)
        c, s_ = np.cos(np.radians(rot)), np.sin(np.radians(rot))
        x2 = x * c + np.cross(-n, x) * s_
        T = place(np.zeros(3), x2, -n)
        T[:3, 3] = P
        return T
    T_rest = tool_from_hand(tl, rest[side])
    frames = []
    nr = int(0.8 * FPS)
    for k in range(nr):
        T = lerp_T(T_rest, T_at(0.0), ease((k + 1) / nr))
        frames.append(Frame([tl.hand_target(T, side)], tuple(P), {"screwdriver": ("screwdriver", T)}, label="reach"))
    per = int(0.7 * FPS)
    for _ in range(turns):
        for k in range(per):
            u = k / per
            rot = 110.0 * ease(u / 0.65) if u < 0.65 else 110.0 * (1 - ease((u - 0.65) / 0.35))
            T = T_at(rot)
            frames.append(Frame([tl.hand_target(T, side)], tuple(P), {"screwdriver": ("screwdriver", T)}, label="drive"))
    for k in range(nr):
        T = lerp_T(T_at(0.0), T_rest, ease((k + 1) / nr))
        frames.append(Frame([tl.hand_target(T, side)], tuple(P), {"screwdriver": ("screwdriver", T)}, label="return"))
    return frames


def inspect(solver, point, seconds=2.0, side="r"):
    """Visual control: look at the point and point at it."""
    rest, _ = rest_targets(solver)
    P = np.asarray(point, float)
    src = P + np.array([0, 4.0, -12.0])
    d = (P - src) / np.linalg.norm(P - src)
    tgt = HandTarget(side, tuple(src), fingers=tuple(d), palm=(0, -1, 0), grip="point", weight=0.5)
    n = int(0.6 * FPS)
    frames = [Frame([lerp_target(rest[side], tgt, ease((k + 1) / n))], tuple(P), label="inspect") for k in range(n)]
    frames += _hold([tgt], tuple(P), int(seconds * FPS), label="inspect")
    frames += [Frame([lerp_target(tgt, rest[side], ease((k + 1) / n))], tuple(P), label="inspect") for k in range(n)]
    return frames


def idle(seconds, look=None, label="wait"):
    return [Frame([], look, static=(k > 0), label=label) for k in range(max(1, int(seconds * FPS)))]
