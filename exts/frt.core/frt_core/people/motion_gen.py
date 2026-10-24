"""Motion generators: a task in, per-frame specs out (hand targets from tools, look point, tool placements).

Every generator works in the worker frame (Y up, facing +Z) and returns a list of Frame. The baker (bake.py) solves
each frame with the posture solver (warm-started from the previous one) and writes the skeleton and tool animation.

Phases follow the MTM-UAS logic of the catalog: reach / get tool -> position -> act (motion cycle) -> return.
"""
import math
from dataclasses import dataclass, field

import numpy as np

from .solver import HandTarget
from .tools import TOOLS, place

FPS = 24.0


@dataclass
class Frame:
    targets: list
    look: tuple = None
    tools: dict = field(default_factory=dict)     # instance name -> (tool key, 4x4 tool -> worker)
    root: tuple = None                            # (x, z, yaw) in the world while walking; None = at the work spot
    stance: object = None                         # solver.Stance in the root frame (walking feet)
    swing: dict = None                            # free-arm swing (cm forward per side) while walking
    static: bool = False                          # same as the previous frame (waiting): reuse its solution
    label: str = ""                               # step / phase name (timeline, findings)
    base: object = None                           # full-body base pose (mocap) the solver works on top of
    world: object = None                          # 4x4: this frame's worker frame -> the scene collision field


def ease(u):
    """Minimum-jerk profile (Flash & Hogan): the bell-shaped speed of real human reaching movements."""
    u = min(1.0, max(0.0, u))
    return u * u * u * (10 - 15 * u + 6 * u * u)


def ease_in(u):                                   # accelerating (a strike)
    u = min(1.0, max(0.0, u))
    return u * u


def ease_out(u):                                  # decelerating (a lift)
    u = min(1.0, max(0.0, u))
    return 1 - (1 - u) * (1 - u)


def _frame_of(t):
    """4x4 frame of a hand target: X = fingers, Y = palm normal, origin = palm centre."""
    f = np.asarray(t.fingers, float)
    f = f / np.linalg.norm(f)
    p = np.asarray(t.palm, float)
    p = p - f * np.dot(p, f)
    p = p / np.linalg.norm(p)
    M = np.eye(4)
    M[:3, 0], M[:3, 1], M[:3, 2], M[:3, 3] = f, p, np.cross(f, p), t.point
    return M


def _grip_frame(g):
    return _frame_of(HandTarget("r", g.point, g.fingers, g.palm))


def tool_from_hand(tool, hand_target):
    """Tool transform when the hand (at hand_target) holds the tool by its grip."""
    return _frame_of(hand_target) @ np.linalg.inv(_grip_frame(tool.grip))


def lerp_T(A, B, u):
    """Blend two rigid transforms: positions linearly, rotations by turning steadily about the one axis between them
    (a matrix lerp + re-orthogonalise flips suddenly for turns past ~90 deg - a tool put away spun in a frame)."""
    M = np.eye(4)
    M[:3, 3] = A[:3, 3] * (1 - u) + B[:3, 3] * u
    Ra, Rb = A[:3, :3], B[:3, :3]
    D = Ra.T @ Rb
    c = max(-1.0, min(1.0, (np.trace(D) - 1.0) / 2.0))
    th = math.acos(c)
    if th < 1e-6:
        M[:3, :3] = Ra
        return M
    w = np.array([D[2, 1] - D[1, 2], D[0, 2] - D[2, 0], D[1, 0] - D[0, 1]])
    if np.linalg.norm(w) < 1e-6:                          # (half a turn: the axis from the symmetric part)
        k = int(np.argmax(np.diag(D)))
        w = (D[:, k] + np.eye(3)[:, k])
    ax = w / np.linalg.norm(w)
    K_ = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    a = th * u
    M[:3, :3] = Ra @ (np.eye(3) + math.sin(a) * K_ + (1 - math.cos(a)) * K_ @ K_)
    return M


CLOSING = ("power", "pinch", "tripod", "lateral", "hook", "trigger", "point")


def grip_blend(ga, gb, u):
    """Hand shape between two grips. Reaching for something to hold, the hand opens on the way (pre-shaping) and
    closes on contact - as people do."""
    if ga == gb:
        return ga
    if ga == "open" or gb == "open":                       # already open (or opening fully): one smooth change
        return ("blend", ga, gb, ease(u))
    if str(gb).split(":")[0] in CLOSING and not (isinstance(gb, tuple)) and str(ga).split(":")[0] not in CLOSING:
        if u < 0.7:
            return ("blend", ga, "open", ease(u / 0.7))
        return ("blend", "open", gb, ease((u - 0.7) / 0.3))
    if str(ga).split(":")[0] in CLOSING and not isinstance(ga, tuple) and str(gb).split(":")[0] not in CLOSING:
        if u < 0.4:                                             # letting go: open first, then relax
            return ("blend", ga, "open", ease(u / 0.4))
        return ("blend", "open", gb, ease((u - 0.4) / 0.6))
    return ("blend", ga, gb, u)


def _hand_frame(t):
    f = np.asarray(t.fingers, float)
    f = f / (np.linalg.norm(f) or 1.0)
    p = np.asarray(t.palm, float)
    p = p - f * np.dot(p, f)
    p = p / (np.linalg.norm(p) or 1.0)
    return np.column_stack([f, p, np.cross(f, p)])


def lerp_target(a, b, u, grip=None):
    """Hand target between two others: position linear, orientation by a true rotation interpolation (slerp of
    the hand frame - never through a degenerate in-between that would flip the hand), hand shape blended."""
    from .bake import _R, _q, _slerp
    Fa, Fb = _hand_frame(a), _hand_frame(b)
    c = max(-1.0, min(1.0, (np.trace(Fa.T @ Fb) - 1.0) / 2.0))
    if math.degrees(math.acos(c)) > 120.0:
        # near half a turn the shortest rotation is ambiguous and its middle can be a hand the wrist cannot make
        # (overhead palm-up back down to the side): turn through a natural hand instead - fingers forward, palm in
        sx = 1.0 if a.side == "l" else -1.0
        N = _hand_frame(HandTarget(a.side, a.point, (0.0, -0.45, 1.0), (-sx, 0.0, 0.0), a.grip, a.weight))
        Fu = _R(_slerp(_q(Fa), _q(N), u * 2.0)) if u < 0.5 else _R(_slerp(_q(N), _q(Fb), u * 2.0 - 1.0))
    else:
        Fu = _R(_slerp(_q(Fa), _q(Fb), u))
    pt = np.asarray(a.point, float) * (1 - u) + np.asarray(b.point, float) * u
    return HandTarget(a.side, tuple(pt), tuple(Fu[:, 0]), tuple(Fu[:, 1]),
                      grip or grip_blend(a.grip, b.grip, u), a.weight * (1 - u) + b.weight * u)


def _rest_disk(solver):
    """The free standing solve for this body, kept on disk by body and solver code (a cold solve: ~4 s a task)."""
    import hashlib
    import os
    import pickle
    sk = solver.sk
    h = hashlib.sha1()
    for j in sorted(sk.world):
        h.update(j.encode() + np.asarray(sk.world[j], float).tobytes())
    here = os.path.dirname(os.path.abspath(__file__))
    for f in ("solver.py", "kinematics.py", "collide.py"):
        h.update(open(os.path.join(here, f), "rb").read())
    d = os.path.join(here, "..", "..", "..", "..", "out", "people", "cache")
    path = os.path.join(d, f"rest_{h.hexdigest()[:16]}.pkl")
    try:
        return pickle.load(open(path, "rb"))
    except Exception:
        pass
    sol = solver.solve([], look=None)
    try:
        os.makedirs(d, exist_ok=True)
        pickle.dump(sol, open(path + ".tmp", "wb"), protocol=4)
        os.replace(path + ".tmp", path)
    except OSError:                                                 # (read-only install: just solve each time)
        pass
    return sol


def rest_targets(solver):
    """Where each hand is when the worker stands at ease (from a free solve): HandTargets to blend from / back to.
    Solved once per solver (every motion builder asks for it: it was most of a task's compile time)."""
    from .kinematics import fk
    memo = getattr(solver, "_rest_memo", None)
    if memo is not None:
        return dict(memo[0]), memo[1]
    sol = _rest_disk(solver)
    R, P = fk(solver.sk, sol.pose)
    out = {}
    for s, sx in (("l", 1.0), ("r", -1.0)):
        Rh = R[f"hand_{s}"]
        fingers = Rh @ np.array([sx, 0.0, 0.0])
        palm = Rh @ np.array([0.0, -1.0, 0.0])
        pt = P[f"hand_{s}"] + fingers * solver.hand_len * 0.40 + palm * 1.6
        pt = pt + np.array([0.0, 7.0, 6.0])                      # hand at the hip, elbow eased: always reachable
        out[s] = HandTarget(s, tuple(pt), tuple(fingers), tuple(palm), "relaxed", 0.3)
    solver._rest_memo = (dict(out), sol)
    return out, sol


# ---------------------------------------------------------------------------------------------- ratchet
def ratchet(solver, bolt, axis=(0, 1, 0), side="r", strokes=8, swing=50.0, period=0.85, tool="ratchet_38",
            handle_hint=None, brace=True, reach=0.9):
    """Tighten a bolt with a ratchet: bring the tool from the hip onto the bolt, short power strokes and quicker
    returns over `swing` degrees, support hand cupping the head, then back to the hip."""
    tl = TOOLS[tool]
    rest, _ = rest_targets(solver)
    n = np.asarray(axis, float)
    n = n / np.linalg.norm(n)
    P = np.asarray(bolt, float)
    sx = 1.0 if side == "l" else -1.0
    # handle across the body on the working hand's side, so the forearm runs in line with the fingers
    hint = np.asarray(handle_hint if handle_hint is not None else (sx * 1.0, 0.0, -0.25), float)
    d0 = hint - n * np.dot(hint, n)
    d0 = d0 / np.linalg.norm(d0)
    other = "l" if side == "r" else "r"

    def T_at(theta):
        c, s_ = math.cos(math.radians(theta)), math.sin(math.radians(theta))
        d = d0 * c + np.cross(n, d0) * s_
        return place(P + n * 3.0, d, n)                    # tool X = handle, tool Y = bolt axis out of the work
    T_rest = tool_from_hand(tl, rest[side])
    frames = []
    nr = int(reach * FPS)
    for k in range(nr):                                    # reach: tool from the hip onto the bolt
        u = ease((k + 1) / nr)
        T = lerp_T(T_rest, T_at(-swing / 2), u)
        tg = [tl.hand_target(T, side)]
        if brace and u > 0.5:
            tg.append(lerp_target(rest[other], tl.hand_target(T_at(-swing / 2), other, "grip2", 0.6), ease((u - 0.5) * 2)))
        frames.append(Frame(tg, tuple(P), {tool: (tool, T)}))
    per = int(period * FPS)
    for i in range(strokes):                               # power stroke (60 %) then a quicker return
        for k in range(per):
            u = k / per
            th = -swing / 2 + swing * (ease(u / 0.6) if u < 0.6 else 1 - ease((u - 0.6) / 0.4))
            T = T_at(th)
            tg = [tl.hand_target(T, side)]
            if brace:
                tg.append(tl.hand_target(T, other, "grip2", 0.6))
            frames.append(Frame(tg, tuple(P), {tool: (tool, T)}))
    for k in range(nr):                                    # back to the hip
        u = ease((k + 1) / nr)
        T = lerp_T(T_at(-swing / 2), T_rest, u)
        tg = [tl.hand_target(T, side)]
        if brace and u < 0.5:
            tg.append(lerp_target(tl.hand_target(T_at(-swing / 2), other, "grip2", 0.6), rest[other], ease(u * 2)))
        frames.append(Frame(tg, tuple(P), {tool: (tool, T)}))
    return frames


# ---------------------------------------------------------------------------------------------- hammer
def hammer(solver, point, normal=(0, 1, 0), side="r", strikes=4, period=0.9, lift=30.0, hold=True, reach=0.9):
    """Strike a point (a pin, a part to seat): raise - accelerate down - impact - short rebound, support hand holding
    the work beside the strike point."""
    tl = TOOLS["hammer"]
    rest, _ = rest_targets(solver)
    S = np.asarray(point, float)
    nn = np.asarray(normal, float)
    nn = nn / np.linalg.norm(nn)
    sx = 1.0 if side == "l" else -1.0
    sh = np.array([sx * 18.0, 140.0, 0.0])                 # roughly the shoulder: the swing comes from here
    u_dir = S - sh
    u_dir = u_dir - nn * np.dot(u_dir, nn)
    u_dir = u_dir / np.linalg.norm(u_dir)                  # handle direction at impact: away from the worker
    face = np.asarray(tl.work["face"], float)

    T0 = place(np.zeros(3), u_dir, nn)
    T0[:3, 3] = S - T0[:3, :3] @ face                       # impact: face on the strike point, handle away from us
    g0 = T0[:3, :3] @ np.asarray(tl.grip.point, float) + T0[:3, 3]
    a_ax = np.cross(nn, u_dir)                               # swing axis (horizontal, across the swing)

    def T_strike(lift_deg):
        """Hammer cocked by lift_deg about the wrist (head up and back), the hand raised and drawn back with it."""
        th = -math.radians(lift_deg)
        K = np.array([[0, -a_ax[2], a_ax[1]], [a_ax[2], 0, -a_ax[0]], [-a_ax[1], a_ax[0], 0]])
        R = np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K
        T = np.eye(4)
        T[:3, :3] = R @ T0[:3, :3]
        g = g0 + nn * lift_deg * 0.22 - u_dir * lift_deg * 0.02          # the forearm rises, the wrist cocks
        T[:3, 3] = g - T[:3, :3] @ np.asarray(tl.grip.point, float)
        return T
    other = "l" if side == "r" else "r"
    hold_t = HandTarget(other, tuple(S + nn * 2.5 - u_dir * 9.0 + np.cross(nn, u_dir) * sx * 4.0),
                        fingers=tuple(u_dir), palm=tuple(-nn), grip="pinch", weight=0.6) if hold else None
    T_rest = tool_from_hand(tl, rest[side])
    T_up = T_strike(65.0)
    frames = []
    nr = int(reach * FPS)
    for k in range(nr):
        u = ease((k + 1) / nr)
        T = lerp_T(T_rest, T_up, u)
        tg = [tl.hand_target(T, side)]
        if hold_t:
            tg.append(lerp_target(rest[other], hold_t, u))
        frames.append(Frame(tg, tuple(S), {"hammer": ("hammer", T)}))
    per = int(period * FPS)
    for i in range(strikes):
        for k in range(per):
            u = k / per
            if u < 0.45:                                    # downswing, accelerating
                lift = 65.0 * (1 - ease_in(u / 0.45))
            elif u < 0.55:                                  # impact and rebound
                lift = 6.0 * math.sin(math.pi * (u - 0.45) / 0.1)
            else:                                           # raise again, decelerating
                lift = 65.0 * ease_out((u - 0.55) / 0.45)
            T = T_strike(lift)
            tg = [tl.hand_target(T, side)]
            if hold_t:
                tg.append(hold_t)
            frames.append(Frame(tg, tuple(S), {"hammer": ("hammer", T)}))
    for k in range(nr):
        u = ease((k + 1) / nr)
        T = lerp_T(T_up, T_rest, u)
        tg = [tl.hand_target(T, side)]
        if hold_t:
            tg.append(lerp_target(hold_t, rest[other], u))
        frames.append(Frame(tg, tuple(S), {"hammer": ("hammer", T)}))
    return frames
