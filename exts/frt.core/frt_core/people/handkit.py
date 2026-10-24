"""Hand placement rules and a keyframe engine for hand work.

Hand orientation conventions (HandTarget): `fingers` = the hand's long axis (wrist -> knuckles), `palm` = the
direction the palm faces. Thumb = fingers x palm (right hand) / palm x fingers (left hand).

Rules from how people hold things:
- power(): a handle runs across the palm from the little finger to the thumb; the hand continues the forearm
  (its long axis ~ shoulder -> grip, made square to the handle).
- pinch(): fingertips point at the part, palm towards it or down.
- brace(): palm flat on a surface, fingers pointing away from the body.
- trigger(): a pistol grip (drill, scanner, caulking gun) - a power grip on the handle, index on the trigger.

keyed() turns tracks of keyframes into Frames: hand targets and tools / parts interpolated between keys with the
minimum-jerk profile, hand shapes blended (pre-shaping on reaches).
"""
import math

import numpy as np

from .motion_gen import FPS, Frame, ease, grip_blend, lerp_T, lerp_target, rest_targets, tool_from_hand
from .solver import HandTarget
from .tools import TOOLS

UP = np.array([0.0, 1.0, 0.0])


def unit(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def shoulder(sk, side):
    return np.asarray(sk.world[f"upperarm_{side}"], float)


def _perp(v, axis):
    v = np.asarray(v, float)
    a = unit(axis)
    return v - a * np.dot(v, a)


def power(sk, side, point, handle_axis, grip="power:1.5", weight=1.0, lean=None):
    """Power grip at `point` on a handle along `handle_axis` (pointing to the thumb side). `lean` tilts the hand
    axis (e.g. downwards for a hammer)."""
    p = np.asarray(point, float)
    h = unit(handle_axis)
    f = _perp((lean if lean is not None else (p - shoulder(sk, side))), h)
    if np.linalg.norm(f) < 1e-6:
        f = _perp(np.array([0.0, 0.0, 1.0]), h)
    f = unit(f)
    palm = np.cross(h, f) if side == "r" else np.cross(f, h)
    return HandTarget(side, tuple(p), tuple(f), tuple(unit(palm)), grip, weight)


def tilt(t, toward, deg):
    """The same grip with the hand's long axis turned `deg` towards `toward` about the palm normal: a handle held
    diagonally in the palm (a screwdriver, a file) - the hand then continues the forearm instead of a bent wrist."""
    f = np.asarray(t.fingers, float)
    a = math.radians(deg)
    f2 = unit(f * math.cos(a) + unit(_perp(toward, f)) * math.sin(a))
    pm = unit(_perp(t.palm, f2))
    return HandTarget(t.side, t.point, tuple(f2), tuple(pm), t.grip, t.weight)


def pinch(sk, side, tip_point, approach=None, palm=None, grip="pinch", weight=1.0):
    """Pinch at tip_point: fingertips arriving along `approach` (default: from the shoulder), palm facing `palm`
    (default down). The target point is the palm centre, set back from the fingertips."""
    tp = np.asarray(tip_point, float)
    a = unit(approach if approach is not None else tp - shoulder(sk, side))
    pn = unit(_perp(palm if palm is not None else (0.0, -1.0, 0.0), a))
    if np.linalg.norm(pn) < 1e-6:
        pn = unit(_perp((0.0, 0.0, 1.0), a))
    return HandTarget(side, tuple(tp - a * 7.5 - pn * 1.0), tuple(a), tuple(pn), grip, weight)


def brace(sk, side, point, normal=(0.0, 1.0, 0.0), toward=None, weight=0.6, grip="flat"):
    """Palm flat on a surface at `point` (surface normal `normal`), fingers away from the body (or `toward`)."""
    n = unit(normal)
    p = np.asarray(point, float)
    f = _perp(toward if toward is not None else (p - shoulder(sk, side)), n)
    f = unit(f if np.linalg.norm(f) > 1e-6 else _perp((0.0, 0.0, 1.0), n))
    return HandTarget(side, tuple(p + n * 1.8), tuple(f), tuple(-n), grip, weight)


def trigger(sk, side, point, barrel, pull=0.0, weight=1.0, radius=1.7):
    """Pistol grip: the handle runs down from the barrel; the hand axis points along the barrel's side (across),
    palm facing forward along the barrel, index on the trigger (pull 0..1)."""
    b = unit(barrel)                                     # barrel direction (towards the tip)
    p = np.asarray(point, float)
    down = unit(_perp(-UP, b))
    if np.linalg.norm(down) < 1e-6:
        down = unit(_perp((0.0, 0.0, -1.0), b))
    side_ax = np.cross(b, down)                          # handle's thumb side
    t = HandTarget(side, tuple(p), None, None, f"trigger:{radius}:{pull:.2f}", weight)
    h = -down if side == "r" else -down                  # the handle axis to the thumb side points up the handle
    pw = power(sk, side, p, h, grip=t.grip, weight=weight, lean=b)
    return pw


# ---------------------------------------------------------------------------------------------- keyframes
def _interp(keys, t, kind):
    """Value of a track at time t: keys = [(t, value)], eased between keys (min-jerk), held outside."""
    if t <= keys[0][0]:
        return keys[0][1]
    for (t0, a), (t1, b) in zip(keys, keys[1:]):
        if t <= t1:
            u = 0.0 if t1 <= t0 else (t - t0) / (t1 - t0)
            if a is None or b is None:
                return a if u < 0.5 else b
            if kind == "hand":
                return lerp_target(a, b, ease(u))
            if kind == "T":
                return lerp_T(a, b, ease(u))
            if kind == "grip":
                return ("blend", a, b, ease(u))
            return a * (1 - ease(u)) + b * ease(u)
    return keys[-1][1]


def kneel_stance(sk, knee_down="r", forward=34.0, width=None):
    """One knee on the floor (toes tucked), the other foot planted in front: the pelvis about a thigh length above
    the floor. knee_down: which knee is on the floor."""
    from .solver import Stance
    ank = float(sk.world["foot_l"][1])
    hw = abs(sk.world["thigh_l"][0] - sk.world["thigh_r"][0]) / 2 + (width or 3.0)
    thigh = sk.length("thigh_l", "calf_l")
    shin = sk.length("calf_l", "foot_l")
    pel = float(sk.world["pelvis"][1])
    hip_y = thigh * 0.93 + 6.0                                  # knee pad on the floor, thigh near vertical
    st = Stance()
    up = knee_down == "l"
    front, back = ("r", "l") if up else ("l", "r")
    fx = hw if front == "l" else -hw
    bx = hw if back == "l" else -hw
    setattr(st, "left" if front == "l" else "right", (fx, ank, forward))
    setattr(st, "left" if back == "l" else "right", (bx, 11.0, -shin * 0.92))
    setattr(st, f"lift_{back}", -58.0)                         # toes tucked under, heel up
    st.drop = pel - hip_y
    st.hinge = (-2.0, 25.0)
    return st


def squat_stance(sk, depth=0.55, width=None, heels_up=12.0):
    """Feet a little wider than the hips, the pelvis lowered by `depth` x the leg length (heels slightly up)."""
    from .solver import Stance
    ank = float(sk.world["foot_l"][1])
    hw = abs(sk.world["thigh_l"][0] - sk.world["thigh_r"][0]) / 2 + (width or 8.0)
    leg = sk.length("thigh_l", "calf_l") + sk.length("calf_l", "foot_l")
    st = Stance(left=(hw, ank + 1.0, 4.0), right=(-hw, ank + 1.0, 4.0), yaw=0.0)
    st.yaw_l, st.yaw_r = -12.0, 12.0                           # toes turned out a little
    st.lift_l = st.lift_r = -heels_up
    st.drop = leg * depth
    return st


PATH_GUARD = None          # optional (side, HandTarget) -> HandTarget applied to every keyed hand (see tasks.Task)


def keyed(duration, hands=None, tools=None, parts=None, look=None, label="work", stance=None):
    """Frames from keyframe tracks.

    tools: {name: (tool key, [(t, 4x4 tool transform in the worker frame)])}
    parts: {"part:<id>": (spec, [(t, 4x4)])}
    hands: {"r": [(t, value)], "l": [...]} - value: a HandTarget, or a function of the current tool transforms
           (dict name -> 4x4) returning one (a hand that holds a tool follows it), or None (hanging relaxed)
    look:  a point, or [(t, point)]
    """
    hands, tools, parts = hands or {}, tools or {}, parts or {}
    n = max(2, int(round(duration * FPS)))
    frames = []
    for i in range(n):
        t = i / FPS
        Td = {name: _interp(keys, t, "T") for name, (key, keys) in tools.items()}
        tl = {name: (key, Td[name]) for name, (key, keys) in tools.items()}
        for name, (spec, keys) in parts.items():
            tl[name] = (spec, _interp(keys, t, "T"))
        tg = []
        for side, keys in hands.items():
            ev = [(tk, v(Td) if callable(v) else v) for tk, v in keys]
            v = _interp(ev, t, "hand")
            if v is not None and PATH_GUARD is not None:
                v = PATH_GUARD(side, v)
            if v is not None:
                tg.append(v)
        if isinstance(look, list):
            lk = _interp([(a, np.asarray(b, float)) for a, b in look], t, "v")
        else:
            lk = look
        frames.append(Frame(tg, None if lk is None else tuple(lk), tl, stance=stance, label=label))
    return frames


def T_at(origin, x_axis, y_axis):
    from .tools import place
    return place(np.asarray(origin, float), x_axis, y_axis)


def rot_about(T, axis, deg, pivot):
    """Rotate a 4x4 transform about an axis through `pivot` (worker frame)."""
    a = unit(axis)
    th = math.radians(deg)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    R = np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K
    out = np.eye(4)
    out[:3, :3] = R @ T[:3, :3]
    out[:3, 3] = R @ (T[:3, 3] - np.asarray(pivot, float)) + np.asarray(pivot, float)
    return out
