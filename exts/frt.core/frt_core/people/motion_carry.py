"""Handling parts and wiring: carry (pick -> walk -> place, the part in the world), floor / pallet lifts, cable
tie, crimp. Parts are world objects named "part:<id>" in Frame.tools with a geometry spec ("box:30x14x22:look")
and a transform in the worker frame (bake.py places them in the world).
"""
import numpy as np

from .motion_gen import FPS, Frame, ease, lerp_target, rest_targets
from .solver import HandTarget
from .tools import place


def _T(p):
    T = np.eye(4)
    T[:3, 3] = p
    return T


def box_hands(p, size, two_hands=True, side="r"):
    """Hand targets holding a box centred at p: palms on the side faces (two hands) or a pinch on top (one)."""
    w, h, d = size
    p = np.asarray(p, float)
    if not two_hands:
        return [HandTarget(side, tuple(p + np.array([0, h / 2 + 1.5, 0])), fingers=(0, 0, 1), palm=(0, -1, 0),
                           grip="pinch")]
    return [HandTarget("l", tuple(p + np.array([w / 2 + 1.4, h * 0.1, 0])), fingers=(0, -0.25, 1), palm=(-1, 0, 0),
                       grip="hook:3"),
            HandTarget("r", tuple(p + np.array([-w / 2 - 1.4, h * 0.1, 0])), fingers=(0, -0.25, 1), palm=(1, 0, 0),
                       grip="hook:3")]


def carry_pose(sk, size, two_hands=True, side="r"):
    """Where a carried part sits in the worker frame: close in front, hands near elbow height."""
    elbow = float(sk.world["upperarm_l"][1]) - sk.length("upperarm_l", "lowerarm_l")   # standing elbow height
    w, h, d = size
    if two_hands:
        return np.array([0.0, elbow - 6.0 - h * 0.1, 18.0 + d / 2])
    sx = 1.0 if side == "l" else -1.0
    return np.array([sx * 12.0, elbow - 10.0 - h / 2, 22.0 + d / 2])


def carry_pick(solver, pick, size, part, spec, two_hands=True, side="r", label="pick"):
    """Reach, grasp, lift the part into the carry position. Returns (frames, carry hand targets, part carry T)."""
    rest, _ = rest_targets(solver)
    pick = np.asarray(pick, float)
    hold = box_hands(pick, size, two_hands, side)
    cp = carry_pose(solver.sk, size, two_hands, side)
    frames = []
    nr = int(0.9 * FPS)
    name = f"part:{part}"
    for k in range(nr):
        u = ease((k + 1) / nr)
        frames.append(Frame([lerp_target(rest[t.side], t, u) for t in hold], tuple(pick), {name: (spec, _T(pick))},
                            label=label))
    frames += [Frame(hold, tuple(pick), {name: (spec, _T(pick))}, static=k > 0, label=label) for k in range(int(0.25 * FPS))]
    nl = int(1.0 * FPS)
    up = pick + np.array([0, 10.0, 0])
    for k in range(nl):                                         # lift: up first, then in towards the body
        u = ease((k + 1) / nl)
        p = (1 - u) ** 2 * pick + 2 * (1 - u) * u * up + u * u * cp
        frames.append(Frame(box_hands(p, size, two_hands, side), tuple(p + np.array([0, 0, 20])),
                            {name: (spec, _T(p))}, label=label))
    return frames, box_hands(cp, size, two_hands, side), _T(cp)


def carry_place(solver, place_at, size, part, spec, two_hands=True, side="r", label="place"):
    """From the carry position down onto the place point, release, hands back."""
    rest, _ = rest_targets(solver)
    place_at = np.asarray(place_at, float)
    cp = carry_pose(solver.sk, size, two_hands, side)
    name = f"part:{part}"
    frames = []
    nl = int(1.0 * FPS)
    over = place_at + np.array([0, 8.0, 0])
    for k in range(nl):
        u = ease((k + 1) / nl)
        p = (1 - u) ** 2 * cp + 2 * (1 - u) * u * over + u * u * place_at
        frames.append(Frame(box_hands(p, size, two_hands, side), tuple(place_at), {name: (spec, _T(p))}, label=label))
    hold = box_hands(place_at, size, two_hands, side)
    nr = int(0.8 * FPS)
    for k in range(nr):
        u = ease((k + 1) / nr)
        frames.append(Frame([lerp_target(t, rest[t.side], u) for t in hold], tuple(place_at),
                            {name: (spec, _T(place_at))}, label=label))
    return frames


def cable_tie(solver, at, axis=(1, 0, 0), side="r", part="tie"):
    """Install a cable tie on a harness at `at` (harness along `axis`): support hand holds the harness, the working
    hand wraps the tie around it, threads the tail, pulls it tight with a short jerk."""
    rest, _ = rest_targets(solver)
    P = np.asarray(at, float)
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    up = np.array([0.0, 1.0, 0.0]) if abs(a[1]) < 0.9 else np.array([0.0, 0.0, 1.0])
    e1 = up - a * np.dot(up, a)
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(a, e1)
    other = "l" if side == "r" else "r"
    so = 1.0 if other == "l" else -1.0
    hold = HandTarget(other, tuple(P + a * so * 6.0 + e1 * 1.5), fingers=tuple(-e2), palm=tuple(-e1),
                      grip="hook:1.5", weight=0.6)
    R = 3.2
    ring = place(P, e1, e2)                                  # tie lies in the plane across the harness
    ring[:3, :3] = np.column_stack([e1, e2, a])
    spec = "ring:1.6:0.18:plastic_black"
    name = f"part:{part}"
    frames = []
    nr = int(0.8 * FPS)
    start = HandTarget(side, tuple(P + e1 * R + e2 * 1.0), fingers=tuple(-e2), palm=tuple(-e1), grip="pinch")
    for k in range(nr):
        u = ease((k + 1) / nr)
        frames.append(Frame([lerp_target(rest[side], start, u), lerp_target(rest[other], hold, u)], tuple(P),
                            label="tie"))
    nw = int(0.8 * FPS)
    for k in range(nw):                  # wrap: the fingertips feed the tie under the harness (half a turn of the
        th = 2 * np.pi * ease((k + 1) / nw) * 0.45      # hand - the forearm can roll that far, not a full circle)
        q = P + (e1 * np.cos(th) + e2 * np.sin(th)) * R
        tho = th * 0.2                    # the hand rolls only a little about the harness; the fingers do the rest
        fng = -e2 * np.cos(tho) + e1 * np.sin(tho)
        plm = -e1 * np.cos(tho) - e2 * np.sin(tho)
        frames.append(Frame([HandTarget(side, tuple(q), fingers=tuple(fng), palm=tuple(plm), grip="pinch"), hold],
                            tuple(P), label="tie"))
    tail0 = P + e1 * R
    last = frames[-1].targets[0]
    grab = HandTarget(side, tuple(tail0), fingers=tuple(-e2), palm=tuple(-e1), grip="pinch")   # pinch from above
    nb = int(0.35 * FPS)
    for k in range(nb):                                      # back over the top to the tie head: thread the tail
        u = ease((k + 1) / nb)
        frames.append(Frame([lerp_target(last, grab, u), hold], tuple(P), label="tie"))
    for k in range(int(0.5 * FPS)):                          # pull the tail: slow, then a short jerk
        u = (k + 1) / (0.5 * FPS)
        d = 5.0 * ease(min(1.0, u / 0.7)) + (2.0 * (u - 0.7) / 0.3 if u > 0.7 else 0.0)
        frames.append(Frame([HandTarget(side, tuple(tail0 + e1 * d), fingers=tuple(-e2), palm=tuple(-e1), grip="pinch"),
                             hold], tuple(P), {name: (spec, ring)}, label="tie pull"))
    end = HandTarget(side, tuple(tail0 + e1 * 7.0), fingers=tuple(-e2), palm=tuple(-e1), grip="pinch")
    for k in range(nr):
        u = ease((k + 1) / nr)
        frames.append(Frame([lerp_target(end, rest[side], u), lerp_target(hold, rest[other], u)], tuple(P),
                            {name: (spec, ring)}, label="tie"))
    return frames


def crimp(solver, at, side="r", squeezes=1):
    """Crimp a terminal: the support hand feeds the wire into the jaws at `at`, the working hand closes the crimper
    handles (power grip closing from open handles to the ratchet release), then lets go."""
    rest, _ = rest_targets(solver)
    P = np.asarray(at, float)
    other = "l" if side == "r" else "r"
    sx = 1.0 if side == "l" else -1.0
    grip_pt = P + np.array([sx * 2.0, -2.0, -14.0])
    feed = HandTarget(other, tuple(P + np.array([-sx * 9.0, 0.5, 0.0])), fingers=(sx, 0, 0), palm=(0, -1, 0),
                      grip="pinch", weight=0.6)
    frames = []
    nr = int(0.8 * FPS)

    def hand(r):
        return HandTarget(side, tuple(grip_pt), fingers=(0, -0.3, 1), palm=(-sx, 0, 0), grip=f"power:{r:.2f}")
    for k in range(nr):
        u = ease((k + 1) / nr)
        frames.append(Frame([lerp_target(rest[side], hand(3.2), u), lerp_target(rest[other], feed, u)], tuple(P),
                            label="crimp"))
    for _ in range(squeezes):
        n = int(0.9 * FPS)
        for k in range(n):                                   # close hard (60 %), hold, open
            u = k / n
            r = 3.2 - 1.6 * (ease(u / 0.6) if u < 0.6 else (1.0 if u < 0.75 else 1 - ease((u - 0.75) / 0.25)))
            frames.append(Frame([hand(r), feed], tuple(P), label="crimp squeeze"))
    for k in range(nr):
        u = ease((k + 1) / nr)
        frames.append(Frame([lerp_target(hand(3.2), rest[side], u), lerp_target(feed, rest[other], u)], tuple(P),
                            label="crimp"))
    return frames
