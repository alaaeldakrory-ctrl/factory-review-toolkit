"""Full-body posture solver with comfort priors.

A worker is asked for hand targets ("right palm here, fingers this way, power grip"), optionally a point to look at
and a stance (where the feet are). Limbs are solved exactly (two-bone IK with the elbow down / knees forward); an
optimiser chooses the few whole-body degrees of freedom - pelvis shift and drop (knee bend), pelvis yaw, hip hinge,
spine flexion / side bend / twist - by minimising discomfort the way people do:

    trunk flexion (hinge at the hips first, spread over the spine), side bend, twist, knee bend (squatting costs),
    shoulder elevation, elbow far from ~80 deg, wrist deviation, neck bend, reach shortfall, centre of mass outside
    the feet, and - for animation - distance from the previous frame (smooth, no popping).

    sol = PostureSolver(sk).solve([HandTarget("r", (10, 95, 45), fingers=(0, 0, 1), palm=(0, -1, 0))],
                                  look=(10, 95, 45))
    sol.pose, sol.angles, sol.cost
"""
import math
from dataclasses import dataclass, field

import numpy as np

from .kinematics import I3, Pose, angle, axis_angle, com, fk, frame_rot, rx, ry, rz, unit

UP = np.array([0.0, 1.0, 0.0])
DOWN = -UP
FWD = np.array([0.0, 0.0, 1.0])
SPINE = (("spine_01", 0.30, 0.34, 0.30), ("spine_02", 0.35, 0.33, 0.30), ("spine_03", 0.35, 0.33, 0.40))
VARS = ("dx", "drop", "dz", "yaw", "hinge", "flex", "lat", "twist")
LO = np.array([-15.0, -2.0, -25.0, -45.0, -10.0, -3.0, -25.0, -40.0])        # no back arching to cheat flexion
HI = np.array([15.0, 70.0, 25.0, 45.0, 75.0, 50.0, 25.0, 40.0])
SCALE = np.array([4.0, 8.0, 5.0, 10.0, 8.0, 8.0, 6.0, 8.0])        # simplex size and smoothness scale per variable

GRIPS = {   # finger curl per joint (deg) for index..pinky, thumb (01, 02, 03 flex) - simple, readable presets
    "relaxed": ((14, 14, 10), (8, 10, 8)),
    "open": ((4, 4, 2), (0, 4, 4)),
    "power": ((62, 72, 48), (30, 30, 25)),        # around a handle (ratchet, hammer, wrench)
    "pinch": ((40, 45, 30), (25, 35, 30)),        # small part between thumb and index
    "hook": ((35, 80, 60), (0, 10, 10)),          # carrying a tote / bag handle
    "flat": ((2, 0, 0), (-5, 0, 0)),              # pushing with the palm, bracing
    "point": ((0, 0, 0), (20, 20, 10)),           # pressing a button
}


@dataclass
class HandTarget:
    side: str                                     # "l" / "r"
    point: tuple                                  # grip point (palm centre) in the worker frame, cm
    fingers: tuple = None                         # finger direction (default: away from the shoulder)
    palm: tuple = None                            # palm normal (default: facing inward, thumb up)
    grip: str = "power"
    weight: float = 1.0


@dataclass
class Stance:
    left: tuple = None                            # ankle point of the left / right foot (worker frame)
    right: tuple = None
    yaw: float = 0.0                              # feet direction (deg about Y; 0 = facing +Z)


@dataclass
class Solution:
    pose: Pose
    cost: float
    terms: dict = field(default_factory=dict)
    angles: dict = field(default_factory=dict)
    v: np.ndarray = None


def _hull(pts):
    pts = sorted(map(tuple, pts))
    if len(pts) < 3:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lo, hi = [], []
    for p in pts:
        while len(lo) >= 2 and cross(lo[-2], lo[-1], p) <= 0:
            lo.pop()
        lo.append(p)
    for p in reversed(pts):
        while len(hi) >= 2 and cross(hi[-2], hi[-1], p) <= 0:
            hi.pop()
        hi.append(p)
    return lo[:-1] + hi[:-1]


def _outside(p, hull):
    """Distance of 2-D point p outside a counter-clockwise convex polygon (0 inside)."""
    worst = 0.0
    n = len(hull)
    for i in range(n):
        a, b = np.array(hull[i]), np.array(hull[(i + 1) % n])
        e = b - a
        nrm = np.array([e[1], -e[0]]) / (np.linalg.norm(e) or 1.0)          # outward normal (ccw)
        worst = max(worst, float(np.dot(np.asarray(p) - a, nrm)))
    return worst


def nelder_mead(f, x0, step, lo, hi, iters=160, tol=1e-4):
    n = len(x0)
    xs = [np.clip(x0, lo, hi)]
    for i in range(n):
        x = xs[0].copy()
        x[i] += step[i]
        xs.append(np.clip(x, lo, hi))
    fs = [f(x) for x in xs]
    for _ in range(iters):
        order = np.argsort(fs)
        xs, fs = [xs[i] for i in order], [fs[i] for i in order]
        if abs(fs[-1] - fs[0]) < tol:
            break
        c = np.mean(xs[:-1], axis=0)
        xr = np.clip(c + (c - xs[-1]), lo, hi)
        fr = f(xr)
        if fr < fs[0]:
            xe = np.clip(c + 2 * (c - xs[-1]), lo, hi)
            fe = f(xe)
            xs[-1], fs[-1] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < fs[-2]:
            xs[-1], fs[-1] = xr, fr
        else:
            xc = np.clip(c + 0.5 * (xs[-1] - c), lo, hi)
            fc = f(xc)
            if fc < fs[-1]:
                xs[-1], fs[-1] = xc, fc
            else:
                for i in range(1, n + 1):
                    xs[i] = xs[0] + 0.5 * (xs[i] - xs[0])
                    fs[i] = f(xs[i])
    i = int(np.argmin(fs))
    return xs[i], fs[i]


class PostureSolver:
    def __init__(self, sk, worker=None):
        self.sk = sk
        self.L = {s: (sk.length(f"upperarm_{s}", f"lowerarm_{s}"), sk.length(f"lowerarm_{s}", f"hand_{s}"))
                  for s in ("l", "r")}
        self.leg = {s: (sk.length(f"thigh_{s}", f"calf_{s}"), sk.length(f"calf_{s}", f"foot_{s}")) for s in ("l", "r")}
        self.hand_len = sk.length("hand_l", "middle_01_l") / 0.47 * 1.0 if "middle_01_l" in sk.world else 19.0
        self.worker = worker

    # ------------------------------------------------------------------ building a pose from the variables
    def build(self, v, targets, look, stance, free_arms="hang"):
        dx, drop, dz, yaw, hinge, flex, lat, twist = v
        sk = self.sk
        pose = Pose(pelvis=np.array([dx, -drop, dz]))
        pose.rot["pelvis"] = ry(yaw) @ rx(hinge)
        for j, kf, kl, kt in SPINE:
            pose.rot[j] = rx(flex * kf) @ rz(-lat * kl) @ ry(twist * kt)
        R, P = fk(sk, pose, ["spine_03", "thigh_l", "thigh_r"])
        terms = {}
        # legs: knees forward, feet flat on the stance
        fwd = unit(R["pelvis"] @ FWD * np.array([1, 0, 1]))
        for s in ("l", "r"):
            ank = np.asarray(getattr(stance, "left" if s == "l" else "right"), float)
            hip = P[f"thigh_{s}"]
            L1, L2 = self.leg[s]
            d = np.linalg.norm(ank - hip)
            terms[f"leg_{s}"] = max(0.0, d - (L1 + L2) * 0.999) ** 2 * 60.0
            dd = min(d, (L1 + L2) * 0.999)
            u = unit(ank - hip)
            a = (L1 * L1 - L2 * L2 + dd * dd) / (2 * dd)
            h = math.sqrt(max(L1 * L1 - a * a, 0.0))
            pole = unit(fwd - u * np.dot(fwd, u))
            knee = hip + u * a + pole * h
            td, cd = unit(knee - hip), unit(ank - knee)
            rest_thigh = unit(sk.world[f"calf_{s}"] - sk.world[f"thigh_{s}"])
            Rt = frame_rot(rest_thigh, FWD, td, pole)
            n = np.cross(td, cd)
            Rc = axis_angle(n, math.radians(angle(td, cd))) @ Rt if np.linalg.norm(n) > 1e-9 else Rt
            Rf = ry(stance.yaw)
            pose.rot[f"thigh_{s}"] = R["pelvis"].T @ Rt
            pose.rot[f"calf_{s}"] = Rt.T @ Rc
            pose.rot[f"foot_{s}"] = Rc.T @ Rf
            terms[f"knee_{s}"] = (angle(td, cd) / 45.0) ** 2 * 0.6
        # arms
        R, P = fk(sk, pose, ["spine_03", "clavicle_l", "clavicle_r", "upperarm_l", "upperarm_r", "head"])
        tg = {t.side: t for t in targets}
        angles = {}
        for s, sx in (("l", 1.0), ("r", -1.0)):
            t = tg.get(s)
            S0 = P[f"upperarm_{s}"]
            if t is None:
                grip_pt = S0 + DOWN * (sum(self.L[s]) + self.hand_len * 0.45) * 0.97 + R["spine_03"] @ FWD * 4.0
                fingers, palm, gname = DOWN + R["spine_03"] @ FWD * 0.1, np.array([-sx, 0, 0.0]), "relaxed"
                wgt = 0.0
            else:
                grip_pt = np.asarray(t.point, float)
                gname, wgt = t.grip, t.weight
                fingers = unit(t.fingers) if t.fingers is not None else unit(grip_pt - S0)
                palm = unit(t.palm) if t.palm is not None else unit(sx * np.cross(fingers, UP)) if abs(fingers[1]) < 0.95 \
                    else np.array([-sx, 0.0, 0.0])
            # clavicle: shrug a little for high targets
            rise = float(grip_pt[1] - S0[1])
            cl = min(18.0, max(0.0, rise - 5.0) * 0.35) if t is not None else 0.0
            pose.rot[f"clavicle_{s}"] = rz(sx * cl)
            R2, P2 = fk(sk, pose, [f"upperarm_{s}"])
            S = P2[f"upperarm_{s}"]
            palm = unit(palm - fingers * np.dot(palm, fingers))
            wrist = grip_pt - fingers * self.hand_len * 0.40 - palm * 1.6
            L1, L2 = self.L[s]
            d = np.linalg.norm(wrist - S)
            terms[f"reach_{s}"] = max(0.0, d - (L1 + L2) * 0.985) ** 2 * 8.0 * max(wgt, 0.05)
            dd = min(max(d, abs(L1 - L2) + 1e-3), (L1 + L2) * 0.985)
            u = unit(wrist - S)
            a = (L1 * L1 - L2 * L2 + dd * dd) / (2 * dd)
            h = math.sqrt(max(L1 * L1 - a * a, 0.0))
            trunk = R2["spine_03"]
            pole = trunk @ unit(np.array([sx * 0.35, -1.0, -0.25]))              # elbow down, a little out and back
            pole = unit(pole - u * np.dot(pole, u))
            elbow = S + u * a + pole * h
            ue, fe = unit(elbow - S), unit(S + u * dd - elbow)
            bend = fe - ue * np.dot(fe, ue)
            Ru = frame_rot(np.array([sx, 0, 0]), FWD, ue, bend if np.linalg.norm(bend) > 1e-6 else trunk @ FWD)
            n = np.cross(ue, fe)
            Rf = axis_angle(n, math.radians(angle(ue, fe))) @ Ru if np.linalg.norm(n) > 1e-9 else Ru
            Rh = frame_rot(np.array([sx, 0, 0]), DOWN, fingers, palm)
            Rcl = R2[f"clavicle_{s}"]
            pose.rot[f"upperarm_{s}"] = Rcl.T @ Ru
            pose.rot[f"lowerarm_{s}"] = Ru.T @ Rf
            pose.rot[f"hand_{s}"] = Rf.T @ Rh
            elev = angle(ue, trunk @ DOWN)
            eflex = angle(ue, fe)
            wdev = angle(fe, fingers)
            angles.update({f"shoulder_{s}": elev, f"elbow_{s}": eflex, f"wrist_{s}": wdev})
            if t is not None:
                terms[f"shoulder_{s}"] = (max(0.0, elev - 20.0) / 45.0) ** 2 * wgt
                terms[f"elbow_{s}"] = ((eflex - 80.0) / 55.0) ** 2 * 0.3 * wgt
                terms[f"wrist_{s}"] = (wdev / 30.0) ** 2 * 0.6 * wgt
            self._grip(pose, s, gname)
        # head and neck: look at the work
        if look is not None:
            R3, P3 = fk(sk, pose, ["head"])
            dloc = R3["spine_03"].T @ (np.asarray(look, float) - P3["head"])
            yw = math.degrees(math.atan2(dloc[0], dloc[2]))
            pt = math.degrees(math.atan2(-dloc[1], math.hypot(dloc[0], dloc[2])))
            ywc, ptc = max(-75.0, min(75.0, yw)), max(-45.0, min(70.0, pt))
            pose.rot["neck_01"] = ry(ywc * 0.4) @ rx(ptc * 0.45)
            pose.rot["head"] = ry(ywc * 0.6) @ rx(ptc * 0.55)
            terms["neck"] = (max(0.0, ptc - 15.0) / 30.0) ** 2 * 0.6 + (ywc / 45.0) ** 2 * 0.3 + \
                ((abs(yw - ywc) + abs(pt - ptc)) / 10.0) ** 2
            angles.update({"neck_flexion": ptc, "neck_twist": ywc})
        # trunk and balance
        terms["trunk"] = ((hinge + flex) / 28.0) ** 2 + (flex / 35.0) ** 2 * 0.5 + (lat / 10.0) ** 2 + (twist / 14.0) ** 2             + (min(0.0, hinge + flex) / 6.0) ** 2                       # leaning back is rare and costly
        terms["pelvis"] = (dx / 8.0) ** 2 * 0.3 + (dz / 10.0) ** 2 * 0.3 + (yaw / 25.0) ** 2 * 0.4
        Rall, Pall = fk(sk, pose)
        c = com(Pall)
        feet = []
        for s in ("l", "r"):
            a = np.asarray(getattr(stance, "left" if s == "l" else "right"), float)
            for ox, oz in ((-5, -6), (5, -6), (-5, 18), (5, 18)):
                q = a + ry(stance.yaw) @ np.array([ox, 0, oz])
                feet.append((q[0], q[2]))
        terms["balance"] = (_outside((c[0], c[2]), _hull(feet)) / 1.5) ** 2 * 20.0
        angles.update({"trunk_flexion": angle(Rall["spine_03"] @ UP, UP), "trunk_side": lat, "trunk_twist": twist,
                       "knee_l": angle(Pall["calf_l"] - Pall["thigh_l"], Pall["foot_l"] - Pall["calf_l"]),
                       "knee_r": angle(Pall["calf_r"] - Pall["thigh_r"], Pall["foot_r"] - Pall["calf_r"])})
        return pose, terms, angles, (Rall, Pall)

    def _grip(self, pose, s, name):
        sx = 1.0 if s == "l" else -1.0
        (f1, f2, f3), (t1, t2, t3) = GRIPS.get(name, GRIPS["relaxed"])
        for f in ("index", "middle", "ring", "pinky"):
            spread = {"index": 1.0, "middle": 1.05, "ring": 1.1, "pinky": 1.15}[f]
            for i, a in zip((1, 2, 3), (f1, f2, f3)):
                pose.rot[f"{f}_0{i}_{s}"] = rz(-sx * a * spread)
        pose.rot[f"thumb_01_{s}"] = ry(sx * t1 * 0.6) @ rz(-sx * t1 * 0.5)
        pose.rot[f"thumb_02_{s}"] = rz(-sx * t2)
        pose.rot[f"thumb_03_{s}"] = rz(-sx * t3)

    # ------------------------------------------------------------------ solving
    def solve(self, targets, look=None, stance=None, prev=None, smooth=1.0, seeds=None):
        stance = stance or self.default_stance()
        prev_v = prev.v if isinstance(prev, Solution) else (None if prev is None else np.asarray(prev, float))

        def f(v):
            _, terms, _, _ = self.build(v, targets, look, stance)
            c = sum(terms.values())
            if prev_v is not None:
                c += smooth * float(np.sum(((v - prev_v) / SCALE) ** 2)) * 0.5
            return c
        starts = [prev_v] if prev_v is not None else (seeds or [np.zeros(8), np.array([0, 25, 0, 0, 15, 5, 0, 0.0]),
                                                                  np.array([0, 4, 3, 0, 35, 15, 0, 0.0])])
        best = None
        for x0 in starts:
            x, fx = nelder_mead(f, np.asarray(x0, float), SCALE, LO, HI, iters=220 if prev_v is None else 90)
            x, fx = nelder_mead(f, x, SCALE * 0.3, LO, HI, iters=80)              # polish
            if best is None or fx < best[1]:
                best = (x, fx)
        x, fx = best
        pose, terms, angles, _ = self.build(x, targets, look, stance)
        return Solution(pose, fx, terms, angles, x)

    def default_stance(self, width=None, yaw=0.0):
        sk = self.sk
        w = width if width is not None else abs(sk.world["foot_l"][0] - sk.world["foot_r"][0]) + 6.0
        y = sk.world["foot_l"][1]
        return Stance(left=(w / 2, y, 0.0), right=(-w / 2, y, 0.0), yaw=yaw)
