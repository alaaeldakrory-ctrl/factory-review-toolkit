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

from .kinematics import I3, Pose, angle, axis_angle, com, cross3, fk, frame_rot, norm3, quat_from_matrix, rx, ry, rz, unit

UP = np.array([0.0, 1.0, 0.0])
DOWN = -UP
FWD = np.array([0.0, 0.0, 1.0])
SPINE = (("spine_01", 0.30, 0.34, 0.30), ("spine_02", 0.35, 0.33, 0.30), ("spine_03", 0.35, 0.33, 0.40))
VARS = ("dx", "drop", "dz", "yaw", "hinge", "flex", "lat", "twist")
LUMBAR_SHARE = 0.45         # of a forward bend, the part the spine takes (the rest: the pelvis tilting at the hips)
LO = np.array([-15.0, -2.0, -25.0, -45.0, -2.0, -3.0, -25.0, -40.0])         # no back arching / pelvis tilted back
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
    yaw_l: float = None                           # per-foot direction (walking, turning); default `yaw`
    yaw_r: float = None
    lift_l: float = 0.0                           # swing foot: toe-up pitch (deg) while it is in the air
    lift_r: float = 0.0
    drop: float = None                            # posture: pelvis this far below standing (kneel / squat), cm
    hinge: tuple = None                           # posture: (min, max) hip hinge allowed, deg

    def foot_yaw(self, s):
        v = self.yaw_l if s == "l" else self.yaw_r
        return self.yaw if v is None else v


WRIST_MAX = 88.0             # deg: the hand never turns further than this from the forearm (checks flag 95)


def _clamp_rot(R, max_deg, prev=None):
    """R, or the same turn about the same axis cut to max_deg (a target orientation the wrist can't reach - e.g.
    a blended hand path rolling past the forearm's twist range - is followed only as far as a wrist goes). Past a
    half turn the axis flips: prev (the frame before's hand) picks the side it stays on."""
    c = (R[0, 0] + R[1, 1] + R[2, 2] - 1.0) / 2.0
    if c >= math.cos(math.radians(max_deg)):
        return R
    q = quat_from_matrix(R)
    ax = np.array(q[1:], float)
    n = norm3(ax)
    if n < 1e-9:
        return R
    C = axis_angle(ax / n, math.radians(max_deg))
    if prev is not None:
        C2 = axis_angle(-ax / n, math.radians(max_deg))
        if np.trace(prev.T @ C2) > np.trace(prev.T @ C):              # (closer to the frame before)
            return C2
    return C


def _prev_arm(prev):
    """The arm branch of the frame before (swivel, twist, elbow direction): the next frame stays on it."""
    if not isinstance(prev, Solution):
        return None
    return {s_: (prev.angles.get(f"swivel_{s_}"), prev.angles.get(f"twist_{s_}"),
                 tuple(prev.angles[f"elbow_dir{k}_{s_}"] for k in "xyz") if f"elbow_dirx_{s_}" in prev.angles else None,
                 prev.angles.get(f"twist_raw_{s_}")) for s_ in ("l", "r")}


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


def _quat(R):
    from .kinematics import quat_from_matrix
    return quat_from_matrix(R)


def keep_out_cost(P, rules):
    """Penalty for body joints in front of a fixture face: rules = [(joint names, max z)] or
    [(joint names, plane point, plane normal)] (the normal points into the fixture; worker frame, cm)."""
    c = 0.0
    for r in rules:
        for q in r[0]:
            over = keep_out_over(P[q], r)
            if over > 0.0:
                c += (over / 2.0) ** 2 * 3.0
    return c


def keep_out_over(p, rule):
    """How far point p is past a keep-out rule (cm; negative = clear)."""
    if len(rule) == 3:
        return float(np.dot(np.asarray(p, float) - np.asarray(rule[1], float), np.asarray(rule[2], float)))
    return float(p[2]) - rule[1]


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


def torso_profile_of(sk):
    """[(t, half width, front, back)] along pelvis -> chest: the character's measured trunk (skin and clothes,
    Character.torso_profile) or, for the frt bodies, one from the shoulder breadth."""
    prof = getattr(sk, "torso", None)
    if not prof:
        half = abs(float(sk.world["upperarm_l"][0] - sk.world["upperarm_r"][0])) / 2.0
        prof = [(0.0, 0.95 * half, 0.72 * half, 0.75 * half), (1.0, 0.92 * half, 0.70 * half, 0.75 * half)]
    return prof


class PostureSolver:
    def __init__(self, sk, worker=None):
        self.sk = sk
        self.L = {s: (sk.length(f"upperarm_{s}", f"lowerarm_{s}"), sk.length(f"lowerarm_{s}", f"hand_{s}"))
                  for s in ("l", "r")}
        self.leg = {s: (sk.length(f"thigh_{s}", f"calf_{s}"), sk.length(f"calf_{s}", f"foot_{s}")) for s in ("l", "r")}
        self.hand_len = sk.length("hand_l", "middle_01_l") / 0.47 * 1.0 if "middle_01_l" in sk.world else 19.0
        self.worker = worker
        self._wrap = {}
        self._thumb_cache = {}
        self.keep_out = []          # [(joint names, max z in the worker frame)]: fixture faces the body stays behind
        self.field = None           # collide.SceneField: the scene's equipment the body stays out of
        self._world = None          # 4x4 worker frame -> field, per solve
        self._prev_hand = None      # the frame before's hand rotations (which way a clamped wrist stays)
        self.base_correct = False   # base (generated / mocap) pose: allow small pelvis / trunk corrections on top
        self.effort = 1.0           # search iterations x this (a preview: fewer - its poses start from the one before)

    # ------------------------------------------------------------------ building a pose from the variables
    def build(self, v, targets, look, stance, free_arms="hang", swing=None, base=None, prev_arm=None):
        """base: a full-body Pose (mocap) for the trunk, head and free arms - only the legs (feet on the stance) and
        the hands that have targets are solved on top of it."""
        dx, drop, dz, yaw, hinge, flex, lat, twist = v
        sk = self.sk
        if base is not None:
            pose = base.copy()
            pose.pelvis = base.pelvis + np.array([dx, -drop, dz])
            if self.base_correct:                      # corrections on top of the base trunk
                pose.rot["pelvis"] = ry(yaw) @ rx(hinge) @ base.local("pelvis")
                for j, kf, kl, kt in SPINE:
                    pose.rot[j] = rx(flex * kf) @ rz(-lat * kl) @ ry(twist * kt) @ base.local(j)
        else:
            pose = Pose(pelvis=np.array([dx, -drop, dz]))
            pose.rot["pelvis"] = ry(yaw) @ rx(hinge)
            for j, kf, kl, kt in SPINE:
                pose.rot[j] = rx(flex * kf) @ rz(-lat * kl) @ ry(twist * kt)
        R, P = fk(sk, pose, ["spine_03", "thigh_l", "thigh_r"])
        terms = {}
        # legs: knees forward, feet flat on the stance (a free stance: the legs of the base pose as they are)
        fwd = unit(R["pelvis"] @ FWD * np.array([1, 0, 1]))
        for s in (() if getattr(stance, "free", False) and base is not None else ("l", "r")):
            ank = np.asarray(getattr(stance, "left" if s == "l" else "right"), float)
            hip = P[f"thigh_{s}"]
            L1, L2 = self.leg[s]
            d = norm3(ank - hip)
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
            n = cross3(td, cd)
            Rc = axis_angle(n, math.radians(angle(td, cd))) @ Rt if norm3(n) > 1e-9 else Rt
            Rf = ry(stance.foot_yaw(s)) @ rx(-getattr(stance, f"lift_{s}"))
            pose.rot[f"thigh_{s}"] = R["pelvis"].T @ Rt
            pose.rot[f"calf_{s}"] = Rt.T @ Rc
            pose.rot[f"foot_{s}"] = Rc.T @ Rf
            terms[f"knee_{s}"] = (angle(td, cd) / 45.0) ** 2 * 0.6
        # arms
        R, P = fk(sk, pose, ["spine_03", "clavicle_l", "clavicle_r", "upperarm_l", "upperarm_r", "head"])
        tg = {t.side: t for t in targets}
        angles = {}
        # the torso the arms stay out of: an elliptic cylinder pelvis -> chest, sized from the shoulder breadth
        torso = (P["pelvis"], P["spine_03"] + R["spine_03"] @ UP * 6.0, R["spine_03"], self.torso_profile())
        for s, sx in (("l", 1.0), ("r", -1.0)):
            t = tg.get(s)
            S0 = P[f"upperarm_{s}"]
            if t is None and base is not None:                                   # mocap arm swing as it is
                self._grip(pose, s, "relaxed")
                self._clear_free_arm(pose, s, sx, torso)
                continue
            if t is None:
                grip_pt = S0 + DOWN * (sum(self.L[s]) + self.hand_len * 0.45) * 0.97 + R["spine_03"] @ FWD * 4.0
                if swing and s in swing:                                         # walking arm swing
                    grip_pt = grip_pt + R["pelvis"] @ FWD * swing[s] + UP * abs(swing[s]) * 0.18
                fingers = unit(DOWN + R["spine_03"] @ FWD * 0.1)
                palm = R["spine_03"] @ np.array([-sx, 0, 0.0])                   # palm towards the thigh
                gname, wgt = "relaxed", 0.0
            else:
                grip_pt = np.asarray(t.point, float)
                gname, wgt = t.grip, t.weight
                fingers = unit(t.fingers) if t.fingers is not None else unit(grip_pt - S0)
                palm = unit(t.palm) if t.palm is not None else unit(sx * cross3(fingers, UP)) if abs(fingers[1]) < 0.95 \
                    else np.array([-sx, 0.0, 0.0])
            # clavicle (scapula): elevates for high targets, protracts as the hand reaches forward / across the body -
            # a long reach comes from the shoulder girdle too, not only the arm
            rise = float(grip_pt[1] - S0[1])
            cl = min(18.0, max(0.0, rise - 5.0) * 0.35) if t is not None else 0.0
            prot = 0.0
            if t is not None:
                vt = R["spine_03"].T @ (grip_pt - S0)
                arm_len = sum(self.L[s]) + self.hand_len * 0.45
                frac = float(norm3(vt)) / arm_len
                fwd_k = max(0.0, float(vt[2]) / (float(norm3(vt)) + 1e-9)) + max(0.0, -sx * float(vt[0])) / arm_len
                prot = min(22.0, max(0.0, (frac - 0.45) / 0.45) * 20.0 * min(1.0, fwd_k))
            pose.rot[f"clavicle_{s}"] = ry(-sx * prot) @ rz(sx * cl)
            angles[f"clavicle_{s}"] = prot
            R2, P2 = fk(sk, pose, [f"upperarm_{s}"])
            S = P2[f"upperarm_{s}"]
            palm = unit(palm - fingers * np.dot(palm, fingers))
            wrist = grip_pt - fingers * self.hand_len * 0.40 - palm * 1.6
            L1, L2 = self.L[s]
            d = norm3(wrist - S)
            terms[f"reach_{s}"] = max(0.0, d - (L1 + L2) * 0.985) ** 2 * 8.0 * max(wgt, 0.05)
            dd = min(max(d, abs(L1 - L2) + 1e-3), (L1 + L2) * 0.985)
            u = unit(wrist - S)
            a = (L1 * L1 - L2 * L2 + dd * dd) / (2 * dd)
            h = math.sqrt(max(L1 * L1 - a * a, 0.0))
            trunk = R2["spine_03"]
            Rh = frame_rot(np.array([sx, 0, 0]), DOWN, fingers, palm)
            pa = (prev_arm or {}).get(s)
            ref = None
            if base is not None:                               # the recorded person's elbow: where this one goes
                Rb, Pb = fk(sk, pose, [f"lowerarm_{s}"])
                ref = unit(Pb[f"lowerarm_{s}"] - Pb[f"upperarm_{s}"])
            Ru, Rf, wsw, tau, elbow, sw = self._arm(s, sx, S, u, a, h, dd, trunk, Rh, True, pa, torso, ref)
            angles[f"swivel_{s}"], angles[f"twist_{s}"] = sw, tau
            angles[f"twist_raw_{s}"] = float(self._raw_twist)
            ed = trunk.T @ unit(elbow - S)                     # where the elbow points (trunk frame), for the next frame
            angles[f"elbow_dirx_{s}"], angles[f"elbow_diry_{s}"], angles[f"elbow_dirz_{s}"] = (float(v_) for v_ in ed)
            ue, fe = unit(elbow - S), unit(S + u * dd - elbow)
            Rcl = R2[f"clavicle_{s}"]
            pose.rot[f"upperarm_{s}"] = Rcl.T @ Ru
            pose.rot[f"lowerarm_{s}"] = Ru.T @ Rf
            pose.rot[f"hand_{s}"] = _clamp_rot(Rf.T @ Rh, WRIST_MAX,  # (a target past the wrist's range: as far as it goes)
                                               (self._prev_hand or {}).get(s))
            elev = angle(ue, trunk @ DOWN)
            eflex = angle(ue, fe)
            wdev = wsw
            angles.update({f"shoulder_{s}": elev, f"elbow_{s}": eflex, f"wrist_{s}": wdev})
            if t is not None:
                wr = S + u * dd                                # the arm in the body: the trunk makes room too (the
                pen = max(self.torso_pen(S + (elbow - S) * f_ if f_ <= 1.0 else elbow + (wr - elbow) * (f_ - 1.0),
                                         r_, torso) for f_, r_ in self.ARM_SAMPLES)   # elbow search alone can't)
                if pen > 0.0:                                  # (bounded: never worth a squat or a missed reach)
                    terms[f"arm_body_{s}"] = ((min(pen, 6.0) / 1.5) ** 2 + max(0.0, pen - 6.0)) * 0.8 * wgt
                back_e = -float((trunk.T @ (elbow - S))[2]) - self.BEHIND_ELBOW     # cm behind the chest's line
                back_w = -float((trunk.T @ (wr - S))[2]) - self.BEHIND_WRIST
                if back_e > 0.0 or back_w > 0.0:                # no one works with the arms behind the back
                    terms[f"behind_{s}"] = ((max(0.0, back_e) / 3.0) ** 2 + (max(0.0, back_w) / 2.0) ** 2) * 4.0 * wgt
                terms[f"shoulder_{s}"] = (max(0.0, elev - 20.0) / 45.0) ** 2 * wgt
                terms[f"elbow_{s}"] = ((eflex - 80.0) / 55.0) ** 2 * 0.3 * wgt
                terms[f"wrist_{s}"] = (wdev / 30.0) ** 2 * 0.6 * wgt + (max(0.0, abs(tau) - 70.0) / 20.0) ** 2 * wgt
            self._grip(pose, s, gname)
            if t is None:                                      # a hanging arm under a bent-over chest: swung clear
                self._clear_free_arm(pose, s, sx, torso)
        # head and neck: look at the work
        if look is not None:
            R3, P3 = fk(sk, pose, ["head"])
            dloc = R3["spine_03"].T @ (np.asarray(look, float) - P3["head"])
            yw = math.degrees(math.atan2(dloc[0], dloc[2]))
            pt = math.degrees(math.atan2(-dloc[1], math.hypot(dloc[0], dloc[2])))
            ywc, ptc = max(-75.0, min(75.0, yw)), max(-45.0, min(70.0, pt))
            miss = abs(yw - ywc) + abs(pt - ptc)
            ywc -= max(-12.0, min(12.0, ywc * 0.25))               # the eyes take part of the gaze: up to 12 deg
            ptc -= max(-8.0, min(22.0, ptc * 0.4))                 # sideways, 22 deg down (comfortable eye depression)
            pose.rot["neck_01"] = ry(ywc * 0.4) @ rx(ptc * 0.45)
            pose.rot["head"] = ry(ywc * 0.6) @ rx(ptc * 0.55)
            terms["neck"] = (max(0.0, ptc - 15.0) / 30.0) ** 2 * 0.6 + (ywc / 45.0) ** 2 * 0.3 + (miss / 10.0) ** 2
            angles.update({"neck_flexion": ptc, "neck_twist": ywc})
        # trunk and balance
        lean = 0.0
        for t in targets:
            if t.weight >= 0.5:
                sh = self.sk.world[f"upperarm_{t.side}"]
                v_ = np.asarray(t.point, float) - sh
                frac = float(norm3(v_)) / (sum(self.L[t.side]) + self.hand_len * 0.45)
                if v_[2] > 0 and v_[1] < 5.0:                       # forward and not above the shoulder
                    lean = max(lean, min(1.0, max(0.0, (frac - 0.55) / 0.4)) * 14.0)
        hinge_eff = hinge + flex - lean
        # a forward bend is shared by the pelvis and the lower back (lumbo-pelvic rhythm, about 55 : 45 through the
        # working range): all of it at the hips is a stiff, hinged back, all of it in the spine a hunch
        split = (flex - LUMBAR_SHARE * max(0.0, hinge + flex)) / 12.0
        terms["trunk"] = (hinge_eff / 28.0) ** 2 + (min(0.0, hinge) / 2.0) ** 2 + split ** 2 * 0.5 + (lat / 10.0) ** 2 + (twist / 14.0) ** 2             + (min(0.0, hinge + flex) / 6.0) ** 2                       # leaning back is rare and costly
        terms["pelvis"] = (dx / 8.0) ** 2 * 0.3 + (dz / 10.0) ** 2 * 0.3 + (yaw / 25.0) ** 2 * 0.4
        Rall, Pall = fk(sk, pose)
        if self.keep_out:
            terms["keep_out"] = keep_out_cost(Pall, self.keep_out)
        if self.field is not None:
            terms["collide"] = self.field.cost(Pall, self._world)
        # knees: a deep knee bend is real effort too (so squatting is chosen over stooping, not for free)
        terms["knees"] = (max(0.0, drop - 6.0) / 60.0) ** 2 * 0.5
        c = com(Pall)
        feet = []
        for s in ("l", "r"):
            a = np.asarray(getattr(stance, "left" if s == "l" else "right"), float)
            for ox, oz in ((-5, -6), (5, -6), (-5, 18), (5, 18)):
                q = a + ry(stance.foot_yaw(s)) @ np.array([ox, 0, oz])
                feet.append((q[0], q[2]))
        terms["balance"] = (_outside((c[0], c[2]), _hull(feet)) / 1.5) ** 2 * (20.0 if not swing else 2.0)
        angles.update({"trunk_flexion": angle(Rall["spine_03"] @ UP, UP), "trunk_side": lat, "trunk_twist": twist,
                       "knee_l": angle(Pall["calf_l"] - Pall["thigh_l"], Pall["foot_l"] - Pall["calf_l"]),
                       "knee_r": angle(Pall["calf_r"] - Pall["thigh_r"], Pall["foot_r"] - Pall["calf_r"])})
        return pose, terms, angles, (Rall, Pall)

    # a working arm behind the back (trunk frame, from the shoulder joint, cm): elbows drawn back past this, or the
    # wrist behind the shoulder line. Measured on the reviewed library clips: elbows up to 17 cm back, wrists in front
    BEHIND_ELBOW = 16.0
    BEHIND_WRIST = 2.0
    HANDLE_DEPTH = 0.7              # cm from the palm centre's surface to a held handle's skin (the authored grips)
    FINGER_HALF = 0.9               # cm from a finger's bone line to its palm-side skin

    def wrap(self, s, f, radius):
        """Joint flexions (deg) that close finger f onto a handle of `radius` cm held across the palm: the handle's
        axis lies along the thumb direction, `radius + HANDLE_DEPTH` out from the palm centre (where a grip's
        HandTarget puts it), and each joint bends, base first, just until the next joint of the finger reaches the
        handle's skin plus the finger's half thickness - the fingers stop on the handle. (A curl from chord angles
        alone closed them past a thin handle: it hung below the fingertips, held as if by the middle fingers.)"""
        key = (s, f, round(radius, 2))
        if key in self._wrap:
            return self._wrap[key]
        sk, sx = self.sk, (1.0 if s == "l" else -1.0)
        h = sk.world[f"hand_{s}"]
        rest = [sk.world[f"{f}_0{i}_{s}"] - h for i in (1, 2, 3)] + [sk.tips[f"{f}_tip_{s}"] - h]
        C = np.array([sx * self.hand_len * 0.40, -1.6 - radius - self.HANDLE_DEPTH])     # handle axis (x, y)
        Rr = radius + self.FINGER_HALF
        lim = (90.0, 105.0, 80.0)
        pos, acc, out = rest[0][:2], 0.0, []
        for i in range(3):
            seg = (rest[i + 1] - rest[i])[:2]
            best = lim[i]
            for th in np.arange(0.0, lim[i] + 1e-6, 0.5):
                a = math.radians(-sx * (acc + th))
                q = pos + np.array([seg[0] * math.cos(a) - seg[1] * math.sin(a), seg[0] * math.sin(a) + seg[1] * math.cos(a)])
                if float(np.linalg.norm(q - C)) <= Rr:
                    best = th
                    break
            acc += best
            a = math.radians(-sx * acc)
            pos = pos + np.array([seg[0] * math.cos(a) - seg[1] * math.sin(a), seg[0] * math.sin(a) + seg[1] * math.cos(a)])
            out.append(float(best))
        self._wrap[key] = tuple(out)
        return self._wrap[key]

    def torso_profile(self):
        return torso_profile_of(self.sk)

    @classmethod
    def arm_in_body(cls, R, P, prof):
        """cm each arm (l, r) reaches into the trunk at worst, for posed joints R, P."""
        tor = (P["pelvis"], P["spine_03"] + R["spine_03"] @ UP * 6.0, R["spine_03"], prof)
        out = []
        for s_ in ("l", "r"):
            S, E, W = P[f"upperarm_{s_}"], P[f"lowerarm_{s_}"], P[f"hand_{s_}"]
            out.append(max(cls.torso_pen(S + (E - S) * f if f <= 1.0 else E + (W - E) * (f - 1.0), r, tor)
                           for f, r in cls.ARM_SAMPLES))
        return out

    TORSO_TOL = 1.5                 # cm of contact (sleeve on the side of the chest, soft tissue) before it counts

    @classmethod
    def torso_pen(cls, q, r, torso):
        """cm a sphere (centre q, radius r) reaches into the trunk (<= 0: clear); torso = (pelvis, chest top,
        trunk rotation, profile)."""
        a, b, Rt, prof = torso
        ax = b - a
        t = min(1.0, max(0.0, float(np.dot(q - a, ax) / (np.dot(ax, ax) + 1e-9))))
        d = Rt.T @ (q - (a + ax * t))
        k = next((i for i in range(1, len(prof)) if prof[i][0] >= t), len(prof) - 1)
        t0, t1 = prof[k - 1][0], prof[k][0]
        u = 0.0 if t1 <= t0 else min(1.0, max(0.0, (t - t0) / (t1 - t0)))
        A, F, Bk = (prof[k - 1][j] * (1 - u) + prof[k][j] * u for j in (1, 2, 3))
        x, z = float(d[0]), float(d[2])
        Bz = F if z >= 0.0 else Bk
        e = math.hypot(x / (A + r), z / (Bz + r))
        return (1.0 - e) * min(A + r, Bz + r) - cls.TORSO_TOL

    # along the upper arm (0-1) and forearm (1-2): from below the armpit down (higher up, the upper arm lies against
    # the side of the chest - contact, and the trunk measurement there includes the shoulder of the clothes)
    ARM_SAMPLES = ((0.85, 4.5), (1.0, 4.5), (1.3, 4.0), (1.6, 3.5))

    def _clear_free_arm(self, pose, s, sx, torso):
        """A recorded free arm placed by direction on a bulkier body can sit inside it: swing it out from the
        shoulder (about the trunk's forward axis) just until it is clear."""
        for _ in range(10):
            R, P = fk(self.sk, pose, [f"hand_{s}"])
            S, E, W = P[f"upperarm_{s}"], P[f"lowerarm_{s}"], P[f"hand_{s}"]
            pen = max(self.torso_pen(q, r, torso) for q, r in
                      ((S + (E - S) * f, r) if f <= 1.0 else (E + (W - E) * (f - 1.0), r) for f, r in self.ARM_SAMPLES))
            if pen <= self.TORSO_TOL:
                return
            Rot = axis_angle(R["spine_03"] @ np.array([0.0, 0.0, 1.0]), math.radians(sx * 4.0))
            Rcl = R[f"clavicle_{s}"]
            pose.rot[f"upperarm_{s}"] = Rcl.T @ Rot @ R[f"upperarm_{s}"]

    def _arm(self, s, sx, S, u, a, h, dd, trunk, Rh, has_target, prev=None, torso=None, ref_elbow=None):
        """Elbow swivel + forearm twist: try elbow positions around the shoulder-wrist line (low and slightly out
        is preferred), give the forearm up to +-85 deg of the hand's roll (pronation / supination), leave the rest
        to the wrist; keep the swivel that bends the wrist least."""
        base_pole = trunk @ unit(np.array([sx * 0.35, -1.0, -0.25]))
        base_pole = unit(base_pole - u * np.dot(base_pole, u))
        xa = np.array([sx, 0.0, 0.0])
        best = None
        cands = (-75.0, -50.0, -30.0, -15.0, 0.0, 15.0, 30.0, 50.0, 75.0)
        pe = None
        if prev is not None and len(prev) > 2 and prev[2] is not None:
            pe = trunk @ np.asarray(prev[2], float)             # last frame's elbow direction
        if prev is not None and prev[0] is not None:            # search near the last frame's elbow (coherent)
            p0 = prev[0]
            near = [p0]
            if pe is not None:
                # the swivel is measured round the shoulder-wrist line: when that line swings (the hand passing
                # close to the shoulder) the same swivel is another elbow. Search round the swivel that puts the
                # elbow back where it was, too.
                ps = pe - u * np.dot(pe, u)
                if norm3(ps) > 1e-3:
                    ps = unit(ps)
                    ang = math.degrees(math.atan2(float(np.dot(cross3(base_pole, ps), u)), float(np.dot(base_pole, ps))))
                    if abs(ang * sx - p0) > 4.0:                # (only when it differs: half the work otherwise)
                        near.append(ang * sx)
            cands = tuple(sorted({max(-90.0, min(90.0, c + d)) for c in near for d in (-12.0, -6.0, -3.0, 0.0, 3.0, 6.0, 12.0)}))
        if ref_elbow is not None:                               # (a recorded elbow: search round it too)
            ps = ref_elbow - u * np.dot(ref_elbow, u)
            if norm3(ps) > 1e-3:
                ps = unit(ps)
                ang = math.degrees(math.atan2(float(np.dot(cross3(base_pole, ps), u)), float(np.dot(base_pole, ps))))
                cands = tuple(sorted(set(cands) | {max(-90.0, min(90.0, ang * sx + d)) for d in (-8.0, 0.0, 8.0)}))
        for sw in cands:
            pole = axis_angle(u, math.radians(sw * sx)) @ base_pole
            elbow = S + u * a + pole * h
            ue, fe = unit(elbow - S), unit(S + u * dd - elbow)
            bend = fe - ue * np.dot(fe, ue)
            Ru = frame_rot(xa, FWD, ue, bend if norm3(bend) > 1e-6 else trunk @ FWD)
            n = cross3(ue, fe)
            Rf0 = axis_angle(n, math.radians(angle(ue, fe))) @ Ru if norm3(n) > 1e-9 else Ru
            Q = Rf0.T @ Rh                                     # hand relative to the untwisted forearm
            w_, x_, y_, z_ = _quat(Q)
            tau = math.degrees(2.0 * math.atan2(x_ * sx, w_)) if abs(w_) + abs(x_) > 1e-9 else 0.0
            tau = (tau + 180.0) % 360.0 - 180.0
            # the twist continues, it never wraps around - judged on the roll the hand really needed last frame
            # (the forearm took at most 85 of it: comparing with that clipped value flipped it past half a turn)
            ref = prev[3] if prev is not None and len(prev) > 3 and prev[3] is not None else                 (prev[1] if prev is not None else None)
            if ref is not None:
                tau = min((tau - 360.0, tau, tau + 360.0), key=lambda x: abs(x - ref))
            tc = max(-85.0, min(85.0, tau))
            Rf = Rf0 @ axis_angle(xa, math.radians(tc))
            wsw = angle(Rf @ xa, Rh @ xa)                       # wrist bend left after the forearm twist
            roll = abs(tau - tc)                                # roll the forearm could not take
            elbow_up = max(0.0, float(np.dot(ue, trunk @ UP)) + 0.3)
            score = (wsw / 30.0) ** 2 + (roll / 20.0) ** 2 + (sw / 60.0) ** 2 * 0.3 + elbow_up ** 2 * 2.0
            if torso is not None:                              # the arm never goes through the body
                wr = S + u * dd
                for f_, r_ in self.ARM_SAMPLES:
                    q = S + (elbow - S) * f_ if f_ <= 1.0 else elbow + (wr - elbow) * (f_ - 1.0)
                    pen = self.torso_pen(q, r_, torso)
                    if pen > 0.0:
                        score += (pen / 1.5) ** 2 * 2.0
            if prev is not None and prev[0] is not None:
                cont = angle(ue, pe) / 8.0 if pe is not None else (sw - prev[0]) / 10.0   # the elbow itself stays put
                score += cont ** 2 + ((tc - (prev[1] or 0.0)) / 25.0) ** 2
            if ref_elbow is not None:
                score += (angle(ue, ref_elbow) / 20.0) ** 2
            if best is None or score < best[0]:
                best = (score, Ru, Rf, wsw + roll, tc, elbow, sw, tau)
        self._raw_twist = best[7]
        return best[1], best[2], best[3], best[4], best[5], best[6]

    def finger_angles(self, s, name):
        """Hand shape -> {finger: (flex1, flex2, flex3)} (deg; thumb: (opposition, flex2, flex3)).
        name: a preset ("relaxed", "open", "flat", "point", "pinch", "tripod", "lateral"), a wrap "power:<r>" /
        "hook:<r>" around a handle of radius r cm, "trigger:<r>:<pull 0-1>" (index on a trigger), or a blend
        ("blend", a, b, u) - so hand shapes change smoothly from frame to frame."""
        if isinstance(name, tuple) and name and name[0] == "blend":
            A, Bn, u = self.finger_angles(s, name[1]), self.finger_angles(s, name[2]), float(name[3])
            return {k: tuple(x * (1 - u) + y * u for x, y in zip(A[k], Bn[k])) for k in A}
        kind, *args = str(name).split(":")
        out = {}
        if kind in ("power", "hook", "trigger") and args:
            r = float(args[0])
            for f in ("index", "middle", "ring", "pinky"):
                out[f] = self.wrap(s, f, r)
            if kind == "trigger":                              # index finger on the trigger, pulled by `pull`
                pull = float(args[1]) if len(args) > 1 else 0.0
                out["index"] = (18.0 + 22.0 * pull, 40.0 + 30.0 * pull, 25.0 + 20.0 * pull)
            out["thumb"] = self.thumb(s, "hook" if kind == "hook" else ("wrap_mid" if kind == "trigger" else "wrap"),
                                      out, r)
            return out
        P = {"relaxed": ((14, 14, 10), (8, 10, 8)),
             "open": ((5, 5, 3), (2, 4, 4)),
             "flat": ((2, 0, 0), (-5, 0, 0)),
             "power": ((62, 72, 48), (34, 30, 25)),
             "hook": ((35, 80, 60), (4, 10, 10)),
             "pinch": ((38, 42, 28), (40, 32, 26)),
             "point": ((0, 0, 0), (25, 20, 10)),
             "tripod": ((40, 40, 25), (42, 28, 22)),
             "lateral": ((68, 80, 50), (12, 18, 22))}
        (f1, f2, f3), th = P.get(kind, P["relaxed"])
        spread = {"index": 1.0, "middle": 1.05, "ring": 1.1, "pinky": 1.15}
        for f in ("index", "middle", "ring", "pinky"):
            out[f] = (f1 * spread[f], f2 * spread[f], f3 * spread[f])
        if kind == "point":                                     # index straight, the others tucked
            for f in ("middle", "ring", "pinky"):
                out[f] = (70.0, 85.0, 55.0)
        if kind in ("pinch", "tripod"):                         # spare fingers tuck in
            for f in ("ring", "pinky") if kind == "tripod" else ("middle", "ring", "pinky"):
                out[f] = (55.0, 70.0, 45.0)
        mode = {"pinch": "pinch", "tripod": "tripod", "lateral": "lateral", "power": "wrap", "hook": "hook",
                "point": "wrap_mid", "relaxed": "relaxed", "open": "open", "flat": "open"}.get(kind, "relaxed")
        out["thumb"] = self.thumb(s, mode, out, 1.5)
        return out

    def contact_offset(self, s, grip):
        """Where a fingertip grip holds, from the palm centre (the point a HandTarget places), in the hand's own
        (fingers, palm, thumb side) axes, cm - from this hand's finger lengths and the grip's curl. A fixed set-back
        put the pinch of a long- or short-fingered character several cm off the nut it was meant to hold."""
        key = ("contact", s, str(grip))
        if key in self._thumb_cache:
            return self._thumb_cache[key]
        sx = 1.0 if s == "l" else -1.0
        ang = self.finger_angles(s, grip)
        kind = str(grip).split(":")[0]
        ix = self._finger_pts(s, "index", ang["index"])[3]
        th, _ = self._thumb_tip(s, ang["thumb"])
        pt = (ix + th) / 2.0
        if kind == "tripod":
            pt = (ix + self._finger_pts(s, "middle", ang["middle"])[3] + th) / 3.0
        hl = self.hand_len
        rel = pt - np.array([sx * hl * 0.40, -1.6, 0.0])
        out = (float(rel[0] * sx), float(-rel[1]), float(rel[2]))      # along the fingers, towards the palm, thumb side
        self._thumb_cache[key] = out
        return out

    # ------------------------------------------------------------------ thumb
    def _finger_pts(self, s, f, curl):
        """Hand-frame (rest hand orientation, origin at the wrist) joint points of a curled finger: base, mid, end, tip."""
        sk, sx = self.sk, (1.0 if s == "l" else -1.0)
        h = sk.world[f"hand_{s}"]
        names = [f"{f}_01_{s}", f"{f}_02_{s}", f"{f}_03_{s}"]
        rest = [sk.world[n] - h for n in names] + [sk.tips[f"{f}_tip_{s}"] - h]
        pts = [rest[0]]
        R = I3
        for i in range(3):
            R = R @ rz(-sx * curl[i])
            pts.append(pts[-1] + R @ (rest[i + 1] - rest[i]))
        return pts

    def _thumb_axes(self, s):
        """Anatomical axes from the thumb's own rest geometry: flexion bends the thumb towards the palm centre;
        opposition swings it across under the palm."""
        key = ("thax", s)
        if key in self._thumb_cache:
            return self._thumb_cache[key]
        sk = self.sk
        h = sk.world[f"hand_{s}"]
        t1, t2, t3 = (sk.world[f"thumb_0{i}_{s}"] - h for i in (1, 2, 3))
        tip = sk.tips[f"thumb_tip_{s}"] - h
        palm_c = (sk.world[f"middle_01_{s}"] - h) * 0.6 + np.array([0.0, -2.5, 0.0])   # palm centre, palm side
        out = []
        for a_, b_ in ((t2, t3), (t3, tip)):
            d = unit(b_ - a_)
            n_in = unit((palm_c - a_) - d * np.dot(palm_c - a_, d))
            out.append(unit(cross3(d, n_in)))
        self._thumb_cache[key] = (t1, t2, t3, tip, out[0], out[1])
        return self._thumb_cache[key]

    def _thumb_tip(self, s, x):
        t1, t2, t3, tip, k2, k3 = self._thumb_axes(s)
        rv = np.radians(np.asarray(x[:3], float))
        ang = float(norm3(rv))
        R1 = axis_angle(rv / ang, ang) if ang > 1e-9 else I3
        R2 = R1 @ axis_angle(k2, math.radians(x[3]))
        R3 = R2 @ axis_angle(k3, math.radians(x[4]))
        p2 = t1 + R1 @ (t2 - t1)
        p3 = p2 + R2 @ (t3 - t2)
        return p3 + R3 @ (tip - t3), (p2, p3)

    def thumb(self, s, mode, fingers, radius=1.5):
        """Thumb joint values (thumb_01 rotation vector deg x3, thumb_02 / 03 flexion deg) that put the thumb tip
        where it goes for the grip - solved against the curled fingers, so the thumb wraps, pinches or rests the way
        a real one does instead of swinging about a fixed axis."""
        key = (s, mode, tuple(round(v, 1) for f in ("index", "middle") for v in fingers[f]), round(radius, 1))
        if key in self._thumb_cache:
            return self._thumb_cache[key]
        t1, t2, t3, tip0, k2, k3 = self._thumb_axes(s)
        ix = self._finger_pts(s, "index", fingers["index"])
        md = self._finger_pts(s, "middle", fingers["middle"])
        side = np.array([0.0, 0.0, 1.0])                      # thumb side of the hand (both hands: +Z in the rest pose)
        palm = np.array([0.0, -1.0, 0.0])
        if mode == "pinch":
            tgt = ix[3] + side * 0.9 + palm * 0.6                 # pad to pad with the index tip
        elif mode == "tripod":
            tgt = (ix[3] + md[3]) / 2 + side * 0.6 + palm * 0.9
        elif mode == "lateral":
            tgt = (ix[1] + ix[2]) / 2 + side * 1.4                # key grip: on the side of the index middle phalanx
        elif mode in ("wrap", "wrap_mid"):
            # around the handle: the tip over the curled index (power) / middle (pointing, trigger) middle phalanx
            f = ix if mode == "wrap" else md
            tgt = (f[1] + f[2]) / 2 + side * 1.2 - palm * 0.6
        elif mode == "hook":
            tgt = tip0 + (ix[1] - tip0) * 0.25
        elif mode == "relaxed":
            tgt = tip0 + ((ix[1] + ix[2]) / 2 - tip0) * 0.35
        else:                                                    # open / flat
            tgt = tip0.copy()

        def cost(x):
            p, (p2, p3) = self._thumb_tip(s, x)
            c = float(np.sum((p - tgt) ** 2))
            c += 0.0004 * float(np.sum(np.asarray(x[:3]) ** 2)) + 0.0002 * (x[3] ** 2 + x[4] ** 2)
            c += 0.002 * (x[4] - 0.9 * x[3]) ** 2                # the two thumb joints bend together
            for q in (p2, p3, p):                                # never through the index finger
                dmin = min(float(norm3(q - (ix[i] + ix[i + 1]) / 2)) for i in range(3))
                c += max(0.0, 1.6 - dmin) ** 2 * 20.0
            return c
        lo = np.array([-60.0, -60.0, -60.0, -15.0, -15.0])
        hi = np.array([60.0, 60.0, 60.0, 80.0, 85.0])
        x, _ = nelder_mead(cost, np.array([0.0, 0.0, 0.0, 20.0, 18.0]), np.array([15.0, 15.0, 15.0, 15.0, 15.0]),
                           lo, hi, iters=400)
        x, _ = nelder_mead(cost, x, np.array([4.0, 4.0, 4.0, 4.0, 4.0]), lo, hi, iters=200)
        out = tuple(float(v) for v in x)
        self._thumb_cache[key] = out
        return out

    def _grip(self, pose, s, name):
        sx = 1.0 if s == "l" else -1.0
        A = self.finger_angles(s, name)
        for f in ("index", "middle", "ring", "pinky"):
            for i, a in zip((1, 2, 3), A[f]):
                pose.rot[f"{f}_0{i}_{s}"] = rz(-sx * a)
        x = A["thumb"]
        _, _, _, _, k2, k3 = self._thumb_axes(s)
        rv = np.radians(np.asarray(x[:3], float))
        ang = float(norm3(rv))
        pose.rot[f"thumb_01_{s}"] = axis_angle(rv / ang, ang) if ang > 1e-9 else I3
        pose.rot[f"thumb_02_{s}"] = axis_angle(k2, math.radians(x[3]))
        pose.rot[f"thumb_03_{s}"] = axis_angle(k3, math.radians(x[4]))

    # ------------------------------------------------------------------ solving
    def low_work(self, targets):
        """Hands working below mid-thigh: squatting has to be among the postures tried, not only stooping."""
        hip = float(self.sk.world["thigh_l"][1])
        return any(t.weight >= 0.5 and float(t.point[1]) < hip * 0.75 for t in targets)

    def solve(self, targets, look=None, stance=None, prev=None, smooth=1.0, seeds=None, swing=None, base=None,
              world=None):
        self._world = world
        self._prev_hand = {s_: prev.pose.local(f"hand_{s_}") for s_ in ("l", "r")} if isinstance(prev, Solution) else None
        stance = stance or self.default_stance()
        prev_arm = _prev_arm(prev)
        if base is not None:                       # mocap-driven frame: no search, just exact limbs on the base pose
            v = np.zeros(8) if getattr(stance, "free", False) else self.fit_pelvis(base, stance)
            if self.base_correct:                  # generated base: small corrections (keep-outs, reach, feet)
                prev_v = prev.v if isinstance(prev, Solution) else None
                v0 = v.copy()
                lo_c = np.array([-12.0, -6.0, -15.0, -20.0, -15.0, -12.0, -12.0, -20.0])
                hi_c = np.array([12.0, 25.0, 15.0, 20.0, 20.0, 15.0, 12.0, 20.0])

                def fb(x):
                    _, tm, _, _ = self.build(x, targets, look, stance, swing=swing, base=base, prev_arm=prev_arm)
                    tm.pop("trunk", None)              # the base trunk is the natural one: penalise change, not lean
                    c = sum(tm.values()) + float(np.sum(((x - v0) / SCALE) ** 2)) * 0.15
                    if prev_v is not None:
                        c += float(np.sum(((x - prev_v) / SCALE) ** 2)) * 0.5
                    return c
                x0 = prev_v if prev_v is not None else v0
                v, _ = nelder_mead(fb, np.clip(x0, v0 + lo_c, v0 + hi_c), SCALE * 0.5, v0 + lo_c, v0 + hi_c,
                                   iters=self._it(90 if prev_v is not None else 200))
            pose, terms, angles, _ = self.build(v, targets, look, stance, swing=swing, base=base, prev_arm=prev_arm)
            return Solution(pose, float(sum(terms.values())), terms, angles, v)
        prev_v = prev.v if isinstance(prev, Solution) else (None if prev is None else np.asarray(prev, float))

        def f(v):
            _, terms, _, _ = self.build(v, targets, look, stance, swing=swing, prev_arm=prev_arm)
            c = sum(terms.values())
            if prev_v is not None:
                c += smooth * float(np.sum(((v - prev_v) / SCALE) ** 2)) * 0.5
            return c
        starts = [prev_v] if prev_v is not None else (seeds or [np.zeros(8), np.array([0, 25, 0, 0, 15, 5, 0, 0.0]),
                                                                  np.array([0, 4, 3, 0, 35, 15, 0, 0.0])])
        lo, hi = LO.copy(), HI.copy()
        if getattr(stance, "drop", None) is not None:              # kneeling / squatting: the pelvis height is given
            lo[1], hi[1] = stance.drop - 3.0, stance.drop + 3.0
            if prev_v is not None:                                  # and an upright fresh start: no inherited lean
                starts = list(starts) + [np.array([0.0, stance.drop, 0.0, 0.0, 5.0, 3.0, 0.0, 0.0])]
            starts = [np.clip(np.asarray(x0, float), lo, hi) for x0 in starts]
        if getattr(stance, "drop", None) is None and self.low_work(targets):     # squat as well as stoop
            hip = float(self.sk.world["thigh_l"][1])
            low = min(float(t.point[1]) for t in targets if t.weight >= 0.5)
            d1 = min(HI[1] - 2.0, max(15.0, (hip - low) * 0.55))
            starts = list(starts) + [np.array([0.0, d1, 4.0, 0.0, 20.0, 8.0, 0.0, 0.0]),
                                     np.array([0.0, min(HI[1] - 2.0, d1 + 15.0), 6.0, 0.0, 30.0, 12.0, 0.0, 0.0])]
        if getattr(stance, "hinge", None) is not None:
            lo[4], hi[4] = stance.hinge
        best = None
        for x0 in starts:
            x, fx = nelder_mead(f, np.asarray(x0, float), SCALE, lo, hi, iters=self._it(220 if prev_v is None else 90))
            x, fx = nelder_mead(f, x, SCALE * 0.3, lo, hi, iters=self._it(80))    # polish
            if best is None or fx < best[1]:
                best = (x, fx)
        x, fx = best
        pose, terms, angles, _ = self.build(x, targets, look, stance, swing=swing, prev_arm=prev_arm)
        return Solution(pose, fx, terms, angles, x)

    def _it(self, n):
        return n if self.effort >= 1.0 else max(20, int(n * self.effort))

    def at(self, v, targets, look=None, stance=None, prev=None, swing=None, base=None, world=None):
        """The pose for given trunk / pelvis values, no search: the limbs still go exactly onto the targets (the
        in-between frames of a preview, the trunk blended between two solved key poses)."""
        self._world = world
        self._prev_hand = {s_: prev.pose.local(f"hand_{s_}") for s_ in ("l", "r")} if isinstance(prev, Solution) else None
        stance = stance or self.default_stance()
        pose, terms, angles, _ = self.build(v, targets, look, stance, swing=swing, base=base, prev_arm=_prev_arm(prev))
        return Solution(pose, float(sum(terms.values())), terms, angles, np.asarray(v, float))

    def fit_pelvis(self, base, stance, knee=0.975):
        """Pelvis drop / fore-aft shift so both feet reach their stance points with the knees just unlocked:
        the pelvis is lowest in double support and rides over the planted foot - as in real walking."""
        sk = self.sk
        v = np.zeros(8)
        for _ in range(3):
            pose = base.copy()
            pose.pelvis = base.pelvis + np.array([v[0], -v[1], v[2]])
            R, P = fk(sk, pose, ["thigh_l", "thigh_r"])
            need_drop, dz = 0.0, []
            for s in ("l", "r"):
                ank = np.asarray(getattr(stance, "left" if s == "l" else "right"), float)
                hip = P[f"thigh_{s}"]
                Lmax = sum(self.leg[s]) * knee
                horiz = math.hypot(ank[0] - hip[0], ank[2] - hip[2])
                if horiz >= Lmax:
                    dz.append((ank[2] - hip[2]) - math.copysign(Lmax * 0.95, ank[2] - hip[2]))
                    horiz = Lmax * 0.95
                y_ok = ank[1] + math.sqrt(Lmax * Lmax - horiz * horiz)
                need_drop = max(need_drop, hip[1] - y_ok)
            v[1] += max(0.0, need_drop)
            if dz:
                v[2] += float(np.mean(dz)) * 0.8
            if need_drop <= 0.05 and not dz:
                break
        return v

    def default_stance(self, width=None, yaw=0.0):
        sk = self.sk
        w = width if width is not None else abs(sk.world["thigh_l"][0] - sk.world["thigh_r"][0]) + 4.0   # ~ hip width
        y = sk.world["foot_l"][1]
        return Stance(left=(w / 2, y, 0.0), right=(-w / 2, y, 0.0), yaw=yaw)
