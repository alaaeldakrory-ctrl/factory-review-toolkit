"""Walking: a floor path in, frames out (root placement per frame + planted / swinging feet + arm swing).

- Speed profile: accelerate over the first step, cruise at the worker's pace, slow into the last step.
- Footsteps: alternate feet, step length ~0.41 x stature, plants offset half the stance width from the path and turned
  with the heading; a foot swings while the root covers the middle 80 % of its step (double support between), lifting
  in a sine arc with the toe pitched up.
- Turns: the root heading follows the path; at the end the worker turns to the work direction with two short
  shuffle steps (as people do) instead of spinning on the spot.
- Arms swing opposite to the legs (free hands only - a hand holding something keeps its target).

All positions are world XZ (cm); frames carry the root (x, z, yaw) and the stance in the root's local frame.
"""
import math

import numpy as np

from .motion_gen import FPS, Frame, ease
from .solver import Stance


def _yaw_of(d):
    return math.degrees(math.atan2(d[0], d[1]))          # 0 = facing +Z (world), positive towards +X


def _rot2(yaw):
    a = math.radians(yaw)
    return np.array([[math.cos(a), math.sin(a)], [-math.sin(a), math.cos(a)]])   # local x,z -> world x,z


def _wrap(a):
    return (a + 180.0) % 360.0 - 180.0


def path_length(pts):
    return sum(float(np.linalg.norm(np.subtract(b, a))) for a, b in zip(pts, pts[1:]))


def _sample(pts, s):
    """Point and heading at arc length s along a polyline."""
    acc = 0.0
    for a, b in zip(pts, pts[1:]):
        a, b = np.asarray(a, float), np.asarray(b, float)
        L = float(np.linalg.norm(b - a))
        if acc + L >= s or b is pts[-1]:
            u = 0.0 if L < 1e-9 else min(1.0, max(0.0, (s - acc) / L))
            return a + (b - a) * u, _yaw_of(b - a) if L > 1e-9 else None
        acc += L
    b = np.asarray(pts[-1], float)
    return b, _yaw_of(np.subtract(pts[-1], pts[-2])) if len(pts) > 1 else None


def _smooth(path, D):
    """Corner-rounded point and heading at arc length s (moving average over ~90 cm, heading from +-50 cm)."""
    def f(s):
        pts = [_sample(path, min(D, max(0.0, s + o)))[0] for o in (-45, -30, -15, 0, 15, 30, 45)]
        p = np.mean(pts, axis=0)
        if s <= 1e-6:
            p = _sample(path, 0.0)[0]
        elif s >= D - 1e-6:
            p = _sample(path, D)[0]
        else:                                                    # blend to the true end points near the ends
            k = min(1.0, s / 45.0, (D - s) / 45.0)
            p = _sample(path, s)[0] * (1 - k) + p * k if k < 1 else p
        a, b = _sample(path, max(0.0, s - 50))[0], _sample(path, min(D, s + 50))[0]
        h = _yaw_of(b - a) if np.linalg.norm(b - a) > 1e-6 else None
        return p, h
    return f


def _mocap_base(sk, phase, w, hang):
    """Walking base pose from the mocap cycle at gait phase, faded in from the standing pose by w (0..1)."""
    from . import mocap as MC
    from .bake import blend
    if not MC.available("Walking"):
        return None
    walkp = MC.cycle("Walking").pose(sk, phase)
    if w >= 0.999:
        return walkp
    return blend(hang, walkp, w)


def _hang_pose(sk, worker):
    from .solver import PostureSolver
    key = id(sk)
    if key not in _HANG:
        _HANG[key] = PostureSolver(sk, worker).solve([], look=None).pose
    return _HANG[key]


_HANG = {}


def walk(worker, sk, path, yaw_start, yaw_end, carry=None, speed=None):
    """Frames walking `path`: with the mocap walk available, the captured person's legs, hips, trunk and arms as
    they are (mocap_walk); otherwise planned footsteps and leg IK (planned_walk)."""
    from . import locomotion as LO
    from . import mocap as MC
    if LO.available() and speed is None:
        return clip_walk(worker, sk, path, yaw_start, yaw_end, carry)
    if MC.available("Walking") and speed is None:
        return mocap_walk(worker, sk, path, yaw_start, yaw_end, carry)
    return planned_walk(worker, sk, path, yaw_start, yaw_end, carry, speed)


def _e3(u):
    u = min(1.0, max(0.0, u))
    return u * u * (3 - 2 * u)


def clip_walk(worker, sk, path, yaw_start, yaw_end, carry=None):
    """Walking assembled from motion-capture clips (locomotion.py): turn on the spot towards the path if needed,
    start walking, the walk cycle, stop (start clip reversed), turn on the spot to the work. The clips are joined
    where their leg poses match, with short cross-fades; the root follows the path at the captured person's own
    pace (all distances scaled to this body, then stretched by a few percent to fit the path exactly)."""
    from . import locomotion as LO
    from . import mocap as MC
    D = path_length(path)
    if D < 5.0:
        return turn_in_place(worker, sk, path[-1], yaw_start, yaw_end, carry)
    cy, st = LO.clip("Walking", cyclic=True), LO.clip("Female_Start_Walking")
    k = float(sk.world["pelvis"][1]) / cy.hip_height
    drop = MC.ground_drop(sk, MC.cycle("Walking"))
    P = LO.plan_walk(D, k)
    sm = _smooth(path, D)
    h0 = sm(0.0)[1]
    h0 = yaw_start if h0 is None else h0
    hD = sm(D)[1]
    hD = h0 if hD is None else hD
    segs = []                                          # (kind, n_out, f(local_t) -> (joints, s, yaw_add))
    r = cy.fps / FPS                                    # clip frames per output frame (clips are 30 fps)

    def turn_seg(a_deg, s_at):
        tf = LO.turn_frames(sk, a_deg, FPS)
        return ("turn", len(tf), lambda i: (tf[min(i, len(tf) - 1)][0], s_at, tf[min(i, len(tf) - 1)][1]))
    da = _wrap(h0 - yaw_start)
    if abs(da) > 25.0:
        segs.append(turn_seg(da, 0.0))
    sc = P["stretch"] * k
    n_start = int(round(P["start_cut"] / r))
    segs.append(("start", n_start, lambda i: (st.at(i * r), st.s_at(i * r) * sc, 0.0)))
    s1 = st.s[P["start_cut"]] * sc
    per = cy.loop_dist / cy.n * sc                     # distance per cycle frame
    n_cyc = int(round(P["c_len"] / r))
    segs.append(("cycle", n_cyc, lambda i: (cy.at(P["c0"] + i * r), s1 + i * r * per, 0.0)))
    s2 = s1 + P["c_len"] * per
    n_stop = int(round(0.55 * FPS))                   # step-together: the rear foot comes up beside the front one
    sj, adv = LO.stop_frames(cy, st, P["stop_at"], P["front"], n_stop)
    segs.append(("stop", n_stop, lambda i: (sj[min(i, n_stop - 1)], s2 + adv * sc * _e3((i + 1) / n_stop), 0.0)))
    dz = _wrap(yaw_end - hD)
    if abs(dz) > 20.0:
        segs.append(turn_seg(dz, None))
    frames = []
    xf = 5                                             # cross-fade frames at each hand-over
    s_end = None
    yaw_base = yaw_start if segs[0][0] == "turn" else h0
    for si, (kind, n, f) in enumerate(segs):
        for i in range(n):
            joints, s_, yadd = f(i)
            if i < xf and si > 0 and segs[si - 1][0] != "turn" and kind not in ("turn", "stop"):
                pj, _, _ = segs[si - 1][2](segs[si - 1][1] + i)
                u = (i + 1) / (xf + 1)
                joints = {q: pj[q] * (1 - u) + joints[q] * u for q in joints}
            if kind == "turn" and s_ is None:
                s_ = s_end
            s_end = s_ if s_ is not None else s_end
            sr = min(max(s_, 0.0), D)
            p, h = sm(sr)
            if kind == "turn":
                h = (yaw_start if si == 0 else hD) + yadd
            elif h is None:
                h = h0
            pose = MC.retarget(sk, joints)
            pose.pelvis = (joints["pelvis"] - np.array([0.0, cy.hip_height, 0.0])) * k + np.array([0.0, -drop, 0.0])
            stc = Stance()
            stc.free = True
            ahead = (0.0, float(sk.world["head"][1]) - 30.0, 350.0)   # eyes on the way ahead, ~5 deg down
            frames.append(Frame(list(carry or []), ahead, {}, root=(float(p[0]), float(p[1]), float(h)), stance=stc,
                                base=pose))
    return frames


def mocap_walk(worker, sk, path, yaw_start, yaw_end, carry=None):
    """A real walk: the motion capture played on this body (legs too - nothing re-solved), the root travelling at
    the capture's own pace scaled to the leg length, so a planted foot stays planted. Starts and stops blend from /
    to the standing pose over half a step; the heading follows the corner-rounded path."""
    from . import mocap as MC
    from .bake import blend
    cyc = MC.cycle("Walking")
    k = float(sk.world["pelvis"][1]) / cyc.hip_height
    v = cyc.speed * k                                        # cm/s
    period = cyc.n / cyc.fps                                 # s per two steps
    drop = MC.ground_drop(sk, cyc)
    D = path_length(path)
    if D < 5.0:
        return turn_in_place(worker, sk, path[-1], yaw_start, yaw_end, carry)
    sm = _smooth(path, D)
    hang = _hang_pose(sk, worker)
    t_ramp = 0.35 * period                                   # start / stop: speed ramps over ~2/3 of a step
    T = D / v + t_ramp                                       # trapezoid: cruise time + one ramp
    n = max(2, int(round(T * FPS)))

    def s_of(t):
        if t < t_ramp:
            return 0.5 * v / t_ramp * t * t
        if t > T - t_ramp:
            tr = T - t
            return D - 0.5 * v / t_ramp * tr * tr
        return 0.5 * v * t_ramp + v * (t - t_ramp)
    frames = []
    extra = int(0.3 * FPS)
    for i in range(n + extra):
        t = min(i / FPS, T)
        s = s_of(t)
        p, h = sm(min(s, D))
        h = yaw_start if h is None else h
        if i < int(0.4 * FPS):
            h = yaw_start + _wrap(h - yaw_start) * ease(i / (0.4 * FPS))
        if i >= n - int(0.6 * FPS):
            u = ease(min(1.0, (i - (n - int(0.6 * FPS))) / (0.6 * FPS + extra)))
            h = h + _wrap(yaw_end - h) * u
        # gait phase from the distance covered (feet stay planted at any speed), 0 = right toe-off
        phase = s / (v * period)
        w = min(1.0, t / t_ramp) if i < n else 0.0
        if i < n and t > T - t_ramp:
            w = min(w, (T - t) / t_ramp)
        if i >= n:
            w = 0.0
        w = ease(w)
        wp = cyc.pose(sk, phase).copy()
        wp.pelvis = wp.pelvis + np.array([0.0, -drop, 0.0])
        base = blend(hang, wp, w) if w < 0.999 else wp
        st = Stance()
        st.free = True
        frames.append(Frame(list(carry or []), None, {}, root=(float(p[0]), float(p[1]), float(h)), stance=st,
                            base=base))
    return frames


def planned_walk(worker, sk, path, yaw_start, yaw_end, carry=None, speed=None):
    """Frames walking `path` (world XZ points, first = start, last = destination), arriving facing yaw_end.
    `carry`: hand targets kept during the walk (local frame), e.g. holding a tote."""
    v = speed or worker.walk_speed
    L = 0.41 * worker.stature
    D = path_length(path)
    stance_w = abs(sk.world["foot_l"][0] - sk.world["foot_r"][0]) + 4.0
    ank_y = float(sk.world["foot_l"][1])
    frames = []
    if D < 5.0:
        return turn_in_place(worker, sk, path[-1], yaw_start, yaw_end, carry)
    sm = _smooth(path, D)
    nst = max(2, int(math.ceil(D / L)))
    L = D / nst
    # time for each arc length: trapezoid speed profile, accelerate / decelerate over one step
    acc_d = min(L, D / 2)
    t_acc = 2 * acc_d / v
    T = 2 * t_acc + (D - 2 * acc_d) / v
    n = max(2, int(T * FPS))

    def s_of(t):
        if t < t_acc:
            return 0.5 * v / t_acc * t * t
        if t > T - t_acc:
            tr = T - t
            return D - 0.5 * v / t_acc * tr * tr
        return acc_d + v * (t - t_acc)
    # foot plants along the path: k = 0 is both feet at the start; then alternate (right first)
    start_pt, _ = _sample(path, 0.0)
    plants = {"l": [(0.0, start_pt + _rot2(yaw_start) @ np.array([stance_w / 2, 0.0]), yaw_start)],
              "r": [(0.0, start_pt + _rot2(yaw_start) @ np.array([-stance_w / 2, 0.0]), yaw_start)]}
    for k in range(1, nst + 1):
        side = "r" if k % 2 == 1 else "l"
        s_k = min(k * L, D)
        p, h = sm(s_k)
        h = yaw_end if k == nst else (h if h is not None else yaw_end)
        sx = 1.0 if side == "l" else -1.0
        plants[side].append((s_k, p + _rot2(h) @ np.array([sx * stance_w / 2, 0.0]), h))
    last = "l" if nst % 2 == 1 else "r"                        # the other foot closes up beside the last step
    p, _ = sm(D)
    sx = 1.0 if last == "l" else -1.0
    plants[last].append((D + 0.5 * L, p + _rot2(yaw_end) @ np.array([sx * stance_w / 2, 0.0]), yaw_end))

    def foot_at(side, s):
        """(world xz, yaw, height, pitch) of a foot when the root is at arc length s."""
        seq = plants[side]
        for (s0, p0, h0), (s1, p1, h1) in zip(seq, seq[1:]):
            # a foot lifts just after the other one lands and lands when the body is half a step behind it, so the
            # body is always between the feet (double support ~10 % of the step)
            a, b = max(0.0, s1 - 1.4 * L), s1 - 0.5 * L
            if s1 > D:                                           # closing step beside the last one, as the body arrives
                a, b = D - 0.45 * L, D
            if s < a:
                return p0, h0, 0.0, 0.0
            if s <= b:
                u = (s - a) / max(b - a, 1e-6)
                e = ease(u)
                # toe-off (heel up, toe down) -> level -> heel strike (toe up)
                pitch = -28.0 * math.exp(-((u - 0.04) / 0.12) ** 2) + 14.0 * math.exp(-((u - 0.96) / 0.08) ** 2)
                return (p0 + (p1 - p0) * e, h0 + _wrap(h1 - h0) * e, 6.0 * math.sin(math.pi * u) ** 0.8, pitch)
        s1, p1, h1 = seq[-1]
        return p1, h1, 0.0, 0.0
    extra = int(0.45 * FPS)                                      # the closing step after the root stops
    for i in range(n + extra):
        t = min(i / FPS, T)
        s = s_of(t) if i < n else D + (i - n + 1) / extra * 0.5 * L
        sr = min(s, D)
        p, h = sm(sr)
        if i >= n - int(0.6 * FPS):                              # ease into the final heading over the last steps
            u = ease((i - (n - int(0.6 * FPS))) / (0.6 * FPS + extra))
            h0 = h if h is not None else yaw_end
            h = h0 + _wrap(yaw_end - h0) * u
        elif h is None:
            h = yaw_start
        if i < int(0.4 * FPS):                                   # turn out of the start heading over the first step
            h = yaw_start + _wrap(h - yaw_start) * ease(i / (0.4 * FPS))
        root = (float(p[0]), float(p[1]), float(h))
        Rinv = _rot2(h).T
        st = Stance()
        for side in ("l", "r"):
            fp, fh, fy, fpitch = foot_at(side, s)
            loc = Rinv @ (np.asarray(fp) - p)
            setattr(st, "left" if side == "l" else "right", (float(loc[0]), ank_y + fy, float(loc[1])))
            setattr(st, f"yaw_{side}", _wrap(fh - h))
            setattr(st, f"lift_{side}", fpitch)
        st.yaw = 0.0
        phase = s / (2 * L) * 2 * math.pi                        # arm swing: opposite arm to the stepping foot
        amp = 14.0 * min(1.0, v / 120.0) * (1.0 if i < n else max(0.0, 1 - (i - n) / extra))
        if s < acc_d:
            amp *= s / max(acc_d, 1e-6)
        swing = {"l": amp * math.sin(phase), "r": -amp * math.sin(phase)}
        if carry:
            for t_ in carry:
                swing.pop(t_.side, None)
        w = min(1.0, (s_of(min(t, T)) if i < n else D) / max(acc_d, 1e-6), 1.0) if i < n else max(0.0, 1 - (i - n + 1) / extra)
        if i < n and t > T - t_acc:
            w = min(w, (T - t) / t_acc * 0.7 + 0.3)
        base = _mocap_base(sk, s / (2 * L) - 0.05, w, _hang_pose(sk, worker))
        frames.append(Frame(list(carry or []), None, {}, root=root, stance=st, swing=swing, base=base))
    return frames


def turn_in_place(worker, sk, at, yaw0, yaw1, carry=None):
    """Turn on the spot with two shuffle steps (outer foot first)."""
    d = _wrap(yaw1 - yaw0)
    if abs(d) < 5.0:
        return []
    stance_w = abs(sk.world["foot_l"][0] - sk.world["foot_r"][0]) + 4.0
    ank_y = float(sk.world["foot_l"][1])
    n = max(int(abs(d) / 90.0 * 0.9 * FPS), 10)
    first = "l" if d < 0 else "r"
    frames = []
    p = np.asarray(at, float)
    for i in range(n):
        u = ease((i + 1) / n)
        h = yaw0 + d * u
        st = Stance(yaw=0.0)
        for side in ("l", "r"):
            lead = side == first
            w0, w1 = (0.0, 0.55) if lead else (0.45, 1.0)          # each foot re-plants in half the turn
            uu = min(1.0, max(0.0, (u - w0) / (w1 - w0)))
            fh = yaw0 + d * ease(uu)
            sx = 1.0 if side == "l" else -1.0
            fp = p + _rot2(fh) @ np.array([sx * stance_w / 2, 0.0])
            loc = _rot2(h).T @ (fp - p)
            setattr(st, "left" if side == "l" else "right", (float(loc[0]), ank_y + 5.0 * math.sin(math.pi * uu),
                                                              float(loc[1])))
            setattr(st, f"yaw_{side}", _wrap(fh - h))
        frames.append(Frame(list(carry or []), None, {}, root=(float(p[0]), float(p[1]), float(h)), stance=st))
    return frames
