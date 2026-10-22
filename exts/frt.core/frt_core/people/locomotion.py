"""Locomotion from motion capture, the way game engines do it: real clips (start walking, walk cycle, stop, turn on
the spot) played on the worker and joined where their leg poses match, with the root moving exactly as the captured
person's hips did (smoothed) - so the feet stay planted and the motion is a real person's, not a synthesis.

    plan = walk_clips(sk, D)          # [(clip, frame range / phase range, root distances)] for a path length D
    frames = mocap_walk(worker, sk, path, yaw_start, yaw_end)    (gait.py)

Clips (Mixamo, in MIXAMO_DIR): Walking (cycle), Female_Start_Walking (stand -> walk; reversed = walk -> stand),
Left_Turn_90 / Right_Turn_90 (turn on the spot, scaled to any angle).

Each clip frame is stored relative to its root: the root is the hips' floor track smoothed over about a stride
(position) and the facing from the hip line (yaw), so sway, bob and surge stay in the body and the root carries only
the travel. A joint set in that root frame is retargeted onto the canonical skeleton (mocap.retarget).
"""
import math
import os

import numpy as np

from . import mocap as MC

FEAT = ("foot_l", "foot_r", "ball_l", "ball_r", "calf_l", "calf_r", "lowerarm_l", "lowerarm_r")


def _rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _smooth(x, win, cyclic=False):
    if win <= 1:
        return x.copy()
    k = np.ones(win) / win
    if cyclic:
        xx = np.concatenate([x[-win:], x, x[:win]])
        return np.convolve(xx, k, mode="same")[win:-win]
    xx = np.concatenate([np.repeat(x[:1], win), x, np.repeat(x[-1:], win)])
    return np.convolve(xx, k, mode="same")[win:-win]


class Clip:
    """A clip as root track (forward distance s, yaw) + per-frame joint sets in the root frame."""

    def __init__(self, name, cyclic=False, smooth=None):
        from .fbx_anim import FbxClip
        clip = FbxClip(os.path.join(MC.MIXAMO_DIR, name + ".fbx"))
        pre = next(n.split(":")[0] + ":" for n in clip.names if n.endswith("Hips"))
        self.name, self.n, self.fps, self.cyclic = name, clip.frames, float(clip.fps), cyclic
        raw = []
        for f in range(self.n):
            g = clip.global_mats(f)
            raw.append({k: np.array(g[pre + v].ExtractTranslation(), float) for k, v in MC.MAP.items() if pre + v in g})
        H = np.array([r["pelvis"] for r in raw])
        hl = np.array([r["thigh_l"] - r["thigh_r"] for r in raw])
        yaw = np.unwrap(np.array([math.atan2(-d[2], d[0]) for d in hl]))      # facing +Z = 0, left turn positive
        if cyclic:                                               # a loop: remove the straight travel, then smooth
            t = np.arange(self.n)
            A = np.column_stack([t, np.ones_like(t)])
            trav = {ax: np.linalg.lstsq(A, H[:, ax], rcond=None)[0] for ax in (0, 2)}
            rx = trav[0][0] * t + trav[0][1]
            rz = trav[2][0] * t + trav[2][1]
            ry = np.full(self.n, float(np.mean(yaw)))
            self.loop_dist = math.hypot(trav[0][0], trav[2][0]) * self.n      # distance per loop (two steps)
        else:
            w = smooth or max(3, int(0.9 * self.fps))
            rx, rz = _smooth(H[:, 0], w), _smooth(H[:, 2], w)
            ry = _smooth(yaw, max(3, w // 3))
            self.loop_dist = None
        self.root_xz = np.column_stack([rx, rz])
        self.yaw = ry - ry[0] if not cyclic else ry * 0.0
        d = np.linalg.norm(np.diff(self.root_xz, axis=0), axis=1)
        self.s = np.concatenate([[0.0], np.cumsum(d)])              # distance along the root track
        self.hip_height = float(np.mean(H[:, 1])) if cyclic else float(H[0][1])
        self.local = []
        for i, r in enumerate(raw):
            o = np.array([rx[i], 0.0, rz[i]])
            Rinv = _rot_y(-(ry[i]))
            self.local.append({k: Rinv @ (v - o) for k, v in r.items()})
        self._feat = np.array([self.feature(i) for i in range(self.n)])

    def feature(self, i):
        L = self.local[i]
        p = L["pelvis"]
        return np.concatenate([L[k] - p * np.array([1, 0, 1]) for k in FEAT])

    def at(self, x):
        """Joint set at fractional frame x (looping for a cycle)."""
        if self.cyclic:
            x = x % self.n
            i0 = int(math.floor(x))
            i1 = (i0 + 1) % self.n
        else:
            x = min(max(x, 0.0), self.n - 1)
            i0 = int(math.floor(x))
            i1 = min(i0 + 1, self.n - 1)
        u = x - i0
        a, b = self.local[i0], self.local[i1]
        return {k: a[k] * (1 - u) + b[k] * u for k in a}

    def s_at(self, x):
        x = min(max(x, 0.0), self.n - 1)
        i0 = int(math.floor(x))
        i1 = min(i0 + 1, self.n - 1)
        u = x - i0
        return self.s[i0] * (1 - u) + self.s[i1] * u

    def yaw_at(self, x):
        x = min(max(x, 0.0), self.n - 1)
        i0 = int(math.floor(x))
        i1 = min(i0 + 1, self.n - 1)
        u = x - i0
        return self.yaw[i0] * (1 - u) + self.yaw[i1] * u

    def match(self, feat):
        """(best cycle frame, error) for a feature vector (cycle clips)."""
        e = np.linalg.norm(self._feat - feat[None, :], axis=1)
        i = int(np.argmin(e))
        return i, float(e[i])


def heel_strikes(cy):
    """Cycle frames where a foot has just landed in front (both feet down, widest spread): [(frame, front side)]."""
    out = []
    for side, other in (("l", "r"), ("r", "l")):
        best = None
        for i in range(cy.n):
            L = cy.local[i]
            low = max(L[f"ball_{side}"][1], L[f"ball_{other}"][1])
            spread = L[f"foot_{side}"][2] - L[f"foot_{other}"][2]
            score = spread - max(0.0, low - 3.0) * 8.0
            if best is None or score > best[0]:
                best = (score, i)
        out.append((best[1], side))
    return out


def stop_frames(cy, st, c_hs, front, n_out):
    """Step-together stop from cycle frame c_hs (front foot just landed): the front foot stays where it is, the rear
    foot lifts and steps up beside it, the body settles to standing (the start clip's first frame). Returns
    ([joint sets in the moving root frame], root advance) - the root moves forward by `advance` over the stop."""
    A = cy.local[c_hs]
    B = st.local[0]
    rear = "r" if front == "l" else "l"
    # stand B placed so its front-side foot is on A's front foot (x, z)
    off = A[f"foot_{front}"] - B[f"foot_{front}"]
    off[1] = 0.0
    advance = float(off[2])
    Bw = {q: v + off for q, v in B.items()}
    out = []
    for i in range(n_out):
        u = (i + 1) / n_out
        e = u * u * (3 - 2 * u)
        root = np.array([0.0, 0.0, advance * e])
        J = {}
        for q in A:
            v = A[q] * (1 - e) + Bw[q] * e
            if q in (f"foot_{rear}", f"ball_{rear}", f"toe_{rear}"):
                v = v + np.array([0.0, 7.0 * math.sin(math.pi * min(1.0, u * 1.15)), 0.0])
            elif q == f"calf_{rear}":
                v = v + np.array([0.0, 2.5, 3.5]) * math.sin(math.pi * min(1.0, u * 1.15))
            J[q] = v - root
        out.append(J)
    return out, advance


_CLIPS = {}


def clip(name, cyclic=False):
    if name not in _CLIPS:
        _CLIPS[name] = Clip(name, cyclic=cyclic)
    return _CLIPS[name]


def available():
    return all(MC.available(n) for n in ("Walking", "Female_Start_Walking", "Left_Turn_90", "Right_Turn_90"))


def plan_walk(D, k):
    """Choose where the start clip hands over to the cycle, how far the cycle runs, and on which landing step the
    stop happens, so that the scaled distances add up to D with the least stretch. k = body scale.
    Returns dict(start_cut, c0, c_len, stop_at, front, stop_adv, stretch)."""
    st, cy = clip("Female_Start_Walking"), clip("Walking", cyclic=True)
    cands = []
    for i in range(int(st.n * 0.45), st.n):                    # hand-over points once walking has begun
        c, e = cy.match(st._feat[i])
        cands.append((i, c, e))
    emin = min(c[2] for c in cands)
    good = [g for g in cands if g[2] < emin * 1.6 + 3.0]
    hs = heel_strikes(cy)
    best = None
    stride = cy.loop_dist * k                                   # distance per cycle
    for (i, c0, e0) in good:
        for (c1, front) in hs:
            _, adv = stop_frames(cy, st, c1, front, 2)
            d_start, d_stop = st.s[i] * k, adv * k
            frac = ((c1 - c0) % cy.n) / cy.n
            for m in range(0, 400):
                L = d_start + (frac + m) * stride + d_stop
                if L > D * 1.25 and m > 0:
                    break
                stretch = D / L
                score = abs(math.log(stretch)) * 40.0 + e0 * 0.05
                if best is None or score < best[0]:
                    best = (score, dict(start_cut=i, c0=c0, c_len=(frac + m) * cy.n, stop_at=c1, front=front,
                                        stop_adv=adv, stretch=stretch))
    return best[1]


def turn_frames(sk, angle_deg, fps_out):
    """(joint sets, yaw offsets deg) of a turn on the spot by angle_deg (left positive), from the 90 deg clips."""
    c = clip("Left_Turn_90" if angle_deg > 0 else "Right_Turn_90")
    total = math.degrees(c.yaw[-1])
    scale = angle_deg / total if abs(total) > 1e-6 else 0.0
    dur = c.n / c.fps * max(0.6, min(1.3, abs(angle_deg) / 90.0))
    n = max(4, int(round(dur * fps_out)))
    out = []
    for i in range(n):
        x = i / (n - 1) * (c.n - 1)
        out.append((c.at(x), math.degrees(c.yaw_at(x)) * scale))
    return out
