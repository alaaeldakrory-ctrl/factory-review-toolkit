"""Where the worker stands: moving the work relative to the body, and choosing a better standing spot when the work
does not suit the body (a short worker steps closer to high work, a tall one back from low, close work).

move_work(frames, M) moves the work (hand targets at the work, tools at the work, parts, gaze) by a 4x4 M in the
worker frame while hands / tools at their start and end positions (at rest, at the hip - body-relative) stay put;
targets blend in over the first 30 cm of a reach, so reaches stay smooth.

solve_placed(S, frames, keep_out) solves as authored; if the checks fail, it tries standing spots (the work shifted
the other way) on the job's key frames, re-solves the most promising and keeps the best - returning the offset so
the scene can stand the worker there.
"""
import copy
import math

import numpy as np

# standing spots tried (worker steps dx to its left, dz forward; cm)
SPOTS = [(0.0, -10.0), (0.0, 8.0), (0.0, -18.0), (8.0, 0.0), (-8.0, 0.0), (8.0, -10.0), (-8.0, -10.0), (0.0, 15.0),
         (0.0, 22.0), (8.0, 15.0), (-8.0, 15.0)]


def _w(dist):
    u = min(1.0, max(0.0, (dist - 4.0) / 26.0))
    return u * u * (3 - 2 * u)


def move_work(frames, M):
    from .bake import _R, _q, _slerp
    from .solver import HandTarget
    qR = _q(M[:3, :3])
    out = copy.deepcopy(frames)
    ends = {s: [np.asarray(t.point, float) for f in (frames[0], frames[-1]) for t in f.targets if t.side == s]
            for s in ("l", "r")}
    tool_ends = {n: [f.tools[n][1][:3, 3] for f in (frames[0], frames[-1]) if n in f.tools]
                 for n in {k for f in frames for k in f.tools if not k.startswith("part:")}}

    def moved_T(T, w):
        T2 = M @ T
        if w >= 0.999:
            return T2
        out_ = np.eye(4)
        out_[:3, :3] = _R(_slerp(_q(np.eye(3)), qR, w)) @ T[:3, :3]
        out_[:3, 3] = T[:3, 3] * (1 - w) + T2[:3, 3] * w
        return out_
    for f, g in zip(frames, out):
        tg = []
        for t in f.targets:
            p = np.asarray(t.point, float)
            w = _w(min([np.linalg.norm(p - e) for e in ends.get(t.side, [])] or [99.0]))
            Rw = _R(_slerp(_q(np.eye(3)), qR, w))
            p2 = (M[:3, :3] @ p + M[:3, 3]) * w + p * (1 - w)
            tg.append(HandTarget(t.side, tuple(p2), None if t.fingers is None else tuple(Rw @ np.asarray(t.fingers)),
                                 None if t.palm is None else tuple(Rw @ np.asarray(t.palm)), t.grip, t.weight))
        g.targets = tg
        for name, (key, T) in f.tools.items():
            if name.startswith("part:"):
                g.tools[name] = (key, M @ T)
            else:
                g.tools[name] = (key, moved_T(T, _w(min([np.linalg.norm(T[:3, 3] - e) for e in tool_ends.get(name, [])]
                                                       or [99.0]))))
        if f.look is not None:
            g.look = tuple(M[:3, :3] @ np.asarray(f.look, float) + M[:3, 3])
    return out


def move_rules(rules, M):
    out = []
    for r in rules:
        if len(r) == 3:
            p0, n = np.asarray(r[1], float), np.asarray(r[2], float)
        else:
            p0, n = np.array([0.0, 0.0, r[1]]), np.array([0.0, 0.0, 1.0])
        out.append((r[0], tuple(M[:3, :3] @ p0 + M[:3, 3]), tuple(M[:3, :3] @ n)))
    return out


def translation(dx, dz):
    M = np.eye(4)
    M[0, 3], M[2, 3] = dx, dz
    return M


def _key_frames(frames, n=8):
    idx = [i for i, f in enumerate(frames) if any(t.weight >= 0.5 for t in f.targets)]
    if not idx:
        return []
    step = max(1, len(idx) // n)
    return idx[::step][:n]


def _badness(rep):
    c = rep["counts"]
    return (c.get("limit", 0) + c.get("awkward", 0) + c.get("collision", 0) * 2 + c.get("pop", 0) * 2,
            rep["worst_limit_over"] + rep["worst_pop"] * 0.2)


def solve_placed(S, frames, keep_out=(), log=None):
    """(frames, sols, report, offset (dx, dz) of the worker, report at the authored spot) - the authored spot when
    it passes the checks, otherwise the best standing spot found."""
    from .bake import solve_frames
    from .check import check_frames
    S.keep_out = list(keep_out)
    sols = solve_frames(S, frames, log=lambda *a: None)
    rep = check_frames(S.sk, frames, sols, keep_out=S.keep_out)
    raw = rep
    if rep["ok"]:
        return frames, sols, rep, (0.0, 0.0), raw
    keys = _key_frames(frames)
    scored = []
    for dx, dz in SPOTS:                         # the worker steps by (dx, dz): the work moves the other way
        M = translation(-dx, -dz)
        fr = move_work([frames[i] for i in [0] + keys + [len(frames) - 1]], M)[1:-1]
        S.keep_out = move_rules(keep_out, M)
        cost = sum(S.solve(f.targets, look=f.look, stance=f.stance, base=f.base).cost for f in fr)
        scored.append((cost, dx, dz))
    scored.sort()
    best = (_badness(rep), frames, sols, rep, (0.0, 0.0))
    for cost, dx, dz in scored[:2]:
        M = translation(-dx, -dz)
        fr = move_work(frames, M)
        S.keep_out = move_rules(keep_out, M)
        so = solve_frames(S, fr, log=lambda *a: None)
        rp = check_frames(S.sk, fr, so, keep_out=S.keep_out)
        if log:
            log(f"  spot ({dx:+.0f}, {dz:+.0f}): {'ok' if rp['ok'] else rp['counts']}")
        cand = (_badness(rp), fr, so, rp, (dx, dz))
        if cand[0] < best[0]:
            best = cand
        if rp["ok"]:
            break
    S.keep_out = move_rules(keep_out, translation(-best[4][0], -best[4][1]))
    return best[1], best[2], best[3], best[4], raw


def diagnose(S, frames, rep, sols=None):
    """Why a job still fails after stepping, in workplace terms: for the flagged frames, where the working hand's
    target is relative to this body (above the shoulder, beyond comfortable reach, below the knee, behind the body).
    Returns a list of findings like {"kind": "too high", "cm": 18, "side": "r", "frames": 24}.
    With `sols`, reach is measured from the shoulder where it actually is in each solved frame (kneeling,
    crouching, leaning), not from the standing shoulder."""
    from .kinematics import fk
    sk = S.sk
    flagged = sorted({i for i, *_ in rep["issues"]})
    if not flagged:
        return []
    arm = {s: sum(S.L[s]) + S.hand_len * 0.45 for s in ("l", "r")}
    knee = float(sk.world["calf_l"][1])
    tally = {}
    for i in flagged:
        for t in frames[i].targets:
            if t.weight < 0.5:
                continue
            if sols is not None and sols[i] is not None:
                _, P_ = fk(sk, sols[i].pose)
                sh = np.asarray(P_[f"upperarm_{t.side}"], float)
            else:
                sh = np.asarray(sk.world[f"upperarm_{t.side}"], float)
            p = np.asarray(t.point, float)
            d = p - sh
            kinds = []
            if d[1] > 8.0:
                kinds.append(("too high", d[1]))
            if np.linalg.norm(d) > 0.9 * arm[t.side]:
                kinds.append(("too far", np.linalg.norm(d) - 0.9 * arm[t.side]))
            if p[1] < knee:
                kinds.append(("too low", knee - p[1]))
            if d[2] < 5.0:
                kinds.append(("too close / behind", 5.0 - d[2]))
            for k, v in kinds:
                key = (k, t.side)
                n, worst = tally.get(key, (0, 0.0))
                tally[key] = (n + 1, max(worst, float(v)))
    out = [{"kind": k, "side": s, "frames": n, "cm": round(v)} for (k, s), (n, v) in tally.items()]
    return sorted(out, key=lambda x: -x["frames"])
