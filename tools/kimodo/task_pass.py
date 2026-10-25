"""Kimodo for a whole task (walks between stations, several steps): the body of every frame from Kimodo, the hands,
tools and footing from our solve - tools/task_run.py --kimodo.

A task is a minute or more, longer than Kimodo's motions: it is cut into overlapping windows (~8 s, 1.5 s overlap).
Per window, in the window's own frame (its first frame's root at the origin, facing +Z):
    - the root path while walking (Root2D waypoints every 1/4 s): Kimodo walks it its own way
    - the working hands (wrist position + rotation) while a hand works
    - the planted feet while standing / kneeling, and the whole solved body every 3rd key while kneeling / squatting
The windows are generated in one Kimodo process, blended across the overlaps (joint positions, smoothstep), taken
back into each frame's worker frame and retargeted; then the usual final pass ("keep the body, solve the hands").
"""
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np   # noqa: E402

WIN, OVER, EVERY = 192, 36, 4                     # frames (24 fps): 8 s windows, 1.5 s overlap, a key every 1/6 s


def _rootT(r):
    from frt_core.people.bake import _rootT as rt
    return rt(r or (0.0, 0.0, 0.0))


def windows(n):
    out, a = [], 0
    while True:
        b = min(n, a + WIN)
        out.append((a, b))
        if b >= n:
            return out
        a = b - OVER


SIDE, BEHIND, AROUND = 26.0, 5.0, 12                 # cm to the side / in front of the pelvis, frames either side


def turns(T):
    """Frames where a working hand is far to the side or behind (a bin on a cart beside the spot): the feet are
    left to Kimodo there, so the worker turns or steps towards it instead of folding sideways on planted feet."""
    n = len(T.frames)
    far = np.zeros(n, bool)
    for i, f in enumerate(T.frames):
        if f.label.endswith(":walk") or getattr(f.stance, "drop", None) is not None:
            continue
        for t in f.targets:
            if t.weight >= 0.5 and (abs(float(t.point[0])) > SIDE or float(t.point[2]) < BEHIND):
                far[i] = True
    if not os.environ.get("FRT_KIMODO_TURN"):                       # (tried: worse - the paths were the problem)
        return np.zeros(n, bool)
    out = far.copy()
    for i in np.nonzero(far)[0]:
        out[max(0, i - AROUND):i + AROUND + 1] = True
    return out


SLOW = 1.2                                          # cm / frame: a hand target slower than this is at the work


def contact(T):
    """{side: bool per frame}: the hand works (target weight >= 0.5) and its target is nearly still - at a nut, a
    bolt, a socket. Between contacts (reaching, carrying a part over) the arm is Kimodo's own."""
    n = len(T.frames)
    out = {}
    for side in "lr":
        pts = [next((np.asarray(t.point, float) for t in f.targets if t.side == side and t.weight >= 0.5), None)
               for f in T.frames]
        c = np.zeros(n, bool)
        for i in range(n):
            if pts[i] is None:
                continue
            nb = [pts[j] for j in (i - 2, i + 2) if 0 <= j < n and pts[j] is not None]
            c[i] = bool(nb) and max(float(np.linalg.norm(q - pts[i])) for q in nb) / 2.0 < SLOW
        out[side] = c
    return out


PHASE = {"walk": "walk", "reach": "pick_bin", "move": "pick_bin", "get tool": "get_tool", "put tool back": "put_tool",
         "thread": "hand_nut", "tighten": "ratchet", "position": "ratchet", "return": "idle"}


def phase_prompt(label):
    """A frame's phase ('01_hand_start:thread') -> a prompt key of cloud/prompts.json."""
    step, _, ph = label.partition(":")
    verb = step.split("_", 1)[-1]
    if verb in ("carry", "transfer", "push_door", "latch", "inspect", "cable_tie"):
        return {"push_door": "push_door", "latch": "latch"}.get(verb, verb if verb != "cable_tie" else "cable_tie")
    if verb == "connect":
        return "connector"
    return PHASE.get(ph, "idle")


def prepare(T, sols, a, b, out_dir, seed=0, steps=100, turn=None, touch=None):
    """Window a..b-1 -> keys npz (kimodo_constrained.py's input) and the window frame (4x4 window -> worker world)."""
    from frt_core.people.kinematics import fk
    from frt_core.people.motion_gen import FPS
    from frt_core.people.skeleton import NAMES
    import kimodo_job as KJ
    sk = T.sk
    s = KJ.KIMODO_HIP_M / float(sk.world["pelvis"][1])
    W0 = _rootT(T.frames[a].root)                                   # window frame -> world
    Wi = np.linalg.inv(W0)
    idx = list(range(a, b, EVERY))
    if idx[-1] != b - 1:
        idx.append(b - 1)
    P, R, ur, ul, feet, full = [], [], [], [], [], []
    for i in idx:
        f = T.frames[i]
        Rw, Pw = fk(sk, sols[i].pose)
        M = Wi @ _rootT(f.root)                                     # this frame's worker frame -> window frame
        P.append([M[:3, :3] @ Pw[n_] + M[:3, 3] for n_ in NAMES])
        R.append([M[:3, :3] @ Rw[n_] for n_ in NAMES])
        tg = {t.side: t for t in f.targets}
        ur.append("r" in tg and tg["r"].weight >= 0.5 and (touch is None or touch["r"][i]))
        ul.append("l" in tg and tg["l"].weight >= 0.5 and (touch is None or touch["l"][i]))
        walking = f.label.endswith(":walk")
        feet.append(not walking and not getattr(f.stance, "free", False) and not (turn is not None and turn[i]))
        full.append(getattr(f.stance, "drop", None) is not None and (len(full) % 3 == 0))
    r2t, r2 = [], []
    for i in range(a, b, 6):                                        # the route, while walking
        if T.frames[i].label.endswith(":walk"):
            p = Wi @ _rootT(T.frames[i].root) @ np.array([0.0, 0.0, 0.0, 1.0])
            r2t.append((i - a) / FPS)
            r2.append((p[0], p[2]))
    src = os.path.join(out_dir, f"keys_{a:05d}.npz")
    from collections import Counter                                 # the window's main activity -> its prompt
    act = Counter(phase_prompt(T.frames[i].label) for i in range(a, b)).most_common(1)[0][0]
    np.savez(src, seconds=(b - a) / FPS, fps_in=FPS, frames=np.array(idx) - a, t=(np.array(idx) - a) / FPS,
             names=np.array(NAMES), P=np.array(P), R=np.array(R), use_r=np.array(ur), use_l=np.array(ul),
             use_feet=np.array(feet), use_full=np.array(full), scale=s, steps=steps, seed=seed,
             r2_t=np.array(r2t), r2_xz=np.array(r2).reshape(-1, 2), prompt=KJ.PROMPTS.get(act, ""))
    return src, os.path.join(out_dir, f"kimodo_{a:05d}.npz"), W0


def kimodo_pass(T, sols, out_dir, log=print):
    """sols (our solve of task T) -> new sols: Kimodo's body, our hands / tools / footing; T.frames get the bases."""
    import kimodo_job as KJ
    from frt_core.people.bake import solve_frames
    from frt_core.people.mocap import retarget
    from frt_core.people.motion_gen import FPS
    os.makedirs(out_dir, exist_ok=True)
    sk, n = T.sk, len(T.frames)
    s = KJ.KIMODO_HIP_M / float(sk.world["pelvis"][1])
    wins = windows(n)
    turn = turns(T)
    log(f"Kimodo: {int(turn.sum())} frames where the worker may turn / step to a side reach")
    touch = contact(T) if os.environ.get("FRT_KIMODO_PIN") == "contact" else None   # (tried: arms too free)
    jobs = [prepare(T, sols, a, b, out_dir, turn=turn, touch=touch) for a, b in wins]
    so, se = KJ.generate([(k, o) for k, o, _ in jobs])
    got = so.count("ok (")
    log(f"Kimodo: {got} of {len(jobs)} windows")
    if got < len(jobs):
        log(se[-1500:])
    # joint positions per frame (world, cm), blended over the overlaps
    acc, wsum = [None] * n, np.zeros(n)
    names77 = None
    for (a, b), (_, gen, W0) in zip(wins, jobs):
        if not os.path.isfile(gen):
            continue
        g = np.load(gen)
        pj, names77, kfps = g["posed_joints"], [str(x) for x in g["names77"]], float(g["fps"])
        for i in range(a, b):
            x = (i - a) / FPS * kfps
            i0 = min(int(math.floor(x)), len(pj) - 1)
            i1 = min(i0 + 1, len(pj) - 1)
            u = x - i0
            J = KJ.soma_to_canonical(pj[i0] * (1 - u) + pj[i1] * u, names77, s)
            Jw = {k: W0[:3, :3] @ v + W0[:3, 3] for k, v in J.items()}
            w = 1.0                                                  # fade in / out across the overlaps
            if a > 0 and i < a + OVER:
                w = (i - a + 0.5) / OVER
            if b < n and i >= b - OVER:
                w = min(w, (b - i - 0.5) / OVER)
            w = w * w * (3 - 2 * w)
            if acc[i] is None:
                acc[i] = {k: v * w for k, v in Jw.items()}
            else:
                for k, v in Jw.items():
                    acc[i][k] = acc[i].get(k, 0.0) + v * w
            wsum[i] += w
    cal = KJ.soma_calibration(sk, s)
    from dataclasses import replace
    new = []
    for i, f in enumerate(T.frames):
        if acc[i] is None or wsum[i] <= 1e-6:
            new.append(f)
            continue
        Mi = np.linalg.inv(_rootT(f.root))
        J = {k: Mi[:3, :3] @ (v / wsum[i]) + Mi[:3, 3] for k, v in acc[i].items()}
        pose = retarget(sk, J, cal)
        pose.pelvis = J["pelvis"] - sk.world["pelvis"]
        settle_clavicles(sk, pose, J)
        st = f.stance
        if turn[i]:                                                    # the legs as Kimodo turned them
            st = replace(st if st is not None else T.S.default_stance())
            st.free = True
        new.append(replace(f, base=pose, static=False, stance=st, tools=dict(f.tools)))     # (a wait moves a little now: breathing)
    if touch is not None:
        import copy
        for i, f in enumerate(new):                   # between contacts: the target only eases the arm in / out
            f.targets = [copy.copy(t) for t in f.targets]
            for t in f.targets:
                if t.weight >= 0.5 and not touch[t.side][i]:
                    t.weight = 0.49
        KJ.pin_hands(sk, new)
    relax_hip_hands(sk, sols, new)
    S = T.S
    keep = S.base_correct
    S.base_correct = True
    try:
        ks = solve_frames(S, new, log=lambda *a_: None)
    finally:
        S.base_correct = keep
    follow_hands(T, sols, ks, new)
    T.frames[:] = new
    return ks


CLAV_MAX = 12.0                                     # deg: the collarbone's turn from rest kept from a retargeted body


def _swing(a, b):
    from frt_core.people.kinematics import axis_angle, unit
    a, b = unit(a), unit(b)
    ax = np.cross(a, b)
    s_ = float(np.linalg.norm(ax))
    if s_ < 1e-9:
        return np.eye(3)
    return axis_angle(ax / s_, math.atan2(s_, float(np.dot(a, b))))


def settle_clavicles(sk, pose, J):
    """Kimodo's shoulder girdle does not map onto the character's collarbone (it came out raised past its range: a
    shrug). The collarbone is kept within CLAV_MAX of rest and the arm re-aimed so that elbow and wrist stay where
    Kimodo put them."""
    from frt_core.people.kinematics import fk
    from frt_core.people.solver import _clamp_rot
    for s_ in "lr":
        c, u, l_, h = (f"{j}_{s_}" for j in ("clavicle", "upperarm", "lowerarm", "hand"))
        if c not in pose.rot or l_ not in J or h not in J:
            continue
        pose.rot[c] = _clamp_rot(pose.rot[c], CLAV_MAX)
        R, P = fk(sk, pose, [h])
        Rc = R[c]
        Ru = _swing(P[l_] - P[u], J[l_] - P[u]) @ R[u]                  # upper arm at Kimodo's elbow
        pose.rot[u] = Rc.T @ Ru
        R, P = fk(sk, pose, [h])
        Rl = _swing(P[h] - P[l_], J[h] - P[l_]) @ R[l_]                 # forearm at Kimodo's wrist
        pose.rot[l_] = R[u].T @ Rl


ARM = ("clavicle", "upperarm", "lowerarm", "hand")


def _slerp(Ra, Rb, u):
    from frt_core.people.kinematics import axis_angle
    D = Ra.T @ Rb
    ang = math.acos(max(-1.0, min(1.0, (np.trace(D) - 1.0) / 2.0)))
    if ang < 1e-6:
        return Ra
    ax = np.array([D[2, 1] - D[1, 2], D[0, 2] - D[2, 0], D[1, 0] - D[0, 1]]) / (2.0 * math.sin(ang))
    return Ra @ axis_angle(ax, ang * u)


def relax_hip_hands(sk, rule_sols, frames, ramp=8):
    """Without a text prompt Kimodo parks an idle hand on the hip (elbow out, wrist at the waist): where a free hand
    does that, the arm eases (over `ramp` frames) into the first solve's relaxed hang."""
    from frt_core.people.kinematics import fk
    n = len(frames)
    for s_ in "lr":
        flag = np.zeros(n)
        for i, f in enumerate(frames):
            if f.base is None or any(t.side == s_ and t.weight >= 0.3 for t in f.targets):
                continue
            R, P = fk(sk, f.base, [f"hand_{s_}"])
            h, pel = P[f"hand_{s_}"], P["pelvis"]
            up, fa = P[f"lowerarm_{s_}"] - P[f"upperarm_{s_}"], P[f"hand_{s_}"] - P[f"lowerarm_{s_}"]
            bend = math.degrees(math.acos(max(-1.0, min(1.0, float(up @ fa) / (np.linalg.norm(up) * np.linalg.norm(fa) + 1e-9)))))
            if abs(h[1] - pel[1]) < 14.0 and bend > 55.0:
                flag[i] = 1.0
        if not flag.any():
            continue
        w = np.array([flag[max(0, i - ramp):i + ramp + 1].mean() for i in range(n)])   # eased in / out
        w = np.clip(w * 2.0, 0.0, 1.0)
        w = w * w * (3 - 2 * w)
        for i, f in enumerate(frames):
            if w[i] <= 1e-3 or f.base is None:
                continue
            for j in ARM:
                k = f"{j}_{s_}"
                if k in f.base.rot and k in rule_sols[i].pose.rot:
                    f.base.rot[k] = _slerp(f.base.rot[k], rule_sols[i].pose.rot[k], float(w[i]))


HELD = 16.0                                          # cm: a tool / part this close to a hand, rigidly, is held by it


def follow_hands(T, old_sols, new_sols, frames):
    """Things in a hand move with the hand that is shown: where a tool or part keeps the same pose relative to a hand
    of the first solve (it is held), it is put at that same pose relative to the final hand."""
    from frt_core.people.kinematics import fk

    def hand(sols, i, s_):
        R, P = fk(T.sk, sols[i].pose, [f"hand_{s_}"])
        M = np.eye(4)
        M[:3, :3], M[:3, 3] = R[f"hand_{s_}"], P[f"hand_{s_}"]
        return M
    names = sorted({k for f in frames for k in f.tools})
    cache = {}
    for i, f in enumerate(frames):
        for name in names:
            if name not in f.tools:
                continue
            key, Tw = f.tools[name]
            Tw = np.asarray(Tw, float)
            for s_ in "lr":
                if (i, s_) not in cache:
                    cache[(i, s_)] = (hand(old_sols, i, s_), hand(new_sols, i, s_))
                Ho, Hn = cache[(i, s_)]
                O = np.linalg.inv(Ho) @ Tw
                if np.linalg.norm(O[:3, 3]) > HELD:
                    continue
                if i > 0 and name in frames[i - 1].tools:            # rigid: same offset as a frame ago
                    Hp = hand(old_sols, i - 1, s_) if (i - 1, s_) not in cache else cache[(i - 1, s_)][0]
                    if name not in T.frames[i - 1].tools:
                        continue
                    Op = np.linalg.inv(Hp) @ np.asarray(T.frames[i - 1].tools[name][1], float)   # (as authored)
                    if np.linalg.norm(Op[:3, 3] - O[:3, 3]) > 1.0:
                        continue
                f.tools[name] = (key, Hn @ O)
                break
