"""Gait diagnosis: the source motion capture (a real person) next to what a worker does, as side-view stick figures,
plus joint-angle curves over the cycle (knee, hip, ankle, elbow, shoulder) against the mocap reference.

    python tools/walk_compare.py [W50M] -> out/people/walk_lab/compare.gif, compare_curves.png
"""
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "exts", "frt.core"))
import numpy as np   # noqa: E402

from frt_core.people import mocap as MC   # noqa: E402
from frt_core.people.bake import solve_frames   # noqa: E402
from frt_core.people.gait import walk   # noqa: E402
from frt_core.people.kinematics import fk   # noqa: E402
from frt_core.people.skeleton import Skeleton   # noqa: E402
from frt_core.people.solver import PostureSolver   # noqa: E402
from frt_core.people.workers import WORKERS   # noqa: E402

OUT = os.path.join(HERE, "..", "out", "people", "walk_lab")
BONES = [("pelvis", "spine_01"), ("spine_01", "spine_02"), ("spine_02", "spine_03"), ("spine_03", "neck_01"),
         ("neck_01", "head")]
for s in ("l", "r"):
    BONES += [("spine_03", f"clavicle_{s}"), (f"clavicle_{s}", f"upperarm_{s}"), (f"upperarm_{s}", f"lowerarm_{s}"),
              (f"lowerarm_{s}", f"hand_{s}"), ("pelvis", f"thigh_{s}"), (f"thigh_{s}", f"calf_{s}"),
              (f"calf_{s}", f"foot_{s}"), (f"foot_{s}", f"ball_{s}")]


def ang(a, b, c):
    u, v = a - b, c - b
    return math.degrees(math.acos(max(-1, min(1, np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-9)))))


def metrics(P, fwd):
    """Sagittal angles: knee flexion, hip flexion (thigh vs trunk), ankle, elbow, shoulder swing - per side."""
    up = np.array([0.0, 1.0, 0.0])
    m = {}
    for s in ("l", "r"):
        m[f"knee_{s}"] = 180 - ang(P[f"thigh_{s}"], P[f"calf_{s}"], P[f"foot_{s}"])
        th = P[f"calf_{s}"] - P[f"thigh_{s}"]
        m[f"hip_{s}"] = math.degrees(math.atan2(np.dot(th, fwd), -np.dot(th, up)))
        ua = P[f"lowerarm_{s}"] - P[f"upperarm_{s}"]
        m[f"shoulder_{s}"] = math.degrees(math.atan2(np.dot(ua, fwd), -np.dot(ua, up)))
        m[f"elbow_{s}"] = 180 - ang(P[f"upperarm_{s}"], P[f"lowerarm_{s}"], P[f"hand_{s}"])
        m[f"foot_y_{s}"] = P[f"foot_{s}"][1]
    tr = P["neck_01"] - P["pelvis"]
    m["trunk"] = math.degrees(math.atan2(np.dot(tr, fwd), np.dot(tr, up)))
    m["pelvis_y"] = P["pelvis"][1]
    return m


def main():
    wid = next((a for a in sys.argv[1:] if not a.startswith("--")), "W50M")
    w = WORKERS[wid]
    sk = Skeleton.for_worker(w)
    S = PostureSolver(sk, w)
    cyc = MC.cycle("Walking")
    # ours: a straight walk; keep the steady middle
    frames = walk(w, sk, [(0.0, 0.0), (900.0, 0.0)], 90.0, 90.0) if '--turn' not in sys.argv else         walk(w, sk, [(-100.0, 0.0), (420.0, 0.0), (520.0, -20.0)], 90.0, 180.0)
    sols = solve_frames(S, frames, log=lambda *a: None)
    ours = []
    contact = []                                          # per frame: world point of the lowest heel / toe
    for f, s in zip(frames, sols):
        R, P = fk(sk, s.pose)
        ours.append(P)
        x, z, yaw = f.root
        a = math.radians(yaw)
        Rw = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
        pts = {}
        for sd in ("l", "r"):
            for nm in (f"foot_{sd}", f"ball_{sd}"):
                q = Rw @ P[nm] + np.array([x, 0, z])
                pts[nm] = q
        contact.append(pts)
    slips, lows = [], []
    a0, a1 = (len(frames) // 4, 3 * len(frames) // 4) if '--all' not in sys.argv else (0, len(frames))
    for i in range(a0 + 1, a1):
        # the supporting foot: the side whose lower point (heel = ankle, toe = ball) is lowest
        best = min(("l", "r"), key=lambda sd: min(contact[i][f"foot_{sd}"][1], contact[i][f"ball_{sd}"][1]))
        nm = min((f"foot_{best}", f"ball_{best}"), key=lambda q: contact[i][q][1])
        d = contact[i][nm] - contact[i - 1][nm]
        slips.append(float(np.hypot(d[0], d[2])))
        lows.append(contact[i][nm][1])
    print("support-foot slip per frame cm: median %.2f  p90 %.2f  (body moves %.1f per frame)" % (
        np.median(slips), np.percentile(slips, 90), np.median([np.hypot(*(np.subtract(frames[i].root[:2], frames[i - 1].root[:2]))) for i in range(a0 + 1, a1)])))
    print("support point height: %.1f .. %.1f (heel ankle rest %.1f, ball rest %.1f)" % (
        min(lows), max(lows), sk.world["foot_l"][1], sk.world["ball_l"][1]))
    n = len(ours)
    ours = ours[n // 3: n // 3 + cyc.n]
    ref = [cyc.frames[(cyc.offset + i) % cyc.n] for i in range(cyc.n)]
    fwd_o = np.array([0.0, 0.0, 1.0])                    # worker frame faces +Z
    # mocap travel direction
    tv = np.array([cyc.travel[0], 0.0, cyc.travel[2]])
    fwd_m = tv / (np.linalg.norm(tv) or 1)
    mo = [metrics(P, fwd_o) for P in ours]
    mm = [metrics(P, fwd_m) for P in ref]
    keys = ["knee_r", "hip_r", "shoulder_r", "elbow_r", "trunk", "pelvis_y", "foot_y_r"]
    print(f"{'':10s} {'mocap min..max':>18s} {'ours min..max':>18s}")
    for k in keys:
        a = [x[k] for x in mm]
        b = [x[k] for x in mo]
        print(f"{k:10s} {min(a):7.1f} .. {max(a):7.1f}   {min(b):7.1f} .. {max(b):7.1f}")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    fig, axs = plt.subplots(1, 2, figsize=(8, 4.5))

    def side(P, fwd):
        # (forward, up) coordinates relative to the pelvis
        o = P["pelvis"]
        return {k: (np.dot(v - o, fwd), v[1]) for k, v in P.items()}

    def draw(i):
        for ax, P, fwd, title in ((axs[0], ref[i % len(ref)], fwd_m, "mocap (real person)"),
                                  (axs[1], ours[i % len(ours)], fwd_o, f"ours ({wid})")):
            ax.clear()
            Q = side(P, fwd)
            for a, b in BONES:
                if a in Q and b in Q:
                    col = "tab:red" if a.endswith("_r") or b.endswith("_r") else "tab:blue"
                    ax.plot([Q[a][0], Q[b][0]], [Q[a][1], Q[b][1]], color=col, lw=2.5)
            ax.set_xlim(-80, 80)
            ax.set_ylim(-5, 190)
            ax.set_aspect("equal")
            ax.set_title(title)
            ax.axhline(0, color="k", lw=0.5)
    anim = FuncAnimation(fig, draw, frames=max(len(ref), len(ours)), interval=1000 / 30)
    anim.save(os.path.join(OUT, "compare.gif"), writer=PillowWriter(fps=30))
    plt.close(fig)
    fig, axs = plt.subplots(2, 3, figsize=(12, 6))
    for ax, k in zip(axs.flat, ["knee_r", "hip_r", "shoulder_r", "elbow_r", "trunk", "pelvis_y"]):
        ax.plot(np.linspace(0, 1, len(mm)), [x[k] for x in mm], label="mocap")
        ax.plot(np.linspace(0, 1, len(mo)), [x[k] for x in mo], label="ours")
        ax.set_title(k)
        ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "compare_curves.png"), dpi=80)
    print("->", OUT)


if __name__ == "__main__":
    main()
