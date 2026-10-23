"""Walking shoot-out: our clip-based walk vs Kimodo (root-path constraint, no text) on several routes and all three
bodies, measured the same way: planted-foot slip, floor contact (sink / float), route following, knee and hip
ranges against the captured walk (Mixamo), and frame-to-frame jumps.

    python tools/kimodo/walk_shootout.py -> out/people/kimodo/walk_shootout.json + a table
"""
import json
import math
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "exts", "frt.core"))
sys.path.insert(0, HERE)
import numpy as np   # noqa: E402

ROUTES = {"straight 4 m": [(0.0, 0.0), (0.0, 400.0)],
          "L turn": [(0.0, 0.0), (0.0, 250.0), (200.0, 250.0)],
          "U turn": [(0.0, 0.0), (0.0, 250.0), (150.0, 250.0), (150.0, 0.0)],
          "short 1 m": [(0.0, 0.0), (0.0, 100.0)]}
OUT = os.path.join(ROOT, "out", "people", "kimodo", "shootout")


def rootT(r):
    a = math.radians(r[2])
    T = np.eye(4)
    T[:3, :3] = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
    T[:3, 3] = (r[0], 0.0, r[1])
    return T


def measure(sk, frames, sols, route):
    from frt_core.people.check import check_frames
    from frt_core.people.kinematics import fk
    feet, knees, hips, pel = [], [], [], []
    for f, s in zip(frames, sols):
        T = rootT(f.root or (0.0, 0.0, 0.0))
        R, P = fk(sk, s.pose)
        feet.append({n: T[:3, :3] @ P[n] + T[:3, 3] for n in ("foot_l", "ball_l", "foot_r", "ball_r")})
        pel.append(T[:3, :3] @ P["pelvis"] + T[:3, 3])
        for sd in ("l", "r"):
            th, ca = P[f"calf_{sd}"] - P[f"thigh_{sd}"], P[f"foot_{sd}"] - P[f"calf_{sd}"]
            knees.append(math.degrees(math.acos(np.clip(np.dot(th, ca) / np.linalg.norm(th) / np.linalg.norm(ca), -1, 1))))
    slip, low = [], []
    for i in range(1, len(feet)):
        nm = min(feet[i], key=lambda q: feet[i][q][1])
        d = feet[i][nm] - feet[i - 1][nm]
        slip.append(float(np.hypot(d[0], d[2])))
        low.append(float(feet[i][nm][1]))
    end = np.asarray(route[-1], float)
    rep = check_frames(sk, frames, sols)
    return {"slip_median_cm": round(float(np.median(slip)), 2), "slip_p90_cm": round(float(np.percentile(slip, 90)), 2),
            "contact_cm": [round(min(low), 1), round(max(low), 1)], "knee_deg": [round(min(knees), 1), round(max(knees), 1)],
            "end_error_cm": round(float(np.linalg.norm(pel[-1][[0, 2]] - end)), 1),
            "pops": rep["counts"].get("pop", 0), "limits": rep["counts"].get("limit", 0), "frames": len(frames)}


def main():
    from frt_core.people.bake import solve_frames
    from frt_core.people.gait import walk
    from frt_core.people.mocap import retarget
    from frt_core.people.motion_gen import FPS, Frame
    from frt_core.people.motion_more import idle
    from frt_core.people.skeleton import Skeleton
    from frt_core.people.solver import PostureSolver, Stance
    from frt_core.people.workers import WORKERS
    from kimodo_job import KIMODO_HIP_M, KIMODO_PY, soma_calibration, soma_to_canonical
    from kimodo_names import NAMES77
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for wid in ("W05F", "W50M", "W95M"):
        w = WORKERS[wid]
        sk = Skeleton.for_worker(w)
        S = PostureSolver(sk, w)
        s = KIMODO_HIP_M / float(sk.world["pelvis"][1])
        cal = soma_calibration(sk, s)
        for rname, route in ROUTES.items():
            d = np.subtract(route[1], route[0])
            y0 = math.degrees(math.atan2(d[0], d[1]))
            dl = np.subtract(route[-1], route[-2])
            y1 = math.degrees(math.atan2(dl[0], dl[1]))
            ours = idle(0.4) + walk(w, sk, route, y0, y1)
            last = next(f.root for f in reversed(ours) if f.root is not None)
            for f in ours:
                f.root = f.root or (route[0][0], route[0][1], y0)
            tail = idle(0.6)
            for f in tail:
                f.root = last
            ours += tail
            so = solve_frames(S, ours, log=lambda *a: None)
            m_o = measure(sk, ours, so, route)
            # Kimodo along our walk's timing: root waypoints every 8 frames
            wp = [(i / FPS, (ours[i].root[0] * s, ours[i].root[1] * s)) for i in range(0, len(ours), 8)]
            job = {"seconds": len(ours) / FPS, "seed": 1, "steps": 100, "first_heading": math.radians(y0),
                   "constraints": [{"type": "root2d", "frame_indices": [int(round(t * 30)) for t, _ in wp],
                                    "smooth_root_2d": [[x, z] for _, (x, z) in wp]}]}
            tag = f"{wid}_{rname.replace(' ', '_')}"
            jp, gp = os.path.join(OUT, tag + ".json"), os.path.join(OUT, tag + ".npz")
            json.dump(job, open(jp, "w"))
            subprocess.run([KIMODO_PY, os.path.join(HERE, "kimodo_run.py"), jp, gp], capture_output=True,
                           env=dict(os.environ, HF_HOME=r"C:/OmniverseData/kimodo/hf"))
            g = np.load(gp)
            pj, kfps = g["posed_joints"], float(g["fps"])
            kf = []
            for i in range(len(ours)):
                x = min(i / FPS * kfps, len(pj) - 1)
                a = int(x)
                b = min(a + 1, len(pj) - 1)
                J = soma_to_canonical(pj[a] * (1 - (x - a)) + pj[b] * (x - a), NAMES77, s)
                pose = retarget(sk, J, cal)
                pose.pelvis = J["pelvis"] - sk.world["pelvis"]
                st = Stance()
                st.free = True
                kf.append(Frame([], None, {}, root=(0.0, 0.0, 0.0), stance=st, base=pose))
            from kimodo_job import floor_fix
            floor_fix(sk, [f.base for f in kf])
            sk_ = solve_frames(S, kf, log=lambda *a: None)
            m_k = measure(sk, kf, sk_, route)
            rows.append({"body": wid, "route": rname, "ours": m_o, "kimodo": m_k})
            print(f"{wid} {rname:13s} ours slip {m_o['slip_median_cm']:5.2f}/{m_o['slip_p90_cm']:5.2f} end {m_o['end_error_cm']:5.1f}"
                  f" pops {m_o['pops']:2d} | kimodo slip {m_k['slip_median_cm']:5.2f}/{m_k['slip_p90_cm']:5.2f}"
                  f" end {m_k['end_error_cm']:5.1f} pops {m_k['pops']:2d} contact {m_k['contact_cm']}", flush=True)
    json.dump(rows, open(os.path.join(ROOT, "out", "people", "kimodo", "walk_shootout.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
