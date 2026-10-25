"""A line of operators for the ergonomics showcase: several library jobs (solved on the website's character, Kimodo
bodies, reviewed) placed side by side as stations, each looping its own cycle, composed into ONE USD scene - then
scanned by Ergo Scan (the USD_Ergo repo's scanner, the same one an agent runs on any scene) as a scene it knows
nothing about.

    python tools/ergo_line.py <out dir> [--web=<site>/ergo/line] [--seconds=24]

-> <out>/line.usda (+ ops/*.usda), <out>/report/ (ergo_report.md / .json, ergo_dashboard.html, timeline, floor map,
   score panel), and with --web the viewer folders (one per operator: fixture, tools, one cycle) + line.json with
   placements, cycle lengths and the scan's per-frame scores and body-part scores.
"""
import json
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path[:0] = [HERE, os.path.join(HERE, "kimodo"), os.path.join(ROOT, "exts", "frt.core")]
ERGO = os.environ.get("FRT_ERGO_SCAN", os.path.join(ROOT, "..", "ergo-scan", "exts", "ergo.scan"))
LIB = os.path.join(ROOT, "out", "people", "library")
FPS = 24

# (operator, library job, row, what the station does, the work moved (cm: x, y up, z away) or None): row 0 faces +Z,
# row 1 across the aisle faces -Z. Two stations are the same job as their neighbour with the work set too low - the
# layouts the scan is there to find.
LINE = [("Operator_01", "screwdriver", 0, "Drive the cover screws - panel at elbow height", None),
        ("Operator_02", "screwdriver", 0, "Drive the cover screws - panel 40 cm lower", (0.0, -40.0, 0.0)),
        ("Operator_03", "torque", 0, "Torque a bolt to the click", None),
        ("Operator_04", "kneel_drive", 0, "Kneel at the base and drive a fastener", None),
        ("Operator_05", "guide_hoist", 1, "Guide a hoisted counterweight", None),
        ("Operator_06", "lug_torque", 1, "Torque the hub's lug nuts", None),
        ("Operator_07", "box", 1, "Move a box along the bench", None),
        ("Operator_08", "box", 1, "Move a box along a bench 45 cm lower", (0.0, -45.0, 0.0))]
LOADS = {"Operator_07": 1.5, "Operator_08": 1.5}     # kg handled (a light carton); all cycles repeat (+1 muscle use)
AISLE = 260.0          # cm between the two rows of operators (backs to the aisle)
GAP = 90.0             # cm between neighbouring stations


def loop_cycle(frames, sols, n_blend=12):
    """One cycle that loops: the last n_blend frames ease back into the first pose."""
    from frt_core.people.bake import blend
    from frt_core.people.solver import Solution
    frames, sols = list(frames), list(sols)
    n = len(sols)
    for k in range(n_blend):
        i = n - n_blend + k
        u = (k + 1) / (n_blend + 1)
        u = u * u * (3 - 2 * u)
        sols[i] = Solution(blend(sols[i].pose, sols[0].pose, u), sols[i].cost)
    return frames, sols


def solved(jid, ch, out, d=None):
    """(frames, sols, M) of a library job on the character, Kimodo-bodied: the library's own clip, or with the work
    moved by d the same pipeline run again (rules on the character -> Kimodo -> final pass), cached in out/var."""
    if d is None:
        k = pickle.load(open(os.path.join(LIB, jid, "kimodo.pkl"), "rb"))
        return k["frames"], k["sols"], np.eye(4)
    vid = f"{jid}_{'_'.join(f'{v:+.0f}' for v in d)}"
    vout = os.path.join(out, "var", vid)
    p = os.path.join(vout, "kimodo.pkl")
    if not os.path.isfile(p):
        import kimodo_job as KJ
        from frt_core.people import api
        from frt_core.people.placement import move_rules
        r = api.solve_job(jid, worker=ch, work={"d": d}, step=False, cache=False)
        src = {"frames": r.frames, "sols": r.sols, "sk": ch.skeleton, "worker": ch.worker, "out": vout}
        so, se = KJ.generate([KJ.prepare(jid, src=src)])
        if "ok (" not in so:
            raise RuntimeError(f"Kimodo failed on {vid}: {se[-1500:]}")
        print(f"  {vid}: {KJ.finalize(jid, src, keep_out=move_rules(r.job.keep_out_rules(), r.M))}", flush=True)
        np.save(os.path.join(vout, "M.npy"), r.M)
    k = pickle.load(open(p, "rb"))
    return k["frames"], k["sols"], np.load(os.path.join(vout, "M.npy"))


def op_usd(name, jid, ch, out, reps, d=None):
    """The job looped `reps` times, as a standalone job USD (fixture, worker, tools)."""
    from frt_core.people.api import JobResult, write_job_usd
    from frt_core.people.jobs import BY_ID
    f0, s0, M = solved(jid, ch, out, d)
    fr, so = loop_cycle(f0, s0)
    res = JobResult(BY_ID[jid], ch.worker, M, fr * reps, so * reps, {"ok": True, "counts": {}, "issues": []},
                    (0.0, 0.0), [], None)
    res.character = ch
    p = os.path.join(out, "ops", f"{name}.usda")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    write_job_usd(res, p)
    return p, (fr, so, M)


def fixture_extent(usd):
    """x range (cm) of the station's fixture and the worker's reach, in the job frame."""
    from pxr import Usd, UsdGeom
    st = Usd.Stage.Open(usd)
    bb = UsdGeom.BBoxCache(0, ["default", "render"]).ComputeWorldBound(st.GetPrimAtPath("/World/Job/Fixture"))
    r = bb.ComputeAlignedRange()
    return min(float(r.GetMin()[0]), -45.0), max(float(r.GetMax()[0]), 45.0)


def compose(out, ops):
    """line.usda: every station referenced at its place on the line (api.add_to_scene)."""
    from pxr import Usd, UsdGeom
    from frt_core.people.api import add_to_scene
    from frt_core import units as U
    p = os.path.join(out, "line.usda")
    st = Usd.Stage.CreateNew(p)
    U.setup_stage(st, fps=FPS)
    st.SetDefaultPrim(UsdGeom.Xform.Define(st, "/World").GetPrim())
    st.SetStartTimeCode(0)
    for o in ops:
        add_to_scene(st, f"/World/{o['name']}", o["usd"], at=(o["x"], 0.0, o["z"]), yaw=o["yaw"])
    st.GetRootLayer().Save()
    return p


def main():
    opt = {a.split("=")[0][2:]: a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--") and "=" in a}
    out = os.path.abspath([a for a in sys.argv[1:] if not a.startswith("--")][0])
    seconds = float(opt.get("seconds", 24))
    from library_build import character
    ch = character()
    ops = []
    for name, jid, row, what, d in LINE:
        n = len(pickle.load(open(os.path.join(LIB, jid, "kimodo.pkl"), "rb"))["sols"])
        reps = int(np.ceil(seconds * FPS / n))
        usd, cyc = op_usd(name, jid, ch, out, reps, d)
        x0, x1 = fixture_extent(usd)
        ops.append({"name": name, "job": jid, "row": row, "what": what, "usd": usd, "cycle": n, "reps": reps,
                    "x0": x0, "x1": x1, "_cyc": cyc})
        print(f"{name}: {jid} cycle {n / FPS:.1f} s x {reps}", flush=True)
    for row in (0, 1):                                   # pack each row along X, centred on 0
        rs = [o for o in ops if o["row"] == row]
        w = sum(o["x1"] - o["x0"] for o in rs) + GAP * (len(rs) - 1)
        x = -w / 2
        for o in (rs if row == 0 else rs[::-1]):
            if row == 0:
                o["x"] = x - o["x0"]
            else:                                        # turned 180: its +x side is on the line's -x side
                o["x"] = x + o["x1"]
            x += (o["x1"] - o["x0"]) + GAP
            o["z"] = 0.0 if row == 0 else -AISLE
            o["yaw"] = 0.0 if row == 0 else 180.0
    line = compose(out, ops)
    print(f"scene: {line}", flush=True)

    # --- the scan: Ergo Scan on the composed scene, as it would run on any customer scene
    from pxr import Usd
    ES = ergo_scanner()
    st = Usd.Stage.Open(line)
    end = int(round(seconds * FPS)) - 1
    res = ES.scan(st, root="/World", step=1, frames=(0, end), loads=LOADS, repeated=[o[0] for o in LINE])
    res["stage"] = "line.usda"                         # (the report is published: no local paths)
    rep = os.path.join(out, "report")
    summ = ES.write_report(res, rep, title="Assembly line - ergonomic scan (RULA)")
    for s in summ:
        print(f"  {s['operator']}: max {s.get('max')} p90 {s.get('p90')} median {s.get('median')} "
              f"driver {s.get('main_driver')}", flush=True)
    if "web" in opt:
        web(opt["web"], ops, res, summ, ch, end + 1)


def ergo_scanner():
    """The USD_Ergo package's scan module (tools/ergo_scan.py, this repo's CLI, has the same name)."""
    import importlib
    sys.modules.pop("ergo_scan", None)
    sys.path.insert(0, ERGO)
    try:
        return importlib.import_module("ergo_scan.scan")
    finally:
        sys.path.remove(ERGO)


PARTS = ("trunk_s", "neck_s", "upper_arm_s", "lower_arm_s", "wrist_s", "legs_s")


def web(dst, ops, res, summ, ch, n):
    """The viewer's line: one job folder per operator (one cycle, fixture + tools) and line.json with the scan."""
    from web_export import export_job
    os.makedirs(dst, exist_ok=True)
    by = {o["name"]: o for o in res["operators"]}
    sm = {s["operator"]: s for s in summ}
    meta = {"fps": FPS, "frames": n, "aisle_cm": AISLE, "operators": []}
    for o in ops:
        fr, so, M = o["_cyc"]
        export_job(o["job"], fr, so, ch, os.path.join(dst, o["name"]), M=M)
        rows = by[o["name"]]["rows"][:n]
        s = sm[o["name"]]
        meta["operators"].append({
            "name": o["name"], "job": o["job"], "dir": o["name"], "what": o["what"], "x": o["x"] / 100.0, "z": o["z"] / 100.0,
            "yaw": o["yaw"], "cycle": o["cycle"],
            "rula": [int(r["score"]) for r in rows],
            "parts": {k: [int(r[k]) for r in rows] for k in PARTS},
            "angles": {k: [round(float(r[k]), 0) for r in rows] for k in ("trunk_deg", "neck_deg", "upper_arm_deg",
                                                                          "knee_deg")},
            "summary": {k: s[k] for k in ("max", "p90", "median", "band_pct", "main_driver", "worst", "walking_pct")}})
    json.dump(meta, open(os.path.join(dst, "line.json"), "w"), separators=(",", ":"))
    print(f"web: {dst}")


if __name__ == "__main__":
    main()
