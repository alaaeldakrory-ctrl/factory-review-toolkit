"""Run a process file end to end: build the demo workplace, plan the steps (walks, waits, giving way), solve and bake
every worker, check the rules, write the scene + a report (findings, per-worker summary, timeline).

    python tools/process_demo.py [docs/people/example_process.toml] [out/people/process/process.usd] [--step 2]
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "exts", "frt.core"))
from pxr import UsdGeom   # noqa: E402

from frt_core import catalog as CAT   # noqa: E402
from frt_core import units as U   # noqa: E402
from frt_core.geom import Shapes   # noqa: E402
from frt_core.materials import MaterialLibrary   # noqa: E402
from frt_core.people import process as PR   # noqa: E402
from frt_core.people import rules as RU   # noqa: E402
from frt_core.people.bake import bake   # noqa: E402
from frt_core.people.motion_gen import FPS   # noqa: E402
from frt_core.people.navigation import NavGrid   # noqa: E402
from frt_core.studio import dress   # noqa: E402
from frt_core.usdio import safe_stage   # noqa: E402


def workplace(st, lib):
    """Demo workplace matching docs/people/example_process.toml."""
    CAT.place(st, "/World/Asset", "industrial_table", "Bench", (0, 0, 60), 180.0, {"length": 200, "depth": 80,
              "height": 92}, lib)
    CAT.place(st, "/World/Asset", "industrial_table", "PartsTable", (-250, 0, -150), 0.0, {"length": 120, "depth": 70,
              "height": 90}, lib)
    s = Shapes()
    for bx in (-45.0, 45.0):                                       # brackets with bolts on the bench
        s.rbox((bx, 94.0, 38.0), (30.0, 4.0, 20.0), 0.8, "powder_blue", seg=2)
        for dx in (-8.0, 8.0):
            s.bolt((bx + dx, 96.0, 38.0), (0, 1, 0), 1.6, "bolt_zinc")
    s.rbox((248.0, 124.0, 37.0), (70.0, 1.6, 1.6), 0.7, "cable_pvc", seg=1)                 # harness on the panel
    for bx in (-275.0,):                                           # spare bin on the parts table
        s.rbox((bx, 97.0, -140.0), (30.0, 14.0, 22.0), 1.0, "plastic_blue", seg=2)
    s.rbox((230.0, 60.0, 42.0), (90.0, 120.0, 4.0), 1.0, "powder_lightgrey", seg=2)       # wiring panel on legs
    for lx in (192.0, 268.0):
        s.rbox((lx, 0.5, 42.0), (6.0, 1.0, 30.0), 0.3, "steel_dark", seg=1)
    s.rbox((230.0, 112.0, 39.0), (4.0, 3.0, 2.0), 0.4, "plastic_black", seg=1)            # connector socket
    s.rbox((212.0, 100.0, 39.6), (16.0, 12.0, 0.8), 0.4, "powder_dark", seg=1)            # cover with a screw
    s.build(st, "/World/Asset/Work", lib.refs(list(s.groups)))
    UsdGeom.Xform.Define(st, "/World/Asset/Work")


def main(proc_path, out, step=2):
    t0 = time.time()
    proc = PR.load(proc_path)
    with safe_stage(out) as st:
        U.setup_stage(st, fps=FPS)
        st.SetDefaultPrim(UsdGeom.Xform.Define(st, "/World").GetPrim())
        lib = MaterialLibrary(st, "/World/Looks")
        UsdGeom.Xform.Define(st, "/World/Asset")
        workplace(st, lib)
        nav = NavGrid.from_stage(st, ["/World/Asset"])
        pl = PR.Planner(proc, nav)
        plan = pl.plan()
        print(f"plan: {plan.length} frames ({plan.length / FPS:.1f} s); " + "; ".join(plan.notes))
        UsdGeom.Xform.Define(st, "/World/Asset/People")
        baked = {}
        for wid in pl.workers:
            frames = pl.frames_of(plan, wid)
            w = pl.workers[wid]
            root = (float(w["start"][0]), float(w["start"][1]), float(w.get("yaw", 0.0)))
            for sg in plan.segments[wid]:                         # frames at a work spot carry it as their root
                for f in sg.frames:
                    if f.root is None:
                        f.root = sg.spot
            cur = root
            for f in frames:
                if f.root is None:
                    f.root = cur
                cur = f.root
            UsdGeom.Xform.Define(st, f"/World/Asset/People/{wid}")
            baked[wid] = bake(st, f"/World/Asset/People/{wid}", pl.body[wid], frames, lib, solver=pl.solver[wid],
                              step=step)
        findings, summary = RU.evaluate(baked, plan, pl)
        dress(st, lib, st.GetPrimAtPath("/World/Asset"))
    timeline = {wid: [{"kind": sg.kind, "step": sg.step, "start_s": round(sg.start / FPS, 2),
                       "end_s": round(sg.end / FPS, 2)} for sg in plan.segments[wid]] for wid in plan.segments}
    report = {"process": proc.get("process", {}).get("name", ""), "seconds": round(plan.length / FPS, 1),
              "notes": plan.notes, "summary": summary, "findings": findings, "timeline": timeline}
    with open(os.path.splitext(out)[0] + "_report.json", "w") as f:
        json.dump(report, f, indent=1, default=str)
    print(json.dumps({k: report[k] for k in ("seconds", "notes", "summary")}, indent=1))
    for fd in findings:
        print(f"  [{fd['severity']}] {fd['worker']} {fd['step']}: {fd['what']} - {fd['why']} -> {fd['fix']}")
    print(f"-> {out} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    step = int(next((a.split("=")[1] for a in sys.argv[1:] if a.startswith("--step=")), 2))
    main(args[0] if args else os.path.join(HERE, "..", "docs", "people", "example_process.toml"),
         args[1] if len(args) > 1 else os.path.join(HERE, "..", "out", "people", "process", "process.usd"), step)
