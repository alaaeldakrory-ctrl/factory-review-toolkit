"""Kit script: one picture per step of solved tasks, at the middle of each step's main working phase.

    env TASK_RESULTS = result.json paths separated by ';'    TASK_SHOTS = output folder
    kit.exe ... --enable frt.scene --exec tools/kit/task_shots.py
-> <out>/<task>/<nn>_<step>.png and shots.json
"""
import asyncio
import json
import math
import os
import traceback

import carb
import omni.kit.app
import omni.kit.renderer_capture
import omni.timeline
import omni.usd
from pxr import Gf, Usd, UsdGeom

RESULTS = [p for p in os.environ.get("TASK_RESULTS", "").split(";") if p]
OUT = os.environ.get("TASK_SHOTS", os.path.join(os.environ.get("TEMP", "."), "task_shots"))
WORK = ("thread", "tighten", "drive", "push", "turn", "grasp", "transfer", "tie", "align", "pick", "place",
        "inspect", "press", "move")


async def frames(n):
    for _ in range(n):
        await omni.kit.app.get_app().next_update_async()


def look_at(eye, at):
    from omni.kit.viewport.utility import get_active_viewport
    from omni.kit.viewport.utility.camera_state import ViewportCameraState
    vp = get_active_viewport()
    vp.camera_path = "/OmniverseKit_Persp"
    cs = ViewportCameraState("/OmniverseKit_Persp", vp)
    cs.set_position_world(Gf.Vec3d(*eye), True)
    cs.set_target_world(Gf.Vec3d(*at), True)


def pick_frame(step):
    best = None
    for lab, a, b in step["phases"]:
        if lab == "walk":
            continue
        score = (b - a) * (2.0 if any(w in lab for w in WORK[:6]) else 1.0)
        if best is None or score > best[0]:
            best = (score, (a + b) // 2, lab)
    return (best[1], best[2]) if best else ((step["start"] + step["end"]) // 2, "")


async def main():
    rep = []
    try:
        s = carb.settings.get_settings()
        s.set("/app/renderer/skipWhileMinimized", False)
        s.set("/app/viewport/grid/enabled", False)
        import omni.ui as ui
        await frames(120)
        for name in ("Ergonomics", "Factory Sim"):
            w = ui.Workspace.get_window(name)
            if w is not None:
                w.visible = False
        rc = omni.kit.renderer_capture.acquire_renderer_capture_interface()
        from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport
        vp = get_active_viewport()
        tl = omni.timeline.get_timeline_interface()
        for res_path in RESULTS:
            res = json.load(open(res_path))
            task = os.path.basename(os.path.dirname(res_path))
            od = os.path.join(OUT, task)
            os.makedirs(od, exist_ok=True)
            await omni.usd.get_context().open_stage_async(res["usd"])
            await frames(240)
            st = omni.usd.get_context().get_stage()
            wp = st.GetPrimAtPath("/World/Asset/Worker")
            for k, step in enumerate(res["steps"]):
                f, lab = pick_frame(step)
                tl.set_current_time(f / 24.0)
                M = UsdGeom.Xformable(wp).ComputeLocalToWorldTransform(Usd.TimeCode(f))
                p = M.ExtractTranslation()
                yaw = 0.0
                for op in UsdGeom.Xformable(wp).GetOrderedXformOps():
                    if op.GetOpName().endswith("rotateY"):
                        yaw = float(op.Get(Usd.TimeCode(f)) or 0.0)
                a = math.radians(yaw)
                fwd = (math.sin(a), math.cos(a))
                left = (math.cos(a), -math.sin(a))
                foc = step.get("focus") or (p[0] + fwd[0] * 35, 105.0, p[2] + fwd[1] * 35)
                at = ((foc[0] + p[0]) / 2, (foc[1] + 95.0) / 2, (foc[2] + p[2]) / 2)   # the worker and the work
                side = -1.0 if k % 2 else 1.0                                       # alternate sides
                eye = (at[0] - fwd[0] * 110 + side * left[0] * 210, at[1] + 70.0, at[2] - fwd[1] * 110 + side * left[1] * 210)
                look_at(eye, at)
                await frames(90)
                fn = os.path.join(od, f"{k + 1:02d}_{step['do']}.png")
                cap = capture_viewport_to_file(vp, fn)
                if hasattr(cap, "wait_for_result"):
                    await cap.wait_for_result(completion_frames=10)
                await frames(10)
                rep.append({"task": task, "step": step["id"], "frame": f, "phase": lab, "png": fn})
    except Exception:
        rep.append({"error": traceback.format_exc()})
    json.dump(rep, open(os.path.join(OUT, "shots.json"), "w"), indent=1)
    omni.kit.app.get_app().post_quit()
    await frames(30)
    os._exit(0)


asyncio.ensure_future(main())
