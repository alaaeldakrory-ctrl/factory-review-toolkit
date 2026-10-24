"""Kit script: a multi-camera review of solved tasks - for each step, its main moments (the middle of each working
phase) from four sides (front, left, right, back-high), framed on the worker and the work.

    env TASK_RESULTS = result.json paths separated by ';'    TASK_SHOTS = output folder
        TASK_VIEWS = views to take (default "front,left,right,back")    TASK_MOMENTS = moments per step (default 3)
    kit.exe ... --enable frt.scene --exec tools/kit/task_review.py
-> <out>/<task>/<step>_<frame>_<view>.png, shots.json; then tools/review_sheet.py makes the contact sheets.
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
OUT = os.environ.get("TASK_SHOTS", os.path.join(os.environ.get("TEMP", "."), "task_review"))
VIEWS = os.environ.get("TASK_VIEWS", "front,left,right,back").split(",")
MOMENTS = int(os.environ.get("TASK_MOMENTS", "3"))
# view -> (forward, left, up) camera offset in the worker's frame, cm
OFFSET = {"front": (230.0, 45.0, 30.0), "left": (25.0, 230.0, 30.0), "right": (25.0, -230.0, 30.0),
          "back": (-170.0, 60.0, 130.0), "top": (20.0, 10.0, 260.0)}


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


def scene_boxes(stage):
    """World boxes of the still equipment (not the worker, the parts or the floor), for line-of-sight tests."""
    from pxr import UsdGeom
    bc = UsdGeom.BBoxCache(Usd.TimeCode(0), ["default", "render"])
    out = []
    root = stage.GetPrimAtPath("/World/Asset/Scene")
    for p in (root.GetChildren() if root else []):
        for q in Usd.PrimRange(p):
            if not q.IsA(UsdGeom.Gprim):
                continue
            r = bc.ComputeWorldBound(q).ComputeAlignedRange()
            if r.IsEmpty() or r.GetMax()[1] < 3.0:
                continue
            out.append((tuple(r.GetMin()), tuple(r.GetMax())))
    return out


def blocked(eye, at, boxes):
    """Whether the segment eye -> at passes through any box (slab test), ignoring the last 35 cm near the target."""
    d = [at[i] - eye[i] for i in range(3)]
    L = math.sqrt(sum(x * x for x in d)) or 1.0
    t_end = max(0.0, 1.0 - 35.0 / L)
    for lo, hi in boxes:
        t0, t1 = 0.0, t_end
        for i in range(3):
            if abs(d[i]) < 1e-9:
                if eye[i] < lo[i] or eye[i] > hi[i]:
                    break
                continue
            a, b = (lo[i] - eye[i]) / d[i], (hi[i] - eye[i]) / d[i]
            t0, t1 = max(t0, min(a, b)), min(t1, max(a, b))
            if t0 > t1:
                break
        else:
            return True
    return False


def camera(at, fwd, left, view, boxes):
    """A clear eye position for a view: the nominal one, else closer, else swung round a little."""
    o = OFFSET[view]
    for k, swing in ((1.0, 0), (0.7, 0), (1.0, 30), (1.0, -30), (0.5, 0), (0.7, 55), (0.7, -55)):
        a = math.radians(swing)
        f_, l_ = o[0] * math.cos(a) - o[1] * math.sin(a), o[0] * math.sin(a) + o[1] * math.cos(a)
        eye = (at[0] + (fwd[0] * f_ + left[0] * l_) * k, at[1] + o[2] * (0.6 + 0.4 * k),
               at[2] + (fwd[1] * f_ + left[1] * l_) * k)
        if not blocked(eye, at, boxes):
            return eye
    return eye


def moments(step, n):
    """The middle frames of the step's longest working phases (not walks), in time order."""
    ph = [(b - a, (a + b) // 2, lab) for lab, a, b in step["phases"] if lab != "walk" and b > a]
    ph = sorted(sorted(ph, reverse=True)[:n], key=lambda x: x[1])
    return [(f, lab) for _, f, lab in ph] or [((step["start"] + step["end"]) // 2, "")]


async def main():
    rep = []
    try:
        s = carb.settings.get_settings()
        s.set("/app/renderer/skipWhileMinimized", False)
        s.set("/app/viewport/grid/enabled", False)
        import omni.ui as ui
        await frames(120)
        for name in ("Ergonomics", "Factory Sim", "Task Builder"):
            w = ui.Workspace.get_window(name)
            if w is not None:
                w.visible = False
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
            boxes = scene_boxes(st)
            for step in res["steps"]:
                for f, lab in moments(step, MOMENTS):
                    tl.set_current_time(f / 24.0)
                    p = UsdGeom.Xformable(wp).ComputeLocalToWorldTransform(Usd.TimeCode(f)).ExtractTranslation()
                    yaw = 0.0
                    for op in UsdGeom.Xformable(wp).GetOrderedXformOps():
                        if op.GetOpName().endswith("rotateY"):
                            yaw = float(op.Get(Usd.TimeCode(f)) or 0.0)
                    a = math.radians(yaw)
                    fwd, left = (math.sin(a), math.cos(a)), (math.cos(a), -math.sin(a))
                    foc = step.get("focus")                   # between the chest and the work, so both are in
                    hy = 118.0 if foc is None else 0.5 * 118.0 + 0.5 * float(foc[1])
                    at = (p[0] + fwd[0] * 22.0, hy, p[2] + fwd[1] * 22.0)
                    for v in VIEWS:
                        eye = camera(at, fwd, left, v, boxes)
                        look_at(eye, at)
                        await frames(45)
                        fn = os.path.join(od, f"{step['id']}_{f:05d}_{v}.png")
                        cap = capture_viewport_to_file(vp, fn)
                        if hasattr(cap, "wait_for_result"):
                            await cap.wait_for_result(completion_frames=8)
                        await frames(5)
                        rep.append({"task": task, "step": step["id"], "frame": f, "phase": lab, "view": v, "png": fn})
    except Exception:
        rep.append({"error": traceback.format_exc()})
    os.makedirs(OUT, exist_ok=True)
    json.dump(rep, open(os.path.join(OUT, "shots.json"), "w"), indent=1)
    omni.kit.app.get_app().post_quit()
    await frames(30)
    os._exit(0)


asyncio.ensure_future(main())
