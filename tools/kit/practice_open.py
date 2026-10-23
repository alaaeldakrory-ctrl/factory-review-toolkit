"""Kit startup script for the practice session: open a scene, place the camera, open Window > Task Builder.

    kit.exe apps/my_company.my_usd_composer.kit --ext-folder <repo>/exts --enable frt.scene
            --exec "tools/kit/practice_open.py"
    env PRACTICE_USD = the scene to open (default: out/people/task/my_task/scene.usda)
        PRACTICE_TASK = the task file the window opens (default: docs/people/practice/my_task.toml)
        PRACTICE_SHOT = a png to capture once the window is up (optional, for checking)
"""
import asyncio
import os

import carb
import omni.kit.app
import omni.kit.renderer_capture
import omni.usd
from pxr import Gf

REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
USD = os.environ.get("PRACTICE_USD", os.path.join(REPO, "out", "people", "task", "my_task", "scene.usda"))
SHOT = os.environ.get("PRACTICE_SHOT", "")


async def frames(n):
    for _ in range(n):
        await omni.kit.app.get_app().next_update_async()


async def main():
    await frames(60)
    await omni.usd.get_context().open_stage_async(USD)
    await frames(120)
    from omni.kit.viewport.utility import get_active_viewport
    from omni.kit.viewport.utility.camera_state import ViewportCameraState
    vp = get_active_viewport()
    vp.camera_path = "/OmniverseKit_Persp"
    cs = ViewportCameraState("/OmniverseKit_Persp", vp)
    cs.set_position_world(Gf.Vec3d(-55.0, 470.0, -690.0), True)          # behind-above the worker side
    cs.set_target_world(Gf.Vec3d(-55.0, 70.0, 80.0), True)
    import omni.ui as ui
    for name in ("Ergonomics", "Factory Sim"):                         # the app's older panels, not needed here
        old = ui.Workspace.get_window(name)
        if old is not None:
            old.visible = False
    import frt_scene.extension as X
    w = X.show_task()
    await frames(60)
    task = os.environ.get("PRACTICE_TASK", os.path.join(REPO, "docs", "people", "practice", "my_task.toml"))
    if w is not None:
        w.load(task)
        w.window.frame.rebuild()
    if SHOT:
        await frames(200)
        rc = omni.kit.renderer_capture.acquire_renderer_capture_interface()
        rc.capture_next_frame_swapchain(SHOT)
        await frames(30)
    carb.log_warn("[practice] ready: " + USD)


asyncio.ensure_future(main())
