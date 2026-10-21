"""Window > Factory Review: Scene Builder, findings, revision compare, packaging.

All the work is done by frt_core (the same code as the command line); this file is only buttons and lists.
"""
import asyncio
import os
import traceback

import carb
import omni.kit.app
import omni.ui as ui
import omni.usd
from pxr import Gf

from frt_core import drawing as DRW
from frt_core import package as PKG
from frt_core import recipe as RCP
from frt_core import revision as REV
from frt_core import tags as T
from frt_core import usdio
from frt_core import validate as VAL
from frt_core.findings import FindingList

TITLE = "Factory Review"
REPO = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DEFAULT_RECIPE = os.path.join(REPO, "samples", "recipes", "demo_plant.toml")
SETTING = "/persistent/exts/frt.scene/"

GREY, WHITE = 0xFF9A9A9A, 0xFFE8E8E8
SEV = {"critical": 0xFF3A3AE0, "warning": 0xFF30A8F0, "info": 0xFFC8A040}      # ABGR: red, amber, blue


def _setting(key, default=""):
    v = carb.settings.get_settings().get(SETTING + key)
    return v if v else default


def _set_setting(key, value):
    carb.settings.get_settings().set(SETTING + key, value)


def _stage_file():
    st = omni.usd.get_context().get_stage()
    if not st:
        return ""
    p = st.GetRootLayer().realPath
    return p if p and os.path.isfile(p) else ""


def _same(a, b):
    return bool(a and b) and os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))


def look_at(eye, at):
    """Move the viewport's own perspective camera (never a scene camera) to eye, looking at `at`."""
    from omni.kit.viewport.utility import get_active_viewport
    from omni.kit.viewport.utility.camera_state import ViewportCameraState
    vp = get_active_viewport()
    if vp is None:
        return
    vp.camera_path = "/OmniverseKit_Persp"
    cs = ViewportCameraState("/OmniverseKit_Persp", vp)
    cs.set_position_world(Gf.Vec3d(*eye), True)
    cs.set_target_world(Gf.Vec3d(*at), True)


class ReviewWindow:
    def __init__(self):
        old = ui.Workspace.get_window(TITLE)
        if old is not None:
            old.visible = False
        self.findings = FindingList()
        self._busy = False
        self.window = ui.Window(TITLE, width=480, height=860)
        self._build()
        self._stage_sub = omni.usd.get_context().get_stage_event_stream().create_subscription_to_pop(
            self._on_stage_event, name="frt review stage")
        self._refresh_scene()

    def destroy(self):
        self._stage_sub = None
        if self.window:
            self.window.destroy()
            self.window = None

    # ------------------------------------------------------------------ layout
    def _build(self):
        with self.window.frame:
            with ui.ScrollingFrame():
                with ui.VStack(spacing=6, height=0):
                    self._scene_lbl = ui.Label("", word_wrap=True, style={"color": GREY})
                    with ui.CollapsableFrame("1  Scene Builder", collapsed=False):
                        with ui.VStack(spacing=4, height=0):
                            ui.Label("Recipe (.toml): drawing, building, equipment, extra items", style={"color": GREY})
                            self._recipe = ui.StringField(height=24)
                            self._recipe.model.set_value(_setting("recipe", DEFAULT_RECIPE))
                            with ui.HStack(height=28, spacing=4):
                                ui.Button("Build + open", clicked_fn=lambda: self._run(self._build_scene(True)))
                                ui.Button("Build only", clicked_fn=lambda: self._run(self._build_scene(False)))
                                ui.Button("Show folder", width=90, clicked_fn=self._show_folder)
                            self._build_lbl = ui.Label("", word_wrap=True, style={"color": GREY})
                    with ui.CollapsableFrame("2  Review", collapsed=False):
                        with ui.VStack(spacing=4, height=0):
                            with ui.HStack(height=28, spacing=4):
                                ui.Button("Validate open scene", clicked_fn=lambda: self._run(self._validate()))
                                ui.Button("Load saved findings", clicked_fn=self._load_findings)
                            self._count_lbl = ui.Label("", style={"color": WHITE})
                            self._list = ui.VStack(spacing=2, height=0)
                    with ui.CollapsableFrame("3  Stations", collapsed=True):
                        self._stations = ui.VStack(spacing=2, height=0)
                    with ui.CollapsableFrame("4  New drawing revision", collapsed=False):
                        with ui.VStack(spacing=4, height=0):
                            ui.Label("New DXF or PDF of this plant: see what changed, then update the scene",
                                     word_wrap=True, style={"color": GREY})
                            self._newdrw = ui.StringField(height=24)
                            self._newdrw.model.set_value(_setting("new_drawing", ""))
                            with ui.HStack(height=28, spacing=4):
                                ui.Button("Compare with scene", clicked_fn=lambda: self._run(self._compare()))
                                ui.Button("Update scene to it", clicked_fn=lambda: self._run(self._update_to_new()))
                    with ui.CollapsableFrame("5  Package for another PC", collapsed=True):
                        with ui.VStack(spacing=4, height=0):
                            ui.Label("Output folder (everything is copied in, paths made relative)",
                                     style={"color": GREY})
                            self._pkg = ui.StringField(height=24)
                            self._pkg.model.set_value(_setting("package_dir", ""))
                            ui.Button("Package open scene", height=28, clicked_fn=lambda: self._run(self._package()))
                            self._pkg_lbl = ui.Label("", word_wrap=True, style={"color": GREY})

    # ------------------------------------------------------------------ helpers
    def _run(self, coro):
        if self._busy:
            carb.log_warn("[frt] busy, wait for the current step to finish")
            coro.close()
            return
        asyncio.ensure_future(self._guard(coro))

    async def _guard(self, coro):
        self._busy = True
        try:
            await coro
        except Exception as e:   # noqa: BLE001
            carb.log_error("[frt] " + traceback.format_exc())
            self._build_lbl.text = f"Error: {e}"
        finally:
            self._busy = False

    async def _frames(self, n=2):
        for _ in range(n):
            await omni.kit.app.get_app().next_update_async()

    def _on_stage_event(self, e):
        if e.type == int(omni.usd.StageEventType.OPENED):
            self._refresh_scene()

    def _refresh_scene(self):
        path = _stage_file()
        self._scene_lbl.text = f"Open scene: {path}" if path else "Open scene: (unsaved or none)"
        saved = os.path.splitext(path)[0] + ".findings.json" if path else ""
        if saved and os.path.isfile(saved):
            self.findings = FindingList.load(saved)
        else:
            self.findings = FindingList()
        self._show_findings()
        self._show_stations()

    def _show_folder(self):
        r = self._recipe.model.get_value_as_string().strip()
        if os.path.isfile(r):
            os.startfile(os.path.dirname(r))

    # ------------------------------------------------------------------ 1 build
    async def _build_scene(self, open_after, recipe=None):
        path = self._recipe.model.get_value_as_string().strip().strip('"')
        _set_setting("recipe", path)
        r = recipe or RCP.load(path)
        out = r["scene"]["output"]
        ctx = omni.usd.get_context()
        was_open = _same(_stage_file(), out)
        if was_open:                                     # never rebuild a file the app has open
            self._build_lbl.text = "Closing the scene to rebuild it..."
            await ctx.new_stage_async()
            await self._frames(3)
            usdio.clear_open(out)                        # closed now; our own marker must not block the build
        self._build_lbl.text = "Building... (reading the drawing, placing the building and equipment)"
        await self._frames(2)
        lines = []
        res = RCP.build(r, validate=True, log=lines.append)
        self._build_lbl.text = "\n".join(lines[:2] + [f"-> {res['usd']}"])
        if open_after or was_open:
            await ctx.open_stage_async(res["usd"])
            await self._frames(3)
            if ctx.get_stage().GetPrimAtPath("/World/Cameras/Overview"):
                from omni.kit.viewport.utility import get_active_viewport
                vp = get_active_viewport()
                if vp:
                    vp.camera_path = "/World/Cameras/Overview"
        self._refresh_scene()

    # ------------------------------------------------------------------ 2 review
    async def _validate(self):
        stage = omni.usd.get_context().get_stage()
        if not stage:
            return
        if not T.tagged(stage):
            n = len(T.auto_tag(stage))
            carb.log_info(f"[frt] auto-tagged {n} prims by name")
        self._count_lbl.text = "Validating..."
        await self._frames(2)
        self.findings = VAL.validate(stage, package_root=os.path.dirname(_stage_file()) or None)
        path = _stage_file()
        if path:
            self.findings.save(os.path.splitext(path)[0] + ".findings.json")
        self._show_findings()

    def _load_findings(self):
        self._refresh_scene()

    def _show_findings(self, extra_title=""):
        self._list.clear()
        c = self.findings.counts()
        self._count_lbl.text = (extra_title + "  " if extra_title else "") + \
            f"{c.get('critical', 0)} critical, {c.get('warning', 0)} warnings, {c.get('info', 0)} info"
        with self._list:
            for f in list(self.findings)[:300]:
                with ui.HStack(height=0, spacing=6):
                    ui.Rectangle(width=6, style={"background_color": SEV.get(f.severity, GREY), "border_radius": 2})
                    ui.Label(f.message, word_wrap=True, style={"color": WHITE})
                    if f.view or f.prims:
                        ui.Button("Go", width=40, height=22, clicked_fn=lambda f=f: self._go(f))

    def _go(self, f):
        if f.prims:
            omni.usd.get_context().get_selection().set_selected_prim_paths(list(f.prims), True)
        if f.view:
            look_at(f.view["eye"], f.view["at"])

    # ------------------------------------------------------------------ 3 stations
    def _show_stations(self):
        self._stations.clear()
        stage = omni.usd.get_context().get_stage()
        rows = []
        if stage:
            for p in T.tagged(stage, role="station"):
                c, s = p.GetAttribute("frt:center").Get(), p.GetAttribute("frt:size").Get()
                if c is not None:
                    rows.append((T.id_of(p) or p.GetName(), c, s))
        with self._stations:
            if not rows:
                ui.Label("No stations in this scene (station rectangles on the drawing become stations).",
                         word_wrap=True, style={"color": GREY})
            for name, c, s in sorted(rows, key=lambda r: (len(r[0]), r[0])):
                with ui.HStack(height=22, spacing=6):
                    ui.Label(name, width=110, style={"color": WHITE})
                    ui.Label(f"{s[0] / 100:.1f} x {s[1] / 100:.1f} m", style={"color": GREY})
                    d = max(s[0], s[1]) * 1.1 + 300
                    ui.Button("Go", width=40, clicked_fn=lambda c=c, d=d: look_at(      # below the lamps (~8 m)
                        (c[0] + d * 0.5, min(d * 0.55, 600), c[2] + d * 0.8), (c[0], 40, c[2])))

    # ------------------------------------------------------------------ 4 revision
    def _new_drawing(self):
        p = self._newdrw.model.get_value_as_string().strip().strip('"')
        _set_setting("new_drawing", p)
        if not os.path.isfile(p):
            raise FileNotFoundError(f"drawing not found: {p!r}")
        return p

    async def _compare(self):
        scene = _stage_file()
        old = os.path.splitext(scene)[0] + ".plant.json" if scene else ""
        if not os.path.isfile(old):
            self._count_lbl.text = "The open scene has no .plant.json (build it from a drawing first)."
            return
        self._count_lbl.text = "Reading the new drawing..."
        await self._frames(2)
        new = DRW.read_plant(self._new_drawing())
        changes = REV.diff(old, new)
        self.findings = REV.to_findings(changes, new, revision=os.path.basename(self._new_drawing()))
        self._show_findings(f"{len(changes)} changes in {os.path.basename(self._new_drawing())}:")

    async def _update_to_new(self):
        path = self._recipe.model.get_value_as_string().strip().strip('"')
        r = RCP.load(path)
        r.setdefault("drawing", {})["file"] = self._new_drawing()
        await self._build_scene(True, recipe=r)

    # ------------------------------------------------------------------ 5 package
    async def _package(self):
        scene = _stage_file()
        out = self._pkg.model.get_value_as_string().strip().strip('"')
        _set_setting("package_dir", out)
        if not scene or not out:
            self._pkg_lbl.text = "Save the scene and give an output folder first."
            return
        if omni.usd.get_context().has_pending_edit():
            self._pkg_lbl.text = "The scene has unsaved changes; save it first (the saved file is what gets packaged)."
            return
        self._pkg_lbl.text = "Packaging... (copying every layer, material and texture)"
        await self._frames(2)
        rep = await asyncio.get_event_loop().run_in_executor(None, PKG.package, scene, out)
        left = PKG.check(rep["scene"])
        self._pkg_lbl.text = (f"{rep['files']} files, {rep['bytes'] / 1e9:.2f} GB -> {rep['scene']}\n"
                              + ("all dependencies resolve" if not left else f"{len(left)} unresolved: {left[:3]}"))
