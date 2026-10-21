"""Window > Drawing Review: read a DXF / PDF with a drawing-standard profile, see what was understood, correct it.

Click an object in the view to select it; the panel shows its type and why. Then:
  Teach shape     - this shape is a <type> everywhere (saved in your team profile; identical shapes follow)
  Only here       - this one object is a <type> in this drawing (saved beside the drawing)
  Ignore          - leave it out of the scene
Calibrate: click two points of a known dimension, type it (40', 16'-10", 12 m), Save.
Confirm saves the reviewed reading as the drawing's answer key; Score compares a fresh reading against it.
Build scene turns the reviewed reading into a USD scene beside the drawing and opens it.
All the work is done by frt_core.drawing (same as the command line).
"""
import asyncio
import glob
import os
import tempfile
import traceback

import carb
import omni.kit.app
import omni.ui as ui
import omni.usd

from frt_core import recipe as RCP
from frt_core.drawing import answers as ANS
from frt_core.drawing import classify as CL
from frt_core.drawing import read as RD
from frt_core.drawing import view as VW

TITLE = "Drawing Review"
SETTING = "/persistent/exts/frt.scene/review_"
GREY, WHITE, ORANGE = 0xFF9A9A9A, 0xFFE8E8E8, 0xFF2090F0
VIEW_W, VIEW_H = 1000, 680


def _profiles():
    names = ["base"]
    for d in (CL.USER_PROFILES, CL.PROFILES):
        for p in sorted(glob.glob(os.path.join(d, "*.toml"))):
            n = os.path.splitext(os.path.basename(p))[0]
            if n not in names:
                names.append(n)
    return names


class DrawingReviewWindow:
    def __init__(self):
        old = ui.Workspace.get_window(TITLE)
        if old is not None:
            old.visible = False
        self.reading = None
        self.sel = None
        self.center, self.ppm = None, 10.0
        self.calib_pts = []
        self.calib_mode = False
        self._n = 0
        self._map = None
        self._tmp = os.path.join(tempfile.gettempdir(), "frt_review")
        os.makedirs(self._tmp, exist_ok=True)
        self.window = ui.Window(TITLE, width=1480, height=900)
        self._build()

    def destroy(self):
        if self.window:
            self.window.destroy()
            self.window = None

    # ------------------------------------------------------------------ layout
    def _build(self):
        s = carb.settings.get_settings()
        self.profiles = _profiles()
        with self.window.frame:
            with ui.VStack(spacing=6):
                with ui.HStack(height=26, spacing=6):
                    ui.Label("Drawing", width=55)
                    self._path = ui.StringField()
                    self._path.model.set_value(s.get(SETTING + "path") or "")
                    ui.Label("Profile", width=45)
                    cur = s.get(SETTING + "profile") or "base"
                    self._prof = ui.ComboBox(self.profiles.index(cur) if cur in self.profiles else 0, *self.profiles,
                                             width=150)
                    ui.Button("Read", width=70, clicked_fn=lambda: self._run(self._read()))
                    ui.Button("Calibrate", width=80, clicked_fn=self._start_calib)
                    ui.Button("Confirm", width=70, clicked_fn=self._confirm,
                              tooltip="save the reviewed reading as this drawing's answer key")
                    ui.Button("Score", width=60, clicked_fn=self._score)
                    ui.Button("Build scene", width=95, clicked_fn=lambda: self._run(self._build_scene()))
                self._status = ui.Label("Pick a DXF or PDF and press Read.", height=20, style={"color": GREY})
                with ui.HStack(spacing=8):
                    with ui.VStack(width=VIEW_W + 4, spacing=4):
                        with ui.HStack(height=24, spacing=4):
                            ui.Button("Fit", width=50, clicked_fn=lambda: self._zoom(None))
                            ui.Button("-", width=30, clicked_fn=lambda: self._zoom(0.6))
                            ui.Button("+", width=30, clicked_fn=lambda: self._zoom(1.6))
                            for lbl, dx, dy in (("<", -0.35, 0), (">", 0.35, 0), ("^", 0, 0.35), ("v", 0, -0.35)):
                                ui.Button(lbl, width=26, clicked_fn=lambda dx=dx, dy=dy: self._pan(dx, dy))
                            self._hint = ui.Label("click: select   double-click: zoom in   wheel: zoom",
                                                  style={"color": GREY})
                        with ui.ZStack(width=VIEW_W, height=VIEW_H):
                            ui.Rectangle(style={"background_color": 0xFFFFFFFF})
                            self._img = ui.Image("", width=VIEW_W, height=VIEW_H, fill_policy=ui.FillPolicy.STRETCH,
                                                 mouse_pressed_fn=self._click, mouse_double_clicked_fn=self._dclick,
                                                 mouse_wheel_fn=self._wheel, mouse_moved_fn=self._moved)
                        self._calib_row = ui.HStack(height=26, spacing=6, visible=False)
                        with self._calib_row:
                            self._calib_lbl = ui.Label("Calibrate: click the two ends of a known dimension", width=360)
                            self._calib_dist = ui.StringField(width=120)
                            self._calib_dist.model.set_value("40'")
                            ui.Button("Save scale", width=90, clicked_fn=lambda: self._run(self._save_calib()))
                            ui.Button("Cancel", width=60, clicked_fn=self._cancel_calib)
                    with ui.ScrollingFrame():
                        with ui.VStack(spacing=6, height=0):
                            with ui.CollapsableFrame("Selected", collapsed=False):
                                with ui.VStack(spacing=4, height=0):
                                    self._sel_lbl = ui.Label("nothing selected", word_wrap=True, style={"color": WHITE})
                                    with ui.HStack(height=24, spacing=4):
                                        ui.Label("is a", width=30)
                                        self._type = ui.ComboBox(0, *CL.KNOWN_TYPES)
                                    with ui.HStack(height=24, spacing=4):
                                        ui.Label("name", width=30)
                                        self._name = ui.StringField(tooltip="optional: room name, model, capacity...")
                                    with ui.HStack(height=26, spacing=4):
                                        ui.Button("Teach shape", clicked_fn=lambda: self._run(self._teach()),
                                                  tooltip="this shape is that type everywhere (team profile)")
                                        ui.Button("Only here", clicked_fn=lambda: self._run(self._only_here()),
                                                  tooltip="this one object, in this drawing only")
                                        ui.Button("Ignore", clicked_fn=lambda: self._run(self._only_here("none")))
                            with ui.CollapsableFrame("Found", collapsed=False):
                                self._found = ui.Label("", word_wrap=True, style={"color": GREY})
                            with ui.CollapsableFrame("Stations", collapsed=False):
                                self._stations = ui.VStack(spacing=2, height=0)
                            with ui.CollapsableFrame("Unknown (largest first)", collapsed=False):
                                self._unknown = ui.VStack(spacing=2, height=0)

    # ------------------------------------------------------------------ helpers
    def _run(self, coro):
        asyncio.ensure_future(self._guard(coro))

    async def _guard(self, coro):
        try:
            await coro
        except Exception as e:   # noqa: BLE001
            carb.log_error("[frt review] " + traceback.format_exc())
            self._status.text = f"Error: {e}"

    async def _frames(self, n=2):
        for _ in range(n):
            await omni.kit.app.get_app().next_update_async()

    def _drawing(self):
        return self._path.model.get_value_as_string().strip().strip('"')

    def _profile_name(self):
        return self.profiles[self._prof.model.get_item_value_model().as_int]

    # ------------------------------------------------------------------ reading
    async def _read(self):
        path = self._drawing()
        if not os.path.isfile(path):
            self._status.text = f"No such file: {path}"
            return
        s = carb.settings.get_settings()
        s.set(SETTING + "path", path)
        s.set(SETTING + "profile", self._profile_name())
        RD.save_settings(path, profile=self._profile_name())
        self._status.text = "Reading..."
        await self._frames(2)
        self.reading = RD.read(path, self._profile_name())
        self.sel = None
        self.center = None
        self._refresh_lists()
        self._render()

    def _refresh_lists(self):
        r = self.reading
        dr = r.drawing
        self._status.text = " ; ".join(dr.notes[:2]) + "   |   profile: " + " > ".join(r.profile["names"])
        self._found.text = ", ".join(f"{n} {t.replace('_', ' ')}" for t, n in r.counts().items())
        self._stations.clear()
        with self._stations:
            ui.Label("Go flies to it; type a name (and tasks, separated by ;) and Save", style={"color": GREY})
            for st in r.stations:
                with ui.HStack(height=24, spacing=4):
                    ui.Button("Go", width=34, clicked_fn=lambda st=st: self._goto(((st.box[0] + st.box[2]) / 2,
                                                                                   (st.box[1] + st.box[3]) / 2), 40))
                    nm = ui.StringField(width=110)
                    nm.model.set_value(st.name)
                    tk = ui.StringField(tooltip="tasks, separated by ;")
                    tk.model.set_value("; ".join(st.tasks))
                    ui.Button("Save", width=44, clicked_fn=lambda st=st, nm=nm, tk=tk: self._run(self._name_station(st, nm, tk)))
        self._unknown.clear()
        unk = sorted((o for o, c in r.items if c.type == "unknown" and max(o.size) >= 40),
                     key=lambda o: -o.size[0] * o.size[1])
        with self._unknown:
            ui.Label(f"{len(unk)} over 40 cm", style={"color": GREY})
            for o in unk[:80]:
                ui.Button(f"#{o.id}  {o.layer[:18]}  {max(o.size) / 100:.1f} x {min(o.size) / 100:.1f} m", height=20,
                          clicked_fn=lambda o=o: self._select(o, center=True))

    # ------------------------------------------------------------------ view
    def _extent(self):
        """Extent of the objects found (the drawing's own frame and title block included), fast on big sheets."""
        objs = [o for o, c in self.reading.items if c.type not in ("ignore",)] or self.reading.objects
        return (min(o.bbox[0] for o in objs), min(o.bbox[1] for o in objs), max(o.bbox[2] for o in objs),
                max(o.bbox[3] for o in objs))

    def _render(self):
        if self.reading is None:
            return
        if self.center is None:
            x0, y0, x1, y1 = self._extent()
            self.center = ((x0 + x1) / 2, (y0 + y1) / 2)
            self.ppm = min(VIEW_W / ((x1 - x0) / 100), VIEW_H / ((y1 - y0) / 100)) * 0.98
        k = self.ppm / 100.0
        hw, hh = VIEW_W / 2 / k, VIEW_H / 2 / k
        crop = (self.center[0] - hw, self.center[1] - hh, self.center[0] + hw, self.center[1] + hh)
        self._n += 1
        out = os.path.join(self._tmp, f"view_{self._n % 20}.png")
        self._map = VW.render(self.reading, out, crop, self.ppm, selected=self.sel, legend=False)
        self._img.source_url = out

    def _to_drawing(self, x, y):
        if not self._map:
            return None
        u, v = x - self._img.screen_position_x, y - self._img.screen_position_y
        k = self._map["k"] * (VIEW_W / self._map["size"][0])
        x0, _, _, y1 = self._map["crop"]
        return x0 + u / k, y1 - v / k

    def _zoom(self, f, at=None):
        if f is None:
            self.center = None
        else:
            if at is not None:
                self.center = at
            self.ppm = max(0.5, min(400.0, self.ppm * f))
        self._render()

    def _pan(self, dx, dy):
        if self.center is None:
            return
        k = self.ppm / 100.0
        self.center = (self.center[0] + dx * VIEW_W / k, self.center[1] + dy * VIEW_H / k)
        self._render()

    def _goto(self, p, ppm=None):
        self.center = p
        if ppm:
            self.ppm = ppm
        self._render()

    def _moved(self, x, y, mod, pressed):
        self._mouse = (x, y)

    def _wheel(self, dx, dy, mod):
        """Wheel: zoom about the point under the mouse (omni.ui passes scroll amounts, not a position)."""
        p = self._to_drawing(*self._mouse) if getattr(self, "_mouse", None) else None
        if p and self.center:
            f = 1.25 if dy > 0 else 0.8
            # keep the point under the mouse where it is: move the centre toward it by the zoom ratio
            self.center = (p[0] + (self.center[0] - p[0]) / f, p[1] + (self.center[1] - p[1]) / f)
            self._zoom(f)
        else:
            self._zoom(1.25 if dy > 0 else 0.8)

    def _dclick(self, x, y, b, m):
        p = self._to_drawing(x, y)
        if p:
            self._zoom(2.0, p)

    def _click(self, x, y, b, m):
        p = self._to_drawing(x, y)
        if p is None or self.reading is None or b != 0:
            return
        if self.calib_mode:
            self.calib_pts.append(p)
            self._calib_lbl.text = f"Calibrate: {len(self.calib_pts)} of 2 points" + (
                f" - {((self.calib_pts[0][0] - p[0]) ** 2 + (self.calib_pts[0][1] - p[1]) ** 2) ** 0.5 / 100:.2f} m at "
                f"the current scale; type the real length" if len(self.calib_pts) == 2 else "")
            if len(self.calib_pts) > 2:
                self.calib_pts = [p]
            return
        o = RD.pick(self.reading.objects, p)
        self._select(o)

    def _select(self, o, center=False):
        self.sel = o
        if o is None:
            self._sel_lbl.text = "nothing selected"
        else:
            c = self.reading.classes[self.reading.objects.index(o)]
            self._sel_lbl.text = (f"#{o.id}  {c.type.replace('_', ' ')}  ({c.confidence:.0%})\n{c.reason}\n"
                                  f"layer {o.layer}, {max(o.size) / 100:.2f} x {min(o.size) / 100:.2f} m, {o.n} strokes"
                                  + (f"\nlabels: {'; '.join(o.texts + o.near_texts)[:120]}" if o.texts or o.near_texts else "")
                                  + (f"\n{c.params}" if c.params else ""))
            if c.type in CL.KNOWN_TYPES:
                self._type.model.get_item_value_model().set_value(CL.KNOWN_TYPES.index(c.type))
            if center:
                self.center = o.center
                self.ppm = max(self.ppm, 30.0)
        self._render()

    # ------------------------------------------------------------------ corrections
    def _chosen(self):
        t = CL.KNOWN_TYPES[self._type.model.get_item_value_model().as_int]
        name = self._name.model.get_value_as_string().strip()
        return t, ({"name": name} if name else None)

    async def _after_change(self, msg):
        self.reading.reclassify()
        self._refresh_lists()
        self._status.text = msg
        if self.sel is not None:
            self._select(self.sel)
        else:
            self._render()

    async def _teach(self):
        if self.sel is None:
            return
        t, params = self._chosen()
        o = self.sel
        ex = f"{os.path.basename(self._drawing())} #{o.id} at ({o.center[0] / 100:.1f}, {o.center[1] / 100:.1f}) m"
        entry = CL.teach_symbol(self.reading.profile, o, t, ex, params)
        same = sum(1 for x in self.reading.objects if x is not o and CL._symbol_match(x, entry) is not None)
        await self._after_change(f"Taught: #{o.id} is a {t} ({os.path.basename(self.reading.profile['path'])}); "
                                 f"{same} more like it in this drawing")

    async def _only_here(self, t=None):
        if self.sel is None:
            return
        o = self.sel
        tt, params = self._chosen()
        t = t or tt
        RD.add_override(self._drawing(), o.center, t, params, [max(o.size), min(o.size)])
        await self._after_change(f"#{o.id} is {'ignored' if t == 'none' else 'a ' + t} in this drawing")

    async def _name_station(self, st, nm, tk):
        name = nm.model.get_value_as_string().strip()
        tasks = [t.strip() for t in tk.model.get_value_as_string().split(";") if t.strip()]
        RD.name_station(self._drawing(), st, name or st.name, tasks)
        await self._after_change(f"{st.name} is now {name or st.name}" + (f" ({len(tasks)} tasks)" if tasks else ""))

    # ------------------------------------------------------------------ calibration / key / build
    def _start_calib(self):
        if self.reading is None or not self._drawing().lower().endswith(".pdf"):
            self._status.text = "Calibration is for PDFs (DXF files carry their units)."
            return
        self.calib_mode, self.calib_pts = True, []
        self._calib_row.visible = True

    def _cancel_calib(self):
        self.calib_mode, self.calib_pts = False, []
        self._calib_row.visible = False

    async def _save_calib(self):
        if len(self.calib_pts) != 2:
            self._calib_lbl.text = "Click exactly two points first"
            return
        dr = self.reading.drawing
        k, hpage = dr.cm_per_pt, dr.page_size[1]
        to_pt = lambda p: [round(p[0] / k, 2), round(hpage - p[1] / k, 2)]
        cal = {"a": to_pt(self.calib_pts[0]), "b": to_pt(self.calib_pts[1]),
               "distance": self._calib_dist.model.get_value_as_string().strip()}
        RD.save_settings(self._drawing(), calibration=cal)
        self._cancel_calib()
        await self._read()

    def _confirm(self):
        if self.reading is not None:
            self._status.text = "Answer key saved: " + ANS.save_key(self.reading)

    def _score(self):
        if self.reading is None:
            return
        try:
            sc = ANS.score(RD.read(self._drawing()))
        except FileNotFoundError:
            self._status.text = "No answer key yet: review the drawing and press Confirm first."
            return
        bad = [f"{t}: -{v['missed']}/+{v['extra']}" for t, v in sc["types"].items() if v["missed"] or v["extra"] or v["wrong"]]
        self._status.text = f"Score F1 {sc['f1']:.3f} over {sc['items']} items; stations {sc['stations']}  " + ", ".join(bad[:8])

    async def _build_scene(self):
        path = self._drawing()
        if not os.path.isfile(path):
            return
        out = os.path.splitext(path)[0] + "_scene.usd"
        r = {"scene": {"name": os.path.basename(path), "output": out}, "drawing": {"file": path},
             "building": {"generator": "warehouse_a", "height": "H10m", "wall_style": "mixed"},
             "equipment": {"from_drawing": True}, "look": {"sky": True}, "_path": path}
        ctx = omni.usd.get_context()
        st = ctx.get_stage()
        if st and os.path.normcase(st.GetRootLayer().realPath or "") == os.path.normcase(out):
            await ctx.new_stage_async()
            await self._frames(3)
            from frt_core import usdio
            usdio.clear_open(out)
        self._status.text = "Building the scene..."
        await self._frames(2)
        lines = []
        res = RCP.build(r, validate=True, log=lines.append)
        await ctx.open_stage_async(res["usd"])
        self._status.text = f"Built and opened {res['usd']}  |  {res.get('summary', '')}"
