"""Window > Task Builder: author a worker's task as steps on the scene's assets, solve it, review it per step.

The window only edits a task file (TOML, the same file an AI agent or a script would write) and runs
tools/task_run.py on it in the background; the result scene is opened when it is done. Nothing here animates.

  Task file   the steps (and which scene file, which worker, where they start)
  Scene       the assets and their tags (bins, tray slots, holes, studs, tools, doors) - read from the scene file
  Steps       verb + the assets it uses, picked from menus or from the viewport selection
  Solve       compile -> solve -> check -> RULA -> bake; per step: time, checks, RULA max; click a step to go there
              (preview: key poses, unchanged steps reused - seconds; Final quality: every frame - minutes)
"""
import asyncio
import json
import os
import shutil
import subprocess
import sys
import time

import carb
import omni.kit.app
import omni.timeline
import omni.ui as ui
import omni.usd

try:
    import tomllib
except ImportError:                                                  # pragma: no cover
    import tomli as tomllib

TITLE = "Task Builder"
REPO = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
PRACTICE = os.path.join(REPO, "docs", "people", "practice")
SETTING = "/persistent/exts/frt.scene/task/"
GREY, WHITE, DIM = 0xFF9A9A9A, 0xFFE8E8E8, 0xFF7A7A7A
GREEN, YELLOW, RED = 0xFF50AA2E, 0xFF1EC8F0, 0xFF2828D7

# verb -> [(field, kind of choice, help)]; kinds are filled from the scene (see Choices)
VERBS = {
    "transfer": ([("what", "small_kind", "what to take"), ("count", "int", "how many"),
                  ("from", "bin", "the bin it is in"), ("to", "slots", "where they go (tray compartment...)")],
                 "get small parts from a bin and drop them in a place"),
    "carry": ([("what", "movable", "the part (both hands)"), ("to", "place", "where it is put down")],
              "pick up a part with both hands, walk if needed, put it down"),
    "hand_start": ([("what", "small_kind", "nut or screw"), ("on", "work_points", "studs / holes"),
                    ("brace", "place", "where the other hand rests (optional)"), ("posture", "posture", "auto: kneels for low work")],
                   "take a nut / screw and thread it on by hand"),
    "fasten": ([("tool", "tool", "the tool (where it lies)"), ("on", "work_points", "the started nuts / screws"),
                ("brace", "place", "where the other hand rests (optional)"), ("posture", "posture", "auto: kneels for low work")],
               "drive / tighten at each point (gets the tool and puts it back, unless it is already in hand)"),
    "get_tool": ([("tool", "tool", "the tool to keep in hand")],
                 "pick a tool up and keep it for the next steps (walk / kneel with it)"),
    "put_back": ([("tool", "tool", "the tool in hand")], "put the tool in hand back where it was"),
    "connect": ([("on", "work_points", "the sockets"), ("posture", "posture", "auto: kneels for low work")], "plug connector halves into the sockets"),
    "cable_tie": ([("on", "work_points", "the tie points on a harness"), ("posture", "posture", "auto: kneels for low work")], "tie the harness at each point"),
    "push_door": ([("door", "door", "the cabinet")], "push the open door so it swings shut"),
    "latch": ([("door", "door", "the cabinet")], "press the door home and turn the latch"),
    "inspect": ([("on", "any_point", "what to look at"), ("seconds", "float", "how long")],
                "look at something and point at it"),
    "wait": ([("seconds", "float", "how long")], "stand and wait"),
}
OPTIONAL = {"brace"}


def _setting(key, default=""):
    v = carb.settings.get_settings().get(SETTING + key)
    return v if v else default


def _set_setting(key, value):
    carb.settings.get_settings().set(SETTING + key, value)


def _python():
    """A Python with numpy + pxr for the solver: the setting, else the one on PATH, else Kit's own."""
    p = _setting("python")
    if p and os.path.isfile(p):
        return p
    w = shutil.which("python")
    if w:
        return w
    kp = os.path.join(os.path.dirname(sys.executable), "python", "python.exe")
    return kp if os.path.isfile(kp) else sys.executable


# ---------------------------------------------------------------------------------------------- task file io
def _toml_value(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_toml_value(x) for x in v) + "]"
    return json.dumps(str(v))


def write_task(path, head, steps):
    lines = ["# Task file written by Window > Task Builder (you can edit it by hand too; the window re-reads it).",
             "[task]"]
    lines += [f"{k} = {_toml_value(v)}" for k, v in head.items()]
    for s in steps:
        lines += ["", "[[step]]"]
        lines += [f"{k} = {_toml_value(v)}" for k, v in s.items() if v not in ("", None)]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


class Choices:
    """What the scene offers each kind of field."""

    def __init__(self, assets):
        self.assets = assets
        A = list(assets.values())
        self.c = {
            "small_kind": list(dict.fromkeys(a["holds"] for a in A if a.get("kind") == "bin" and a.get("holds"))),
            "bin": [a["id"] for a in A if a.get("kind") == "bin"],
            "movable": [a["id"] for a in A if a.get("kind") in ("tray", "plate")],
            "tool": [a["id"] for a in A if a.get("kind") == "tool"],
            "door": [a["id"] for a in A if a.get("kind") == "cabinet"],
            "slots": [], "place": [], "work_points": [], "any_point": [],
            "posture": ["auto", "stand", "kneel"],
        }
        for a in A:
            for name, pts in a.get("points", {}).items():
                if name.endswith(("_axis", "_dir")):          # directions of the other tags, not places
                    continue
                ref = f"{a['id']}.{name}"
                many = isinstance(pts[0], (list, tuple))
                self.c["any_point"].append(ref)
                if many and a.get("kind") in ("tray",):
                    self.c["slots"].append(ref)
                elif many:
                    self.c["work_points"].append(ref)
                elif name != "grab":
                    self.c["place"].append(ref)

    def of(self, kind):
        return self.c.get(kind, [])


# ---------------------------------------------------------------------------------------------- the window
class TaskWindow:
    def __init__(self):
        old = ui.Workspace.get_window(TITLE)
        if old is not None:
            old.visible = False
        self.path = _setting("file", os.path.join(PRACTICE, "my_task.toml"))
        self.head, self.steps, self.assets, self.spots = {}, [], {}, {}
        self.result = None
        self.sel = None                    # selected step index
        self.proc = None
        self._status_label = None
        self.load(self.path)
        self.window = ui.Window(TITLE, width=520, height=920, position_x=1360, position_y=40)
        self.window.frame.set_build_fn(self._build)

    # ------------------------------------------------------------------ data
    def load(self, path):
        self.path = path
        self.error = ""
        try:
            with open(path, "rb") as f:
                data = tomllib.load(f)
        except Exception as e:
            data = {"task": {}, "step": []}
            self.error = f"could not read {path}: {e}"
        self.head = dict(data.get("task", {}))
        self.steps = [dict(s) for s in data.get("step", [])]
        scene = self.head.get("scene", "")
        sp = scene if os.path.isabs(scene) else os.path.join(os.path.dirname(path), scene)
        self.assets, self.spots = {}, {}
        try:
            with open(sp, "rb") as f:
                sc = tomllib.load(f)
            self.assets = {a["id"]: a for a in sc.get("asset", [])}
            self.spots = {s["id"]: s for s in sc.get("spot", [])}
        except Exception as e:
            self.error = f"scene file: {e}"
        self.choices = Choices(self.assets)
        rp = self._out_dir()
        res = os.path.join(rp, "result.json")
        self.result = json.load(open(res)) if os.path.isfile(res) else None
        _set_setting("file", path)

    def save(self):
        write_task(self.path, self.head, self.steps)
        self._say(f"saved {os.path.basename(self.path)}")

    def _out_dir(self):
        name = os.path.splitext(os.path.basename(self.path))[0]
        return os.path.join(REPO, "out", "people", "task", name)

    # ------------------------------------------------------------------ UI
    def _build(self):
        with ui.ScrollingFrame():
            with ui.VStack(spacing=6, height=0):
                ui.Label("Task Builder", style={"font_size": 22}, height=28)
                ui.Label("Steps use the scene's assets; the worker walks between them by itself.", height=16,
                         style={"color": DIM})
                self._file_row()
                if self.error:
                    ui.Label(self.error, word_wrap=True, style={"color": RED}, height=0)
                with ui.CollapsableFrame("Scene assets and their tags", collapsed=True, height=0):
                    self._assets_list()
                with ui.CollapsableFrame(f"Steps ({len(self.steps)})", height=0):
                    self._steps_list()
                with ui.CollapsableFrame("Add / edit a step", height=0):
                    self._editor()
                with ui.CollapsableFrame("Solve and review", height=0):
                    self._solve_panel()

    def _file_row(self):
        with ui.HStack(height=24, spacing=4):
            ui.Label("Task file", width=70)
            self.f_path = ui.StringField()
            self.f_path.model.set_value(self.path)
            ui.Button("Open", width=60, clicked_fn=lambda: self._reload(self.f_path.model.get_value_as_string()))
            ui.Button("Save", width=60, clicked_fn=self.save)
        with ui.HStack(height=24, spacing=4):
            ui.Label("Worker", width=70)
            self.f_worker = ui.StringField()
            self.f_worker.model.set_value(self.head.get("worker", "W50M"))
            self.f_worker.model.add_end_edit_fn(lambda m: self.head.__setitem__("worker", m.get_value_as_string()))
        with ui.HStack(height=24, spacing=4):
            ui.Label("Scene", width=70)
            ui.Label(self.head.get("scene", "-"), style={"color": GREY})
            ui.Button("Show scene", width=100, clicked_fn=lambda: self.solve(scene_only=True))

    def _reload(self, path):
        self.load(path)
        self.window.frame.rebuild()

    def _assets_list(self):
        with ui.VStack(spacing=2, height=0):
            for a in self.assets.values():
                tags = ", ".join(a.get("points", {}).keys())
                extra = f"holds {a['holds']}" if a.get("holds") else (a.get("tool", ""))
                with ui.HStack(height=20):
                    ui.Button(a["id"], width=120, clicked_fn=lambda a=a: self.select_asset(a["id"]))
                    ui.Label(f"{a['kind']:8s} {extra}", width=130, style={"color": GREY})
                    ui.Label(tags or "-", style={"color": DIM}, elided_text=True)

    def _summary(self, s):
        v = s.get("do", "?")
        keys = [k for k, _, _ in VERBS.get(v, ([], ""))[0]]
        return (v + "  " + "  ".join(f"{s[k]}" for k in keys if k in s and k not in ("brace", "posture"))
                + (f"  ({s['posture']})" if s.get("posture") else ""))

    def _step_result(self, i):
        if not self.result or i >= len(self.result.get("steps", [])):
            return None
        return self.result["steps"][i]

    def _steps_list(self):
        with ui.VStack(spacing=2, height=0):
            if not self.steps:
                ui.Label("No steps yet - add the first one below.", style={"color": DIM}, height=20)
            for i, s in enumerate(self.steps):
                r = self._step_result(i)
                col = GREY
                info = "not solved"
                if r is not None:
                    bad = sum(r.get("checks", {}).values())
                    rm = r.get("rula_max")
                    col = RED if bad > 20 or (rm or 0) >= 7 else YELLOW if bad or (rm or 0) >= 5 else GREEN
                    info = f"{r['seconds']} s   RULA max {rm}   " + (", ".join(f"{k} {v}" for k, v in r["checks"].items())
                                                                     or "clean")
                with ui.HStack(height=22, spacing=3):
                    ui.Rectangle(width=6, style={"background_color": col})
                    ui.Button(f"{i + 1:2d}. {self._summary(s)}", alignment=ui.Alignment.LEFT_CENTER,
                              clicked_fn=lambda i=i: self.pick_step(i),
                              style={"background_color": 0xFF4A4038} if i == self.sel else {})
                    ui.Button("^", width=22, clicked_fn=lambda i=i: self.move(i, -1))
                    ui.Button("v", width=22, clicked_fn=lambda i=i: self.move(i, 1))
                    ui.Button("x", width=22, clicked_fn=lambda i=i: self.delete(i))
                ui.Label("      " + info, height=14, style={"color": col if r else DIM, "font_size": 12})

    def _editor(self):
        self.ed = {}
        cur = self.steps[self.sel] if self.sel is not None and self.sel < len(self.steps) else {}
        verbs = list(VERBS)
        v0 = verbs.index(cur["do"]) if cur.get("do") in verbs else 0
        with ui.VStack(spacing=4, height=0):
            with ui.HStack(height=24):
                ui.Label("Verb", width=80)
                self.ed_verb = ui.ComboBox(v0, *verbs)
                self.ed_verb.model.add_item_changed_fn(lambda *a: self._verb_changed())
            self.ed_help = ui.Label(VERBS[verbs[v0]][1], style={"color": DIM}, height=16)
            self.ed_fields = ui.Frame(height=0)
            self.ed_fields.set_build_fn(lambda: self._fields(cur))
            with ui.HStack(height=26, spacing=4):
                ui.Button("Add as new step", clicked_fn=lambda: self.add_step(None))
                ui.Button("Update selected", clicked_fn=lambda: self.add_step(self.sel))
                ui.Button("Use viewport selection", clicked_fn=self.use_selection)

    def _verb(self):
        return list(VERBS)[self.ed_verb.model.get_item_value_model().as_int]

    def _verb_changed(self):
        self.ed_help.text = VERBS[self._verb()][1]
        cur = self.steps[self.sel] if self.sel is not None and self.sel < len(self.steps) else {}
        self.ed_fields.set_build_fn(lambda: self._fields(cur if cur.get("do") == self._verb() else {}))
        self.ed_fields.rebuild()

    def _fields(self, cur):
        self.ed = {}
        with ui.VStack(spacing=3, height=0):
            for name, kind, help_ in VERBS[self._verb()][0]:
                with ui.HStack(height=22):
                    ui.Label(name, width=80, tooltip=help_)
                    opts = []
                    if kind == "int":
                        w = ui.IntField()
                        w.model.set_value(int(cur.get(name, 1)))
                    elif kind == "float":
                        w = ui.FloatField()
                        w.model.set_value(float(cur.get(name, 2.0)))
                    else:
                        opts = self.choices.of(kind)
                        if name in OPTIONAL:
                            opts = ["(none)"] + opts
                        if not opts:
                            opts = ["(nothing in the scene)"]
                        i0 = opts.index(cur[name]) if cur.get(name) in opts else 0
                        w = ui.ComboBox(i0, *opts)
                    ui.Label(help_, width=190, style={"color": DIM, "font_size": 12})
                self.ed[name] = (kind, w, opts)

    def _read_editor(self):
        s = {"do": self._verb()}
        for name, (kind, w, opts) in self.ed.items():
            if kind == "int":
                s[name] = w.model.get_value_as_int()
            elif kind == "float":
                s[name] = round(w.model.get_value_as_float(), 2)
            else:
                v = opts[w.model.get_item_value_model().as_int]
                if not v.startswith("(") and not (kind == "posture" and v == "auto"):
                    s[name] = v
        return s

    def check_step(self, s):
        """Quick sense checks before a step goes in the file (the solver checks the rest)."""
        A = self.assets
        if s["do"] == "transfer" and s.get("from") in A and A[s["from"]].get("holds") != s.get("what"):
            return f"{s['from']} holds {A[s['from']].get('holds')}s, not {s.get('what')}s"
        if s["do"] == "transfer" and s.get("to"):
            aid, _, name = s["to"].partition(".")
            n = len(A.get(aid, {}).get("points", {}).get(name, []))
            if s.get("count", 1) > n:
                return f"{s['to']} has {n} slots, you are moving {s.get('count')}"
            kinds = {a.get("holds") for a in A.values() if a.get("holds")}
            if name.rstrip("s") in kinds and name.rstrip("s") != s.get("what"):
                return f"{s['to']} is for {name}, not {s.get('what')}s"
        for k, kind, _ in VERBS[s["do"]][0]:
            if k not in s and k not in OPTIONAL and kind not in ("int", "float", "posture"):
                return f"pick a value for '{k}'"
        return ""

    def check_order(self, steps):
        """Tool in hand: get_tool / put_back must pair up, and nothing two-handed in between."""
        held = None
        for i, s in enumerate(steps):
            v = s.get("do")
            if v == "get_tool":
                if held:
                    return f"step {i + 1}: already holding {held} (put_back first)"
                held = s.get("tool")
            elif v == "put_back":
                if held != s.get("tool"):
                    return f"step {i + 1}: {s.get('tool')} is not in hand (get_tool first)"
                held = None
            elif held and v in ("carry", "latch", "cable_tie"):
                return f"step {i + 1}: {v} needs both hands, the worker holds {held} (put_back first)"
            elif held and v == "fasten" and s.get("tool") != held:
                return f"step {i + 1}: the worker holds {held}, not {s.get('tool')}"
        return ""

    def order_warning(self, steps):
        """Allowed but probably not meant: starting nuts / screws that are still in a bin away from the work."""
        moved = set()
        for i, s in enumerate(steps):
            if s.get("do") == "transfer":
                moved.add(s.get("what"))
            elif s.get("do") == "hand_start" and not s.get("from") and s.get("what") not in moved:
                bins = [a["id"] for a in self.assets.values() if a.get("kind") == "bin" and a.get("holds") == s.get("what")]
                where = f" ({', '.join(bins)})" if bins else ""
                return (f"step {i + 1}: the {s.get('what')}s are still in their bin{where}; the worker will reach "
                        f"for them from the work. Transfer them to the tray first?")
        return ""

    def add_step(self, at):
        s = self._read_editor()
        bad = self.check_step(s)
        if not bad:
            trial = list(self.steps)
            if at is None:
                trial.insert(self.sel + 1 if self.sel is not None else len(trial), s)
            else:
                trial[at] = s
            bad = self.check_order(trial)
        if bad:
            self._say("not added: " + bad)
            return
        if at is None:
            self.steps.insert(self.sel + 1 if self.sel is not None else len(self.steps), s)
            self.sel = (self.sel + 1) if self.sel is not None else len(self.steps) - 1
        else:
            self.steps[at] = s
        self.save()
        self.window.frame.rebuild()
        warn = self.order_warning(self.steps)
        if warn:
            self._say("added, but: " + warn)

    def move(self, i, d):
        j = i + d
        if 0 <= j < len(self.steps):
            self.steps[i], self.steps[j] = self.steps[j], self.steps[i]
            self.sel = j
            self.save()
            self.window.frame.rebuild()

    def delete(self, i):
        del self.steps[i]
        self.sel = None
        self.save()
        self.window.frame.rebuild()

    def pick_step(self, i):
        self.sel = i
        r = self._step_result(i)
        if r is not None:
            tl = omni.timeline.get_timeline_interface()
            tl.set_current_time(r["start"] / tl.get_time_codes_per_seconds())
        self.window.frame.rebuild()

    def use_selection(self):
        """Fill the first field that can take the asset selected in the viewport."""
        paths = omni.usd.get_context().get_selection().get_selected_prim_paths()
        if not paths:
            self._say("select something in the viewport first")
            return
        aid = self._asset_of(paths[0])
        if aid is None:
            self._say(f"{paths[0]} is not one of the task's assets")
            return
        for name, (kind, w, opts) in self.ed.items():
            hit = [k for k, o in enumerate(opts) if o == aid or o.startswith(aid + ".")]
            if hit:
                w.model.get_item_value_model().set_value(hit[0])
                self._say(f"{name} = {opts[hit[0]]}")
                return
        self._say(f"{aid} does not fit any field of '{self._verb()}'")

    def _asset_of(self, path):
        parts = path.split("/")
        for key in ("Scene", "Parts"):
            if key in parts:
                k = parts.index(key)
                if k + 1 < len(parts):
                    name = parts[k + 1]
                    if name in self.assets:
                        return name
                    for aid in self.assets:                  # bin contents, the cabinet's door / latch
                        if name.endswith("_" + aid) or name.startswith(aid + "_") or f"_{aid}_" in name:
                            return aid
        return None

    def select_asset(self, aid):
        st = omni.usd.get_context().get_stage()
        if not st:
            return
        for p in (f"/World/Asset/Scene/{aid}", f"/World/Asset/Parts/{aid}", f"/World/Asset/Parts/{aid}_door"):
            if st.GetPrimAtPath(p):
                omni.usd.get_context().get_selection().set_selected_prim_paths([p], True)
                try:
                    import omni.kit.commands
                    omni.kit.commands.execute("FramePrimsCommand", prim_to_move="/OmniverseKit_Persp",
                                              prims_to_frame=[p])
                except Exception:
                    pass
                return

    # ------------------------------------------------------------------ solve
    def _solve_panel(self):
        with ui.VStack(spacing=4, height=0):
            with ui.HStack(height=28, spacing=4):
                ui.Button("Solve (preview)", clicked_fn=self.solve, style={"background_color": 0xFF3D6B2A},
                          tooltip="Key poses solved, blended between them; unchanged steps reused. Seconds to a minute.")
                ui.Button("Final quality", width=100, clicked_fn=lambda: self.solve(final=True),
                          tooltip="Every frame solved (for the video / the report). Several minutes.")
                ui.Button("Play", width=60, clicked_fn=lambda: omni.timeline.get_timeline_interface().play())
                ui.Button("Pause", width=60, clicked_fn=lambda: omni.timeline.get_timeline_interface().pause())
            self.progress = ui.ProgressBar(height=14)
            self._status_label = ui.Label("", word_wrap=True, height=0, style={"color": GREY})
            if self.result:
                c = self.result.get("counts") or {}
                ui.Label(f"Last solve ({self.result.get('quality', 'final')}): {self.result['frames']} frames "
                         f"({self.result['frames'] / 24:.1f} s), "
                         f"{self.result.get('solve_s', '?')} s to solve. Checks: "
                         f"{', '.join(f'{k} {v}' for k, v in c.items()) or 'clean'}", word_wrap=True, height=0)
                ui.Label("Colour per step: green = clean and RULA under 5, yellow = a few check findings or RULA 5-6,"
                         " red = many findings or RULA 7.", word_wrap=True, height=0, style={"color": DIM})

    def _say(self, msg):
        carb.log_info(f"[task builder] {msg}")
        if self._status_label is not None:
            self._status_label.text = msg

    def solve(self, scene_only=False, final=False):
        if self.proc is not None and self.proc.poll() is None:
            self._say("already solving...")
            return
        self.save()
        out = self._out_dir()
        os.makedirs(out, exist_ok=True)
        ctx = omni.usd.get_context()                     # the last result open here would block writing the new one
        cur = os.path.normcase(os.path.abspath(ctx.get_stage_url() or "")) if ctx.get_stage() else ""
        if cur.startswith(os.path.normcase(os.path.abspath(out))):
            ctx.new_stage()
        args = [_python(), "-u", os.path.join(REPO, "tools", "task_run.py"), self.path, f"--out={out}"]
        if scene_only:
            args.append("--scene-only")
        if final:
            args.append("--final")
        st = os.path.join(out, "status.json")
        if os.path.isfile(st):
            os.remove(st)
        log = open(os.path.join(out, "run.log"), "w")
        flags = 0x08000000 if os.name == "nt" else 0                 # no console window
        self.proc = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT, cwd=REPO, creationflags=flags)
        self._say("started: " + " ".join(os.path.basename(a) for a in args[2:]))
        asyncio.ensure_future(self._watch(out, scene_only))

    async def _watch(self, out, scene_only):
        st = os.path.join(out, "status.json")
        t0 = time.time()
        while True:
            for _ in range(20):
                await omni.kit.app.get_app().next_update_async()
            s = None
            if os.path.isfile(st):
                try:
                    s = json.load(open(st))
                except Exception:
                    s = None
            if s:
                self.progress.model.set_value(float(s.get("progress", 0.0)))
                self._say(f"{s['msg']}   ({time.time() - t0:.0f} s)")
                if s.get("done"):
                    break
            if self.proc.poll() is not None and not (s and s.get("done")):
                self._say(f"the solver stopped (exit {self.proc.returncode}) - see {os.path.join(out, 'run.log')}")
                return
        if s.get("error"):
            self._say("task error: " + s["error"])
            return
        usd = s.get("usd") or os.path.join(out, "task.usda")
        cam = _camera_state()
        await omni.usd.get_context().open_stage_async(usd)
        for _ in range(10):
            await omni.kit.app.get_app().next_update_async()
        _restore_camera(cam)
        res = os.path.join(out, "result.json")
        if not scene_only and os.path.isfile(res):
            self.result = json.load(open(res))
        self.window.frame.rebuild()
        self._say(s["msg"])


def _camera_state():
    try:
        from omni.kit.viewport.utility import get_active_viewport
        from omni.kit.viewport.utility.camera_state import ViewportCameraState
        cs = ViewportCameraState("/OmniverseKit_Persp", get_active_viewport())
        return cs.position_world, cs.target_world
    except Exception:
        return None


def _restore_camera(c):
    if not c:
        return
    try:
        from omni.kit.viewport.utility import get_active_viewport
        from omni.kit.viewport.utility.camera_state import ViewportCameraState
        cs = ViewportCameraState("/OmniverseKit_Persp", get_active_viewport())
        cs.set_position_world(c[0], True)
        cs.set_target_world(c[1], True)
    except Exception:
        pass
