"""Task files: a job written as assets + steps (verbs), compiled into one continuous, solved take.

A task file (TOML) has three parts:

  [task]      name, the worker (an frt body id like "W50M" or a character .usd), where they start
  [[spot]]    places to stand: id, at = [x, z], yaw (deg, 0 = facing +Z)
  [[asset]]   the physical things: id, kind, at = [x, y, z] (world cm, Y up), size, the spot to work it from,
              and its tags - named points (grab points, holes, studs, slots...) in the asset's own frame
  [[step]]    what the worker does, in order: do = <verb> plus the assets it uses

Verbs (each one expands into reach / grasp / move / act / release / return phases, see the functions below):

  transfer    get small parts from a bin and drop them somewhere      what, count, from, to
  carry       pick a part up with both hands, walk if needed, put it down   what, to
  hand_start  pick a nut / screw (from where it lies) and thread it by hand   what, on (points), count
  fasten      get a tool, drive / tighten at each point, put the tool back    tool, on (points)
              (if the worker already holds that tool - get_tool - it stays in hand afterwards)
  get_tool    pick a tool up and keep it in hand for the next steps          tool
  put_back    put the held tool back where it was taken from                 tool
  connect     plug connector halves into sockets                             on
  cable_tie   tie a harness at its tie points                                on
  push_door   push an open door so it swings shut                     door
  latch       press the door home and turn its latch                  door
  inspect     look at a point and point at it                         on, seconds

The worker walks to the spot of the step's asset by itself (spots are on the assets), so a step never says
where to stand unless you want to override it (at = "<spot id>").

    task = Task.load("docs/people/practice/battery_box.toml")
    take = task.compile()          # -> frames, steps (id, verb, text, frame range), parts, notes
"""
import math
import os

import numpy as np

try:
    import tomllib
except ImportError:
    import tomli as tomllib

from ..geom import Shapes
from . import handkit as K
from . import motion_carry as C
from .bake import PART_BUILDERS, _rootT
from .gait import walk
from .motion_gen import FPS, Frame, _frame_of, ease, lerp_target, rest_targets
from .motion_more import idle
from .motion_more import inspect as inspect_motion
from .solver import HandTarget
from .tools import TOOLS

Y = np.array([0.0, 1.0, 0.0])
X = np.array([1.0, 0.0, 0.0])
Z = np.array([0.0, 0.0, 1.0])


# ---------------------------------------------------------------------------------------------- small helpers
def T_pos(p, R=None):
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = R
    T[:3, 3] = p
    return T


def Ry(deg):
    a = math.radians(deg)
    return np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])


def rot_axis(axis, deg):
    a = K.unit(axis)
    th = math.radians(deg)
    Kx = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(th) * Kx + (1 - math.cos(th)) * Kx @ Kx


def rot_target(t, axis, deg, pivot):
    R = rot_axis(axis, deg)
    pv = np.asarray(pivot, float)
    return HandTarget(t.side, tuple(R @ (np.asarray(t.point) - pv) + pv), tuple(R @ np.asarray(t.fingers)),
                      tuple(R @ np.asarray(t.palm)), t.grip, t.weight)


def with_grip(t, grip, weight=None):
    return HandTarget(t.side, t.point, t.fingers, t.palm, grip, t.weight if weight is None else weight)


def moved(t, d):
    return HandTarget(t.side, tuple(np.asarray(t.point) + np.asarray(d, float)), t.fingers, t.palm, t.grip, t.weight)


def perp(v, n):
    v = np.asarray(v, float)
    n = K.unit(n)
    return v - n * np.dot(v, n)


def R_up(n):
    """Rotation taking +Y to n (and +X to the nearest direction across it): a part / tool frame on a surface whose
    outward normal is n (n = +Y: identity)."""
    n = K.unit(n)
    ref = X if abs(n[0]) < 0.9 else Z
    e1 = K.unit(perp(ref, n))
    return np.column_stack([e1, n, np.cross(e1, n)])


def target_of(f, side):
    return next((t for t in f.targets if t.side == side), None)


def lerp4(A, B, u):
    from .motion_gen import lerp_T
    return lerp_T(A, B, u)


# ---------------------------------------------------------------------------------------------- part geometry
def _screw(s, rest):
    L = float(rest[0]) if rest else 2.0
    s.cylinder((0, 0, 0), (0, -L, 0), 0.22, "bolt_zinc", sides=10)
    s.cylinder((0, 0, 0), (0, 0.35, 0), 0.5, "bolt_zinc", sides=14)


def _nut(s, rest):
    s.cylinder((0, -0.25, 0), (0, 0.25, 0), 0.6, "bolt_zinc", sides=6)


def _tray(s, rest):
    w, h, d = (float(v) for v in rest[0].split("x"))
    t = 0.5
    look = rest[1] if len(rest) > 1 else "plastic_black"
    s.rbox((0, -h / 2 + t / 2, 0), (w, t, d), 0.2, look, seg=1)
    for sx in (-1, 1):
        s.rbox((sx * (w / 2 - t / 2), 0, 0), (t, h, d), 0.2, look, seg=1)
        s.rbox((0, 0, sx * (d / 2 - t / 2)), (w, h, t), 0.2, look, seg=1)
    s.rbox((0, -h * 0.1, 0), (t, h * 0.8, d), 0.2, look, seg=1)         # divider: two compartments


def _plate(s, rest):
    w, h, d = (float(v) for v in rest[0].split("x"))
    s.rbox((0, 0, 0), (w, h, d), 0.3, rest[1] if len(rest) > 1 else "powder_lightgrey", seg=2)


def _door(s, rest):
    """A cabinet door: origin on the hinge line at the bottom, closed along +X, outer face towards -Z."""
    w, h, t = (float(v) for v in rest[0].split("x"))
    s.rbox((w / 2, h / 2, -t / 2), (w, h, t), 0.4, rest[1] if len(rest) > 1 else "powder_lightgrey", seg=2)
    for y in (h * 0.15, h * 0.85):                                     # hinge knuckles
        s.cylinder((0, y - 4, -t / 2), (0, y + 4, -t / 2), 0.9, "galvanised", sides=12)


def _latch(s, rest):
    """A quarter-turn latch lever: origin at the spindle on the door's outer face, lever along -Y (closed: down)."""
    s.cylinder((0, 0, 0), (0, 0, -1.4), 1.6, "plastic_black", sides=16)
    s.rbox((0, -3.0, -1.8), (1.6, 7.0, 1.4), 0.5, "plastic_black", seg=1)


def _tool(s, rest):
    TOOLS[rest[0]].build(s)


PART_BUILDERS.update({"screw": _screw, "nut": _nut, "tray": _tray, "plate": _plate, "door": _door,
                      "latch": _latch, "tool": _tool})


# ---------------------------------------------------------------------------------------------- static assets
def build_asset(s, a):
    """Static geometry of an asset (world cm): tables, shelves, bins, carts, cabinets. Movable things (parts,
    tools, the tray, the door) are parts and are built by the baker."""
    k, at, size = a["kind"], np.asarray(a["at"], float), a.get("size")
    if k == "table":
        w, h, d = size
        s.rbox((at[0], h - 2.0, at[2]), (w, 4.0, d), 0.6, "wood_top", seg=2)
        for x in (-w / 2 + 4, w / 2 - 4):
            for z in (-d / 2 + 4, d / 2 - 4):
                s.rprism((at[0] + x, 0.0, at[2] + z), (at[0] + x, h - 4.0, at[2] + z), 4.0, 4.0, 0.6, "powder_blue",
                         up=(1, 0, 0))
        s.rprism((at[0] - w / 2 + 4, 14.0, at[2]), (at[0] + w / 2 - 4, 14.0, at[2]), 3.0, 3.0, 0.5, "powder_blue")
    elif k == "shelf":
        w, h, d = size
        for x in (-w / 2 + 2, w / 2 - 2):
            for z in (-d / 2 + 2, d / 2 - 2):
                s.rprism((at[0] + x, 0.0, at[2] + z), (at[0] + x, h + 40.0, at[2] + z), 3.5, 3.5, 0.5, "powder_dark",
                         up=(1, 0, 0))
        for y in (18.0, h - 1.5, h + 38.0):
            s.rbox((at[0], y, at[2]), (w, 3.0, d), 0.4, "galvanised", seg=1)
    elif k == "cart":
        w, h, d = size
        s.rbox((at[0], h - 2.0, at[2]), (w, 3.0, d), 0.5, "powder_dark", seg=1)
        s.rbox((at[0], 22.0, at[2]), (w, 3.0, d), 0.5, "powder_dark", seg=1)
        for x in (-w / 2 + 2, w / 2 - 2):
            for z in (-d / 2 + 2, d / 2 - 2):
                s.rprism((at[0] + x, 8.0, at[2] + z), (at[0] + x, h - 3.0, at[2] + z), 2.5, 2.5, 0.4, "galvanised",
                         up=(1, 0, 0))
                s.cylinder((at[0] + x, 4.0, at[2] + z - 2), (at[0] + x, 4.0, at[2] + z + 2), 4.0, "plastic_black",
                           sides=14)
    elif k == "bin":
        w, h, d = size
        t = 0.6
        y0 = at[1]
        s.rbox((at[0], y0 + t / 2, at[2]), (w, t, d), 0.2, "plastic_blue", seg=1)
        s.rbox((at[0] - w / 2 + t / 2, y0 + h / 2, at[2]), (t, h, d), 0.2, "plastic_blue", seg=1)
        s.rbox((at[0] + w / 2 - t / 2, y0 + h / 2, at[2]), (t, h, d), 0.2, "plastic_blue", seg=1)
        s.rbox((at[0], y0 + h / 2, at[2] + d / 2 - t / 2), (w, h, t), 0.2, "plastic_blue", seg=1)
        s.rbox((at[0], y0 + h * 0.3, at[2] - d / 2 + t / 2), (w, h * 0.6, t), 0.2, "plastic_blue", seg=1)  # low lip
        rng = np.random.default_rng(len(a["id"]))
        for _ in range(int(a.get("fill", 40))):                          # the loose parts in the bin
            p = at + np.array([rng.uniform(-w / 2 + 2, w / 2 - 2), t + rng.uniform(0.3, 1.6),
                               rng.uniform(-d / 2 + 2, d / 2 - 2)])
            if a.get("holds") == "nut":
                s.cylinder(tuple(p - Y * 0.25), tuple(p + Y * 0.25), 0.6, "bolt_zinc", sides=6)
            else:
                ax = K.unit(rng.normal(size=3) * np.array([1, 0.3, 1]))
                s.cylinder(tuple(p), tuple(p + ax * 2.0), 0.22, "bolt_zinc", sides=6, caps=False)
                s.cylinder(tuple(p - ax * 0.3), tuple(p), 0.5, "bolt_zinc", sides=8)
    elif k == "box":                                                     # the work: a battery box with studs
        w, h, d = size
        s.rbox((at[0], at[1] + h / 2, at[2]), (w, h, d), 0.8, "powder_dark", seg=2)
        for p in a.get("points", {}).get("studs", []):
            q = at + np.asarray(p, float)
            s.cylinder(tuple(q - Y * 0.1), tuple(q + Y * 2.4), 0.45, "steel_machined", sides=12)
            s.cylinder(tuple(q - Y * 0.1), tuple(q + Y * 0.4), 1.6, "galvanised", sides=16)   # terminal pad
    elif k in ("fixture", "wheel"):
        _fixture(s, a)
    elif k == "cabinet":                                                 # wall cabinet on a stand, door is a part
        w, h, d = size
        y0 = at[1]
        z0 = at[2]                                                       # front face plane (the door side)
        s.rbox((at[0] + w / 2, y0 + h / 2, z0 + d / 2), (w, h, d), 0.6, "powder_lightgrey", seg=2)
        s.rbox((at[0] + w / 2, y0 + h / 2, z0 + 0.4), (w - 4.0, h - 4.0, 0.8), 0.2, "powder_dark", seg=1)  # opening
        for x in (4.0, w - 4.0):
            s.rprism((at[0] + x, 0.0, z0 + d / 2), (at[0] + x, y0, z0 + d / 2), 4.0, 4.0, 0.5, "powder_dark",
                     up=(1, 0, 0))
        s.rbox((at[0] + w / 2, 1.0, z0 + d / 2), (w + 10.0, 2.0, d + 20.0), 0.4, "powder_dark", seg=1)


def _features(s, a, T):
    """Draw an asset's tagged work features by their names: studs*, holes*, sockets*, harness* (a cable through
    its points), ties* (nothing: they go on the harness). Each tag can have '<name>_axis' (the outward normal)."""
    pts = a.get("points", {})
    for name, P in pts.items():
        if name.endswith(("_axis", "_dir")):
            continue
        P = [P] if not isinstance(P[0], (list, tuple)) else P
        W = [T[:3, :3] @ np.asarray(p, float) + T[:3, 3] for p in P]
        n = K.unit(T[:3, :3] @ np.asarray(pts.get(f"{name}_axis", (0, 1, 0)), float))
        Rn = R_up(n)
        if name.startswith("studs"):
            for q in W:
                s.cylinder(tuple(q - n * 0.1), tuple(q + n * 2.6), 0.55, "steel_machined", sides=12)
                s.cylinder(tuple(q - n * 0.2), tuple(q + n * 0.3), 1.7, "galvanised", sides=16)
        elif name.startswith("holes"):
            for q in W:
                s.cylinder(tuple(q - n * 0.3), tuple(q + n * 0.05), 0.75, "powder_black", sides=12)
        elif name.startswith("sockets"):
            for q in W:
                s.obox(tuple(q - n * 1.2), (3.2, 2.4, 2.2), [Rn[:, 0], n, Rn[:, 2]], "plastic_black")
        elif name.startswith("harness") and len(W) >= 2:
            s.sweep([tuple(q) for q in W], 0.9, "cable_pvc", sides=10)


def _fixture(s, a):
    """A generic piece of the machine: a block (chassis rail, battery tray, platform, control box...) or a wheel,
    optionally on legs / a stand, with its tagged features drawn."""
    at = np.asarray(a["at"], float)
    T = T_pos(at, Ry(float(a.get("yaw", 0.0))))
    look = a.get("look", "powder_dark")
    if a["kind"] == "wheel":
        n = K.unit(T[:3, :3] @ np.asarray(a.get("axis", (0, 0, -1)), float))
        R0 = float(a.get("radius", 30.0))
        s.torus(tuple(at - n * 4.0), tuple(n), R0 - 6.0, 6.5, "rubber_black", seg=36, sides=12)
        s.cylinder(tuple(at - n * 9.0), tuple(at + n * 0.2), R0 - 11.0, "powder_lightgrey", sides=32)
        s.cylinder(tuple(at - n * 30.0), tuple(at - n * 8.0), 9.0, "steel_dark", sides=20)       # axle housing
        if a.get("stand", True):
            s.rbox((at[0], (at[1] - R0 + 2) / 2, at[2] + 26.0 * (1 if n[2] < 0 else -1)), (20.0, at[1] - R0 + 2, 20.0),
                   0.5, "powder_dark", seg=1)
    else:
        w, h, d = a["size"]
        c = T[:3, :3] @ np.array([0.0, h / 2, 0.0]) + at
        s.obox(tuple(c), (w, h, d), [T[:3, 0], T[:3, 1], T[:3, 2]], look)
        if a.get("legs"):
            for sx in (-1, 1):
                for sz in (-1, 1):
                    q = T[:3, :3] @ np.array([sx * (w / 2 - 4), 0.0, sz * (d / 2 - 4)]) + at
                    s.rprism((q[0], 0.0, q[2]), tuple(q), 5.0, 5.0, 0.6, "powder_dark", up=(1, 0, 0))
    _features(s, a, T)


# ---------------------------------------------------------------------------------------------- the compiler
class Task:
    def __init__(self, data, path=None):
        self.data = data
        self.path = path
        self.name = data.get("task", {}).get("name", "task")
        self.spots = {s["id"]: (float(s["at"][0]), float(s["at"][1]), float(s.get("yaw", 0.0)))
                      for s in data.get("spot", [])}
        self.assets = {a["id"]: a for a in data.get("asset", [])}
        self.notes = []

    @classmethod
    def load(cls, path):
        """A task file; `scene = "<file>"` in [task] pulls in the spots and assets of a scene file (relative path)."""
        with open(path, "rb") as f:
            data = tomllib.load(f)
        scene = data.get("task", {}).get("scene")
        if scene:
            sp = scene if os.path.isabs(scene) else os.path.join(os.path.dirname(os.path.abspath(path)), scene)
            with open(sp, "rb") as f:
                sc = tomllib.load(f)
            data["spot"] = sc.get("spot", []) + data.get("spot", [])
            data["asset"] = sc.get("asset", []) + data.get("asset", [])
        return cls(data, path)

    # ------------------------------------------------------------------ worker
    def worker(self):
        from .character import load as load_character
        from .skeleton import Skeleton
        from .workers import WORKERS
        w = self.data["task"].get("worker", "W50M")
        if w.lower().endswith((".usd", ".usda", ".usdc")):
            ch = load_character(w)
            return ch.worker, ch.skeleton, ch
        if w.lower().endswith(".glb"):                                 # a Mixamo character (the website's)
            from .mixamo import load as load_mixamo
            ch = load_mixamo(w)
            return ch.worker, ch.skeleton, ch
        return WORKERS[w], Skeleton.for_worker(WORKERS[w]), None

    # ------------------------------------------------------------------ points
    def asset_T(self, aid):
        """World 4x4 of an asset (movable parts: where they are now)."""
        if aid in self.parts:
            return self.parts[aid]["W"]
        a = self.assets[aid]
        return T_pos(np.asarray(a["at"], float), Ry(float(a.get("yaw", 0.0))))

    def points(self, ref):
        """'asset.name' -> list of world points (a single point is a list of one). 'asset' alone -> its origin."""
        aid, _, name = ref.partition(".")
        T = self.asset_T(aid)
        if not name:
            return [T[:3, 3].copy()]
        pts = self.assets[aid].get("points", {}).get(name)
        if pts is None:
            raise KeyError(f"asset {aid!r} has no point {name!r} (tags: {list(self.assets[aid].get('points', {}))})")
        pts = [pts] if not isinstance(pts[0], (list, tuple)) else pts
        return [T[:3, :3] @ np.asarray(p, float) + T[:3, 3] for p in pts]

    def axis_of(self, ref):
        """Outward normal of a tagged surface: '<asset>.<name>' looks for the tag '<name>_axis' (asset frame);
        without one the work faces up (+Y)."""
        aid, _, name = ref.partition(".")
        a = self.assets.get(aid, {}).get("points", {}).get(f"{name}_axis")
        if a is None:
            return Y.copy()
        return K.unit(self.asset_T(aid)[:3, :3] @ np.asarray(a, float))

    def spot_of(self, step, aid):
        if "at" in step:
            return self.spots[step["at"]]
        if aid in getattr(self, "parts", {}) and self.parts[aid].get("spot"):
            return self.spots[self.parts[aid]["spot"]]          # a part that was moved: worked where it is now
        a = self.assets[aid]
        if "spot" not in a:
            raise KeyError(f"asset {aid!r} has no spot to work it from - give it spot = \"...\" or the step at = ...")
        return self.spots[a["spot"]]

    # ------------------------------------------------------------------ parts (things that move)
    def _init_parts(self):
        self.parts = {}
        for a in self.assets.values():
            k = a["kind"]
            at = np.asarray(a["at"], float)
            if k == "bin" and a.get("holds") in ("screw", "nut"):
                kind = a["holds"]
                n = int(a.get("count", 6))
                w, h, d = a["size"]
                for i in range(n):                                       # the parts that will be picked: on top
                    p = at + np.array([(i % 3 - 1) * 2.2, 1.6 + 0.2 * (i // 3), 2.0 + (i // 3) * 2.2])
                    spec = "screw:2.2" if kind == "screw" else "nut"
                    R = rot_axis(Z, 90.0) if kind == "screw" else np.eye(3)   # screws lie on their side
                    self._add_part(f"{kind}_{a['id']}_{i + 1}", spec, T_pos(p, R), kind=kind, home=a["id"])
            elif k in ("tray", "plate"):
                w, h, d = a["size"]
                spec = f"{k}:{w:g}x{h:g}x{d:g}:{a.get('look', 'plastic_black' if k == 'tray' else 'powder_lightgrey')}"
                self._add_part(a["id"], spec, T_pos(at, Ry(float(a.get("yaw", 0.0)))), kind=k, size=(w, h, d))
            elif k == "tool":
                tool = a["tool"]
                T = np.eye(4)
                T[:3, :3] = K.T_at(np.zeros(3), a.get("x_axis", [1, 0, 0]), a.get("y_axis", [0, 1, 0]))[:3, :3]
                T[:3, 3] = at
                self._add_part(a["id"], f"tool:{tool}", T, kind="tool", tool=tool)
            elif k == "cabinet":
                w, h, d = a["size"]
                door = a.get("door", {})
                dw, dh = float(door.get("width", w)), float(door.get("height", h))
                ang = float(door.get("open_deg", 90.0))
                hinge = np.array([at[0], at[1] + (h - dh) / 2, at[2]])
                Wd = T_pos(hinge, Ry(ang))
                self._add_part(f"{a['id']}_door", f"door:{dw:g}x{dh:g}x2:powder_lightgrey", Wd, kind="door",
                               cabinet=a["id"], hinge=hinge, open=ang, width=dw, height=dh)
                rel = T_pos((dw - 5.0, dh * 0.5, -2.0))
                self._add_part(f"{a['id']}_latch", "latch", Wd @ rel, kind="latch", parent=f"{a['id']}_door",
                               rel=rel, turned=0.0)

    def _add_part(self, pid, spec, W, **kw):
        self.parts[pid] = dict(spec=spec, W=W, **kw)

    def children(self, pid):
        return [q for q, p in self.parts.items() if p.get("parent") == pid]

    # ------------------------------------------------------------------ compile
    def compile(self, log=print, field=True):
        """Every step expanded into frames (the worker walks between spots by itself), one continuous take."""
        self._init_parts()
        self.w, self.sk, self.character = self.worker()
        from .solver import PostureSolver
        self.S = PostureSolver(self.sk, self.w)
        self.field = self.equipment() if field else None    # the body keeps out of the scene's equipment
        self.S.field = self.field
        self.holding = None                      # (tool part id, hand) while a tool stays in hand between steps
        self.rest, _ = rest_targets(self.S)
        t = self.data["task"]
        st = t.get("start", [0, 0, 0])
        self.at = (float(st[0]), float(st[1]), float(st[2]) if len(st) > 2 else 0.0)
        self.frames = []
        self.steps = []
        self.carrying = None
        self.kneeling = None                     # (spot, stance) while the worker is down on one knee
        self._want_kneel = False
        self.rest_stand = self._rest_base
        K.PATH_GUARD = self._path_guard
        try:
            self._compile_steps(log)
        finally:
            K.PATH_GUARD = None
        self.stand_up("end")
        self._add(idle(0.8), self.at, "end:stand")
        self._seams()
        for f in self.frames:                    # worker frame -> world, for the equipment field
            f.world = _rootT(f.root)
        return self

    def _path_guard(self, side, t, w=1.0):
        """A hand on its way between keys keeps clear of its own shoulder: interpolated paths can pass right by it
        (a tray at the side to the work in front, kneeling), which folds the arm shut - real hands swing round."""
        sh = self._shoulders_kneel[side] if self.kneeling else self.sk.world[f"upperarm_{side}"]
        v = np.asarray(t.point, float) - sh
        d = float(np.linalg.norm(v))
        m = 0.7 * (self.sk.length(f"upperarm_{side}", f"lowerarm_{side}") + self.sk.length(f"lowerarm_{side}", f"hand_{side}"))
        if d >= m:
            return t
        # push forward (and a little out to its side) by just enough: a smooth bow in front of the shoulder, where
        # pushing straight away from it would swing round as the path passes close by
        F = K.unit(np.array([0.25 if side == "l" else -0.25, 0.0, 1.0]))
        vf = float(np.dot(v, F))
        k = -vf + math.sqrt(max(0.0, vf * vf - (d * d - m * m)))
        return moved(t, F * k * w)

    def _compile_steps(self, log):
        for k, step in enumerate(self.data.get("step", [])):
            sid = step.get("id", f"{k + 1:02d}_{step['do']}")
            fn = getattr(self, "do_" + step["do"], None)
            if fn is None:
                raise ValueError(f"step {sid}: unknown verb {step['do']!r}")
            a0 = len(self.frames)
            try:
                self._want_kneel = self.wants_kneel(step)
                if self.kneeling and not self._want_kneel:
                    self.stand_up(sid)
                self._run_holding(fn, step, sid)
            except KeyError as e:                                    # a missing field or an unknown name
                msg = e.args[0] if e.args else e
                if isinstance(msg, str) and msg in step.keys() | {"to", "from", "what", "on", "tool", "door"}:
                    msg = f"the step has no '{msg}'"
                raise ValueError(f"step {k + 1} ({step['do']}): {msg}") from None
            except ValueError as e:
                raise ValueError(f"step {k + 1} ({step['do']}): {e}") from None
            self.steps.append({"id": sid, "do": step["do"], "text": step.get("text", ""), "start": a0,
                               "end": len(self.frames) - 1, "phases": self._phases(a0)})
            log(f"  {sid:24s} {step['do']:10s} {a0:5d}-{len(self.frames) - 1:5d}  ({(len(self.frames) - a0) / FPS:4.1f} s)")

    def equipment(self):
        """The static assets as a distance field (collide.SceneField) around the spots."""
        try:
            from pxr import Usd

            from ..materials import MaterialLibrary
            from .collide import SceneField, region_around
            st = Usd.Stage.CreateInMemory()
            self.build_static(st, "/Scene", MaterialLibrary(st, "/Looks"))
            t = self.data["task"].get("start", [0, 0])
            pts = [[float(t[0]), 0.0, float(t[1])]] + [[sp[0], 0.0, sp[1]] for sp in self.spots.values()]
            return SceneField.from_stage(st, region_around(np.array(pts), margin=150.0))
        except Exception as e:                   # (no pxr: the motion is made without it)
            self.notes.append(f"no equipment field: {e}")
            return None

    # ------------------------------------------------------------------ a tool kept in hand between steps
    OWN_TOOL = ("fasten", "get_tool", "put_back")

    @property
    def rest(self):
        """Hands at ease between actions; the hand that holds a tool keeps it ready in front."""
        if not self.holding:
            return self._rest_base
        out = dict(self._rest_base)
        out[self.holding[1]] = self.ready(self.holding[1])[0]
        return out

    @rest.setter
    def rest(self, v):
        self._rest_base = v

    def grip_fn(self, tool, side):
        """The hand on a tool at tool pose Td["t"]: the right hand at the grip, the left one nearer the head (so the
        two hands do not overlap when it is passed across)."""
        sk = self.sk
        if tool == "screwdriver":
            d = 18.0 if side == "r" else 10.0
            return lambda Td: K.power(sk, side, Td["t"][:3, :3] @ np.array([0.0, d, 0.0]) + Td["t"][:3, 3],
                                      Td["t"][:3, :3] @ Y, grip="power:1.6")
        d = 19.0 if side == "r" else 12.0
        return lambda Td: K.power(sk, side, Td["t"][:3, :3] @ np.array([d, 1.0, 0.0]) + Td["t"][:3, 3],
                                  -(Td["t"][:3, :3] @ X), grip="power:1.25")

    def ready(self, side, base=None, tid=None):
        """(hand target, tool pose) for a held tool at ease: fist a little in front of the resting hand, the tool's
        head forward and down."""
        tid = tid or self.holding[0]
        tool = self.parts[tid]["tool"]
        base = base or self._rest_base
        p = np.asarray(base[side].point, float) + np.array([0.0, 8.0, 17.0])   # (clear of the belly)
        if tool == "screwdriver":
            Rt = K.T_at(np.zeros(3), X, K.unit(np.array([0.0, 0.5, -1.0])))[:3, :3]
        else:
            Rt = K.T_at(np.zeros(3), K.unit(np.array([0.0, 0.35, -1.0])), Y)[:3, :3]
        g = self.grip_fn(tool, side)
        T = T_pos(p, Rt)
        for _ in range(2):                                   # the grip depends a little on where it is
            h = g({"t": T})
            T = T_pos(T[:3, 3] + p - np.asarray(h.point), Rt)
        return g({"t": T}), T

    def _snapshot(self):
        import copy
        return (len(self.frames), copy.deepcopy(self.parts), self.at, self.kneeling, self._rest_base,
                getattr(self, "rest_kneel", None), self.holding)

    def _restore(self, s):
        n, self.parts, self.at, self.kneeling, self._rest_base, self.rest_kneel, self.holding = s
        del self.frames[n:]

    def _run_holding(self, fn, step, sid):
        """Run a step. If it turns out to need the hand that holds the tool, take it back, pass the tool to the
        other hand first and run it again; if it needs both hands, say so."""
        if not self.holding or step["do"] in self.OWN_TOOL:
            fn(step, sid)
            return
        snap = self._snapshot()
        self._busy = 0
        fn(step, sid)
        if self._busy < 6:
            return
        self._restore(snap)
        side = self.holding[1]
        self.swap_hands(sid)
        self._busy = 0
        fn(step, sid)
        if self._busy >= 6:
            raise ValueError(f"it needs both hands and the worker holds {self.holding[0]}: put it back first "
                             f"(put_back)")

    def _hold_override(self, frames):
        """Frames of a step done with a tool in hand: the holding hand keeps the tool ready (counting the frames
        where the step wanted that hand somewhere else), the tool follows it."""
        if not self.holding:
            return
        tid, side = self.holding
        key = f"part:{tid}"
        h, T = self.ready(side)
        for f in frames:
            if key in f.tools:
                continue
            mine = [t for t in f.targets if t.side == side]
            if mine and math.dist(mine[0].point, h.point) > 12.0:
                self._busy = getattr(self, "_busy", 0) + 1
            f.targets = [t for t in f.targets if t.side != side] + [h]
            self.set_part(f, tid, T)

    def swap_hands(self, sid):
        """Pass the held tool to the other hand in front of the body."""
        tid, a = self.holding
        b = "l" if a == "r" else "r"
        tool = self.parts[tid]["tool"]
        from .jobs import held
        ha, Ta = self.ready(a)
        hb, Tb = self.ready(b)
        Tm = lerp4(Ta, Tb, 0.5)
        Tm[:3, 3] += np.array([0.0, 8.0, 10.0])
        ga, gb = self.grip_fn(tool, a), self.grip_fn(tool, b)
        tk = [(0.0, Ta), (0.55, Tm), (1.0, Tm), (1.55, Tb)]
        base = self._rest_base
        meet = with_grip(held(gb, Tb)({"t": Tm}), "open")
        way = moved(lerp_target(base[b], meet, 0.5), np.array([0.0, 0.0, 12.0]))   # round the belly, not through it
        hands = {a: [(0.0, held(ga, Ta)), (0.95, held(ga, Ta)), (1.55, base[a])],     # lets go once b has it
                 b: [(0.0, base[b]), (0.28, way), (0.5, meet), (0.85, held(gb, Tb)), (1.55, held(gb, Tb))]}
        fr = K.keyed(1.6, hands=hands, tools={"t": (tool, tk)}, look=tuple(Tm[:3, 3]), label="swap hands")
        for f in fr:
            self.set_part(f, tid, f.tools.pop("t")[1])
        self.holding = None                                  # (the frames carry both hands themselves)
        self._add(fr, self.at, sid)
        self.holding = (tid, b)

    def do_get_tool(self, step, sid):
        """Pick a tool up where it lies and keep it in hand: the next fasten steps use it without fetching it,
        and the worker can walk and kneel with it."""
        tid = step["tool"]
        if self.holding:
            raise ValueError(f"the worker already holds {self.holding[0]}")
        if self.parts.get(tid, {}).get("kind") != "tool":
            raise ValueError(f"{tid!r} is not a tool")
        spot = self.spot_of(step, tid)
        self.go(spot, sid)
        side = step.get("hand", "r")
        tool = self.parts[tid]["tool"]
        from .jobs import held
        T0 = self.L(spot) @ self.parts[tid]["W"]
        g = self.grip_fn(tool, side)
        _, Tr = self.ready(side, tid=tid)
        R = self._rest_base
        tk = [(0.0, T0), (1.15, T0), (1.45, T_pos(T0[:3, 3] + Y * 8.0, T0[:3, :3])), (2.2, Tr), (2.5, Tr)]
        hk = [(0.0, R[side]), (0.8, with_grip(moved(held(g, T0)({"t": T0}), Y * 6.0), "open")),
              (1.05, held(g, T0)), (2.5, held(g, T0))]
        fr = K.keyed(2.5, hands={side: hk}, tools={"t": (tool, tk)},
                     look=[(0.0, T0[:3, 3]), (1.4, T0[:3, 3]), (2.5, Tr[:3, 3])], label="get tool")
        ph = [(0.8, "reach"), (1.15, "grasp"), (2.2, "move"), (99, "hold")]
        for i, f in enumerate(fr):
            f.label = next(lab for a_, lab in ph if i / FPS <= a_)
            self.set_part(f, tid, f.tools.pop("t")[1])
        self._add(fr, spot, sid)
        self.parts[tid]["taken_from"] = (self.parts[tid]["W"].copy(), spot)
        self.holding = (tid, side)

    def do_put_back(self, step, sid):
        """Put the held tool back where it was taken from."""
        tid = step.get("tool") or (self.holding or [None])[0]
        if not self.holding or self.holding[0] != tid:
            raise ValueError(f"the worker does not hold {tid!r}")
        W, spot = self.parts[tid]["taken_from"]
        spot = self.spots[step["at"]] if "at" in step else spot
        self.go(spot, sid)
        side = self.holding[1]
        tool = self.parts[tid]["tool"]
        from .jobs import held
        T0 = self.L(spot) @ W
        _, Tr = self.ready(side)
        g = self.grip_fn(tool, side)
        R = self._rest_base
        tk = [(0.0, Tr), (0.9, T_pos(T0[:3, 3] + Y * 8.0, T0[:3, :3])), (1.2, T0), (2.4, T0)]
        hk = [(0.0, held(g, T0)), (1.25, held(g, T0)), (1.5, with_grip(moved(held(g, T0)({"t": T0}), Y * 6.0), "open")),
              (2.4, R[side])]
        fr = K.keyed(2.4, hands={side: hk}, tools={"t": (tool, tk)},
                     look=[(0.0, Tr[:3, 3]), (0.6, T0[:3, 3]), (2.4, T0[:3, 3])], label="put tool back")
        ph = [(1.2, "move"), (1.5, "release"), (99, "return")]
        self.holding = None
        for i, f in enumerate(fr):
            f.label = next(lab for a_, lab in ph if i / FPS <= a_)
            self.set_part(f, tid, f.tools.pop("t")[1])
        self._add(fr, spot, sid)
        self.commit(tid, spot, T0)

    # ------------------------------------------------------------------ posture: kneel for low work
    KNEEL_BELOW = 0.30                           # work below this x stature -> down on one knee

    def wants_kneel(self, step):
        """posture = "kneel" / "stand" / "auto" (default): auto kneels when the step's work points are low."""
        mode = step.get("posture", "auto")
        if mode != "auto":
            return mode == "kneel"
        refs = [step[k] for k in ("on",) if k in step]
        ys = [float(p[1]) for ref in refs for p in self.points(ref)]
        return bool(ys) and min(ys) < self.KNEEL_BELOW * self.w.stature

    def rest_for(self, stance):
        """Hands at ease for a stance (kneeling: lower, with the body)."""
        sol = self.S.solve([], look=None, stance=stance)
        from .kinematics import fk
        R, P = fk(self.sk, sol.pose)
        out = {}
        for s_, sx in (("l", 1.0), ("r", -1.0)):
            Rh = R[f"hand_{s_}"]
            fingers = Rh @ np.array([sx, 0.0, 0.0])
            palm = Rh @ np.array([0.0, -1.0, 0.0])
            pt = P[f"hand_{s_}"] + fingers * self.S.hand_len * 0.40 + palm * 1.6 + np.array([0.0, 7.0, 6.0])
            out[s_] = HandTarget(s_, tuple(pt), tuple(fingers), tuple(palm), "relaxed", 0.3)
        return out

    def _stance_between(self, a, b, u, front="l"):
        """Standing -> kneeling at u in 0..1: the front foot steps forward first, then the back foot goes back,
        toes tucked, as the pelvis lowers."""
        from .solver import Stance
        back = "r" if front == "l" else "l"
        o = Stance(yaw=a.yaw)
        uf = ease(min(1.0, u / 0.45))
        ub = ease(max(0.0, (u - 0.35) / 0.65))
        for s_, uu, arc in ((front, uf, 6.0 * math.sin(math.pi * uf)), (back, ub, 3.0 * math.sin(math.pi * ub))):
            pa = np.asarray(getattr(a, "left" if s_ == "l" else "right"), float)
            pb = np.asarray(getattr(b, "left" if s_ == "l" else "right"), float)
            setattr(o, "left" if s_ == "l" else "right", tuple(pa * (1 - uu) + pb * uu + np.array([0.0, arc, 0.0])))
        setattr(o, f"lift_{front}", -10.0 * math.sin(math.pi * uf))
        setattr(o, f"lift_{back}", getattr(b, f"lift_{back}") * ub)
        o.drop = (b.drop or 0.0) * ease(max(0.0, (u - 0.3) / 0.7))
        o.hinge = b.hinge if u > 0.5 else None
        return o

    def _transition(self, sid, down):
        st = self.kneeling[1]
        s0 = self.S.default_stance()
        R0, R1 = self.rest_stand, self.rest_kneel
        n = int(1.3 * FPS)
        for k in range(n):
            u = (k + 1) / (n + 1)
            u = u if down else 1.0 - u
            hands = [lerp_target(R0[s_], R1[s_], ease(u)) for s_ in ("l", "r")]
            tools = {}
            if self.holding:                                 # the tool comes along in its hand
                tid, hs = self.holding
                (h0, T0), (h1, T1) = self.ready(hs, R0), self.ready(hs, R1)
                hands = [h for h in hands if h.side != hs] + [lerp_target(h0, h1, ease(u))]
                tools[f"part:{tid}"] = (self.parts[tid]["spec"], lerp4(T0, T1, ease(u)))
            self.frames.append(Frame(hands, None, tools, root=self.at, stance=self._stance_between(s0, st, u),
                                     label=f"{sid}:{'kneel down' if down else 'stand up'}"))

    def kneel(self, sid, step):
        if self.kneeling:
            return
        from .handkit import kneel_stance
        st = kneel_stance(self.sk, "r", forward=float(step.get("kneel_forward", 24.0)), width=6.0)
        st = self._kneel_clear(st)
        self.kneeling = (self.at, st)
        self.rest_kneel = self.rest_for(st)
        from .kinematics import fk
        _, P = fk(self.sk, self.S.solve([], look=None, stance=st).pose)
        self._shoulders_kneel = {s_: P[f"upperarm_{s_}"].copy() for s_ in ("l", "r")}     # (trunk leant forward)
        self._transition(sid, True)
        self.rest = self.rest_kneel

    def _kneel_clear(self, st):
        """Kneel a little further back if the knee, thigh or body would be inside the equipment (a tyre, a chassis
        rail) where the worker stands."""
        if self.field is None:
            return st
        import copy

        from .collide import body_points
        from .kinematics import fk
        W = _rootT(self.at)
        best = None
        for back in (0.0, 6.0, 12.0, 18.0, 24.0, 32.0):
            s2 = copy.copy(st)
            s2.left = tuple(np.asarray(st.left, float) - np.array([0.0, 0.0, back]))
            s2.right = tuple(np.asarray(st.right, float) - np.array([0.0, 0.0, back]))
            _, P = fk(self.sk, self.S.solve([], look=None, stance=s2, world=W).pose)
            pts, rad, part = body_points(P)
            pen = float(np.max(self.field.penetration(((W[:3, :3] @ pts.T).T + W[:3, 3], rad, part)), initial=0.0))
            if best is None or pen < best[0] - 0.5:
                best = (pen, s2, back)
            if pen <= 0.0:
                break
        if best[2]:
            self.notes.append(f"kneels {best[2]:.0f} cm further back to clear the equipment")
        return best[1]

    def generator_posture(self):
        """For the library generators that work out their own rest pose from the solver's default stance: while
        kneeling that default is the kneel, so their reaches start and end at the kneeling hands."""
        import contextlib

        @contextlib.contextmanager
        def cm():
            if not self.kneeling:
                yield
                return
            st = self.kneeling[1]
            self.S.default_stance = lambda *a, **k: st
            try:
                yield
            finally:
                del self.S.default_stance
        return cm()

    def stand_up(self, sid):
        if not self.kneeling:
            return
        self._transition(sid, False)
        self.kneeling = None
        self.rest = self.rest_stand

    def _phases(self, a0, b=None):
        out = []
        for i in range(a0, (b + 1) if b is not None else len(self.frames)):
            lab = self.frames[i].label.split(":", 1)[-1]
            if not out or out[-1][0] != lab:
                out.append([lab, i, i])
            else:
                out[-1][2] = i
        return out

    # frames are in the frame of the spot they are worked from; parts given as local 4x4 per frame
    def _add(self, frames, spot, label):
        for f in frames:
            if f.root is None:
                f.root = spot
            if self.kneeling and f.stance is None:
                f.stance = self.kneeling[1]
            f.label = f"{label.split(':')[0]}:{f.label or label.split(':')[-1]}"
        self._hold_override(frames)
        self.frames.extend(frames)

    def L(self, spot):
        return np.linalg.inv(_rootT(spot))

    def loc(self, spot, p):
        return self.L(spot)[:3, :3] @ np.asarray(p, float) + self.L(spot)[:3, 3]

    def locd(self, spot, d):
        return self.L(spot)[:3, :3] @ np.asarray(d, float)

    def set_part(self, f, pid, T_local):
        f.tools[f"part:{pid}"] = (self.parts[pid]["spec"], T_local)

    def commit(self, pid, spot, T_local):
        """A part's new resting place (world) after a step; its children follow."""
        W = _rootT(spot) @ T_local
        self.parts[pid]["W"] = W
        for c in self.children(pid):
            self.parts[c]["W"] = W @ self.parts[c]["rel"]

    CLEAR = 28.0                 # cm a walking body keeps from the equipment (half the shoulders + a little)
    HEIGHTS = (30.0, 70.0, 110.0)   # where it is checked: knee, hip, chest

    def _blocked(self, pts, a, b):
        """For floor points (x, z): too close to the equipment to walk past (not near the two ends - the work spots
        are next to the equipment on purpose)."""
        pts = np.asarray(pts, float).reshape(-1, 2)
        P = np.array([[x, h, z] for x, z in pts for h in self.HEIGHTS])
        d = self.field.distance(P).reshape(len(pts), len(self.HEIGHTS)).min(axis=1)
        near = np.minimum(np.linalg.norm(pts - np.asarray(a), axis=1), np.linalg.norm(pts - np.asarray(b), axis=1))
        return (d < self.CLEAR) & (near > 35.0)

    def _clear(self, a, b):
        n = max(2, int(math.dist(a, b) / 5.0))
        return not self._blocked([np.asarray(a) + (np.asarray(b) - a) * k / n for k in range(n + 1)], a, b).any()

    def route(self, a, b, h=10.0):
        """Floor path a -> b round the equipment: straight when it is clear; else the shortest way on an h cm grid
        (A*), pulled tight into a few straight legs. A route too hard to find: straight (and the check flags it)."""
        import heapq
        a, b = np.asarray(a, float), np.asarray(b, float)
        if self.field is None or self._clear(a, b):
            return [tuple(a), tuple(b)]
        lo = np.minimum(a, b) - 250.0
        nx, nz = (np.ceil((np.maximum(a, b) + 250.0 - lo) / h)).astype(int) + 1
        gx, gz = np.meshgrid(np.arange(nx), np.arange(nz), indexing="ij")
        cells = np.stack([gx.ravel(), gz.ravel()], 1) * h + lo
        bad = self._blocked(cells, a, b).reshape(nx, nz)
        ia = tuple(np.round((a - lo) / h).astype(int))
        ib = tuple(np.round((b - lo) / h).astype(int))
        steps = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
                 (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414)]
        g, came, q = {ia: 0.0}, {}, [(0.0, ia)]
        while q:
            _, c = heapq.heappop(q)
            if c == ib:
                break
            for dx, dz, w in steps:
                n_ = (c[0] + dx, c[1] + dz)
                if not (0 <= n_[0] < nx and 0 <= n_[1] < nz) or bad[n_]:
                    continue
                gn = g[c] + w
                if gn < g.get(n_, 1e18):
                    g[n_], came[n_] = gn, c
                    heapq.heappush(q, (gn + math.dist(n_, ib), n_))
        if ib not in came:
            self.notes.append(f"no clear walking route from {a.round()} to {b.round()}: walks straight")
            return [tuple(a), tuple(b)]
        cell = [ib]
        while cell[-1] != ia:
            cell.append(came[cell[-1]])
        pts = [a] + [np.array(c_, float) * h + lo for c_ in reversed(cell[1:-1])] + [b]
        out, i = [a], 0                                              # pulled tight: the farthest point seen
        while i < len(pts) - 1:
            j = len(pts) - 1
            while j > i + 1 and not self._clear(pts[i], pts[j]):
                j -= 1
            out.append(pts[j])
            i = j
        return [tuple(p) for p in out]

    def go(self, spot, sid, carry=None, parts=None):
        """Walk (or turn) to a spot if the worker is not there yet."""
        if math.dist(self.at[:2], spot[:2]) < 5.0 and abs(((spot[2] - self.at[2]) + 180) % 360 - 180) < 5:
            if self._want_kneel:
                self.kneel(sid, {})
            return
        self.stand_up(sid)
        if self.holding and carry is None:                    # walk with the tool held ready
            h, T = self.ready(self.holding[1])
            carry, parts = [h], dict(parts or {}, **{self.holding[0]: T})
        fr = walk(self.w, self.sk, self.route(self.at[:2], spot[:2]), self.at[2], spot[2], carry=carry)
        for f in fr:
            f.label = f"{sid}:walk"
            for pid, T in (parts or {}).items():
                self.set_part(f, pid, T)
        self.frames.extend(fr)
        self.at = spot
        if self._want_kneel:
            self.kneel(sid, {})

    # ------------------------------------------------------------------ verbs
    def take_free(self, kind, home, n):
        got = [q for q, p in self.parts.items() if p.get("kind") == kind and p.get("home") == home]
        if len(got) < n:
            raise ValueError(f"{home!r} has only {len(got)} {kind}s left, the step wants {n}")
        return got[:n]

    def do_transfer(self, step, sid):
        """Small parts from a bin to a place: reach into the bin, pinch a few, lift, move over, open the hand
        (they drop into their slots), hand back. One-hand get & place (MTM-UAS)."""
        src = step["from"]
        spot = self.spot_of(step, src)
        self.go(spot, sid)
        n = int(step.get("count", 1))
        kind = step["what"]
        pids = self.take_free(kind, src, n)
        drops = self.points(step["to"])
        if len(drops) < n:
            raise ValueError(f"{step['to']!r} has {len(drops)} slots, the step drops {n}")
        tgt_asset = step["to"].split(".")[0]
        grab = self.loc(spot, self.points(f"{src}.grab")[0])
        side = step.get("hand", "r" if self.loc(spot, self.points(f"{src}.grab")[0])[0] <= 15 else "l")
        R = self.rest
        pin = K.pinch(self.sk, side, grab, approach=(0.0, -0.75, 0.65), palm=(0, -1, 0.3))
        hover = moved(with_grip(pin, "open"), Y * 9.0)
        drop_c = np.mean([self.loc(spot, p) for p in drops[:n]], axis=0)
        over = K.pinch(self.sk, side, drop_c + Y * 7.0, approach=(0.0, -0.7, 0.7), palm=(0, -1, 0.3))
        keys = [(0.0, R[side]), (0.75, hover), (1.05, pin), (1.35, pin), (1.7, moved(pin, Y * 12.0)),
                (2.5, over), (2.62, with_grip(over, "open")), (2.9, moved(with_grip(over, "open"), Y * 4.0)),
                (3.6, R[side])]
        look = [(0.0, grab), (1.6, grab), (2.1, drop_c), (3.6, drop_c)]
        fr = K.keyed(3.6, hands={side: keys}, look=look, label="transfer")
        labels = [(0.75, "reach"), (1.35, "grasp"), (2.5, "move"), (2.9, "release"), (9, "return")]
        T0 = {q: self.L(spot) @ self.parts[q]["W"] for q in pids}
        off = {q: np.linalg.inv(_frame_of(pin)) @ T0[q] for q in pids}
        # in the hand they bunch up between the fingertips
        tip = np.linalg.inv(_frame_of(pin)) @ np.append(grab, 1.0)
        for k_, q in enumerate(pids):
            off[q][:3, 3] = tip[:3] + np.array([0.0, 0.0, (k_ - (n - 1) / 2) * 0.6])
        Tdrop = {}
        for k_, q in enumerate(pids):
            p = self.loc(spot, drops[k_])
            Rr = rot_axis(Y, 37.0 * k_) @ (rot_axis(Z, 90.0) if kind == "screw" else np.eye(3))
            Tdrop[q] = T_pos(p + Y * (0.25 if kind == "nut" else 0.5), Rr)
        for i, f in enumerate(fr):
            t = i / FPS
            f.label = next(lab for tt, lab in labels if t <= tt)
            for q in pids:
                if t < 1.35:
                    T = T0[q]
                elif t < 2.62:
                    T = _frame_of(target_of(f, side)) @ off[q]
                else:
                    u = min(1.0, (t - 2.62) / 0.18)                    # falls into the slot
                    Th = _frame_of(over) @ off[q]
                    T = lerp4(Th, Tdrop[q], u * u)
                self.set_part(f, q, T)
        self._add(fr, spot, sid)
        for q in pids:
            self.parts[q]["home"] = tgt_asset
            self.commit(q, spot, Tdrop[q])
            if tgt_asset in self.parts:                                  # they now ride on the tray
                self.parts[q]["parent"] = tgt_asset
                self.parts[q]["rel"] = np.linalg.inv(self.parts[tgt_asset]["W"]) @ self.parts[q]["W"]

    def do_carry(self, step, sid):
        """Two-hand pick of a part (hands on its sides), walk with it if the place is elsewhere, put it down.
        Whatever rides on it (parts in a tray) comes along."""
        pid = step["what"]
        p = self.parts[pid]
        size = p.get("size", (30, 10, 20))
        s1 = self.spot_of(step, step.get("from_spot_asset", pid) if "from" not in step else step["from"])
        if "spot" not in self.assets.get(pid, {}) and "from" not in step and "at" not in step:
            raise KeyError(f"carry {pid}: give the part a spot (where it is picked from)")
        self.go(s1, sid)
        dest_ref = step["to"]
        dest_aid = dest_ref.split(".")[0]
        place_w = self.points(dest_ref)[0]
        s2 = self.spots[step["place_at"]] if "place_at" in step else self.spot_of({}, dest_aid)
        kids = [(c, self.parts[c]["rel"]) for c in self.children(pid)]
        Tp = self.L(s1) @ p["W"]
        c0 = Tp[:3, 3]
        f1, hold, Tc = C.carry_pick(self.S, c0, size, pid, p["spec"], True, "r")
        for f in f1:
            f.label = "pick"
            T = f.tools[f"part:{pid}"][1].copy()
            T[:3, :3] = Tp[:3, :3]
            f.tools[f"part:{pid}"] = (p["spec"], T)
        Tc = Tc.copy()
        Tc[:3, :3] = Tp[:3, :3]
        # walk with it (orientation as it lay, relative to the worker)
        self._add(f1, s1, sid)
        parts = {pid: Tc}
        parts.update({c: Tc @ r for c, r in kids})
        if math.dist(s1[:2], s2[:2]) > 5.0 or abs(s1[2] - s2[2]) > 5:
            self.go(s2, sid, carry=hold, parts=parts)
        R_place = Ry(float(step.get("yaw", 0.0)))[:3, :3]
        place_l = self.loc(s2, place_w) + Y * size[1] / 2
        f3 = C.carry_place(self.S, place_l, size, pid, p["spec"], True, "r")
        Tend = None
        for f in f3:
            f.label = "place"
            T = f.tools[f"part:{pid}"][1].copy()
            T[:3, :3] = Tc[:3, :3] if Tend is None else Tend[:3, :3]
            f.tools[f"part:{pid}"] = (p["spec"], T)
        Tend = T_pos(place_l, self.L(s2)[:3, :3] @ _rootT(s2)[:3, :3] @ Tc[:3, :3] if "yaw" not in step else R_place)
        for f in f3:
            T = f.tools[f"part:{pid}"][1]
            T[:3, :3] = Tend[:3, :3]
        self._add(f3, s2, sid)
        for f in f1 + f3:
            T = f.tools[f"part:{pid}"][1]
            for c, r in kids:
                self.set_part(f, c, T @ r)
        self.commit(pid, s2, Tend)
        self.parts[pid]["spot"] = step.get("place_at") or self.assets[dest_aid].get("spot")

    def _brace_on(self, spot, step):
        if "brace" not in step:
            return None
        side = "l"
        if self.holding and self.holding[1] == side:
            return None
        # the free (left) hand braces on its own side: of the points given, the one furthest left; a point right of
        # the body's middle would put the arm across the chest - nobody braces like that, so then no brace
        b = max((self.loc(spot, p) for p in self.points(step["brace"])), key=lambda q: float(q[0]))
        if float(b[0]) < -5.0:
            self.notes.append(f"{step.get('do')} on {step.get('on')}: brace {step['brace']} is on the worker's "
                              f"right ({-float(b[0]):.0f} cm) - the left hand can't brace there without crossing "
                              f"the body; it stays free")
            return None
        return K.brace(self.sk, side, b, self.locd(spot, self.axis_of(step["brace"])), weight=0.6)

    def do_hand_start(self, step, sid):
        """For each point: pick the part from where it lies (tray slot), bring it over the stud / hole, thread it
        on with a few finger twists, let go. The other hand rests on the work."""
        kind = step["what"]
        on = self.points(step["on"])
        aid = step["on"].split(".")[0]
        spot = self.spot_of(step, aid if aid in self.assets and "spot" in self.assets[aid] else step["on_spot"])
        self.go(spot, sid)
        holder = step.get("from")
        pool = [q for q, p in self.parts.items() if p.get("kind") == kind and (holder is None or p.get("home") == holder)]
        n = int(step.get("count", len(on)))
        if len(pool) < n:
            raise ValueError(f"step {sid}: needs {n} {kind}s, {len(pool)} available")
        R = self.rest
        side = "r"
        br = self._brace_on(spot, step)
        rise = 0.9 if kind == "screw" else 0.6                          # how far above seated it is left
        nrm = self.locd(spot, self.axis_of(step["on"]))                 # the surface faces this way (worker frame)
        fw = perp(Z, nrm)
        fw = K.unit(fw) if np.linalg.norm(fw) > 0.3 else -Y            # a wall facing the worker: come in from above
        Rn = R_up(nrm)
        for k_ in range(n):
            q = pool[k_]
            P = self.loc(spot, on[k_])                                    # seat point (top of the surface / stud)
            T0 = self.L(spot) @ self.parts[q]["W"]
            p0 = T0[:3, 3]
            pin = K.pinch(self.sk, side, p0 + Y * 0.5, approach=(0.0, -0.8, 0.6), palm=(0, -1, 0.3))
            top = P + nrm * (rise + 1.2 + (0.35 if kind == "screw" else 0.25))
            if abs(nrm[1]) < 0.5:                       # a face towards the worker: fingertips in, palm down
                hold = K.pinch(self.sk, side, top + nrm * 0.4, approach=-nrm * 1.0 - Y * 0.25 + X * 0.1,
                               palm=(0.0, -1.0, 0.0))
            else:                                       # up (or down, overhead): come in over / under it
                hold = K.pinch(self.sk, side, top + nrm * 0.4, approach=-nrm * 0.6 + fw * 1.0 + X * 0.15,
                               palm=-nrm + fw * 0.2)
            keys = [(0.0, R[side]), (0.75, with_grip(moved(pin, Y * 6), "open")), (1.0, pin), (1.2, pin),
                    (1.5, moved(pin, Y * 8)), (2.2, moved(hold, nrm * 3)), (2.45, hold)]
            t = 2.45
            twists = int(step.get("twists", 3))
            for _ in range(twists):
                loose = "tripod"                    # fingers ease off for the regrip (a plain blend, no fling open)
                keys += [(t + 0.4, rot_target(hold, nrm, -55.0, top)),
                         (t + 0.5, with_grip(rot_target(hold, nrm, -55.0, top), loose)),
                         (t + 0.65, hold)]
                t += 0.65
            keys += [(t + 0.2, with_grip(moved(hold, nrm * 3), "open")), (t + 0.9, R[side])]
            dur = t + 0.95
            hands = {side: keys}
            if br is not None:
                hands["l"] = [(0.0, R["l"]), (0.9, br), (t, br), (t + 0.8, R["l"])]
            fr = K.keyed(dur, hands=hands, look=[(0.0, p0), (1.3, p0), (1.9, P), (dur, P)], label="hand start")
            Tseat = lambda h, ang: T_pos(P + nrm * h, rot_axis(nrm, ang) @ Rn)   # noqa: E731
            off = np.linalg.inv(_frame_of(pin)) @ T0
            off_h = None
            ph = [(0.9, "reach"), (1.2, "grasp"), (2.45, "move"), (t, "thread"), (99, "return")]
            for i, f in enumerate(fr):
                tt = i / FPS
                f.label = next(lab for a_, lab in ph if tt <= a_)
                if tt < 1.2:
                    T = T0
                elif tt < 2.45:
                    u = ease((tt - 1.2) / 1.25)
                    Tf = _frame_of(target_of(f, side)) @ off
                    Tu = Tseat(rise + 1.2, 0.0)
                    T = lerp4(Tf, Tu, min(1.0, u * 1.15))                # it turns upright in the fingers on the way
                elif tt < t:
                    k2 = (tt - 2.45) / 0.65
                    j = int(k2)
                    v = min(1.0, (k2 - j) / (0.4 / 0.65))
                    ang = -55.0 * (j + ease(v))
                    T = Tseat(rise + 1.2 - 1.2 * min(1.0, (tt - 2.45) / (t - 2.45)), ang)
                else:
                    T = Tseat(rise, -55.0 * twists)
                self.set_part(f, q, T)
            self._add(fr, spot, sid)
            self.parts[q].update(home=f"{step['on']}#{k_}", seat=self.points(step["on"])[k_], parent=None,
                                 rise=rise, angle=-55.0 * twists, axis=self.axis_of(step["on"]))
            self.commit(q, spot, Tseat(rise, -55.0 * twists))

    def do_fasten(self, step, sid):
        """Get the tool where it lies, drive / tighten each started part at the points, put the tool back."""
        tid = step["tool"]
        tool = self.parts[tid]["tool"]
        on = self.points(step["on"])
        aid = step["on"].split(".")[0]
        spot = self.spot_of(step, aid if "spot" in self.assets.get(aid, {}) else step["on_spot"])
        self.go(spot, sid)
        sk = self.sk
        T_rest = self.L(spot) @ self.parts[tid]["W"]
        hold = self.holding                                  # already in hand (get_tool): no fetching, keep it
        if hold and hold[0] != tid:
            raise ValueError(f"the worker holds {hold[0]}, not {tid}: put it back first")
        from_l = bool(hold) and hold[1] == "l"               # in the left hand: the right one takes it over
        T_home = self.ready(hold[1])[1] if hold else T_rest
        h_l = self.ready("l")[0] if from_l else None
        if hold:
            self.holding = (tid, "r")
        T_end = self.ready("r")[1] if hold else T_rest
        R = self._rest_base
        seated = []
        for p in on:
            q = next((q for q, v in self.parts.items() if v.get("seat") is not None
                      and np.linalg.norm(np.asarray(v["seat"]) - p) < 0.5), None)
            seated.append(q)
        Ps = [self.loc(spot, p) for p in on]
        br = self._brace_on(spot, step)
        tk, hk, part_keys, look = [], [], [], []
        nrm = self.locd(spot, self.axis_of(step["on"]))
        Rn = R_up(nrm)
        e1 = Rn[:, 0]
        if tool == "screwdriver":
            def T_work(P, h, ang):
                x = rot_axis(-nrm, ang) @ e1
                return K.T_at(P + nrm * h, x, nrm)                         # tip on the screw, handle out along it

            def rh(Td):
                T = Td["t"]
                return K.power(sk, "r", T[:3, :3] @ np.array([0, 18.0, 0]) + T[:3, 3], T[:3, :3] @ Y,
                               grip="power:1.6")
            strokes, per, step_ang = int(step.get("strokes", 5)), 0.75, 60.0
        else:                                                            # ratchet on the nuts
            def T_work(P, h, ang):
                d = rot_axis(-nrm, ang) @ K.unit(perp(np.array([-1.0, 0.0, -0.45]), nrm))
                return K.T_at(P + nrm * (h + 2.0), d, nrm)

            def rh(Td):
                T = Td["t"]
                return K.power(sk, "r", T[:3, :3] @ np.array([19.0, 1.0, 0]) + T[:3, 3], -(T[:3, :3] @ X),
                               grip="power:1.25")
            strokes, per, step_ang = int(step.get("strokes", 4)), 0.8, 45.0

        def tool_ang(cum, power):
            """Tool angle after `cum` degrees of turning: a screwdriver turns with the screw; a ratchet's handle
            swings through the stroke and clicks back (only the nut keeps the turn)."""
            if tool == "screwdriver":
                return cum
            return step_ang / 2 if power else -step_ang / 2
        from .jobs import held
        T_first = T_work(Ps[0], self.parts[seated[0]]["rise"] if seated[0] else 0.0, tool_ang(0.0, False))
        if tool == "screwdriver" and not hold:
            # round about its handle: picked up (and put back) turned so the work grip is a palm-down grab where
            # it lies - the fingers roll it on the way, the hand does not arrive upside down
            def at_roll(T, deg):
                return T @ T_pos(np.zeros(3), rot_axis(Y, deg))
            g = held(rh, T_first)
            best = max(range(0, 360, 20), key=lambda d: -float(np.asarray(g({"t": at_roll(T_rest, d)}).palm)[1]))
            T_home = T_end = at_roll(T_rest, best)
        tk.append((0.0, T_home))
        tk.append((1.0, T_home))
        hk += [(0.0, held(rh, T_first) if hold and not from_l else R["r"]), (0.95, held(rh, T_first)),
               (1.0, held(rh, T_first))]
        look += [(0.0, T_home[:3, 3]), (1.0, T_home[:3, 3])]
        t = 1.0
        phases = [(1.0, "take over" if from_l else "ready" if hold else "get tool")]
        for k_, (P, q) in enumerate(zip(Ps, seated)):
            rise = self.parts[q]["rise"] if q else 0.0
            a0 = self.parts[q]["angle"] if q else 0.0
            t_in = t + (1.1 if k_ == 0 else 0.7)
            if k_ > 0:
                tk.append((t + 0.25, T_work(Ps[k_ - 1], 3.0, tool_ang(ang, False))))
                tk.append((t + 0.55, T_work(P, rise + 3.0, tool_ang(0.0, False))))
            tk.append((t_in, T_work(P, rise, tool_ang(0.0, False))))
            look.append((t_in - 0.3, P))
            phases.append((t_in, "position"))
            hk.append((t_in, rh))
            t = t_in
            h0 = rh({"t": T_work(P, rise, tool_ang(0.0, False))})
            ang = 0.0
            for s_ in range(strokes):
                h = rise * (1.0 - (s_ + 1) / strokes)
                ang += step_ang
                tk += [(t + per * 0.62, T_work(P, h, tool_ang(ang, True))), (t + per, T_work(P, h, tool_ang(ang, False)))]
                part_keys.append((q, t, t + per * 0.62, rise * (1.0 - s_ / strokes), h, a0 + 0.0, ang))
                if tool == "screwdriver":                                 # the hand rolls with the tool, loosens, rolls back
                    last = s_ == strokes - 1
                    hk += [(t + 0.08, h0 if s_ == 0 else with_grip(h0, "power:1.6")),
                           (t + per * 0.62, with_grip(rot_target(h0, -nrm, step_ang, P), "power:1.6")),
                           (t + per * 0.72, with_grip(rot_target(h0, -nrm, step_ang, P), "power:2.3" if not last else "power:1.6")),
                           (t + per, with_grip(h0, "power:2.3" if not last else "power:1.6"))]
                    h0 = moved(h0, -nrm * rise / strokes)
                else:
                    hk.append((t + per * 0.62, rh))
                t += per
            phases.append((t, "drive" if tool == "screwdriver" else "tighten"))
            if tool == "screwdriver":
                hk.append((t + 0.05, rh))
        # tool back where it was
        t_back = t + 1.2
        last_T = T_work(Ps[-1], 0.0, tool_ang(ang, False))
        if tool == "screwdriver" and not hold:                   # put back turned for the grip it has now
            g2 = held(rh, T_work(Ps[-1], 4.0, tool_ang(ang, False)))
            best2 = max(range(0, 360, 20), key=lambda d: -float(np.asarray(g2({"t": at_roll(T_rest, d)}).palm)[1]))
            T_end = at_roll(T_rest, best2)
        tk += [(t + 0.3, T_work(Ps[-1], 4.0, tool_ang(ang, False))), (t_back, T_end), (t_back + 0.8, T_end)]
        hk += [(t + 0.3, rh), (t_back - 0.02, held(rh, last_T if False else T_first)), (t_back, held(rh, T_first))]
        hk = [(a_, v) for a_, v in hk]
        hk.append((t_back + 0.75, held(rh, T_first) if hold else R["r"]))
        look.append((t + 0.3, Ps[-1]))
        look.append((t_back - 0.2, T_end[:3, 3]))
        phases += [(t_back, "tool to ready" if hold else "put tool back"), (99, "hold" if hold else "return")]
        dur = t_back + 0.8
        hands = {"r": self._held_back(hk, tk, rh, T_first, t, t_back, held)}
        if from_l:                                           # holds it out until the right hand has it
            hands["l"] = [(0.0, h_l), (1.0, h_l)] + ([(1.6, br), (t, br), (t + 0.9, R["l"])] if br is not None
                                                      else [(1.8, R["l"])])
        elif br is not None:
            hands["l"] = [(0.0, R["l"]), (1.6, br), (t, br), (t + 0.9, R["l"])]
        fr = K.keyed(dur, hands=hands, tools={"t": (tool, tk)}, look=look, label="fasten")
        for i, f in enumerate(fr):
            tt = i / FPS
            f.label = next(lab for a_, lab in phases if tt <= a_)
            key, T = f.tools.pop("t")
            self.set_part(f, tid, T)
            for q, ta, tb, h0_, h1_, a0_, ang_ in part_keys:
                if q is None:
                    continue
                P = Ps[seated.index(q)]
            # the started parts turn and go down with the tool on each power stroke
            for q in [s_ for s_ in seated if s_]:
                P = Ps[seated.index(q)]
                hs = [pk for pk in part_keys if pk[0] == q]
                a0_ = self.parts[q]["angle"]
                if tt < hs[0][1]:
                    h_, ang_ = hs[0][3], 0.0
                else:
                    h_, ang_ = hs[-1][4], hs[-1][6]
                    for pk in hs:
                        if pk[1] <= tt <= pk[2]:
                            u = ease((tt - pk[1]) / (pk[2] - pk[1]))
                            h_ = pk[3] + (pk[4] - pk[3]) * u
                            ang_ = (pk[6] - step_ang) + step_ang * u
                            break
                        if tt < pk[1]:
                            h_, ang_ = pk[3], pk[6] - step_ang
                            break
                self.set_part(f, q, T_pos(P + nrm * h_, rot_axis(nrm, a0_ - ang_) @ Rn))
        self._add(fr, spot, sid)
        if not hold:
            self.commit(tid, spot, T_end)                        # (as it was put back)
        for q in [s_ for s_ in seated if s_]:
            P = Ps[seated.index(q)]
            a_end = self.parts[q]["angle"] - step_ang * strokes
            self.parts[q].update(rise=0.0, angle=a_end, fastened=True)
            self.commit(q, spot, T_pos(P, rot_axis(nrm, a_end) @ Rn))

    @staticmethod
    def _held_back(hk, tk, rh, T_first, t_end, t_back, held):
        """While the tool travels back to its place the hand stays rigid on it (as it was gripped)."""
        out = []
        for a_, v in hk:
            if t_end + 0.3 < a_ <= t_back:
                continue
            out.append((a_, v))
        last = next(T for a_, T in reversed(tk) if a_ <= t_end + 0.3 + 1e-6)
        out.insert(len(out) - 1, (t_end + 0.31, held(rh, last)))
        out.insert(len(out) - 1, (t_back, held(rh, last)))
        out.sort(key=lambda kv: kv[0])
        return out

    def _door_T(self, d, ang):
        return T_pos(d["hinge"], Ry(ang))

    def do_push_door(self, step, sid):
        """Hand flat on the open door's outer face, push it most of the way, let go - it swings shut by itself."""
        cab = step["door"]
        did = f"{cab}_door"
        d = self.parts[did]
        spot = self.spots[step["at"]] if "at" in step else self.spots[self.assets[cab]["door"]["push_spot"]]
        self.go(spot, sid)
        r = float(step.get("radius", d["width"] * 0.7))
        yh = float(step.get("height", 120.0)) - d["hinge"][1]
        a_open, a_rel = d["open"], float(step.get("release_deg", 45.0))
        Linv = self.L(spot)

        def door_l(ang):
            return Linv @ self._door_T(d, ang)

        def hand_at(ang):
            Td = door_l(ang)
            p = Td[:3, :3] @ np.array([r, yh, -2.0]) + Td[:3, 3]          # on the outer face
            n = Td[:3, :3] @ np.array([0.0, 0.0, -1.0])
            side = step.get("hand", "l" if p[0] > 0 else "r")
            return K.brace(self.sk, side, p, n, toward=Td[:3, :3] @ Y, weight=1.0)
        h0 = hand_at(a_open)
        side = h0.side
        R = self.rest
        n1, n2 = int(0.8 * FPS), int(0.9 * FPS)
        fr = []
        Dl = d
        for i in range(n1):                                              # reach to the door
            u = ease((i + 1) / n1)
            fr.append(Frame([lerp_target(R[side], h0, u)], tuple(h0.point), label="reach"))
            fr[-1].tools[f"part:{did}"] = (Dl["spec"], door_l(a_open))
        for i in range(n2):                                              # push
            u = ease((i + 1) / n2)
            ang = a_open + (a_rel - a_open) * u
            fr.append(Frame([hand_at(ang)], tuple(hand_at(ang).point), label="push"))
            fr[-1].tools[f"part:{did}"] = (Dl["spec"], door_l(ang))
        hl = hand_at(a_rel)
        n3 = int(0.9 * FPS)
        swing = int(0.55 * FPS)
        closed = door_l(0.0)[:3, 3]
        for i in range(n3):                                              # let go; the door swings shut and settles
            u = ease((i + 1) / n3)
            ft = Frame([lerp_target(hl, R[side], u)], tuple(closed + np.array([0, 40.0, 0])), label="let go")
            if i < swing:
                ang = a_rel * (1.0 - ((i + 1) / swing) ** 2)
            else:
                k = (i - swing) / max(1, n3 - swing)
                ang = 3.0 * math.sin(math.pi * k) * (1 - k)
            ft.tools[f"part:{did}"] = (Dl["spec"], door_l(ang))
            fr.append(ft)
        lat = f"{cab}_latch"
        for f in fr:
            Td = f.tools[f"part:{did}"][1]
            self.set_part(f, lat, Td @ self.parts[lat]["rel"])
        self._add(fr, spot, sid)
        self.commit(did, spot, door_l(0.0))
        self.parts[did]["open"] = 0.0

    def do_latch(self, step, sid):
        """Press the closed door home with one hand, turn the quarter-turn latch with the other."""
        cab = step["door"]
        did, lat = f"{cab}_door", f"{cab}_latch"
        spot = self.spots[step["at"]] if "at" in step else self.spots[self.assets[cab]["spot"]]
        self.go(spot, sid)
        Linv = self.L(spot)
        Td = Linv @ self.parts[did]["W"]
        rel = self.parts[lat]["rel"]
        dw, dh = self.parts[did]["width"], self.parts[did]["height"]
        n = Td[:3, :3] @ np.array([0.0, 0.0, -1.0])                      # outer face normal (towards the worker)
        press_p = Td[:3, :3] @ np.array([dw * 0.45, dh * 0.55, -2.0]) + Td[:3, 3]
        Tl0 = Td @ rel
        spindle = Tl0[:3, 3]
        R = self.rest
        lever = Tl0[:3, :3] @ np.array([0, -3.0, -2.2]) + spindle
        side_l = "l" if spindle[0] > press_p[0] else "r"
        side_p = "r" if side_l == "l" else "l"
        press = K.brace(self.sk, side_p, press_p, n, toward=Td[:3, :3] @ Y, weight=1.0)
        grab = K.pinch(self.sk, side_l, lever, approach=-n * 0.8 + Td[:3, :3] @ -Y * 0.2, palm=-(Td[:3, :3] @ X)
                       if side_l == "l" else Td[:3, :3] @ X, grip="lateral")
        turn = float(step.get("turn_deg", 90.0))
        keys_l = [(0.0, R[side_l]), (1.0, with_grip(moved(grab, -n * 4), "open")), (1.25, grab), (1.45, grab),
                  (2.15, rot_target(grab, -n, turn, spindle)), (2.35, with_grip(rot_target(grab, -n, turn, spindle), "open")),
                  (3.0, R[side_l])]
        keys_p = [(0.0, R[side_p]), (0.8, press), (1.3, moved(press, -n * -0.6)), (2.4, press), (3.0, R[side_p])]
        fr = K.keyed(3.1, hands={side_l: keys_l, side_p: keys_p}, look=tuple(spindle), label="latch")
        ph = [(0.8, "reach"), (1.45, "press"), (2.15, "turn"), (99, "return")]
        for i, f in enumerate(fr):
            tt = i / FPS
            f.label = next(lab for a_, lab in ph if tt <= a_)
            u = 0.0 if tt < 1.45 else min(1.0, ease((tt - 1.45) / 0.7))
            self.set_part(f, lat, Td @ rel @ T_pos(np.zeros(3), rot_axis(Z, -turn * u)))
            self.set_part(f, did, Td)
        self._add(fr, spot, sid)
        self.parts[lat]["rel"] = rel @ T_pos(np.zeros(3), rot_axis(Z, -turn))
        self.commit(did, spot, Td)

    def do_connect(self, step, sid):
        """For each socket: bring the connector half up, push it home to the click, tug test, let go (it stays)."""
        from .motion_more import connector
        ref = step["on"]
        aid = ref.split(".")[0]
        spot = self.spot_of(step, aid)
        self.go(spot, sid)
        nrm = self.locd(spot, self.axis_of(ref))
        for k_, p in enumerate(self.points(ref)[:int(step.get("count", 99))]):
            P = self.loc(spot, p)
            side = step.get("hand", "r" if P[0] <= 10 else "l")
            with self.generator_posture():
                fr = connector(self.S, P, -nrm, side)
            # the free hand rests flat on the panel beside the socket (nothing to hold on a panel-mounted socket)
            other = "l" if side == "r" else "r"
            sx = 1.0 if other == "l" else -1.0
            brace = None                                     # (not with the hand that holds a tool)
            if not (self.holding and self.holding[1] == other):
                brace = K.brace(self.sk, other, P + np.array([sx * 12.0, 0.0, 0.0]) - nrm * 0.0, nrm, weight=0.5)
            nr = int(0.8 * FPS)
            for i, f in enumerate(fr if brace is not None else []):
                w = min(1.0, (i + 1) / nr, (len(fr) - i) / nr)
                f.targets = [t for t in f.targets if t.side != other] + [lerp_target(self.rest[other], brace, ease(w))]
            pid = f"conn_{aid}_{k_ + 1}"
            self.parts[pid] = dict(spec="tool:connector", W=np.eye(4), kind="connector")
            for f in fr:
                if "connector" in f.tools:
                    self.set_part(f, pid, f.tools.pop("connector")[1])
            last = next(f.tools[f"part:{pid}"][1] for f in reversed(fr) if f"part:{pid}" in f.tools)
            self._add(fr, spot, sid)
            self.commit(pid, spot, last)

    def do_cable_tie(self, step, sid):
        """For each tie point on a harness: the free hand holds the harness, the other wraps the tie round, threads
        it and pulls it tight. '<points>_dir' gives the harness direction (asset frame)."""
        ref = step["on"]
        aid, _, name = ref.partition(".")
        spot = self.spot_of(step, aid)
        self.go(spot, sid)
        d = self.assets[aid].get("points", {}).get(f"{name}_dir", (1, 0, 0))
        dl = K.unit(self.locd(spot, self.asset_T(aid)[:3, :3] @ np.asarray(d, float)))
        up = Y if abs(dl[1]) < 0.9 else Z
        if np.cross(dl, K.unit(perp(up, dl)))[2] > 0:        # either way along the harness is the same harness: pick
            dl = -dl                                          # the one that has the fingers pointing away from the body
        for k_, p in enumerate(self.points(ref)[:int(step.get("count", 99))]):
            pid = f"tie_{aid}_{k_ + 1}"
            with self.generator_posture():
                fr = C.cable_tie(self.S, self.loc(spot, p), dl, "r", pid)
            self.parts[pid] = dict(spec="ring:1.6:0.18:plastic_black", W=np.eye(4), kind="tie")
            last = next(f.tools[f"part:{pid}"][1] for f in reversed(fr) if f"part:{pid}" in f.tools)
            self._add(fr, spot, sid)
            self.commit(pid, spot, last)

    def do_inspect(self, step, sid):
        ref = step["on"]
        aid = ref.split(".")[0]
        spot = self.spot_of(step, aid)
        self.go(spot, sid)
        pts = [self.loc(spot, p) for p in self.points(ref)]
        P = np.mean(pts, axis=0)
        with self.generator_posture():
            fr = inspect_motion(self.S, P, float(step.get("seconds", 2.0)))
        self._add(fr, spot, sid)

    def do_wait(self, step, sid):
        self._add(idle(float(step.get("seconds", 1.0))), self.at, sid)

    # ------------------------------------------------------------------ seams
    def _seams(self, n=8):
        """Where one step hands over to the next, each hand blends from where it was into the next target."""
        from .motion_gen import ease as e_
        bounds = [i for i in range(1, len(self.frames)) if self.frames[i].label.split(":")[0]
                  != self.frames[i - 1].label.split(":")[0]]
        for b in bounds:
            if self.frames[b - 1].base is not None or self.frames[b].base is not None:
                continue                                                   # walking frames: the mocap arms
            prev = {t.side: t for t in self.frames[b - 1].targets}
            for k in range(min(n, len(self.frames) - b)):
                u = e_((k + 1) / (n + 1))
                f = self.frames[b + k]
                cur = {t.side: t for t in f.targets}
                out = []
                for side in ("l", "r"):
                    a_, c_ = prev.get(side), cur.get(side)
                    if a_ is None and c_ is None:
                        continue
                    a_ = a_ or self.rest[side]
                    c_ = c_ or self.rest[side]
                    out.append(lerp_target(a_, c_, u))
                f.targets = out
        self._level_walk_pelvis()
        self._walk_blends()

    def _level_walk_pelvis(self):
        """The captured walks carry the recording skeleton's pelvis tilted back ~20 deg with the lower back bent forward
        to make up for it (the chest is upright): on a character's body that is a tucked-under seat. Per walk, the
        clip's mean tilt is taken off the pelvis and given back to the spine and the thighs, so the chest, the legs and
        the feet stay exactly where they were."""
        from .kinematics import axis_angle
        F = self.frames
        i = 0
        while i < len(F):
            if F[i].base is None:
                i += 1
                continue
            j = i
            while j + 1 < len(F) and F[j + 1].base is not None:
                j += 1
            run = [F[k].base for k in range(i, j + 1)]
            ups = [b.local("pelvis") @ np.array([0.0, 1.0, 0.0]) for b in run]
            tilt = float(np.mean([math.atan2(u[2], u[1]) for u in ups]))       # + forward, - back (rad)
            if abs(tilt) > math.radians(4.0):
                for b in run:
                    Rp = b.local("pelvis")
                    side = Rp @ np.array([1.0, 0.0, 0.0])                         # the pelvis' own left-right axis
                    C = axis_angle(side, -tilt)                                   # level it
                    Rn = C @ Rp
                    fix = Rn.T @ Rp                                               # children keep their world turn
                    for c in ("spine_01", "thigh_l", "thigh_r"):
                        b.rot[c] = fix @ b.local(c)
                    b.rot["pelvis"] = Rn
            i = j + 1

    def _walk_blends(self, n=10):
        """Standing still <-> walking: the first / last frames of a walk blend the captured body from / back to
        the standing pose, so the hips and arms do not jump where the clip starts or ends."""
        from .bake import blend
        stand = self.S.solve([self.rest["l"], self.rest["r"]], look=None,          # as it stands between steps
                             prev=rest_targets(self.S)[1]).pose                     # (from the free stand: quick)
        F = self.frames
        i = 0
        while i < len(F):
            if F[i].base is None:
                i += 1
                continue
            j = i
            while j + 1 < len(F) and F[j + 1].base is not None:
                j += 1
            m = min(n, (j - i + 1) // 2)
            if i > 0:
                for k in range(m):
                    F[i + k].base = blend(stand, F[i + k].base, ease((k + 1) / (m + 1)))
            if j < len(F) - 1:
                for k in range(m):
                    F[j - k].base = blend(stand, F[j - k].base, ease((k + 1) / (m + 1)))
            i = j + 1

    def smooth_solutions(self, sols, n=10):
        """After solving: where a walk starts or ends, ease the solved body from the standing frame next to it into
        the walk (and back), so nothing jumps at the hand-over between the captured walk and the solved work."""
        from .bake import blend
        F = self.frames
        out = list(sols)

        def put(i, ref, u):
            a = out[i]
            out[i] = type(a)(blend(ref, a.pose, u), a.cost, a.terms, a.angles, a.v)
        for i in range(1, len(F)):
            walk_in = F[i].base is not None and F[i - 1].base is None
            walk_out = F[i].base is None and F[i - 1].base is not None
            if walk_in:
                ref = sols[i - 1].pose
                for k in range(min(n, len(F) - i)):
                    if F[i + k].base is None:
                        break
                    put(i + k, ref, ease((k + 1) / (n + 1)))
            elif walk_out:
                ref = sols[i].pose
                for k in range(min(n, i)):
                    if F[i - 1 - k].base is None:
                        break
                    put(i - 1 - k, ref, ease((k + 1) / (n + 1)))
        return out

    # ------------------------------------------------------------------ scene
    def build_static(self, stage, root, lib):
        """The static assets as one geometry prim per asset under root."""
        from pxr import Tf, UsdGeom
        UsdGeom.Xform.Define(stage, root)
        for a in self.assets.values():
            s = Shapes()
            build_asset(s, a)
            if not s.groups:
                continue
            p = f"{root}/{Tf.MakeValidIdentifier(a['id'])}"
            UsdGeom.Xform.Define(stage, p)
            s.build(stage, f"{p}/Geom", lib.refs(list(s.groups)))


def default_task():
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(here, "..", "..", "..", "..", "docs", "people", "practice",
                                         "battery_box.toml"))
