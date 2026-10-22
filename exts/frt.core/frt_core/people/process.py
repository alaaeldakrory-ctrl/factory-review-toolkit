"""Process steps -> scheduled, animated workers.

A process is data (TOML): workers, stations (where a worker stands to work and which way they face) and steps.
Each step names a worker, a station, a catalog motion and the motion's targets in world coordinates:

    [[worker]]   id = "A"   body = "W50M"   start = [0, -150]   yaw = 0
    [[station]]  id = "bench"   spot = [0, -10]   yaw = 0
    [[step]]     id = "tighten"   worker = "A"   station = "bench"   motion = "fasten.ratchet.tighten"
                 target = [8, 99, 38]   after = ["place_bracket"]   strokes = 6

The planner walks each worker to the station of their next step (A* around the equipment), waits for the
steps it depends on, generates the motion from the catalog (pattern -> generator), and keeps workers from walking
into each other (a later walk waits until the way is clear). Everything is then solved and baked; the rules check
the result (rules.py).
"""
import math
from dataclasses import dataclass, field

import numpy as np

try:
    import tomllib
except ImportError:
    import tomli as tomllib

from . import motion_carry as C
from . import motion_gen as G
from . import motion_more as M
from .gait import turn_in_place, walk
from .motions import load as load_catalog
from .skeleton import Skeleton
from .solver import PostureSolver
from .workers import WORKERS

CLEARANCE = 65.0            # two workers' roots closer than this while one is walking = a near miss
WAIT_STEP = 12              # frames a walk is held back per conflict


@dataclass
class Segment:
    worker: str
    kind: str               # walk / work / wait
    step: str
    start: int
    frames: list
    spot: tuple             # (x, z, yaw) at the end of the segment

    @property
    def end(self):
        return self.start + len(self.frames)


@dataclass
class Plan:
    segments: dict = field(default_factory=dict)       # worker -> [Segment]
    step_end: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)
    length: int = 0


def load(path):
    with open(path, "rb") as f:
        return tomllib.load(f)


def _rot2(yaw):
    a = math.radians(yaw)
    return np.array([[math.cos(a), math.sin(a)], [-math.sin(a), math.cos(a)]])


def to_local(spot, p):
    """World point (x, y, z) -> the frame of a worker standing at spot (x, z, yaw)."""
    xz = _rot2(spot[2]).T @ (np.array([p[0], p[2]], float) - np.array(spot[:2], float))
    return (float(xz[0]), float(p[1]), float(xz[1]))


def dir_local(spot, d):
    xz = _rot2(spot[2]).T @ np.array([d[0], d[2]], float)
    return (float(xz[0]), float(d[1]), float(xz[1]))


def part_spec(step):
    size = [float(v) for v in step.get("size", (30, 15, 20))]
    return size, f"box:{size[0]:g}x{size[1]:g}x{size[2]:g}:{step.get('look', 'cardboard')}"


def lift_record(spot, pick, place_at, mass, two_hands):
    """NIOSH inputs in the worker frame: horizontal hand distance from the ankles and hand heights."""
    a, b = to_local(spot, pick), to_local(spot, place_at)
    return {"H1": math.hypot(a[0], a[2]), "V1": a[1], "H2": math.hypot(b[0], b[2]), "V2": b[1],
            "mass_kg": float(mass), "two_hands": two_hands}


def generate(cat, S, step, spot, notes, lifts=None):
    """Frames for one step at a worker spot (catalog pattern -> generator)."""
    m = cat.get(step["motion"])
    if m is None:
        notes.append(f"{step['id']}: unknown motion {step['motion']!r} - shown as a wait")
        return M.idle(float(step.get("seconds", 2.0)), label=f"{step['id']}:unknown")
    kind, cyc = m.generator
    side = step.get("hand", "r")
    tgt = lambda k="target": to_local(spot, step[k])                         # noqa: E731
    axis = dir_local(spot, step.get("axis", (0, 1, 0)))
    p = dict(m.params)
    p.update({k: v for k, v in step.items() if k in ("strokes", "strikes", "turns", "seconds")})
    if "size" in step and "pick" in step and "to" not in step:   # handle a real part (it stays where it is put)
        size, spec = part_spec(step)
        two = m.hands in ("two", "team")
        f1, _, _ = C.carry_pick(S, tgt("pick"), size, step.get("part", step["id"]), spec, two, side)
        f2 = C.carry_place(S, tgt("place"), size, step.get("part", step["id"]), spec, two, side)
        fr = f1 + f2
        if lifts is not None and "mass_kg" in step:
            lifts[step["id"]] = lift_record(spot, step["pick"], step["place"], step["mass_kg"], two)
    elif kind == "cycle" and cyc == "tie_pull":
        fr = C.cable_tie(S, tgt(), dir_local(spot, step.get("axis", (1, 0, 0))), side, step.get("part", step["id"]))
    elif kind == "cycle" and cyc == "squeeze":
        fr = C.crimp(S, tgt(), side, int(p.get("squeezes", 1)))
    elif kind == "cycle" and cyc in ("ratchet", "wrench_swing"):
        fr = G.ratchet(S, tgt(), axis, side, strokes=int(p.get("turns", p.get("strokes", 6))))
    elif kind == "cycle" and cyc == "torque_click":
        fr = M.torque(S, tgt(), axis, side)
    elif kind == "cycle" and cyc == "hammer":
        fr = G.hammer(S, tgt(), axis, side, strikes=int(p.get("strikes", 4)))
    elif kind == "cycle" and cyc in ("screwdriver", "power_drive"):
        fr = M.screwdriver(S, tgt(), dir_local(spot, step.get("axis", (0, 0, 1))), side,
                           turns=int(p.get("turns", 2 if cyc == "power_drive" else 6)))
    elif kind in ("reach_place", "carry") and "pick" in step:
        fr = M.pick_place(S, tgt("pick"), tgt("place"), side, two_hands=m.hands == "two",
                          width=float(step.get("width", 30.0)))
    elif kind == "operate" and m.object == "connector":
        fr = M.connector(S, tgt(), dir_local(spot, step.get("axis", (0, 0, 1))), side)
    elif kind in ("inspect", "operate", "reach_place", "body"):
        if kind != "inspect":
            notes.append(f"{step['id']}: no dedicated generator for {m.id} yet - shown as reach and hold")
        fr = M.inspect(S, tgt(), float(p.get("seconds", 2.0)), side)
    elif step["motion"].startswith("walk.wait"):
        fr = M.idle(float(p.get("seconds", 2.0)))
    else:
        notes.append(f"{step['id']}: no generator for pattern {m.pattern!r} yet - shown as a wait")
        fr = M.idle(float(p.get("seconds", 2.0)))
    for f in fr:
        f.label = f"{step['id']}:{f.label or 'work'}"
    return fr


class Planner:
    def __init__(self, proc, nav=None, catalog=None):
        self.proc = proc
        self.nav = nav
        self.cat = catalog or load_catalog()
        self.workers = {w["id"]: w for w in proc["worker"]}
        self.stations = {s["id"]: s for s in proc.get("station", [])}
        self.body = {wid: WORKERS[w.get("body", "W50M")] for wid, w in self.workers.items()}
        self.sk = {wid: Skeleton.for_worker(b) for wid, b in self.body.items()}
        self.solver = {wid: PostureSolver(self.sk[wid], self.body[wid]) for wid in self.workers}
        self._gen = {}                                    # step id -> frames (generated once)
        self.lifts = {}                                   # step id -> NIOSH inputs

    def spot(self, sid):
        s = self.stations[sid]
        return (float(s["spot"][0]), float(s["spot"][1]), float(s.get("yaw", 0.0)))

    def _work(self, step):
        if step["id"] not in self._gen:
            notes = []
            fr = generate(self.cat, self.solver[step["worker"]], step, self.spot(step["station"]), notes,
                          self.lifts)
            self._gen[step["id"]] = (fr, notes)
        return self._gen[step["id"]]

    def schedule(self, delays=None):
        delays = delays or {}
        plan = Plan({wid: [] for wid in self.workers})
        t = {wid: 0 for wid in self.workers}
        at = {wid: (float(w["start"][0]), float(w["start"][1]), float(w.get("yaw", 0.0))) for wid, w in self.workers.items()}
        done = {}
        pending = list(self.proc["step"])
        guard = 0
        while pending and guard < 10000:
            guard += 1
            step = next((s for s in pending if all(d in done for d in s.get("after", []))), None)
            if step is None:
                plan.notes.append("circular or missing dependencies: " + ", ".join(s["id"] for s in pending))
                break
            pending.remove(step)
            wid = step["worker"]
            spot = self.spot(step["station"])
            sk, body = self.sk[wid], self.body[wid]
            # walk there (held back by `delays` when the way was not clear)
            hold = delays.get(step["id"], 0)
            if hold:
                plan.segments[wid].append(Segment(wid, "wait", step["id"], t[wid], M.idle(hold / G.FPS,
                                                  label=f"{step['id']}:give way"), at[wid]))
                t[wid] += hold
            if math.dist(at[wid][:2], spot[:2]) > 5.0:
                path = self.nav.path(at[wid][:2], spot[:2]) if self.nav else [at[wid][:2], spot[:2]]
                fr = walk(body, sk, path, at[wid][2], spot[2])
                for f in fr:
                    f.label = f"{step['id']}:walk"
                plan.segments[wid].append(Segment(wid, "walk", step["id"], t[wid], fr, spot))
                t[wid] += len(fr)
            elif abs(((spot[2] - at[wid][2]) + 180) % 360 - 180) > 5:
                fr = turn_in_place(body, sk, spot[:2], at[wid][2], spot[2])
                plan.segments[wid].append(Segment(wid, "walk", step["id"], t[wid], fr, spot))
                t[wid] += len(fr)
            at[wid] = spot
            ready = max([done[d] for d in step.get("after", [])] + [t[wid]])
            if ready > t[wid]:                                    # wait for the steps this one depends on
                plan.segments[wid].append(Segment(wid, "wait", step["id"], t[wid],
                                                  M.idle((ready - t[wid]) / G.FPS, label=f"{step['id']}:wait"), spot))
                t[wid] = ready
            if "to" in step:                                      # pick here, walk carrying, place at `to`
                t[wid], at[wid] = self._carry(plan, step, t[wid], spot)
                done[step["id"]] = t[wid]
                continue
            fr, notes = self._work(step)
            plan.notes += [n for n in notes if n not in plan.notes]
            plan.segments[wid].append(Segment(wid, "work", step["id"], t[wid], fr, spot))
            t[wid] += len(fr)
            done[step["id"]] = t[wid]
        plan.step_end = done
        plan.length = max(t.values()) if t else 0
        return plan

    def _carry(self, plan, step, t, spot):
        wid = step["worker"]
        S, sk, body = self.solver[wid], self.sk[wid], self.body[wid]
        m = self.cat.get(step["motion"])
        two = (m.hands in ("two", "team")) if m else True
        side = step.get("hand", "r")
        size, spec = part_spec(step)
        part = step.get("part", step["id"])
        dest = self.spot(step["to"])
        key = step["id"] + "/carry"
        if key not in self._gen:
            f1, hold_t, Tc = C.carry_pick(S, to_local(spot, step["pick"]), size, part, spec, two, side)
            path = self.nav.path(spot[:2], dest[:2]) if self.nav else [spot[:2], dest[:2]]
            fw = walk(body, sk, path, spot[2], dest[2], carry=hold_t)
            for f in fw:
                f.tools[f"part:{part}"] = (spec, Tc)
            f3 = C.carry_place(S, to_local(dest, step["place"]), size, part, spec, two, side)
            for fr, lab in ((f1, "pick"), (fw, "carry"), (f3, "place")):
                for f in fr:
                    f.label = f"{step['id']}:{lab}"
            self._gen[key] = (f1, fw, f3)
            if "mass_kg" in step:
                cp = C.carry_pose(sk, size, two, side)
                rec = lift_record(spot, step["pick"], step["pick"], step["mass_kg"], two)
                rec2 = lift_record(dest, step["place"], step["place"], step["mass_kg"], two)
                self.lifts[step["id"] + ":pick"] = {**rec, "H2": math.hypot(cp[0], cp[2]), "V2": float(cp[1])}
                self.lifts[step["id"] + ":place"] = {**rec2, "H1": math.hypot(cp[0], cp[2]), "V1": float(cp[1])}
        f1, fw, f3 = self._gen[key]
        for kind, fr, sp in (("work", f1, spot), ("walk", fw, dest), ("work", f3, dest)):
            plan.segments[wid].append(Segment(wid, kind, step["id"], t, fr, sp))
            t += len(fr)
        return t, dest

    def roots(self, plan):
        """Per worker, per frame (x, z, yaw, walking?, walk origin, walk destination)."""
        out = {}
        for wid, segs in plan.segments.items():
            w = self.workers[wid]
            cur = (float(w["start"][0]), float(w["start"][1]), float(w.get("yaw", 0.0)))
            arr = [None] * plan.length
            k = 0
            for sg in segs:
                for i in range(k, sg.start):
                    arr[i] = (*cur, False, None, None)
                origin = cur
                for i, f in enumerate(sg.frames):
                    r = f.root if f.root is not None else sg.spot
                    arr[sg.start + i] = (*r, sg.kind == "walk", origin[:2], sg.spot[:2])
                cur = sg.spot
                k = sg.end
            for i in range(k, plan.length):
                arr[i] = (*cur, False, None, None)
            out[wid] = arr
        return out

    def conflicts(self, plan):
        R = self.roots(plan)
        ids = list(R)
        out = []
        for i in range(plan.length):
            for a in range(len(ids)):
                for b in range(a + 1, len(ids)):
                    ra, rb = R[ids[a]][i], R[ids[b]][i]
                    if not (ra[3] or rb[3]) or math.dist(ra[:2], rb[:2]) >= CLEARANCE:
                        continue
                    # stepping into / out of a spot beside a coworker who stands still is normal work, not a near miss
                    if not (ra[3] and rb[3]):
                        w = ra if ra[3] else rb
                        if min(math.dist(w[:2], w[5]), math.dist(w[:2], w[4])) < CLEARANCE:
                            continue
                    out.append((i, ids[a], ids[b], ra[3], rb[3]))
        return out

    def adjacent_spots(self):
        """Stations closer than the clearance (two people working shoulder to shoulder)."""
        ids = list(self.stations)
        return [(a, b) for i, a in enumerate(ids) for b in ids[i + 1:]
                if math.dist(self.spot(a)[:2], self.spot(b)[:2]) < CLEARANCE]

    def plan(self, max_iter=40):
        """Schedule, then give way: while two workers would meet, hold back the walk that started later."""
        delays = {}
        for _ in range(max_iter):
            p = self.schedule(delays)
            c = self.conflicts(p)
            if not c:
                p.notes.append(f"traffic: clear ({sum(delays.values()) / G.FPS:.1f} s of giving way)" if delays
                               else "traffic: clear")
                return p
            i, a, b, wa, wb = c[0]
            if not (wa and wb) and _ > 12:                            # the other one is not going to move away
                break
            # the walking worker whose walk started later gives way
            cand = []
            for wid, walking in ((a, wa), (b, wb)):
                if walking:
                    sg = next(s for s in p.segments[wid] if s.kind == "walk" and s.start <= i < s.end)
                    cand.append((sg.start, sg.step))
            step = max(cand)[1]
            delays[step] = delays.get(step, 0) + WAIT_STEP
        p.notes.append(f"traffic: {len(c)} conflict frames left after {max_iter} tries")
        return p

    def frames_of(self, plan, wid):
        out = []
        for sg in plan.segments[wid]:
            while len(out) < sg.start:
                out.append(G.Frame([], None, static=True, label="idle"))
            out += sg.frames
        while len(out) < plan.length:
            out.append(G.Frame([], None, static=True, label="idle"))
        return out
