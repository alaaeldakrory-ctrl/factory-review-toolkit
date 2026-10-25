"""People API: put a checked, animated worker doing one of the library jobs into any scene.

This is the entry point for scripts and AI agents (see docs/people/AGENT_GUIDE.md). Four calls:

    from frt_core.people import api
    api.list_jobs()                                    # [{id, title, group, worker, fixture, work_point, ...}]
    res = api.solve_job("ratchet", worker="W05F",      # solve + check; the work can be moved / turned / tilted
                        work={"d": (0, 10, 0)})        #   relative to the authored station (cm, deg)
    api.write_job_usd(res, "C:/out/ratchet_W05F.usd")  # standalone USD: Y-up, cm, 24 fps, default prim /World
    api.add_to_scene(stage, "/World/People/Op1", "C:/out/ratchet_W05F.usd",
                     at=(x, y, z), yaw=90.0, start_frame=120)   # reference it into your scene (any up axis / units)

Job frame (every job is authored in it): the worker stands at the origin facing +Z, Y up, cm. A bench top is at
92 cm with its front edge 25 cm ahead; a wall panel face at z = 42. `work_point` in list_jobs() is roughly where the
hands work, so a scene can place a job by its work: api.station_for(job, work_point_world, facing_yaw).

Nothing here needs Kit: it runs in Kit's Python or any Python with pxr (usd-core) and numpy.
"""
import math
import os

import numpy as np

FPS = 24.0


def _rot(axis, deg):
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    th = math.radians(deg)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K


def work_transform(focus, d=(0.0, 0.0, 0.0), yaw=0.0, tilt=0.0):
    """4x4 that moves the work about `focus` (job frame): shift d (x left, y up, z away from the worker, cm), turn
    yaw (deg, about Y) and tilt (deg; > 0 tips the work face back / up)."""
    R = _rot((0, 1, 0), yaw) @ _rot((1, 0, 0), -tilt)
    c = np.asarray(focus, float)
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = c + np.asarray(d, float) - R @ c
    return M


def list_jobs():
    """Every job in the library as plain data (JSON-friendly)."""
    from .jobs import JOBS
    out = []
    for j in JOBS:
        out.append({"id": j.id, "title": j.title, "group": j.group, "worker": j.worker,
                    "fixture": getattr(j.fixture, "kind", "") or "custom",
                    "work_point": [round(float(v), 1) for v in j.focus],
                    "partner": j.partner[0] if j.partner else None})
    return out


def station_for(job_id, work_point, facing_yaw=0.0):
    """Where to stand the job's station so its work lands on `work_point` (world, Y-up cm) with the worker facing
    `facing_yaw` (deg about Y, 0 = +Z). Returns (at=(x, z, yaw), dy): dy is how much higher (+) the real work is than
    the authored one - pass it as work={"d": (0, dy, 0)} to solve_job (the library holds up well to about +-15 cm)."""
    from .jobs import BY_ID
    f = np.asarray(BY_ID[job_id].focus, float)
    R = _rot((0, 1, 0), facing_yaw)
    p = np.asarray(work_point, float)
    o = p - R @ np.array([f[0], 0.0, f[2]])
    return (float(o[0]), float(o[2]), float(facing_yaw)), float(p[1] - f[1])


class JobResult:
    """A solved job: frames + solutions for the worker (and the partner), the check report, the standing spot."""

    def __init__(self, job, worker, M, frames, sols, report, spot, findings, partner=None):
        self.job, self.worker, self.M = job, worker, M
        self.frames, self.sols, self.report, self.spot = frames, sols, report, spot
        self.findings, self.partner = findings, partner
        self.character = None                  # a character.Character when solved for a real character

    @property
    def ok(self):
        return bool(self.report["ok"]) and (self.partner is None or bool(self.partner["report"]["ok"]))

    def summary(self):
        """JSON-friendly result: pass / fail, issue counts, where the worker stepped to, reach findings."""
        top = {}
        for it in self.report["issues"]:
            top.setdefault(f"{it[1]}:{it[2]}", []).append(it[3])
        s = {"job": self.job.id, "worker": self.worker.id, "ok": self.ok, "frames": len(self.frames),
             "seconds": round(len(self.frames) / FPS, 2), "counts": dict(self.report["counts"]),
             "stepped_cm": [round(v, 1) for v in self.spot], "findings": self.findings,
             "top_issues": {k: [len(v), round(max(v), 1)] for k, v in
                            sorted(top.items(), key=lambda kv: -len(kv[1]))[:5]}}
        if self.partner is not None:
            s["partner"] = {"worker": self.partner["worker"].id, "ok": bool(self.partner["report"]["ok"]),
                            "counts": dict(self.partner["report"]["counts"])}
        return s


CACHE = os.environ.get("FRT_PEOPLE_CACHE") or os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "data", "solved"))


_CODE = []


def _code_hash():
    """Hash of the files that shape a solve: a cached result is only reused while they are unchanged."""
    import hashlib
    if not _CODE:
        h = hashlib.sha1()
        here = os.path.dirname(__file__)
        for n in ("solver.py", "check.py", "jobs.py", "jobs_awp.py", "handkit.py", "placement.py", "kinematics.py",
                  "skeleton.py", "workers.py", "motion_gen.py", "motion_carry.py", "bake.py", "character.py"):
            with open(os.path.join(here, n), "rb") as f:
                h.update(f.read().replace(b"\r\n", b"\n"))
        _CODE.append(h.hexdigest()[:8])
    return _CODE[0]


def _cache_key(job_id, w, M, step):
    import hashlib
    m = ",".join(f"{v:.2f}" for v in np.asarray(M, float).flatten())
    tag = f"{m}|{step}|{w.dims.get('stature', 0):.1f}|{_code_hash()}"
    return f"{job_id}_{w.id}_{hashlib.sha1(tag.encode()).hexdigest()[:10]}"


def solve_job(job_id, worker=None, work=None, step=True, cache=True):
    """Generate, solve and check one job (a few minutes; up to ~20 when the worker has to step). Results are cached
    in data/solved (FRT_PEOPLE_CACHE), so the same job / body / placement is instant the next time; the offline
    bundle ships every job on every body at its authored placement already solved.
    worker: "W05F" / "W50M" / "W95M", a Worker from workers.custom(), or a real character (character.load(asset)):
            the job is then solved on that character's own proportions and baked onto its rig; default: the
            job's own worker.
    work: None, a 4x4 matrix in the job frame, or {"d": (dx, dy, dz), "yaw": deg, "tilt": deg} about the work point.
    step: when the authored spot fails the checks, let the worker step to a better spot (reported in .spot)."""
    from .jobs import BY_ID
    from .skeleton import Skeleton
    from .solver import PostureSolver
    from .workers import WORKERS
    job = BY_ID[job_id]
    ch = worker if hasattr(worker, "skeleton") and hasattr(worker, "write") else None     # a real character
    if ch is not None:
        w, sk = ch.worker, ch.skeleton
    else:
        w = WORKERS[worker or job.worker] if not hasattr(worker, "id") else worker
        sk = Skeleton.for_worker(w)
    S = PostureSolver(sk, w)
    if work is None:
        M = np.eye(4)
    elif isinstance(work, dict):
        M = work_transform(job.focus, **work)
    else:
        M = np.asarray(work, float)
    path = os.path.join(CACHE, _cache_key(job_id, w, M, step) + ".pkl") if cache else None
    if path and os.path.isfile(path):
        import pickle
        try:
            with open(path, "rb") as f:
                d = pickle.load(f)
            r = JobResult(job, w, M, d["frames"], d["sols"], d["report"], d["spot"], d["findings"], d["partner"])
            r.character = ch
            return r
        except Exception:
            pass                                                # stale cache (code changed): solve again
    res = _solve(job, w, sk, S, M, step)
    res.character = ch
    if path:
        import pickle
        os.makedirs(CACHE, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"frames": res.frames, "sols": res.sols, "report": res.report, "spot": res.spot,
                         "findings": res.findings, "partner": res.partner}, f)
    return res


def clear_feet(sk, frames, field, step=1.0, most=24.0):
    """Planted feet placed (by a job's stance) with the toes in the fixture are drawn back until they are clear:
    the feet are fixed in the solve, so a foot in a panel would otherwise hold the whole body against it."""
    from .collide import TOL
    from .solver import ry
    done = set()
    for f in frames:
        st = f.stance
        if st is None or id(st) in done or getattr(st, "free", False):
            continue
        done.add(id(st))
        for s_ in ("l", "r"):
            a = getattr(st, "left" if s_ == "l" else "right")
            if a is None:
                continue
            a = np.asarray(a, float)
            Rf = ry(st.foot_yaw(s_))
            toe = Rf @ (sk.world[f"ball_{s_}"] - sk.world[f"foot_{s_}"])
            fwd = Rf @ np.array([0.0, 0.0, 1.0])
            pts = np.array([a, a + toe * 0.5, a + toe, a + toe + fwd * 5.0])
            pts[:, 1] = np.maximum(pts[:, 1], 3.0)
            back = 0.0
            while back < most and np.any(4.0 - field.distance(pts - fwd * back) - TOL > 0.0):
                back += step
            if back > 0.0:
                setattr(st, "left" if s_ == "l" else "right", tuple(a - fwd * back))


def _solve(job, w, sk, S, M, step):
    from .bake import solve_frames
    from .check import check_frames
    from .placement import diagnose, move_rules, move_work, solve_placed
    from .skeleton import Skeleton
    from .solver import PostureSolver
    from .workers import WORKERS
    job_id = job.id
    frames = job.frames(S)
    rules = job.keep_out_rules()
    if not np.allclose(M, np.eye(4)):
        frames = move_work(frames, M)
        rules = move_rules(rules, M)
    from .collide import fixture_field
    S.field = fixture_field(job, M)                               # the body (legs and feet too) out of the fixture
    if S.field is not None:
        clear_feet(sk, frames, S.field)
    if step:
        frames, sols, rep, spot, _ = solve_placed(S, frames, rules)
    else:
        S.keep_out = rules
        sols = solve_frames(S, frames, log=lambda *a: None)
        rep = check_frames(sk, frames, sols, job_id, keep_out=S.keep_out, field=S.field)
        spot = (0.0, 0.0)
    findings = diagnose(S, frames, rep, sols) if rep["counts"] else []
    partner = None
    if job.partner:
        wid2, place, build2 = job.partner
        w2 = WORKERS[wid2]
        sk2 = Skeleton.for_worker(w2)
        S2 = PostureSolver(sk2, w2)
        f2 = build2(S2, sk2, place)
        if not np.allclose(M, np.eye(4)):                       # the partner's frames are in its own spot's frame
            off = np.eye(4)
            off[:3, 3] = (place[0], 0.0, place[1])
            f2 = move_work(f2, np.linalg.inv(off) @ M @ off)
        s2 = solve_frames(S2, f2, log=lambda *a: None)
        partner = {"worker": w2, "place": place, "frames": f2, "sols": s2,
                   "report": check_frames(sk2, f2, s2, job_id + "/partner")}
    return JobResult(job, w, M, frames, sols, rep, spot, findings, partner)


def bake_into(stage, path, res, lib=None, fixture=True, start=0):
    """Bake a solved job under `path` (an Xform in a Y-up, cm stage; the job frame = that prim's frame):
    <path>/Fixture (optional, moved with the work), <path>/<worker id> (+ /Tools), <path>/Parts, partner."""
    from pxr import Gf, UsdGeom

    from ..geom import Shapes
    from ..materials import MaterialLibrary
    from .bake import bake
    lib = lib or MaterialLibrary(stage, "/World/Looks")
    UsdGeom.Xform.Define(stage, path)
    if fixture:
        s = Shapes()
        res.job.fixture(s)
        fx = UsdGeom.Xform.Define(stage, f"{path}/Fixture")
        fx.AddTransformOp().Set(Gf.Matrix4d(*res.M.T.flatten().tolist()))
        s.build(stage, f"{path}/Fixture/Geom", lib.refs(list(s.groups)))
    UsdGeom.Xform.Define(stage, f"{path}/Parts")
    from pxr import Tf
    bake(stage, f"{path}/{Tf.MakeValidIdentifier(res.worker.id)}", res.worker, res.frames, lib, sols=res.sols,
         start=start, root=(res.spot[0], res.spot[1], 0.0), parts_root=f"{path}/Parts", log=lambda *a: None,
         character=res.character)
    if res.partner is not None:
        p = res.partner
        bake(stage, f"{path}/{p['worker'].id}_2", p["worker"], p["frames"], lib, sols=p["sols"], start=start,
             root=tuple(p["place"]), parts_root=f"{path}/Parts", log=lambda *a: None)
    end = start + max(len(res.frames), len(res.partner["frames"]) if res.partner else 0) - 1
    if stage.GetEndTimeCode() < end:
        stage.SetEndTimeCode(end)
    return end


def write_job_usd(res, out_path, fixture=True):
    """A standalone USD of the solved job (Y-up, cm, 24 fps, default prim /World; job frame = /World/Job)."""
    from pxr import UsdGeom

    from .. import units as U
    from ..materials import MaterialLibrary
    from ..usdio import safe_stage
    if isinstance(res, str):
        res = solve_job(res)
    with safe_stage(out_path) as st:
        U.setup_stage(st, fps=FPS)
        st.SetDefaultPrim(UsdGeom.Xform.Define(st, "/World").GetPrim())
        lib = MaterialLibrary(st, "/World/Job/Looks")         # inside the job prim: one reference carries all
        st.SetStartTimeCode(0)
        bake_into(st, "/World/Job", res, lib, fixture=fixture)
        import json
        st.GetRootLayer().customLayerData = {"frt_summary": json.dumps(res.summary())}     # the check result, as JSON
    return out_path


def add_to_scene(stage, prim_path, usd_path, at=(0.0, 0.0, 0.0), yaw=0.0, start_frame=0.0):
    """Reference a job USD (from write_job_usd) into any stage at `at` (in that stage's own units and up axis),
    turned `yaw` degrees about the stage's up axis, starting at `start_frame` of the stage's timeline.
    Units, up axis (Y or Z) and frame rate are converted; the job's own frame (worker at origin facing +Z, Y up)
    is put at `at`. Z-up stage: the worker faces -Y at yaw 0, +X at 90, +Y at 180 (Y-up: +Z at 0)."""
    from pxr import Gf, Sdf, Usd, UsdGeom
    mpu = UsdGeom.GetStageMetersPerUnit(stage) or 0.01
    zup = UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.z
    xf = UsdGeom.Xform.Define(stage, prim_path)
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(*at))
    if zup:
        xf.AddRotateZOp().Set(float(yaw))
        xf.AddRotateXOp().Set(90.0)                            # job Y-up -> stage Z-up (job +Z -> stage -Y)
    else:
        xf.AddRotateYOp().Set(float(yaw))
    s = 0.01 / mpu
    if abs(s - 1.0) > 1e-9:
        xf.AddScaleOp().Set(Gf.Vec3d(s, s, s))
    inner = stage.DefinePrim(f"{prim_path}/Job")
    src = Usd.Stage.Open(usd_path)
    src_tcps = src.GetTimeCodesPerSecond()
    dst_tcps = stage.GetTimeCodesPerSecond()
    # USD rescales a referenced layer's time by timeCodesPerSecond itself; the offset is in the stage's frames
    inner.GetReferences().AddReference(Sdf.Reference(usd_path.replace("\\", "/"), "/World/Job",
                                                     Sdf.LayerOffset(float(start_frame))))
    length = (src.GetEndTimeCode() - src.GetStartTimeCode()) * dst_tcps / src_tcps
    end = float(start_frame) + length
    if stage.GetEndTimeCode() < end:
        stage.SetEndTimeCode(end)
    return {"prim": prim_path, "start": float(start_frame), "end": end, "scale": s, "z_up": zup}
