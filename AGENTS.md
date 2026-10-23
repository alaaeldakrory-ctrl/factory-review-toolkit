# AGENTS.md: how an AI agent uses the Factory Review Toolkit (people & motion)

You are an agent working in someone's Omniverse / USD scene. You want workers doing manufacturing hand jobs at the
right places in that scene, with motion that has already been checked. This file is everything you need. Use the API
below; do not write joint angles or keyframes yourself, and do not re-tune the solver per scene.

## 0. Setup (once)

- The repository root is `<FRT>`: the folder holding `exts/`, `tools/`, `data/` and this file.
- **Python outside Kit:** Python 3.10-3.12 with `usd-core` and `numpy`. On an offline PC, install them from the
  bundle:

  ```
  pip install --no-index --find-links <FRT>/wheels usd-core numpy tomli
  ```

- **Inside Kit:**
  - Add `<FRT>/exts` to the extension search paths.
  - Enable `frt.core`. It needs no internet: its wheels come from `<FRT>/wheels`.
  - Then `from frt_core.people import api` works in the Script Editor or an extension.
- **Outside Kit:** run `sys.path.insert(0, "<FRT>/exts/frt.core")` first.
- **Smoke test:** `python <FRT>/tools/people_job.py --list` prints the job library as JSON.

## 1. The model you are working with

- **Workers.** Three bodies on one skeleton:
  - `W05F`: 5th percentile female, the shortest reach;
  - `W50M`: 50th percentile male;
  - `W95M`: 95th percentile male, the most likely to stoop.

  Real meshes come from `data/bodies`. If those are missing, a mannequin is used. `workers.custom(id, stature_cm)`
  gives any stature, drawn as a mannequin.
- **Jobs.** 50 checked recipes, listed in `docs/people/JOBS.md` or from `api.list_jobs()`. A recipe is a keyframed
  hand plan made of grips, tool paths, gaze and stance. It is not baked animation: the posture solver turns it into
  a full-body pose for each body and each placement.
  - Groups: fastening, electrical, assembly, operate, hydraulics, overhead / low, heavy, finishing.
  - `cylinder` is a two-person job.
- **Job frame.** Every job is authored here:
  - the worker stands at the origin, facing +Z;
  - Y is up, units are cm, 24 fps;
  - a bench top is at 92 cm with its front edge 25 cm ahead;
  - a wall panel face is at z = 42.

  `work_point` is roughly where the hands work.
- **Checks.** Every frame of every solve goes through the same checks:
  - joint limits;
  - pops (frame-to-frame jumps);
  - hand contact;
  - awkward postures (trunk, neck, elbow);
  - fixture collision (body inside the bench or panel).

  A result is `ok` only when nothing is flagged.
- **Robustness.** The library is stress-tested with work moved ±25 cm sideways, ±15 cm in height, -6 / +10 cm in
  depth, turned ±25° and tilted -15 / +20°, on all three bodies. With stepping allowed, 85 % of those cases pass.
  The rest are mostly real workplace problems ("too far", "too high" for that body), reported as `findings`.

## 2. The four calls

```python
from frt_core.people import api

api.list_jobs()                       # [{id, title, group, worker, fixture, work_point, partner}]

# 1. place by the scene's work point (world, Y-up cm) and which way the worker faces
at, dy = api.station_for("ratchet", work_point=(300, 100, 200), facing_yaw=90)

# 2. solve + check (work moved dy up / down to the real height; the worker may step to a better spot)
res = api.solve_job("ratchet", worker="W05F", work={"d": (0, dy, 0)})   # also "yaw" / "tilt" in degrees
res.ok, res.summary()                 # summary: counts, stepped_cm, findings, top_issues (JSON-friendly)

# 3. write a standalone USD (Y-up, cm, 24 fps; worker + tools + parts (+ bench/panel if fixture=True))
api.write_job_usd(res, "C:/scenes/ops/ratchet_W05F.usd", fixture=False)   # fixture=False: the scene has its own

# 4. reference it into the target scene: any up axis / units / fps; `at` in the scene's own units
api.add_to_scene(stage, "/World/People/Op1", "C:/scenes/ops/ratchet_W05F.usd",
                 at=(3.0, 2.0, 0.0), yaw=90.0, start_frame=120)
```

### Speed

A fresh solve takes 3-4 minutes per job, and up to about 20 minutes when the worker has to step. Plan for that and
run several in parallel processes. Results are cached in `data/solved`, so the same job, body and placement is
instant the next time. Every job on every body at its authored placement (`work=None`) ships already solved. Small
height changes (`dy`) are new placements and are solved fresh.

### Coordinates and timing in `add_to_scene`

- **`at` and `yaw`.**
  - `at` is where the job frame's origin (the worker's feet at the authored spot) goes, in the target stage's units
    and axes.
  - `yaw` turns the job about the stage's up axis.
  - In a Z-up stage the worker faces -Y at yaw 0 (the standard Y-up to Z-up turn), +X at yaw 90 and +Y at yaw
    180. In a Y-up stage the worker faces +Z at yaw 0.
- **`at` versus `station_for`.**
  - `station_for` works in the job's convention (Y-up cm, x / z on the floor).
  - In a Y-up cm scene, pass its result straight to `at=(x, 0, z), yaw=yaw`.
  - In a Z-up metre scene, convert first: `at=(x/100, ?, 0)`, and mind the axis swap. The simplest way is to work
    out the target point in the scene, then choose `at` and `yaw` so the worker's +Z (the job frame's forward)
    points at it.
- **Timing.** `start_frame` is in the target stage's frames. USD rescales 24 fps automatically. A job plays once and
  holds its last pose; the part stays where it was put.

### Without Kit, from the command line (prints JSON; exit code 2 when not ok)

```
python tools/people_job.py ratchet --worker=W05F --work-point=300,100,200 --facing=90 --out=C:/scenes/ops/r.usd --no-fixture
python tools/people_job.py ratchet --out=C:/scenes/ops/r.usd --scene=C:/scenes/cell.usd --prim=/World/People/Op1 --at=3,2,0 --yaw=90 --start=120
```

## 3. Choosing the job, body and placement

1. **Pick the job.** Match the task to `group` and `title`, for example "torque the lug nuts" → `lug_torque`, or
   "plug the harness" → `connector` or `deutsch`. If nothing fits, use the closest job and say so in your report.
   Do not invent a new recipe silently.
2. **Pick the body.** Default to the job's own worker. For a design review, run all three bodies: W05F shows
   over-reach, W95M shows stooping.
3. **Place it.**
   - Use `station_for` with the real work point.
   - Keep `dy` within ±15 cm and turns within ±25°. Beyond that, expect failures.
   - If the real work is much higher or lower, that is a finding about the workplace (platform, lift table). Do not
     force the job.
4. **Read the result.**
   - `res.ok` means the motion is clean. Use it.
   - `stepped_cm` non-zero means the worker stood somewhere else (dx left, dz forward). This is already applied in
     the USD.
   - `findings` (for example `{"kind": "too far", "cm": 12, "side": "r"}`) are ergonomic or layout problems. Report
     them to the user; they are results, not bugs to hide.
   - `top_issues` lists what the checks flagged (for example `awkward:elbow_r [n, worst]`).
   - A job that is not ok can still be written, but say so and name the issue.
5. **Two people.** `cylinder` writes both workers. Other jobs are one worker. Several jobs in one scene: one USD
   each, each `add_to_scene` at its own spot and start frame.

## 4. Walking, carrying and whole processes

To move workers between stations, use a **process file** instead of single jobs:
`docs/people/example_process.toml` (workers, stations, steps with catalog motion ids and world targets, `after`
dependencies). The planner:

- walks workers between stations (A* around equipment);
- uses captured clips for walking, turning and carrying (`data/mixamo`);
- waits for dependencies;
- keeps workers out of each other's way;
- bakes the result and checks the rules (RULA, NIOSH lifting, reach zones).

```
python tools/process_demo.py my_process.toml out/my_process.usd
```

The catalog motion ids are in `exts/frt.core/frt_core/people/data/motions.toml`. Process steps use the generic
generators (ratchet, torque, screwdriver, hammer, connector, cable tie, crimp, pick / place, carry, lift). For the
best hand detail at a station, use the 50 jobs above: walk with a process, then play the job at that station.

## 5. Rules for agents

- Do not edit `solver.py`, `check.py`, `jobs*.py` or `handkit.py` to make one scene pass. Those are shared and
  tested on 300 variants. Change the placement, the body or the workplace instead, and report it.
- Always report `res.summary()` (or the CLI JSON) with what you placed. Never claim a motion is clean when
  `ok` is false.
- Units: the job side is always Y-up cm. Do the conversion through `add_to_scene`, not by hand-scaling prims.
- Use red only for E-stops and fire equipment. Never colour equipment red or orange: in this plant red means a
  quality issue.
- Kimodo (an NVIDIA generative motion model) is optional and not part of the offline bundle. The rule-based solver
  is the default and needs nothing else.

## 6. Where things are

| What | Where |
|---|---|
| API (start here) | `exts/frt.core/frt_core/people/api.py` |
| CLI | `tools/people_job.py`, `tools/process_demo.py` |
| Job list | `docs/people/JOBS.md` |
| Design of the people module | `docs/people/DESIGN.md` |
| Job recipes | `exts/frt.core/frt_core/people/jobs.py`, `jobs_awp.py` |
| Solver and checks | `people/solver.py`, `people/check.py`, `people/placement.py` |
| Stress test (how robust is a recipe) | `tools/stress_test.py --only=<job>` |
| Scene builder from drawings (non-people) | `README.md` |
