# Solver validation on AWP-style jobs (2026-10-23)

Four tasks written as task files and run through the same pipeline the Task Builder uses: compile, solve, check, RULA.
- Three tasks use the AWP scene (`awp_scene.toml`), with generic shapes rather than real drawings.
- The fourth is the battery-box practice task on a different body.
- The worker is the NVIDIA Uniform_M character unless noted.

The table compares the first run with the run after the fixes listed below.
- **Limit:** frames × joints past their range.
- **Pop:** a joint jumping more than 32° between frames.

| Task | Work | Limit | Pop | RULA max | Notes |
|---|---|---|---|---|---|
| `scissor.toml` | battery terminals at 40 cm, connectors at 30 cm, harness ties at 21 cm | 443 → **27** | 35 → 22 | 6 → 5–6 | now kneels |
| `telehandler.toml` | 5 lug nuts on a hub at 51–73 cm | 67 → 105 | 40 → 49 | 5 → 6 | kneels; see finding 1 |
| `boom.toml` | control-box cover screws and connectors at 109–131 cm, jib bracket overhead at 172 cm | 252 → **113** | 97 → 81 | 5 | overhead left; see finding 3 |
| `battery_box_W05F.toml` | the practice task on the small female body | 689 → **161** | 63 → 64 | 4 | walks fixed; see finding 2 |
| `reference_task.toml` | the practice task, Uniform_M | 19 | 41 | 4 | unchanged |

Pictures, one per step: `img/sheet_<task>.jpg`.

## What was fixed
- **Kneeling for low work.** When a step's work points are below 30% of stature, the worker kneels on one knee, does the work, then gets up (`posture = "auto"`).
  - Before, the worker stooped with the feet fixed, the same fault as in the rack re-solve.
  - Kneeling: the front foot steps forward, then the back knee goes down with the toes tucked.
  - The worker stays down for consecutive low steps at the same spot.
  - The hands' rest position drops with the body.
  - `posture = "stand"` or `"kneel"` overrides the choice.
- **Work on any surface.** `hand_start` and `fasten` now handle work facing any direction: up, toward the worker (wall or hub), or down (overhead). A tag's `<name>_axis` gives the direction.
- **New verbs.**
  - `connect`: plug a connector into a socket; the free hand braces on the panel.
  - `cable_tie`: the free hand holds the harness.
  - The tie motion no longer rolls the wrist round the harness (the fingers feed the tie), and the harness direction is picked so the fingers point away from the body. Cable tie limit hits: 105 → 15.
- **Walk calibration for other skeletons.** The small female body's lower back is much more curved at rest than the captured walker's, which measured as 50° of spine bend on every walking frame. The trunk now keeps its own shape over the captured hips. The frt bodies W50M and W95M are unchanged.
- **New asset kinds.** A generic `fixture` (chassis, tray, platform, control box, bracket, optionally on legs) and a `wheel` (hub with tagged studs). Tagged studs, holes, sockets and harnesses are drawn from their tags.

## Findings
1. **Tools should be fetched before kneeling.**
   - On the telehandler, the worker kneels for the lug nuts, then reaches from the knee up to the ratchet on the cart (79 cm, 60 cm to the side). That reach causes the foot and wrist limits.
   - A real fitter picks up the tool first. The fix is a "tool in hand" state that lasts across steps: get the tool, then several fasten steps, then put it back. It would also save time.
2. **The small body's reach at the battery box.**
   - W05F threads the terminal nuts with the tray 30 cm to her right and the studs in front, and her right forearm runs out of range. W50M does not.
   - This is a layout finding (tray too far for a 5th-percentile woman), the kind of result the tool should report.
3. **The overhead jib at 172 cm** sits right at shoulder range for a 177 cm worker. Real practice would be a step platform or a lower bracket; the shoulder limit hits say the same.
4. **The remaining pops are mostly where motions join:** walking-clip seams, and the moments the tool changes hands. Neither is a pose error.

## Run it yourself
```bash
python tools/task_run.py docs/people/validation/scissor.toml
```

Results go to `out/people/task/<name>/` as `result.json` (per step: checks, joints, RULA, focus point) and `task.usda`.
