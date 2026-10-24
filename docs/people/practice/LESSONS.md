# Task Builder lessons

Do these in Window > Task Builder with `my_task.toml`, in order. Each one takes about 5 minutes plus a solve. After each solve, check three things:
- **Step colours:** green, yellow or red.
- **The numbers under each step:** time, RULA max, and what the checks found.
- **The animation:** click a step to jump there, then Play.

## 1. One step: a reach into a bin
- Add `transfer`: what = screw, count = 2, from = screw_bin, to = tray.screws. Then Solve.
- Notice that you never said where to stand. The worker walks from the start point to the shelf, because the bin's spot is `shelf`.
- Watch the hand: it opens as it reaches, closes on the screws, and opens over the tray.

## 2. A refused step
- Try `transfer` with what = nut, from = screw_bin. The window refuses it and says why.
- Try count = 5 into `tray.screws`, which has 4 slots. It is refused again.
- These checks are cheap: they run before anything is solved.

## 3. Things that move stay moved
- Add `carry`: what = tray, to = table.tray_place. Solve.
- The screws you dropped in the tray ride along with it, and stay on the table afterwards.
- Every step after this one finds the tray on the table, not on the shelf.

## 4. Order matters (no check for it yet)
- Add `hand_start` screw on `cover.holes` before carrying the cover to the table.
- Solve and watch: the worker walks to the cart and starts the screws there, where the cover lies.
- Move the step after the carry (^ v) and solve again.

## 5. Tool use
- Add `hand_start` nut on `battery_box.studs`, then `fasten` with the ratchet on the same studs. Use brace = battery_box.corner.
- The worker takes the ratchet from where it lies and does short strokes. The handle swings back while the nut keeps turning. Then the ratchet goes back.

## 6. Low work
- The AWP validation scene is `docs/people/validation/awp_scene.toml`. In `scissor.toml`, the battery terminals are 40 cm high.
- Without being told, the worker kneels: down on one knee, the work, then back up.
- Set `posture = "stand"` on a step to see the stooped version. Type it into the task file: the window has no field for it yet. The RULA difference is the argument for kneeling.

## 7. Same task, other body
- Change Worker to `W05F` (small female) or `W95M` (large male) and solve again.
- Steps that turn yellow or red on one body but not another are design findings (reach, heights), not animation bugs.

## The same file from an AI agent or a script
Everything the window does is in the task file. An agent writes the same file and runs:

```bash
python tools/task_run.py docs/people/practice/my_task.toml
```

It then reads `out/people/task/my_task/result.json`, which lists each step's checks and RULA score, the same numbers the window shows.
