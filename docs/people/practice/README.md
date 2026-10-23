# Practice: author a task in the Task Builder

The practice scene is a small battery-box cell:

| Asset | Kind | Tags (named points) |
|---|---|---|
| `screw_bin`, `nut_bin` | bins on the shelf | `grab` (where the fingers go in); they hold screws / nuts |
| `tray` | kitting tray on the shelf | `screws`, `nuts` (compartment slots) |
| `cover` | plate on the cover cart | `holes` (4 screw holes) |
| `table` | work table | `tray_place`, `cover_place` |
| `battery_box` | the work, on the table | `studs` (2 terminal studs), `corner` (where the free hand rests) |
| `screwdriver`, `ratchet` | tools lying on the table | - |
| `cabinet` | wall cabinet, door open | its door and latch |

Files:
- `battery_box_scene.toml`: the scene (assets and tags). No steps.
- `my_task.toml`: your task, still empty. The window writes it.
- `reference_task.toml`: my version, for comparison afterwards.

## In Kit

1. **Open the window.** Window > Task Builder opens `my_task.toml`. Show scene builds the scene with the worker standing at the start.
2. **Add a step.** Choose a verb, then fill its fields. Each field only offers what fits from the scene:
   - **Transfer**: what = screw/nut, from = a bin, to = tray slots.
   - **Carry**: a tray or plate, and a place to put it.
   - **Hand_start**: nut or screw, onto studs or holes.
   - **Fasten**: a tool, onto the started studs or holes.
   - **Push_door, latch**: the cabinet.
   - **Inspect**: something to look at.

   Use viewport selection fills a field from what you clicked in the viewport.
3. **Edit the list.** Use ^ v x to reorder or remove steps. Update selected replaces the step you clicked.
4. **Solve and preview.** This runs the solver in the background, about 2-5 minutes for a 2-minute task, then opens the result. Each step then shows:
   - its time;
   - the RULA maximum;
   - what the checks found (limit = a joint past its range, pop = a jump between frames);
   - a colour.

   Click a step to jump the timeline there.

You never say where the worker stands or when to walk. Every asset has a spot, so the worker walks there by itself.

## Mistakes to try on purpose

- **Hand-start screws before the cover is on.** The holes belong to the cover, so the screws go where the cover lies, on the cart.
- **Fasten before hand_start.** The tool turns, but there is nothing started to drive.
- **Transfer 5 screws into a 4-slot compartment.** You get an error message, not a broken animation.
- **Latch before push_door.** It runs, but on an open door. There is no rule for that yet.

The same task file can be written by an AI agent or a script. The window is just one way to write it:

```bash
python tools/task_run.py docs/people/practice/my_task.toml
```
