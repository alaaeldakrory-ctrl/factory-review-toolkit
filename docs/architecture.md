# Architecture

## Pieces

| Piece | What | Depends on |
|---|---|---|
| `frt.core` (`frt_core`) | All logic: catalog, drawings, building, recipes, validation, findings, packaging, revision diff | pxr, numpy, ezdxf, pymupdf. No Kit, no UI |
| `frt.scene` (`frt_scene`) | Window > Factory Review: buttons and lists over `frt_core` | Kit (omni.ui, omni.usd, viewport utility) |
| `tools/` | CLI and offline previews over `frt_core` | usd-core, pillow |

Every feature lands in `frt_core` first, with a CLI entry and a test. The Kit window only calls it. The same code
therefore runs:
- headless, e.g. for CI or batch rebuilds;
- in Kit;
- later, in other Kit apps (USD Composer, Isaac Sim) unchanged.

## Data flow

1. `drawing.load(path)` reads a file into a **Drawing**: segments, polygons, block inserts and texts, in cm, cleaned.
2. `interpret(drawing, legend)` turns it into a **Plant**: outline, grid, doors, aisles, walkways, stations,
   equipment and unknowns. The Plant is plain dataclasses and is saved as `<scene>.plant.json`.
3. `recipe.build(recipe)` writes the scene:
   - `/World/Building`: the generated building.
   - `/World/FloorMarking`: geometry, plus Scopes per station, aisle and walkway.
   - `/World/Equipment/<ID>`: one per catalog item.
   - `/World/Lights` and `/World/Cameras`.
4. `validate(stage)` produces a **FindingList**, saved as `<scene>.findings.json`.
5. `revision.diff(old_plant, new_plant)` gives **Changes**, and `to_findings` turns them into review items.

## Conventions

- **Units and axes.** The scene is Y-up, cm, 30 fps. Plant coordinates are cm from the building's south-west corner,
  x east, y north. Scene X = x − W/2 and Z = D/2 − y, so north is −Z and the building is centred on the origin.
- **Tags.** Every placed item carries these attributes:
  - `frt:role` (table, lift_table, pallet_rack, crane_jib, station, ...);
  - `frt:id` (T3, R1, Station 5);
  - `frt:source` (drawing reference);
  - `frt:catalog`;
  - `frt:params` (JSON).

  Checks and tools work on tags, not prim names.
- **Catalog items.** The local frame has its origin at the footprint centre on the floor, +X along the length and
  +Z toward the front. `footprint(params)` must equal what the drawing shows, so the drawing rectangle is the item's
  envelope. Items return named attachment points (hook, beam levels, bins).
- **Findings.** Each finding has: check, severity (info / warning / critical), message, prims, location, view, value,
  limit, rule and reference.
- **Never overwrite an open scene.**
  - Builds go to `<scene>.building.usd` and are swapped in afterwards.
  - Kit marks open scenes with `<scene>.open`.
  - The Factory Review window closes the scene before rebuilding and reopens it afterwards.

## Planned modules (from the plan)

- M1 layout checks: aisle widths, clearances, flow lines.
- M2 safety: crane hook clearance, rack frontage, egress, forklift zones.
- M3 ergonomics: the existing RULA/spine pipeline moves in here.
- M4 vision: camera planner, video to posture.
- M5 characters and tasks.
- M6 flow / process animation without collisions.
- M7 robots.
- M8 reports.
- M9 assistant.
