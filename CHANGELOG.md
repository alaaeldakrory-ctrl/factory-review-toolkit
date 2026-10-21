# Changelog

## 0.1.0 (2026-10-21)

First working slice: Phase 0 (foundation) and the start of Phase 1 (drawings to a reviewed factory).

### Core (`frt.core`)
- Scene conventions:
  - Y-up, cm, 30 fps.
  - Plant coordinates: cm from the building's south-west corner, x east, y north.
- Material library:
  - OmniPBR looks with a UsdPreviewSurface fallback, RAL powder-coat colours and textures from the NVIDIA base
    materials.
- Geometry:
  - Batched shape builder: one mesh per material.
  - Unit-safe asset placement that wraps any asset into the scene's units and up-axis.
- Tags, findings, rule packs, validator.
  - Validator checks: units, floating/sunk, overlaps (cranes exempt), sizes, missing and absolute assets.
- Safe writing:
  - Builds into `<scene>.building.usd`, then swaps it in.
  - Cached layers are reloaded after the swap.
  - Refuses to replace a scene another app has open (`<scene>.open` marker).
- Packager:
  - Mirrors the source folder layout, so relative paths inside asset packs keep working.
  - UDIM tiles and MDL texture references are followed.
  - Replaces `UsdUtils.LocalizeAsset`, which broke nested-layer paths.
- Layout checks on the plan:
  - aisle and walkway widths;
  - equipment in aisles or walkways;
  - columns in aisles, with the clear width beside them;
  - walkway separation from forklift aisles;
  - rectilinear exit travel distance to personnel doors;
  - items crossing their station's marking.
- Revision diff:
  - Equipment by tag, stations by name, doors, building size.
  - Results can become findings with views.

### Drawings
- DXF import:
  - Units, cleaning (strays, duplicates, gap snapping).
  - Block extents taken from geometry only.
  - Circles and arcs.
- PDF import:
  - Vector paths, scale read from "1:N" or given.
  - Colours and line widths kept.
  - Curves flattened, so circles become polygons.
- Two-point alignment and move-to-origin.
- Interpreter:
  - Outline from the perimeter column grid; column grid; doors and docks; aisles; walkways.
  - Stations, named inside or beside the rectangle.
  - Equipment from blocks (DXF) or labelled shapes (PDF), with whole-word label matching where legend order sets
    priority.
  - IDs taken from nearby tags.
- Default legend (TOML), so a plant's own CAD standard is a copy-and-edit away.

### Building
- Generator on NVIDIA Warehouse_A modules:
  - Stretched to the real column grid.
  - Dock and roll-up variants placed where the doors are.
  - Generated personnel door sets.
- Floor marking:
  - Aisles (yellow) and walkways (green with chevrons).
  - Stations: white borders and a yellow plate with the station number in seven-segment digits.

### Catalog (11 items)
- Industrial table, scissor lift table, carton flow rack, belt/roller conveyor, selective pallet racking (single or
  double), bin rack, pillar jib crane, freestanding bridge crane.
- Pallet with boxes, crate, forklift (from the NVIDIA packs).

### Kit (`frt.scene`)
- Window > Factory Review:
  - Scene Builder.
  - Review (findings with Go).
  - Stations (Go).
  - New drawing revision (compare, update).
  - Package.
- Pinned pip dependencies installed with `--no-deps`, so Kit's own numpy is never shadowed.

### Samples and tools
- Generic "Demo Plant": a 60 x 40 m lift assembly hall, laid out with its aisles between the column lines.
  - Two lines, seven stations, three pallet rack rows.
  - Bin racks, jib cranes, a bridge crane, a kitting conveyor and a receiving area.
  - Rev A and rev B, each as DXF and PDF. Rev B stages a pallet in the main aisle, which the layout check catches.
- CLI: build, drawing, diff, validate, package, catalog, preview.
- Offline previewer and catalog sheet.
- 47 pytest tests.
