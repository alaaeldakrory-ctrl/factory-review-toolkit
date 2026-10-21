# NVIDIA Warehouse_A modules: measured facts

Source: `Warehouse_NVD/Assets/DigitalTwin/Assets/Warehouse/Modular/Warehouse_A/`.

File names follow the pattern `Warehouse_<variant>_<height>_<kind>[Mirror]_01.usd`:

| Part | Values |
|---|---|
| Height | `H10m` (1000 cm tile, 10 m clear) or `H450cm` (500 cm tile) |
| Kind | `Center`, `CenterMirror`, `Straight`, `StraightMirror`, `Straight90`, `CornerIn`, ... |

## Frame

- Z-up, cm.
- A tile spans x 0..T and y −T..0, where T is the tile size.
- `Straight` carries its wall just outside the x = 0 edge.
- `Straight90` carries its wall just outside the y = 0 edge.
- `CornerIn` carries both.
- Dock modules extend outside the wall line.

Placing a module in a bay (scene Y-up), with ops in this order:

```
translate(bay centre) · scale(sx, 1, sz) · rotateY(turn) · translate(−T/2, 0, −T/2) · rotateX(−90)
```

Here `sx` and `sz` are the bay's real size divided by T, so a 12 m bay is stretched 1.2×.

| Wall on side | Kind | turn |
|---|---|---|
| W | Straight | 0 |
| E | Straight | 180 |
| N | Straight90 | 0 |
| S | Straight90 | 180 |

| Corner | turn |
|---|---|
| W+N | 0 |
| N+E | −90 |
| E+S | 180 |
| S+W | 90 |

## Wall variants used

| Use | Variants |
|---|---|
| Plain | A04 (braced panel), A01 (panel with electrics), A12 (small window) |
| Windows | A05 |
| Roll-up door | A09 (medium), A02 (large), A10 (double sectional), A08 (high-speed) |
| Dock | A03 (single), A07 (with leveler), A11 (double), A06 (with trailer) |

The pack has no personnel-door module, so the generator adds its own door sets: frame, leaf, handle and exit sign.

## Materials

The modules bind MDL materials from the pack: concrete floor, painted metal and so on. They also author magenta
display colours as placeholders. Only offline previews see those; RTX uses the MDLs.
