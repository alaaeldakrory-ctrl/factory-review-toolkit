# People & Ergonomics module (`frt_core.people`)

Goal: put realistic workers into any toolkit scene and animate manufacturing work **without hand-keying people** -
natural, fully controllable motion generated from process steps, checked against ergonomics, efficiency and
traffic rules before anything is baked.

Lessons from the rail / rack assembly demos (kit-app-template/factory_sim/assembly) that this design removes:
every motion was scripted per scene; characters had different skeletons; Mixamo arms are ~10 cm short
(inflated trunk lean); awkward postures were found by eye after baking and fixed one at a time.

## 1. Workers (`people/skeleton.py`, `people/workers.py`)

- **One canonical skeleton** for every worker: the game-engine biped layout (pelvis, spine_01..03, neck_01, head,
  clavicle/upperarm/lowerarm/hand, five fingers x3, thigh/calf/foot/ball). Every motion is authored against it once and
  plays on every worker - no per-character retargeting. Mixamo / other mocap maps on by joint name.
- **Bodies**: MakeHuman / MPFB (Blender add-on; exports are CC0, so they ship in the portable library) exported once
  per worker with the game-engine rig, converted to USD (UsdSkel) by our importer. Until then a generated segmented
  mannequin on the same skeleton (portable, instant) stands in.
- **Anthropometry-driven proportions**: segment lengths from published percentile data (stature, shoulder, elbow,
  knuckle, eye heights, arm reach). Default crew:
  - `W05F` - 5th percentile female (short reach; the person most likely to over-reach)
  - `W50M` - 50th percentile male
  - `W95M` - 95th percentile male (most likely to stoop at low work heights)
- **Worker profile**: id, percentile / stature, handedness, PPE (vest, gloves, helmet, glasses), skills / tools
  qualified, walking speed, strength class (for NIOSH / push-pull limits).

## 2. Motion library (`people/data/motions.toml`, `people/motions.py`)

A motion is a **recipe**, not keyframes:

| field | meaning |
|---|---|
| `id`, `name`, `category` | e.g. `fasten.ratchet.tighten`, "Tighten bolt with ratchet", fastening |
| `hands` | `one`, `two`, `either`, `dominant+support`, `team` |
| `tool` / `object` | tool from the tool library; what it acts on (bolt, connector, part, cart, ...) |
| `postures` | allowed body postures: standing, kneeling, squatting, overhead, seated, lying, on_stand |
| `pattern` | the generator that makes the frames: `reach_place`, `cycle:<name>`, `route_path`, `walk`, `carry`, `team`, `operate`, `inspect` |
| `uas` | MTM-UAS building blocks (Get & Place, Place, Handle Tool, Operate, Motion Cycle, Fasten/Loosen, Body Motion, Visual Control) with their parameters (distance class, weight class, accuracy) |
| `params` | defaults (turns, strikes, force, length, ...) |
| `risks` | what the rule engine watches for this motion |

Times: from the MTM-UAS building blocks. MTM-UAS values are licensed data cards, so the toolkit ships
**estimates** and loads the plant's card (`~/.frt/mtm_uas.toml`) when present.

**How frames are made** - three layers, all on the canonical skeleton:
1. **Posture solver** (full-body IK with comfort priors): hands / tool tip to targets, feet planted or stepping,
   trunk flexion shared over the spine, elbows low, wrists near neutral, head to the point of regard, weight over
   the base of support. It picks the posture (stand / kneel / squat / step stand) the motion allows that scores best.
2. **Motion cycles**: rhythmic actions as parametric curves on top of a solved posture - ratchet arc, hammer swing,
   torque pull to click, crimp squeeze, tie pull, screwdriver turns, sanding strokes, wire push / route.
3. **Mocap layer**: walking, turning, starting / stopping, carrying (Mixamo clips mapped by name) phase-locked to
   planned footsteps (the gait planner from the assembly demos).

Transitions between motions are planned (approach, settle, act, release, return), so chains of motions look
continuous; the free hand gets a sensible job (brace, hold part, rest) instead of hanging.

## 3. Process steps -> animation (`people/process.py`)

A process is data: steps with worker, station / location, part, tool, motion id(s), predecessor steps.
The planner resolves targets in the scene (bolts, connectors, rack lanes, crane hook ...), plans walks and hand-offs,
schedules workers on a shared timeline, runs the rules, then bakes UsdSkel animation (plus tool / part motion).

## 4. Rules (`people/rules/`) - checked before baking, reported as findings

- **Ergonomics**: RULA / REBA per frame, NIOSH lifting equation, push / pull limits, sustained awkward postures
  (time over trunk / neck / shoulder thresholds), kneeling and overhead time per shift, repetition.
- **Efficiency (MTM / lean)**: reach zones (primary / secondary / out of reach), walking distance per cycle,
  unnecessary regrasps / transports, idle or holding hands, two-handed work balance, tool pick-up / put-down
  repeats, waiting.
- **Safety & traffic**: space-time reservation of floor cells so two workers never occupy the same spot; walkway
  use; no standing under a suspended crane load; line of fire; keep-out zones around moving equipment.
- **Feasibility**: reachability for the worker percentile, clearance (body vs scene), tool access.

## Build order

1. Skeleton + three anthropometric workers (mannequin bodies) + posture solver + pose lab (offline renders).
2. Motion catalog (data) + loader + validation; first generators: reach_place, walk / carry (mocap), ratchet,
   hammer, connector push, cable tie, crimp.
3. MakeHuman / MPFB bodies (needs Blender + MPFB, one-time export).
4. Process -> schedule -> bake; first rules (RULA / REBA, reach zones, traffic reservation).
5. Grow the generators until every catalog motion has one; NIOSH, push / pull, time standards card.
