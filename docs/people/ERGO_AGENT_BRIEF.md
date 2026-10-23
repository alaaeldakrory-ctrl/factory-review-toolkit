# Brief for an AI agent: ergonomic assessment (RULA) of the operators in a scene

Give this whole file to the agent.

---

## Your task

Assess the ergonomics of every operator in the scene `<SCENE>` (about 15 animated people) with RULA. Produce the
report and dashboards, put the risk markers into the scene, and tell me which operators and moments need changing
and why.

You are not building the assessment. The Factory Review Toolkit (`<FRT>`, the folder with `exts/`, `tools/` and
`AGENTS.md`) already has it:

- `tools/ergo_scan.py` (command line);
- `frt_core.people.ergo_scan` / `ergo_dash` (Python).

Use them as they are. Do not write your own angle maths or RULA tables, and do not edit the toolkit to change a
score.

## Steps

1. **Setup (once).**
   - Outside Kit: Python 3.12 with `usd-core`, `numpy` and `pillow`. On an offline PC run
     `pip install --no-index --find-links <FRT>/wheels usd-core numpy pillow`.
   - Inside Kit: enable the `frt.core` extension (search path `<FRT>/exts`). Or run
     `sys.path.insert(0, r"<FRT>/exts/frt.core")` first.

2. **Find the operators.**

   ```
   python <FRT>/tools/ergo_scan.py "<SCENE>" --list
   ```

   This prints every character found, its name, and how its joints were recognised.
   - Check the count matches the operators you expect.
   - If an operator shows `missing` joints, its rig uses unusual names. Write a `mapping.json` like
     `{"Operator_07": {"pelvis": "Bip_Pelvis", "chest": "Bip_Spine2", ...}}`, using the role names printed by
     `--list`, and pass `--mapping=mapping.json`.
   - If the names are only the character asset's own name (several the same), that is fine: the tool adds the
     parent prim's name.

3. **Gather what the skeleton cannot show.** Loads and repeated work don't appear in a skeleton.
   - Read the scene, its process description, or ask me: which operators carry or hold weight (kg), and which do
     repeated cycles (ratcheting, drilling, more than ~4 times a minute)?
   - Pass them as `--loads=Operator_03:12,Operator_07:5` and `--repeated=Operator_02,Operator_09`.
   - If you don't know, leave them out and say so in the summary. Do not guess weights.

4. **Run the scan.**

   ```
   python <FRT>/tools/ergo_scan.py "<SCENE>" --out="<OUT>" --step=2 [--loads=...] [--repeated=...] [--range=start,end]
   ```

   - `--step=2` scores every second frame, which is plenty. Use `--range` to limit it to the shift or cycle you
     care about.
   - Inside Kit, use the open stage:

     ```python
     import omni.usd
     from frt_core.people import ergo_scan
     st = omni.usd.get_context().get_stage()
     res = ergo_scan.scan(st, "/World", step=2, loads={"Operator_03": 12}, repeated={"Operator_02"})
     ergo_scan.write_report(res, r"<OUT>")
     ```

5. **Check the output is real before reporting.**
   - Every operator has frames, and none is marked **"Not moving in the USD file"**. That marker means the
     animation isn't saved in USD: it is driven at runtime by Animation Graph or omni.anim.people behaviour
     scripts.
     - If you see it, record or bake the animation to USD first (for example with Kit's Stage Recorder while
       playing), then scan the recording.
     - Never report the scores of a static operator as an assessment.
   - Spot-check the worst moment of the top 2 operators. Go to that frame in the scene (the frame number is in the
     report) and confirm the posture described ("trunk bent 50°, right arm raised 95°") is what you see. If it
     isn't, the joint mapping is wrong: fix `mapping.json` and re-run.

6. **Show the markers in the viewport.**
   - Sublayer `<OUT>/ergo_overlay.usda` into the scene. In Kit, use the Layer window, or run
     `st.GetRootLayer().subLayerPaths.append(r"<OUT>/ergo_overlay.usda")`.
   - It adds, under `/World/ErgoScan`, a sphere above each head coloured by the RULA band over time, plus dots on
     the back, shoulders and knees coloured by that body part's score.
   - It changes nothing else in the scene. Remove the sublayer to take it away.

7. **Optional: movie with the score panel.**
   - Render the scene's frames (for example Movie Capture to PNG, frame number in the file name).
   - Then run `--frames-dir=<render folder>`. Every frame gets the score panel beside it, in
     `<OUT>/frames_with_panel`.

8. **Report back to me** (short):
   - The overview table from `ergo_report.md`.
   - The 3 to 5 most important findings. For each: operator, time and frame, score, what the body is doing, and the
     suggested fix in workplace terms (raise or tilt the work, bring it closer, add a platform, rotate a fixture).
   - Which operators are fine (max 4 or below).
   - The links to the files (see the next section).
   - The caveats: loads and repetition you assumed or left out; that scores describe the animated method, not
     measured people; any operators you skipped and why.

## What you get in `<OUT>` (these are the dashboards)

| file | what it is |
|---|---|
| `ergo_dashboard.html` | **Main dashboard.** One file, works offline in any browser. It has: KPI tiles (operators, how many reach 7, % of time at 5+, worst operator); the ranking table (click an operator); the RULA-over-time chart; the body-angle chart; the body part at risk; the worst moments with fixes; the floor map of where risky work happens. |
| `ergo_panel_worst.png` | Score panel at the worst moment. One card per operator with the score badge, band, angles, a body figure coloured by part, and the last 20 s chart. |
| `ergo_timeline.png` | Every operator's RULA over time on one image, worst first. |
| `ergo_floor.png` | Plan view of where RULA 5-7 happens, with operators labelled at their usual spot. |
| `ergo_report.md` / `.json` | Overview table and worst moments with fixes. The `.json` is for scripts. |
| `ergo_frames.csv` | Every scored frame: score, body-part scores, angles. |
| `ergo_overlay.usda` | Viewport markers, to sublayer into the scene. |

## Reading the scores (RULA 1-7)

- **Bands:** 1-2 acceptable · 3-4 investigate · 5-6 change soon · 7 change now.
- **Walking empty-handed** is not a task posture, so it is capped at 2.
- **"Main driver"** is the body part that pushes the score up: trunk bent, neck bent, arm raised, deep knee bend,
  or twist. Each one has a matching fix:
  - trunk → bring the work up or closer;
  - neck → raise or tilt the work or display;
  - arm raised → lower the work or raise the operator;
  - knees → raise the part or give a stool or creeper;
  - twist → rotate the fixture or move the bin in front.
- **Rank by time as well as peak.** A short spike to 7 matters less than a long stretch at 5-6. Use the "time at
  5-6 / 7" columns and the "lasts" seconds on each worst moment.

## Rules

- Report the numbers the tool gives. Never round a 5 down to "acceptable".
- If something failed (skipped operators, static operators, wrong joint mapping), say so plainly. Do not leave
  them out quietly.
- In the scene, use red only for the risk markers and for E-stops and fire equipment. Never colour equipment red or
  orange: in our plant red means a quality issue.
