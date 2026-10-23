# Kimodo in the toolkit: generated body motion, exact hands

[Kimodo](https://research.nvidia.com/labs/sil/projects/kimodo/) (NVIDIA, 2026) is a motion diffusion model trained
on about 700 hours of production optical mocap. It generates whole-body motion from a text prompt and/or
**kinematic constraints**: end-effector keyframes (hand / foot positions and rotations), full-body keyframes and
root paths. Code: Apache-2.0. Recommended models (Kimodo-SOMA-RP) are under the NVIDIA Open Model License (read it
before shipping). The SMPL-X variant is research-only - not used.

## Where it fits

The toolkit's motion stack is built in layers:

| Layer | Does | Kimodo's role |
|---|---|---|
| process / jobs | what happens, where, with what tool | - |
| our solver (work poses) | exact hand / tool / posture at the work, keep-outs, body size | gives Kimodo its constraints |
| **Kimodo** | natural whole-body motion meeting those constraints: trunk, shoulders, head, weight shift, how the arms travel, transitions, steps | **the in-betweening / body-motion engine** |
| our solver (final pass) | "keep the body, solve the hands": exact grips, tools, fingers, thumb; small pelvis / trunk corrections for keep-outs and planted feet | corrects Kimodo where the station demands it |
| checks | limits, jumps, contact, awkward postures, collisions | the same gate for everything |

Kimodo knows nothing about the station (fixtures, bench edges, panels), so it never has the last word on the body:
the final solver pass enforces the keep-outs (the kneel-and-drive test leaned into the plate until that pass was
added).

## Pipeline (tools/kimodo/)

1. `kimodo_job.py <job>` takes our solved job (`out/people/jobs/cache/<job>.pkl`), samples keyframes every 4
   frames: wrists (when that hand works), feet (planted), and for kneel / squat postures the whole solved body
   every 3rd keyframe. Positions are scaled to Kimodo's body (hip height 1.00 m).
2. `kimodo_constrained.py` (runs in Kimodo's own Python env) builds `EndEffectorConstraintSet` /
   `FullBodyConstraintSet` and generates with **no text**. A stand-in encoder returns zero features, which is
   Kimodo's own handling of an empty prompt, so the gated Llama text encoder is not needed.
3. The output (SOMA 77 joints, 30 fps) is mapped onto our canonical joints, resampled to 24 fps and retargeted
   onto the worker.
4. The final pass solves with `S.base_correct = True`. Kimodo's pose is the base, the legs are solved to the job's
   stance, arms / hands / fingers go to the job's targets, and pelvis / trunk corrections are penalised for leaving
   the base. Then `check_frames`.
5. `compare_scene.py` renders rule-based vs Kimodo-based side by side.

Speed: about 3 s per 4-5 s motion on an RTX 5060 Ti (50-100 denoising steps), plus our final pass.

## Results so far (2026-10-22)

| Job | Rule-based | Kimodo-based |
|---|---|---|
| ratchet | passes | passes; trunk more upright (2-8 deg vs 13-20), head less dropped, shoulder girdle and off arm move naturally; hands identical |
| kneel and drive | passes | passes after the final pass corrects the trunk for the plate (Kimodo alone leaned in) |
| move a box | passes | passes |
| overhead ratchet | passes | 2 borderline awkward frames |

## In the extension

- An optional **"natural motion"** switch per process step. Off: the rule-based solver alone (fast, deterministic).
  On: Kimodo plus the final pass.
- Kimodo runs as a local service in its own Python env (GPU). The Kit extension sends keyframes and gets joint
  tracks back, as these tools do with npz files. Nothing of Kimodo runs inside Kit's Python.
- Text prompts ("a worker kneels and drives a bolt") need the gated Llama 3 text encoder: request access to
  `meta-llama/Meta-Llama-3-8B-Instruct` on Hugging Face and log in (`hf auth login`). Constraint-only generation
  works without it.
- Walking between stations: Kimodo's `root2d` path constraint could replace or blend with the clip-based
  locomotion; to be compared.

## Install (as done here)

```
python -m pip install --user uv
python -m uv venv --python 3.11 C:/OmniverseData/kimodo/env
python -m uv pip install --python C:/OmniverseData/kimodo/env/Scripts/python.exe torch --index-url https://download.pytorch.org/whl/cu128
python -m uv pip install --python C:/OmniverseData/kimodo/env/Scripts/python.exe hydra-core omegaconf numpy scipy transformers==5.1.0 peft einops tqdm pydantic trimesh pillow bvhio boto3 filelock packaging huggingface_hub safetensors
git clone https://github.com/nv-tlabs/kimodo.git C:/OmniverseData/kimodo/src     (run from source; the optional
                                                                                   C++ MotionCorrection is skipped)
```

The model downloads on first use to `HF_HOME=C:/OmniverseData/kimodo/hf`.
