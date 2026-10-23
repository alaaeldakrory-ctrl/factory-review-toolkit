# Installing on an air-gapped PC

1. Copy `frt_offline_<version>_<commit>.zip` to the PC (USB stick or internal file share) and unzip it, e.g. to
   `C:\frt`. You get `C:\frt\frt\exts`, `tools`, `data`, `wheels`, `AGENTS.md` ... Below, `<FRT>` = `C:\frt\frt`.
2. **Kit app** (USD Composer, Isaac Sim, your own kit-app-template app; Kit 107 or later, Python 3.12):
   - **Window > Extensions** > gear icon > **Extension Search Paths** > add `<FRT>\exts`, or start Kit with
     `--ext-folder <FRT>\exts`.
   - Enable **frt.scene** (it enables **frt.core**). Its Python packages install from `<FRT>\wheels`; no internet.
   - Check: in **Window > Script Editor** run `from frt_core.people import api; print(len(api.list_jobs()))` → `50`.
3. **Command line / AI agent** (outside Kit), any Python 3.12 (64-bit Windows):

   ```
   python -m pip install --no-index --find-links <FRT>\wheels usd-core numpy tomli pillow imageio-ffmpeg
   python <FRT>\tools\people_job.py --list
   ```

   For a different Python version, download matching `usd-core` / `numpy` wheels on a connected PC
   (`pip download usd-core numpy --python-version 3.11 --only-binary=:all: --platform win_amd64 -d wheels`).
4. Point your agent at `<FRT>\AGENTS.md`.

Optional on that PC:
- **NVIDIA materials**: set `FRT_NVIDIA_MATERIALS` to the Warehouse pack's `Materials/Base` folder for textured
  equipment. Without it, plain materials are used.
- **Data folders**: `FRT_BODIES` and `FRT_MIXAMO` override the bundled `data\bodies` and `data\mixamo`.

The bundle has the Mixamo walk clips in it. They are for your own use, so do not post the zip publicly.
