"""Offline bundle for an air-gapped PC: one zip with the extensions, tools, docs, the worker bodies, the walk clips
and every Python wheel the toolkit needs, so nothing is fetched from the internet there.

    python tools/make_offline_bundle.py [out_dir]      -> <out_dir>/frt_offline_<version>_<commit>.zip

Inside the zip (unzip anywhere, e.g. C:/frt):
    exts/           frt.core, frt.scene, frt.bundle  (add this folder to Kit's extension search paths)
    tools/, docs/, samples/, tests/, README.md, AGENTS.md
    data/bodies/    MPFB worker bodies (CC0)                  - found automatically by people/body.py
    data/mixamo/    walk / turn / carry clips (Mixamo)         - found automatically by people/mocap.py
    data/solved/    every job on every body, already solved (tools/people_presolve.py) - instant api.solve_job
    wheels/         ezdxf, pymupdf, pyparsing, fonttools (Kit installs them from here: archiveDirs in frt.core)
                    + usd-core, numpy, tomli for running the tools outside Kit (pip install --no-index ...)
    OFFLINE_INSTALL.md

The code comes from the last commit (git archive), so uncommitted edits are not in the bundle. Mixamo clips are for
your own use only (Adobe's terms do not allow redistributing them), so the bundle stays off public hosting.
"""
import os
import shutil
import subprocess
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
BODIES = os.environ.get("FRT_BODIES") or r"C:/OmniverseData/mpfb_bodies"
MIXAMO = os.environ.get("FRT_MIXAMO") or r"C:/OmniverseData/mixamo/fbx"
KIT_PY = "312"                       # Kit 107/108 ships Python 3.12
KIT_WHEELS = ["pyparsing>=3.0", "fonttools>=4.40", "ezdxf==1.4.5", "pymupdf>=1.24,<2"]
TOOL_WHEELS = ["usd-core", "numpy", "tomli", "pillow", "imageio-ffmpeg"]


def git(*a):
    return subprocess.run(["git", "-C", ROOT, *a], check=True, capture_output=True, text=True).stdout.strip()


def pip_download(reqs, dest, deps):
    cmd = [sys.executable, "-m", "pip", "download", "--dest", dest, "--only-binary=:all:", "--platform", "win_amd64",
           "--python-version", KIT_PY, "--implementation", "cp"] + ([] if deps else ["--no-deps"]) + reqs
    subprocess.run(cmd, check=True)


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "build", "offline")
    commit = git("rev-parse", "--short", "HEAD")
    version = "0.1.0"
    stage = os.path.join(out_dir, "frt")
    if os.path.isdir(stage):
        shutil.rmtree(stage)
    os.makedirs(stage)
    tar = os.path.join(out_dir, "src.zip")
    git("archive", "--format=zip", "-o", tar, "HEAD")
    with zipfile.ZipFile(tar) as z:
        z.extractall(stage)
    os.remove(tar)
    for src, dst, ext in ((BODIES, "bodies", ".npz"), (MIXAMO, "mixamo", ".fbx")):
        d = os.path.join(stage, "data", dst)
        os.makedirs(d)
        n = 0
        for f in sorted(os.listdir(src)):
            if f.endswith(ext):
                shutil.copy2(os.path.join(src, f), d)
                n += 1
        print(f"data/{dst}: {n} files")
    solved = os.path.join(ROOT, "data", "solved")
    if os.path.isdir(solved):
        shutil.copytree(solved, os.path.join(stage, "data", "solved"))
        print(f"data/solved: {len(os.listdir(solved))} files")
    else:
        print("data/solved missing - run tools/people_presolve.py first for instant results on the offline PC")
    wh = os.path.join(stage, "wheels")
    os.makedirs(wh)
    pip_download(KIT_WHEELS, wh, deps=False)
    pip_download(TOOL_WHEELS, wh, deps=True)
    shutil.copy2(os.path.join(ROOT, "docs", "OFFLINE_INSTALL.md"), os.path.join(stage, "OFFLINE_INSTALL.md"))
    with open(os.path.join(stage, "BUNDLE.txt"), "w") as f:
        f.write(f"frt offline bundle {version}, commit {commit}\n")
    zp = os.path.join(out_dir, f"frt_offline_{version}_{commit}.zip")
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for base, _, files in os.walk(stage):
            for f in files:
                p = os.path.join(base, f)
                z.write(p, os.path.relpath(p, out_dir))
    print("->", zp, f"{os.path.getsize(zp) / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
