"""The cloud engine's build folder (out/engine_image): the code the website's engine runs, the sample tasks, and the
data files it reads that are not in the repository - the Mixamo walk clips, the website's worker (.glb) and its skin
measurement (out/people/cache/skin_*.npz, made with Blender, which the container does not have).

    python deploy/engine/assemble.py [--site=../frt-site]      -> out/engine_image (then deploy.sh builds it)

The image goes only to the project's own private registry: the Mixamo files never go into the repository or onto
public hosting (the worker .glb is on the website already).
"""
import glob
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(ROOT, "out", "engine_image")
CODE = ["exts/frt.core", "tools/task_run.py", "tools/web_export.py", "tools/clip_export.py", "tools/clip_audit.py",
        "tools/blender", "docs/people/demo"]
DROP = ("__pycache__", "*.pyc", "tests")


def main():
    opt = {a.split("=")[0][2:]: a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--") and "=" in a}
    site = os.path.abspath(opt.get("site", os.path.join(ROOT, "..", "frt-site")))
    shutil.rmtree(OUT, ignore_errors=True)
    for c in CODE:
        src, dst = os.path.join(ROOT, c), os.path.join(OUT, c)
        if os.path.isdir(src):
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns(*DROP))
        else:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy(src, dst)
    shutil.copy(os.path.join(ROOT, "tools", "engine_server.py"), os.path.join(OUT, "tools", "engine_server.py"))
    mix = os.environ.get("FRT_MIXAMO", r"C:/OmniverseData/mixamo/fbx")              # walk clips -> data/mixamo
    shutil.copytree(mix, os.path.join(OUT, "data", "mixamo"))
    os.makedirs(os.path.join(OUT, "assets"), exist_ok=True)
    shutil.copy(os.path.join(site, "assets", "worker.glb"), os.path.join(OUT, "assets", "worker.glb"))
    cache = os.path.join(OUT, "out", "people", "cache")
    os.makedirs(cache, exist_ok=True)
    for p in glob.glob(os.path.join(ROOT, "out", "people", "cache", "skin_*.npz")):
        shutil.copy(p, cache)
    for f in ("Dockerfile", "requirements.txt", ".dockerignore"):
        shutil.copy(os.path.join(HERE, f), os.path.join(OUT, f))
    n = sum(len(fs) for _, _, fs in os.walk(OUT))
    mb = sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(OUT) for f in fs) / 1e6
    print(f"{OUT}: {n} files, {mb:.0f} MB")


if __name__ == "__main__":
    main()
