"""Package a scene into one portable folder: the scene and every layer, texture and asset it uses are copied in and
all paths are rewritten relative, so the folder opens on any PC (no C:/Nvidia_Assets needed).

    report = package("C:/scenes/plant.usd", "C:/share/plant_pkg")
    # -> C:/share/plant_pkg/plant.usd + C:/share/plant_pkg/<dependencies>; report = {"files": n, "bytes": ..., "missing": [...]}
"""
import os
import shutil

from pxr import Sdf, UsdUtils


def package(scene, out_dir, overwrite=True):
    scene = os.path.abspath(scene)
    if overwrite and os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    loc = getattr(UsdUtils, "LocalizeAsset", None)
    if loc is None:
        raise RuntimeError("this USD build has no UsdUtils.LocalizeAsset (needs OpenUSD 23.11+)")
    _, _, unresolved_before = UsdUtils.ComputeAllDependencies(scene)
    ok = loc(scene, out_dir)
    if not ok:
        raise RuntimeError(f"localizing {scene} failed")
    n, size = 0, 0
    for dirpath, _, files in os.walk(out_dir):
        for f in files:
            n += 1
            size += os.path.getsize(os.path.join(dirpath, f))
    root = os.path.join(out_dir, os.path.basename(scene))
    return {"scene": root, "files": n, "bytes": size, "missing": list(unresolved_before)}
