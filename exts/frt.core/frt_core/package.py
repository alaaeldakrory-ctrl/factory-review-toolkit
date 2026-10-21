"""Package a scene into one portable folder: the scene and every layer, material, texture and asset it uses are copied in
and the paths rewritten relative, so the folder opens on any PC (no C:/Nvidia_Assets needed).

    report = package("C:/scenes/plant.usd", "C:/share/plant_pkg")
    # -> plant_pkg/plant.usd, plant_pkg/<files beside the scene>, plant_pkg/assets/C/Nvidia_Assets/... (mirrored)
    # report = {"scene", "files", "bytes", "missing": [...], "rewritten": n}

The source folder layout is mirrored, so relative paths inside asset packs (sub-layers, MDL textures, UDIM tiles)
keep working untouched; only absolute paths and paths that leave their tree are rewritten. Built-in MDLs (OmniPBR.mdl,
...) are left as they are: every Omniverse app ships them. (UsdUtils.LocalizeAsset in OpenUSD 26.08 writes nested-layer
paths relative to the package root, which breaks the asset packs; hence this.)
"""
import glob
import os
import re
import shutil

from pxr import Sdf, UsdUtils

BUILTIN_MDL = re.compile(r"^(Omni[A-Za-z]*|nvidia/.*|core_definitions)\.mdl$|^nvidia/", re.I)
LAYER_EXT = (".usd", ".usda", ".usdc")
MDL_REF = re.compile(r'"([^"<>]+?\.(?:png|jpg|jpeg|exr|tga|hdr|tif|tiff|dds|mdl))"', re.I)
UDIM = "<UDIM>"


def _norm(p):
    return os.path.normcase(os.path.normpath(os.path.abspath(p)))


class _Packer:
    def __init__(self, scene, out_dir):
        self.scene = os.path.abspath(scene)
        self.src_root = os.path.dirname(self.scene)
        self.out = os.path.abspath(out_dir)
        self.map = {}            # normalised source file -> destination file
        self.todo = []
        self.missing = []
        self.rewritten = 0

    def dest(self, src):
        src = os.path.abspath(src)
        rel = os.path.relpath(src, self.src_root) if _norm(src).startswith(_norm(self.src_root) + os.sep) else None
        if rel is None:
            drive, rest = os.path.splitdrive(src)
            rel = os.path.join("assets", drive.rstrip(":\\/") or "root", rest.lstrip("\\/"))
        return os.path.join(self.out, rel)

    def add(self, src):
        k = _norm(src)
        if k not in self.map:
            self.map[k] = self.dest(src)
            self.todo.append(os.path.abspath(src))
        return self.map[k]

    def resolve(self, anchor_file, path):
        """-> list of existing source files for an asset path written in anchor_file (UDIM expands to its tiles)."""
        if not path or "://" in path or path.startswith("omniverse:"):
            return None
        full = path if os.path.isabs(path) else os.path.join(os.path.dirname(anchor_file), path)
        if UDIM in full:
            tiles = glob.glob(full.replace(UDIM, "[1-9][0-9][0-9][0-9]"))
            return tiles or []
        return [full] if os.path.isfile(full) else []

    def relink(self, anchor_src, path):
        """Path to write in the packaged copy of anchor_src for `path`; copies what it points at."""
        if path.lower().endswith(".mdl") and BUILTIN_MDL.search(path.replace("\\", "/")):
            return path
        files = self.resolve(anchor_src, path)
        if files is None:
            return path
        if not files:
            self.missing.append(f"{path} (in {os.path.basename(anchor_src)})")
            return path
        for f in files:
            self.add(f)
        if UDIM in path:
            target = self.dest(os.path.join(os.path.dirname(anchor_src), path) if not os.path.isabs(path) else path)
        else:
            target = self.dest(files[0])
        new = os.path.relpath(target, os.path.dirname(self.dest(anchor_src))).replace("\\", "/")
        if not new.startswith("."):
            new = "./" + new
        if new != path.replace("\\", "/"):
            self.rewritten += 1
        return new

    def layer(self, src, dst):
        lay = Sdf.Layer.OpenAsAnonymous(src)                    # a private copy: the source is never edited
        if lay is None:
            self.missing.append(f"{src} (could not open)")
            shutil.copy2(src, dst)
            return
        UsdUtils.ModifyAssetPaths(lay, lambda p: self.relink(src, p))
        lay.Export(dst)

    def mdl(self, src, dst):
        text = open(src, encoding="utf-8", errors="replace").read()

        def sub(m):
            p = m.group(1)
            return '"' + self.relink(src, p) + '"' if not p.startswith("/") or os.path.isfile(p) else m.group(0)
        open(dst, "w", encoding="utf-8").write(MDL_REF.sub(sub, text))

    def run(self):
        root_dst = os.path.join(self.out, os.path.basename(self.scene))
        self.map[_norm(self.scene)] = root_dst
        self.todo.append(self.scene)
        n, size = 0, 0
        while self.todo:
            src = self.todo.pop()
            dst = self.map[_norm(src)]
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            ext = os.path.splitext(src)[1].lower()
            if ext in LAYER_EXT:
                self.layer(src, dst)
            elif ext == ".mdl":
                self.mdl(src, dst)
            else:
                shutil.copy2(src, dst)
            n += 1
            size += os.path.getsize(dst)
        return {"scene": root_dst, "files": n, "bytes": size, "missing": sorted(set(self.missing)),
                "rewritten": self.rewritten}


def package(scene, out_dir, overwrite=True):
    if overwrite and os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    return _Packer(scene, out_dir).run()


def check(scene):
    """Unresolved dependencies of a (packaged) scene, built-in MDLs ignored."""
    _, _, unresolved = UsdUtils.ComputeAllDependencies(scene)
    return [u for u in unresolved if not BUILTIN_MDL.search(os.path.basename(u))]
