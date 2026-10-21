"""Drawings in: DXF and vector PDF import, alignment and interpretation into a Plant (walls, grid, doors, aisles,
equipment). `load(path)` picks the importer from the extension."""
import os

from .interpret import Door, Equip, Plant, interpret, load_legend   # noqa: F401


def load(path, **kw):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".dxf":
        from .dxf_import import load as _load
    elif ext == ".pdf":
        from .pdf_import import load as _load
    else:
        raise ValueError(f"unsupported drawing type {ext} (DXF or PDF)")
    return _load(path, **kw)


def read_plant(path, legend=None, **kw):
    return interpret(load(path, **kw), legend)
