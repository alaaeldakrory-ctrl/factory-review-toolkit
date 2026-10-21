import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "exts", "frt.core"))
os.environ.setdefault("PXR_USDC_EMIT_DEPRECATION_WARNINGS", "0")

DRAWINGS = os.path.join(ROOT, "samples", "drawings")
NV = os.environ.get("FRT_NVIDIA_ASSETS", r"C:/Nvidia_Assets")
needs_packs = pytest.mark.skipif(not os.path.isdir(NV), reason="NVIDIA asset packs not installed")


def drawing(name):
    return os.path.join(DRAWINGS, name)


@pytest.fixture
def stage(tmp_path):
    from pxr import Usd
    from frt_core import units
    st = Usd.Stage.CreateNew(str(tmp_path / "t.usda"))
    units.setup_stage(st)
    return st
