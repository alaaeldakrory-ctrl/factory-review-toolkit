"""Showcase scene for the workstation crane: a 20 x 20 m hall of Warehouse_A modules with a standard 4-leg crane
(single girder, chain hoist) and a double-girder crane beside it, a scissor unit under each hook.

    python tools/showcase_crane.py [out.usd]
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "exts", "frt.core"))
from pxr import UsdGeom   # noqa: E402

from frt_core import catalog as CAT   # noqa: E402
from frt_core import recipe as RCP   # noqa: E402
from frt_core import units as U   # noqa: E402
from frt_core.building import warehouse_a as WA   # noqa: E402
from frt_core.drawing.interpret import Plant   # noqa: E402
from frt_core.materials import MaterialLibrary   # noqa: E402
from frt_core.usdio import safe_stage   # noqa: E402


def main(out):
    plant = Plant("showcase", 3000.0, 2000.0, grid_x=[0.0, 1000.0, 2000.0, 3000.0], grid_y=[0.0, 1000.0, 2000.0])
    with safe_stage(out) as st:
        U.setup_stage(st)
        st.SetDefaultPrim(UsdGeom.Xform.Define(st, "/World").GetPrim())
        lib = MaterialLibrary(st, "/World/Looks")
        WA.build(st, "/World/Building", plant, "H10m", "plain", lib)
        UsdGeom.Xform.Define(st, "/World/Equipment")
        CAT.place(st, "/World/Equipment", "bridge_crane", "BC1", (-1000, 0, 0), 0.0,
                  {"span": 700, "runway": 600, "height": 480, "capacity": "2000", "hook_height": 140,
                   "hoists": 2}, lib)
        CAT.place(st, "/World/Equipment", "scissor_unit", "U1", (-1030, 0, 130), 0.0, {}, lib)
        CAT.place(st, "/World/Equipment", "bridge_crane", "BC2", (1000, 0, 0), 0.0,
                  {"span": 800, "runway": 1400, "height": 550, "girders": 2, "bridges": 2,
                   "bridge_positions": "0.3,0.75", "capacity": "5000", "hook_height": 180, "trolley": 0.4}, lib)
        CAT.place(st, "/World/Equipment", "scissor_unit", "U2", (960, 0, -330), 90.0, {"length": 245}, lib)
        CAT.place(st, "/World/Equipment", "industrial_table", "T1", (-1000, 0, -200), 0.0, {}, lib)
        RCP._look(st, {"look": {"sky": True}})
    print("->", out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "out", "showcase", "crane.usd"))
