"""Showcase scene for the modular carton flow (roller) rack: a 20 x 20 m hall with four variants -
3 bays of wheel tracks with totes, a single roller-bed bay with cartons and a return level, 2 bays of ball mats with
blue beams, and a mixed 4-bay run (90 / 120 cm bays, a different bed per level).

    python tools/showcase_rack.py [out.usd]
"""
import os
import sys
import time

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

RACKS = [
    ("FR1", (-500, 0, -300), 0.0, {"bays": 3, "levels": 4, "bed": "wheel", "load": "totes"}),
    ("FR2", (250, 0, -300), 0.0, {"bays": 1, "bay_width": 130, "levels": 3, "height": 190, "bed": "roller",
                                  "load": "cartons", "return_top": True}),
    ("FR3", (700, 0, -300), 0.0, {"bays": 2, "levels": 3, "bed": "ball", "load": "mixed", "beams": "powder_blue"}),
    ("FR4", (-200, 0, 350), 180.0, {"bay_widths": "120,90,90,120", "levels": 5, "height": 240,
                                    "bed": "wheel,wheel,roller,ball,wheel", "return_top": True}),
]


def main(out):
    plant = Plant("showcase_rack", 2000.0, 2000.0, grid_x=[0.0, 1000.0, 2000.0], grid_y=[0.0, 1000.0, 2000.0])
    with safe_stage(out) as st:
        U.setup_stage(st)
        st.SetDefaultPrim(UsdGeom.Xform.Define(st, "/World").GetPrim())
        lib = MaterialLibrary(st, "/World/Looks")
        WA.build(st, "/World/Building", plant, "H10m", "plain", lib)
        UsdGeom.Xform.Define(st, "/World/Equipment")
        for rid, pos, yaw, params in RACKS:
            t0 = time.time()
            CAT.place(st, "/World/Equipment", "flow_rack", rid, pos, yaw, params, lib)
            print(f"{rid}: {time.time() - t0:.1f} s")
        CAT.place(st, "/World/Equipment", "industrial_table", "T1", (-500, 0, -80), 0.0, {}, lib)
        RCP._look(st, {"look": {"sky": True}})
    print("->", out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "out", "showcase", "rack.usd"))
