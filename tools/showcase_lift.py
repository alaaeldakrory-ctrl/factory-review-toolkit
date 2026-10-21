"""Showcase scene for the scissor lift table: a 20 x 20 m hall with mobile tables (one carrying a scissor unit,
one with a plain painted deck) and a fixed floor table, animated over 240 frames (24 fps).

    python tools/showcase_lift.py [out.usd]
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "exts", "frt.core"))
from pxr import Sdf, UsdGeom   # noqa: E402

from frt_core import catalog as CAT   # noqa: E402
from frt_core import recipe as RCP   # noqa: E402
from frt_core import units as U   # noqa: E402
from frt_core.building import warehouse_a as WA   # noqa: E402
from frt_core.drawing.interpret import Plant   # noqa: E402
from frt_core.materials import MaterialLibrary   # noqa: E402
from frt_core.rig import lift_table as LT   # noqa: E402
from frt_core.usdio import safe_stage   # noqa: E402

EQ = "/World/Equipment"
TABLES = [
    ("LT1", (-450, 0, -250), 0.0, {"length": 260, "width": 110}),
    ("LT2", (-450, 0, 150), 0.0, {"length": 220, "width": 95, "deck": "plain", "capacity": "750 KG"}),
    ("LT3", (300, 0, -250), 0.0, {"length": 220, "width": 110, "style": "fixed", "lowered": 30, "raised": 90}),
    ("LT4", (300, 0, 150), 0.0, {"length": 200, "width": 90, "lift": 0.6}),
]


def main(out):
    plant = Plant("showcase_lift", 2000.0, 2000.0, grid_x=[0.0, 1000.0, 2000.0], grid_y=[0.0, 1000.0, 2000.0])
    with safe_stage(out) as st:
        U.setup_stage(st)
        st.SetDefaultPrim(UsdGeom.Xform.Define(st, "/World").GetPrim())
        lib = MaterialLibrary(st, "/World/Looks")
        WA.build(st, "/World/Building", plant, "H10m", "plain", lib)
        UsdGeom.Xform.Define(st, EQ)
        for tid, pos, yaw, params in TABLES:
            CAT.place(st, EQ, "lift_table", tid, pos, yaw, params, lib)
        # a scissor unit being built on LT1 (stands on the deck and rides with it)
        lt1 = st.GetPrimAtPath(f"{EQ}/LT1/Attach/platform")
        px, py, pz = UsdGeom.Xformable(lt1).GetOrderedXformOps()[0].Get()
        u, _ = CAT.place(st, EQ, "scissor_unit", "U1", (-450 + px, py, -250 + pz), 0.0, {"length": 230}, lib)
        u.CreateAttribute("frt:on", Sdf.ValueTypeNames.String, custom=True).Set("LT1")
        RCP._look(st, {"look": {"sky": True}})
        st.SetTimeCodesPerSecond(24)
        LT.animate(st, f"{EQ}/LT1", [(0, {"lift": 0}), (70, {"lift": 1}), (150, {"lift": 1}), (220, {"height": 70}),
                                     (240, {"height": 70})])
        LT.animate(st, f"{EQ}/LT2", [(0, {"lift": 0.8}), (100, {"lift": 0}), (240, {"lift": 0.5})])
        LT.animate(st, f"{EQ}/LT3", [(0, {"lift": 0}), (40, {"lift": 0}), (160, {"lift": 1}), (240, {"lift": 1})])
    print("->", out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "out", "showcase", "lift.usd"))
