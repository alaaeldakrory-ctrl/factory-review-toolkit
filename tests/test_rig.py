"""Crane animation rig: the parts move, the chain is re-laid at a true pitch, limits clamp, attach points follow."""
from pxr import Sdf, Usd, UsdGeom

from frt_core import catalog as CAT
from frt_core.catalog.items_crane import PITCH
from frt_core.rig import crane as R


def _world(stage, path):
    return UsdGeom.Xformable(stage.GetPrimAtPath(path)).ComputeLocalToWorldTransform(Usd.TimeCode.Default())


def test_single_girder_rig(stage):
    CAT.place(stage, "/World", "bridge_crane", "BC", params={"span": 700, "runway": 600, "height": 480,
                                                            "hoists": 2, "hook_height": 140})
    lim = R.limits(stage, "/World/BC")
    z_lo, z_hi, _ = lim["Bridge_1"]["z"]
    assert z_lo < 0 < z_hi and set(lim["Bridge_1"]) == {"z", "Hoist_1", "Hoist_2"}
    pi = UsdGeom.PointInstancer(stage.GetPrimAtPath("/World/BC/Bridge_1/Hoist_2/Chain"))
    n0 = len(pi.GetPositionsAttr().Get())
    got = R.pose(stage, "/World/BC", bridge=1, z=9999, hoist=2, x=100, hook=60)
    assert got["z"] == z_hi and got["x"] == 100 and got["hook"] == 60
    assert len(pi.GetPositionsAttr().Get()) > n0 + 15                       # 80 cm lower: ~28 more links
    ys = [p[1] for p in pi.GetPositionsAttr().Get()]
    assert all(abs((a - b) - PITCH) < 1e-3 for a, b in zip(ys, ys[1:]))      # links keep their pitch
    t = UsdGeom.Xformable(stage.GetPrimAtPath("/World/BC/Attach/hook2")).GetOrderedXformOps()[0].Get()
    assert abs(t[1] - (60 + 12)) < 1e-6 and abs(t[2] - z_hi) < 1e-6          # hook tip follows hook and bridge
    # the festoon follows the last hoist
    tr = stage.GetPrimAtPath("/World/BC/Bridge_1/Festoon/Trolley_5")
    assert abs(_world(stage, tr.GetPath()).ExtractTranslation()[0] -
               (100 - R._rig(stage.GetPrimAtPath("/World/BC/Bridge_1/Festoon"), "rest")[-1])) < 1e-6


def test_double_girder_rig_animates(stage):
    CAT.place(stage, "/World", "bridge_crane", "BC2", params={"span": 800, "runway": 1400, "girders": 2,
                                                             "bridges": 2, "hook_height": 180})
    R.animate(stage, "/World/BC2", [(0, {"z": -300, "x": -200, "hook": 200}), (48, {"z": 100}), (96, {"hook": 60})],
              bridge=2)
    assert stage.GetStartTimeCode() == 0 and stage.GetEndTimeCode() == 96
    b = UsdGeom.Xformable(stage.GetPrimAtPath("/World/BC2/Bridge_2")).GetOrderedXformOps()[0]
    assert len(b.GetTimeSamples()) == 97                                    # baked every frame
    assert b.Get(24)[2] - b.Get(0)[2] == (b.Get(48)[2] - b.Get(0)[2]) / 2     # eased: halfway at mid-key
    assert b.Get(6)[2] - b.Get(0)[2] < (b.Get(48)[2] - b.Get(0)[2]) / 8      # and slow to start
    ropes = stage.GetPrimAtPath("/World/BC2/Bridge_2/Hoist_1/Ropes")
    sc = [op for op in UsdGeom.Xformable(ropes).GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeScale][0]
    assert sc.Get(96)[1] > sc.Get(0)[1]                                     # lower hook, longer ropes


def test_lift_table_rig(stage):
    from frt_core.rig import lift_table as LT
    for style in ("mobile", "fixed"):
        CAT.place(stage, "/World", "lift_table", f"LT_{style}", params={"length": 240, "width": 110, "style": style})
    CAT.place(stage, "/World", "scissor_unit", "U", params={})
    stage.GetPrimAtPath("/World/U").CreateAttribute("frt:on", Sdf.ValueTypeNames.String).Set("LT_mobile")
    lo, hi, rest = LT.limits(stage, "/World/LT_mobile")
    assert lo == rest and hi > lo + 50
    h = LT.pose(stage, "/World/LT_mobile", lift=1.0)
    assert h == hi
    deck = UsdGeom.Xformable(stage.GetPrimAtPath("/World/LT_mobile/Deck")).GetOrderedXformOps()[0].Get()
    assert abs(deck[1] - (hi - lo)) < 1e-6
    u = UsdGeom.Xformable(stage.GetPrimAtPath("/World/U")).GetOrderedXformOps()[0].Get()
    assert abs(u[1] - (hi - lo)) < 1e-6                                     # the unit rides up with the deck
    # the scissor closes: the outer arm's far end stays under the deck and the rod meets the arm
    p = stage.GetPrimAtPath("/World/LT_mobile")
    g = LT._geo(p)
    s = g.solve(hi)
    import math
    top = g.y_b + g.La * math.sin(math.radians(s["theta"]))
    assert abs(top - (hi - g.top_off)) < 1e-6
    assert s["ext"] > 10 and g.solve(lo)["ext"] == 0
    assert LT.pose(stage, "/World/LT_fixed", height=999) == LT.limits(stage, "/World/LT_fixed")[1]
