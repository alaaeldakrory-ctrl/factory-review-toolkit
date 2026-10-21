"""Crane animation rig: the parts move, the chain is re-laid at a true pitch, limits clamp, attach points follow."""
from pxr import Usd, UsdGeom

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
