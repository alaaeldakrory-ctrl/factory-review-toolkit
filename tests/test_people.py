"""People module foundation: canonical skeleton per worker, motion catalog, UsdSkel worker."""
from pxr import Usd, UsdSkel

from frt_core.materials import MaterialLibrary
from frt_core.people import motions as M
from frt_core.people.skeleton import NAMES, PARENT, Skeleton
from frt_core.people.usd_worker import relaxed, write_worker
from frt_core.people.workers import WORKERS, custom


def test_skeleton_per_worker():
    reach = {}
    for wid, w in WORKERS.items():
        sk = Skeleton.for_worker(w)
        assert len(NAMES) == 53 and all(PARENT[n] in (None, *NAMES) for n in NAMES)
        assert abs(sk.length("upperarm_l", "lowerarm_l") - w.dims["upper_arm"]) < 1e-6
        assert abs(sk.world["upperarm_l"][0] + sk.world["upperarm_r"][0]) < 1e-9      # symmetric
        reach[wid] = sk.length("upperarm_l", "hand_l")
    assert reach["W05F"] < reach["W50M"] < reach["W95M"]
    assert abs(custom("X", 180.0).stature - 180.0) < 1e-6


def test_motion_catalog():
    lib = M.load()
    assert len(lib) >= 150
    cats = M.by_category(lib)
    assert {"fastening", "joining", "wiring", "handling", "equipment"} <= set(cats)
    m = lib["fasten.ratchet.tighten"]
    assert m.generator == ("cycle", "ratchet") and "ratchet_38" in m.tools
    assert M.validate([{"id": "x", "hands": "three", "postures": ["standing"], "pattern": "walk"}])


def test_worker_usdskel(stage):
    lib = MaterialLibrary(stage, "/World/Looks")
    w = WORKERS["W05F"]
    sk = Skeleton.for_worker(w)
    write_worker(stage, "/World/W", w, lib, pose=relaxed(sk), sk=sk)
    cache = UsdSkel.Cache()
    root = UsdSkel.Root(stage.GetPrimAtPath("/World/W"))
    cache.Populate(root, Usd.TraverseInstanceProxies())
    q = cache.GetSkelQuery(UsdSkel.Skeleton(stage.GetPrimAtPath("/World/W/Skeleton")))
    assert q
    xf = q.ComputeJointSkelTransforms(Usd.TimeCode.Default())
    hand = xf[NAMES.index("hand_l")].ExtractTranslation()
    assert 60 < hand[1] < 80 and abs(hand[0]) < 35                       # relaxed: hand hangs at the hip


def test_solver_and_generators():
    import numpy as np
    from frt_core.people import motion_gen as G
    from frt_core.people.kinematics import fk
    from frt_core.people.solver import HandTarget, PostureSolver
    sk = Skeleton.for_worker(WORKERS["W50M"])
    S = PostureSolver(sk)
    sol = S.solve([HandTarget("r", (-8.0, 95.0, 42.0), fingers=(0, 0, 1), palm=(0, -1, 0))], look=(-8.0, 95.0, 42.0))
    assert sol.terms["reach_r"] < 1e-3 and sol.angles["trunk_flexion"] < 20     # bench height: upright
    low = S.solve([HandTarget("r", (-8.0, 40.0, 42.0), fingers=(0, 0, 1), palm=(0, -1, 0))], look=(-8.0, 40.0, 42.0))
    assert low.angles["knee_r"] > 45 or low.angles["trunk_flexion"] > 25          # low target: bend or squat
    fr = G.ratchet(S, bolt=(8.0, 99.0, 48.0), side="r", strokes=1)
    assert len(fr) > 30 and all("ratchet_38" in f.tools for f in fr)
    prev = None
    for f in fr[20:30]:
        prev = S.solve(f.targets, look=f.look, prev=prev)
    R, P = fk(sk, prev.pose)
    grip = np.asarray(f.targets[0].point)
    assert np.linalg.norm(P["hand_r"] - grip) < 12.0                               # hand on the tool
